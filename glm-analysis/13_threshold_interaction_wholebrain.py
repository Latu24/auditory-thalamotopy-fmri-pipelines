"""Same three threshold/correction methods as 12_threshold_interaction.py,
applied to the FULL functional-coverage volume (all slabs / whole imaged
field of view) instead of the thalamus-restricted subset. Reuses the same
GLM fit (cfxam_interaction_native.npz) — no re-fitting needed, since that
file already stores the whole-volume t_native and mask_native arrays; the
thalamus script just intersected them with the thalamus mask before
thresholding, this one uses mask_native directly.

Same method choices/caveats as before:
  1. p<0.05 uncorrected, two-tailed.
  2. FDR q<0.05 (Benjamini-Hochberg), two-tailed.
  3. Monte Carlo cluster-extent, parametric ACF-matched-noise (5000 sims),
     cluster-forming threshold p<0.001. Smoothness estimated from the whole
     mask_native itself (no separate "outside ROI" region needed here, since
     the ROI now IS the whole covered volume — same logic AFNI uses: any
     true effect occupies too small a fraction of voxels to bias a
     whole-volume smoothness estimate).
"""
import os
import json
import numpy as np
from scipy import stats, ndimage

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

WORK = f"{PROJECT_ROOT}/derivatives/sub-01/analysis/thalamus_work"

ALPHA = 0.05
CLUSTER_FORMING_P = 0.001
N_MONTE_CARLO = 5000
RNG_SEED = 42


def method_uncorrected(t_vals, dof):
    tcrit = float(stats.t.isf(ALPHA / 2, dof))
    sig = np.abs(t_vals) > tcrit
    return {
        "method": "p<0.05 uncorrected (two-tailed)",
        "t_critical": tcrit,
        "n_significant": int(sig.sum()),
        "n_total": int(t_vals.size),
        "fraction": float(sig.mean()),
    }


def method_fdr_bh(t_vals, dof):
    p = 2 * stats.t.sf(np.abs(t_vals), dof)
    order = np.argsort(p)
    p_sorted = p[order]
    m = p.size
    ranks = np.arange(1, m + 1)
    bh_crit = ranks / m * ALPHA
    below = p_sorted <= bh_crit
    q_thresh_p = p_sorted[np.max(np.where(below)[0])] if below.any() else 0.0
    sig = p <= q_thresh_p
    return {
        "method": "FDR q<0.05 (Benjamini-Hochberg, two-tailed)",
        "p_threshold": float(q_thresh_p),
        "n_significant": int(sig.sum()),
        "n_total": int(m),
        "fraction": float(sig.mean()),
    }


def estimate_axis_fwhm(vol, mask):
    fwhms = []
    for axis in range(3):
        a = np.take(vol, range(0, vol.shape[axis] - 1), axis=axis)
        b = np.take(vol, range(1, vol.shape[axis]), axis=axis)
        ma = np.take(mask, range(0, mask.shape[axis] - 1), axis=axis)
        mb = np.take(mask, range(1, mask.shape[axis]), axis=axis)
        both = ma & mb
        if both.sum() < 100:
            fwhms.append(2.0)
            continue
        av, bv = a[both], b[both]
        rho = float(np.corrcoef(av, bv)[0, 1])
        rho = np.clip(rho, 1e-4, 0.999)
        fwhm = np.sqrt(-4 * np.log(2) / np.log(rho))
        fwhms.append(float(np.clip(fwhm, 0.5, 10.0)))
    return fwhms


def method_monte_carlo_cluster(t_native, mask_native, dof):
    fwhm_zyx = estimate_axis_fwhm(t_native, mask_native)
    sigma_zyx = [f / 2.3548 for f in fwhm_zyx]
    print(f"    estimated smoothness FWHM (voxels, z/y/x): "
          f"{[round(f, 2) for f in fwhm_zyx]}")

    zz = np.where(mask_native.any(axis=(1, 2)))[0]
    yy = np.where(mask_native.any(axis=(0, 2)))[0]
    xx = np.where(mask_native.any(axis=(0, 1)))[0]
    pad = 6
    z0, z1 = max(0, zz.min() - pad), min(mask_native.shape[0], zz.max() + pad + 1)
    y0, y1 = max(0, yy.min() - pad), min(mask_native.shape[1], yy.max() + pad + 1)
    x0, x1 = max(0, xx.min() - pad), min(mask_native.shape[2], xx.max() + pad + 1)
    box_mask = mask_native[z0:z1, y0:y1, x0:x1]
    box_shape = box_mask.shape
    print(f"    simulation box shape: {box_shape}, "
          f"{int(box_mask.sum())} covered voxels inside it")

    tcrit_forming = float(stats.t.isf(CLUSTER_FORMING_P / 2, dof))
    structure = np.ones((3, 3, 3), dtype=int)

    rng = np.random.default_rng(RNG_SEED)
    max_cluster_sizes = np.zeros(N_MONTE_CARLO, dtype=int)
    for i in range(N_MONTE_CARLO):
        noise = rng.standard_normal(box_shape)
        smoothed = ndimage.gaussian_filter(noise, sigma=sigma_zyx)
        smoothed = smoothed / smoothed[box_mask].std()
        sig = (np.abs(smoothed) > tcrit_forming) & box_mask
        if not sig.any():
            continue
        labels, n_cl = ndimage.label(sig, structure=structure)
        if n_cl == 0:
            continue
        sizes = ndimage.sum(sig, labels, index=np.arange(1, n_cl + 1))
        max_cluster_sizes[i] = int(sizes.max())
        if (i + 1) % 1000 == 0:
            print(f"    ... {i + 1}/{N_MONTE_CARLO} simulations done")

    cluster_thresh = int(np.ceil(np.percentile(max_cluster_sizes, 95)))

    real_sig = (np.abs(t_native) > tcrit_forming) & mask_native
    labels_real, n_cl_real = ndimage.label(real_sig, structure=structure)
    sizes_real = (ndimage.sum(real_sig, labels_real,
                              index=np.arange(1, n_cl_real + 1))
                 if n_cl_real else np.array([]))
    surviving = sizes_real >= cluster_thresh
    n_voxels_surviving = int(sizes_real[surviving].sum()) if surviving.any() else 0

    return {
        "method": "Monte Carlo cluster-extent (parametric, ACF-matched noise)",
        "cluster_forming_p": CLUSTER_FORMING_P,
        "cluster_forming_t_threshold": tcrit_forming,
        "estimated_fwhm_voxels_zyx": fwhm_zyx,
        "n_monte_carlo_iterations": N_MONTE_CARLO,
        "null_max_cluster_size_95th_pct": cluster_thresh,
        "null_max_cluster_size_distribution_summary": {
            "mean": float(max_cluster_sizes.mean()),
            "median": float(np.median(max_cluster_sizes)),
            "max": int(max_cluster_sizes.max()),
        },
        "real_data_n_clusters_at_primary_threshold": int(n_cl_real),
        "real_data_largest_clusters": sorted([int(s) for s in sizes_real], reverse=True)[:20],
        "n_clusters_surviving_extent_threshold": int(surviving.sum()),
        "n_voxels_surviving": n_voxels_surviving,
    }


def main():
    d = np.load(f"{WORK}/cfxam_interaction_native.npz")
    t_native, mask_native = d["t_native"], d["mask_native"]
    dof = float(d["dof"])
    t_vals = t_native[mask_native]
    print(f"loaded interaction map: {t_vals.size} whole-brain covered voxels, dof={dof:.1f}")

    print("\n[1/3] p<0.05 uncorrected ...")
    r1 = method_uncorrected(t_vals, dof)
    print(f"    {r1}")

    print("\n[2/3] FDR q<0.05 (Benjamini-Hochberg) ...")
    r2 = method_fdr_bh(t_vals, dof)
    print(f"    {r2}")

    print(f"\n[3/3] Monte Carlo cluster-extent ({N_MONTE_CARLO} iterations) ...")
    r3 = method_monte_carlo_cluster(t_native, mask_native, dof)
    print(f"    {r3}")

    out = {"dof": dof, "n_wholebrain_coverage_voxels": int(t_vals.size),
           "uncorrected_p05": r1, "fdr_q05": r2, "monte_carlo_cluster": r3}
    with open(f"{WORK}/cfxam_interaction_thresholds_wholebrain.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {WORK}/cfxam_interaction_thresholds_wholebrain.json")


if __name__ == "__main__":
    main()
