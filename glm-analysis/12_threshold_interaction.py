"""Three threshold/correction methods on the CF x AM interaction t-map
produced by 11_glm_crossed_2x2_interaction.py, restricted to the native-space
thalamus mask (same ROI convention used for this pipeline's other
significance gates):

  1. p<0.05 uncorrected, two-tailed (interaction can go either direction).
  2. FDR q<0.05 (Benjamini-Hochberg), two-tailed, within the thalamus mask
     (same BH procedure used for this pipeline's other FDR q<0.05 exports).
  3. Monte Carlo cluster-extent correction, AFNI-3dClustSim-style:
     - spatial smoothness (FWHM per axis) estimated empirically from the
       observed interaction t-map's own autocorrelation, in covered voxels
       OUTSIDE the thalamus (treated as a null-ish region for smoothness
       estimation, standard practice — true interaction effects, if any,
       occupy a small fraction of voxels and don't meaningfully bias a
       smoothness estimate taken from the rest of the covered volume).
     - primary (cluster-forming) voxel threshold: p<0.001 uncorrected,
       two-tailed — the conventional AFNI/SPM default for this step, distinct
       from the p<0.05 whole-map test above (that one has no cluster-extent
       requirement at all).
     - 5000 simulated Gaussian random fields, matched smoothness, generated
       only inside a padded box around the thalamus (cheap — no VTC re-read,
       no GLM refit) — thresholded the same way, max cluster size recorded
       per iteration, 95th percentile taken as the cluster-extent threshold.
     - A permutation-based null (shuffle condition labels, refit the whole
       multi-run GLM, repeat thousands of times) was considered but is
       computationally prohibitive here, since each refit streams several GB
       of VTC data per run. The parametric route is on firmer footing for
       THIS design than it would be for the marginal 45-predictor design,
       since this crossed design is well-conditioned OLS (cond~5), not the
       severely collinear design the earlier ridge-based analyses required.
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


def method_uncorrected(t_thal, dof):
    tcrit = float(stats.t.isf(ALPHA / 2, dof))
    sig = np.abs(t_thal) > tcrit
    return {
        "method": "p<0.05 uncorrected (two-tailed)",
        "t_critical": tcrit,
        "n_significant": int(sig.sum()),
        "n_total": int(t_thal.size),
        "fraction": float(sig.mean()),
    }, sig


def method_fdr_bh(t_thal, dof):
    p = 2 * stats.t.sf(np.abs(t_thal), dof)  # two-tailed
    order = np.argsort(p)
    p_sorted = p[order]
    m = p.size
    ranks = np.arange(1, m + 1)
    bh_crit = ranks / m * ALPHA
    below = p_sorted <= bh_crit
    if below.any():
        k = np.max(np.where(below)[0])
        q_thresh_p = p_sorted[k]
    else:
        q_thresh_p = 0.0
    sig_sorted = p_sorted <= q_thresh_p
    sig = np.zeros(m, dtype=bool)
    sig[order] = sig_sorted
    return {
        "method": "FDR q<0.05 (Benjamini-Hochberg, two-tailed)",
        "p_threshold": float(q_thresh_p),
        "n_significant": int(sig.sum()),
        "n_total": int(m),
        "fraction": float(sig.mean()),
    }, sig


def estimate_axis_fwhm(vol, mask):
    """Empirical lag-1 spatial autocorrelation -> Gaussian FWHM (voxels),
    per axis, AFNI-style: rho = exp(-4 ln2 (1/FWHM)^2)."""
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


def method_monte_carlo_cluster(t_native, mask_native, thal, in_thal, dof):
    outside_thal_covered = mask_native & (~thal)
    fwhm_zyx = estimate_axis_fwhm(t_native, outside_thal_covered)
    sigma_zyx = [f / 2.3548 for f in fwhm_zyx]  # FWHM -> sigma
    print(f"    estimated smoothness FWHM (voxels, z/y/x): "
          f"{[round(f, 2) for f in fwhm_zyx]}")

    zmin, zmax = np.where(in_thal.any(axis=(1, 2)))[0][[0, -1]]
    ymin, ymax = np.where(in_thal.any(axis=(0, 2)))[0][[0, -1]]
    xmin, xmax = np.where(in_thal.any(axis=(0, 1)))[0][[0, -1]]
    pad = 6
    z0, z1 = max(0, zmin - pad), min(in_thal.shape[0], zmax + pad + 1)
    y0, y1 = max(0, ymin - pad), min(in_thal.shape[1], ymax + pad + 1)
    x0, x1 = max(0, xmin - pad), min(in_thal.shape[2], xmax + pad + 1)
    box_mask = in_thal[z0:z1, y0:y1, x0:x1]
    box_shape = box_mask.shape
    print(f"    simulation box shape: {box_shape}, "
          f"{int(box_mask.sum())} thalamus-coverage voxels inside it")

    tcrit_forming = float(stats.t.isf(CLUSTER_FORMING_P / 2, dof))
    structure = np.ones((3, 3, 3), dtype=int)  # 26-connectivity

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

    cluster_thresh = int(np.ceil(np.percentile(max_cluster_sizes, 95)))

    # ---- apply to the REAL map, same primary threshold + connectivity ----
    real_sig = (np.abs(t_native) > tcrit_forming) & in_thal
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
        "real_data_cluster_sizes": sorted([int(s) for s in sizes_real], reverse=True),
        "n_clusters_surviving_extent_threshold": int(surviving.sum()),
        "n_voxels_surviving": n_voxels_surviving,
    }


def main():
    d = np.load(f"{WORK}/cfxam_interaction_native.npz")
    t_native, mask_native, thal, in_thal, t_thal = (
        d["t_native"], d["mask_native"], d["thal"], d["in_thal"], d["t_thal"])
    dof = float(d["dof"])
    print(f"loaded interaction map: {t_thal.size} thalamus-coverage voxels, dof={dof:.1f}")

    print("\n[1/3] p<0.05 uncorrected ...")
    r1, sig1 = method_uncorrected(t_thal, dof)
    print(f"    {r1}")

    print("\n[2/3] FDR q<0.05 (Benjamini-Hochberg) ...")
    r2, sig2 = method_fdr_bh(t_thal, dof)
    print(f"    {r2}")

    print(f"\n[3/3] Monte Carlo cluster-extent ({N_MONTE_CARLO} iterations) ...")
    r3 = method_monte_carlo_cluster(t_native, mask_native, thal, in_thal, dof)
    print(f"    {r3}")

    out = {"dof": dof, "n_thalamus_coverage_voxels": int(t_thal.size),
           "uncorrected_p05": r1, "fdr_q05": r2, "monte_carlo_cluster": r3}
    with open(f"{WORK}/cfxam_interaction_thresholds.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {WORK}/cfxam_interaction_thresholds.json")


if __name__ == "__main__":
    main()
