"""Diagnostic re-analysis: can spatial smoothing (as commonly applied in a
standard BrainVoyager preprocessing pipeline) explain a broader, less
anatomically constrained pattern of activity than this project's default
unsmoothed analysis, without spatial smoothing being mistaken for a
stronger true effect?

This pipeline deliberately applies NO spatial smoothing by default.

Method: spatially smooth the ALREADY-COMPUTED native-space t-maps
(interaction, CF_main, AM_main from 11/14_*.py) with an assumed FWHM=4mm
Gaussian (a common BrainVoyager default; flagged here as an assumption, not
a matched value). This is exact for the beta/contrast numerator —
per-voxel GLM fitting with a SHARED design matrix X across voxels is a
linear operator (beta_v = (X'X)^-1 X' y_v), so spatially smoothing y (data)
before fitting is mathematically identical to spatially smoothing beta (or
c'beta) after fitting. It is NOT exact for the t-statistic's denominator
(residual variance would also shrink under a true data-level resmooth+refit,
which this shortcut doesn't capture) — so this is a conservative/lower-bound
estimate of the true smoothing effect, not an exact replication. Temporal
smoothing is not retrofittable post-hoc and is out of scope for this test.

Mask-aware ("normalized") convolution is used throughout to avoid
zero-padding bias at the edge of the coverage mask.

Reruns the same three threshold methods, both thalamus and whole-brain
scope, for all 3 contrasts (interaction, CF_main, AM_main) — directly
comparable to 11/12/13/14_*.py's unsmoothed numbers.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import stats, ndimage
import glmlib2 as G2

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

WORK = f"{PROJECT_ROOT}/derivatives/sub-01/analysis/thalamus_work"
CKPT = f"{WORK}/_ckpt_crossed2x2_fit.npz"
INTERACTION_NPZ = f"{WORK}/cfxam_interaction_native.npz"

FBZ, FBY, FBX = 240, 320, 320
NATIVE_VOXEL_MM = 0.7  # framebox matches the 0.7mm VMR
ASSUMED_FWHM_MM = 4.0  # ASSUMPTION -- a typical default, not a matched value
SIGMA_VOX = (ASSUMED_FWHM_MM / NATIVE_VOXEL_MM) / 2.3548

ALPHA = 0.05
CLUSTER_FORMING_P = 0.001
N_MONTE_CARLO = 2000  # reduced from 5000 for this diagnostic (6 combinations to run)
RNG_SEED = 42

CONTRASTS = {
    "interaction": np.array([1.0, -1.0, -1.0, 1.0]),
    "CF_main": np.array([-1.0, -1.0, 1.0, 1.0]),
    "AM_main": np.array([-1.0, 1.0, -1.0, 1.0]),
}


def insert_framebox(vtc_raw, bbox):
    fb = np.zeros((FBZ, FBY, FBX), np.float32)
    zs, ze = bbox["ZStart"], bbox["ZEnd"]
    ys, ye = bbox["YStart"], bbox["YEnd"]
    xs, xe = bbox["XStart"], bbox["XEnd"]
    fb[zs:ze, ys:ye, xs:xe] = vtc_raw
    return fb


def documented_fb_to_native(fb):
    return np.ascontiguousarray(np.transpose(fb[::-1, ::-1, ::-1], (0, 2, 1)))


def to_native(vtc_raw, bbox):
    return documented_fb_to_native(insert_framebox(vtc_raw, bbox))


def mask_aware_smooth(t_native, mask, sigma_vox):
    """Normalized convolution: avoids zero-padding bias at mask edges."""
    num = ndimage.gaussian_filter((t_native * mask).astype(np.float64), sigma=sigma_vox)
    den = ndimage.gaussian_filter(mask.astype(np.float64), sigma=sigma_vox)
    out = np.zeros_like(t_native, dtype=np.float64)
    valid = den > 0.05
    out[valid] = num[valid] / den[valid]
    return out.astype(np.float32)


def method_uncorrected(t_vals, dof):
    tcrit = float(stats.t.isf(ALPHA / 2, dof))
    sig = np.abs(t_vals) > tcrit
    return {"method": "p<0.05 uncorrected", "t_critical": tcrit,
            "n_significant": int(sig.sum()), "n_total": int(t_vals.size),
            "fraction": float(sig.mean())}


def method_fdr_bh(t_vals, dof):
    p = 2 * stats.t.sf(np.abs(t_vals), dof)
    order = np.argsort(p)
    p_sorted = p[order]
    m = p.size
    bh_crit = np.arange(1, m + 1) / m * ALPHA
    below = p_sorted <= bh_crit
    q_thresh_p = p_sorted[np.max(np.where(below)[0])] if below.any() else 0.0
    sig = p <= q_thresh_p
    return {"method": "FDR q<0.05", "p_threshold": float(q_thresh_p),
            "n_significant": int(sig.sum()), "n_total": int(m),
            "fraction": float(sig.mean())}


def estimate_axis_fwhm(vol, mask):
    fwhms = []
    for axis in range(3):
        a = np.take(vol, range(0, vol.shape[axis] - 1), axis=axis)
        b = np.take(vol, range(1, vol.shape[axis]), axis=axis)
        ma = np.take(mask, range(0, mask.shape[axis] - 1), axis=axis)
        mb = np.take(mask, range(1, mask.shape[axis]), axis=axis)
        both = ma & mb
        if both.sum() < 100:
            fwhms.append(2.0); continue
        rho = float(np.clip(np.corrcoef(a[both], b[both])[0, 1], 1e-4, 0.999))
        fwhms.append(float(np.clip(np.sqrt(-4*np.log(2)/np.log(rho)), 0.5, 15.0)))
    return fwhms


def method_monte_carlo_cluster(t_native, roi_mask, smoothness_mask, dof, label):
    fwhm_zyx = estimate_axis_fwhm(t_native, smoothness_mask)
    sigma_zyx = [f / 2.3548 for f in fwhm_zyx]
    zz = np.where(roi_mask.any(axis=(1, 2)))[0]
    yy = np.where(roi_mask.any(axis=(0, 2)))[0]
    xx = np.where(roi_mask.any(axis=(0, 1)))[0]
    pad = 8
    z0, z1 = max(0, zz.min()-pad), min(roi_mask.shape[0], zz.max()+pad+1)
    y0, y1 = max(0, yy.min()-pad), min(roi_mask.shape[1], yy.max()+pad+1)
    x0, x1 = max(0, xx.min()-pad), min(roi_mask.shape[2], xx.max()+pad+1)
    box_mask = roi_mask[z0:z1, y0:y1, x0:x1]
    box_shape = box_mask.shape
    print(f"    [{label}] FWHM(z/y/x)={[round(f,2) for f in fwhm_zyx]} "
          f"box={box_shape} roi_vox={int(box_mask.sum())}")

    tcrit_forming = float(stats.t.isf(CLUSTER_FORMING_P / 2, dof))
    structure = np.ones((3, 3, 3), dtype=int)
    rng = np.random.default_rng(RNG_SEED)
    max_sizes = np.zeros(N_MONTE_CARLO, dtype=int)
    for i in range(N_MONTE_CARLO):
        noise = rng.standard_normal(box_shape)
        smoothed = ndimage.gaussian_filter(noise, sigma=sigma_zyx)
        smoothed = smoothed / smoothed[box_mask].std()
        sig = (np.abs(smoothed) > tcrit_forming) & box_mask
        if sig.any():
            labels, n_cl = ndimage.label(sig, structure=structure)
            if n_cl:
                sizes = ndimage.sum(sig, labels, index=np.arange(1, n_cl+1))
                max_sizes[i] = int(sizes.max())

    cluster_thresh = int(np.ceil(np.percentile(max_sizes, 95)))
    real_sig = (np.abs(t_native) > tcrit_forming) & roi_mask
    labels_real, n_cl_real = ndimage.label(real_sig, structure=structure)
    sizes_real = (ndimage.sum(real_sig, labels_real, index=np.arange(1, n_cl_real+1))
                 if n_cl_real else np.array([]))
    surviving = sizes_real >= cluster_thresh
    n_vox_surv = int(sizes_real[surviving].sum()) if surviving.any() else 0
    return {
        "cluster_forming_t_threshold": tcrit_forming,
        "estimated_fwhm_voxels_zyx": fwhm_zyx,
        "n_monte_carlo_iterations": N_MONTE_CARLO,
        "null_max_cluster_size_95th_pct": cluster_thresh,
        "real_n_clusters": int(n_cl_real),
        "real_largest_clusters": sorted([int(s) for s in sizes_real], reverse=True)[:10],
        "n_clusters_surviving": int(surviving.sum()),
        "n_voxels_surviving": n_vox_surv,
    }


def main():
    ck = np.load(CKPT)
    fit = {"betas": ck["betas"], "rss": ck["rss"], "dof": float(ck["dof"]),
           "XtXws_reg_inv": ck["XtXws_reg_inv"], "col_scale": ck["col_scale"],
           "mask": ck["mask"]}
    bbox = {"XStart": int(ck["bbox_XStart"]), "XEnd": int(ck["bbox_XEnd"]),
            "YStart": int(ck["bbox_YStart"]), "YEnd": int(ck["bbox_YEnd"]),
            "ZStart": int(ck["bbox_ZStart"]), "ZEnd": int(ck["bbox_ZEnd"])}
    n_cond = int(ck["n_cond"])
    dof = fit["dof"]

    inter = np.load(INTERACTION_NPZ)
    mask_native, thal = inter["mask_native"], inter["thal"]
    in_thal = thal & mask_native

    print(f"Assumed spatial smoothing: FWHM={ASSUMED_FWHM_MM}mm "
          f"-> sigma={SIGMA_VOX:.3f} voxels (native voxel size {NATIVE_VOXEL_MM}mm)")

    all_results = {"assumed_fwhm_mm": ASSUMED_FWHM_MM, "native_voxel_mm": NATIVE_VOXEL_MM}
    for name, c_task in CONTRASTS.items():
        print(f"\n=== {name} (smoothed) ===")
        c_full = np.zeros(fit["betas"].shape[0])
        c_full[:n_cond] = c_task
        t_map = G2.t_contrast_v2(fit, c_full)
        t_native_raw = to_native(t_map, bbox)
        t_native = mask_aware_smooth(t_native_raw, mask_native, SIGMA_VOX)

        results = {}
        for scope, roi in (("thalamus", in_thal), ("wholebrain", mask_native)):
            t_vals = t_native[roi]
            r1 = method_uncorrected(t_vals, dof)
            r2 = method_fdr_bh(t_vals, dof)
            smoothness_mask = mask_native if scope == "wholebrain" else (mask_native & (~thal))
            r3 = method_monte_carlo_cluster(t_native, roi, smoothness_mask, dof,
                                            label=f"{name}/{scope}")
            print(f"  -- {scope}: p<0.05={r1['n_significant']}/{r1['n_total']} "
                  f"({r1['fraction']*100:.2f}%), FDR={r2['n_significant']}, "
                  f"MonteCarlo: {r3['n_voxels_surviving']} voxels survive "
                  f"(largest={r3['real_largest_clusters'][:1]}, "
                  f"thresh={r3['null_max_cluster_size_95th_pct']})")
            results[scope] = {"uncorrected_p05": r1, "fdr_q05": r2,
                              "monte_carlo_cluster": r3}
        all_results[name] = results

    out_path = f"{WORK}/cfxam_smoothed_reanalysis_thresholds.json"
    with open(out_path, "w") as f:
        json.dump({"dof": dof, **all_results}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
