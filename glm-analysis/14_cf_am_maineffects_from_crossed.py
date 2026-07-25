"""CF-main and AM-main effect contrasts computed from the SAME crossed 2x2
GLM fit as the interaction test (11_glm_crossed_2x2_interaction.py) -- reuses
the checkpointed betas (_ckpt_crossed2x2_fit.npz), no VTC re-read needed.

Condition order: [CFlo_AMlo, CFlo_AMhi, CFhi_AMlo, CFhi_AMhi]
  CF main effect: c = [-1, -1, +1, +1]  (mean(CFhi) - mean(CFlo))
  AM main effect: c = [-1, +1, -1, +1]  (mean(AMhi) - mean(AMlo))
(t-statistic is invariant to positive rescaling of c, so unweighted +-1 is
equivalent to the properly-averaged 0.5-weighted contrast.)

Unlike the earlier 45-predictor marginal design (severely collinear,
cond~1e9, CF and AM there were separate partitions of the same underlying
events), THIS design's CF/AM main effects come from a well-conditioned
crossed fit (cond~4.6) where all 4 joint cells are estimated together -- a
cleaner, independent replication check on whether this smaller/differently
binned design still recovers the known CF/AM tuning already established
with the marginal design (54 CF-sig / 189 AM-sig voxels @ p<0.01
uncorrected omnibus-F).

Runs the same three threshold methods as 12/13_threshold_interaction*.py,
both thalamus-restricted and whole-brain, for both contrasts (4 runs total).
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

ALPHA = 0.05
CLUSTER_FORMING_P = 0.001
N_MONTE_CARLO = 5000
RNG_SEED = 42

CONTRASTS = {
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


def method_uncorrected(t_vals, dof):
    tcrit = float(stats.t.isf(ALPHA / 2, dof))
    sig = np.abs(t_vals) > tcrit
    return {"method": "p<0.05 uncorrected (two-tailed)", "t_critical": tcrit,
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
    return {"method": "FDR q<0.05 (Benjamini-Hochberg, two-tailed)",
            "p_threshold": float(q_thresh_p), "n_significant": int(sig.sum()),
            "n_total": int(m), "fraction": float(sig.mean())}


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
        rho = float(np.clip(np.corrcoef(av, bv)[0, 1], 1e-4, 0.999))
        fwhm = np.sqrt(-4 * np.log(2) / np.log(rho))
        fwhms.append(float(np.clip(fwhm, 0.5, 10.0)))
    return fwhms


def method_monte_carlo_cluster(t_native, roi_mask, smoothness_mask, dof, label):
    fwhm_zyx = estimate_axis_fwhm(t_native, smoothness_mask)
    sigma_zyx = [f / 2.3548 for f in fwhm_zyx]
    print(f"    [{label}] smoothness FWHM (z/y/x): {[round(f,2) for f in fwhm_zyx]}")

    zz = np.where(roi_mask.any(axis=(1, 2)))[0]
    yy = np.where(roi_mask.any(axis=(0, 2)))[0]
    xx = np.where(roi_mask.any(axis=(0, 1)))[0]
    pad = 6
    z0, z1 = max(0, zz.min() - pad), min(roi_mask.shape[0], zz.max() + pad + 1)
    y0, y1 = max(0, yy.min() - pad), min(roi_mask.shape[1], yy.max() + pad + 1)
    x0, x1 = max(0, xx.min() - pad), min(roi_mask.shape[2], xx.max() + pad + 1)
    box_mask = roi_mask[z0:z1, y0:y1, x0:x1]
    box_shape = box_mask.shape
    print(f"    [{label}] sim box {box_shape}, {int(box_mask.sum())} roi voxels")

    tcrit_forming = float(stats.t.isf(CLUSTER_FORMING_P / 2, dof))
    structure = np.ones((3, 3, 3), dtype=int)
    rng = np.random.default_rng(RNG_SEED)
    max_cluster_sizes = np.zeros(N_MONTE_CARLO, dtype=int)
    for i in range(N_MONTE_CARLO):
        noise = rng.standard_normal(box_shape)
        smoothed = ndimage.gaussian_filter(noise, sigma=sigma_zyx)
        smoothed = smoothed / smoothed[box_mask].std()
        sig = (np.abs(smoothed) > tcrit_forming) & box_mask
        if sig.any():
            labels, n_cl = ndimage.label(sig, structure=structure)
            if n_cl:
                sizes = ndimage.sum(sig, labels, index=np.arange(1, n_cl + 1))
                max_cluster_sizes[i] = int(sizes.max())
        if (i + 1) % 2000 == 0:
            print(f"    [{label}] ... {i+1}/{N_MONTE_CARLO}")

    cluster_thresh = int(np.ceil(np.percentile(max_cluster_sizes, 95)))
    real_sig = (np.abs(t_native) > tcrit_forming) & roi_mask
    labels_real, n_cl_real = ndimage.label(real_sig, structure=structure)
    sizes_real = (ndimage.sum(real_sig, labels_real, index=np.arange(1, n_cl_real + 1))
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
        "null_summary": {"mean": float(max_cluster_sizes.mean()),
                         "median": float(np.median(max_cluster_sizes)),
                         "max": int(max_cluster_sizes.max())},
        "real_n_clusters_at_primary_threshold": int(n_cl_real),
        "real_largest_clusters": sorted([int(s) for s in sizes_real], reverse=True)[:20],
        "n_clusters_surviving": int(surviving.sum()),
        "n_voxels_surviving": n_voxels_surviving,
    }


def main():
    ck = np.load(CKPT)
    fit = {
        "betas": ck["betas"], "rss": ck["rss"], "dof": float(ck["dof"]),
        "XtXws_reg_inv": ck["XtXws_reg_inv"], "col_scale": ck["col_scale"],
        "mask": ck["mask"],
    }
    bbox = {"XStart": int(ck["bbox_XStart"]), "XEnd": int(ck["bbox_XEnd"]),
            "YStart": int(ck["bbox_YStart"]), "YEnd": int(ck["bbox_YEnd"]),
            "ZStart": int(ck["bbox_ZStart"]), "ZEnd": int(ck["bbox_ZEnd"])}
    n_cond = int(ck["n_cond"])
    dof = fit["dof"]

    inter = np.load(INTERACTION_NPZ)
    mask_native, thal = inter["mask_native"], inter["thal"]
    in_thal = thal & mask_native

    all_results = {}
    for name, c_task in CONTRASTS.items():
        print(f"\n=== {name} ===")
        c_full = np.zeros(fit["betas"].shape[0])
        c_full[:n_cond] = c_task
        t_map = G2.t_contrast_v2(fit, c_full)
        t_native = to_native(t_map, bbox)

        results = {}
        for scope, roi in (("thalamus", in_thal), ("wholebrain", mask_native)):
            t_vals = t_native[roi]
            print(f"  -- {scope} ({t_vals.size} voxels) --")
            r1 = method_uncorrected(t_vals, dof)
            print(f"    p<0.05 uncorrected: {r1['n_significant']}/{r1['n_total']} "
                  f"({r1['fraction']*100:.2f}%)")
            r2 = method_fdr_bh(t_vals, dof)
            print(f"    FDR q<0.05: {r2['n_significant']} voxels")
            smoothness_mask = mask_native if scope == "wholebrain" else (mask_native & (~thal))
            r3 = method_monte_carlo_cluster(t_native, roi, smoothness_mask, dof,
                                            label=f"{name}/{scope}")
            print(f"    Monte Carlo: {r3['n_clusters_surviving']} clusters, "
                  f"{r3['n_voxels_surviving']} voxels survive "
                  f"(largest real cluster={r3['real_largest_clusters'][:1]}, "
                  f"null 95th pct={r3['null_max_cluster_size_95th_pct']})")
            results[scope] = {"uncorrected_p05": r1, "fdr_q05": r2,
                              "monte_carlo_cluster": r3}
        all_results[name] = results

    out_path = f"{WORK}/cf_am_maineffects_from_crossed_thresholds.json"
    with open(out_path, "w") as f:
        json.dump({"dof": dof, **all_results}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
