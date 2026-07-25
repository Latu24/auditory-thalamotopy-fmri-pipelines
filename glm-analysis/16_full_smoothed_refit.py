"""Full VTC-level smoothing + GLM refit (not the post-hoc t-map shortcut in
15_smoothed_reanalysis.py). Spatially AND temporally smooths the raw VTC data
itself before fitting — matching what BrainVoyager's SD3D (spatial) + TDTS
(temporal) preprocessing steps do — then reruns the full crossed 2x2 GLM
(interaction, CF_main, AM_main) on the smoothed data.

Values used (matched to the corrected CF-only/AM-only smoothed GLMs —
21_cf_smoothed_glm.py / 22_am_smoothed_glm.py):
  - Spatial: FWHM=1.5mm, user-specified.
  - Temporal: FWHM=1.0dp (=1 TR=1.6s). An earlier 3.0dp/4.8s value caused an
    AR(2)-clipping / anti-conservative-dof problem here as it did for the
    CF-only/AM-only GLMs (an earlier run of this script pinned phi1 at the
    +-0.9 stability clip — a symptom of an unconverged fit, not a true
    estimate). 1.0dp avoids that.

Memory safety: smoothing needs neighboring voxels across Z-slab boundaries,
so each read slab is padded with a `halo` of extra Z-slices (computed from
the spatial kernel's truncation radius), smoothed as one block (spatial x,y,z
+ temporal t combined in a single separable 4D Gaussian -- mathematically
equivalent to doing each dimension sequentially), then the halo is cropped
back off before accumulating into XtY/sumYsq.

AR(2) is RE-ESTIMATED on temporally-smoothed sample voxel timeseries (not
reused from the unsmoothed fit's checkpoint), since temporal smoothing
measurably increases autocorrelation and whitening should reflect that.

Output: cfxam_smoothed_refit_native.npz (t-maps for all 3 contrasts) +
cfxam_smoothed_refit_thresholds.json (same 3 threshold methods, same 2
scopes, directly comparable to the unsmoothed 11/12/13/14_*.py numbers and
to 15_smoothed_reanalysis.py's post-hoc-shortcut numbers).
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import stats, ndimage
import nibabel as nib
import glmlib as G
import glmlib2 as G2
import common2 as C2

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ROOT = PROJECT_ROOT
ANA = f"{ROOT}/derivatives/sub-01/analysis"
WORK = f"{ANA}/thalamus_work"
SDM_DIR = f"{ANA}/sdm"
CKPT = f"{WORK}/_ckpt_crossed2x2_fit.npz"
INTERACTION_NPZ = f"{WORK}/cfxam_interaction_native.npz"

CONDITION_ORDER = ["CFlo_AMlo", "CFlo_AMhi", "CFhi_AMlo", "CFhi_AMhi"]
CONTRASTS = {
    "interaction": np.array([1.0, -1.0, -1.0, 1.0]),
    "CF_main": np.array([-1.0, -1.0, 1.0, 1.0]),
    "AM_main": np.array([-1.0, 1.0, -1.0, 1.0]),
}

NATIVE_VOXEL_MM = 0.7
FWHM_SPATIAL_MM = 1.5          # matched to CF-only/AM-only smoothed GLMs
FWHM_TEMPORAL_DP = 1.0         # = 1 TR = 1.6s; matched to CF-only/AM-only, corrected from 3.0dp
SIGMA_SPATIAL_VOX = (FWHM_SPATIAL_MM / NATIVE_VOXEL_MM) / 2.3548
SIGMA_TEMPORAL_VOL = FWHM_TEMPORAL_DP / 2.3548

ZCHUNK = 20
HALO = 12  # >= 4*SIGMA_SPATIAL_VOX (~9.7 vox) so filter truncation is negligible

FBZ, FBY, FBX = 240, 320, 320

ALPHA = 0.05
CLUSTER_FORMING_P = 0.001
N_MONTE_CARLO = 2000
RNG_SEED = 42


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


def prep_smoothed(vtc_paths, run_task_designs, log=print, ar_sample=4000):
    """Same as glmlib2.prep_multirun_design, but AR(2) is estimated on
    TEMPORALLY-smoothed sample voxel timeseries (spatial smoothing's direct
    effect on a single voxel's own temporal ACF is second-order and skipped
    here for tractability -- see module docstring)."""
    headers = [G.read_vtc_header(p) for p in vtc_paths]
    h0 = headers[0]
    DimZ, DimY, DimX = h0["DimZ"], h0["DimY"], h0["DimX"]
    n_cond = run_task_designs[0].shape[1]
    nrun = len(vtc_paths)
    n_conf = 2
    run_nvols = [d.shape[0] for d in run_task_designs]
    Ttotal = sum(run_nvols)
    P = n_cond + nrun * n_conf
    X = np.zeros((Ttotal, P))
    row = 0
    for r in range(nrun):
        nv = run_nvols[r]
        X[row:row + nv, :n_cond] = run_task_designs[r]
        c0 = n_cond + r * n_conf
        X[row:row + nv, c0:c0 + n_conf] = G.confounds(nv)
        row += nv
    run_row_bounds = np.cumsum([0] + run_nvols)

    log("  sampling voxels + estimating AR(2) on TEMPORALLY-smoothed sample ...")
    rng = np.random.default_rng(42)
    mm0 = G.vtc_memmap(vtc_paths[0], headers[0])
    tmean = np.asarray(mm0[:, :, :, ::8].mean(-1), dtype=np.float32)
    pos = tmean[tmean > 0]
    mask_thresh = float(np.percentile(pos, 40)) if pos.size else 0.0
    mask = tmean > mask_thresh
    vox_idx = np.argwhere(mask)
    if vox_idx.shape[0] > ar_sample:
        sel = rng.choice(vox_idx.shape[0], ar_sample, replace=False)
        vox_idx = vox_idx[sel]
    Ysamp = np.zeros((Ttotal, vox_idx.shape[0]), dtype=np.float64)
    for r in range(nrun):
        mm = G.vtc_memmap(vtc_paths[r], headers[r])
        a, b = run_row_bounds[r], run_row_bounds[r + 1]
        raw = np.asarray(mm[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2], :].T,
                         dtype=np.float64)
        # temporal-only smoothing of each sampled voxel's own timecourse
        smoothed = ndimage.gaussian_filter1d(raw, sigma=SIGMA_TEMPORAL_VOL, axis=0)
        Ysamp[a:b] = smoothed
        del mm

    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta_s = XtX_inv @ (X.T @ Ysamp)
    resid = Ysamp - X @ beta_s
    phis = []
    for r in range(nrun):
        a, b = run_row_bounds[r], run_row_bounds[r + 1]
        phis.append(G.estimate_ar2(resid[a:b]))
    phi1 = float(np.mean([p[0] for p in phis]))
    phi2 = float(np.mean([p[1] for p in phis]))
    log(f"  AR(2) on smoothed sample: phi1={phi1:.4f} phi2={phi2:.4f} "
        f"(per-run {[(round(p[0],3),round(p[1],3)) for p in phis]})")

    Xw = X.copy()
    for r in range(nrun):
        a, b = run_row_bounds[r], run_row_bounds[r + 1]
        Xw[a:b] = G.ar2_whiten(X[a:b], phi1, phi2)

    return {
        "vtc_paths": vtc_paths, "headers": headers,
        "DimZ": DimZ, "DimY": DimY, "DimX": DimX,
        "n_cond": n_cond, "nrun": nrun, "n_conf": n_conf, "P": P,
        "run_row_bounds": run_row_bounds, "Ttotal": Ttotal,
        "Xw": Xw, "phi": (phi1, phi2), "mask": mask,
    }


def fit_smoothed(prep, log=print, zchunk=ZCHUNK, halo=HALO,
                 sigma_spatial=SIGMA_SPATIAL_VOX, sigma_temporal=SIGMA_TEMPORAL_VOL):
    vtc_paths = prep["vtc_paths"]; headers = prep["headers"]
    DimZ, DimY, DimX = prep["DimZ"], prep["DimY"], prep["DimX"]
    n_cond, nrun, n_conf, P = prep["n_cond"], prep["nrun"], prep["n_conf"], prep["P"]
    run_row_bounds = prep["run_row_bounds"]
    Xw = prep["Xw"]
    mask = prep["mask"]

    Nvox = DimZ * DimY * DimX
    XtY = np.zeros((P, Nvox), dtype=np.float32)
    sumYsq = np.zeros(Nvox, dtype=np.float32)

    t_start = time.time()
    for r in range(nrun):
        mm = G.vtc_memmap(vtc_paths[r], headers[r])
        a, b = run_row_bounds[r], run_row_bounds[r + 1]
        Xwr = Xw[a:b]
        z = 0
        while z < DimZ:
            z0, z1 = z, min(z + zchunk, DimZ)
            zp0, zp1 = max(0, z0 - halo), min(DimZ, z1 + halo)
            crop_lo, crop_hi = z0 - zp0, z1 - zp0  # where the real chunk sits inside the padded read
            slab = np.asarray(mm[zp0:zp1], dtype=np.float32)  # (zp, DimY, DimX, nvols)
            # joint separable 4D Gaussian: spatial (z,y,x) + temporal (t) in one pass
            smoothed = ndimage.gaussian_filter(
                slab, sigma=(sigma_spatial, sigma_spatial, sigma_spatial, sigma_temporal))
            del slab
            cropped = smoothed[crop_lo:crop_hi]
            del smoothed
            zc = z1 - z0
            Y = cropped.reshape(-1, cropped.shape[-1]).T  # (nvols, zc*DimY*DimX)
            del cropped
            Yw = G.ar2_whiten(Y, prep["phi"][0], prep["phi"][1])
            del Y
            off = z0 * DimY * DimX
            n_sl = zc * DimY * DimX
            XtY[:, off:off + n_sl] += (Xwr.T @ Yw).astype(np.float32)
            sumYsq[off:off + n_sl] += np.einsum("tn,tn->n", Yw, Yw).astype(np.float32)
            del Yw
            z = z1
        del mm
        log(f"    run {r+1}/{nrun} accumulated ({time.time()-t_start:.0f}s elapsed)")

    XtXw = Xw.T @ Xw
    XtXw_inv = np.linalg.pinv(XtXw)
    betas = (XtXw_inv.astype(np.float32) @ XtY)
    rss = sumYsq.astype(np.float64) - np.einsum(
        "pn,pn->n", betas.astype(np.float64), XtY.astype(np.float64))
    rss = np.clip(rss, 0, None)
    dof = prep["Ttotal"] - P
    shape3 = (DimZ, DimY, DimX)
    return {"betas": betas.reshape((P,) + shape3), "rss": rss.reshape(shape3),
           "dof": dof, "XtXw_inv": XtXw_inv, "mask": mask, "P": P,
           "n_cond": n_cond, "headers": headers, "shape3": shape3}


def t_contrast(fit, c):
    c = np.asarray(c, dtype=np.float64)
    betas = fit["betas"]
    cb = np.tensordot(c, betas, axes=([0], [0]))
    var_unit = float(c @ fit["XtXw_inv"] @ c)
    s2 = fit["rss"] / fit["dof"]
    denom = np.sqrt(np.clip(var_unit * s2, 1e-20, None))
    t = cb / denom
    t[~fit["mask"]] = 0.0
    t[~np.isfinite(t)] = 0.0
    return t


def method_uncorrected(t_vals, dof):
    tcrit = float(stats.t.isf(ALPHA / 2, dof))
    sig = np.abs(t_vals) > tcrit
    return {"t_critical": tcrit, "n_significant": int(sig.sum()),
            "n_total": int(t_vals.size), "fraction": float(sig.mean())}


def method_fdr_bh(t_vals, dof):
    p = 2 * stats.t.sf(np.abs(t_vals), dof)
    order = np.argsort(p)
    p_sorted = p[order]
    m = p.size
    bh_crit = np.arange(1, m + 1) / m * ALPHA
    below = p_sorted <= bh_crit
    q_thresh_p = p_sorted[np.max(np.where(below)[0])] if below.any() else 0.0
    sig = p <= q_thresh_p
    return {"p_threshold": float(q_thresh_p), "n_significant": int(sig.sum()),
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
    print(f"    [{label}] FWHM(z/y/x)={[round(f,2) for f in fwhm_zyx]} "
          f"box={box_mask.shape} roi_vox={int(box_mask.sum())}")
    tcrit_forming = float(stats.t.isf(CLUSTER_FORMING_P / 2, dof))
    structure = np.ones((3, 3, 3), dtype=int)
    rng = np.random.default_rng(RNG_SEED)
    max_sizes = np.zeros(N_MONTE_CARLO, dtype=int)
    for i in range(N_MONTE_CARLO):
        noise = rng.standard_normal(box_mask.shape)
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
    return {"cluster_forming_t_threshold": tcrit_forming,
            "estimated_fwhm_voxels_zyx": fwhm_zyx,
            "null_max_cluster_size_95th_pct": cluster_thresh,
            "real_n_clusters": int(n_cl_real),
            "real_largest_clusters": sorted([int(s) for s in sizes_real], reverse=True)[:10],
            "n_clusters_surviving": int(surviving.sum()),
            "n_voxels_surviving": n_vox_surv}


def main():
    print(f"Spatial: FWHM={FWHM_SPATIAL_MM}mm (matched to CF/AM-only) -> sigma={SIGMA_SPATIAL_VOX:.3f} vox")
    print(f"Temporal: FWHM={FWHM_TEMPORAL_DP}dp (matched to CF/AM-only) -> sigma={SIGMA_TEMPORAL_VOL:.3f} vol")
    print(f"zchunk={ZCHUNK}, halo={HALO} "
          f"(padded slab peak ~{(ZCHUNK+2*HALO)*92*215*350*4/1e9:.2f} GB worst-case run)")

    designs = []
    for r in (1, 2, 3, 4):
        X, names, nv = G2.read_sdm_predictors(
            f"{SDM_DIR}/sub-01_run-{r}_CFxAM_2x2_crossed_nilearn.sdm", CONDITION_ORDER)
        designs.append(X)

    print("\n[smoothed-refit] prepping design + AR(2) on temporally-smoothed sample ...")
    prep = prep_smoothed(C2.VTCS_REFINED, designs, log=print)

    print("\n[smoothed-refit] fitting voxel-wise on FULLY SMOOTHED data "
          "(this streams all 4 VTCs again with padded reads -- will take a while) ...")
    fit = fit_smoothed(prep, log=print)
    hdr = fit["headers"][0]
    bbox = {k: hdr[k] for k in ("XStart","XEnd","YStart","YEnd","ZStart","ZEnd")}
    mask_native = to_native(fit["mask"].astype(np.float32), bbox) > 0.5

    thal = np.asarray(nib.load(f"{WORK}/sub-01_thalamus_native.nii.gz").dataobj) > 0
    in_thal = thal & mask_native
    dof = fit["dof"]
    print(f"[smoothed-refit] dof={dof}, coverage voxels: thalamus={int(in_thal.sum())}, "
          f"wholebrain={int(mask_native.sum())}")

    # checkpoint immediately -- the fit just streamed several GB per run with
    # padded reads; don't lose that if anything below has a bug.
    ckpt_path = f"{WORK}/_ckpt_smoothed_refit_fit.npz"
    np.savez(ckpt_path, betas=fit["betas"], rss=fit["rss"], dof=fit["dof"],
             XtXw_inv=fit["XtXw_inv"], mask=fit["mask"], P=fit["P"],
             n_cond=fit["n_cond"], **{f"bbox_{k}": v for k, v in bbox.items()})
    print(f"[smoothed-refit] checkpointed fit essentials to {ckpt_path}")

    native_maps = {}
    all_results = {"fwhm_spatial_mm": FWHM_SPATIAL_MM,
                   "fwhm_temporal_dp": FWHM_TEMPORAL_DP, "dof": float(dof)}
    for name, c_task in CONTRASTS.items():
        print(f"\n=== {name} (full smoothed refit) ===")
        c_full = np.zeros(fit["P"])
        c_full[:fit["n_cond"]] = c_task
        t_map = t_contrast(fit, c_full)
        t_native = to_native(t_map, bbox)
        native_maps[name] = t_native

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
                  f"MonteCarlo voxels survive={r3['n_voxels_surviving']} "
                  f"(largest real cluster={r3['real_largest_clusters'][:1]}, "
                  f"null thresh={r3['null_max_cluster_size_95th_pct']})")
            results[scope] = {"uncorrected_p05": r1, "fdr_q05": r2,
                              "monte_carlo_cluster": r3}
        all_results[name] = results

    np.savez(f"{WORK}/cfxam_smoothed_refit_native.npz",
             mask_native=mask_native, thal=thal, **native_maps)
    out_path = f"{WORK}/cfxam_smoothed_refit_thresholds.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {out_path}")
    print(f"wrote {WORK}/cfxam_smoothed_refit_native.npz")


if __name__ == "__main__":
    main()
