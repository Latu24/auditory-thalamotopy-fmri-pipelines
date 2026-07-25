"""Sanity-test GLM: the earlier marginal 45-predictor design (36 CF + 9 AM
conditions in a single combined GLM), refit WITH spatial + temporal
smoothing applied directly to the VTC data, using the same smoothing
machinery as 16_full_smoothed_refit.py. Spatial FWHM=1.0mm is a conservative
value (revised down from an initial 4.0mm, then 2.0mm, after visual
inspection showed over-smoothing); temporal FWHM=3.0dp is a commonly used
BrainVoyager default, kept here for comparison against the CF-only/AM-only
exports' corrected 1.0dp value (see 21/22_*_smoothed_glm.py).

This design is severely collinear (cond~1e9, established in
07_combined_ridge.py — CF and AM partition the same underlying events) even
before smoothing, and smoothing only adds spatial redundancy on top — so
this uses RIDGE regularization + GCV lambda search, exactly like
07_combined_ridge.py, and for the same reason produces DESCRIPTIVE outputs
only (beta maps, mean-beta, best-condition), not t/F stat maps — ridge
estimates are biased by construction and classical inference doesn't apply
to them (see 07_combined_ridge.py's own documented inference_caveat).

Memory-safe padded-slab smoothing identical to 16_full_smoothed_refit.py.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import ndimage
import glmlib as G
import glmlib2 as G2
import common2 as C2
from glm_writer import write_glm

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ROOT = PROJECT_ROOT
ANA = f"{ROOT}/derivatives/sub-01/analysis"
OUT_DIR = f"{ANA}/smoothed_glm"
os.makedirs(OUT_DIR, exist_ok=True)

NATIVE_VOXEL_MM = 0.7
FWHM_SPATIAL_MM = 1.0          # revised down from 4.0mm -> 2.0mm -> 1.0mm, too strong per visual inspection
FWHM_TEMPORAL_DP = 3.0
SIGMA_SPATIAL_VOX = (FWHM_SPATIAL_MM / NATIVE_VOXEL_MM) / 2.3548
SIGMA_TEMPORAL_VOL = FWHM_TEMPORAL_DP / 2.3548
ZCHUNK = 20
HALO = 12


def prep_smoothed_marginal(vtc_paths, run_task_designs, log=print, ar_sample=4000):
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
        X[row:row+nv, :n_cond] = run_task_designs[r]
        c0 = n_cond + r * n_conf
        X[row:row+nv, c0:c0+n_conf] = G.confounds(nv)
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
        a, b = run_row_bounds[r], run_row_bounds[r+1]
        raw = np.asarray(mm[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2], :].T, dtype=np.float64)
        Ysamp[a:b] = ndimage.gaussian_filter1d(raw, sigma=SIGMA_TEMPORAL_VOL, axis=0)
        del mm

    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta_s = XtX_inv @ (X.T @ Ysamp)
    resid = Ysamp - X @ beta_s
    phis = []
    for r in range(nrun):
        a, b = run_row_bounds[r], run_row_bounds[r+1]
        phis.append(G.estimate_ar2(resid[a:b]))
    phi1 = float(np.mean([p[0] for p in phis]))
    phi2 = float(np.mean([p[1] for p in phis]))
    log(f"  AR(2) on smoothed sample: phi1={phi1:.4f} phi2={phi2:.4f} "
        f"(per-run {[(round(p[0],3),round(p[1],3)) for p in phis]})")

    Xw = X.copy()
    for r in range(nrun):
        a, b = run_row_bounds[r], run_row_bounds[r+1]
        Xw[a:b] = G.ar2_whiten(X[a:b], phi1, phi2)

    Yw = Ysamp.copy()
    for r in range(nrun):
        a, b = run_row_bounds[r], run_row_bounds[r+1]
        Yw[a:b] = G.ar2_whiten(Ysamp[a:b], phi1, phi2)

    return {"vtc_paths": vtc_paths, "headers": headers, "DimZ": DimZ, "DimY": DimY,
           "DimX": DimX, "n_cond": n_cond, "nrun": nrun, "n_conf": n_conf, "P": P,
           "run_row_bounds": run_row_bounds, "Ttotal": Ttotal, "Xw": Xw, "Ysamp_w": Yw,
           "phi": (phi1, phi2), "mask": mask}


def fit_smoothed_ridge(prep, ridge_lambda, log=print, zchunk=ZCHUNK, halo=HALO):
    vtc_paths = prep["vtc_paths"]; headers = prep["headers"]
    DimZ, DimY, DimX = prep["DimZ"], prep["DimY"], prep["DimX"]
    n_cond, nrun, n_conf, P = prep["n_cond"], prep["nrun"], prep["n_conf"], prep["P"]
    run_row_bounds = prep["run_row_bounds"]
    Xw = prep["Xw"]; mask = prep["mask"]

    col_scale = np.ones(P)
    if ridge_lambda > 0:
        task_norms = np.linalg.norm(Xw[:, :n_cond], axis=0)
        task_norms[task_norms == 0] = 1.0
        col_scale[:n_cond] = task_norms
    Xws = Xw / col_scale[None, :]
    XtXws = Xws.T @ Xws
    D = np.zeros((P, P))
    if ridge_lambda > 0:
        D[np.arange(n_cond), np.arange(n_cond)] = 1.0
    XtXws_reg = XtXws + ridge_lambda * D
    XtXws_reg_inv = np.linalg.pinv(XtXws_reg)

    Nvox = DimZ * DimY * DimX
    XtY = np.zeros((P, Nvox), dtype=np.float32)
    sumYsq = np.zeros(Nvox, dtype=np.float32)
    t0 = time.time()
    for r in range(nrun):
        mm = G.vtc_memmap(vtc_paths[r], headers[r])
        a, b = run_row_bounds[r], run_row_bounds[r+1]
        Xwr = Xws[a:b]
        z = 0
        while z < DimZ:
            z0, z1 = z, min(z + zchunk, DimZ)
            zp0, zp1 = max(0, z0-halo), min(DimZ, z1+halo)
            crop_lo, crop_hi = z0-zp0, z1-zp0
            slab = np.asarray(mm[zp0:zp1], dtype=np.float32)
            smoothed = ndimage.gaussian_filter(
                slab, sigma=(SIGMA_SPATIAL_VOX, SIGMA_SPATIAL_VOX, SIGMA_SPATIAL_VOX, SIGMA_TEMPORAL_VOL))
            del slab
            cropped = smoothed[crop_lo:crop_hi]
            del smoothed
            zc = z1 - z0
            Y = cropped.reshape(-1, cropped.shape[-1]).T
            del cropped
            Yw = G.ar2_whiten(Y, prep["phi"][0], prep["phi"][1])
            del Y
            off = z0 * DimY * DimX
            n_sl = zc * DimY * DimX
            XtY[:, off:off+n_sl] += (Xwr.T @ Yw).astype(np.float32)
            sumYsq[off:off+n_sl] += np.einsum("tn,tn->n", Yw, Yw).astype(np.float32)
            del Yw
            z = z1
        del mm
        log(f"    run {r+1}/{nrun} accumulated ({time.time()-t0:.0f}s elapsed)")

    betas_scaled = XtXws_reg_inv.astype(np.float32) @ XtY
    rss = sumYsq.astype(np.float64) - np.einsum(
        "pn,pn->n", betas_scaled.astype(np.float64), XtY.astype(np.float64))
    rss = np.clip(rss, 0, None)
    betas = betas_scaled.astype(np.float64) / col_scale[:, None]

    if ridge_lambda > 0:
        eff_df = float(np.trace(XtXws_reg_inv @ XtXws))
        dof = prep["Ttotal"] - eff_df
    else:
        dof = prep["Ttotal"] - P
    shape3 = (DimZ, DimY, DimX)
    return {"betas": betas.reshape((P,)+shape3), "rss": rss.reshape(shape3),
           "dof": dof, "XtXws_reg_inv": XtXws_reg_inv, "col_scale": col_scale,
           "mask": mask, "P": P, "n_cond": n_cond, "headers": headers,
           "ridge_lambda": ridge_lambda, "shape3": shape3}


def gcv_select_lambda_smoothed(prep, lambda_grid, log=print):
    Xw = prep["Xw"]; Yw = prep["Ysamp_w"]; n_cond = prep["n_cond"]
    Ttotal, P = Xw.shape
    task_norms = np.linalg.norm(Xw[:, :n_cond], axis=0)
    task_norms[task_norms == 0] = 1.0
    col_scale = np.ones(P); col_scale[:n_cond] = task_norms
    Xws = Xw / col_scale[None, :]
    XtXws = Xws.T @ Xws
    D = np.zeros((P, P)); D[np.arange(n_cond), np.arange(n_cond)] = 1.0
    XtY = Xws.T @ Yw
    diagnostics, best = [], None
    for lam in lambda_grid:
        M = XtXws + lam * D
        Minv = np.linalg.pinv(M)
        eff_df = float(np.trace(Minv @ XtXws))
        betas = Minv @ XtY
        fitted = Xws @ betas
        rss = float(np.sum((Yw - fitted) ** 2))
        n = Ttotal * Yw.shape[1]
        denom = max((1.0 - eff_df / Ttotal) ** 2, 1e-8)
        gcv = (rss / n) / denom
        diagnostics.append({"lambda": float(lam), "eff_df": eff_df, "gcv": gcv})
        if best is None or gcv < best[1]:
            best = (lam, gcv)
    log(f"  GCV best lambda={best[0]:g} (gcv={best[1]:.6g})")
    for d in diagnostics:
        log(f"    lambda={d['lambda']:>10.4g}  eff_df={d['eff_df']:>7.2f}  gcv={d['gcv']:.6g}")
    return best[0], diagnostics


def main():
    cf_order = C2.FREQ_ORDER
    am_order = C2.AM_ORDER
    combined_order = cf_order + am_order
    n_cond = len(combined_order)

    designs = []
    for cf_sdm, am_sdm, vtc in zip(C2.CF_SDM, C2.AM_SDM, C2.VTCS_REFINED):
        Xc, names, nv = G2.combined_design_from_sdms(cf_sdm, am_sdm, cf_order, am_order)
        h = G.read_vtc_header(vtc)
        assert h["DimT"] == nv
        designs.append(Xc)

    print(f"Spatial: FWHM={FWHM_SPATIAL_MM}mm -> sigma={SIGMA_SPATIAL_VOX:.3f} vox")
    print(f"Temporal: FWHM={FWHM_TEMPORAL_DP}dp -> sigma={SIGMA_TEMPORAL_VOL:.3f} vol")
    print(f"cond(run1 45-col raw task design) = {np.linalg.cond(designs[0]):.3e} "
          f"(severely collinear, as established -- ridge required)")

    print("\n[sanity-45cond] prepping + AR(2) on temporally-smoothed sample ...")
    prep = prep_smoothed_marginal(C2.VTCS_REFINED, designs, log=print)

    lambda_grid = np.concatenate([[0.0], np.logspace(-2, 4, 13)])
    print(f"\n[sanity-45cond] GCV lambda search on smoothed sample ...")
    best_lambda, gcv_diag = gcv_select_lambda_smoothed(prep, lambda_grid, log=print)

    print(f"\n[sanity-45cond] fitting voxel-wise on FULLY SMOOTHED data, "
          f"ridge_lambda={best_lambda:g} (streams all 4 VTCs, padded reads) ...")
    fit = fit_smoothed_ridge(prep, best_lambda, log=print)
    hdr = fit["headers"][0]
    bbox = {k: hdr[k] for k in ("XStart","XEnd","YStart","YEnd","ZStart","ZEnd")}
    mask = fit["mask"]
    print(f"[sanity-45cond] dof(informational only, ridge biased)={fit['dof']:.1f}, "
          f"mask voxels={int(mask.sum())}")

    ckpt_path = f"{ANA}/thalamus_work/_ckpt_smoothed_45cond_ridge_fit.npz"
    np.savez(ckpt_path, betas=fit["betas"], rss=fit["rss"], dof=fit["dof"],
             XtXws_reg_inv=fit["XtXws_reg_inv"], col_scale=fit["col_scale"],
             mask=mask, P=fit["P"], n_cond=fit["n_cond"], ridge_lambda=best_lambda,
             **{f"bbox_{k}": v for k, v in bbox.items()})
    print(f"[sanity-45cond] checkpointed to {ckpt_path}")

    betas_cond = fit["betas"][:n_cond]
    beta_absmax = float(np.percentile(np.abs(betas_cond[:, mask]), 99))
    mean_beta = betas_cond.mean(axis=0)
    beta_range = betas_cond.max(axis=0) - betas_cond.min(axis=0)
    best_idx = np.argmax(betas_cond, axis=0).astype(np.float32) + 1.0
    range_thr = float(np.percentile(beta_range[mask], 75))
    desc_sig = (beta_range > range_thr) & mask
    best_map = np.where(desc_sig, best_idx, 0.0).astype(np.float32)

    maps = []
    for i, name in enumerate(combined_order):
        maps.append({"name": f"beta_{name}_SMOOTHED", "data": betas_cond[i],
                    "type": 1, "df1": 0, "threshold": 0.0, "upper": beta_absmax,
                    "showposneg": 3})
    maps.append({"name": "combinedCFAM_SMOOTHED mean-beta (descriptive)",
                "data": mean_beta, "type": 1, "df1": 0, "threshold": 0.0,
                "upper": beta_absmax, "showposneg": 3})
    maps.append({"name": "combinedCFAM_SMOOTHED beta-range (descriptive)",
                "data": beta_range, "type": 1, "df1": 0, "threshold": 0.0,
                "upper": float(np.percentile(beta_range[mask], 99))})
    maps.append({"name": "combinedCFAM_SMOOTHED best-cond idx (DESCRIPTIVE, top-quartile beta-range)",
                "data": best_map, "type": 1, "df1": 0, "threshold": 0.5,
                "upper": float(n_cond), "showposneg": 1})

    hdr_for_vmp = {"XStart": bbox["XStart"], "XEnd": bbox["XEnd"],
                  "YStart": bbox["YStart"], "YEnd": bbox["YEnd"],
                  "ZStart": bbox["ZStart"], "ZEnd": bbox["ZEnd"],
                  "VTC resolution": 1, "DimX": fit["shape3"][2],
                  "DimY": fit["shape3"][1], "DimZ": fit["shape3"][0]}
    vmp_path = f"{OUT_DIR}/sub-01_combinedCFAM_45cond_SMOOTHED_ridge.vmp"
    G.write_stat_vmp(vmp_path, maps, hdr_for_vmp,
                     vtc_name="sub-01_run-*_preproc_coreg-refined.vtc (smoothed in-memory)",
                     prt_name="combined_CF+AM (45cond, sanity test)")
    print(f"[sanity-45cond] wrote {vmp_path} ({len(maps)} sub-maps)")

    log = {
        "analysis": "combinedCFAM_45cond_SMOOTHED_ridge_sanity_test",
        "purpose": "Replicate the earlier marginal 36-CF+9-AM design with typical "
                  "BrainVoyager smoothing defaults, as a direct comparison point -- "
                  "not the crossed 2x2 interaction design.",
        "fwhm_spatial_mm": FWHM_SPATIAL_MM,
        "fwhm_temporal_dp": FWHM_TEMPORAL_DP,
        "condition_order": combined_order,
        "collinearity": {"cond_run1_45col_raw": float(np.linalg.cond(designs[0]))},
        "gcv_lambda_grid": gcv_diag, "chosen_lambda": best_lambda,
        "phi_ar2": fit["phi"] if "phi" in fit else prep["phi"],
        "dof_informational_ridge_biased": float(fit["dof"]),
        "mask_voxels": int(mask.sum()),
        "inference_caveat": ("Ridge estimates are BIASED by construction; no t/F stat "
                            "map produced, matching 07_combined_ridge.py's own documented "
                            "convention for this same collinear design."),
        "glm_ckpt": ckpt_path, "vmp": vmp_path,
    }
    with open(f"{ANA}/thalamus_work/smoothed_45cond_sanity_stats.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"wrote {ANA}/thalamus_work/smoothed_45cond_sanity_stats.json")


if __name__ == "__main__":
    main()
