"""CF-ONLY smoothed GLM (36 conditions), separated from AM entirely -- unlike
the combined 45-predictor design, CF-only is well-conditioned on its own
(cond(X)~5.5, cond(X+confounds)~42, verified before writing this script), so
NO RIDGE is needed. This means BrainVoyager's live per-predictor t-computation
(which always assumes classical unbiased OLS, regardless of what's actually
stored) is now genuinely VALID for this file -- unlike the combined
45-condition ridge GLM, where that same live computation was silently invalid.

Smoothing matched to the AM-only export (22_am_smoothed_glm.py): spatial
FWHM=1.5mm (user-specified), temporal FWHM=1.0dp (=1 TR=1.6s). Temporal
FWHM was brought back down from an initial 3.0dp/4.8s test: at 3.0dp the
Yule-Walker AR(2) estimate pinned at the +-0.9 stability clip identically
across all 4 runs (not a converged fit), and nominal dof=Ttotal-P doesn't
discount the effective-independent-timepoint loss from that much temporal
smoothing, so s2=rss/dof came out anti-conservative (AM omnibus F-max
78.9->321.6 between 1.0dp and 3.0dp with dof unchanged -- smoothing alone
doesn't create real signal, so that jump is the bias, not stronger
activation). At 1.0dp, AR(2) converges to distinct per-run values (no
clipping) and is a light, safe compromise. Same memory-safe padded-slab
approach as 16_full_smoothed_refit.py.

Produces: per-condition beta maps, per-condition t-maps (valid now), and an
omnibus F-map (all 36 CF conditions vs. implicit baseline) -- all with a
single well-matched color ceiling since this file no longer mixes CF and AM
magnitude scales.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import ndimage, stats
import glmlib as G
import glmlib2 as G2
import common2 as C2
from glm_writer import write_glm
from vmp_fdr_table import compute_fdr_table

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ANA = f"{PROJECT_ROOT}/derivatives/sub-01/analysis"
WORK = f"{ANA}/thalamus_work"
OUT_DIR = f"{ANA}/smoothed_glm"
os.makedirs(OUT_DIR, exist_ok=True)

BLOCK_NAME = "CF"
SDM_PATHS = C2.CF_SDM
CONDITION_ORDER = C2.FREQ_ORDER
ANAT_V2_Z_OFFSET = 40

NATIVE_VOXEL_MM = 0.7
FWHM_SPATIAL_MM = 1.5          # user-specified, matched with AM-only export
FWHM_TEMPORAL_DP = 1.0         # = 1 TR = 1.6s; reduced from 3.0dp -- see docstring (AR(2) clip / dof bias)
SIGMA_SPATIAL_VOX = (FWHM_SPATIAL_MM / NATIVE_VOXEL_MM) / 2.3548
SIGMA_TEMPORAL_VOL = FWHM_TEMPORAL_DP / 2.3548
ZCHUNK = 20
HALO = 12


def prep_smoothed(vtc_paths, run_task_designs, log=print, ar_sample=4000):
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
    p10 = float(np.percentile(pos, 10))
    mask = tmean > p10
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

    return {"vtc_paths": vtc_paths, "headers": headers, "DimZ": DimZ, "DimY": DimY,
           "DimX": DimX, "n_cond": n_cond, "nrun": nrun, "n_conf": n_conf, "P": P,
           "run_row_bounds": run_row_bounds, "Ttotal": Ttotal, "Xw": Xw,
           "phi": (phi1, phi2), "mask": mask, "p10": p10}


def fit_smoothed(prep, log=print, zchunk=ZCHUNK, halo=HALO):
    vtc_paths = prep["vtc_paths"]; headers = prep["headers"]
    DimZ, DimY, DimX = prep["DimZ"], prep["DimY"], prep["DimX"]
    P = prep["P"]
    run_row_bounds = prep["run_row_bounds"]
    Xw = prep["Xw"]

    XtXw = Xw.T @ Xw
    XtXw_inv = np.linalg.pinv(XtXw)

    Nvox = DimZ * DimY * DimX
    XtY = np.zeros((P, Nvox), dtype=np.float32)
    sumYsq = np.zeros(Nvox, dtype=np.float32)
    t0 = time.time()
    for r in range(prep["nrun"]):
        mm = G.vtc_memmap(vtc_paths[r], headers[r])
        a, b = run_row_bounds[r], run_row_bounds[r+1]
        Xwr = Xw[a:b]
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
        log(f"    run {r+1}/{prep['nrun']} accumulated ({time.time()-t0:.0f}s elapsed)")

    betas = (XtXw_inv.astype(np.float32) @ XtY)
    rss = sumYsq.astype(np.float64) - np.einsum(
        "pn,pn->n", betas.astype(np.float64), XtY.astype(np.float64))
    rss = np.clip(rss, 0, None)
    dof = prep["Ttotal"] - P
    shape3 = (DimZ, DimY, DimX)
    return {"betas": betas.reshape((P,)+shape3), "rss": rss.reshape(shape3),
           "dof": dof, "XtXw_inv": XtXw_inv, "mask": prep["mask"], "P": P,
           "n_cond": prep["n_cond"], "headers": headers, "shape3": shape3}


def main():
    print(f"=== {BLOCK_NAME}-only smoothed GLM (no ridge -- well-conditioned) ===")
    print(f"Spatial: FWHM={FWHM_SPATIAL_MM}mm -> sigma={SIGMA_SPATIAL_VOX:.3f} vox")
    print(f"Temporal: FWHM={FWHM_TEMPORAL_DP}dp -> sigma={SIGMA_TEMPORAL_VOL:.3f} vol")

    designs = []
    for sdm_path in SDM_PATHS:
        X, _, nv = G2.read_sdm_predictors(sdm_path, CONDITION_ORDER)
        designs.append(X)
    print(f"cond(run1 {BLOCK_NAME}-only task design)={np.linalg.cond(designs[0]):.3f}")

    print(f"\n[{BLOCK_NAME}] prepping + AR(2) on temporally-smoothed sample ...")
    prep = prep_smoothed(C2.VTCS_REFINED, designs, log=print)
    cond_full = np.linalg.cond(prep["Xw"])
    print(f"cond(full whitened stacked design)={cond_full:.3f}")
    assert cond_full < 200, f"unexpectedly ill-conditioned ({cond_full:.1f}) -- stop and recheck"

    print(f"\n[{BLOCK_NAME}] fitting voxel-wise on FULLY SMOOTHED data (OLS, no ridge) ...")
    fit = fit_smoothed(prep, log=print)
    hdr = fit["headers"][0]
    mask = fit["mask"]
    dof = fit["dof"]
    bbox = {"XStart": hdr["XStart"], "XEnd": hdr["XEnd"], "YStart": hdr["YStart"],
           "YEnd": hdr["YEnd"], "ZStart": hdr["ZStart"] + ANAT_V2_Z_OFFSET,
           "ZEnd": hdr["ZEnd"] + ANAT_V2_Z_OFFSET}
    print(f"[{BLOCK_NAME}] dof={dof} (valid, unbiased OLS), mask voxels={int(mask.sum())}")

    ckpt_path = f"{WORK}/_ckpt_smoothed_{BLOCK_NAME}only_fit.npz"
    np.savez(ckpt_path, betas=fit["betas"], rss=fit["rss"], dof=dof,
             XtXw_inv=fit["XtXw_inv"], mask=mask, P=fit["P"], n_cond=fit["n_cond"],
             phi1=prep["phi"][0], phi2=prep["phi"][1],
             **{f"bbox_{k}": v for k, v in bbox.items()})
    print(f"[{BLOCK_NAME}] checkpointed to {ckpt_path}")

    P = fit["P"]; n_cond = fit["n_cond"]
    betas = fit["betas"]; rss = fit["rss"]; XtXw_inv = fit["XtXw_inv"]
    DimZ, DimY, DimX = fit["shape3"]

    # per-condition valid t-maps (single-predictor contrast vs. implicit baseline)
    t_maps = np.zeros((n_cond,) + fit["shape3"], dtype=np.float32)
    s2 = rss / dof
    for i in range(n_cond):
        c = np.zeros(P); c[i] = 1.0
        cb = betas[i]
        var_unit = float(XtXw_inv[i, i])
        denom = np.sqrt(np.clip(var_unit * s2, 1e-20, None))
        t = cb / denom
        t[~mask] = 0; t[~np.isfinite(t)] = 0
        t_maps[i] = t

    # omnibus F: all n_cond task predictors vs baseline (valid now, no ridge)
    C = np.zeros((n_cond, P))
    C[np.arange(n_cond), np.arange(n_cond)] = 1.0
    Cb = np.tensordot(C, betas, axes=([1], [0]))  # (n_cond, Z,Y,X)
    M = np.linalg.pinv(C @ XtXw_inv @ C.T)
    Cb2 = Cb.reshape(n_cond, -1)
    num = np.einsum("in,ij,jn->n", Cb2, M, Cb2).reshape(fit["shape3"])
    with np.errstate(invalid="ignore", divide="ignore"):
        Fmap = (num / n_cond) / np.clip(s2, 1e-20, None)
    Fmap[~mask] = 0; Fmap[~np.isfinite(Fmap)] = 0

    beta_absmax = float(np.percentile(np.abs(betas[:n_cond][:, mask]), 99.9))
    t_absmax = float(np.percentile(np.abs(t_maps[:, mask]), 99.9))
    Fcrit_p001 = float(stats.f.isf(0.001, n_cond, dof))
    print(f"[{BLOCK_NAME}] beta p99.9={beta_absmax:.1f}, t p99.9={t_absmax:.2f}, "
          f"F crit(p<0.001,{n_cond},{dof:.0f})={Fcrit_p001:.2f}, "
          f"F max={Fmap[mask].max():.2f}")

    # ---- VMP: per-condition beta + per-condition t + omnibus F ----
    hdr_for_vmp = {"XStart": bbox["XStart"], "XEnd": bbox["XEnd"],
                  "YStart": bbox["YStart"], "YEnd": bbox["YEnd"],
                  "ZStart": bbox["ZStart"], "ZEnd": bbox["ZEnd"],
                  "VTC resolution": 1, "DimX": DimX, "DimY": DimY, "DimZ": DimZ}
    maps = []
    for i, name in enumerate(CONDITION_ORDER):
        maps.append({"name": f"beta_{name}_SMOOTHED", "data": betas[i], "type": 1,
                    "df1": 0, "threshold": 0.0, "upper": beta_absmax, "showposneg": 3})
    for i, name in enumerate(CONDITION_ORDER):
        fdr_table = compute_fdr_table(np.abs(t_maps[i][mask]), dof)
        fdr_q05 = float(fdr_table[1, 1])
        maps.append({"name": f"t_{name}_SMOOTHED", "data": t_maps[i], "type": 1,
                    "df1": int(dof), "threshold": fdr_q05, "upper": t_absmax,
                    "fdr_table": fdr_table, "fdr_table_index": 1, "cluster_size": 4})
    maps.append({"name": f"{BLOCK_NAME}_omnibus_F_SMOOTHED", "data": Fmap, "type": 4,
                "df1": n_cond, "df2": int(dof), "threshold": Fcrit_p001,
                "upper": max(float(Fmap[mask].max()), Fcrit_p001 + 1), "showposneg": 1})

    vmp_path = f"{OUT_DIR}/sub-01_{BLOCK_NAME}only_SMOOTHED.vmp"
    G.write_stat_vmp(vmp_path, maps, hdr_for_vmp,
                     vtc_name="sub-01_run-*_preproc_coreg-refined.vtc (smoothed in-memory)",
                     prt_name=f"{BLOCK_NAME}_only")
    print(f"[{BLOCK_NAME}] wrote {vmp_path} ({len(maps)} sub-maps)")

    # ---- GLM export (valid OLS InvXtX -- BrainVoyager's live t-computation
    #      will now be genuinely correct, not ridge-invalid) ----
    XtY = XtXw_inv_inv_at_betas = np.linalg.pinv(XtXw_inv).astype(np.float64) @ betas.reshape(P, -1).astype(np.float64)
    fitted_ss = np.einsum("pn,pn->n", betas.reshape(P, -1).astype(np.float64), XtY)
    fitted_ss = fitted_ss.reshape(rss.shape)
    SS_total = np.clip(rss.astype(np.float64) + fitted_ss, 1e-12, None)
    R2 = np.sqrt(np.clip(1.0 - rss.astype(np.float64)/SS_total, 0, 1)).astype(np.float32)
    SS_total = SS_total.astype(np.float32)
    SS_XiY = XtY.reshape((P,) + rss.shape).transpose(1, 2, 3, 0).astype(np.float32)
    beta_zyxp = betas.transpose(1, 2, 3, 0).astype(np.float32)
    R2[~mask] = 0; SS_total[~mask] = 0
    meantc = np.zeros(rss.shape, dtype=np.float32)
    ARlag = np.zeros(rss.shape + (2,), dtype=np.float32)
    ARlag[..., 0] = prep["phi"][0]; ARlag[..., 1] = prep["phi"][1]

    predictor_info = []
    for i, name in enumerate(CONDITION_ORDER):
        predictor_info.append({"NameInternal": f"Predictor: {i+1}", "NameCustom": name,
                               "Color": np.array([200, 0, 0] if BLOCK_NAME == "CF" else [0, 0, 200], dtype=np.uint8)})
    for r in (1, 2, 3, 4):
        for label, color in (("Constant", [128, 0, 0]), ("Linear", [0, 128, 0])):
            predictor_info.append({"NameInternal": f"Predictor: {len(predictor_info)+1}",
                                   "NameCustom": f"Study {r}: {label}", "Color": np.array(color, dtype=np.uint8)})

    RUN_NVOLS = [350, 340, 340, 340]
    study_info = []
    for r, nv in zip((1, 2, 3, 4), RUN_NVOLS):
        study_info.append({"NrTimePoints": nv,
                           "NameOfStudyData": f"sub-01_run-{r}_preproc_coreg-refined.vtc [SMOOTHED in-memory]",
                           "NameOfSDM": SDM_PATHS[r-1]})

    header = dict(NrTimePoints=sum(RUN_NVOLS), NrAllPredictors=P, NrConfoundPredictors=8,
                  NrStudies=4, NrConfoundsPerStudy=[2, 2, 2, 2], ResolutionMultiplier=1,
                  SerialCorrelation=2, MeanSerialCorrBefore=float(prep["phi"][0]),
                  MeanSerialCorrAfter=0.0, **bbox, StudyInfo=study_info,
                  PredictorInfo=predictor_info, DesignMatrix=prep["Xw"], InvXtX=XtXw_inv)

    glm_path = f"{OUT_DIR}/sub-01_{BLOCK_NAME}only_SMOOTHED.glm"
    write_glm(glm_path, header, R2, SS_total, beta_zyxp, SS_XiY, meantc, ARlag)
    print(f"[{BLOCK_NAME}] wrote {glm_path}")

    import bvbabel.glm as glmmod
    h2, R2r, SSr, betar, _, _, _ = glmmod.read_glm(glm_path)
    print(f"[{BLOCK_NAME}] round-trip check: Nr all predictors={h2['Nr all predictors']} "
          f"(expect {P}), beta shape={betar.shape}")


if __name__ == "__main__":
    main()
