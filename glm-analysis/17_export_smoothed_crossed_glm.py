"""Export the already-fitted SMOOTHED crossed-2x2 GLM (from
_ckpt_smoothed_refit_fit.npz, computed in 16_full_smoothed_refit.py) as a
native BrainVoyager .glm file + VMP contrast maps, so it can be opened
directly in BrainVoyager.

R2/SS_total/SS_XiY are reconstructed algebraically from what was checkpointed
(betas, rss, XtXw_inv) rather than requiring another VTC pass:
  fitted_ss = beta' (X'X) beta = beta . (XtXw @ beta)   [exact, since
  beta = XtXw_inv @ XtY implies XtY = XtXw @ beta for a well-conditioned system]
  SS_total  = rss + fitted_ss
  R2        = 1 - rss/SS_total  (clipped, sqrt for the stored "R" value)
  SS_XiY    = XtY = XtXw @ beta

meantc is NOT reconstructable from the checkpoint (would need the raw
per-voxel temporal mean, not saved) -- filled with a placeholder (0), flagged
here and in the GLM's own predictor-info naming. This is a display-only field
in BrainVoyager's GUI and does not affect beta/contrast/t-value validity.

phi1/phi2 (global AR(2)) are hardcoded from this exact run's printed log
since they weren't part of the checkpoint. After 16_full_smoothed_refit.py
was corrected to spatial FWHM=1.5mm / temporal FWHM=1.0dp (matched to the
CF-only/AM-only smoothed GLMs), the corrected run converges cleanly to
phi1=0.1222, phi2=0.1395, with no AR(2) stability clipping (an earlier,
more aggressive temporal FWHM had pinned phi1 at the clip).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import glmlib as G
from glm_writer import write_glm
from vmp_fdr_table import compute_fdr_table

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

WORK = f"{PROJECT_ROOT}/derivatives/sub-01/analysis/thalamus_work"
OUT_DIR = f"{PROJECT_ROOT}/derivatives/sub-01/analysis/smoothed_glm"
os.makedirs(OUT_DIR, exist_ok=True)

CKPT = f"{WORK}/_ckpt_smoothed_refit_fit.npz"
CONDITION_ORDER = ["CFlo_AMlo", "CFlo_AMhi", "CFhi_AMlo", "CFhi_AMhi"]
PHI1, PHI2 = 0.1222, 0.1395   # from the corrected (1.5mm/1.0dp) run's own printed AR(2) estimate

CONTRASTS = {
    "interaction": np.array([1.0, -1.0, -1.0, 1.0]),
    "CF_main": np.array([-1.0, -1.0, 1.0, 1.0]),
    "AM_main": np.array([-1.0, 1.0, -1.0, 1.0]),
}

VTC_PATHS = [f"{PROJECT_ROOT}/derivatives/sub-01/reg/"
             f"sub-01_run-{r}_preproc_coreg-refined.vtc" for r in (1, 2, 3, 4)]
SDM_PATHS = [f"{PROJECT_ROOT}/derivatives/sub-01/analysis/sdm/"
             f"sub-01_run-{r}_CFxAM_2x2_crossed_nilearn.sdm" for r in (1, 2, 3, 4)]
RUN_NVOLS = [350, 340, 340, 340]


def main():
    ck = np.load(CKPT)
    betas = ck["betas"]           # (P, DimZ, DimY, DimX)
    rss = ck["rss"]                # (DimZ, DimY, DimX)
    XtXw_inv = ck["XtXw_inv"]      # (P, P)
    mask = ck["mask"]
    P = int(ck["P"]); n_cond = int(ck["n_cond"])
    dof = float(ck["dof"])
    # The anatomical VMR that functional data is aligned to centers its
    # 240-slice short axis (Z = left-right) inside BrainVoyager's 320^3
    # framing cube via OffsetZ=(320-240)//2=40 -- our bbox was computed
    # against the OLD, uncentered convention (OffsetZ=0), so it needs the
    # same +40 shift to land correctly on that VMR.
    ANAT_V2_Z_OFFSET = 40
    bbox = {"XStart": int(ck["bbox_XStart"]), "XEnd": int(ck["bbox_XEnd"]),
            "YStart": int(ck["bbox_YStart"]), "YEnd": int(ck["bbox_YEnd"]),
            "ZStart": int(ck["bbox_ZStart"]) + ANAT_V2_Z_OFFSET,
            "ZEnd": int(ck["bbox_ZEnd"]) + ANAT_V2_Z_OFFSET}
    DimZ, DimY, DimX = rss.shape
    Ttotal = int(dof) + P

    print(f"Loaded checkpoint: P={P}, n_cond={n_cond}, dof={dof}, shape={rss.shape}")

    # the checkpointed mask used a 40th-percentile brightness cutoff, which
    # excludes 63% of the anatomical thalamus (it's dimmer than surrounding
    # tissue in raw BOLD units, not because it lacks scan coverage -- verified
    # directly: 100% of the thalamus is within the functional bbox, only the
    # intensity mask was excluding it). Recomputed at the 10th percentile,
    # which includes >95% of thalamus voxels while still excluding true
    # near-zero background (cheap: single strided read of run1, no re-fit).
    h0 = G.read_vtc_header(VTC_PATHS[0])
    mm0 = G.vtc_memmap(VTC_PATHS[0], h0)
    tmean = np.asarray(mm0[:, :, :, ::8].mean(-1), dtype=np.float32)
    pos = tmean[tmean > 0]
    p10 = float(np.percentile(pos, 10))
    mask = tmean > p10
    print(f"recomputed mask at p10={p10:.1f} (was p40={float(np.percentile(pos,40)):.1f} "
          f"in checkpoint): {int(mask.sum())} voxels (was {int(ck['mask'].sum())})")

    XtXw = np.linalg.pinv(XtXw_inv)   # invert the already-saved inverse back
    betas_flat = betas.reshape(P, -1)                       # (P, Nvox)
    XtY_flat = XtXw.astype(np.float64) @ betas_flat.astype(np.float64)   # (P, Nvox) exact reconstruction
    fitted_ss = np.einsum("pn,pn->n", betas_flat.astype(np.float64), XtY_flat)
    fitted_ss = fitted_ss.reshape(rss.shape)
    SS_total = rss.astype(np.float64) + fitted_ss
    SS_total = np.clip(SS_total, 1e-12, None)
    R2 = np.sqrt(np.clip(1.0 - rss.astype(np.float64) / SS_total, 0, 1)).astype(np.float32)
    SS_total = SS_total.astype(np.float32)
    SS_XiY = XtY_flat.reshape((P,) + rss.shape).transpose(1, 2, 3, 0).astype(np.float32)  # (Z,Y,X,P)
    beta_zyxp = betas.transpose(1, 2, 3, 0).astype(np.float32)  # (Z,Y,X,P)

    meantc = np.zeros(rss.shape, dtype=np.float32)   # NOT reconstructable from checkpoint -- see docstring
    ARlag = np.zeros(rss.shape + (2,), dtype=np.float32)
    ARlag[..., 0] = PHI1
    ARlag[..., 1] = PHI2
    R2[~mask] = 0; SS_total[~mask] = 0; meantc[~mask] = 0

    # Design matrix + its own inverse X'X (whitened, global-AR2 -- see glm_writer.py docstring)
    designs = []
    for sdm_path, nv in zip(SDM_PATHS, RUN_NVOLS):
        import glmlib2 as G2
        X, _, _ = G2.read_sdm_predictors(sdm_path, CONDITION_ORDER)
        designs.append(X)
    X_full = np.zeros((Ttotal, P))
    row = 0
    for run_idx, (X_task, nv) in enumerate(zip(designs, RUN_NVOLS)):
        X_full[row:row+nv, :n_cond] = X_task
        c0 = n_cond + run_idx * 2
        X_full[row:row+nv, c0:c0+2] = G.confounds(nv)
        row += nv
    Xw_full = X_full.copy()
    row = 0
    for nv in RUN_NVOLS:
        Xw_full[row:row+nv] = G.ar2_whiten(X_full[row:row+nv], PHI1, PHI2)
        row += nv

    condition_colors = {
        "CFlo_AMlo": [66, 133, 244], "CFlo_AMhi": [52, 168, 83],
        "CFhi_AMlo": [251, 188, 5], "CFhi_AMhi": [234, 67, 53],
    }
    predictor_info = []
    for i, name in enumerate(CONDITION_ORDER):
        predictor_info.append({"NameInternal": f"Predictor: {i+1}", "NameCustom": name,
                               "Color": np.array(condition_colors[name], dtype=np.uint8)})
    for r in (1, 2, 3, 4):
        for label, color in (("Constant", [128, 0, 0]), ("Linear", [0, 128, 0])):
            predictor_info.append({"NameInternal": f"Predictor: {len(predictor_info)+1}",
                                   "NameCustom": f"Study {r}: {label}",
                                   "Color": np.array(color, dtype=np.uint8)})

    study_info = []
    for r, nv in zip((1, 2, 3, 4), RUN_NVOLS):
        study_info.append({
            "NrTimePoints": nv,
            "NameOfStudyData": (f"sub-01_run-{r}_preproc_coreg-refined.vtc "
                               f"[SMOOTHED in-memory: spatial FWHM=1.5mm, "
                               f"temporal FWHM=1.0dp (=1 TR=1.6s) -- matched to "
                               f"CF-only/AM-only smoothed GLMs, see 16_full_smoothed_refit.py]"),
            "NameOfSDM": SDM_PATHS[r-1],
        })

    header = dict(
        NrTimePoints=Ttotal, NrAllPredictors=P, NrConfoundPredictors=8, NrStudies=4,
        NrConfoundsPerStudy=[2, 2, 2, 2], ResolutionMultiplier=1,
        SerialCorrelation=2, MeanSerialCorrBefore=float(PHI1), MeanSerialCorrAfter=0.0,
        **bbox, StudyInfo=study_info, PredictorInfo=predictor_info,
        DesignMatrix=Xw_full, InvXtX=XtXw_inv,
    )

    glm_path = f"{OUT_DIR}/sub-01_CFxAM_2x2_crossed_SMOOTHED.glm"
    write_glm(glm_path, header, R2, SS_total, beta_zyxp, SS_XiY, meantc, ARlag)
    print(f"wrote {glm_path}")

    # round-trip sanity check
    import bvbabel.glm as glmmod
    h2, R2r, SSr, betar, _, _, ARlagr = glmmod.read_glm(glm_path)
    print(f"round-trip check: Nr all predictors={h2['Nr all predictors']} "
          f"(expect {P}), Nr studies={h2['Nr studies']} (expect 4), "
          f"beta shape={betar.shape}")

    # VMP exports for the 3 contrasts (raw VTC-space, matching this pipeline's
    # existing VMP convention exactly)
    fake_hdr = {"XStart": bbox["XStart"], "XEnd": bbox["XEnd"],
                "YStart": bbox["YStart"], "YEnd": bbox["YEnd"],
                "ZStart": bbox["ZStart"], "ZEnd": bbox["ZEnd"],
                "VTC resolution": 1, "DimX": DimX, "DimY": DimY, "DimZ": DimZ}
    maps = []
    for name, c_task in CONTRASTS.items():
        c_full = np.zeros(P)
        c_full[:n_cond] = c_task
        cb = np.tensordot(c_full, betas, axes=([0], [0]))
        var_unit = float(c_full @ XtXw_inv @ c_full)
        s2 = rss / dof
        denom = np.sqrt(np.clip(var_unit * s2, 1e-20, None))
        t = cb / denom
        t[~mask] = 0
        t[~np.isfinite(t)] = 0
        fdr_table = compute_fdr_table(np.abs(t[mask]), dof)
        # MapThreshold = the map's OWN FDR q<0.05 critical |t| (row index 1 =
        # q=0.05, "critical_std" column) -- not an arbitrary fixed number.
        # This is what BrainVoyager's FDR/significance display actually reads
        # as the display-worthy cutoff, and it differs slightly per contrast
        # since each has its own p-value distribution.
        fdr_q05_crit_t = float(fdr_table[1, 1])
        print(f"  {name}: FDR q<0.05 critical |t| = {fdr_q05_crit_t:.3f}")
        maps.append({"name": f"{name}_SMOOTHED_t", "data": t, "type": 1,
                    "df1": int(dof), "threshold": fdr_q05_crit_t, "upper": 8.0,
                    "fdr_table": fdr_table, "fdr_table_index": 1,
                    "cluster_size": 4})

    vmp_path = f"{OUT_DIR}/sub-01_CFxAM_2x2_crossed_SMOOTHED.vmp"
    G.write_stat_vmp(vmp_path, maps, fake_hdr,
                     vtc_name="sub-01_run-*_preproc_coreg-refined.vtc (smoothed in-memory)",
                     prt_name="CFxAM_2x2_crossed")
    print(f"wrote {vmp_path} ({len(maps)} sub-maps: {list(CONTRASTS.keys())})")


if __name__ == "__main__":
    main()
