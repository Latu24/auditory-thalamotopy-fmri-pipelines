"""Export the smoothed 45-condition RIDGE sanity-test GLM (checkpoint from
18_marginal_45cond_smoothed_sanity.py) as a native BrainVoyager .glm file.

Ridge complicates the R2/SS_XiY reconstruction used in
17_export_smoothed_crossed_glm.py (that trick relied on the plain OLS
identity XtY = XtXw @ beta, which only holds when there's no ridge penalty).
Under ridge, the correct relationship is in the RAW (unscaled) parameter
space: converting the ridge penalty from scaled-space (lambda * D, D=identity
on task columns) into raw-space gives an equivalent penalty of
lambda * diag(col_scale^2) on task columns (derived in-line below), so that
  (Xw'Xw + lambda*diag(col_scale^2)) @ beta_raw = Xw'Y = XtY
holds exactly, letting SS_XiY/R2 be reconstructed the same way as script 17,
just with this ridge-adjusted matrix instead of the plain XtXw.

phi1/phi2 hardcoded from this run's own printed log, same convention as
script 17.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import glmlib as G
import glmlib2 as G2
import common2 as C2
from glm_writer import write_glm

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ANA = f"{PROJECT_ROOT}/derivatives/sub-01/analysis"
WORK = f"{ANA}/thalamus_work"
OUT_DIR = f"{ANA}/smoothed_glm"

CKPT = f"{WORK}/_ckpt_smoothed_45cond_ridge_fit.npz"
PHI1, PHI2 = 0.9000, -0.6438
RUN_NVOLS = [350, 340, 340, 340]


def main():
    ck = np.load(CKPT)
    betas = ck["betas"]
    rss = ck["rss"]
    XtXws_reg_inv = ck["XtXws_reg_inv"]
    col_scale = ck["col_scale"]
    mask = ck["mask"]
    P = int(ck["P"]); n_cond = int(ck["n_cond"])
    ridge_lambda = float(ck["ridge_lambda"])
    dof = float(ck["dof"])
    # the anatomical VMR centers its 240-slice short axis (Z=left-right)
    # inside BrainVoyager's 320^3 framing cube via OffsetZ=40 -- our bbox was
    # computed against the old, uncentered convention, needs the same shift.
    ANAT_V2_Z_OFFSET = 40
    bbox = {"XStart": int(ck["bbox_XStart"]), "XEnd": int(ck["bbox_XEnd"]),
            "YStart": int(ck["bbox_YStart"]), "YEnd": int(ck["bbox_YEnd"]),
            "ZStart": int(ck["bbox_ZStart"]) + ANAT_V2_Z_OFFSET,
            "ZEnd": int(ck["bbox_ZEnd"]) + ANAT_V2_Z_OFFSET}
    DimZ, DimY, DimX = rss.shape
    Ttotal = sum(RUN_NVOLS)
    print(f"Loaded checkpoint: P={P}, n_cond={n_cond}, ridge_lambda={ridge_lambda}, "
          f"dof={dof}, shape={rss.shape}")

    # checkpointed mask (p40 brightness cutoff) excludes 63% of the anatomical
    # thalamus -- it's dimmer than surrounding tissue in raw BOLD units, not
    # outside scan coverage (100% of the thalamus is within the functional
    # bbox). Recomputed at p10, which includes >95% of thalamus voxels.
    h0 = G.read_vtc_header(C2.VTCS_REFINED[0])
    mm0 = G.vtc_memmap(C2.VTCS_REFINED[0], h0)
    tmean = np.asarray(mm0[:, :, :, ::8].mean(-1), dtype=np.float32)
    pos = tmean[tmean > 0]
    p10 = float(np.percentile(pos, 10))
    mask = tmean > p10
    print(f"recomputed mask at p10={p10:.1f}: {int(mask.sum())} voxels "
          f"(was {int(ck['mask'].sum())} at p40 in checkpoint)")

    cf_order = C2.FREQ_ORDER
    am_order = C2.AM_ORDER
    combined_order = cf_order + am_order
    assert len(combined_order) == n_cond

    designs = []
    for cf_sdm, am_sdm in zip(C2.CF_SDM, C2.AM_SDM):
        Xc, _, _ = G2.combined_design_from_sdms(cf_sdm, am_sdm, cf_order, am_order)
        designs.append(Xc)
    X_full = np.zeros((Ttotal, P))
    row = 0
    for run_idx, (X_task, nv) in enumerate(zip(designs, RUN_NVOLS)):
        X_full[row:row+nv, :n_cond] = X_task
        c0 = n_cond + run_idx * 2
        X_full[row:row+nv, c0:c0+2] = G.confounds(nv)
        row += nv
    Xw = X_full.copy()
    row = 0
    for nv in RUN_NVOLS:
        Xw[row:row+nv] = G.ar2_whiten(X_full[row:row+nv], PHI1, PHI2)
        row += nv

    # ridge penalty in RAW (unscaled) parameter space: lambda*diag(col_scale^2)
    # on task columns only -- derived so (Xw'Xw + D_raw) @ beta_raw = Xw'Y exactly
    XtXw_raw = Xw.T @ Xw
    D_raw = np.zeros((P, P))
    D_raw[np.arange(n_cond), np.arange(n_cond)] = ridge_lambda * (col_scale[:n_cond] ** 2)
    M_raw = XtXw_raw + D_raw
    InvXtX_raw = np.linalg.pinv(M_raw)

    betas_flat = betas.reshape(P, -1).astype(np.float64)
    XtY_flat = M_raw.astype(np.float64) @ betas_flat   # exact, by ridge normal-equation construction
    fitted_ss = np.einsum("pn,pn->n", betas_flat, XtY_flat).reshape(rss.shape)
    SS_total = np.clip(rss.astype(np.float64) + fitted_ss, 1e-12, None)
    R2 = np.sqrt(np.clip(1.0 - rss.astype(np.float64) / SS_total, 0, 1)).astype(np.float32)
    SS_total = SS_total.astype(np.float32)
    SS_XiY = XtY_flat.reshape((P,) + rss.shape).transpose(1, 2, 3, 0).astype(np.float32)
    beta_zyxp = betas.transpose(1, 2, 3, 0).astype(np.float32)
    R2[~mask] = 0; SS_total[~mask] = 0

    meantc = np.zeros(rss.shape, dtype=np.float32)   # not reconstructable, see script 17 note
    ARlag = np.zeros(rss.shape + (2,), dtype=np.float32)
    ARlag[..., 0] = PHI1; ARlag[..., 1] = PHI2

    predictor_info = []
    for i, name in enumerate(combined_order):
        color = [200, 0, 0] if i < 36 else [0, 0, 200]
        predictor_info.append({"NameInternal": f"Predictor: {i+1}", "NameCustom": name,
                               "Color": np.array(color, dtype=np.uint8)})
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
                               f"[SMOOTHED in-memory: spatial FWHM=4mm ASSUMED, "
                               f"temporal FWHM=3.0dp; RIDGE lambda={ridge_lambda:g} "
                               f"-- see 18_marginal_45cond_smoothed_sanity.py]"),
            "NameOfSDM": f"docs/sub-01_run-{r}_{{CF,AM}}_only.sdm",
        })

    header = dict(
        NrTimePoints=Ttotal, NrAllPredictors=P, NrConfoundPredictors=8, NrStudies=4,
        NrConfoundsPerStudy=[2, 2, 2, 2], ResolutionMultiplier=1,
        SerialCorrelation=2, MeanSerialCorrBefore=float(PHI1), MeanSerialCorrAfter=0.0,
        **bbox, StudyInfo=study_info, PredictorInfo=predictor_info,
        DesignMatrix=Xw, InvXtX=InvXtX_raw,
    )

    glm_path = f"{OUT_DIR}/sub-01_combinedCFAM_45cond_SMOOTHED_ridge.glm"
    write_glm(glm_path, header, R2, SS_total, beta_zyxp, SS_XiY, meantc, ARlag)
    print(f"wrote {glm_path}")

    import bvbabel.glm as glmmod
    h2, R2r, SSr, betar, _, _, ARlagr = glmmod.read_glm(glm_path)
    print(f"round-trip check: Nr all predictors={h2['Nr all predictors']} (expect {P}), "
          f"Nr studies={h2['Nr studies']} (expect 4), beta shape={betar.shape}")


if __name__ == "__main__":
    main()
