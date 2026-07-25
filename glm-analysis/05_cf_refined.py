"""Carrier-frequency GLM (36 conditions), refit on refined VTCs.

Rerun of the earlier carrier-frequency GLM, but:
- on the refined VTCs (larger bounding box after additional nonlinear
  registration, re-read fresh from disk)
- on the corrected VMR
- predictors taken directly from precomputed CF-only SDM files (built with
  nilearn's make_first_level_design_matrix, hrf_model="spm") instead of
  rebuilt HRF convolution
- produces one beta sub-map per condition (36), plus the omnibus-F,
  all-tones, and best-condition summary maps kept from the earlier round
  (this design is well-conditioned, cond ~5.5, so full OLS inference
  remains valid)

Output: derivatives/sub-01/analysis/sub-01_carrierFrequency_refined.vmp
(does not touch the earlier sub-01_carrierFrequency.vmp)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import stats
import glmlib as G
import glmlib2 as G2
import common2 as C2
import common as C1


def main():
    designs = []
    for cf_sdm, vtc in zip(C2.CF_SDM, C2.VTCS_REFINED):
        X, names, nv = G2.read_sdm_predictors(cf_sdm, C2.FREQ_ORDER)
        h = G.read_vtc_header(vtc)
        assert h["DimT"] == nv, f"SDM/VTC volume mismatch: {nv} vs {h['DimT']}"
        designs.append(X)
    n_cond = len(C2.FREQ_ORDER)

    print("[CF-refined] fitting FFX GLM (OLS, well-conditioned), "
          f"{n_cond} conditions, 4 runs, refined VTCs ...")
    fit = G2.fit_multirun_glm_v2(C2.VTCS_REFINED, designs, log=print,
                                 ridge_lambda=0.0)
    hdr = fit["headers"][0]
    dof = fit["dof"]
    mask = fit["mask"]

    log = {"analysis": "carrierFrequency_refined", "n_conditions": n_cond,
           "condition_order": C2.FREQ_ORDER, "vtc_source": "refined (SyN)",
           "predictor_source": "SDM (nilearn spm HRF)",
           "effects": "fixed-effects multi-run, OLS", "dof": dof,
           "Ttotal": fit["Ttotal"], "P": fit["P"], "phi_ar2": fit["phi"],
           "mask_voxels": int(mask.sum()),
           "vtc_bbox": {k: hdr[k] for k in
                       ("XStart","XEnd","YStart","YEnd","ZStart","ZEnd",
                        "VTC resolution","DimX","DimY","DimZ")}}

    maps = []
    # ---- per-condition BETA maps (the primary new deliverable) ----
    betas_cond = fit["betas"][:n_cond]
    beta_absmax = float(np.percentile(np.abs(betas_cond[:, mask]), 99))
    for i, name in enumerate(C2.FREQ_ORDER):
        maps.append({"name": f"beta_{name}", "data": betas_cond[i],
                     "type": 1, "df1": 0, "threshold": 0.0,
                     "upper": beta_absmax, "showposneg": 3})

    # ---- omnibus F ----
    F = G2.f_omnibus_v2(fit, cond_cols=list(range(n_cond)))
    F_thr = float(stats.f.isf(0.001, n_cond, dof))
    maps.append({"name": f"carrierFrequency_refined omnibus F ({n_cond} cond)",
                 "data": F, "type": 4, "df1": n_cond, "df2": dof,
                 "threshold": F_thr, "upper": F_thr * 3, "showposneg": 1})
    log["omnibus_F"] = {"df1": n_cond, "df2": dof, "F_thr_p001": F_thr,
                        "peak_F": float(np.max(F)),
                        "n_F_gt_thr": int((F > F_thr).sum())}
    print(f"[CF-refined] omnibus F: peak={np.max(F):.1f} thr={F_thr:.2f} "
          f"nsig={int((F>F_thr).sum())}")

    # ---- all-tones > baseline ----
    c_all = np.zeros(fit["P"]); c_all[:n_cond] = 1.0 / n_cond
    t_all = G2.t_contrast_v2(fit, c_all)
    maps.append({"name": "carrierFrequency_refined all-tones>baseline (t)",
                 "data": t_all, "type": 1, "df1": dof, "threshold": 5.0,
                 "upper": 15.0})
    log["all_tones"] = {"peak_t": float(np.max(t_all)),
                        "n_t_gt_5": int((t_all > 5).sum())}
    print(f"[CF-refined] all-tones: peak t={np.max(t_all):.2f}")

    # ---- best-frequency (voxel-level, F-masked) ----
    best = np.argmax(betas_cond, axis=0).astype(np.float32) + 1.0
    sig = (F > F_thr) & mask
    best_map = np.where(sig, best, 0.0).astype(np.float32)
    maps.append({"name": "carrierFrequency_refined best-freq idx (voxel-level, F-masked)",
                 "data": best_map, "type": 1, "df1": 0, "threshold": 0.5,
                 "upper": float(n_cond), "showposneg": 1})
    log["best_condition_map"] = {
        "kind": "VOXEL-LEVEL tuning proxy (argmax beta), NOT layer-resolved",
        "n_voxels": int(sig.sum())}

    out = f"{C2.ANA}/sub-01_carrierFrequency_refined.vmp"
    G.write_stat_vmp(out, maps, hdr,
                     vtc_name="sub-01_run-1_preproc_coreg-refined.vtc",
                     prt_name="sub-01_run-1_CF_only.sdm")
    log["vmp"] = out
    log["vmp_n_submaps"] = len(maps)
    C1.save_json(f"{C2.LOGS}/analyses_carrierFrequency_refined.json", log)
    print(f"[CF-refined] wrote {out} ({len(maps)} sub-maps)")


if __name__ == "__main__":
    main()
