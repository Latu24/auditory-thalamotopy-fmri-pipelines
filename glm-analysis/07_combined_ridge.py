"""Combined CF+AM (45 conditions) GLM on refined VTCs, fit with ridge
regression.

Unlike the earlier CF-only / AM-only refits, this is a new analysis: all 36
carrier-frequency + 9 amplitude-modulation predictors are estimated
TOGETHER in one design, on the refined VTCs.

Critical collinearity issue (verified independently below): the
by-frequency and by-AM PRTs partition the exact SAME underlying tone events
— sum(36 CF columns) == sum(9 AM columns) at every timepoint (correlation
1.000000). A plain unconstrained combined design has condition number
~1e9-1e10 (reproduced in this script's own check below). Plain OLS betas
would be numerically unstable, so this analysis uses RIDGE (L2-penalized)
least squares instead:
  - task columns (45) standardized to unit column-norm so one lambda is
    comparable across predictors; confound columns (const+linear per run)
    are never penalized or scaled.
  - lambda chosen via a quick GCV grid search (glmlib2.gcv_select_lambda),
    using the same voxel sample already gathered for AR(2) estimation —
    cheap, no extra VTC I/O.
  - because ridge estimates are BIASED (that is the whole point — trading
    bias for a large reduction in variance/instability), the classical
    OLS t/F distributional theory does NOT apply to these betas. This
    script therefore does NOT produce an "omnibus F" or thresholded t-map
    for the combined model (that would misrepresent shrunken, biased
    coefficients as if they were valid unbiased test statistics). It
    provides the 45 per-condition BETA maps (the primary deliverable) plus
    purely DESCRIPTIVE summaries (mean-beta, argmax best-condition,
    beta-range) instead.

Output: derivatives/sub-01/analysis/sub-01_combinedCFAM_ridge.vmp
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import glmlib as G
import glmlib2 as G2
import common2 as C2
import common as C1


def main():
    cf_order = C2.FREQ_ORDER
    am_order = C2.AM_ORDER
    combined_order = cf_order + am_order
    n_cond = len(combined_order)  # 45

    designs = []
    for cf_sdm, am_sdm, vtc in zip(C2.CF_SDM, C2.AM_SDM, C2.VTCS_REFINED):
        Xc, names, nv = G2.combined_design_from_sdms(cf_sdm, am_sdm,
                                                      cf_order, am_order)
        h = G.read_vtc_header(vtc)
        assert h["DimT"] == nv, f"SDM/VTC volume mismatch: {nv} vs {h['DimT']}"
        designs.append(Xc)

    # ---- independently re-verify the collinearity finding on THIS design ----
    Xstack = np.vstack(designs)  # not physically meaningful across runs (no
    # confounds), but sufficient to check the raw per-run column relationship
    cf_sum = np.vstack([d[:, :36] for d in designs]).sum(axis=1)
    am_sum = np.vstack([d[:, 36:] for d in designs]).sum(axis=1)
    corr = float(np.corrcoef(cf_sum, am_sum)[0, 1])
    cond_raw = float(np.linalg.cond(designs[0]))
    cond_combined_r1 = float(np.linalg.cond(
        np.column_stack([designs[0], G.confounds(designs[0].shape[0])])))
    print(f"[combined-ridge] collinearity check: corr(sumCF,sumAM)={corr:.9f} "
          f"cond(run1 45-col)={cond_raw:.3e} "
          f"cond(run1 45-col+confounds)={cond_combined_r1:.3e}")

    print("[combined-ridge] prepping design + AR(2) + voxel sample ...")
    prep = G2.prep_multirun_design(C2.VTCS_REFINED, designs, log=print)
    phi1, phi2 = prep["phi"]

    # ---- GCV lambda search ----
    lambda_grid = np.concatenate([[0.0], np.logspace(-2, 4, 13)])
    print(f"[combined-ridge] GCV lambda grid: {lambda_grid.tolist()}")
    best_lambda, gcv_diag = G2.gcv_select_lambda(
        prep["Xw"], prep["run_row_bounds"], prep["Ysamp"], n_cond,
        lambda_grid, phi1, phi2, log=print)

    # sanity: report condition number of the REGULARIZED, whitened+scaled
    # design at best_lambda vs. at lambda=0
    task_norms = np.linalg.norm(prep["Xw"][:, :n_cond], axis=0)
    task_norms[task_norms == 0] = 1.0
    col_scale = np.ones(prep["P"]); col_scale[:n_cond] = task_norms
    Xws = prep["Xw"] / col_scale[None, :]
    XtXws = Xws.T @ Xws
    D = np.zeros((prep["P"], prep["P"]))
    D[np.arange(n_cond), np.arange(n_cond)] = 1.0
    cond_before = float(np.linalg.cond(XtXws))
    cond_after = float(np.linalg.cond(XtXws + best_lambda * D))
    print(f"[combined-ridge] cond(X'X) whitened+scaled: before={cond_before:.3e} "
          f"after ridge(lambda={best_lambda:g})={cond_after:.3e}")

    print(f"[combined-ridge] fitting voxel-wise with lambda={best_lambda:g} ...")
    fit = G2.fit_from_prep(prep, log=print, ridge_lambda=best_lambda)
    hdr = fit["headers"][0]
    mask = fit["mask"]

    log = {
        "analysis": "combinedCFAM_ridge", "n_conditions": n_cond,
        "condition_order": combined_order, "vtc_source": "refined (SyN)",
        "predictor_source": "SDM (nilearn spm HRF), CF+AM concatenated",
        "effects": "fixed-effects multi-run, RIDGE (L2)",
        "collinearity_check": {
            "corr_sumCF_sumAM": corr,
            "cond_run1_45col_raw": cond_raw,
            "cond_run1_45col_plus_confounds": cond_combined_r1,
        },
        "gcv_lambda_grid": gcv_diag,
        "chosen_lambda": best_lambda,
        "cond_XtX_whitened_scaled_before_ridge": cond_before,
        "cond_XtX_whitened_scaled_after_ridge": cond_after,
        "phi_ar2": fit["phi"], "Ttotal": fit["Ttotal"], "P": fit["P"],
        "mask_voxels": int(mask.sum()),
        "vtc_bbox": {k: hdr[k] for k in
                    ("XStart","XEnd","YStart","YEnd","ZStart","ZEnd",
                     "VTC resolution","DimX","DimY","DimZ")},
        "inference_caveat": (
            "Ridge estimates are BIASED by construction (shrunk toward 0); "
            "classical OLS t/F distributional theory does not apply. No "
            "omnibus-F or t-threshold map is produced for this model. Beta "
            "maps and descriptive summaries (mean, argmax, range) only. "
            "Compare against the well-conditioned, unbiased CF-only / "
            "AM-only OLS models (05_cf_refined.py / 06_am_refined.py) for "
            "any claim requiring valid statistical inference."),
    }

    betas_cond = fit["betas"][:n_cond]
    beta_absmax = float(np.percentile(np.abs(betas_cond[:, mask]), 99))
    maps = []
    for i, name in enumerate(combined_order):
        maps.append({"name": f"beta_{name}", "data": betas_cond[i],
                     "type": 1, "df1": 0, "threshold": 0.0,
                     "upper": beta_absmax, "showposneg": 3})

    # ---- descriptive-only summaries (no inferential claims) ----
    mean_beta = betas_cond.mean(axis=0)
    maps.append({"name": "combinedCFAM_ridge mean-beta (descriptive)",
                 "data": mean_beta, "type": 1, "df1": 0, "threshold": 0.0,
                 "upper": beta_absmax, "showposneg": 3})

    beta_range = betas_cond.max(axis=0) - betas_cond.min(axis=0)
    maps.append({"name": "combinedCFAM_ridge beta-range (descriptive, max-min)",
                 "data": beta_range, "type": 1, "df1": 0, "threshold": 0.0,
                 "upper": float(np.percentile(beta_range[mask], 99))})

    best = np.argmax(betas_cond, axis=0).astype(np.float32) + 1.0
    # descriptive mask: top quartile of beta-range within brain mask (no
    # formal significance claim -- see inference_caveat above)
    range_thr = float(np.percentile(beta_range[mask], 75))
    desc_sig = (beta_range > range_thr) & mask
    best_map = np.where(desc_sig, best, 0.0).astype(np.float32)
    maps.append({"name": "combinedCFAM_ridge best-cond idx (voxel-level, DESCRIPTIVE only, top-quartile beta-range)",
                 "data": best_map, "type": 1, "df1": 0, "threshold": 0.5,
                 "upper": float(n_cond), "showposneg": 1})
    log["best_condition_map"] = {
        "kind": "VOXEL-LEVEL, DESCRIPTIVE argmax-beta, NOT a significance "
                "test (ridge bias invalidates classical inference) and NOT "
                "layer-resolved",
        "mask_rule": "beta-range > 75th percentile within brain mask",
        "n_voxels": int(desc_sig.sum())}

    out = f"{C2.ANA}/sub-01_combinedCFAM_ridge.vmp"
    G.write_stat_vmp(out, maps, hdr,
                     vtc_name="sub-01_run-1_preproc_coreg-refined.vtc",
                     prt_name="combined_CF+AM_SDM")
    log["vmp"] = out
    log["vmp_n_submaps"] = len(maps)
    print(f"[combined-ridge] peak |beta| (99th pct within mask) = {beta_absmax:.3f}")
    print(f"[combined-ridge] wrote {out} ({len(maps)} sub-maps)")
    C1.save_json(f"{C2.LOGS}/analyses_combinedCFAM_ridge.json", log)


if __name__ == "__main__":
    main()
