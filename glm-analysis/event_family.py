"""Shared runner for the two event-related tone analyses (carrier frequency,
amplitude modulation). Both label the same underlying tone onsets per run;
only the condition grouping differs. Fits a fixed-effects multi-run GLM and
writes one VMP per family with four sub-maps:

  1. Omnibus F  (any condition differs from baseline)
  2. All-tones > baseline (t; mean of all condition predictors)
  3. Best-condition index (voxel-level tuning proxy, masked to F-significant)
  4. Linear/ordinal parametric contrast across conditions (t)

Sub-map 3 is a voxel-level tuning proxy, not layer-resolved tonotopy — that
would require cortical surface reconstruction, which is out of scope here.
"""
import numpy as np
from scipy import stats
import glmlib as G
import common as C


def run_family(family, prts, condition_order, vtc_paths, hrf,
               parametric_name="linear"):
    n_cond = len(condition_order)
    designs = []
    for prt, vtc in zip(prts, vtc_paths):
        nv = G.read_vtc_header(vtc)["DimT"]
        X, _ = G.build_run_design(prt, nv, hrf, condition_order=condition_order)
        designs.append(X)

    print(f"[{family}] fitting FFX GLM, {n_cond} conditions, 4 runs ...")
    fit = G.fit_multirun_glm(vtc_paths, designs, hrf, log=print)
    hdr = fit["headers"][0]
    dof = fit["dof"]
    mask = fit["mask"]

    log = {"analysis": family, "n_conditions": n_cond,
           "condition_order": condition_order,
           "effects": "fixed-effects multi-run", "dof": dof,
           "Ttotal": fit["Ttotal"], "P": fit["P"], "phi_ar2": fit["phi"],
           "mask_voxels": int(mask.sum())}

    maps = []

    # ---- 1. omnibus F over all condition columns ----
    F = G.f_omnibus(fit, cond_cols=list(range(n_cond)))
    F_thr = float(stats.f.isf(0.001, n_cond, dof))
    log["omnibus_F"] = {"df1": n_cond, "df2": dof,
                        "F_thr_p001": F_thr,
                        "peak_F": float(np.max(F)),
                        "n_F_gt_thr": int((F > F_thr).sum()),
                        "frac_mask_F_gt_thr": float((F > F_thr).sum() / mask.sum())}
    maps.append({"name": f"{family} omnibus F ({n_cond} cond)", "data": F,
                 "type": 4, "df1": n_cond, "df2": dof,
                 "threshold": F_thr, "upper": F_thr * 3, "showposneg": 1})
    print(f"[{family}] omnibus F: peak={np.max(F):.1f} thr(p.001)={F_thr:.2f} "
          f"nsig={int((F>F_thr).sum())}")

    # ---- 2. all-tones > baseline (mean contrast) ----
    c_all = np.zeros(fit["P"]); c_all[:n_cond] = 1.0 / n_cond
    t_all = G.t_contrast(fit, c_all)
    log["all_tones"] = {"peak_t": float(np.max(t_all)),
                        "n_t_gt_5": int((t_all > 5).sum())}
    maps.append({"name": f"{family} all-tones>baseline (t)", "data": t_all,
                 "type": 1, "df1": dof, "threshold": 5.0, "upper": 15.0})
    print(f"[{family}] all-tones>base: peak t={np.max(t_all):.2f} "
          f"t>5={int((t_all>5).sum())}")

    # ---- 3. best-condition index (voxel-level tuning proxy) ----
    betas_cond = fit["betas"][:n_cond]              # (n_cond, Z,Y,X)
    best = np.argmax(betas_cond, axis=0).astype(np.float32) + 1.0  # 1..n_cond
    sig = (F > F_thr) & mask
    best_map = np.where(sig, best, 0.0).astype(np.float32)
    log["best_condition_map"] = {
        "kind": "VOXEL-LEVEL tuning proxy (argmax beta), NOT layer-resolved",
        "masked_to": "omnibus F > p.001",
        "n_voxels": int(sig.sum())}
    maps.append({"name": f"{family} best-cond idx (voxel-level, F-masked)",
                 "data": best_map, "type": 1, "df1": 0,
                 "threshold": 0.5, "upper": float(n_cond), "showposneg": 1})

    # ---- 4. ordinal/linear parametric contrast ----
    w = np.linspace(-1, 1, n_cond)
    w = w - w.mean()
    c_lin = np.zeros(fit["P"]); c_lin[:n_cond] = w
    t_lin = G.t_contrast(fit, c_lin)
    log["parametric_linear"] = {
        "weights_note": f"ordinal ascending across {condition_order[0]}.."
                        f"{condition_order[-1]} (assumes name order == "
                        f"{parametric_name} order)",
        "peak_t_pos": float(np.max(t_lin)),
        "peak_t_neg": float(np.min(t_lin)),
        "n_abs_t_gt_4": int((np.abs(t_lin) > 4).sum())}
    maps.append({"name": f"{family} linear-{parametric_name} (t, +hi/-lo)",
                 "data": t_lin, "type": 1, "df1": dof,
                 "threshold": 3.0, "upper": 10.0, "showposneg": 3})

    out = f"{C.ANA}/sub-01_{family}.vmp"
    G.write_stat_vmp(out, maps, hdr,
                     vtc_name="sub-01_run-1_preproc_coreg.vtc",
                     prt_name=prts[0].split('/')[-1])
    log["vmp"] = out
    log["vmp_submaps"] = [m["name"] for m in maps]
    C.save_json(f"{C.LOGS}/analyses_{family}.json", log)

    # QC overlays
    anatf = np.load("/tmp/anatf.npy")
    C.overlay_tmap(t_all, anatf, f"{C.QC}/{family}_allTones_tmap.png",
                   f"{family} all-tones t", thr=5.0, vmax=15)
    center = int(np.unravel_index(np.argmax(t_all), t_all.shape)[0])
    C.overlay_tmap(best_map, anatf, f"{C.QC}/{family}_bestcond_map.png",
                   f"{family} best-cond", thr=0.5, vmax=float(n_cond),
                   cmap="jet", center=center)
    print("[%s] wrote %s" % (family, out))
    return log
