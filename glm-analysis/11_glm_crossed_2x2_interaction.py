"""GLM fit for the CF x AM 2x2 CROSSED-CELL design (4 predictors:
CFlo_AMlo, CFlo_AMhi, CFhi_AMlo, CFhi_AMhi), built specifically to test a
genuine CF x AM statistical interaction — something the marginal 45-predictor
design (07/08/09_*.py) cannot express, since CF and AM there are separate
partitions of the same events, not crossed.

Design is well-conditioned (verified: cond(X)~3.5 per run, cond(full stacked
design incl. confounds)~5.1 — compare to the marginal design's ~1e9), so
plain OLS is valid here: no ridge needed, classical t/F distributional theory
applies directly (unlike 07_combined_ridge.py's shrunk/biased betas).

Fits on the REFINED VTCs (larger bounding box, better MGB coverage), same
choice as 07_combined_ridge.py. AR(2) prewhitening + memory-safe slab-wise
accumulation via glmlib2.py (the same engine already validated for the
marginal design).

Interaction contrast: c = [+1, -1, -1, +1] on
[CFlo_AMlo, CFlo_AMhi, CFhi_AMlo, CFhi_AMhi]
= (CFhi_AMhi - CFhi_AMlo) - (CFlo_AMhi - CFlo_AMlo)
= does the CF effect change across AM levels (equivalently, does the AM
effect change across CF levels).

Output: derivatives/sub-01/analysis/thalamus_work/cfxam_interaction_native.npz
(t-map + dof, restricted to & indexed within the native-space thalamus mask)
plus a JSON log. Thresholding (p<0.05 / FDR / Monte Carlo cluster) is done
in 12_threshold_interaction.py, kept separate so the (expensive) GLM fit
only needs to run once.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import nibabel as nib
import glmlib as G
import glmlib2 as G2
import common2 as C2

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ROOT = PROJECT_ROOT
ANA = f"{ROOT}/derivatives/sub-01/analysis"
WORK = f"{ANA}/thalamus_work"
SDM_DIR = f"{ANA}/sdm"

CONDITION_ORDER = ["CFlo_AMlo", "CFlo_AMhi", "CFhi_AMlo", "CFhi_AMhi"]
INTERACTION_CONTRAST_TASK = np.array([1.0, -1.0, -1.0, 1.0])  # LL - LH - HL + HH

FBZ, FBY, FBX = 240, 320, 320  # same canonical framebox as scripts 12/16


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


def main():
    print("[interaction] loading crossed-cell SDMs ...")
    designs = []
    for r in (1, 2, 3, 4):
        X, names, nv = G2.read_sdm_predictors(
            f"{SDM_DIR}/sub-01_run-{r}_CFxAM_2x2_crossed_nilearn.sdm",
            CONDITION_ORDER)
        designs.append(X)

    # ---- collinearity check (should be small; this is NOT the marginal design) ----
    cond_r1 = float(np.linalg.cond(designs[0]))
    print(f"[interaction] cond(run1 4-col task design) = {cond_r1:.3f} "
          f"(marginal 45-col design was ~1e9 -- this design is crossed-cell, "
          f"expected to be well-conditioned)")

    print("[interaction] prepping multi-run design + AR(2) + voxel sample "
          "(refined VTCs) ...")
    prep = G2.prep_multirun_design(C2.VTCS_REFINED, designs, log=print)
    phi1, phi2 = prep["phi"]

    cond_full = float(np.linalg.cond(prep["Xw"]))
    print(f"[interaction] cond(full whitened stacked design) = {cond_full:.3f}")
    assert cond_full < 100, (
        f"design less well-conditioned than expected (cond={cond_full:.1f}) "
        f"-- stop and re-check before trusting OLS inference")

    print("[interaction] fitting voxel-wise (OLS, ridge_lambda=0) ...")
    fit = G2.fit_from_prep(prep, log=print, ridge_lambda=0.0)
    hdr = fit["headers"][0]
    mask = fit["mask"]

    # checkpoint immediately -- the fit just streamed several GB per run off
    # disk; don't lose that work if anything below this point has a bug.
    ckpt_path = f"{WORK}/_ckpt_crossed2x2_fit.npz"
    np.savez(ckpt_path, betas=fit["betas"], rss=fit["rss"], dof=fit["dof"],
             XtXws_reg_inv=fit["XtXws_reg_inv"], col_scale=fit["col_scale"],
             mask=fit["mask"], P=fit["P"], n_cond=fit["n_cond"],
             bbox_XStart=hdr["XStart"], bbox_XEnd=hdr["XEnd"],
             bbox_YStart=hdr["YStart"], bbox_YEnd=hdr["YEnd"],
             bbox_ZStart=hdr["ZStart"], bbox_ZEnd=hdr["ZEnd"])
    print(f"[interaction] checkpointed fit essentials to {ckpt_path}")

    print("[interaction] computing interaction contrast t-map ...")
    c_full = np.zeros(fit["P"])
    c_full[:fit["n_cond"]] = INTERACTION_CONTRAST_TASK
    t_map = G2.t_contrast_v2(fit, c_full)
    dof = fit["dof"]

    bbox = {k: hdr[k] for k in ("XStart", "XEnd", "YStart", "YEnd", "ZStart", "ZEnd")}
    print(f"[interaction] bbox (refined VTC): {bbox}")

    t_native = to_native(t_map, bbox)
    mask_native = to_native(mask.astype(np.float32), bbox) > 0.5

    thal = np.asarray(
        nib.load(f"{WORK}/sub-01_thalamus_native.nii.gz").dataobj) > 0
    assert thal.shape == t_native.shape, (
        f"thalamus mask shape {thal.shape} != native t-map shape {t_native.shape}")

    in_thal = thal & mask_native
    n_thal_total = int(thal.sum())
    n_thal_covered = int(in_thal.sum())
    print(f"[interaction] thalamus voxels: {n_thal_total} total, "
          f"{n_thal_covered} with functional coverage in this (refined) VTC bbox")

    t_thal = t_native[in_thal]

    np.savez(f"{WORK}/cfxam_interaction_native.npz",
             t_native=t_native, mask_native=mask_native, thal=thal,
             in_thal=in_thal, t_thal=t_thal, dof=dof)

    betas_cond = fit["betas"][:4]
    log = {
        "analysis": "CFxAM_2x2_crossed_interaction",
        "condition_order": CONDITION_ORDER,
        "interaction_contrast_task_cols": INTERACTION_CONTRAST_TASK.tolist(),
        "interaction_contrast_formula": "(CFhi_AMhi - CFhi_AMlo) - (CFlo_AMhi - CFlo_AMlo)",
        "vtc_source": "refined (SyN), same choice as 07_combined_ridge.py",
        "predictor_source": "SDM (nilearn spm HRF) built from prt_crossed_2x2 PRTs",
        "effects": "fixed-effects multi-run, OLS (ridge_lambda=0)",
        "collinearity_check": {
            "cond_run1_4col_task_only": cond_r1,
            "cond_full_stacked_whitened_design": cond_full,
        },
        "phi_ar2": fit["phi"], "Ttotal": fit["Ttotal"], "P": fit["P"],
        "dof": float(dof),
        "bbox_refined_vtc": bbox,
        "mask_voxels_vtc_space": int(mask.sum()),
        "thalamus_voxels_total": n_thal_total,
        "thalamus_voxels_with_coverage": n_thal_covered,
        "t_thal_summary": {
            "n": int(t_thal.size),
            "min": float(t_thal.min()) if t_thal.size else None,
            "max": float(t_thal.max()) if t_thal.size else None,
            "mean": float(t_thal.mean()) if t_thal.size else None,
        },
        "output_npz": f"{WORK}/cfxam_interaction_native.npz",
        "note": ("t_native/mask_native/thal are full native-space "
                 "(240-flip-transposed) volumes; t_thal is the 1-D array of "
                 "t-values at in_thal voxels only, for fast downstream "
                 "thresholding without re-running this fit."),
    }
    with open(f"{WORK}/cfxam_interaction_stats.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"[interaction] wrote {WORK}/cfxam_interaction_native.npz")
    print(f"[interaction] wrote {WORK}/cfxam_interaction_stats.json")
    print(f"[interaction] dof={dof:.1f}, thalamus t range "
          f"[{t_thal.min():.3f}, {t_thal.max():.3f}] over {t_thal.size} voxels")


if __name__ == "__main__":
    main()
