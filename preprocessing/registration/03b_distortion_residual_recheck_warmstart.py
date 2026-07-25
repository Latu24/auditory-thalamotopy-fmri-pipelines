#!/usr/bin/env python3
"""Redo of 03_distortion_residual_recheck.py using robust, warm-started
registration. Input/output match 03_distortion_residual_recheck.py.

The from-scratch Translation-IA -> Rigid-FA procedure used in the previous
step suffers the same optimizer-convergence fragility already diagnosed for
run-4 in the coregistration step (small EPI slab vs. whole-head anatomical
-> many plausible local optima for a naive translation-only search). A
convergence sanity check (post-registration MI should be more negative /
better than the pre-registration identity-alignment MI) showed several of
the from-scratch fits in the previous step failed this basic check,
including both of run-3's fits (uncorrected and topup-corrected) - meaning
that run's "DEGRADED on both metrics" verdict was computed from two
badly-converged registrations and cannot be trusted as-is.

Fix: warm-start every fit from the shared, already-validated run-1-derived
func->anat rigid transform, which is a physically appropriate starting
point for all of these images (the b0/topup mean images share the same
common post-motion-correction grid/affine as the meanfunc images the shared
transform was derived from). This mirrors the warm-start fix validated for
the run-4 cross-check.

Reports both the warm-started-refined result and a "shared-transform-only,
no further refinement" baseline for comparison, and flags per-fit whether
the refinement actually improved over the baseline.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import ants
from scipy.ndimage import sobel

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ANAT_NII = f"{PROJECT_ROOT}/derivatives/sub-01/anat/sub-01_desc-UNIdenoisedN4.nii.gz"
INV2_NII = f"{PROJECT_ROOT}/derivatives/sub-01/rawdata_nifti/sub-01_acq-mp2rage_INV2.nii.gz"
REG = f"{PROJECT_ROOT}/derivatives/sub-01/reg"
LOG = f"{PROJECT_ROOT}/logs"
SHARED_XFM = f"{REG}/xfm/sub-01_run-1_to_anat_rigid.mat"


def gmag(a):
    gx = sobel(a, axis=0); gy = sobel(a, axis=1); gz = sobel(a, axis=2)
    return np.sqrt(gx**2 + gy**2 + gz**2)


def edge_corr(fixed_np, warped_np, mask):
    gf = gmag(fixed_np)[mask]; gw = gmag(warped_np)[mask]
    gf = (gf - gf.mean()) / (gf.std() + 1e-6)
    gw = (gw - gw.mean()) / (gw.std() + 1e-6)
    return float((gf * gw).mean())


def mi(fixed, moving):
    return float(ants.image_similarity(fixed, moving, metric_type="MattesMutualInformation"))


def evaluate_warmstart(fixed, moving_path, label):
    mov = ants.image_read(moving_path)
    # baseline: shared transform applied directly, no further refinement
    warped_shared = ants.apply_transforms(fixed=fixed, moving=mov, transformlist=[SHARED_XFM], interpolator="linear")
    mi_shared = mi(fixed, warped_shared)
    # warm-started refinement: FA initialized from the shared transform
    fa = ants.registration(fixed=fixed, moving=mov, type_of_transform="Rigid",
                            initial_transform=SHARED_XFM, aff_metric="mattes", verbose=False)
    warped_refined = ants.apply_transforms(fixed=fixed, moving=mov, transformlist=fa["fwdtransforms"], interpolator="linear")
    mi_refined = mi(fixed, warped_refined)

    # use whichever is better (more negative); refinement should never be
    # worse than the shared baseline since it's initialized from it, but
    # guard against stochastic-sampling noise anyway
    if mi_refined <= mi_shared:
        best_warped, best_mi, used = warped_refined, mi_refined, "warmstart_refined"
    else:
        best_warped, best_mi, used = warped_shared, mi_shared, "shared_transform_only"

    f = fixed.numpy(); w = best_warped.numpy()
    mask = w > np.percentile(w[w > 0], 1) if (w > 0).any() else (w != 0)
    ec = edge_corr(f, w, mask)
    print(f"  {label}: MI shared-only={mi_shared:.4f}  MI warmstart-refined={mi_refined:.4f}  "
          f"-> used={used}  edge_corr={ec:.4f}")
    return {"mi_shared_transform_only": mi_shared, "mi_warmstart_refined": mi_refined,
            "mi_best": best_mi, "which_used": used, "edge_grad_corr": ec}


def main():
    anat = ants.image_read(ANAT_NII)
    inv2 = ants.image_read(INV2_NII) if os.path.exists(INV2_NII) else None
    report = {"target_anat_UNI_N4": {}, "target_INV2_crosscheck": {}, "method": "warmstart (fixed)"}

    for r in [1, 2, 3, 4]:
        unc = f"{REG}/sub-01_run-{r}_meanb0_uncorrected.nii.gz"
        cor = f"{REG}/sub-01_run-{r}_meanb0_topupcorrected.nii.gz"
        if not (os.path.exists(unc) and os.path.exists(cor)):
            print(f"run-{r}: missing b0 mean inputs, skipping"); continue
        print(f"\n=== run-{r} (target: anat UNI-N4) ===")
        r_unc = evaluate_warmstart(anat, unc, f"run-{r} uncorrected")
        r_cor = evaluate_warmstart(anat, cor, f"run-{r} topup-corrected")
        delta_mi = r_cor["mi_best"] - r_unc["mi_best"]
        delta_edge = r_cor["edge_grad_corr"] - r_unc["edge_grad_corr"]
        verdict = ("IMPROVED (topup correction confirmed sound)" if (delta_mi < 0 and delta_edge > 0)
                   else ("MIXED (metrics disagree - common for cross-modality edge metrics, inspect QC images)"
                         if (delta_mi < 0 or delta_edge > 0)
                         else "DEGRADED (both metrics worse after topup under a robustly-converged rigid fit - genuine concern, escalate)"))
        report["target_anat_UNI_N4"][f"run-{r}"] = {
            "uncorrected": r_unc, "topup_corrected": r_cor,
            "delta_mi_corrected_minus_uncorrected": delta_mi,
            "delta_edge_corr_corrected_minus_uncorrected": delta_edge,
            "verdict": verdict,
        }
        print(f"  -> delta MI(best)={delta_mi:+.4f} (more negative=better so <0 desired) "
              f"delta edge_corr={delta_edge:+.4f}  verdict={verdict}")

    if inv2 is not None:
        for r in [1, 2]:
            unc = f"{REG}/sub-01_run-{r}_meanb0_uncorrected.nii.gz"
            cor = f"{REG}/sub-01_run-{r}_meanb0_topupcorrected.nii.gz"
            if not (os.path.exists(unc) and os.path.exists(cor)):
                continue
            print(f"\n=== run-{r} (INV2 cross-check) ===")
            r_unc = evaluate_warmstart(inv2, unc, f"run-{r} uncorrected (INV2)")
            r_cor = evaluate_warmstart(inv2, cor, f"run-{r} topup-corrected (INV2)")
            report["target_INV2_crosscheck"][f"run-{r}"] = {
                "uncorrected": r_unc, "topup_corrected": r_cor,
                "delta_mi_corrected_minus_uncorrected": r_cor["mi_best"] - r_unc["mi_best"],
                "delta_edge_corr_corrected_minus_uncorrected": r_cor["edge_grad_corr"] - r_unc["edge_grad_corr"],
            }

    with open(f"{LOG}/registration_03b_distortion_residual_recheck_warmstart.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote logs/registration_03b_distortion_residual_recheck_warmstart.json")


if __name__ == "__main__":
    main()
