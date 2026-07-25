#!/usr/bin/env python3
"""Independent re-check of the distortion-residual QC flag raised during
functional preprocessing: a plain-NCC sanity check there found EPI-to-
anatomy alignment decreasing after topup correction, which was suspected to
be a weak-metric artifact rather than a real distortion problem (that
earlier check used a post-hoc intensity cross-correlation metric rather
than the Mattes MI that ants.registration() actually optimizes, and a
single one-shot rigid registration with no explicit coarse-then-fine
staging).

This script redoes the comparison (uncorrected vs. topup-corrected mean
AP-b0, registered to the same anatomical target) but:
  - uses the same staged IA(Translation)->FA(Rigid) pipeline as the main
    functional<->anatomical coregistration step (not a single-shot fit),
  - scores with the actual Mattes MI value ants optimizes (not a post-hoc
    NCC proxy), reporting both pre-registration (identity) and
    post-registration (best rigid fit found) values,
  - adds an edge-gradient-correlation metric as a second, independent
    boundary-alignment signal (systematic edge misalignment after the best
    achievable rigid fit is the residual-distortion signature being
    checked for),
  - runs across all 4 runs.

Verdict logic: if topup-corrected data shows both higher MI and higher
edge-correlation than uncorrected (under a proper multi-stage rigid fit
with the real MI cost function), that supports the "weak-metric artifact"
explanation and confirms the topup correction is sound. If topup-corrected
is still worse under this proper check, that indicates a genuine
distortion-correction concern.

Output: derivatives/sub-01/reg/*_meanb0_* NIfTI target comparisons and a
JSON report at logs/registration_03_distortion_residual_recheck.json.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import nibabel as nib
import ants
from scipy.ndimage import sobel

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ANAT_NII = f"{PROJECT_ROOT}/derivatives/sub-01/anat/sub-01_desc-UNIdenoisedN4.nii.gz"
INV2_NII = f"{PROJECT_ROOT}/derivatives/sub-01/rawdata_nifti/sub-01_acq-mp2rage_INV2.nii.gz"
REG = f"{PROJECT_ROOT}/derivatives/sub-01/reg"
LOG = f"{PROJECT_ROOT}/logs"


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


def ia_fa(fixed, moving):
    ia = ants.registration(fixed=fixed, moving=moving, type_of_transform="Translation",
                            aff_metric="mattes", verbose=False)
    fa = ants.registration(fixed=fixed, moving=moving, type_of_transform="Rigid",
                            initial_transform=ia["fwdtransforms"][0], aff_metric="mattes",
                            verbose=False)
    warped = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=fa["fwdtransforms"],
                                    interpolator="linear")
    return fa, warped


def evaluate(anat, moving_path, label):
    mov = ants.image_read(moving_path)
    identity = ants.apply_transforms(fixed=anat, moving=mov, transformlist=[], interpolator="linear")
    mi_pre = mi(anat, identity)
    fa, warped = ia_fa(anat, mov)
    mi_post = mi(anat, warped)
    f = anat.numpy(); w = warped.numpy()
    mask = w > np.percentile(w[w > 0], 1) if (w > 0).any() else (w != 0)
    ec = edge_corr(f, w, mask)
    print(f"  {label}: MI pre(identity)={mi_pre:.4f}  MI post(rigid IA+FA)={mi_post:.4f}  edge_corr={ec:.4f}")
    return {"mi_pre_identity": mi_pre, "mi_post_rigid": mi_post, "edge_grad_corr": ec}


def main():
    anat = ants.image_read(ANAT_NII)
    inv2 = ants.image_read(INV2_NII) if os.path.exists(INV2_NII) else None

    report = {"target_anat_UNI_N4": {}, "target_INV2_crosscheck": {}}

    for r in [1, 2, 3, 4]:
        unc = f"{REG}/sub-01_run-{r}_meanb0_uncorrected.nii.gz"
        cor = f"{REG}/sub-01_run-{r}_meanb0_topupcorrected.nii.gz"
        if not (os.path.exists(unc) and os.path.exists(cor)):
            print(f"run-{r}: missing b0 mean inputs, skipping"); continue
        print(f"\n=== run-{r} (target: anat UNI-N4, the actual VTC target space) ===")
        r_unc = evaluate(anat, unc, f"run-{r} uncorrected")
        r_cor = evaluate(anat, cor, f"run-{r} topup-corrected")
        delta_mi = r_cor["mi_post_rigid"] - r_unc["mi_post_rigid"]
        delta_edge = r_cor["edge_grad_corr"] - r_unc["edge_grad_corr"]
        report["target_anat_UNI_N4"][f"run-{r}"] = {
            "uncorrected": r_unc, "topup_corrected": r_cor,
            "delta_mi_corrected_minus_uncorrected": delta_mi,
            "delta_edge_corr_corrected_minus_uncorrected": delta_edge,
            "verdict": "IMPROVED (topup correction confirmed sound)" if (delta_mi > 0 and delta_edge > 0)
                       else ("MIXED (one metric improved, one did not - inspect QC images)" if (delta_mi > 0 or delta_edge > 0)
                       else "DEGRADED (both metrics worse after topup - possible genuine residual distortion, escalate)"),
        }
        print(f"  -> delta MI={delta_mi:+.4f}  delta edge_corr={delta_edge:+.4f}  "
              f"verdict={report['target_anat_UNI_N4'][f'run-{r}']['verdict']}")

    # cross-check against the INV2 target (the original functional-stage QC target) for run-1, run-2
    if inv2 is not None:
        for r in [1, 2]:
            unc = f"{REG}/sub-01_run-{r}_meanb0_uncorrected.nii.gz"
            cor = f"{REG}/sub-01_run-{r}_meanb0_topupcorrected.nii.gz"
            if not (os.path.exists(unc) and os.path.exists(cor)):
                continue
            print(f"\n=== run-{r} (cross-check target: INV2, matching the original functional-stage QC) ===")
            r_unc = evaluate(inv2, unc, f"run-{r} uncorrected (INV2)")
            r_cor = evaluate(inv2, cor, f"run-{r} topup-corrected (INV2)")
            delta_mi = r_cor["mi_post_rigid"] - r_unc["mi_post_rigid"]
            delta_edge = r_cor["edge_grad_corr"] - r_unc["edge_grad_corr"]
            report["target_INV2_crosscheck"][f"run-{r}"] = {
                "uncorrected": r_unc, "topup_corrected": r_cor,
                "delta_mi_corrected_minus_uncorrected": delta_mi,
                "delta_edge_corr_corrected_minus_uncorrected": delta_edge,
            }

    with open(f"{LOG}/registration_03_distortion_residual_recheck.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote logs/registration_03_distortion_residual_recheck.json")


if __name__ == "__main__":
    main()
