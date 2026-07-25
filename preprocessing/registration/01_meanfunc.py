#!/usr/bin/env python3
"""Compute mean-over-time functional images per run from the final
preprocessed FMR (motion-correction + slice-timing + high-pass filtering +
topup distortion correction folded in), for use as the moving image in
coregistration. Also computes mean uncorrected and mean topup-corrected b0
images per run for the independent distortion-residual re-check. Output:
per-run mean NIfTI volumes under derivatives/sub-01/reg and a JSON report.

Note on the FMR vs. the topup-corrected NIfTI it derives from (documented,
benign): the FMR data matches the topup NIfTI to within mean|diff|=0.11
intensity units, but the NIfTI has sparse negative-value spline-
interpolation undershoot artifacts (from applytopup --interp=spline) at
image edges/corners, which bvbabel's FMR writer floors to 0. Since negative
MR magnitude is non-physical, the floored FMR is treated as the canonical
data source for VTC construction.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import nibabel as nib
import bvbabel.fmr

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FUNC = f"{PROJECT_ROOT}/derivatives/sub-01/func"
REG = f"{PROJECT_ROOT}/derivatives/sub-01/reg"
LOG = f"{PROJECT_ROOT}/logs"
os.makedirs(REG, exist_ok=True)

report = {}

for r in [1, 2, 3, 4]:
    fmr_path = f"{FUNC}/sub-01_run-{r}_preproc.fmr"
    fh, fdata = bvbabel.fmr.read_fmr(fmr_path)  # (150,140,42,T) float32, matches nifti orientation
    mean_img = fdata.mean(axis=-1).astype(np.float32)
    # affine: identical to the stage-06dc-topup nifti for this run (shape/
    # orientation-equivalence verified in step 00)
    ref_nii = nib.load(f"{FUNC}/sub-01_run-{r}_stage-06dc-topup.nii.gz")
    out_path = f"{REG}/sub-01_run-{r}_meanfunc_preproc.nii.gz"
    nib.save(nib.Nifti1Image(mean_img, ref_nii.affine, ref_nii.header), out_path)
    report[f"run-{r}"] = {
        "n_vols_averaged": fdata.shape[-1], "out": out_path,
        "mean_img_range": [float(mean_img.min()), float(mean_img.max())],
    }
    print(f"run-{r}: meanfunc from {fdata.shape[-1]} vols -> {out_path}, "
          f"range [{mean_img.min():.1f}, {mean_img.max():.1f}]")

# --- mean uncorrected / topup-corrected b0 for the distortion recheck ---
for r in [1, 2, 3, 4]:
    unc_path = f"{FUNC}/sub-01_run-{r}_AP_b0-first5_cut.nii.gz"
    if os.path.exists(unc_path):
        img = nib.load(unc_path)
        arr = np.asarray(img.dataobj, dtype=np.float32)
        mean_unc = arr.mean(axis=-1) if arr.ndim == 4 else arr
        out = f"{REG}/sub-01_run-{r}_meanb0_uncorrected.nii.gz"
        nib.save(nib.Nifti1Image(mean_unc.astype(np.float32), img.affine, img.header), out)
        report.setdefault(f"run-{r}", {})["meanb0_uncorrected"] = out

    corr_path = f"{FUNC}/sub-01_run-{r}_topup_bunwarped.nii.gz"
    if os.path.exists(corr_path):
        img = nib.load(corr_path)
        arr = np.asarray(img.dataobj, dtype=np.float32)
        mean_corr = arr[..., :5].mean(axis=-1) if arr.ndim == 4 else arr
        out = f"{REG}/sub-01_run-{r}_meanb0_topupcorrected.nii.gz"
        nib.save(nib.Nifti1Image(mean_corr.astype(np.float32), img.affine, img.header), out)
        report.setdefault(f"run-{r}", {})["meanb0_topupcorrected"] = out

with open(f"{LOG}/registration_01_meanfunc.json", "w") as f:
    json.dump(report, f, indent=2)
print("wrote logs/registration_01_meanfunc.json")
