#!/usr/bin/env python3
"""N4 bias-field correction of the denoised MP2RAGE UNI image. Input: the
denoised UNI NIfTI (step 01 output) and a brain mask derived from INV2.
Output: the N4-corrected UNI NIfTI, the estimated bias field, and a JSON
summary of tissue-contrast QC statistics, including a numerical comparison
against a previously computed corrected image if one is present on disk.

ants.n4_bias_field_correction() is called once, with return_bias_field=True
to obtain the bias field, and the corrected image is then derived by direct
division (corrected = uni / bias). The underlying ITK N4 filter computes the
corrected image and the bias field as two components of a single optimizer
run internally; deriving the corrected image by division keeps the two
numerically self-consistent by construction (corrected IS uni/bias, rather
than "should approximately equal"), and only pays for one N4 optimizer run
instead of two independent ones.
"""
import os, json
import numpy as np
import nibabel as nib
import ants

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ANAT = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "anat")
den_path = os.path.join(ANAT, "sub-01_desc-UNIdenoised_lambda6.nii.gz")
mask_path = os.path.join(ANAT, "sub-01_desc-brainmaskINV2.nii.gz")
old_corrected_path = os.path.join(ANAT, "sub-01_desc-UNIdenoisedN4.nii.gz")

uni = ants.image_read(den_path)
mask = ants.image_read(mask_path)

# --- N4, computed ONCE ---
bias = ants.n4_bias_field_correction(
    uni,
    mask=mask,
    shrink_factor=4,
    convergence={"iters": [50, 50, 50, 50], "tol": 1e-7},
    spline_param=200,
    return_bias_field=True,
)

uni_np = uni.numpy()
bias_np = bias.numpy()

# guard against non-positive / degenerate bias values (e.g. far outside mask
# where the fitted spline could in principle dip near/at zero)
n_nonpositive = int(np.sum(bias_np <= 1e-6))
safe_bias = np.where(bias_np > 1e-6, bias_np, 1.0)
corrected_np = (uni_np / safe_bias).astype("float32")
corrected = ants.from_numpy(corrected_np, origin=uni.origin, spacing=uni.spacing, direction=uni.direction)

out_path = os.path.join(ANAT, "sub-01_desc-UNIdenoisedN4_v2.nii.gz")
bias_path = os.path.join(ANAT, "sub-01_desc-N4biasfield_v2.nii.gz")
ants.image_write(corrected, out_path)
ants.image_write(bias, bias_path)

# --- QC numbers: tissue-contrast separability before vs after (within mask) ---
def contrast_stats(img_np, m):
    v = img_np[m > 0.5]
    v = v[v > 0]
    return dict(mean=float(v.mean()), std=float(v.std()),
                cov=float(v.std()/v.mean()),
                p05=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)))

m_np = mask.numpy()
stats = dict(
    method="single N4 computation (bias field only); corrected derived by division uni/bias",
    n_bias_nonpositive_voxels_guarded=n_nonpositive,
    before=contrast_stats(uni_np, m_np),
    after=contrast_stats(corrected_np, m_np),
    bias_min=float(bias_np[m_np > 0.5].min()),
    bias_max=float(bias_np[m_np > 0.5].max()),
)

# --- divergence vs a previously computed corrected image, if present ---
if os.path.exists(old_corrected_path):
    old = np.asarray(nib.load(old_corrected_path).get_fdata(), dtype=np.float64)
    new_full = np.asarray(nib.load(out_path).get_fdata(), dtype=np.float64)
    diff = new_full - old
    m3 = m_np > 0.5
    stats["divergence_vs_previous_corrected_image"] = dict(
        mean_abs_diff_whole_volume=float(np.mean(np.abs(diff))),
        max_abs_diff_whole_volume=float(np.max(np.abs(diff))),
        mean_abs_diff_in_brainmask=float(np.mean(np.abs(diff[m3]))),
        max_abs_diff_in_brainmask=float(np.max(np.abs(diff[m3]))),
        mean_rel_diff_in_brainmask=float(np.mean(np.abs(diff[m3]) / np.maximum(np.abs(old[m3]), 1e-6))),
        max_rel_diff_in_brainmask=float(np.max(np.abs(diff[m3]) / np.maximum(np.abs(old[m3]), 1e-6))),
    )
else:
    stats["divergence_vs_previous_corrected_image"] = "previous corrected image not found; skipped"

print(json.dumps(stats, indent=2))
with open(os.path.join(ANAT, "n4_stats_v2.json"), "w") as f:
    json.dump(stats, f, indent=2)
print("wrote", out_path)
print("wrote", bias_path)
