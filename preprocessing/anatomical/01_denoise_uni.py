#!/usr/bin/env python3
"""MP2RAGE UNI background-noise removal (O'Brien 2014 / Marques robust
combination). Input: MP2RAGE UNI, INV1, and INV2 NIfTI volumes. Output: a
denoised UNI NIfTI (for a range of candidate regularisation strengths) and a
JSON sweep summary.

The MP2RAGE UNI image is defined by the scanner as the real combination
    S = Re(I1* . I2) / (|I1|^2 + |I2|^2)
where I1 = first inversion (INV1), I2 = second inversion (INV2). In air/background
where both inversions are pure noise, this ratio is ill-conditioned -> the
characteristic MP2RAGE "salt-and-pepper" background.

O'Brien et al. (2014, PLoS ONE 9:e99676) regularise this with a constant beta:
    S_denoised = (Re(I1*.I2) - beta) / (|I1|^2 + |I2|^2 + 2*beta)

Siemens stores UNI scaled to integers [0, 4095] representing S in [-0.5, 0.5], so
    S = UNI/4095 - 0.5
and, by the very definition of the UNI combination,
    Re(I1*.I2) = S * (|I1|^2 + |I2|^2)   (exact, up to 0-4095 quantisation).
Hence we can apply the O'Brien regulariser using only the magnitude INV1/INV2 and
the scaled UNI, with no complex data and no polarity recovery. This is algebraically
identical to Marques' robustCombination.m.

beta = (lambda * sigma_bg)^2, where sigma_bg is the background noise level of INV2
(estimated from image corners) and lambda ("multiplying factor") controls denoising
strength. Larger lambda removes more background noise but can slightly bias very
low-SNR tissue. Marques suggests testing a small range; several candidates are
evaluated below and one is chosen.
"""
import sys, os, json
import numpy as np
import nibabel as nib

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RAW = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "rawdata_nifti")
OUT = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "anat")
os.makedirs(OUT, exist_ok=True)

def load(k):
    return nib.load(os.path.join(RAW, f"sub-01_acq-mp2rage_{k}.nii.gz"))

uni_img = load("UNI"); inv1_img = load("INV1"); inv2_img = load("INV2")
UNI = uni_img.get_fdata().astype(np.float64)
I1 = inv1_img.get_fdata().astype(np.float64)
I2 = inv2_img.get_fdata().astype(np.float64)

UNI_MAX = 4095.0
S = UNI / UNI_MAX - 0.5                      # signed combination in [-0.5, 0.5]
sumsq = I1**2 + I2**2
numer = S * sumsq                            # = Re(I1*.I2)

# --- estimate INV2 background noise level from 8 image corners (pure air) ---
def corner_bg(vol, c=12):
    x, y, z = vol.shape
    patches = []
    for xs in (slice(0, c), slice(x-c, x)):
        for ys in (slice(0, c), slice(y-c, y)):
            for zs in (slice(0, c), slice(z-c, z)):
                patches.append(vol[xs, ys, zs].ravel())
    return np.concatenate(patches)

bg = corner_bg(I2)
sigma_bg = float(bg.mean())          # Marques uses mean of background magnitude
sigma_std = float(bg.std())
print(f"INV2 background: mean={sigma_bg:.3f} std={sigma_std:.3f} n={bg.size}")

def robust(lmbda):
    beta = (lmbda * sigma_bg)**2
    S_d = (numer - beta) / (sumsq + 2.0*beta)
    S_d = np.clip(S_d, -0.5, 0.5)
    return (S_d + 0.5) * UNI_MAX, beta

# --- evaluate candidate lambdas: background suppression vs brain preservation ---
brain = I2 > (5*sigma_bg)            # crude brain mask from INV2 (high SNR)
air = I2 < (2*sigma_bg)
report = {}
for lm in [2, 4, 6, 8, 10]:
    den, beta = robust(lm)
    # background should collapse toward mid-grey (2048 = S=0); measure its spread
    bg_std = float(den[air].std())
    bg_mean = float(den[air].mean())
    # brain content should be essentially unchanged
    brain_delta = float(np.abs(den[brain] - UNI[brain]).mean())
    report[lm] = dict(beta=beta, bg_mean=bg_mean, bg_std=bg_std, brain_mean_abs_change=brain_delta)
    print(f"lambda={lm:2d} beta={beta:10.2f}  bg_mean={bg_mean:7.1f} bg_std={bg_std:7.1f}  brain|dU|={brain_delta:.2f}")

# Chosen lambda: smallest that flattens the background (bg_std near floor) while
# keeping the brain-region change negligible.
CHOSEN = int(sys.argv[1]) if len(sys.argv) > 1 else 6
den, beta = robust(CHOSEN)
den16 = np.rint(den).astype(np.int16)
out_img = nib.Nifti1Image(den16, uni_img.affine, uni_img.header)
out_img.set_data_dtype(np.int16)
out_path = os.path.join(OUT, "sub-01_desc-UNIdenoised_lambda%d.nii.gz" % CHOSEN)
nib.save(out_img, out_path)
print("wrote", out_path, "chosen lambda", CHOSEN, "beta", beta)

with open(os.path.join(OUT, "denoise_uni_sweep.json"), "w") as f:
    json.dump(dict(sigma_bg=sigma_bg, sigma_std=sigma_std, chosen_lambda=CHOSEN,
                   chosen_beta=beta, sweep=report), f, indent=2)
