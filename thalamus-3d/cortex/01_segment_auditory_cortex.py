"""Cortex pipeline -- auditory cortex segmentation, mirroring the thalamus
pipeline's "Reg B" path (FSL MNI152_T1_1mm -> native, SyNRA) but warping the
Harvard-Oxford cortical atlas instead of the subcortical one, to build a
native-space auditory-cortex mask comparable to the existing thalamus/MGB
masks.

Auditory cortex definition (Harvard-Oxford cortical maxprob labels, standard
fMRI-tonotopy usage): Heschl's Gyrus (primary auditory cortex) + Planum
Temporale + Planum Polare + Superior Temporal Gyrus (anterior + posterior
divisions) as the auditory belt/parabelt. XML label indices are 0-based; the
maxprob NIfTI label value is index+1 (this offset was confirmed against the
thalamus pipeline's own existing HO_THAL = {4: "L", 15: "R"} <-> XML
"Left/Right Thalamus" index 3/14 mapping in 11_segment_thalamus.py).

  XML index 8  "Superior Temporal Gyrus, anterior division" -> value 9
  XML index 9  "Superior Temporal Gyrus, posterior division" -> value 10
  XML index 43 "Planum Polare"                                -> value 44
  XML index 44 "Heschl's Gyrus (includes H1 and H2)"           -> value 45
  XML index 45 "Planum Temporale"                              -> value 46

Unlike the subcortical atlas, the Harvard-Oxford cortical maxprob atlas is
not lateralized (one label value covers both hemispheres) -- hemisphere is
assigned post-hoc from native-space world X sign (x<0 => L), the same
convention already used per-voxel throughout the thalamus pipeline's
18_cfam_conjunction_smoothed.py.

Inputs: native anatomical NIfTI, FSL MNI152 template + Harvard-Oxford
cortical atlas. Outputs: native-space auditory-cortex binary masks (merged +
per-hemisphere) and a segmentation_stats.json sidecar.
"""
import os, json, time
import numpy as np
import nibabel as nib
from scipy import ndimage
import ants

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

ROOT = PROJECT_ROOT
ANAT = f"{ROOT}/derivatives/sub-01/anat/sub-01_desc-UNIdenoisedN4.nii.gz"
WORK = f"{ROOT}/derivatives/sub-01/analysis/cortex_work"
FSLMNI = os.path.expanduser("~/fsl/data/standard/MNI152_T1_1mm.nii.gz")
HO_CORT = os.path.expanduser("~/fsl/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz")

# XML index + 1 (see module docstring); label -> short name
AUD_CTX_LABELS = {
    9: "STG_anterior",
    10: "STG_posterior",
    44: "Planum_Polare",
    45: "Heschl_Gyrus",
    46: "Planum_Temporale",
}


def largest_cc(binmask):
    lab, n = ndimage.label(binmask)
    if n == 0:
        return binmask
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep = np.argmax(sizes) + 1
    return lab == keep


def main():
    os.makedirs(WORK, exist_ok=True)
    fixed = ants.image_read(ANAT)
    anat_nib = nib.load(ANAT)
    aff = anat_nib.affine
    log(f"native anat {fixed.shape} spacing {np.round(fixed.spacing, 3)}")

    log("Reg B: FSL MNI152_T1_1mm -> native (SyNRA) ...")
    movB = ants.image_read(FSLMNI)
    regB = ants.registration(fixed=fixed, moving=movB, type_of_transform="SyNRA")
    log("   Reg B done")

    ho = ants.image_read(HO_CORT)
    ho_warp = ants.apply_transforms(fixed=fixed, moving=ho,
                                     transformlist=regB["fwdtransforms"],
                                     interpolator="genericLabel")
    ho_arr = ho_warp.numpy()
    log(f"   warped cortical atlas unique labels present: {sorted(np.unique(ho_arr).astype(int).tolist())[:20]}...")

    aud_bin = np.isin(ho_arr, list(AUD_CTX_LABELS.keys()))
    log(f"   raw union of {list(AUD_CTX_LABELS.values())}: {int(aud_bin.sum())} vox (both hemis, pre-cleanup)")

    # per-label breakdown (diagnostic)
    per_label = {name: int((ho_arr == val).sum()) for val, name in AUD_CTX_LABELS.items()}
    log(f"   per-label voxel counts: {per_label}")

    # hemisphere split via native world-space X sign (project convention: x<0 => L)
    ii, jj, kk = np.meshgrid(np.arange(aud_bin.shape[0]), np.arange(aud_bin.shape[1]),
                              np.arange(aud_bin.shape[2]), indexing="ij")
    ones = np.ones_like(ii, dtype=np.float64)
    world_x = (aff[0, 0] * ii + aff[0, 1] * jj + aff[0, 2] * kk + aff[0, 3] * ones)
    is_left = world_x < 0

    results = {"anat": ANAT, "atlas": HO_CORT, "labels_used": AUD_CTX_LABELS,
               "per_label_raw_voxel_counts": per_label, "hemis": {}}
    merged = np.zeros(aud_bin.shape, bool)
    for hemi, sel in (("L", is_left), ("R", ~is_left)):
        m = aud_bin & sel
        n_before = int(m.sum())
        m_cc = largest_cc(m) if n_before > 0 else m
        n_cc = int(m_cc.sum())
        log(f"   {hemi}: {n_before} vox pre-CC -> {n_cc} vox largest-CC "
            f"({100 * n_cc / max(n_before, 1):.1f}% kept)")
        m_clean = ndimage.binary_fill_holes(m_cc)
        n_final = int(m_clean.sum())
        nib.save(nib.Nifti1Image(m_clean.astype(np.uint8), aff),
                 f"{WORK}/sub-01_auditorycortex_{hemi}_native.nii.gz")
        merged |= m_clean
        cen = np.argwhere(m_clean).mean(0) if n_final else [np.nan] * 3
        w = aff @ np.array([*cen, 1]) if n_final else [np.nan] * 4
        results["hemis"][hemi] = {"nvox_pre_cc": n_before, "nvox_final": n_final,
                                   "centroid_world_mm": [round(float(x), 1) for x in w[:3]]}

    nib.save(nib.Nifti1Image(merged.astype(np.uint8), aff),
              f"{WORK}/sub-01_auditorycortex_native.nii.gz")
    results["merged_nvox"] = int(merged.sum())
    log(f"merged auditory-cortex mask: {int(merged.sum())} vox "
        f"(L={results['hemis']['L']['nvox_final']}, R={results['hemis']['R']['nvox_final']})")

    with open(f"{WORK}/segmentation_stats.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {WORK}/segmentation_stats.json")
    log("DONE segmentation")


if __name__ == "__main__":
    main()
