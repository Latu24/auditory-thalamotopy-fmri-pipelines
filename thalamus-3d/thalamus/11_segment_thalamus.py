"""Thalamus pipeline -- Part B: MGB + whole-thalamus segmentation.

Atlas-warp path (no FreeSurfer dependency). Two matched-template SyN
registrations onto the subject's native 0.7 mm anatomy, so all masks are
co-gridded with the native anatomical image and its affine:

  Reg A: ICBM152 2009a sym T1 (1mm)  -> native  ... warp Sitek MGB (labels 7,8)
  Reg B: FSL MNI152_T1_1mm           -> native  ... warp Harvard-Oxford thalamus

MGB (Sitek in-vivo atlas) labels 7 (L) / 8 (R) are identified by their MNI
centroids (+-15, -26.6, -4.5), the postero-inferior geniculate region. These
are discrete ROI labels (not probabilistic), so no threshold is needed.
Cleanup: largest connected component per hemisphere + hole fill on the
native grid.

Inputs: native anatomical NIfTI, MNI template + atlas volumes (ICBM152,
Sitek MGB, FSL MNI152, Harvard-Oxford subcortical atlas). Outputs: native-
space whole-thalamus and MGB binary masks (merged + per-hemisphere) plus a
JSON segmentation-stats sidecar.
"""
import os, sys, json, time
import numpy as np
import nibabel as nib
from scipy import ndimage
import ants

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

ROOT = PROJECT_ROOT
ANAT = f"{ROOT}/derivatives/sub-01/anat/sub-01_desc-UNIdenoisedN4.nii.gz"
WORK = f"{ROOT}/derivatives/sub-01/analysis/thalamus_work"
ATL = f"{WORK}/atlas"
SITEK = f"{ATL}/sub-invivo_MNI_rois.nii.gz"
ICBM = f"{ATL}/icbm152_2009_t1.nii.gz"
FSLMNI = os.path.expanduser("~/fsl/data/standard/MNI152_T1_1mm.nii.gz")
HO = os.path.expanduser("~/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz")

MGB_LABELS = {7: "L", 8: "R"}          # Sitek in-vivo
HO_THAL = {4: "L", 15: "R"}            # Harvard-Oxford subcortical


def largest_cc(binmask):
    lab, n = ndimage.label(binmask)
    if n == 0:
        return binmask
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep = np.argmax(sizes) + 1
    return lab == keep


def clean_hemis(warped_arr, label_map):
    """warped_arr: float label volume on native grid. label_map: {value:hemi}.
    Returns dict hemi->binary cleaned mask, and merged mask."""
    out = {}
    merged = np.zeros(warped_arr.shape, bool)
    for val, hemi in label_map.items():
        m = warped_arr == val
        if m.sum() == 0:
            log(f"   WARN label {val} ({hemi}) empty after warp")
            continue
        m = largest_cc(m)
        m = ndimage.binary_fill_holes(m)
        out[hemi] = m
        merged |= m
        log(f"   {hemi} (label {val}): {int(m.sum())} vox after CC+fill")
    return out, merged


def resample_label_to(mov_label_img, ref_grid_affine, ref_shape):
    """nearest-neighbor resample a label image into a target grid (world-based)."""
    from nilearn.image import resample_img
    return resample_img(mov_label_img, target_affine=ref_grid_affine,
                        target_shape=ref_shape, interpolation="nearest",
                        force_resample=True, copy_header=True)


def main():
    fixed = ants.image_read(ANAT)
    anat_nib = nib.load(ANAT)
    log(f"native anat {fixed.shape} spacing {np.round(fixed.spacing,3)}")

    results = {"anat": ANAT, "mgb": {}, "thal": {}}

    # ---------- Reg A: ICBM152 2009a -> native (for Sitek MGB) ----------
    log("Reg A: ICBM152-2009a -> native (SyNRA) ...")
    movA = ants.image_read(ICBM)
    regA = ants.registration(fixed=fixed, moving=movA, type_of_transform="SyNRA")
    log("   Reg A done")

    # Sitek MGB label -> downsample to 0.5mm in MNI space, then warp
    log("   loading Sitek atlas (0.1mm) + extracting MGB (7,8) ...")
    sit = nib.load(SITEK)
    sd = np.asarray(sit.dataobj)
    mgb01 = np.isin(sd, list(MGB_LABELS.keys())).astype(np.float32)
    # keep original values (7/8) so hemispheres separable after warp
    mgbval = np.where(np.isin(sd, list(MGB_LABELS.keys())), sd, 0).astype(np.float32)
    del sd
    mgb_img = nib.Nifti1Image(mgbval, sit.affine)
    # downsample to 0.5mm grid in same world space
    aff05 = sit.affine.copy()
    aff05[:3, :3] = sit.affine[:3, :3] * 5.0   # 0.1 -> 0.5mm
    shape05 = tuple(int(np.ceil(s / 5.0)) for s in mgbval.shape)
    mgb05 = resample_label_to(mgb_img, aff05, shape05)
    nib.save(mgb05, f"{WORK}/mgb_sitek_0p5mm_MNI.nii.gz")
    del mgbval, mgb01, mgb_img
    log(f"   MGB 0.5mm grid {mgb05.shape} nvox {int((np.asarray(mgb05.dataobj)>0).sum())}")

    mgb_ants = ants.image_read(f"{WORK}/mgb_sitek_0p5mm_MNI.nii.gz")
    mgb_warp = ants.apply_transforms(fixed=fixed, moving=mgb_ants,
                                     transformlist=regA["fwdtransforms"],
                                     interpolator="genericLabel")
    mgb_arr = mgb_warp.numpy()
    log(f"   warped MGB nonzero {int((mgb_arr>0).sum())} vals {np.unique(mgb_arr)}")
    mgb_h, mgb_merged = clean_hemis(mgb_arr, MGB_LABELS)
    nib.save(nib.Nifti1Image(mgb_merged.astype(np.uint8), anat_nib.affine),
             f"{WORK}/sub-01_MGB_native.nii.gz")
    for hemi, m in mgb_h.items():
        nib.save(nib.Nifti1Image(m.astype(np.uint8), anat_nib.affine),
                 f"{WORK}/sub-01_MGB_{hemi}_native.nii.gz")
        # centroid in world coords for QC
        cen = np.argwhere(m).mean(0)
        w = anat_nib.affine @ np.array([*cen, 1])
        results["mgb"][hemi] = {"nvox": int(m.sum()),
                                "centroid_world": [round(float(x), 1) for x in w[:3]]}
    results["mgb"]["merged_nvox"] = int(mgb_merged.sum())

    # ---------- Reg B: FSL MNI152 -> native (for HO thalamus) ----------
    log("Reg B: FSL MNI152_T1_1mm -> native (SyNRA) ...")
    movB = ants.image_read(FSLMNI)
    regB = ants.registration(fixed=fixed, moving=movB, type_of_transform="SyNRA")
    log("   Reg B done")
    ho = ants.image_read(HO)
    ho_warp = ants.apply_transforms(fixed=fixed, moving=ho,
                                    transformlist=regB["fwdtransforms"],
                                    interpolator="genericLabel")
    ho_arr = ho_warp.numpy()
    thal_h, thal_merged = clean_hemis(ho_arr, HO_THAL)
    nib.save(nib.Nifti1Image(thal_merged.astype(np.uint8), anat_nib.affine),
             f"{WORK}/sub-01_thalamus_native.nii.gz")
    for hemi, m in thal_h.items():
        cen = np.argwhere(m).mean(0)
        w = anat_nib.affine @ np.array([*cen, 1])
        results["thal"][hemi] = {"nvox": int(m.sum()),
                                 "centroid_world": [round(float(x), 1) for x in w[:3]]}
    results["thal"]["merged_nvox"] = int(thal_merged.sum())

    # MGB-inside-thalamus QC
    inside = int((mgb_merged & thal_merged).sum())
    results["mgb_inside_thalamus_frac"] = round(inside / max(mgb_merged.sum(), 1), 3)
    log(f"MGB voxels inside HO thalamus: {inside}/{int(mgb_merged.sum())} "
        f"({results['mgb_inside_thalamus_frac']:.2f})")

    with open(f"{WORK}/segmentation_stats.json", "w") as f:
        json.dump(results, f, indent=2)
    log("DONE segmentation")


if __name__ == "__main__":
    main()
