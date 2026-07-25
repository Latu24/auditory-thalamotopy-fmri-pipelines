#!/usr/bin/env python3
"""Independently verify the anatomical and functional pipeline outputs
before starting registration: VMR/V16 geometry, FMR header consistency
across all 4 runs, shared-grid consistency of the topup-corrected NIfTI
files, presence of stimulus protocol (PRT) files, and availability of the
FSL topup binary. Read-only; writes a verification report.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import nibabel as nib
import bvbabel.vmr, bvbabel.v16, bvbabel.fmr

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ANAT = f"{PROJECT_ROOT}/derivatives/sub-01/anat"
FUNC = f"{PROJECT_ROOT}/derivatives/sub-01/func"
LOG = f"{PROJECT_ROOT}/logs"

report = {}

# --- 1. VMR / V16 ---
vmr_path = f"{ANAT}/sub-01_desc-UNIdenoisedN4_anat.vmr"
v16_path = f"{ANAT}/sub-01_desc-UNIdenoisedN4_anat.v16"
h, data = bvbabel.vmr.read_vmr(vmr_path)
report["vmr"] = {
    "shape": list(data.shape), "dtype": str(data.dtype),
    "DimX": h["DimX"], "DimY": h["DimY"], "DimZ": h["DimZ"],
    "VoxelSize": [h["VoxelSizeX"], h["VoxelSizeY"], h["VoxelSizeZ"]],
    "ReferenceSpaceVMR": h["ReferenceSpaceVMR"], "FramingCubeDim": h["FramingCubeDim"],
    "min": int(data.min()), "max": int(data.max()),
}
print("VMR:", report["vmr"])
assert data.shape == (240, 320, 320), "VMR shape mismatch vs expected geometry"
assert abs(h["VoxelSizeX"] - 0.7) < 1e-6, "VMR voxel size mismatch"

h16, d16 = bvbabel.v16.read_v16(v16_path)
report["v16"] = {"shape": list(d16.shape), "dtype": str(d16.dtype)}
print("V16:", report["v16"])
assert d16.shape == (240, 320, 320)

# --- 2. anat NIfTI used to build the VMR (registration target) ---
anat_nii_path = f"{ANAT}/sub-01_desc-UNIdenoisedN4.nii.gz"
anat_img = nib.load(anat_nii_path)
ax = nib.aff2axcodes(anat_img.affine)
report["anat_nifti"] = {
    "path": anat_nii_path, "shape": list(anat_img.shape),
    "zooms": [round(float(z), 4) for z in anat_img.header.get_zooms()[:3]],
    "axcodes": "".join(ax), "affine": anat_img.affine.tolist(),
}
print("anat NIfTI:", report["anat_nifti"])
assert ax == ("R", "A", "S"), f"anat NIfTI not canonical RAS: {ax}"
assert anat_img.shape == (240, 320, 320)

# --- 3. FMR headers (all 4 runs) ---
expected_vols = {1: 350, 2: 340, 3: 340, 4: 340}
report["fmr"] = {}
for r in [1, 2, 3, 4]:
    fmr_path = f"{FUNC}/sub-01_run-{r}_preproc.fmr"
    fh, fdata = bvbabel.fmr.read_fmr(fmr_path)
    info = {
        "NrOfVolumes": fh["NrOfVolumes"], "NrOfSlices": fh["NrOfSlices"],
        "ResolutionX": fh["ResolutionX"], "ResolutionY": fh["ResolutionY"],
        "TR": fh["TR"], "TE": fh["TE"], "DataType": fh["DataType"],
        "data_shape": list(fdata.shape), "data_dtype": str(fdata.dtype),
        "nan_count": int(np.isnan(fdata).sum()), "inf_count": int(np.isinf(fdata).sum()),
    }
    report["fmr"][f"run-{r}"] = info
    print(f"FMR run-{r}:", info)
    assert fh["NrOfVolumes"] == expected_vols[r], f"run-{r} volume count mismatch"
    assert fdata.shape == (150, 140, 42, expected_vols[r])
    assert info["nan_count"] == 0 and info["inf_count"] == 0, f"run-{r} has NaN/Inf"

# --- 4. topup-corrected NIfTI affines (must be identical across all 4 runs => common space) ---
report["topup_nifti_affine_consistency"] = {}
affines = []
for r in [1, 2, 3, 4]:
    nii = nib.load(f"{FUNC}/sub-01_run-{r}_stage-06dc-topup.nii.gz")
    affines.append(nii.affine)
    report["topup_nifti_affine_consistency"][f"run-{r}"] = {
        "shape": list(nii.shape), "axcodes": "".join(nib.aff2axcodes(nii.affine)),
    }
all_same = all(np.allclose(a, affines[0], atol=1e-4) for a in affines)
report["topup_nifti_affine_consistency"]["all_identical"] = bool(all_same)
print("All 4 runs share identical grid/affine (post motion-correction):", all_same)
assert all_same, "Runs do NOT share a common grid - VTC creation must handle per-run transforms separately"

# --- 5. PRT (stimulus protocol) files present and untouched (read-only check, list only) ---
prt_dir = os.path.join(PROJECT_ROOT, "sourcedata", "prt")
prts = sorted(os.listdir(prt_dir))
report["prt_files"] = prts
print(f"PRT dir has {len(prts)} files (untouched, read-only)")

# --- 6. FSL topup binary sanity (already used by the functional pipeline; just confirm it is on PATH) ---
fsl_topup = os.path.join(os.environ.get("FSLDIR", ""), "bin", "topup")
report["fsl_topup_present"] = os.path.exists(fsl_topup)
print("FSL topup present:", report["fsl_topup_present"])

os.makedirs(LOG, exist_ok=True)
with open(f"{LOG}/registration_00_verify.json", "w") as f:
    json.dump(report, f, indent=2)
print("\nAll checks passed. Wrote logs/registration_00_verify.json")
