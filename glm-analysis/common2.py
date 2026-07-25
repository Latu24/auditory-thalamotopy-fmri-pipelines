"""Paths for the refined-VTC / SDM-based round of GLM analyses. Additive
relative to common.py — it does not modify that module, so the earlier
driver scripts remain reproducible against the original (non-refined) VTCs.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import common as C1  # read-only reuse of FREQ_ORDER/AM_ORDER and shared paths

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ROOT = C1.ROOT
ANA = C1.ANA
QC = C1.QC
LOGS = C1.LOGS

# Refined VTCs (ANTs SyN refinement on top of topup + rigid coregistration)
VTCS_REFINED = [
    f"{ROOT}/derivatives/sub-01/reg/sub-01_run-{r}_preproc_coreg-refined.vtc"
    for r in (1, 2, 3, 4)
]

# Corrected VMR (ReferenceSpaceVMR header flag updated; voxel data
# byte-identical to the original)
VMR_PATH = os.path.join(PROJECT_ROOT, "docs", "sub-01_desc-UNIdenoisedN4_anat_refspace-native.vmr")

# SDM predictor files (nilearn make_first_level_design_matrix, hrf_model="spm")
CF_SDM = [os.path.join(PROJECT_ROOT, "docs", f"sub-01_run-{r}_CF_only.sdm")
          for r in (1, 2, 3, 4)]
AM_SDM = [os.path.join(PROJECT_ROOT, "docs", f"sub-01_run-{r}_AM_only.sdm")
          for r in (1, 2, 3, 4)]

FREQ_ORDER = C1.FREQ_ORDER   # Freq_01..Freq_36
AM_ORDER = C1.AM_ORDER       # AM_1..AM_9

# Anatomical NIfTI the VMR is built from (voxel data unchanged; only the
# VMR header's reference-space flag was updated)
ANAT_NII = f"{ROOT}/derivatives/sub-01/anat/sub-01_desc-UNIdenoisedN4.nii.gz"
