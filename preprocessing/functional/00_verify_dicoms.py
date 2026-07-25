#!/usr/bin/env python3
"""Verify raw DICOM series for sub-01 against the expected acquisition
protocol. Reads one representative DICOM header per series and reports
geometry, timing, and phase-encoding metadata. Read-only: never modifies
raw data.
"""
import os, glob, sys
import pydicom
import numpy as np

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RAW = os.path.join(PROJECT_ROOT, "sourcedata", "dicom")

SERIES = [
    "cmrr_mbep2d_bold_pt8FOLD_G2_MB2_AP_run1",
    "cmrr_mbep2d_bold_pt8FOLD_G2_MB2_AP_run1_SBRef",
    "cmrr_mbep2d_bold_pt8FOLD_G2_MB2_AP_run2",
    "cmrr_mbep2d_bold_pt8FOLD_G2_MB2_AP_run2_SBRef",
    "cmrr_mbep2d_bold_pt8FOLD_G2_MB2_AP_run3",
    "cmrr_mbep2d_bold_pt8FOLD_G2_MB2_AP_run3_SBRef",
    "cmrr_mbep2d_bold_pt8FOLD_G2_MB2_AP_run4",
    "cmrr_mbep2d_bold_pt8FOLD_G2_MB2_AP_run4_SBRef",
    "cmrr_mbep2d_bold_pt8FOLD_G2_MB2_PA",
    "cmrr_mbep2d_bold_pt8FOLD_G2_MB2_PA_SBRef",
    "mp2rage_iso0.7mm_iPAT3_UNI_Images",
    "mp2rage_iso0.7mm_iPAT3_INV1",
    "mp2rage_iso0.7mm_iPAT3_INV2",
    "mp2rage_iso0.7mm_iPAT3_T1_Images",
]

def get_private(ds, group, elem):
    try:
        return ds[(group, elem)].value
    except Exception:
        return None

def num_slices_mosaic(ds):
    # CMRR stores NumberOfImagesInMosaic in private tag (0019,100a)
    v = get_private(ds, 0x0019, 0x100a)
    return int(v) if v is not None else None

for s in SERIES:
    d = os.path.join(RAW, s)
    files = sorted(glob.glob(os.path.join(d, "*")))
    files = [f for f in files if os.path.isfile(f)]
    nfiles = len(files)
    ds = pydicom.dcmread(files[0], stop_before_pixels=False)
    itype = list(getattr(ds, "ImageType", []))
    is_mosaic = "MOSAIC" in itype
    rows = int(getattr(ds, "Rows", 0))
    cols = int(getattr(ds, "Columns", 0))
    tr = getattr(ds, "RepetitionTime", None)
    te = getattr(ds, "EchoTime", None)
    px = getattr(ds, "PixelSpacing", None)
    thick = getattr(ds, "SliceThickness", None)
    pedir = getattr(ds, "InPlanePhaseEncodingDirection", None)
    acqmat = getattr(ds, "AcquisitionMatrix", None)
    nmos = num_slices_mosaic(ds)
    iop = getattr(ds, "ImageOrientationPatient", None)
    # bandwidth per pixel PE (0019,1028) CMRR private
    bppe = get_private(ds, 0x0019, 0x1028)
    print(f"### {s}")
    print(f"    files={nfiles}  ImageType={itype}")
    print(f"    Rows x Cols = {rows} x {cols}   AcqMatrix={acqmat}")
    print(f"    mosaic={is_mosaic}  NumberOfImagesInMosaic(0019,100a)={nmos}")
    print(f"    TR={tr}  TE={te}  PixelSpacing={px}  SliceThickness={thick}")
    print(f"    PhaseEncDir(InPlane)={pedir}  BandwidthPerPixelPE(0019,1028)={bppe}")
    print(f"    ImageOrientationPatient={iop}")
    if is_mosaic and nmos:
        print(f"    => derived volumes = {nfiles} (mosaic: 1 file/volume), slices/vol={nmos}")
    else:
        print(f"    => derived volumes = {nfiles} files (single-slice or SBRef)")
    print()
