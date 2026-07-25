#!/usr/bin/env python3
"""Build topup fieldmap inputs (first 5 volumes of each AP run plus the PA
spin-echo pair) and trim/threshold the AP BOLD runs ahead of slice-timing
correction. Input: raw 4D BOLD NIfTI runs and the PA spin-echo NIfTI, plus
their JSON sidecars. Output: trimmed/thresholded NIfTI volumes, topup input
volumes, and an initial trimmed+thresholded FMR per run.

Volume counts (magnitude BOLD, verified): run1=355, run2/3/4=345, PA=10.
After trimming the last 5: run1=350, runs2-4=340. Topup inputs use the
first 5 volumes (copied, not removed from the run), so PRT onset indexing,
which starts at volume 0, stays valid. Every stage below writes a new file
rather than overwriting a previous one.
"""
import os, json, numpy as np, nibabel as nib
import bvbabel.fmr as bvfmr

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RAW = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "rawdata_nifti")
FUNC = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "func")
os.makedirs(FUNC, exist_ok=True)

CUTVAL = 32          # background intensity floor: zeroes out sub-threshold background/air voxels
TRIM_TRAILING = 5    # exclude last 5 volumes per AP run
TOPUP_NVOL = 5        # first 5 vols -> topup input

RUNS = {
    1: f"{RAW}/sub-01_task-asta_run-1_bold.nii.gz",
    2: f"{RAW}/sub-01_task-asta_run-2_bold.nii.gz",
    3: f"{RAW}/sub-01_task-asta_run-3_bold.nii.gz",
    4: f"{RAW}/sub-01_task-asta_run-4_bold.nii.gz",
}
PA = f"{RAW}/sub-01_task-asta_dir-PA_epi.nii.gz"

def apply_cut(arr, cutval=CUTVAL):
    out = arr.copy()
    out[out < cutval] = 0
    return out

def fmr_header_from_nifti(img, nvol, nslices, prefix, tr_ms, te_ms, slice_times_s):
    hdr, _ = bvfmr.create_fmr()
    zooms = img.header.get_zooms()
    hdr["FileVersion"] = 7
    hdr["NrOfVolumes"] = int(nvol)
    hdr["NrOfSlices"] = int(nslices)
    hdr["NrOfSkippedVolumes"] = 0
    hdr["Prefix"] = prefix
    hdr["DataStorageFormat"] = 2   # STC data in separate file
    hdr["DataType"] = 2            # uint16 (per bvbabel convention 2->short)
    hdr["TR"] = float(tr_ms)
    # InterSliceTime: mean spacing between consecutive slice acquisition times
    st = np.array(sorted(slice_times_s))
    hdr["InterSliceTime"] = float(np.round(np.mean(np.diff(st))*1000.0, 3))
    hdr["TE"] = float(te_ms)
    hdr["SliceAcquisitionOrder"] = 0   # 0 = explicit/unknown; real times come from the JSON sidecar
    hdr["SliceAcquisitionOrderVerified"] = 0
    # NOTE: bvbabel write_fmr/read_fmr only round-trip non-square in-plane
    # matrices when ResolutionX=data.shape[1] and ResolutionY=data.shape[0]
    # (read_fmr returns array shaped (ResolutionY, ResolutionX, slices, time)).
    # Data is fed as (X=shape0, Y=shape1, slices, time); read-back matches exactly.
    hdr["ResolutionX"] = int(img.shape[1])
    hdr["ResolutionY"] = int(img.shape[0])
    hdr["InplaneResolutionX"] = float(round(zooms[1], 4))
    hdr["InplaneResolutionY"] = float(round(zooms[0], 4))
    hdr["SliceThickness"] = float(round(zooms[2], 4))
    hdr["SliceGap"] = 0.0
    hdr["LayoutNColumns"] = int(np.ceil(np.sqrt(nslices)))
    hdr["LayoutNRows"] = int(np.ceil(np.sqrt(nslices)))
    hdr["VoxelResolutionVerified"] = 1
    hdr["TimeResolutionVerified"] = 1
    return hdr

def write_fmr(path, img4d_data, img_ref, nvol, prefix, tr_ms, te_ms, slice_times_s):
    nslices = img4d_data.shape[2]
    hdr = fmr_header_from_nifti(img_ref, nvol, nslices, prefix, tr_ms, te_ms, slice_times_s)
    data = np.ascontiguousarray(img4d_data.astype(np.uint16))  # (x,y,slices,time)
    bvfmr.write_fmr(path, hdr, data, rearrange_data_axes=True)

def main():
    jrun1 = json.load(open(f"{RAW}/sub-01_task-asta_run-1_bold.json"))
    tr_ms = jrun1["RepetitionTime"] * 1000.0
    te_ms = jrun1["EchoTime"] * 1000.0
    slice_times = jrun1["SliceTiming"]  # seconds

    # ---- PA topup input (first 5 vols, cut) ----
    pa = nib.load(PA)
    pa_data = np.asarray(pa.dataobj[..., :TOPUP_NVOL], dtype=np.float32)
    pa_cut = apply_cut(pa_data)
    nib.save(nib.Nifti1Image(pa_cut.astype(np.float32), pa.affine, pa.header),
             f"{FUNC}/sub-01_dir-PA_b0-first5_cut.nii.gz")
    print(f"[PA] topup input first{TOPUP_NVOL} cut -> {pa_cut.shape}")

    for r, path in RUNS.items():
        img = nib.load(path)
        nvol_full = img.shape[3]
        nvol_trim = nvol_full - TRIM_TRAILING
        print(f"[run{r}] full={nvol_full} -> trimmed={nvol_trim}")

        # AP topup input: first 5 vols, cut
        ap5 = np.asarray(img.dataobj[..., :TOPUP_NVOL], dtype=np.float32)
        ap5_cut = apply_cut(ap5)
        nib.save(nib.Nifti1Image(ap5_cut.astype(np.float32), img.affine, img.header),
                 f"{FUNC}/sub-01_run-{r}_AP_b0-first5_cut.nii.gz")

        # Trim trailing 5, then cut floor
        data = np.asarray(img.dataobj[..., :nvol_trim], dtype=np.float32)
        # stage 01a: trimmed (pre-cut) saved as NIfTI
        nib.save(nib.Nifti1Image(data.astype(np.float32), img.affine, img.header),
                 f"{FUNC}/sub-01_run-{r}_stage-01trim.nii.gz")
        # stage 01b: cut
        data_cut = apply_cut(data)
        nib.save(nib.Nifti1Image(data_cut.astype(np.float32), img.affine, img.header),
                 f"{FUNC}/sub-01_run-{r}_stage-02cut.nii.gz")

        # Initial trimmed+cut FMR (bvbabel) -- deliverable "run FMR"
        write_fmr(f"{FUNC}/sub-01_run-{r}_stage-02cut.fmr",
                  data_cut, img, nvol_trim, f"sub-01_run-{r}_stage-02cut",
                  tr_ms, te_ms, slice_times)
        print(f"[run{r}] wrote stage-01trim, stage-02cut (nii+fmr)")

if __name__ == "__main__":
    main()
