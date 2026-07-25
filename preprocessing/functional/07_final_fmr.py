#!/usr/bin/env python3
"""Write the final preprocessed FMR (BrainVoyager-native format, via
bvbabel) for each run, selecting the distortion-correction method (topup or
ANTs) per run based on the NCC(AP,PA) comparison recorded by the distortion
QC step. Final = trimmed + thresholded + slice-time-corrected +
motion-corrected + high-pass-filtered + distortion-corrected (winning
method per run). Input: per-run distortion-corrected NIfTI (topup and/or
ANTs) plus JSON sidecars. Output: one FMR per run and a JSON record of the
method chosen per run.

DataType=2 (float32) is used to preserve the processed intensity values
rather than truncating to an integer type.
"""
import os, sys, json, numpy as np, nibabel as nib
import bvbabel.fmr as bvfmr

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FUNC = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "func")
RAW = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "rawdata_nifti")
LOG = os.path.join(PROJECT_ROOT, "logs")
RUNS = [1, 2, 3, 4]
TAG = "v2"
SUFFIX = f"{TAG}_preproc"

def build_header(img, nvol, nslices, prefix, tr_ms, te_ms, slice_times_s):
    hdr, _ = bvfmr.create_fmr()
    zooms = img.header.get_zooms()
    hdr["FileVersion"] = 7
    hdr["NrOfVolumes"] = int(nvol); hdr["NrOfSlices"] = int(nslices)
    hdr["NrOfSkippedVolumes"] = 0
    hdr["Prefix"] = prefix
    hdr["DataStorageFormat"] = 2; hdr["DataType"] = 2   # float32 STC
    hdr["TR"] = float(tr_ms)
    st = np.array(sorted(slice_times_s))
    hdr["InterSliceTime"] = float(np.round(np.mean(np.diff(st)) * 1000.0, 3))
    hdr["TE"] = float(te_ms)
    hdr["SliceAcquisitionOrder"] = 0; hdr["SliceAcquisitionOrderVerified"] = 0
    hdr["ResolutionX"] = int(img.shape[1]); hdr["ResolutionY"] = int(img.shape[0])
    hdr["InplaneResolutionX"] = float(round(zooms[1], 4))
    hdr["InplaneResolutionY"] = float(round(zooms[0], 4))
    hdr["SliceThickness"] = float(round(zooms[2], 4)); hdr["SliceGap"] = 0.0
    hdr["LayoutNColumns"] = int(np.ceil(np.sqrt(nslices)))
    hdr["LayoutNRows"] = int(np.ceil(np.sqrt(nslices)))
    hdr["VoxelResolutionVerified"] = 1; hdr["TimeResolutionVerified"] = 1
    return hdr

def choose_method(r, qc):
    rr = qc.get(str(r), {})
    t = rr.get("ncc_AP_PA_topup"); a = rr.get("ncc_AP_PA_ants")
    if t is None and a is None:
        raise RuntimeError(f"run{r}: no topup or ANTs NCC available in qc_distortion_v2.json")
    if a is None:
        return "topup", t, a
    if t is None:
        return "ants", t, a
    return ("topup" if t >= a else "ants"), t, a

def main():
    qc_path = f"{LOG}/qc_distortion_{TAG}.json"
    if not os.path.exists(qc_path):
        print(f"ERROR: {qc_path} not found -- run 08_distortion_qc.py first"); sys.exit(1)
    qc = json.load(open(qc_path))

    choices = {}
    for r in RUNS:
        j = json.load(open(f"{RAW}/sub-01_task-asta_run-{r}_bold.json"))  # each run's own JSON sidecar
        tr_ms = j["RepetitionTime"] * 1000.0; te_ms = j["EchoTime"] * 1000.0
        slice_times = j["SliceTiming"]

        method, t, a = choose_method(r, qc)
        choices[r] = dict(method=method, ncc_AP_PA_topup=t, ncc_AP_PA_ants=a)
        stage = f"{TAG}_stage-06dc-{method}"
        path = f"{FUNC}/sub-01_run-{r}_{stage}.nii.gz"
        if not os.path.exists(path):
            print(f"[run{r}] SKIP (missing {path})"); continue
        img = nib.load(path)
        data = np.asarray(img.dataobj, dtype=np.float32)
        data[data < 0] = 0.0
        X, Y, S, T = data.shape
        prefix = f"sub-01_run-{r}_{SUFFIX}"
        hdr = build_header(img, T, S, prefix, tr_ms, te_ms, slice_times)
        out = f"{FUNC}/sub-01_run-{r}_{SUFFIX}.fmr"
        bvfmr.write_fmr(out, hdr, np.ascontiguousarray(data), rearrange_data_axes=True)
        print(f"[run{r}] method={method} (topup={t} ants={a})  wrote {out}  ({X}x{Y}x{S}x{T}, float32)")

    json.dump(choices, open(f"{LOG}/qc_final_fmr_{TAG}_method_choice.json", "w"), indent=2)
    print(f"wrote logs/qc_final_fmr_{TAG}_method_choice.json")

if __name__ == "__main__":
    main()
