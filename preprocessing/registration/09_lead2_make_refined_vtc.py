#!/usr/bin/env python3
"""Build refined VTCs (rigid + SyN nonlinear refinement), as additional
outputs alongside the main rigid-only VTCs from 04_make_vtc.py. Input:
per-run preprocessed FMR, mean functional NIfTI, the anatomical NIfTI, and
the per-run SyN transforms from 07_lead2_syn_refinement.py. Output: one
refined VTC per run (suffix "-refined") and a JSON build report.

The bounding box is recomputed fresh from the actual SyN-refined coverage
(rather than reused from the original rigid-only VTC box): a QC check found
the SyN warp's displacement (roughly 7.7-8.5mm across runs) pushes the true
coverage 1-2 voxels beyond the original rigid-derived box on the R-axis
upper side for every run, so generous fresh padding is used here to avoid
any truncation risk. All other conventions (BV-internal coordinate mapping,
VTCResolution=1, float32, data source = the canonical FMR) are identical to
04_make_vtc.py for direct comparability.
"""
import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import nibabel as nib
import ants
import bvbabel.fmr, bvbabel.vtc

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ANAT_NII = f"{PROJECT_ROOT}/derivatives/sub-01/anat/sub-01_desc-UNIdenoisedN4.nii.gz"
FUNC = f"{PROJECT_ROOT}/derivatives/sub-01/func"
REG = f"{PROJECT_ROOT}/derivatives/sub-01/reg"
LOG = f"{PROJECT_ROOT}/logs"
DTYPE = "float32"
RUNS = [1, 2, 3, 4]
PADDING_VOX = 10  # generous padding given the SyN warp's larger displacement range


def bv_internal_from_ras_box(i0, i1, j0, j1, k0, k1):
    ZStart, ZEnd = 240 - i1, 240 - i0
    YStart, YEnd = 320 - k1, 320 - k0
    XStart, XEnd = 320 - j1, 320 - j0
    return XStart, XEnd, YStart, YEnd, ZStart, ZEnd


def ras_to_bv_internal_array(arr4d):
    a = arr4d[::-1, ::-1, ::-1, :]
    a = np.transpose(a, (0, 2, 1, 3))
    return a


def main():
    anat_ants = ants.image_read(ANAT_NII)

    # ---- 1. fresh union bounding box from actual SyN-refined coverage across all 4 runs ----
    ras_boxes = []
    syn_xfm_lists = {}
    for r in RUNS:
        mov = ants.image_read(f"{REG}/sub-01_run-{r}_meanfunc_preproc.nii.gz")
        syn_xfms = [f"{REG}/xfm_lead2_refined/sub-01_run-{r}_syn_fwd_0.nii.gz",
                    f"{REG}/xfm_lead2_refined/sub-01_run-{r}_syn_fwd_1.mat"]
        syn_xfm_lists[r] = syn_xfms
        warped = ants.apply_transforms(fixed=anat_ants, moving=mov, transformlist=syn_xfms, interpolator="linear")
        arr = warped.numpy()
        nz = arr > np.percentile(arr[arr > 0], 1)
        idx = np.where(nz)
        i0, i1 = int(idx[0].min()), int(idx[0].max()) + 1
        j0, j1 = int(idx[1].min()), int(idx[1].max()) + 1
        k0, k1 = int(idx[2].min()), int(idx[2].max()) + 1
        ras_boxes.append((i0, i1, j0, j1, k0, k1))
        print(f"run-{r} SyN-refined nonzero RAS-index box: i[{i0}:{i1}] j[{j0}:{j1}] k[{k0}:{k1}]")

    i0 = max(min(b[0] for b in ras_boxes) - PADDING_VOX, 0)
    i1 = min(max(b[1] for b in ras_boxes) + PADDING_VOX, 240)
    j0 = max(min(b[2] for b in ras_boxes) - PADDING_VOX, 0)
    j1 = min(max(b[3] for b in ras_boxes) + PADDING_VOX, 320)
    k0 = max(min(b[4] for b in ras_boxes) - PADDING_VOX, 0)
    k1 = min(max(b[5] for b in ras_boxes) + PADDING_VOX, 320)
    print(f"UNION bounding box (+{PADDING_VOX}vox padding, fresh from SyN coverage): "
          f"i[{i0}:{i1}] j[{j0}:{j1}] k[{k0}:{k1}]  shape=({i1-i0},{j1-j0},{k1-k0})")

    XStart, XEnd, YStart, YEnd, ZStart, ZEnd = bv_internal_from_ras_box(i0, i1, j0, j1, k0, k1)
    cropped_ref = ants.crop_indices(anat_ants, (i0, j0, k0), (i1, j1, k1))
    print(f"BV-internal box: X[{XStart}:{XEnd}] Y[{YStart}:{YEnd}] Z[{ZStart}:{ZEnd}]  "
          f"cropped ref shape {cropped_ref.shape}")

    bbox_meta = dict(ras_box=dict(i0=i0, i1=i1, j0=j0, j1=j1, k0=k0, k1=k1),
                      bv_internal=dict(XStart=XStart, XEnd=XEnd, YStart=YStart, YEnd=YEnd, ZStart=ZStart, ZEnd=ZEnd),
                      padding_vox=PADDING_VOX,
                      per_run_syn_ras_boxes={f"run-{r}": b for r, b in zip(RUNS, ras_boxes)})

    # ---- 2. resample every volume through rigid+SyN into the cropped reference, build refined VTC ----
    vtc_report = {}
    for r in RUNS:
        t0 = time.time()
        fmr_path = f"{FUNC}/sub-01_run-{r}_preproc.fmr"
        fh, fdata = bvbabel.fmr.read_fmr(fmr_path)
        T = fdata.shape[-1]
        ref_ants = ants.image_read(f"{REG}/sub-01_run-{r}_meanfunc_preproc.nii.gz")
        syn_xfms = syn_xfm_lists[r]

        out = np.zeros((i1 - i0, j1 - j0, k1 - k0, T), dtype=np.float32)
        for t in range(T):
            vol_ants = ref_ants.new_image_like(fdata[..., t].astype(np.float32))
            warped_vol = ants.apply_transforms(fixed=cropped_ref, moving=vol_ants,
                                                transformlist=syn_xfms, interpolator="linear")
            out[..., t] = warped_vol.numpy()
            if t % 100 == 0:
                print(f"  run-{r} vol {t}/{T} ({time.time()-t0:.0f}s elapsed)")

        nan_ct = int(np.isnan(out).sum()); inf_ct = int(np.isinf(out).sum())
        if nan_ct or inf_ct:
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, 0, None)

        bv_data = ras_to_bv_internal_array(out)

        header, _ = bvbabel.vtc.create_vtc()
        header["File version"] = 3
        header["Source FMR name"] = os.path.basename(fmr_path)
        header["Protocol attached"] = 0
        header["Protocol name"] = ""
        header["Current protocol index"] = 0
        header["Data type (1:short int, 2:float)"] = 2
        header["Nr time points"] = T
        header["VTC resolution relative to VMR (1, 2, or 3)"] = 1
        header["XStart"], header["XEnd"] = XStart, XEnd
        header["YStart"], header["YEnd"] = YStart, YEnd
        header["ZStart"], header["ZEnd"] = ZStart, ZEnd
        header["L-R convention (0:unknown, 1:radiological, 2:neurological)"] = 1
        header["Reference space (0:unknown, 1:native, 2:ACPC, 3:Tal, 4:MNI)"] = 1
        header["TR (ms)"] = float(fh["TR"])

        out_path = f"{REG}/sub-01_run-{r}_preproc_coreg-refined.vtc"
        bvbabel.vtc.write_vtc(out_path, header, bv_data, rearrange_data_axes=False)

        elapsed = time.time() - t0
        vtc_report[f"run-{r}"] = {
            "out": out_path, "n_volumes": T, "bv_internal_shape_ZYX": list(bv_data.shape[:3]),
            "XStart": XStart, "XEnd": XEnd, "YStart": YStart, "YEnd": YEnd, "ZStart": ZStart, "ZEnd": ZEnd,
            "VTCResolution": 1, "TR_ms": float(fh["TR"]), "dtype": DTYPE,
            "nan_fixed": nan_ct, "inf_fixed": inf_ct, "elapsed_s": elapsed,
            "file_size_bytes": os.path.getsize(out_path),
        }
        print(f"run-{r}: wrote {out_path} ({T} vols, {elapsed:.0f}s, {os.path.getsize(out_path)/1e9:.2f} GB)")

    report = {"bbox": bbox_meta, "vtc": vtc_report, "syn_transforms_used": syn_xfm_lists, "dtype": DTYPE}
    with open(f"{LOG}/registration_09_lead2_make_refined_vtc.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote logs/registration_09_lead2_make_refined_vtc.json")


if __name__ == "__main__":
    main()
