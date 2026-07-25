#!/usr/bin/env python3
"""Create the final analysis-ready BrainVoyager VTC files, one per run,
folding in motion correction, distortion correction (topup), slice-timing
correction, and high-pass filtering (all baked into the input preproc.fmr
by the functional pipeline) plus the func<->anat coregistration transform
derived in the coregistration steps. Input: per-run preprocessed FMR, the
per-run mean-functional NIfTI, the anatomical N4-corrected NIfTI, and the
coregistration decision record. Output: one VTC per run under
derivatives/sub-01/reg and a JSON build report.

BV-internal coordinate contract (independently derived from bvbabel.vmr's
write_vmr() implementation, matching BrainVoyager's documented internal
convention "X=ant->post, Y=sup->inf, Z=right->left"):

  Given the anat NIfTI (canonical RAS, shape (240,320,320), axes
  i=R-axis(0..239), j=A-axis(0..319), k=S-axis(0..319)) that the VMR was
  built from directly (same array, same affine - no reorientation needed):

      BV_Z = 239 - i        (Z increasing = Right -> Left)
      BV_Y = 319 - k        (Y increasing = Superior -> Inferior)
      BV_X = 319 - j        (X increasing = Anterior -> Posterior)

  and the VMR/VTC on-disk data loop order is (DimZ outer, DimY, DimX inner,
  [DimT innermost for VTC]), obtained from a NIfTI-RAS array via:

      arr[::-1, ::-1, ::-1, ...]        # flip all 3 spatial axes
      np.transpose(arr, (0, 2, 1, ...)) # swap axis1 <-> axis2

  This is the same flip+transpose sequence 03_make_vmr.py used (verified
  from the bvbabel.vmr.write_vmr source), applied here identically so the
  VTC bounding box lines up with the VMR by construction.

VTC resolution chosen: VTCResolution=1 (i.e. 0.7mm, matching the VMR's
native grid exactly). The functional data is only a thin ~150x140x42 slab,
so a full-resolution VTC cropped to a tight bounding box around the
coverage is compact per run and preserves all information; a coarser
VTCResolution would only lose precision with no size benefit at this scale.

Data type: float32 (BV "Data type 2") is used to preserve the full
HPF'd/topup-corrected intensity precision (values already span roughly
0-40000), which would need lossy rescaling to fit a signed int16 range. An
int16 fallback path is available via the DTYPE constant below if a given
BrainVoyager version requires it.
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
DTYPE = "float32"  # or "int16" fallback, see module docstring

RUNS = [1, 2, 3, 4]
PADDING_VOX = 6  # extra 0.7mm voxels of padding around the transformed coverage


def bv_internal_from_ras_box(i0, i1, j0, j1, k0, k1):
    """RAS voxel-index box [i0,i1)x[j0,j1)x[k0,k1) (R,A,S axes) -> BV-internal
    (XStart,XEnd,YStart,YEnd,ZStart,ZEnd)."""
    ZStart, ZEnd = 240 - i1, 240 - i0
    YStart, YEnd = 320 - k1, 320 - k0
    XStart, XEnd = 320 - j1, 320 - j0
    return XStart, XEnd, YStart, YEnd, ZStart, ZEnd


def ras_to_bv_internal_array(arr4d):
    """arr4d: (ni,nj,nk,T) NIfTI-RAS-ordered -> (DimZ,DimY,DimX,T) BV-internal,
    identical flip+transpose sequence to bvbabel.vmr.write_vmr()."""
    a = arr4d[::-1, ::-1, ::-1, :]
    a = np.transpose(a, (0, 2, 1, 3))
    return a


def main():
    anat_nib = nib.load(ANAT_NII)
    anat_ants = ants.image_read(ANAT_NII)
    with open(f"{REG}/xfm/decision.json") as f:
        decision = json.load(f)
    shared_xfm = decision["shared_transform_mat"]
    if not decision["use_shared_transform"]:
        raise RuntimeError("per-run transform path not implemented - decision.json says shared transform not used")
    print("Using shared transform for all runs:", shared_xfm)

    # ---- 1. compute per-run transformed bounding boxes, take the UNION so all 4 VTCs share identical dims ----
    ras_boxes = []
    for r in RUNS:
        mov = ants.image_read(f"{REG}/sub-01_run-{r}_meanfunc_preproc.nii.gz")
        warped = ants.apply_transforms(fixed=anat_ants, moving=mov, transformlist=[shared_xfm], interpolator="linear")
        arr = warped.numpy()
        nz = arr > np.percentile(arr[arr > 0], 1)
        idx = np.where(nz)
        i0, i1 = int(idx[0].min()), int(idx[0].max()) + 1
        j0, j1 = int(idx[1].min()), int(idx[1].max()) + 1
        k0, k1 = int(idx[2].min()), int(idx[2].max()) + 1
        ras_boxes.append((i0, i1, j0, j1, k0, k1))
        print(f"run-{r} transformed nonzero RAS-index box: i[{i0}:{i1}] j[{j0}:{j1}] k[{k0}:{k1}]")

    i0 = min(b[0] for b in ras_boxes) - PADDING_VOX
    i1 = max(b[1] for b in ras_boxes) + PADDING_VOX
    j0 = min(b[2] for b in ras_boxes) - PADDING_VOX
    j1 = max(b[3] for b in ras_boxes) + PADDING_VOX
    k0 = min(b[4] for b in ras_boxes) - PADDING_VOX
    k1 = max(b[5] for b in ras_boxes) + PADDING_VOX
    i0, j0, k0 = max(i0, 0), max(j0, 0), max(k0, 0)
    i1, j1, k1 = min(i1, 240), min(j1, 320), min(k1, 320)
    print(f"UNION bounding box (+{PADDING_VOX}vox padding), used for ALL 4 runs: "
          f"i[{i0}:{i1}] j[{j0}:{j1}] k[{k0}:{k1}]  shape=({i1-i0},{j1-j0},{k1-k0})")

    XStart, XEnd, YStart, YEnd, ZStart, ZEnd = bv_internal_from_ras_box(i0, i1, j0, j1, k0, k1)
    print(f"BV-internal box: X[{XStart}:{XEnd}] Y[{YStart}:{YEnd}] Z[{ZStart}:{ZEnd}]")

    cropped_ref = ants.crop_indices(anat_ants, (i0, j0, k0), (i1, j1, k1))
    print("cropped reference shape:", cropped_ref.shape, "spacing:", cropped_ref.spacing)

    bbox_meta = dict(ras_box=dict(i0=i0, i1=i1, j0=j0, j1=j1, k0=k0, k1=k1),
                      bv_internal=dict(XStart=XStart, XEnd=XEnd, YStart=YStart, YEnd=YEnd, ZStart=ZStart, ZEnd=ZEnd),
                      per_run_ras_boxes={f"run-{r}": b for r, b in zip(RUNS, ras_boxes)})

    # ---- 2. for each run, resample every volume into the cropped reference, build VTC ----
    vtc_report = {}
    for r in RUNS:
        t0 = time.time()
        fmr_path = f"{FUNC}/sub-01_run-{r}_preproc.fmr"
        fh, fdata = bvbabel.fmr.read_fmr(fmr_path)  # (150,140,42,T) float32, matches meanfunc geometry
        T = fdata.shape[-1]
        ref_ants = ants.image_read(f"{REG}/sub-01_run-{r}_meanfunc_preproc.nii.gz")

        out = np.zeros((i1 - i0, j1 - j0, k1 - k0, T), dtype=np.float32)
        for t in range(T):
            vol_ants = ref_ants.new_image_like(fdata[..., t].astype(np.float32))
            warped_vol = ants.apply_transforms(fixed=cropped_ref, moving=vol_ants,
                                                transformlist=[shared_xfm], interpolator="linear")
            out[..., t] = warped_vol.numpy()
            if t % 100 == 0:
                print(f"  run-{r} vol {t}/{T} ({time.time()-t0:.0f}s elapsed)")

        nan_ct = int(np.isnan(out).sum()); inf_ct = int(np.isinf(out).sum())
        if nan_ct or inf_ct:
            print(f"  WARNING run-{r}: {nan_ct} NaN, {inf_ct} Inf voxels - clipping to 0")
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, 0, None)  # resampling can produce tiny negative undershoot at edges; floor at 0 (non-physical values), same convention as the FMR/STC source data

        bv_data = ras_to_bv_internal_array(out)  # (DimZ,DimY,DimX,T)

        header, _ = bvbabel.vtc.create_vtc()
        header["File version"] = 3
        header["Source FMR name"] = os.path.basename(fmr_path)
        header["Protocol attached"] = 0
        header["Protocol name"] = ""
        header["Current protocol index"] = 0
        header["Data type (1:short int, 2:float)"] = 2 if DTYPE == "float32" else 1
        header["Nr time points"] = T
        header["VTC resolution relative to VMR (1, 2, or 3)"] = 1
        header["XStart"], header["XEnd"] = XStart, XEnd
        header["YStart"], header["YEnd"] = YStart, YEnd
        header["ZStart"], header["ZEnd"] = ZStart, ZEnd
        header["L-R convention (0:unknown, 1:radiological, 2:neurological)"] = 1
        header["Reference space (0:unknown, 1:native, 2:ACPC, 3:Tal, 4:MNI)"] = 1  # native, matches VMR ReferenceSpaceVMR=0
        header["TR (ms)"] = float(fh["TR"])

        write_data = bv_data if DTYPE == "float32" else np.clip(np.rint(bv_data), 0, 32767).astype(np.int16)
        out_path = f"{REG}/sub-01_run-{r}_preproc_coreg.vtc"
        bvbabel.vtc.write_vtc(out_path, header, write_data, rearrange_data_axes=False)

        elapsed = time.time() - t0
        vtc_report[f"run-{r}"] = {
            "out": out_path, "n_volumes": T, "bv_internal_shape_ZYX": list(bv_data.shape[:3]),
            "XStart": XStart, "XEnd": XEnd, "YStart": YStart, "YEnd": YEnd, "ZStart": ZStart, "ZEnd": ZEnd,
            "VTCResolution": 1, "TR_ms": float(fh["TR"]), "dtype": DTYPE,
            "nan_fixed": nan_ct, "inf_fixed": inf_ct, "elapsed_s": elapsed,
            "file_size_bytes": os.path.getsize(out_path),
        }
        print(f"run-{r}: wrote {out_path} ({T} vols, {elapsed:.0f}s, "
              f"{os.path.getsize(out_path)/1e9:.2f} GB)")

    report = {"bbox": bbox_meta, "vtc": vtc_report, "transform_used": shared_xfm, "dtype": DTYPE}
    with open(f"{LOG}/registration_04_make_vtc.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote logs/registration_04_make_vtc.json")


if __name__ == "__main__":
    main()
