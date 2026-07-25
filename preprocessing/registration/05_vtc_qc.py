#!/usr/bin/env python3
"""VTC QC: verify dimensions/resolution are consistent across all 4 runs,
round-trip read via bvbabel, check for NaN/Inf, sanity-check the intensity
range, and save a montage overlay of the VTC coverage on the VMR. Input:
per-run VTC files and the VMR. Output: a QC JSON and an overlay PNG.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import bvbabel.vtc, bvbabel.vmr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
REG = f"{PROJECT_ROOT}/derivatives/sub-01/reg"
ANAT = f"{PROJECT_ROOT}/derivatives/sub-01/anat"
LOG = f"{PROJECT_ROOT}/logs"

RUNS = [1, 2, 3, 4]
expected_T = {1: 350, 2: 340, 3: 340, 4: 340}


def main():
    report = {}
    dims = []
    for r in RUNS:
        p = f"{REG}/sub-01_run-{r}_preproc_coreg.vtc"
        h, data = bvbabel.vtc.read_vtc(p, rearrange_data_axes=False)
        shape = data.shape  # (DimZ,DimY,DimX,T)
        box = (h["XStart"], h["XEnd"], h["YStart"], h["YEnd"], h["ZStart"], h["ZEnd"], h["VTC resolution relative to VMR (1, 2, or 3)"])
        nan_ct = int(np.isnan(data).sum()); inf_ct = int(np.isinf(data).sum())
        info = {
            "shape_ZYX_T": list(shape), "box": box, "T": h["Nr time points"],
            "TR_ms": h["TR (ms)"], "dtype_code": h["Data type (1:short int, 2:float)"],
            "nan_count": nan_ct, "inf_count": inf_ct,
            "min": float(data.min()), "max": float(data.max()), "mean": float(data.mean()),
            "reference_space": h["Reference space (0:unknown, 1:native, 2:ACPC, 3:Tal, 4:MNI)"],
        }
        report[f"run-{r}"] = info
        dims.append((shape[0], shape[1], shape[2], box[6]))
        print(f"run-{r}: {info}")
        assert h["Nr time points"] == expected_T[r], f"run-{r} VTC volume count mismatch"
        assert nan_ct == 0 and inf_ct == 0, f"run-{r} VTC has NaN/Inf"

    all_consistent = len(set(dims)) == 1
    report["dims_consistent_across_runs"] = all_consistent
    report["unique_dims_found"] = list(set(dims))
    print("\nSpatial dims + VTCResolution consistent across all 4 runs:", all_consistent)
    assert all_consistent, "VTC spatial dimensions/resolution differ across runs!"

    # quick visual: overlay run-1 VTC mean on VMR at the matching internal box
    hv, vmr = bvbabel.vmr.read_vmr(f"{ANAT}/sub-01_desc-UNIdenoisedN4_anat.vmr")
    h1, d1 = bvbabel.vtc.read_vtc(f"{REG}/sub-01_run-1_preproc_coreg.vtc", rearrange_data_axes=False)
    mean_vtc = d1.mean(axis=-1)  # (DimZ,DimY,DimX)
    Zc = (h1["ZStart"] + h1["ZEnd"]) // 2
    z_local = Zc - h1["ZStart"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(vmr[Zc, :, :], cmap="gray", origin="lower")
    axes[0].set_title(f"VMR slice Z={Zc}")
    axes[1].imshow(vmr[Zc, h1["YStart"]:h1["YEnd"], h1["XStart"]:h1["XEnd"]], cmap="gray", origin="lower")
    axes[1].imshow(mean_vtc[z_local, :, :], cmap="hot", alpha=0.5, origin="lower")
    axes[1].set_title("VMR + mean run-1 VTC overlay (same Z slice)")
    os.makedirs(f"{REG}/qc", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{REG}/qc/vtc_vmr_overlay_run1.png", dpi=120)
    print(f"wrote {REG}/qc/vtc_vmr_overlay_run1.png")

    with open(f"{LOG}/registration_05_vtc_qc.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nAll VTC QC checks passed. wrote logs/registration_05_vtc_qc.json")


if __name__ == "__main__":
    main()
