#!/usr/bin/env python3
"""ANTs (antspyx) susceptibility-distortion correction, evaluated as an
alternative to FSL topup for every run. Input: per-run AP/PA b0 volumes
(first-5, cut) and the high-pass-filtered BOLD NIfTI. Output: a half-warp
displacement field, a distortion-corrected NIfTI per run, and an NCC-based
QC summary comparing AP/PA agreement before and after correction.

Blip-up/blip-down midpoint scheme: fixed = mean PA b0 (blip +j), moving =
mean AP b0 (blip -j). SyN(AP->PA) estimates a displacement field dominated
by the phase-encode axis, where AP and PA disagree. The undistorted image
is approximately the midpoint between the two, so the warp is scaled by 0.5
before being applied to the AP data.
"""
import os, json, numpy as np, nibabel as nib, ants

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FUNC = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "func")
LOG = os.path.join(PROJECT_ROOT, "logs")
TMP = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "tmp", "ants_distortion")
os.makedirs(TMP, exist_ok=True)
RUNS = [1, 2, 3, 4]
TAG = "v2"

def mean3d_file(path, out):
    img = nib.load(path); d = np.asarray(img.dataobj, dtype=np.float32)
    m = d.mean(-1) if d.ndim == 4 else d
    nib.save(nib.Nifti1Image(m, img.affine, img.header), out)
    return out

def ncc(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    a = (a - a.mean()) / (a.std() + 1e-6); b = (b - b.mean()) / (b.std() + 1e-6)
    return float((a * b).mean())

def main():
    pa_img = ants.image_read(mean3d_file(f"{FUNC}/sub-01_dir-PA_b0-first5_cut.nii.gz",
                                         f"{TMP}/pa_mean.nii.gz"))
    summary = {}
    for r in RUNS:
        ap_img = ants.image_read(mean3d_file(f"{FUNC}/sub-01_run-{r}_AP_b0-first5_cut.nii.gz",
                                             f"{TMP}/ap{r}_mean.nii.gz"))
        print(f"[run{r}] SyN AP->PA ...", flush=True)
        reg = ants.registration(fixed=pa_img, moving=ap_img, type_of_transform="SyNOnly",
                                 reg_iterations=(60, 40, 20), flow_sigma=3, total_sigma=0.5,
                                 verbose=False)
        warp_path = [t for t in reg["fwdtransforms"] if t.endswith("Warp.nii.gz")
                     and "Inverse" not in t][0]
        w = nib.load(warp_path)
        half = nib.Nifti1Image(np.asarray(w.dataobj, dtype=np.float32) * 0.5, w.affine, w.header)
        half_path = f"{FUNC}/sub-01_run-{r}_{TAG}_ants_halfwarp.nii.gz"
        nib.save(half, half_path)

        ap_corr = ants.apply_transforms(fixed=pa_img, moving=ap_img,
                                        transformlist=[half_path], interpolator="linear")
        raw = ncc(ap_img.numpy(), pa_img.numpy())
        cor = ncc(ap_corr.numpy(), pa_img.numpy())
        summary[r] = dict(raw_AP_PA_ncc=raw, antscorr_AP_PA_ncc=cor)
        print(f"[run{r}] NCC(AP,PA) raw={raw:.4f} -> ants-corrected={cor:.4f}", flush=True)

        # apply half-warp to the full processed run (STC + motion-correction + HPF)
        src = nib.load(f"{FUNC}/sub-01_run-{r}_{TAG}_stage-05hpf.nii.gz")
        sdata = np.asarray(src.dataobj, dtype=np.float32)
        T = sdata.shape[-1]; out = np.empty_like(sdata)
        for t in range(T):
            vol = pa_img.new_image_like(sdata[..., t].astype(np.float32))
            wv = ants.apply_transforms(fixed=pa_img, moving=vol,
                                       transformlist=[half_path], interpolator="linear")
            out[..., t] = wv.numpy()
        out_path = f"{FUNC}/sub-01_run-{r}_{TAG}_stage-06dc-ants.nii.gz"
        nib.save(nib.Nifti1Image(out.astype(np.float32), src.affine, src.header), out_path)
        print(f"[run{r}] wrote {out_path}  T={T}", flush=True)
    json.dump(summary, open(f"{LOG}/qc_ants_distortion_{TAG}.json", "w"), indent=2)
    print(f"wrote qc_ants_distortion_{TAG}.json")

if __name__ == "__main__":
    main()
