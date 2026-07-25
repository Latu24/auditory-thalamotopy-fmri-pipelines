#!/usr/bin/env python3
"""Distortion-correction QC: compares AP/PA b0 agreement (via NCC)
before correction, after FSL topup, and after ANTs SyN correction, for every
run, and produces a per-run comparison figure. Input: raw AP/PA b0 volumes,
topup outputs (field + unwarped b0), and the ANTs distortion QC summary.
Output: a JSON QC report (including a per-run topup-vs-ANTs winner) and
qc_run-*_distortion.png comparison figures.
"""
import os, json, numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FUNC = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "func")
LOG = os.path.join(PROJECT_ROOT, "logs")
RT = 0.0729774; VOX = 0.9
RUNS = [1, 2, 3, 4]
TAG = "v2"

def ncc(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    a = (a - a.mean()) / (a.std() + 1e-6); b = (b - b.mean()) / (b.std() + 1e-6)
    return float((a * b).mean())

def load(p): return np.asarray(nib.load(p).dataobj, dtype=np.float32)

def main():
    out = {}
    ants_qc = json.load(open(f"{LOG}/qc_ants_distortion_{TAG}.json")) if os.path.exists(f"{LOG}/qc_ants_distortion_{TAG}.json") else {}
    for r in RUNS:
        rr = {}
        apb0 = load(f"{FUNC}/sub-01_run-{r}_AP_b0-first5_cut.nii.gz").mean(-1)
        pab0 = load(f"{FUNC}/sub-01_dir-PA_b0-first5_cut.nii.gz").mean(-1)
        rr["ncc_AP_PA_uncorrected"] = ncc(apb0, pab0)

        fld_p = f"{FUNC}/sub-01_run-{r}_topup_field.nii.gz"
        if os.path.exists(fld_p):
            fld = load(fld_p)
            disp_vox = fld * RT
            disp_mm = disp_vox * VOX
            rr["topup_field_Hz"] = dict(min=float(fld.min()), max=float(fld.max()),
                                        p1=float(np.percentile(fld,1)), p99=float(np.percentile(fld,99)))
            rr["topup_disp_mm"] = dict(max_abs=float(np.abs(disp_mm).max()),
                                       p99_abs=float(np.percentile(np.abs(disp_mm),99)))
            rr["overcorrection_flag"] = bool(np.percentile(np.abs(disp_mm),99) > 15)
        iout_p = f"{FUNC}/sub-01_run-{r}_topup_bunwarped.nii.gz"
        if os.path.exists(iout_p):
            iout = load(iout_p)
            ap_c = iout[..., :5].mean(-1); pa_c = iout[..., 5:].mean(-1)
            rr["ncc_AP_PA_topup"] = ncc(ap_c, pa_c)

        # ANTs, all 4 runs
        if str(r) in ants_qc:
            rr["ncc_AP_PA_ants"] = ants_qc[str(r)]["antscorr_AP_PA_ncc"]

        # winner determination
        if "ncc_AP_PA_topup" in rr and "ncc_AP_PA_ants" in rr:
            rr["winner"] = "topup" if rr["ncc_AP_PA_topup"] >= rr["ncc_AP_PA_ants"] else "ants"
            rr["topup_minus_ants_ncc"] = rr["ncc_AP_PA_topup"] - rr["ncc_AP_PA_ants"]

        out[r] = rr
        print(f"[run{r}] NCC(AP,PA) uncorr={rr['ncc_AP_PA_uncorrected']:.4f} "
              f"topup={rr.get('ncc_AP_PA_topup','-')} ants={rr.get('ncc_AP_PA_ants','-')} "
              f"winner={rr.get('winner','-')} "
              f"maxDisp={rr.get('topup_disp_mm',{}).get('max_abs','-')}mm "
              f"overcorr={rr.get('overcorrection_flag','-')}")

        # comparison figure (mid slice) for all 4 runs
        z = apb0.shape[2] // 2
        panels = [("uncorr AP-b0", apb0[..., z]), ("PA-b0", pab0[..., z])]
        if os.path.exists(iout_p):
            panels.append(("topup AP-b0", load(iout_p)[..., :5].mean(-1)[..., z]))
        ants_p = f"{FUNC}/sub-01_run-{r}_{TAG}_ants_halfwarp.nii.gz"
        ap_corr_p = None
        # use the corrected mean b0 via the same NCC pipeline output if present
        if os.path.exists(f"{FUNC}/sub-01_run-{r}_{TAG}_stage-06dc-ants.nii.gz"):
            panels.append(("ANTs AP (vol0)", load(f"{FUNC}/sub-01_run-{r}_{TAG}_stage-06dc-ants.nii.gz")[..., z, 0]))
        fig, ax = plt.subplots(1, len(panels), figsize=(3.2*len(panels), 3.4))
        for a, (t, im) in zip(np.atleast_1d(ax), panels):
            a.imshow(np.rot90(im), cmap="gray"); a.set_title(t, fontsize=8); a.axis("off")
        fig.suptitle(f"run{r} distortion correction (mid-axial slice)")
        fig.tight_layout(); fig.savefig(f"{LOG}/qc_run-{r}_{TAG}_distortion.png", dpi=95)
        plt.close(fig)
    json.dump(out, open(f"{LOG}/qc_distortion_{TAG}.json", "w"), indent=2)
    print(f"wrote qc_distortion_{TAG}.json + qc_run-*_{TAG}_distortion.png")

if __name__ == "__main__":
    main()
