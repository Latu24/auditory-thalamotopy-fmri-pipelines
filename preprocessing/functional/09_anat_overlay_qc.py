#!/usr/bin/env python3
"""EPI-to-anatomy sanity check for every run: compares uncorrected vs.
topup-corrected alignment to the MP2RAGE INV2 anatomical image using
NCC-based rigid registration, with a pass/warn/fail gate. Input: per-run AP
b0 and topup-corrected b0 volumes, plus the MP2RAGE INV2 NIfTI. Output: a
per-run/per-seed QC JSON with a documented threshold-based gate.

ants.registration's Rigid stage uses Mattes MI with an unseeded stochastic
20% voxel subsample by default (random_seed=None), so a single registration
call can give a noisy, even sign-flipped, "improvement" estimate for the
same pair of images. This script instead runs N_SEEDS independent
registrations per condition with fixed, documented seeds, and uses the
median NCC (robust to the optimizer landing in a bad local optimum on any
single draw) for the improvement metric; the cross-seed IQR is also
reported so an unresolved case remains visible rather than being hidden by
a single lucky or unlucky draw.

Gate: FAIL if median_improvement <= 0; WARN if 0 < median_improvement <
0.01, or either condition's cross-seed IQR is large (>0.05) relative to the
improvement itself (i.e. the "improvement" is not resolved above the
registration's own noise floor); PASS otherwise.
"""
import os, json, numpy as np, nibabel as nib, ants

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FUNC = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "func")
RAW = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "rawdata_nifti")
LOG = os.path.join(PROJECT_ROOT, "logs")
TMP = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "tmp", "anat_overlay_qc")
os.makedirs(TMP, exist_ok=True)
ANAT = f"{RAW}/sub-01_acq-mp2rage_INV2.nii.gz"
RUNS = [1, 2, 3, 4]
TAG = "v2"
WARN_THR = 0.01
NOISE_FLOOR_IQR = 0.05
SEEDS = [0, 1, 2, 3, 4]   # fixed, documented, reproducible

def save3d(arr, ref, out):
    nib.save(nib.Nifti1Image(arr.astype(np.float32), ref.affine, ref.header), out); return out

def ncc_reg(anat, mov, seed):
    reg = ants.registration(fixed=anat, moving=mov, type_of_transform="Rigid",
                            aff_metric="mattes", verbose=False, random_seed=seed)
    warped = ants.apply_transforms(fixed=anat, moving=mov, transformlist=reg["fwdtransforms"])
    a = anat.numpy(); w = warped.numpy(); m = w > 0
    aa = a[m]; ww = w[m]
    aa = (aa - aa.mean()) / (aa.std() + 1e-6); ww = (ww - ww.mean()) / (ww.std() + 1e-6)
    return float((aa * ww).mean())

def main():
    if not os.path.exists(ANAT):
        print("anat INV2 missing; skip"); return
    anat = ants.image_read(ANAT)

    res = {}
    for r in RUNS:
        refimg = nib.load(f"{FUNC}/sub-01_run-{r}_AP_b0-first5_cut.nii.gz")
        apb0 = np.asarray(refimg.dataobj, dtype=np.float32).mean(-1)
        uncorr = ants.image_read(save3d(apb0, refimg, f"{TMP}/ap{r}_uncorr.nii.gz"))
        iout = f"{FUNC}/sub-01_run-{r}_topup_bunwarped.nii.gz"
        corr_arr = np.asarray(nib.load(iout).dataobj, dtype=np.float32)[..., :5].mean(-1)
        corr = ants.image_read(save3d(corr_arr, refimg, f"{TMP}/ap{r}_topup.nii.gz"))

        vals = {"uncorrected": [], "topup_corrected": []}
        for seed in SEEDS:
            vals["uncorrected"].append(ncc_reg(anat, uncorr, seed))
            vals["topup_corrected"].append(ncc_reg(anat, corr, seed))
        run_res = {}
        for name in ("uncorrected", "topup_corrected"):
            arr = np.array(vals[name])
            run_res[name] = dict(
                seeds=SEEDS, values=arr.tolist(),
                median=float(np.median(arr)),
                iqr=float(np.subtract(*np.percentile(arr, [75, 25]))),
                min=float(arr.min()), max=float(arr.max()),
            )
        med_improve = run_res["topup_corrected"]["median"] - run_res["uncorrected"]["median"]
        run_res["median_improvement"] = med_improve
        noisy = (run_res["uncorrected"]["iqr"] > NOISE_FLOOR_IQR or
                 run_res["topup_corrected"]["iqr"] > NOISE_FLOOR_IQR)
        if med_improve <= 0:
            flag = "FAIL"
        elif med_improve < WARN_THR or noisy:
            flag = "WARN"
        else:
            flag = "PASS"
        run_res["qc_flag"] = flag
        run_res["qc_flag_reason"] = (
            f"median_improvement<=0 (n={len(SEEDS)} seeds): topup correction did not "
            "improve (or worsened) EPI-to-anatomy agreement, robust to registration noise"
            if flag == "FAIL" else
            (f"noisy: NCC IQR>{NOISE_FLOOR_IQR} across seeds -- registration itself is "
             "unstable for this run, improvement not resolved above noise floor"
             if noisy and med_improve > 0 else
             f"0<median_improvement<{WARN_THR}: marginal even after robust (median-of-{len(SEEDS)}) estimate")
            if flag == "WARN" else
            f"median_improvement>={WARN_THR}, stable across {len(SEEDS)} seeds (IQR<={NOISE_FLOOR_IQR})"
        )
        res[r] = run_res
        print(f"[run{r}] median NCC uncorr={run_res['uncorrected']['median']:.4f} "
              f"(IQR={run_res['uncorrected']['iqr']:.4f})  "
              f"topup={run_res['topup_corrected']['median']:.4f} "
              f"(IQR={run_res['topup_corrected']['iqr']:.4f})  "
              f"median_improvement={med_improve:+.4f}  flag={flag}")

    res["summary"] = dict(
        n_fail=sum(1 for r in RUNS if res[r]["qc_flag"] == "FAIL"),
        n_warn=sum(1 for r in RUNS if res[r]["qc_flag"] == "WARN"),
        n_pass=sum(1 for r in RUNS if res[r]["qc_flag"] == "PASS"),
        n_seeds=len(SEEDS), seeds=SEEDS,
        warn_threshold=WARN_THR, noise_floor_iqr=NOISE_FLOOR_IQR,
        gate_rule=("FAIL if median_improvement<=0; WARN if 0<median_improvement<0.01 OR "
                   "either condition's cross-seed IQR>0.05; PASS otherwise"),
        root_cause_note=(
            "A naive single-registration version of this QC is confounded by unseeded "
            "stochastic Mattes-MI sampling in ants.registration's Rigid stage "
            "(aff_random_sampling_rate=0.2, random_seed=None by default): repeated "
            "registrations of the identical run-1 images gave improvement values ranging "
            "from roughly +0.06 to -0.13 depending purely on the RNG seed. This script "
            "fixes that by using the median of 5 fixed, documented seeds per condition "
            "per run."
        ),
    )
    json.dump(res, open(f"{LOG}/qc_anat_overlay_{TAG}_robust.json", "w"), indent=2)
    print(f"wrote qc_anat_overlay_{TAG}_robust.json  summary={res['summary']}")

if __name__ == "__main__":
    main()
