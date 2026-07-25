#!/usr/bin/env python3
"""Quantify real head motion between the AP-b0 topup input and the PA-b0
topup input, independently of topup's own internal movement model, via
direct rigid registration on the raw (uncorrected) b0 volumes topup itself
was given. Input: per-run AP b0 and PA b0 (first-5, cut) volumes and
topup's own movpar.txt outputs. Output: a JSON report comparing the
independently measured AP/PA offset against topup's internal estimate.

FSL topup's documentation notes that its internal "movement model"
separates real head motion between the AP/PA acquisitions from the true
susceptibility field, and that if this attribution fails "all differences
would be attributed to the off-resonance field, with potentially
disastrous results." topup's own movpar.txt output shows a consistent
sub-1.6mm AP-to-PA offset across all 4 runs, concentrated in one
translation axis; this script independently verifies that offset via
direct image registration (rather than trusting topup's own internal
attribution), to determine whether it reflects real, substantial head
motion.

Uses the same validated IA(Translation)+FA(Rigid,MattesMI) pipeline used
elsewhere in this registration stage, on the raw (pre-topup) mean AP-b0 vs
mean PA-b0 images -- a same-contrast, same-modality, well-conditioned
registration problem (unlike the cross-modality func->anat case), so no
warm-start/convergence workarounds are expected to be needed, but a
convergence sanity check (post MI more negative than pre-identity MI) is
still verified per run as a safety net.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import nibabel as nib
import ants

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FUNC = f"{PROJECT_ROOT}/derivatives/sub-01/func"
REG = f"{PROJECT_ROOT}/derivatives/sub-01/reg"
LOG = f"{PROJECT_ROOT}/logs"
TMP = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "tmp", "ap_pa_motion_check")
os.makedirs(TMP, exist_ok=True)


def mi(fixed, moving):
    return float(ants.image_similarity(fixed, moving, metric_type="MattesMutualInformation"))


def mean_vol_ants(path):
    img = nib.load(path)
    arr = np.asarray(img.dataobj, dtype=np.float32).mean(axis=-1)
    tmp = os.path.join(TMP, f"_meanb0_{os.path.basename(path)}")
    nib.save(nib.Nifti1Image(arr, img.affine, img.header), tmp)
    return ants.image_read(tmp)


def main():
    pa = mean_vol_ants(f"{FUNC}/sub-01_dir-PA_b0-first5_cut.nii.gz")
    report = {}

    for r in [1, 2, 3, 4]:
        ap = mean_vol_ants(f"{FUNC}/sub-01_run-{r}_AP_b0-first5_cut.nii.gz")

        # pre-registration (identity / native scanner-space alignment) baseline
        identity_warp = ants.apply_transforms(fixed=pa, moving=ap, transformlist=[], interpolator="linear")
        mi_pre = mi(pa, identity_warp)

        # IA (coarse translation) -> FA (rigid, Mattes MI)
        ia = ants.registration(fixed=pa, moving=ap, type_of_transform="Translation", aff_metric="mattes", verbose=False)
        fa = ants.registration(fixed=pa, moving=ap, type_of_transform="Rigid",
                                initial_transform=ia["fwdtransforms"][0], aff_metric="mattes", verbose=False)
        warped = ants.apply_transforms(fixed=pa, moving=ap, transformlist=fa["fwdtransforms"], interpolator="linear")
        mi_post = mi(pa, warped)
        converged = mi_post < mi_pre

        # extract rigid transform parameters (rotation matrix + translation, mm/rad)
        mat_path = [p for p in fa["fwdtransforms"] if p.endswith(".mat")][0]
        t = ants.read_transform(mat_path)
        params = np.array(t.parameters).reshape(-1)
        rot_mat, trans = params[:9].reshape(3, 3), params[9:12]
        trans_mag = float(np.linalg.norm(trans))
        # rotation angle from rotation matrix trace
        cos_ang = np.clip((np.trace(rot_mat) - 1) / 2, -1, 1)
        rot_deg = float(np.degrees(np.arccos(cos_ang)))

        report[f"run-{r}"] = {
            "mi_pre_identity": mi_pre, "mi_post_rigid": mi_post, "converged": bool(converged),
            "translation_vector_mm": trans.tolist(), "translation_magnitude_mm": trans_mag,
            "rotation_deg": rot_deg,
        }
        print(f"run-{r}: MI pre={mi_pre:.4f} post={mi_post:.4f} converged={converged}  "
              f"|translation|={trans_mag:.3f}mm  rotation={rot_deg:.3f}deg  "
              f"trans_vec={np.round(trans,3)}")

    # compare against topup's own movpar-derived AP/PA offset
    print("\n=== comparison: independent direct registration vs topup's own movpar-derived offset ===")
    for r in [1, 2, 3, 4]:
        mp = np.loadtxt(f"{FUNC}/sub-01_run-{r}_topup_movpar.txt")
        ap_mean = mp[:5].mean(axis=0); pa_mean = mp[5:].mean(axis=0)
        diff = pa_mean - ap_mean
        topup_trans_mag = float(np.linalg.norm(diff[:3]))
        topup_rot_deg = float(np.degrees(np.linalg.norm(diff[3:])))
        indep_trans_mag = report[f"run-{r}"]["translation_magnitude_mm"]
        indep_rot_deg = report[f"run-{r}"]["rotation_deg"]
        report[f"run-{r}"]["topup_movpar_trans_mag_mm"] = topup_trans_mag
        report[f"run-{r}"]["topup_movpar_rot_deg"] = topup_rot_deg
        print(f"run-{r}: independent |trans|={indep_trans_mag:.3f}mm rot={indep_rot_deg:.3f}deg   "
              f"vs topup-internal |trans|={topup_trans_mag:.3f}mm rot={topup_rot_deg:.3f}deg")

    with open(f"{LOG}/registration_06_lead1_ap_pa_motion_check.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote logs/registration_06_lead1_ap_pa_motion_check.json")


if __name__ == "__main__":
    main()
