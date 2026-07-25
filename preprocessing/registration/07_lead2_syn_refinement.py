#!/usr/bin/env python3
"""Nonlinear (ANTs SyN) refinement applied after topup + rigid
coregistration, to address the residual distortion/motion-conflation
confirmed real by 06_lead1_ap_pa_motion_check.py: every run shows roughly
1.6-2.3mm of genuine AP/PA motion, of which topup's internal movement model
captured only a fraction. Input: per-run mean functional NIfTI and the
shared rigid func->anat transform. Output: a per-run SyN transform (saved
under reg/xfm_lead2_refined/) and a JSON report of MI improvement and warp
displacement statistics.

FSL's own documentation warns against rigidly registering the raw AP/PA
pair before topup ("the two images will be very different, and the
registration would not perform well"). Instead, SyN is applied after
topup + rigid coregistration, on the already-processed data, refining the
existing validated rigid func->anat transform.

A separate SyN refinement is computed per run (rather than a single shared
one): the residual this step targets originates from each run's own AP/PA
motion during that run's own topup field estimation (topup was run
separately per run), and the real AP/PA motion magnitude differs
substantially by run. A shared correction would not be appropriate for a
per-run-varying residual.

Method: ants.registration(type_of_transform="SyNOnly",
initial_transform=<the validated shared rigid .mat>) keeps the already-
validated rigid alignment as the starting point and estimates only the
additional deformable warp needed on top of it, using ANTsPy's default
(conservative, bending-energy-like) SyN regularization.

Convergence + plausibility checks per run:
  - MI: post-SyN must be more negative (better) than post-rigid-only, the
    same convergence sanity check used throughout this registration stage.
  - Warp-field plausibility: the max/99th-percentile displacement magnitude
    of the estimated SyN field is reported and compared against the real
    motion magnitude found in 06_lead1_ap_pa_motion_check.py (roughly
    1.6-2.3mm) -- a warp of similar order is expected and desired; a warp
    many times larger would indicate overfitting/noise absorption rather
    than correcting the specific known residual.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import ants

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ANAT_NII = f"{PROJECT_ROOT}/derivatives/sub-01/anat/sub-01_desc-UNIdenoisedN4.nii.gz"
REG = f"{PROJECT_ROOT}/derivatives/sub-01/reg"
LOG = f"{PROJECT_ROOT}/logs"
SHARED_RIGID_XFM = f"{REG}/xfm/sub-01_run-1_to_anat_rigid.mat"


def mi(fixed, moving):
    return float(ants.image_similarity(fixed, moving, metric_type="MattesMutualInformation"))


def main():
    anat = ants.image_read(ANAT_NII)
    report = {}
    xfm_dir = f"{REG}/xfm_lead2_refined"
    os.makedirs(xfm_dir, exist_ok=True)

    for r in [1, 2, 3, 4]:
        mov = ants.image_read(f"{REG}/sub-01_run-{r}_meanfunc_preproc.nii.gz")

        # baseline: rigid-only (the existing validated main-pipeline transform)
        warped_rigid = ants.apply_transforms(fixed=anat, moving=mov, transformlist=[SHARED_RIGID_XFM], interpolator="linear")
        mi_rigid = mi(anat, warped_rigid)

        # SyN refinement, initialized from the validated rigid transform
        syn = ants.registration(fixed=anat, moving=mov, type_of_transform="SyNOnly",
                                 initial_transform=SHARED_RIGID_XFM, aff_metric="mattes",
                                 syn_metric="mattes", verbose=False)
        warped_syn = ants.apply_transforms(fixed=anat, moving=mov, transformlist=syn["fwdtransforms"], interpolator="linear")
        mi_syn = mi(anat, warped_syn)
        converged = mi_syn < mi_rigid  # more negative = better

        # persist the per-run refined transform list (warp + affine), copy into a stable dir
        import shutil
        saved_transforms = []
        for i, p in enumerate(syn["fwdtransforms"]):
            ext = ".nii.gz" if p.endswith(".nii.gz") else ".mat"
            dst = f"{xfm_dir}/sub-01_run-{r}_syn_fwd_{i}{ext}"
            shutil.copy(p, dst)
            saved_transforms.append(dst)

        # warp-field displacement magnitude plausibility check
        warp_files = [p for p in syn["fwdtransforms"] if p.endswith(".nii.gz")]
        disp_stats = {}
        if warp_files:
            warp_img = ants.image_read(warp_files[0])
            warp_np = warp_img.numpy()  # shape (...,3) displacement vector field, mm
            disp_mag = np.linalg.norm(warp_np, axis=-1)
            disp_stats = {
                "max_mm": float(disp_mag.max()), "p99_mm": float(np.percentile(disp_mag, 99)),
                "mean_mm": float(disp_mag.mean()),
            }

        report[f"run-{r}"] = {
            "mi_rigid_only": mi_rigid, "mi_syn_refined": mi_syn, "converged": bool(converged),
            "delta_mi_syn_minus_rigid": mi_syn - mi_rigid,
            "warp_displacement_stats_mm": disp_stats,
            "saved_transforms": saved_transforms,
        }
        print(f"run-{r}: MI rigid-only={mi_rigid:.5f}  MI SyN-refined={mi_syn:.5f}  "
              f"converged={converged}  delta={mi_syn-mi_rigid:+.5f}")
        print(f"  warp displacement: max={disp_stats.get('max_mm',0):.2f}mm  "
              f"p99={disp_stats.get('p99_mm',0):.2f}mm  mean={disp_stats.get('mean_mm',0):.2f}mm")
        print(f"  (compare to the independently-measured real AP/PA motion for this run "
              f"in registration_06_lead1_ap_pa_motion_check.json)")

    with open(f"{LOG}/registration_07_lead2_syn_refinement.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote logs/registration_07_lead2_syn_refinement.json")


if __name__ == "__main__":
    main()
