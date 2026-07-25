#!/usr/bin/env python3
"""Functional-to-anatomical coregistration. Input: the per-run mean
functional NIfTI (from 01_meanfunc.py) and the N4-corrected anatomical
NIfTI the VMR was built from. Output: a shared rigid transform (.mat) and a
JSON report of registration scores per run.

Two-stage registration mirrors BrainVoyager's Initial Alignment (IA) ->
Fine-tuning Alignment (FA) workflow:
  - IA: coarse, low-DOF (Translation, center-of-mass init) to resolve the
    large FOV/orientation discrepancy between the 0.9mm EPI slab and the
    0.7mm whole-head anatomical.
  - FA: Rigid, Mattes Mutual-Information-driven, initialized from IA.
    Intensity-driven registration is used (no cortical surface/segmentation
    is available for boundary-based registration). Mattes MI (rather than
    plain correlation) is used because EPI (T2*) and the MP2RAGE UNI
    (T1-like) have substantially different tissue contrast.

Target: the anatomical N4-corrected UNI NIfTI the VMR was built from (same
array/affine/shape as the VMR's native RAS grid), so the resulting
transform maps functional voxel space directly onto the VMR frame with no
extra composition step needed for VTC creation.

Registration is kept rigid only (never affine/nonlinear) deliberately: an
affine/deformable fit could absorb residual distortion and mask a genuine
distortion problem. Systematic edge misalignment after the best possible
rigid fit is the distortion-residual signal this pipeline checks for
downstream.

Procedure:
  1. Register run-1 mean functional -> anat (IA then FA).
  2. Apply the SAME run-1-derived transform to runs 2-4 (valid because step
     00 confirmed all 4 runs share one common post-motion-correction
     grid/affine) and score each with MI + edge-gradient correlation.
  3. Also independently register run-4 from scratch (IA+FA) as a
     cross-check, since a between-run head repositioning of a few mm was
     observed before run 4 during motion QC. Compare the independent run-4
     transform to run-1's transform applied to run-4 - large disagreement
     would mean the shared-space assumption is wrong for run-4 and it needs
     its own transform.
"""
import os, json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import nibabel as nib
import ants
from scipy.ndimage import sobel

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ANAT_NII = f"{PROJECT_ROOT}/derivatives/sub-01/anat/sub-01_desc-UNIdenoisedN4.nii.gz"
REG = f"{PROJECT_ROOT}/derivatives/sub-01/reg"
LOG = f"{PROJECT_ROOT}/logs"
os.makedirs(REG, exist_ok=True)


def edge_gradient_corr(fixed_np, warped_np, mask):
    """Gradient-magnitude (Sobel) correlation in the overlap mask - proxy for
    boundary/edge alignment. Systematic edge misalignment (low correlation)
    after a good rigid fit = leftover distortion, not just misregistration."""
    def gmag(a):
        gx = sobel(a, axis=0); gy = sobel(a, axis=1); gz = sobel(a, axis=2)
        return np.sqrt(gx**2 + gy**2 + gz**2)
    gf = gmag(fixed_np)[mask]
    gw = gmag(warped_np)[mask]
    gf = (gf - gf.mean()) / (gf.std() + 1e-6)
    gw = (gw - gw.mean()) / (gw.std() + 1e-6)
    return float((gf * gw).mean())


def mattes_mi(fixed, warped):
    return float(ants.image_similarity(fixed, warped, metric_type="MattesMutualInformation"))


def register_ia_fa(fixed, moving, label):
    t0 = time.time()
    # --- IA: coarse Translation, center-of-mass initialization ---
    ia = ants.registration(fixed=fixed, moving=moving, type_of_transform="Translation",
                            aff_metric="mattes", verbose=False)
    t_ia = time.time() - t0
    # --- FA: Rigid, Mattes MI, initialized from IA ---
    t1 = time.time()
    fa = ants.registration(fixed=fixed, moving=moving, type_of_transform="Rigid",
                            initial_transform=ia["fwdtransforms"][0],
                            aff_metric="mattes", verbose=False)
    t_fa = time.time() - t1
    warped = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=fa["fwdtransforms"],
                                    interpolator="linear")
    print(f"  [{label}] IA {t_ia:.1f}s, FA {t_fa:.1f}s")
    return fa, warped


def score(fixed, moving_raw, warped):
    f = fixed.numpy(); w = warped.numpy()
    mask = w > np.percentile(w[w > 0], 1) if (w > 0).any() else (w != 0)
    mi_post = mattes_mi(fixed, warped)
    identity_warp = ants.apply_transforms(fixed=fixed, moving=moving_raw, transformlist=[],
                                           interpolator="linear")
    mi_pre = mattes_mi(fixed, identity_warp)
    edge_corr = edge_gradient_corr(f, w, mask)
    return {"mi_pre_identity": mi_pre, "mi_post_rigid": mi_post, "edge_grad_corr": edge_corr}


def main():
    anat = ants.image_read(ANAT_NII)
    report = {}

    # --- run-1: full IA+FA ---
    mov1 = ants.image_read(f"{REG}/sub-01_run-1_meanfunc_preproc.nii.gz")
    fa1, warped1 = register_ia_fa(anat, mov1, "run-1 (reference)")
    s1 = score(anat, mov1, warped1)
    report["run-1"] = {"mode": "independent IA+FA", **s1}
    print(f"run-1: {s1}")

    # save the shared transform
    xfm_dir = f"{REG}/xfm"
    os.makedirs(xfm_dir, exist_ok=True)
    shared_fwd = fa1["fwdtransforms"]
    shared_xfm_path = f"{xfm_dir}/sub-01_run-1_to_anat_rigid.mat"
    import shutil
    for p in shared_fwd:
        if p.endswith(".mat"):
            shutil.copy(p, shared_xfm_path)
    anat_nib = nib.load(ANAT_NII)
    nib.save(nib.Nifti1Image(warped1.numpy().astype(np.float32), anat_nib.affine, anat_nib.header),
             f"{REG}/qc/sub-01_run-1_meanfunc_warped_to_anat.nii.gz")

    # --- apply run-1's transform to runs 2,3,4 (shared common space) ---
    for r in [2, 3, 4]:
        mov = ants.image_read(f"{REG}/sub-01_run-{r}_meanfunc_preproc.nii.gz")
        warped = ants.apply_transforms(fixed=anat, moving=mov, transformlist=shared_fwd, interpolator="linear")
        s = score(anat, mov, warped)
        report[f"run-{r}_shared_transform"] = {"mode": "run-1 transform applied", **s}
        print(f"run-{r} (shared transform from run-1): {s}")
        nib.save(nib.Nifti1Image(warped.numpy().astype(np.float32), nib.load(ANAT_NII).affine, nib.load(ANAT_NII).header),
                 f"{REG}/qc/sub-01_run-{r}_meanfunc_warped_to_anat_sharedxfm.nii.gz")

    # --- independent IA+FA for run-4 as a cross-check (flagged between-run repositioning) ---
    mov4 = ants.image_read(f"{REG}/sub-01_run-4_meanfunc_preproc.nii.gz")
    fa4, warped4 = register_ia_fa(anat, mov4, "run-4 (independent cross-check)")
    s4 = score(anat, mov4, warped4)
    report["run-4_independent"] = {"mode": "independent IA+FA", **s4}
    print(f"run-4 (independent): {s4}")

    # compare transform parameters: read both .mat affine transforms and compare
    t_shared = ants.read_transform(shared_xfm_path)
    t4_ind_path = [p for p in fa4["fwdtransforms"] if p.endswith(".mat")][0]
    t_ind = ants.read_transform(t4_ind_path)
    p_shared = np.array(t_shared.parameters).reshape(-1)
    p_ind = np.array(t_ind.parameters).reshape(-1)
    # parameters = [3x3 rotation matrix (9), 3 translation] for AntsTransform (rigid/affine "MatrixOffsetTransformBase")
    rot_shared, trans_shared = p_shared[:9].reshape(3, 3), p_shared[9:12]
    rot_ind, trans_ind = p_ind[:9].reshape(3, 3), p_ind[9:12]
    trans_delta_mm = float(np.linalg.norm(trans_shared - trans_ind))
    # rotation delta via angle of rot_shared^-1 . rot_ind
    R_delta = np.linalg.inv(rot_shared) @ rot_ind
    cos_ang = np.clip((np.trace(R_delta) - 1) / 2, -1, 1)
    rot_delta_deg = float(np.degrees(np.arccos(cos_ang)))
    report["run4_transform_comparison"] = {
        "translation_delta_mm": trans_delta_mm, "rotation_delta_deg": rot_delta_deg,
        "interpretation": ("shared run1-derived transform and run4's own independent "
                            "IA+FA registration agree closely (small delta) - confirms "
                            "motion correction already resolved run4's between-run "
                            "repositioning into the common space, single shared "
                            "coregistration transform is valid for all 4 runs"
                            if trans_delta_mm < 1.0 and rot_delta_deg < 1.0 else
                            "shared transform and run4's independent fit disagree "
                            "non-trivially - run4 needs its OWN coregistration transform, "
                            "not run1's, for VTC creation"),
    }
    print("run4 transform comparison:", report["run4_transform_comparison"])

    # decide: shared vs per-run transform based on comparison
    use_shared = trans_delta_mm < 1.0 and rot_delta_deg < 1.0
    report["decision"] = "single shared run-1-derived transform used for all 4 runs" if use_shared else \
                          "per-run independent transforms used (run4 differed too much)"

    with open(f"{LOG}/registration_02_coregister.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nDecision:", report["decision"])
    print("wrote logs/registration_02_coregister.json")

    # persist decision + which transform files to use, for step 04
    decision = {
        "use_shared_transform": use_shared,
        "shared_transform_mat": shared_xfm_path,
        "run4_independent_transform_mat": t4_ind_path if not use_shared else None,
    }
    with open(f"{REG}/xfm/decision.json", "w") as f:
        json.dump(decision, f, indent=2)


if __name__ == "__main__":
    main()
