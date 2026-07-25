#!/usr/bin/env python3
"""Visual QC for the distortion-residual comparison: side-by-side anat /
uncorrected-warped / topup-corrected-warped mid-slices, edge overlays, and a
difference map, for all 4 runs, using the exact same best-of
(shared-transform, warmstart-refined) selection logic as
03b_distortion_residual_recheck_warmstart.py, so what is plotted here is
provably the same warped image the MI/edge-correlation numbers were
computed from. Output: three PNG panels per run under
derivatives/sub-01/reg/qc/.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import ants
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ANAT_NII = f"{PROJECT_ROOT}/derivatives/sub-01/anat/sub-01_desc-UNIdenoisedN4.nii.gz"
REG = f"{PROJECT_ROOT}/derivatives/sub-01/reg"
SHARED_XFM = f"{REG}/xfm/sub-01_run-1_to_anat_rigid.mat"


def gmag(a):
    return np.sqrt(sobel(a, axis=0)**2 + sobel(a, axis=1)**2 + sobel(a, axis=2)**2)


def mi(fixed, moving):
    return float(ants.image_similarity(fixed, moving, metric_type="MattesMutualInformation"))


def best_warp(fixed, moving_path):
    """Best-of (shared-transform-only, warmstart-refined) - matches the
    selection logic in 03b_distortion_residual_recheck_warmstart.py exactly."""
    mov = ants.image_read(moving_path)
    warped_shared = ants.apply_transforms(fixed=fixed, moving=mov, transformlist=[SHARED_XFM], interpolator="linear")
    mi_shared = mi(fixed, warped_shared)
    fa = ants.registration(fixed=fixed, moving=mov, type_of_transform="Rigid",
                            initial_transform=SHARED_XFM, aff_metric="mattes", verbose=False)
    warped_refined = ants.apply_transforms(fixed=fixed, moving=mov, transformlist=fa["fwdtransforms"], interpolator="linear")
    mi_refined = mi(fixed, warped_refined)
    if mi_refined <= mi_shared:
        return warped_refined.numpy(), mi_refined, "warmstart_refined"
    return warped_shared.numpy(), mi_shared, "shared_transform_only"


def main():
    anat = ants.image_read(ANAT_NII)
    anat_np = anat.numpy()
    anat_edge = gmag(anat_np)

    for run in [1, 2, 3, 4]:
        unc_path = f"{REG}/sub-01_run-{run}_meanb0_uncorrected.nii.gz"
        cor_path = f"{REG}/sub-01_run-{run}_meanb0_topupcorrected.nii.gz"
        w_unc, mi_unc, which_unc = best_warp(anat, unc_path)
        w_cor, mi_cor, which_cor = best_warp(anat, cor_path)
        print(f"run-{run}: uncorrected MI={mi_unc:.5f} ({which_unc})  topup MI={mi_cor:.5f} ({which_cor})")

        nz = w_cor > np.percentile(w_cor[w_cor > 0], 1)
        idx = np.where(nz)
        zc = int(np.median(idx[0]))  # mid R-axis (sagittal) slice through the coverage
        yc = int(np.median(idx[2]))  # mid S-axis slice through the coverage

        # --- panel 1: anat / uncorrected / topup side by side, two slice planes ---
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        imgs = [("anat", anat_np), (f"uncorrected  MI={mi_unc:.4f}", w_unc), (f"topup-corrected  MI={mi_cor:.4f}", w_cor)]
        for col, (title, img) in enumerate(imgs):
            axes[0, col].imshow(img[zc, :, :].T, cmap="gray", origin="lower")
            axes[0, col].set_title(f"{title}\nR-slice={zc}")
            axes[0, col].axis("off")
            axes[1, col].imshow(img[:, :, yc].T, cmap="gray", origin="lower")
            axes[1, col].set_title(f"{title}\nS-slice={yc}")
            axes[1, col].axis("off")
        plt.suptitle(f"run-{run}: anat vs uncorrected-warped vs topup-corrected-warped (best rigid fit)")
        plt.tight_layout()
        out = f"{REG}/qc/distortion_visual_run{run}_anat_unc_topup.png"
        plt.savefig(out, dpi=110)
        plt.close()
        print("  wrote", out)

        # --- panel 2: anat-edge contour (red) over EPI, uncorrected vs corrected, same slice ---
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        edge_slice = anat_edge[zc, :, :]
        lvl = np.percentile(edge_slice[edge_slice > 0], 90)
        for i, (label, warped) in enumerate([(f"uncorrected MI={mi_unc:.4f}", w_unc), (f"topup-corrected MI={mi_cor:.4f}", w_cor)]):
            ax[i].imshow(warped[zc, :, :].T, cmap="gray", origin="lower")
            ax[i].contour(edge_slice.T, levels=[lvl], colors="red", linewidths=0.6)
            ax[i].set_title(f"run-{run} {label}\nanat edges (red) over warped EPI, R-slice={zc}")
            ax[i].axis("off")
        plt.tight_layout()
        out2 = f"{REG}/qc/distortion_edgeoverlay_run{run}.png"
        plt.savefig(out2, dpi=110)
        plt.close()
        print("  wrote", out2)

        # --- panel 3: difference image (topup-corrected minus uncorrected), to see WHERE topup moved signal ---
        diff = w_cor.astype(np.float32) - w_unc.astype(np.float32)
        vmax = np.percentile(np.abs(diff), 99)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(anat_np[zc, :, :].T, cmap="gray", origin="lower")
        im = ax[0].imshow(diff[zc, :, :].T, cmap="RdBu_r", origin="lower", alpha=0.6, vmin=-vmax, vmax=vmax)
        ax[0].set_title(f"run-{run} (topup - uncorrected) over anat, R-slice={zc}")
        ax[0].axis("off")
        plt.colorbar(im, ax=ax[0], fraction=0.046)
        ax[1].imshow(anat_np[:, :, yc].T, cmap="gray", origin="lower")
        im2 = ax[1].imshow(diff[:, :, yc].T, cmap="RdBu_r", origin="lower", alpha=0.6, vmin=-vmax, vmax=vmax)
        ax[1].set_title(f"run-{run} (topup - uncorrected) over anat, S-slice={yc}")
        ax[1].axis("off")
        plt.colorbar(im2, ax=ax[1], fraction=0.046)
        plt.tight_layout()
        out3 = f"{REG}/qc/distortion_diffmap_run{run}.png"
        plt.savefig(out3, dpi=110)
        plt.close()
        print("  wrote", out3)


if __name__ == "__main__":
    main()
