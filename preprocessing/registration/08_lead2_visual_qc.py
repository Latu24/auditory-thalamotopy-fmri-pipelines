#!/usr/bin/env python3
"""Visual QC for the SyN nonlinear refinement: side-by-side anat /
rigid-only / SyN-refined mean functional, edge overlays, and a difference
map, for all 4 runs, using the same methodology as the earlier distortion
QC visualizations. Output: three PNG panels per run under
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
RIGID_XFM = f"{REG}/xfm/sub-01_run-1_to_anat_rigid.mat"


def gmag(a):
    return np.sqrt(sobel(a, axis=0)**2 + sobel(a, axis=1)**2 + sobel(a, axis=2)**2)


def main():
    anat = ants.image_read(ANAT_NII)
    anat_np = anat.numpy()
    anat_edge = gmag(anat_np)

    for run in [1, 2, 3, 4]:
        mov = ants.image_read(f"{REG}/sub-01_run-{run}_meanfunc_preproc.nii.gz")
        w_rigid = ants.apply_transforms(fixed=anat, moving=mov, transformlist=[RIGID_XFM], interpolator="linear").numpy()
        syn_xfms = [f"{REG}/xfm_lead2_refined/sub-01_run-{run}_syn_fwd_0.nii.gz",
                    f"{REG}/xfm_lead2_refined/sub-01_run-{run}_syn_fwd_1.mat"]
        w_syn = ants.apply_transforms(fixed=anat, moving=mov, transformlist=syn_xfms, interpolator="linear").numpy()

        nz = w_syn > np.percentile(w_syn[w_syn > 0], 1)
        idx = np.where(nz)
        zc = int(np.median(idx[0]))
        yc = int(np.median(idx[2]))

        # panel 1: anat / rigid-only / SyN-refined
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for col, (title, img) in enumerate([("anat", anat_np), ("rigid-only", w_rigid), ("SyN-refined", w_syn)]):
            axes[0, col].imshow(img[zc, :, :].T, cmap="gray", origin="lower")
            axes[0, col].set_title(f"{title}\nR-slice={zc}"); axes[0, col].axis("off")
            axes[1, col].imshow(img[:, :, yc].T, cmap="gray", origin="lower")
            axes[1, col].set_title(f"{title}\nS-slice={yc}"); axes[1, col].axis("off")
        plt.suptitle(f"run-{run}: anat vs rigid-only vs SyN-refined (nonlinear refinement)")
        plt.tight_layout()
        out = f"{REG}/qc/lead2_visual_run{run}_anat_rigid_syn.png"
        plt.savefig(out, dpi=110); plt.close()
        print("wrote", out)

        # panel 2: anat-edge contour over EPI, rigid-only vs SyN-refined
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        edge_slice = anat_edge[zc, :, :]
        lvl = np.percentile(edge_slice[edge_slice > 0], 90)
        for i, (label, warped) in enumerate([("rigid-only", w_rigid), ("SyN-refined", w_syn)]):
            ax[i].imshow(warped[zc, :, :].T, cmap="gray", origin="lower")
            ax[i].contour(edge_slice.T, levels=[lvl], colors="red", linewidths=0.6)
            ax[i].set_title(f"run-{run} {label}: anat edges (red) over warped EPI, R-slice={zc}")
            ax[i].axis("off")
        plt.tight_layout()
        out2 = f"{REG}/qc/lead2_edgeoverlay_run{run}.png"
        plt.savefig(out2, dpi=110); plt.close()
        print("wrote", out2)

        # panel 3: diff map (SyN-refined minus rigid-only)
        diff = w_syn.astype(np.float32) - w_rigid.astype(np.float32)
        vmax = np.percentile(np.abs(diff), 99)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(anat_np[zc, :, :].T, cmap="gray", origin="lower")
        im = ax[0].imshow(diff[zc, :, :].T, cmap="RdBu_r", origin="lower", alpha=0.6, vmin=-vmax, vmax=vmax)
        ax[0].set_title(f"run-{run} (SyN-refined - rigid-only) over anat, R-slice={zc}"); ax[0].axis("off")
        plt.colorbar(im, ax=ax[0], fraction=0.046)
        ax[1].imshow(anat_np[:, :, yc].T, cmap="gray", origin="lower")
        im2 = ax[1].imshow(diff[:, :, yc].T, cmap="RdBu_r", origin="lower", alpha=0.6, vmin=-vmax, vmax=vmax)
        ax[1].set_title(f"run-{run} (SyN-refined - rigid-only) over anat, S-slice={yc}"); ax[1].axis("off")
        plt.colorbar(im2, ax=ax[1], fraction=0.046)
        plt.tight_layout()
        out3 = f"{REG}/qc/lead2_diffmap_run{run}.png"
        plt.savefig(out3, dpi=110); plt.close()
        print("wrote", out3)


if __name__ == "__main__":
    main()
