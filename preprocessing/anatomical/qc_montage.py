#!/usr/bin/env python3
"""Reusable QC montage helper: saves tri-planar slice montages (and optional
before/after comparison) as PNG. Uses matplotlib only."""
import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _slices(vol, axis, n=5, lo=0.2, hi=0.8):
    d = vol.shape[axis]
    return [int(d*f) for f in np.linspace(lo, hi, n)]

def _take(vol, axis, i):
    sl = [slice(None)]*3; sl[axis]=i
    img = vol[tuple(sl)]
    return np.rot90(img)

def montage(vol, out_png, title="", vmax=None, vmin=0):
    if vmax is None:
        vmax = np.percentile(vol[vol>0], 99.5) if (vol>0).any() else vol.max()
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    names=["sagittal (x)","coronal (y)","axial (z)"]
    for a in range(3):
        for j,i in enumerate(_slices(vol, a)):
            ax=axes[a,j]
            ax.imshow(_take(vol,a,i), cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            if j==0: ax.set_ylabel(names[a], fontsize=10)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(out_png, dpi=90); plt.close(fig)
    print("wrote", out_png)

def compare(volA, volB, out_png, titleA="A", titleB="B", axis=0, vmax=None):
    both=np.concatenate([volA[volA>0], volB[volB>0]])
    if vmax is None: vmax=np.percentile(both,99.5)
    idxs=_slices(volA, axis, n=4, lo=0.3, hi=0.7)
    fig, axes=plt.subplots(2, 4, figsize=(14, 7))
    for j,i in enumerate(idxs):
        axes[0,j].imshow(_take(volA,axis,i),cmap="gray",vmin=0,vmax=vmax,origin="lower")
        axes[1,j].imshow(_take(volB,axis,i),cmap="gray",vmin=0,vmax=vmax,origin="lower")
        for r in (0,1):
            axes[r,j].set_xticks([]); axes[r,j].set_yticks([])
    axes[0,0].set_ylabel(titleA,fontsize=11); axes[1,0].set_ylabel(titleB,fontsize=11)
    fig.tight_layout(); fig.savefig(out_png,dpi=95); plt.close(fig)
    print("wrote", out_png)

if __name__=="__main__":
    import sys
    PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ANAT = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "anat")
    RAW = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "rawdata_nifti")
    uni=nib.load(os.path.join(RAW,"sub-01_acq-mp2rage_UNI.nii.gz")).get_fdata()
    den=nib.load(os.path.join(ANAT,"sub-01_desc-UNIdenoised_lambda6.nii.gz")).get_fdata()
    os.makedirs(os.path.join(ANAT,"qc"),exist_ok=True)
    compare(uni,den,os.path.join(ANAT,"qc","qc_01_denoise_sagittal.png"),
            "UNI raw","UNI denoised (lambda6)",axis=0,vmax=4095)
    compare(uni,den,os.path.join(ANAT,"qc","qc_01_denoise_axial.png"),
            "UNI raw","UNI denoised (lambda6)",axis=2,vmax=4095)
