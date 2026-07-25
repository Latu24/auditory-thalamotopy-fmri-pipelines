"""First-pass sanity-check GLM: a single SoundOn block regressor fit on
run 1 only, used to confirm auditory cortex shows a plausible sound-vs-
baseline response before building the full GLM designs.

Reads one VTC (functional data) and one PRT (condition timing) file, fits
a one-condition GLM, and writes a JSON summary plus a QC overlay PNG. Not
intended as a deliverable statistical map.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import glmlib as G

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ROOT = PROJECT_ROOT
PRT = os.path.join(PROJECT_ROOT, "PRTs")
VTC = f"{ROOT}/derivatives/sub-01/reg/sub-01_run-1_preproc_coreg.vtc"
QC = f"{ROOT}/derivatives/sub-01/analysis/qc"
LOG = f"{ROOT}/logs/analyses_01_sanity.json"

def main():
    hrf = G.two_gamma_hrf()
    hdr = G.read_vtc_header(VTC)
    nv = hdr["DimT"]
    Xtask, names = G.build_run_design(f"{PRT}/run1_SoundOn_timestamp.prt", nv, hrf)
    print("sanity run1: conditions", names, "nvols", nv)
    fit = G.fit_multirun_glm([VTC], [Xtask], hrf, log=print)
    # contrast: SoundOn (col 0) vs baseline
    c = np.zeros(fit["P"]); c[0] = 1.0
    t = G.t_contrast(fit, c)
    tmask = t[fit["mask"]]
    peak = float(np.nanmax(t))
    pk_idx = np.unravel_index(np.argmax(t), t.shape)
    n_sig = int((t > 3.0).sum())
    frac_sig = n_sig / int(fit["mask"].sum())
    out = {
        "run": 1, "condition": names, "nvols": nv, "dof": fit["dof"],
        "phi_ar2": fit["phi"], "mask_voxels": int(fit["mask"].sum()),
        "peak_t": peak, "peak_voxel_zyx": [int(x) for x in pk_idx],
        "n_voxels_t_gt_3": n_sig, "frac_mask_t_gt_3": frac_sig,
        "t_percentiles": {p: float(np.percentile(tmask, p))
                          for p in (50, 90, 99, 99.9)},
    }
    print(json.dumps(out, indent=2))
    with open(LOG, "w") as f:
        json.dump(out, f, indent=2)

    # QC overlay
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    anatf = np.load("/tmp/anatf.npy")
    zsh = np.linspace(pk_idx[0]-30, pk_idx[0]+30, 4).astype(int)
    zsh = np.clip(zsh, 0, t.shape[0]-1)
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for i, z in enumerate(zsh):
        axs[i].imshow(anatf[z], cmap="gray", origin="lower")
        m = np.ma.masked_less(t[z], 3.0)
        axs[i].imshow(m, cmap="hot", vmin=3, vmax=12, alpha=0.7, origin="lower")
        axs[i].set_title(f"SoundOn t, bz={z}"); axs[i].axis("off")
    plt.tight_layout()
    plt.savefig(f"{QC}/sanity_soundon_run1_tmap.png", dpi=80)
    print("saved sanity_soundon_run1_tmap.png; peak t=%.2f at %s" % (peak, pk_idx))

if __name__ == "__main__":
    main()
