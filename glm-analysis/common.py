"""Shared paths and plotting/logging helpers for the GLM analysis driver
scripts. Centralizes VTC/PRT file paths, condition orderings, QC overlay
plotting, and JSON logging used by the sound-detection, carrier-frequency,
and amplitude-modulation analyses.
"""
import os, json
import numpy as np

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ROOT = PROJECT_ROOT
PRT = os.path.join(PROJECT_ROOT, "PRTs")
ANA = f"{ROOT}/derivatives/sub-01/analysis"
QC = f"{ANA}/qc"
LOGS = f"{ROOT}/logs"
VTCS = [f"{ROOT}/derivatives/sub-01/reg/sub-01_run-{r}_preproc_coreg.vtc"
        for r in (1, 2, 3, 4)]
SOUNDON = [f"{PRT}/run{r}_SoundOn_timestamp.prt" for r in (1, 2, 3, 4)]
BYFREQ = [f"{PRT}/sub01_run-{r}_tone_events_by_frequency_timestamp.prt"
          for r in (1, 2, 3, 4)]
BYAM = [f"{PRT}/sub01_run-{r}_tone_events_by_AM_timestamp.prt"
        for r in (1, 2, 3, 4)]

FREQ_ORDER = [f"Freq_{i:02d}" for i in range(1, 37)]
AM_ORDER = [f"AM_{i}" for i in range(1, 10)]


def overlay_tmap(t, anatf, path, title, thr=3.0, vmax=12, cmap="hot",
                 center=None, nslices=4, span=30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if center is None:
        center = int(np.unravel_index(np.argmax(np.abs(t)), t.shape)[0])
    zsh = np.clip(np.linspace(center - span, center + span, nslices).astype(int),
                  0, t.shape[0] - 1)
    fig, axs = plt.subplots(1, nslices, figsize=(4 * nslices, 4))
    if nslices == 1:
        axs = [axs]
    for i, z in enumerate(zsh):
        axs[i].imshow(anatf[z], cmap="gray", origin="lower")
        m = np.ma.masked_less(np.abs(t[z]), thr)
        axs[i].imshow(np.ma.masked_less(t[z], thr) if cmap == "hot" else m,
                      cmap=cmap, vmin=thr, vmax=vmax, alpha=0.7, origin="lower")
        axs[i].set_title(f"{title} bz={z}")
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close(fig)


def save_json(path, obj):
    def _ser(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_ser)
