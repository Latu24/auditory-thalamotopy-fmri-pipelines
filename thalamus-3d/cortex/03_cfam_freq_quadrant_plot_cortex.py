"""Cortex pipeline -- mirrors the thalamus pipeline's
22_cfam_freq_quadrant_plot.py exactly (same style/axes/ticks/single-color-
scatter layout), but for the auditory-cortex CF/AM voxel tables
(02_bestfreq_tables_cortex.py) instead of the thalamus conjunction table.

Reads only the already-shipped, untouched cortex CF/AM bestfreq CSVs -- no
GLM recomputation.

Inputs: sub-01_CF_voxel_bestfreq_table_CORTEX_SMOOTHED.csv and
sub-01_AM_voxel_bestfreq_table_CORTEX_SMOOTHED.csv. Output:
sub-01_CFAM_freq_quadrant_cortex.png.
"""
import os, time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

ROOT = PROJECT_ROOT
SG = f"{ROOT}/derivatives/sub-01/analysis/smoothed_glm"
OUTDIR = f"{ROOT}/derivatives/sub-01/analysis/cortex_work/graphs"

CF_HZ = 200.0 * (8000.0 / 200.0) ** (np.arange(36) / 35.0)
AM_HZ = 1.0 * (16.0 / 1.0) ** (np.arange(9) / 8.0)

COLOR_BOTH = "#9467bd"      # purple, same as thalamus version

CF_LO, CF_HI = 150.0, 10000.0
AM_LO, AM_HI = 0.7, 20.0


def main():
    log("loading cortex CF/AM bestfreq tables ...")
    cf = pd.read_csv(f"{SG}/sub-01_CF_voxel_bestfreq_table_CORTEX_SMOOTHED.csv")
    am = pd.read_csv(f"{SG}/sub-01_AM_voxel_bestfreq_table_CORTEX_SMOOTHED.csv")

    merged = cf.merge(am, on=["voxel_i", "voxel_j", "voxel_k"], suffixes=("_cf", "_am"))
    n_both = len(merged)
    log(f"Both/conjunction n={n_both} (CF n={len(cf)}, AM n={len(am)})")

    cf_hz_both = merged["best_frequency_hz_cf"].to_numpy()
    am_hz_both = merged["best_frequency_hz_am"].to_numpy()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    def _fmt(v):
        return f"{v:.2f}".rstrip("0").rstrip(".")

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(CF_LO, CF_HI)
    ax.set_ylim(AM_LO, AM_HI)

    ax.scatter(cf_hz_both, am_hz_both, c=COLOR_BOTH, s=14, alpha=0.5,
               edgecolor="k", linewidth=0.2, label=f"Both / conjunction (n={n_both})")

    ax.text(0.985, 0.97, f"n={n_both}",
            transform=ax.transAxes, fontsize=10.5, ha="right", va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.9, ec=COLOR_BOTH))

    ax.set_xticks(CF_HZ)
    ax.set_xticklabels([_fmt(v) for v in CF_HZ], rotation=90, fontsize=6)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_yticks(AM_HZ)
    ax.set_yticklabels([_fmt(v) for v in AM_HZ], fontsize=8)
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("preferred carrier frequency (200 Hz-8 kHz)")
    ax.set_ylabel("preferred amplitude modulation frequency (1-16 Hz)")
    ax.set_title("AM, CF Auditory Cortex Voxels")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, which="major", alpha=0.25)

    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    out_path = f"{OUTDIR}/sub-01_CFAM_freq_quadrant_cortex.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"wrote {out_path}")
    log("DONE")
    return out_path


if __name__ == "__main__":
    main()
