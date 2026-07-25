"""Thalamus pipeline -- visualization of 24_cfam_combined_quadrant_stats.py's
median-split quadrant analysis (High/Low CF x High/Low AM, across all
CF-only/AM-only/conjunction voxels). Reuses that script's own already-
computed median split + quadrant counts + chi-square/Fisher results (does
not recompute them) -- purely a plotting layer.

Points: CF-only (red), AM-only (blue), Both/conjunction (purple), each at its
real-or-sub-threshold (CF Hz, AM Hz) coordinate (see 23_cfam_subthreshold_
fit.py for what "sub-threshold" means for the CF-only/AM-only groups).
Dashed crosshair lines mark the shared median split; each quadrant is
annotated with its voxel count; chi-square/Fisher result shown top-right.

Inputs: sub-01_CFAM_conjunction_SMOOTHED.json,
sub-01_CFAM_subthreshold_fits_SMOOTHED.json,
sub-01_CFAM_combined_quadrant_SMOOTHED.json. Output:
sub-01_CFAM_combined_quadrant_SMOOTHED.png.
"""
import os, json, time
import numpy as np

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

ROOT = PROJECT_ROOT
SG = f"{ROOT}/derivatives/sub-01/analysis/smoothed_glm"

CF_HZ = 200.0 * (8000.0 / 200.0) ** (np.arange(36) / 35.0)
AM_HZ = 1.0 * (16.0 / 1.0) ** (np.arange(9) / 8.0)

COLOR_CF_ONLY = "#d62728"   # red
COLOR_AM_ONLY = "#1f77b4"   # blue
COLOR_BOTH = "#9467bd"      # purple

CF_LO, CF_HI = 150.0, 10000.0
AM_LO, AM_HI = 0.7, 20.0


def main():
    log("loading conjunction table + sub-threshold fits + combined quadrant stats ...")
    conj = json.load(open(f"{SG}/sub-01_CFAM_conjunction_SMOOTHED.json"))
    sub = json.load(open(f"{SG}/sub-01_CFAM_subthreshold_fits_SMOOTHED.json"))
    quad = json.load(open(f"{SG}/sub-01_CFAM_combined_quadrant_SMOOTHED.json"))

    cf_both = np.array([r["CF_best_fit_freq_hz_continuous"] for r in conj["rows"]])
    am_both = np.array([r["AM_best_fit_freq_hz_continuous"] for r in conj["rows"]])
    cf_cfonly = np.array([r["CF_best_fit_freq_hz_continuous"] for r in sub["CF_only_rows"]])
    am_cfonly = np.array([r["AM_best_fit_freq_hz_continuous_subthreshold"] for r in sub["CF_only_rows"]])
    cf_amonly = np.array([r["CF_best_fit_freq_hz_continuous_subthreshold"] for r in sub["AM_only_rows"]])
    am_amonly = np.array([r["AM_best_fit_freq_hz_continuous"] for r in sub["AM_only_rows"]])

    n_total = len(cf_both) + len(cf_cfonly) + len(cf_amonly)
    assert n_total == quad["n_total"] == 2105

    cf_med, am_med = quad["median_split_CF_hz"], quad["median_split_AM_hz"]
    q = quad["quadrants"]
    chi2, p_chi2 = quad["chi_square"]["chi2"], quad["chi_square"]["p"]

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

    ax.scatter(cf_cfonly, am_cfonly, c=COLOR_CF_ONLY, s=28, alpha=0.6,
               edgecolor="k", linewidth=0.3, label=f"CF only (n={len(cf_cfonly)})")
    ax.scatter(cf_amonly, am_amonly, c=COLOR_AM_ONLY, s=28, alpha=0.6,
               edgecolor="k", linewidth=0.3, label=f"AM only (n={len(cf_amonly)})")
    ax.scatter(cf_both, am_both, c=COLOR_BOTH, s=28, alpha=0.6,
               edgecolor="k", linewidth=0.3, label=f"Both / conjunction (n={len(cf_both)})")

    ax.axvline(cf_med, color="k", linestyle="--", linewidth=1.3)
    ax.axhline(am_med, color="k", linestyle="--", linewidth=1.3)

    # ---- quadrant count annotations, placed near each corner ----
    quad_positions = [
        (0.73, 0.97, f"High CF / High AM\nn={q['high_CF_high_AM']}"),
        (0.73, 0.04, f"High CF / Low AM\nn={q['high_CF_low_AM']}"),
        (0.03, 0.97, f"Low CF / High AM\nn={q['low_CF_high_AM']}"),
        (0.03, 0.04, f"Low CF / Low AM\nn={q['low_CF_low_AM']}"),
    ]
    for xf, yf, txt in quad_positions:
        va = "top" if yf > 0.5 else "bottom"
        ax.text(xf, yf, txt, transform=ax.transAxes, fontsize=9.5, ha="left", va=va,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85, ec="0.4"))

    ax.text(0.985, 0.5,
            f"median split (n={n_total})\nCF={cf_med:.0f} Hz  AM={am_med:.2f} Hz\n"
            f"chi2={chi2:.3f}, p={p_chi2:.3f}\n(no significant association)",
            transform=ax.transAxes, fontsize=10, ha="right", va="center",
            bbox=dict(boxstyle="round", fc="white", alpha=0.92, ec="k"))

    ax.set_xticks(CF_HZ)
    ax.set_xticklabels([_fmt(v) for v in CF_HZ], rotation=90, fontsize=6)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_yticks(AM_HZ)
    ax.set_yticklabels([_fmt(v) for v in AM_HZ], fontsize=8)
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("preferred carrier frequency (200 Hz–8 kHz)")
    ax.set_ylabel("preferred amplitude modulation frequency (1–16 Hz)")
    ax.set_title("High/Low CF x AM Quadrant Split")
    ax.legend(loc="upper left", fontsize=9.5, bbox_to_anchor=(0.20, 1.0))
    ax.grid(True, which="major", alpha=0.2)

    fig.tight_layout()
    out_path = f"{SG}/sub-01_CFAM_combined_quadrant_SMOOTHED.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"wrote {out_path}")
    log("DONE")
    return out_path


if __name__ == "__main__":
    out = main()
    print(json.dumps({"output": out}, indent=2))
