"""Thalamus pipeline -- single-panel CF/AM preferred-frequency scatter for
the conjunction voxels only (active in both CF and AM, p<0.01 uncorrected
omnibus-F each), laid out like script 19's F-value union scatter but
plotting each voxel's actual preferred CF frequency (Hz) / preferred AM rate
(Hz) from the Gaussian fits, rather than F-statistics.

This is a separate deliverable from the marginal-strip figure produced by
script 18 and script 19's F-value plot; it does not replace either.

Reads only the already-shipped, untouched conjunction table from script 18 --
no GLM recomputation.

Input: sub-01_CFAM_conjunction_SMOOTHED.json. Output:
sub-01_CFAM_freq_quadrant_SMOOTHED.png.
"""
import os, json, time
import numpy as np

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

ROOT = PROJECT_ROOT
ANA = f"{ROOT}/derivatives/sub-01/analysis"
SG = f"{ANA}/smoothed_glm"

CF_HZ = 200.0 * (8000.0 / 200.0) ** (np.arange(36) / 35.0)
AM_HZ = 1.0 * (16.0 / 1.0) ** (np.arange(9) / 8.0)

COLOR_BOTH = "#9467bd"      # purple

CF_LO, CF_HI = 150.0, 10000.0
AM_LO, AM_HI = 0.7, 20.0


def main():
    log("loading already-shipped conjunction table ...")
    conj = json.load(open(f"{SG}/sub-01_CFAM_conjunction_SMOOTHED.json"))

    cf_hz_both = np.array([r["CF_best_fit_freq_hz_continuous"] for r in conj["rows"]])
    am_hz_both = np.array([r["AM_best_fit_freq_hz_continuous"] for r in conj["rows"]])
    n_both = len(conj["rows"])
    assert n_both == 639, (
        f"conjunction count changed from the already-shipped run ({n_both}) != 639 "
        f"-- inputs drifted, investigate")
    log(f"Both/conjunction n={n_both}")

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

    # ---- both / conjunction: real CF freq (x), real AM freq (y) ----
    ax.scatter(cf_hz_both, am_hz_both, c=COLOR_BOTH, s=32, alpha=0.7,
               edgecolor="k", linewidth=0.3, label=f"Both / conjunction (n={n_both})")

    ax.text(0.985, 0.97, f"n={n_both}",
            transform=ax.transAxes, fontsize=10.5, ha="right", va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.9, ec=COLOR_BOTH))

    ax.set_xticks(CF_HZ)
    ax.set_xticklabels([_fmt(v) for v in CF_HZ], rotation=90, fontsize=6)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_yticks(AM_HZ)
    ax.set_yticklabels([_fmt(v) for v in AM_HZ], fontsize=8)
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("preferred carrier frequency (200 Hz–8 kHz)")
    ax.set_ylabel("preferred amplitude modulation frequency (1–16 Hz)")
    ax.set_title("AM, CF Thalamus Voxels")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, which="major", alpha=0.25)

    fig.tight_layout()
    out_path = f"{SG}/sub-01_CFAM_freq_quadrant_SMOOTHED.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"wrote {out_path}")
    log("DONE")
    return out_path


if __name__ == "__main__":
    out = main()
    print(json.dumps({"output": out}, indent=2))
