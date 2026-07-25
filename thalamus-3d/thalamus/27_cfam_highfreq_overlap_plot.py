"""Thalamus pipeline -- bar-chart visualization of 26_cfam_highfreq_overlap_
stats.py's four spatial-overlap odds ratios (High/Low-CF x High/Low-AM),
alongside the original CF-active x AM-active baseline enrichment from
21_cfam_conjunction_stats.py.

Plotting them side by side shows whether the four High/Low pairings land
noticeably above the plain-significance baseline (which would indicate an
extra, frequency-specific "high pairs with high" effect) or sit alongside
it (which would indicate the test mostly re-detects the already-known CF/AM
spatial conjunction rather than any additional frequency-specific matching).

Inputs: sub-01_CFAM_highfreq_overlap_SMOOTHED.json and
sub-01_CFAM_conjunction_SMOOTHED_statistics.json. Output:
sub-01_CFAM_highfreq_overlap_SMOOTHED.png.
"""
import os, json, time

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

ROOT = PROJECT_ROOT
SG = f"{ROOT}/derivatives/sub-01/analysis/smoothed_glm"


def main():
    log("loading overlap stats (26) + baseline conjunction stats (21) ...")
    hf = json.load(open(f"{SG}/sub-01_CFAM_highfreq_overlap_SMOOTHED.json"))
    base = json.load(open(f"{SG}/sub-01_CFAM_conjunction_SMOOTHED_statistics.json"))
    base_or = base["conjunction_overlap_test"]["fisher_exact"]["odds_ratio"]

    labels = ["CF-active x\nAM-active\n(baseline)",
              "High-CF x\nHigh-AM", "Low-CF x\nLow-AM",
              "High-CF x\nLow-AM", "Low-CF x\nHigh-AM"]
    ors = [base_or,
           hf["HighCF_x_HighAM"]["fisher_odds_ratio"], hf["LowCF_x_LowAM"]["fisher_odds_ratio"],
           hf["HighCF_x_LowAM"]["fisher_odds_ratio"], hf["LowCF_x_HighAM"]["fisher_odds_ratio"]]
    ps = [base["conjunction_overlap_test"]["fisher_exact"]["p_two_sided"],
          hf["HighCF_x_HighAM"]["fisher_p"], hf["LowCF_x_LowAM"]["fisher_p"],
          hf["HighCF_x_LowAM"]["fisher_p"], hf["LowCF_x_HighAM"]["fisher_p"]]
    colors = ["#7f7f7f", "#2ca02c", "#2ca02c", "#ff7f0e", "#ff7f0e"]
    log(f"odds ratios: {dict(zip(labels, [round(o,2) for o in ors]))}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6.5))
    bars = ax.bar(labels, ors, color=colors, edgecolor="k")
    for b, o, p in zip(bars, ors, ps):
        p_txt = "p < 1e-100" if p < 1e-100 else f"p={p:.2e}"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                f"OR={o:.1f}\n{p_txt}", ha="center", va="bottom", fontsize=9)

    ax.axhline(1.0, color="k", linestyle=":", linewidth=1, label="OR=1 (no association)")
    ax.set_ylabel("Fisher's exact odds ratio (spatial overlap vs. chance)")
    ax.set_title("High/Low CF x AM Overlap vs. Baseline")
    ax.set_ylim(0, max(ors) * 1.25)
    ax.text(0.5, 0.97,
            "all four High/Low pairings sit in a similar range, below the plain-\n"
            "significance baseline -- no pairing (e.g. High-CF x High-AM) stands out above\n"
            "the others, so this reflects the known CF/AM conjunction, not frequency-specific matching.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9, ec="0.4"))
    fig.tight_layout()

    out_path = f"{SG}/sub-01_CFAM_highfreq_overlap_SMOOTHED.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"wrote {out_path}")
    log("DONE")
    return out_path


if __name__ == "__main__":
    out = main()
    print(json.dumps({"output": out}, indent=2))
