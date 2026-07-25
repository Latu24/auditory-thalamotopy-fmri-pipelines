"""Thalamus pipeline -- direct two-group distributional comparison as an
alternative to the contingency/overlap-table framing used in scripts 21, 24,
26:

  1) Split all combined voxels (conjunction + CF-only w/ sub-threshold AM +
     AM-only w/ sub-threshold CF, exactly scripts 24/25's combined
     population) into High-CF vs Low-CF by the shared CF median -- then ask:
     does the AM frequency distribution differ between these two groups?
     (Mann-Whitney U, plus an independent-samples t-test on log10-Hz as a
     parametric cross-check.)
  2) Symmetric: split into High-AM vs Low-AM by the shared AM median -- does
     the CF frequency distribution differ between these two groups?

This tests the same underlying question as the earlier Pearson/Spearman/
Kendall correlation (script 21) and the quadrant/overlap tests (scripts 24,
26), but as a direct two-sample comparison on the continuous other-axis
value rather than a categorical High/Low x High/Low contingency table -- it
avoids the baseline CF/AM spatial conjunction inflating every High/Low
pairing's odds ratio equally, since it only ever uses one modality's median
as the grouping variable, never crosses both into a 2x2.

Uses the same sub-threshold fits as scripts 24/25 (23_cfam_subthreshold_
fit.py) for CF-only/AM-only voxels' other-axis value; same caveat applies
(that value is a non-significant, descriptive Gaussian fit for those two
groups).

Inputs: sub-01_CFAM_conjunction_SMOOTHED.json and
sub-01_CFAM_subthreshold_fits_SMOOTHED.json. Outputs:
sub-01_CFAM_highlow_groupcompare_SMOOTHED.json and a box+strip-plot PNG.
"""
import os, json, time
import numpy as np
from scipy import stats

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

ROOT = PROJECT_ROOT
SG = f"{ROOT}/derivatives/sub-01/analysis/smoothed_glm"


def main():
    log("loading conjunction table + sub-threshold fits ...")
    conj = json.load(open(f"{SG}/sub-01_CFAM_conjunction_SMOOTHED.json"))
    sub = json.load(open(f"{SG}/sub-01_CFAM_subthreshold_fits_SMOOTHED.json"))

    cf_hz, am_hz, cats = [], [], []
    for r in conj["rows"]:
        cf_hz.append(r["CF_best_fit_freq_hz_continuous"]); am_hz.append(r["AM_best_fit_freq_hz_continuous"])
        cats.append("both")
    for r in sub["CF_only_rows"]:
        cf_hz.append(r["CF_best_fit_freq_hz_continuous"])
        am_hz.append(r["AM_best_fit_freq_hz_continuous_subthreshold"])
        cats.append("CF_only")
    for r in sub["AM_only_rows"]:
        cf_hz.append(r["CF_best_fit_freq_hz_continuous_subthreshold"])
        am_hz.append(r["AM_best_fit_freq_hz_continuous"])
        cats.append("AM_only")
    cf_hz, am_hz = np.array(cf_hz), np.array(am_hz)
    n_total = len(cf_hz)
    assert n_total == 2105
    log(f"combined population n={n_total}")

    log_cf, log_am = np.log10(cf_hz), np.log10(am_hz)
    cf_med, am_med = float(np.median(log_cf)), float(np.median(log_am))
    log(f"CF median={10**cf_med:.1f} Hz  AM median={10**am_med:.2f} Hz")

    # =========================================================================
    # (1) High-CF vs Low-CF -> compare their AM distributions
    # =========================================================================
    high_cf, low_cf = log_cf >= cf_med, log_cf < cf_med
    am_given_highcf = am_hz[high_cf]
    am_given_lowcf = am_hz[low_cf]
    u_stat_1, p_mwu_1 = stats.mannwhitneyu(am_given_highcf, am_given_lowcf, alternative="two-sided")
    t_stat_1, p_t_1 = stats.ttest_ind(np.log10(am_given_highcf), np.log10(am_given_lowcf))
    log(f"[High-CF n={high_cf.sum()} vs Low-CF n={low_cf.sum()}] AM median "
        f"{np.median(am_given_highcf):.2f} vs {np.median(am_given_lowcf):.2f} Hz")
    log(f"  Mann-Whitney U={u_stat_1:.1f} p={p_mwu_1:.4f}   t-test(log10-Hz) t={t_stat_1:.3f} p={p_t_1:.4f}")

    # =========================================================================
    # (2) High-AM vs Low-AM -> compare their CF distributions
    # =========================================================================
    high_am, low_am = log_am >= am_med, log_am < am_med
    cf_given_higham = cf_hz[high_am]
    cf_given_lowam = cf_hz[low_am]
    u_stat_2, p_mwu_2 = stats.mannwhitneyu(cf_given_higham, cf_given_lowam, alternative="two-sided")
    t_stat_2, p_t_2 = stats.ttest_ind(np.log10(cf_given_higham), np.log10(cf_given_lowam))
    log(f"[High-AM n={high_am.sum()} vs Low-AM n={low_am.sum()}] CF median "
        f"{np.median(cf_given_higham):.1f} vs {np.median(cf_given_lowam):.1f} Hz")
    log(f"  Mann-Whitney U={u_stat_2:.1f} p={p_mwu_2:.4f}   t-test(log10-Hz) t={t_stat_2:.3f} p={p_t_2:.4f}")

    result = {
        "n_total": n_total,
        "CF_median_hz": 10 ** cf_med, "AM_median_hz": 10 ** am_med,
        "high_CF_vs_low_CF__AM_distribution": {
            "n_high_CF": int(high_cf.sum()), "n_low_CF": int(low_cf.sum()),
            "AM_median_given_highCF": float(np.median(am_given_highcf)),
            "AM_median_given_lowCF": float(np.median(am_given_lowcf)),
            "mann_whitney_U": float(u_stat_1), "mann_whitney_p": float(p_mwu_1),
            "ttest_log10Hz_t": float(t_stat_1), "ttest_log10Hz_p": float(p_t_1),
        },
        "high_AM_vs_low_AM__CF_distribution": {
            "n_high_AM": int(high_am.sum()), "n_low_AM": int(low_am.sum()),
            "CF_median_given_highAM": float(np.median(cf_given_higham)),
            "CF_median_given_lowAM": float(np.median(cf_given_lowam)),
            "mann_whitney_U": float(u_stat_2), "mann_whitney_p": float(p_mwu_2),
            "ttest_log10Hz_t": float(t_stat_2), "ttest_log10Hz_p": float(p_t_2),
        },
        "caveat": "CF_only voxels' AM value and AM_only voxels' CF value are sub-threshold "
                  "(non-significant) Gaussian fits from 23_cfam_subthreshold_fit.py, not "
                  "significant tuning results.",
    }
    out_json = f"{SG}/sub-01_CFAM_highlow_groupcompare_SMOOTHED.json"
    json.dump(result, open(out_json, "w"), indent=2)
    log(f"wrote {out_json}")

    # =========================================================================
    # plot: two side-by-side box+strip plots
    # =========================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    CF_HZ = 200.0 * (8000.0 / 200.0) ** (np.arange(36) / 35.0)
    AM_HZ = 1.0 * (16.0 / 1.0) ** (np.arange(9) / 8.0)

    def _fmt(v):
        return f"{v:.2f}".rstrip("0").rstrip(".")

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))

    ax = axes[0]
    data1 = [am_given_lowcf, am_given_highcf]
    bp = ax.boxplot(data1, labels=[f"Low CF\n(n={low_cf.sum()})", f"High CF\n(n={high_cf.sum()})"],
                     showfliers=False, patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], ["#1f77b4", "#d62728"]):
        patch.set_facecolor(c); patch.set_alpha(0.5)
    rng = np.random.default_rng(0)
    for xi, d, c in zip([1, 2], data1, ["#1f77b4", "#d62728"]):
        ax.scatter(rng.normal(xi, 0.05, size=len(d)), d, s=8, alpha=0.35, color=c)
    ax.set_yscale("log")
    ax.set_yticks(AM_HZ)
    ax.set_yticklabels([_fmt(v) for v in AM_HZ], fontsize=8)
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylabel("AM frequency (Hz)")
    ax.set_title(f"AM freq by CF group\nMann-Whitney p={p_mwu_1:.3f}")

    ax = axes[1]
    data2 = [cf_given_lowam, cf_given_higham]
    bp = ax.boxplot(data2, labels=[f"Low AM\n(n={low_am.sum()})", f"High AM\n(n={high_am.sum()})"],
                     showfliers=False, patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], ["#1f77b4", "#d62728"]):
        patch.set_facecolor(c); patch.set_alpha(0.5)
    for xi, d, c in zip([1, 2], data2, ["#1f77b4", "#d62728"]):
        ax.scatter(rng.normal(xi, 0.05, size=len(d)), d, s=8, alpha=0.35, color=c)
    ax.set_yscale("log")
    ax.set_yticks(CF_HZ)
    ax.set_yticklabels([_fmt(v) for v in CF_HZ], fontsize=6)
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylabel("CF frequency (Hz)")
    ax.set_title(f"CF freq by AM group\nMann-Whitney p={p_mwu_2:.3f}")

    fig.suptitle("High/Low CF vs. AM: Group Comparison", fontsize=14)
    fig.tight_layout()
    out_png = f"{SG}/sub-01_CFAM_highlow_groupcompare_SMOOTHED.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    log(f"wrote {out_png}")
    log("DONE")
    return result, out_png


if __name__ == "__main__":
    res, png = main()
    print(json.dumps(res, indent=2))
