"""Thalamus pipeline -- edge-effect check on the CF/AM preferred-frequency
distributions.

Motivated by a suspiciously dense pile-up of points sitting exactly at the
axis boundaries (CF=200 Hz, AM=1 Hz and 16 Hz) across several of the earlier
scatter plots. That is a well-known bounded-curve_fit artifact:
18_cfam_conjunction_smoothed.py's bestfreq_block_no_gate() fits the
Gaussian's mu with scipy.optimize.curve_fit(..., bounds=(lo, hi)) in
log10(Hz) space, where lo/hi are the tested range's own edges (x[0], x[-1]).
If the true peak of a voxel's tuning curve lies at or beyond the tested
range, or the profile is noisy/flat, curve_fit can saturate at that
boundary -- checked here via the discrete condition_index (the argmax-based
winning condition, 1..q, always defined regardless of fit convergence -- not
affected by the continuous fit's bounds, so a clean, fit-independent way to
check this) rather than the continuous Hz value.

Test: chi-square goodness-of-fit of the condition_index histogram against a
uniform distribution (each of the q conditions equally likely), computed
separately for CF (q=36, 1075 significant voxels) and AM (q=9, 1669
significant voxels). Also a one-sided binomial test on each edge bin
(condition 1, condition q) against the uniform expectation 1/q, and on the
combined edge total against 2/q.

Inputs: sub-01_CF_voxeltable_SMOOTHED.json,
sub-01_AM_voxeltable_SMOOTHED.json. Outputs:
sub-01_CFAM_edge_effect_SMOOTHED.json and a condition-index histogram PNG.
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

CF_HZ = 200.0 * (8000.0 / 200.0) ** (np.arange(36) / 35.0)
AM_HZ = 1.0 * (16.0 / 1.0) ** (np.arange(9) / 8.0)


def check_edges(tag, cond_idx, q, hz_grid):
    n = len(cond_idx)
    counts = np.bincount(cond_idx, minlength=q + 1)[1:]  # counts[0] = condition 1, ..., counts[q-1] = condition q
    expected = n / q
    chi2, p_chi2 = stats.chisquare(counts, f_exp=np.full(q, expected))
    log(f"[{tag}] n={n} q={q}  expected/bin={expected:.1f}  "
        f"chi2 goodness-of-fit vs uniform: chi2={chi2:.2f} p={p_chi2:.2e}")

    n_lo, n_hi = int(counts[0]), int(counts[-1])
    bt_lo = stats.binomtest(n_lo, n, 1 / q, alternative="greater")
    bt_hi = stats.binomtest(n_hi, n, 1 / q, alternative="greater")
    n_edge = n_lo + n_hi
    bt_edge = stats.binomtest(n_edge, n, 2 / q, alternative="greater")
    log(f"[{tag}] lowest condition ({hz_grid[0]:.1f} Hz): n={n_lo} "
        f"({100*n_lo/n:.2f}% vs {100/q:.2f}% expected)  p={bt_lo.pvalue:.2e}")
    log(f"[{tag}] highest condition ({hz_grid[-1]:.1f} Hz): n={n_hi} "
        f"({100*n_hi/n:.2f}% vs {100/q:.2f}% expected)  p={bt_hi.pvalue:.2e}")
    log(f"[{tag}] combined edges: n={n_edge} ({100*n_edge/n:.2f}% vs {200/q:.2f}% expected)  "
        f"p={bt_edge.pvalue:.2e}")

    return dict(n=n, q=q, expected_per_bin=expected, counts_per_condition=counts.tolist(),
                chi2=chi2, chi2_p=p_chi2,
                lowest_condition_hz=float(hz_grid[0]), n_lowest=n_lo, p_lowest_overrep=bt_lo.pvalue,
                highest_condition_hz=float(hz_grid[-1]), n_highest=n_hi, p_highest_overrep=bt_hi.pvalue,
                n_edges_combined=n_edge, pct_edges_combined=100 * n_edge / n,
                pct_edges_expected=200 / q, p_edges_overrep=bt_edge.pvalue)


def main():
    log("loading real (significant-only) per-modality voxel tables ...")
    cf_table = json.load(open(f"{SG}/sub-01_CF_voxeltable_SMOOTHED.json"))
    am_table = json.load(open(f"{SG}/sub-01_AM_voxeltable_SMOOTHED.json"))
    cf_cond = np.array([r["condition_index"] for r in cf_table["rows"]])
    am_cond = np.array([r["condition_index"] for r in am_table["rows"]])
    assert (len(cf_cond), len(am_cond)) == (1075, 1669)

    cf_res = check_edges("CF", cf_cond, 36, CF_HZ)
    am_res = check_edges("AM", am_cond, 9, AM_HZ)

    result = {"CF": cf_res, "AM": am_res}
    out_json = f"{SG}/sub-01_CFAM_edge_effect_SMOOTHED.json"
    json.dump(result, open(out_json, "w"), indent=2)
    log(f"wrote {out_json}")

    # =========================================================================
    # plot: condition-index histograms for CF and AM, edges highlighted
    # =========================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    def _fmt(v):
        return f"{v:.2f}".rstrip("0").rstrip(".")

    for ax, tag, cond, q, hz_grid, res in [
        (axes[0], "CF", cf_cond, 36, CF_HZ, cf_res),
        (axes[1], "AM", am_cond, 9, AM_HZ, am_res),
    ]:
        counts = np.array(res["counts_per_condition"])
        colors = ["#d62728" if (i == 0 or i == q - 1) else "#1f77b4" for i in range(q)]
        ax.bar(np.arange(1, q + 1), counts, color=colors, edgecolor="k", linewidth=0.4)
        ax.axhline(res["expected_per_bin"], color="k", linestyle="--", linewidth=1.2,
                   label=f"expected under uniform ({res['expected_per_bin']:.1f})")
        ax.set_xticks(np.arange(1, q + 1))
        ax.set_xticklabels([_fmt(v) for v in hz_grid], rotation=90, fontsize=6 if q > 10 else 8)
        ax.set_xlabel(f"preferred {tag} condition (Hz)")
        ax.set_ylabel("voxel count")
        ax.set_title(f"{tag}: n={res['n']}, edges={res['n_edges_combined']} "
                     f"({res['pct_edges_combined']:.1f}% vs {res['pct_edges_expected']:.1f}% exp.)\n"
                     f"chi2 p={res['chi2_p']:.1e}, edge-overrep p={res['p_edges_overrep']:.1e}")
        ax.legend(fontsize=8, loc="upper center")

    fig.suptitle("Edge Effect Check: Preferred Condition Distribution", fontsize=14)
    fig.tight_layout()
    out_png = f"{SG}/sub-01_CFAM_edge_effect_SMOOTHED.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    log(f"wrote {out_png}")
    log("DONE")
    return result


if __name__ == "__main__":
    res = main()
    print(json.dumps(res, indent=2))
