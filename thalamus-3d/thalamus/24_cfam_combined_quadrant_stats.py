"""Thalamus pipeline -- combined-population median-split quadrant analysis
(High/Low CF x High/Low AM) across all active-in-thalamus voxels -- CF-only,
AM-only, and conjunction/"both" voxels together -- now that every voxel has
a CF value and an AM value:
  - conjunction voxels: both values are real (significant) Gaussian fits.
  - CF-only voxels: CF value is real; AM value is the sub-threshold fit from
    23_cfam_subthreshold_fit.py (not significant -- descriptive only).
  - AM-only voxels: AM value is real; CF value is the sub-threshold fit from
    23_cfam_subthreshold_fit.py (not significant -- descriptive only).

This extends 21_cfam_conjunction_stats.py's quadrant analysis (which only
covered the conjunction voxels) to the full union, answering: across every
voxel that responds to CF and/or AM at all, is there a relationship between
having a high/low preferred CF and a high/low preferred AM rate -- in
particular the two cross pairings (high-CF/low-AM and low-CF/high-AM).

Median split is computed on log10(Hz) across all voxels together (one shared
CF median, one shared AM median), not per category.

Caveat: the CF-only and AM-only groups' "other-axis" value is the
sub-threshold fit, which is real curve_fit output but describes a
non-significant response (median R^2 = 0.37 for CF-only's AM fit, 0.15 for
AM-only's CF fit) -- so their High/Low label on that axis is noisier and
less trustworthy than the conjunction voxels' labels on either axis.
Reported per-category below so this isn't hidden inside one pooled number.

Inputs: sub-01_CFAM_conjunction_SMOOTHED.json and
sub-01_CFAM_subthreshold_fits_SMOOTHED.json. Output:
sub-01_CFAM_combined_quadrant_SMOOTHED.json.
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

    voxels = []  # (category, cf_hz, am_hz)
    for r in conj["rows"]:
        voxels.append(("both", r["CF_best_fit_freq_hz_continuous"], r["AM_best_fit_freq_hz_continuous"]))
    for r in sub["CF_only_rows"]:
        voxels.append(("CF_only", r["CF_best_fit_freq_hz_continuous"],
                        r["AM_best_fit_freq_hz_continuous_subthreshold"]))
    for r in sub["AM_only_rows"]:
        voxels.append(("AM_only", r["CF_best_fit_freq_hz_continuous_subthreshold"],
                        r["AM_best_fit_freq_hz_continuous"]))

    n_total = len(voxels)
    assert n_total == 2105 == 639 + 436 + 1030, f"unexpected combined n={n_total}"
    cats = np.array([v[0] for v in voxels])
    cf_hz = np.array([v[1] for v in voxels], dtype=np.float64)
    am_hz = np.array([v[2] for v in voxels], dtype=np.float64)
    log_cf, log_am = np.log10(cf_hz), np.log10(am_hz)

    cf_med_log, am_med_log = float(np.median(log_cf)), float(np.median(log_am))
    cf_med_hz, am_med_hz = 10 ** cf_med_log, 10 ** am_med_log
    high_cf, high_am = log_cf >= cf_med_log, log_am >= am_med_log
    log(f"combined median split (n={n_total}): CF median={cf_med_hz:.1f} Hz, "
        f"AM median={am_med_hz:.2f} Hz")

    n_hh = int((high_cf & high_am).sum())    # High CF, High AM
    n_hl = int((high_cf & ~high_am).sum())   # High CF, Low AM
    n_lh = int((~high_cf & high_am).sum())   # Low CF, High AM
    n_ll = int((~high_cf & ~high_am).sum())  # Low CF, Low AM
    table = [[n_hh, n_hl], [n_lh, n_ll]]
    chi2, p_chi2, dof, expected = stats.chi2_contingency(table)
    odds, p_fisher = stats.fisher_exact(table)
    log(f"quadrants: High-CF/High-AM={n_hh}  High-CF/Low-AM={n_hl}  "
        f"Low-CF/High-AM={n_lh}  Low-CF/Low-AM={n_ll}")
    log(f"chi2={chi2:.3f} dof={dof} p={p_chi2:.4f}   Fisher odds ratio={odds:.4f} p={p_fisher:.4f}")

    # ---- same quadrant labels, broken down by category (transparency on
    #      which points are real-fit vs. sub-threshold-fit) ----
    by_cat = {}
    for cat in ("both", "CF_only", "AM_only"):
        sel = cats == cat
        by_cat[cat] = dict(
            n=int(sel.sum()),
            high_CF_high_AM=int((high_cf[sel] & high_am[sel]).sum()),
            high_CF_low_AM=int((high_cf[sel] & ~high_am[sel]).sum()),
            low_CF_high_AM=int((~high_cf[sel] & high_am[sel]).sum()),
            low_CF_low_AM=int((~high_cf[sel] & ~high_am[sel]).sum()),
        )
        log(f"  [{cat}] n={by_cat[cat]['n']}  HH={by_cat[cat]['high_CF_high_AM']}  "
            f"HL={by_cat[cat]['high_CF_low_AM']}  LH={by_cat[cat]['low_CF_high_AM']}  "
            f"LL={by_cat[cat]['low_CF_low_AM']}")

    # ---- the two cross pairings ----
    n_cross = n_hl + n_lh
    n_concordant = n_hh + n_ll
    log(f"CROSS pairings (High-CF/Low-AM + Low-CF/High-AM) = {n_cross} "
        f"({100*n_cross/n_total:.1f}% of {n_total})")
    log(f"CONCORDANT pairings (High-CF/High-AM + Low-CF/Low-AM) = {n_concordant} "
        f"({100*n_concordant/n_total:.1f}% of {n_total})")

    result = {
        "n_total": n_total,
        "median_split_CF_hz": cf_med_hz, "median_split_AM_hz": am_med_hz,
        "quadrants": {"high_CF_high_AM": n_hh, "high_CF_low_AM": n_hl,
                       "low_CF_high_AM": n_lh, "low_CF_low_AM": n_ll},
        "cross_pairings_total": n_cross, "concordant_pairings_total": n_concordant,
        "chi_square": {"chi2": chi2, "dof": dof, "p": p_chi2},
        "fisher_exact": {"odds_ratio": odds, "p": p_fisher},
        "by_category": by_cat,
        "caveat": "CF_only voxels' AM label and AM_only voxels' CF label come from the "
                  "sub-threshold (non-significant) Gaussian fit in "
                  "23_cfam_subthreshold_fit.py, not a significant tuning result -- noisier "
                  "than the 'both' category's labels on either axis (see that script's "
                  "fit_r2 summary).",
    }
    out_path = f"{SG}/sub-01_CFAM_combined_quadrant_SMOOTHED.json"
    json.dump(result, open(out_path, "w"), indent=2)
    log(f"wrote {out_path}")
    log("DONE")
    return result


if __name__ == "__main__":
    res = main()
    print(json.dumps(res, indent=2))
