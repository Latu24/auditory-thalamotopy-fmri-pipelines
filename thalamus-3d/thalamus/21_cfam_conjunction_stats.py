"""Thalamus pipeline -- inferential statistics for the CF/AM conjunction
figure (18_cfam_conjunction_smoothed.py's plot_conjunction() /
sub-01_CFAM_conjunction_SMOOTHED.png). Answers two questions that figure
raises but doesn't itself test:

  1) Is the CF/AM CONJUNCTION (voxels active in BOTH p<0.01 uncorrected
     omnibus-F CF and AM) bigger than expected if CF-sensitivity and
     AM-sensitivity were spatially independent within the thalamus? Tested via
     a 2x2 contingency table (CF active/inactive x AM active/inactive, over
     all N thalamus voxels) with Fisher's exact test (odds ratio + exact p)
     and a chi-square test of independence as a large-sample cross-check, plus
     Jaccard/Dice overlap coefficients.

     Caveat: both tests assume the N thalamus voxels are independent trials.
     fMRI voxels are not independent -- this GLM used smoothed data, and even
     unsmoothed BOLD has intrinsic spatial autocorrelation -- so the p-values
     here are a standard, commonly-reported first-pass enrichment measure, not
     a spatially-valid inferential test. A spatial/cluster permutation test
     would be needed for a fully rigorous p-value; not implemented here.

  2) Within the conjunction voxels, is a voxel's preferred CF frequency
     correlated with its preferred AM rate? Pearson r (already shown on the
     figure, log10-Hz) plus its p-value, Spearman rho (rank-based, robust to
     the non-linear/clustered pattern visible in the scatter), and Kendall tau
     as a third, more conservative rank-based cross-check.

Reads only the already-shipped, untouched tables from scripts 18/19 (does not
recompute anything from the .glm files) -- purely a statistics-on-tables
script.

Inputs: the per-modality and conjunction voxel tables (JSON) plus the
native-space thalamus mask. Output:
sub-01_CFAM_conjunction_SMOOTHED_statistics.json.
"""
import os, json, time
import numpy as np
import nibabel as nib
from scipy import stats

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

ROOT = PROJECT_ROOT
ANA = f"{ROOT}/derivatives/sub-01/analysis"
SG = f"{ANA}/smoothed_glm"
WORK = f"{ANA}/thalamus_work"
THAL_NII = f"{WORK}/sub-01_thalamus_native.nii.gz"


def main():
    log("loading already-shipped voxel tables (CF/AM per-modality + conjunction) ...")
    cf_table = json.load(open(f"{SG}/sub-01_CF_voxeltable_SMOOTHED.json"))
    am_table = json.load(open(f"{SG}/sub-01_AM_voxeltable_SMOOTHED.json"))
    conj = json.load(open(f"{SG}/sub-01_CFAM_conjunction_SMOOTHED.json"))

    cf_ijk = set(tuple(r["voxel_ijk_native"]) for r in cf_table["rows"])
    am_ijk = set(tuple(r["voxel_ijk_native"]) for r in am_table["rows"])
    both_ijk = set(tuple(r["voxel_ijk_native"]) for r in conj["rows"])
    assert both_ijk == cf_ijk & am_ijk, (
        "conjunction table's voxel set != (CF table) & (AM table) -- inconsistent inputs")

    n_cf, n_am, n_both = len(cf_ijk), len(am_ijk), len(both_ijk)
    n_cf_only, n_am_only = n_cf - n_both, n_am - n_both
    assert (n_cf, n_am, n_both) == (1075, 1669, 639), (
        f"active-voxel counts changed from the already-shipped run "
        f"({n_cf},{n_am},{n_both}) != (1075,1669,639) -- inputs drifted, investigate")

    thal_mask = np.asarray(nib.load(THAL_NII).dataobj) > 0
    N = int(thal_mask.sum())
    neither = N - (n_cf_only + n_am_only + n_both)
    log(f"N thalamus voxels={N}  CF active={n_cf}  AM active={n_am}  "
        f"both={n_both}  CF-only={n_cf_only}  AM-only={n_am_only}  neither={neither}")

    # =========================================================================
    # (1) is the conjunction (overlap) bigger than chance?
    # =========================================================================
    table = [[n_both, n_cf_only], [n_am_only, neither]]
    odds_ratio, p_fisher_two = stats.fisher_exact(table, alternative="two-sided")
    _, p_fisher_greater = stats.fisher_exact(table, alternative="greater")
    chi2, p_chi2, dof_chi2, expected = stats.chi2_contingency(table)
    expected_both = float(expected[0][0])
    jaccard = n_both / (n_cf_only + n_am_only + n_both)
    dice = 2 * n_both / (n_cf + n_am)

    log(f"Fisher exact: odds ratio={odds_ratio:.4f}  two-sided p={p_fisher_two:.3e}  "
        f"one-sided(greater) p={p_fisher_greater:.3e}")
    log(f"Chi-square independence: chi2={chi2:.3f} dof={dof_chi2} p={p_chi2:.3e}  "
        f"expected both-active count under independence={expected_both:.1f} (observed={n_both})")
    log(f"Jaccard={jaccard:.4f}  Dice={dice:.4f}")

    # =========================================================================
    # (2) within conjunction voxels: is CF-preferred-freq ~ AM-preferred-rate?
    # =========================================================================
    cf_hz = np.array([r["CF_best_fit_freq_hz_continuous"] for r in conj["rows"]])
    am_hz = np.array([r["AM_best_fit_freq_hz_continuous"] for r in conj["rows"]])
    log_cf, log_am = np.log10(cf_hz), np.log10(am_hz)

    r_pearson, p_pearson = stats.pearsonr(log_cf, log_am)
    rho_spearman, p_spearman = stats.spearmanr(cf_hz, am_hz)
    tau_kendall, p_kendall = stats.kendalltau(cf_hz, am_hz)
    log(f"Pearson r (log10-Hz)={r_pearson:.4f} p={p_pearson:.4f} n={n_both}")
    log(f"Spearman rho={rho_spearman:.4f} p={p_spearman:.4f}")
    log(f"Kendall tau={tau_kendall:.4f} p={p_kendall:.4f}")

    # =========================================================================
    # (3) quadrant analysis: median-split High/Low CF x High/Low AM, within
    #     the conjunction voxels -- do High-CF voxels preferentially pair
    #     with High-AM (or Low-AM), etc.?
    # =========================================================================
    cf_med, am_med = float(np.median(log_cf)), float(np.median(log_am))
    high_cf, high_am = log_cf >= cf_med, log_am >= am_med
    n_hh = int((high_cf & high_am).sum())     # High CF, High AM
    n_hl = int((high_cf & ~high_am).sum())    # High CF, Low AM
    n_lh = int((~high_cf & high_am).sum())    # Low CF, High AM
    n_ll = int((~high_cf & ~high_am).sum())   # Low CF, Low AM
    quad_table = [[n_hh, n_hl], [n_lh, n_ll]]
    quad_chi2, quad_p_chi2, quad_dof, quad_expected = stats.chi2_contingency(quad_table)
    quad_odds, quad_p_fisher = stats.fisher_exact(quad_table)
    log(f"quadrant split (median log-Hz: CF={10**cf_med:.1f} Hz, AM={10**am_med:.2f} Hz): "
        f"High/High={n_hh} High-CF/Low-AM={n_hl} Low-CF/High-AM={n_lh} Low/Low={n_ll}")
    log(f"quadrant chi2={quad_chi2:.3f} dof={quad_dof} p={quad_p_chi2:.4f}  "
        f"Fisher odds ratio={quad_odds:.4f} p={quad_p_fisher:.4f}")

    result = {
        "conjunction_overlap_test": {
            "contingency_table": {"both": n_both, "CF_only": n_cf_only,
                                   "AM_only": n_am_only, "neither": neither,
                                   "N_thalamus_voxels": N},
            "fisher_exact": {"odds_ratio": odds_ratio,
                              "p_two_sided": p_fisher_two,
                              "p_one_sided_greater": p_fisher_greater},
            "chi_square_independence": {"chi2": chi2, "dof": dof_chi2, "p": p_chi2,
                                          "expected_both_under_independence": expected_both},
            "jaccard_index": jaccard,
            "dice_coefficient": dice,
            "caveat": "assumes independent voxel trials; fMRI voxels are spatially "
                      "autocorrelated (this GLM is smoothed), so treat p-values as a "
                      "standard enrichment heuristic, not a spatially-valid test.",
        },
        "cf_am_frequency_correlation_within_conjunction": {
            "n": n_both,
            "pearson_r_log10Hz": r_pearson, "pearson_p": p_pearson,
            "spearman_rho": rho_spearman, "spearman_p": p_spearman,
            "kendall_tau": tau_kendall, "kendall_p": p_kendall,
        },
        "quadrant_analysis_within_conjunction": {
            "median_split_CF_hz": 10 ** cf_med, "median_split_AM_hz": 10 ** am_med,
            "high_CF_high_AM": n_hh, "high_CF_low_AM": n_hl,
            "low_CF_high_AM": n_lh, "low_CF_low_AM": n_ll,
            "chi_square": {"chi2": quad_chi2, "dof": quad_dof, "p": quad_p_chi2},
            "fisher_exact": {"odds_ratio": quad_odds, "p": quad_p_fisher},
        },
    }
    out_path = f"{SG}/sub-01_CFAM_conjunction_SMOOTHED_statistics.json"
    json.dump(result, open(out_path, "w"), indent=2)
    log(f"wrote {out_path}")
    log("DONE")
    return result


if __name__ == "__main__":
    res = main()
    print(json.dumps(res, indent=2))
