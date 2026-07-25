"""Thalamus pipeline -- does "prefers high CF" spatially co-occur with
"prefers high AM" across the thalamus?

Deliberately drops the conjunction-only restriction and the sub-threshold
Gaussian fits from scripts 23/24/25 -- uses only each modality's own real,
significant voxel population (no fabricated/non-significant values):
  - CF median split computed among the CF-significant voxels' own real
    preferred CF frequency (n=1075).
  - AM median split computed among the AM-significant voxels' own real
    preferred AM frequency (n=1669).

This gives four independently-defined spatial voxel sets (High-CF, Low-CF,
High-AM, Low-AM -- each a subset of that modality's own significant voxels,
not requiring the other modality's significance at all), then asks the same
kind of spatial-overlap question as 21_cfam_conjunction_stats.py's original
CF-active x AM-active enrichment test (2x2 contingency table over all N
thalamus voxels, Fisher's exact + chi-square) -- but now for High-CF vs
High-AM (and, for completeness, the other three pairings) instead of for
mere CF-active vs AM-active.

A single 2x2 Fisher/chi-square test on (High-CF, High-AM) already answers
both directions at once ("do high-CF voxels also tend to be high-AM" and its
converse "vice versa") -- it is a symmetric measure of association.

Inputs: sub-01_CF_voxeltable_SMOOTHED.json,
sub-01_AM_voxeltable_SMOOTHED.json, and the native-space thalamus mask.
Output: sub-01_CFAM_highfreq_overlap_SMOOTHED.json.
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


def overlap_test(name_a, ijk_a, name_b, ijk_b, N):
    both = len(ijk_a & ijk_b)
    only_a = len(ijk_a) - both
    only_b = len(ijk_b) - both
    neither = N - both - only_a - only_b
    table = [[both, only_a], [only_b, neither]]
    odds, p_fisher = stats.fisher_exact(table)
    chi2, p_chi2, dof, expected = stats.chi2_contingency(table)
    log(f"[{name_a} x {name_b}] both={both}  {name_a}-only={only_a}  "
        f"{name_b}-only={only_b}  neither={neither}")
    log(f"[{name_a} x {name_b}] Fisher odds={odds:.4f} p={p_fisher:.4f}   "
        f"chi2={chi2:.3f} p={p_chi2:.4f}  (expected both under independence={expected[0][0]:.1f})")
    return dict(both=both, only_a=only_a, only_b=only_b, neither=neither,
                fisher_odds_ratio=odds, fisher_p=p_fisher,
                chi2=chi2, chi2_p=p_chi2, expected_both=float(expected[0][0]))


def main():
    log("loading real (significant-only) per-modality voxel tables ...")
    cf_table = json.load(open(f"{SG}/sub-01_CF_voxeltable_SMOOTHED.json"))
    am_table = json.load(open(f"{SG}/sub-01_AM_voxeltable_SMOOTHED.json"))

    cf_freq = np.array([r["best_fit_freq_hz_continuous"] for r in cf_table["rows"]])
    am_freq = np.array([r["best_fit_freq_hz_continuous"] for r in am_table["rows"]])
    cf_ijk = np.array([tuple(r["voxel_ijk_native"]) for r in cf_table["rows"]])
    am_ijk = np.array([tuple(r["voxel_ijk_native"]) for r in am_table["rows"]])
    assert (len(cf_freq), len(am_freq)) == (1075, 1669)

    cf_med = float(np.median(np.log10(cf_freq)))
    am_med = float(np.median(np.log10(am_freq)))
    log(f"CF median (among 1075 CF-significant voxels) = {10**cf_med:.1f} Hz")
    log(f"AM median (among 1669 AM-significant voxels) = {10**am_med:.2f} Hz")

    high_cf_ijk = set(map(tuple, cf_ijk[np.log10(cf_freq) >= cf_med]))
    low_cf_ijk = set(map(tuple, cf_ijk[np.log10(cf_freq) < cf_med]))
    high_am_ijk = set(map(tuple, am_ijk[np.log10(am_freq) >= am_med]))
    low_am_ijk = set(map(tuple, am_ijk[np.log10(am_freq) < am_med]))
    log(f"High-CF n={len(high_cf_ijk)}  Low-CF n={len(low_cf_ijk)}  "
        f"High-AM n={len(high_am_ijk)}  Low-AM n={len(low_am_ijk)}")

    thal_mask = np.asarray(nib.load(THAL_NII).dataobj) > 0
    N = int(thal_mask.sum())
    log(f"N thalamus voxels = {N}")

    log("--- primary test: does High-CF spatially co-occur with High-AM? ---")
    hh = overlap_test("HighCF", high_cf_ijk, "HighAM", high_am_ijk, N)

    log("--- concordant check: Low-CF x Low-AM ---")
    ll = overlap_test("LowCF", low_cf_ijk, "LowAM", low_am_ijk, N)

    log("--- cross pairings ---")
    hl = overlap_test("HighCF", high_cf_ijk, "LowAM", low_am_ijk, N)
    lh = overlap_test("LowCF", low_cf_ijk, "HighAM", high_am_ijk, N)

    result = {
        "N_thalamus_voxels": N,
        "CF_median_hz_among_significant": 10 ** cf_med,
        "AM_median_hz_among_significant": 10 ** am_med,
        "n_high_CF": len(high_cf_ijk), "n_low_CF": len(low_cf_ijk),
        "n_high_AM": len(high_am_ijk), "n_low_AM": len(low_am_ijk),
        "HighCF_x_HighAM": hh,
        "LowCF_x_LowAM": ll,
        "HighCF_x_LowAM": hl,
        "LowCF_x_HighAM": lh,
        "note": "each test's 'both' overlap voxel count is exactly the number of voxels that "
                "are ALSO in the CF/AM conjunction (they must be significant in both modalities "
                "to have a defined value on both axes) -- this script differs from 24's combined "
                "quadrant analysis only in HOW it computes the High/Low medians (here: each "
                "modality's own real significant population, no sub-threshold fits) and in "
                "testing spatial overlap against the full N-voxel thalamus rather than a shared "
                "per-voxel quadrant table.",
    }
    out_path = f"{SG}/sub-01_CFAM_highfreq_overlap_SMOOTHED.json"
    json.dump(result, open(out_path, "w"), indent=2)
    log(f"wrote {out_path}")
    log("DONE")
    return result


if __name__ == "__main__":
    res = main()
    print(json.dumps(res, indent=2))
