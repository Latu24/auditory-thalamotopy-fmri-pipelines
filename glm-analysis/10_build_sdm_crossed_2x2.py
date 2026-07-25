"""Build SDM files for the CF x AM 2x2 crossed-cell design (4 predictors:
CFlo_AMlo, CFlo_AMhi, CFhi_AMlo, CFhi_AMhi), needed to test a genuine
CF x AM interaction contrast — the marginal 45-predictor design in
07/08/09_*.py cannot express this, since CF and AM there are separate
partitions of the same tone events, not crossed.

Source PRTs: PRTs_join_4/sub01_run-{1-4}_CFxAM_2x2_crossed.prt, built by
matching onset volumes between the original by-frequency and by-AM PRTs
(same underlying events across 4 runs, relabeled by joint cell instead of
marginal CF-or-AM bin; verified 0 mismatches against the existing merged
PRTs).

Two HRF-convolution methods, mirroring 07/08's precedent:
- nilearn's make_first_level_design_matrix (hrf_model='spm') — primary output.
- glmlib.py's hand-rolled two-gamma HRF — cross-check only (correlation
  reported per predictor, not written to its own output here since there's
  no prior crossed-cell SDM to compare against).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix
import bvbabel.sdm as sdmmod
import bvbabel.prt as prtmod
import glmlib as G

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ROOT = PROJECT_ROOT
PRT_DIR = os.path.join(PROJECT_ROOT, "PRTs_join_4")
OUT_DIR = f"{ROOT}/derivatives/sub-01/analysis/sdm"
os.makedirs(OUT_DIR, exist_ok=True)
TR = G.TR_S  # 1.6s

CONDITION_ORDER = ["CFlo_AMlo", "CFlo_AMhi", "CFhi_AMlo", "CFhi_AMhi"]
hrf = G.two_gamma_hrf()


def events_df_from_prt(prt_path):
    _, data = prtmod.read_prt(prt_path)
    rows = []
    for c in data:
        name = c["NameOfCondition"]
        starts = np.asarray(c["Time start"], dtype=float)
        stops = np.asarray(c["Time stop"], dtype=float)
        for s, e in zip(starts, stops):
            onset_s = s * TR
            duration_s = max((e - s + 1) * TR, TR)
            rows.append({"onset": onset_s, "duration": duration_s, "trial_type": name})
    return pd.DataFrame(rows)


def condition_colors(prt_path):
    _, data = prtmod.read_prt(prt_path)
    return {c["NameOfCondition"]: list(int(v) for v in c["Color"]) for c in data}


def run(r):
    prt_path = f"{PRT_DIR}/sub01_run-{r}_CFxAM_2x2_crossed.prt"
    vtc_path = f"{ROOT}/derivatives/sub-01/reg/sub-01_run-{r}_preproc_coreg.vtc"
    h = G.read_vtc_header(vtc_path)
    nvols = h["DimT"]

    events = events_df_from_prt(prt_path)
    assert set(events["trial_type"]) == set(CONDITION_ORDER), \
        f"run-{r}: unexpected condition names {set(events['trial_type'])}"

    # --- primary: nilearn SPM-canonical HRF ---
    dm = make_first_level_design_matrix(
        np.arange(nvols) * TR, events=events, hrf_model="spm",
        drift_model=None, high_pass=None,
    )
    missing = [c for c in CONDITION_ORDER if c not in dm.columns]
    assert not missing, f"run-{r}: nilearn design matrix missing columns: {missing}"
    dm = dm[CONDITION_ORDER]

    colors = condition_colors(prt_path)
    header = {
        "FileVersion": 1,
        "NrOfPredictors": 4,
        "NrOfDataPoints": nvols,
        "IncludesConstant": 0,
        "FirstConfoundPredictor": 5,
    }
    data = []
    for name in CONDITION_ORDER:
        data.append({
            "NameOfPredictor": name,
            "ColorOfPredictor": colors.get(name, [200, 200, 200]),
            "ValuesOfPredictor": dm[name].to_numpy(),
        })

    out_path = f"{OUT_DIR}/sub-01_run-{r}_CFxAM_2x2_crossed_nilearn.sdm"
    sdmmod.write_sdm(out_path, header, data)

    # --- cross-check: hand-rolled two-gamma HRF (same construction as glmlib.py) ---
    X_hand, order_hand = G.build_run_design(prt_path, nvols, hrf,
                                             condition_order=CONDITION_ORDER)
    corrs = []
    for i, name in enumerate(CONDITION_ORDER):
        r_ = np.corrcoef(dm[name].to_numpy(), X_hand[:, i])[0, 1]
        corrs.append(r_)

    # round-trip verify
    h2, d2 = sdmmod.read_sdm(out_path)
    names = [c["NameOfPredictor"] for c in d2]
    assert h2["NrOfPredictors"] == 4 and names == CONDITION_ORDER

    corr_str = ", ".join(f"{n}={c:.4f}" for n, c in zip(CONDITION_ORDER, corrs))
    print(f"run-{r}: wrote {out_path}  (4 predictors x {nvols} timepoints, round-trip OK)")
    print(f"    nilearn vs hand-rolled HRF correlation: {corr_str}")
    return out_path


if __name__ == "__main__":
    outs = [run(r) for r in G.RUNS]
    print("DONE")
    for o in outs:
        print(o)
