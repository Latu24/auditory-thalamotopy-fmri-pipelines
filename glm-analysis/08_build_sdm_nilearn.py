"""Build combined SDM files (36 CF + 9 AM predictors) using nilearn's
validated make_first_level_design_matrix instead of the hand-rolled HRF
convolution in glmlib.py / 07_build_combined_sdm.py.

Rationale: nilearn's HRF convolution is peer-reviewed, widely used, tested
code (part of the standard Python neuroimaging stack) rather than a
from-scratch implementation — preferable for a design matrix that feeds a
GLM meant to be opened directly in BrainVoyager. hrf_model='spm' is used to
match the canonical double-gamma HRF shape (the closest match to what
BrainVoyager itself approximates); drift_model=None because per-run
confounds (constant + trend) are added separately downstream, so this SDM
should contain task predictors only, matching the original script's header
convention (IncludesConstant=0).

Does not touch the original by-frequency/by-AM PRTs (read-only) or the
07_build_combined_sdm.py output — writes new, separately-named SDM files.
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
PRT_DIR = os.path.join(PROJECT_ROOT, "PRTs")
OUT_DIR = f"{ROOT}/derivatives/sub-01/analysis/sdm"
TR = G.TR_S  # 1.6s


def events_df_from_prt(prt_path):
    """PRT (Volumes resolution) -> nilearn events dataframe (seconds)."""
    _, data = prtmod.read_prt(prt_path)
    rows = []
    for c in data:
        name = c["NameOfCondition"]
        starts = np.asarray(c["Time start"], dtype=float)
        stops = np.asarray(c["Time stop"], dtype=float)
        for s, e in zip(starts, stops):
            onset_s = s * TR
            # PRT stop is inclusive volume index; duration spans through end of that volume
            duration_s = max((e - s + 1) * TR, TR)
            rows.append({"onset": onset_s, "duration": duration_s, "trial_type": name})
    return pd.DataFrame(rows)


def condition_colors(prt_path):
    _, data = prtmod.read_prt(prt_path)
    return {c["NameOfCondition"]: list(int(v) for v in c["Color"]) for c in data}


def run(r):
    cf_prt = f"{PRT_DIR}/sub01_run-{r}_tone_events_by_frequency_timestamp.prt"
    am_prt = f"{PRT_DIR}/sub01_run-{r}_tone_events_by_AM_timestamp.prt"

    vtc_path = f"{ROOT}/derivatives/sub-01/reg/sub-01_run-{r}_preproc_coreg.vtc"
    h = G.read_vtc_header(vtc_path)
    nvols = h["DimT"]
    frame_times = np.arange(nvols) * TR

    cf_events = events_df_from_prt(cf_prt)
    am_events = events_df_from_prt(am_prt)
    all_events = pd.concat([cf_events, am_events], ignore_index=True)

    dm = make_first_level_design_matrix(
        frame_times, events=all_events, hrf_model="spm",
        drift_model=None, high_pass=None,
    )

    order_cf = [f"Freq_{i:02d}" for i in range(1, 37)]
    order_am = [f"AM_{i}" for i in range(1, 10)]
    full_order = order_cf + order_am
    missing = [c for c in full_order if c not in dm.columns]
    assert not missing, f"run-{r}: nilearn design matrix missing columns: {missing}"
    dm = dm[full_order]  # enforce exact column order

    cf_colors = condition_colors(cf_prt)
    am_colors = condition_colors(am_prt)
    colors = {**cf_colors, **am_colors}

    header = {
        "FileVersion": 1,
        "NrOfPredictors": 45,
        "NrOfDataPoints": nvols,
        "IncludesConstant": 0,
        "FirstConfoundPredictor": 46,
    }
    data = []
    for name in full_order:
        data.append({
            "NameOfPredictor": name,
            "ColorOfPredictor": colors.get(name, [200, 200, 200]),
            "ValuesOfPredictor": dm[name].to_numpy(),
        })

    out_path = f"{OUT_DIR}/sub-01_run-{r}_CF-AM_combined_nilearn.sdm"
    sdmmod.write_sdm(out_path, header, data)

    # round-trip + correlation check vs the hand-rolled version
    h2, d2 = sdmmod.read_sdm(out_path)
    assert h2["NrOfPredictors"] == 45
    handrolled_path = f"{OUT_DIR}/sub-01_run-{r}_CF-AM_combined.sdm"
    corrs = []
    if os.path.exists(handrolled_path):
        _, d_old = sdmmod.read_sdm(handrolled_path)
        old_by_name = {c["NameOfPredictor"]: c["ValuesOfPredictor"] for c in d_old}
        for c in d2:
            nm = c["NameOfPredictor"]
            if nm in old_by_name:
                v1, v2 = c["ValuesOfPredictor"], old_by_name[nm]
                r_ = np.corrcoef(v1, v2)[0, 1]
                corrs.append(r_)
    corr_summary = f"min_r={min(corrs):.4f} mean_r={np.mean(corrs):.4f}" if corrs else "n/a"
    print(f"run-{r}: wrote {out_path}  (45 predictors x {nvols} tp)  "
          f"vs hand-rolled SDM correlation: {corr_summary}")
    return out_path


if __name__ == "__main__":
    outs = [run(r) for r in G.RUNS]
    print("DONE")
    for o in outs:
        print(o)
