"""Build SEPARATE SDM files per run for carrier-frequency (36 predictors) and
amplitude-modulation (9 predictors) designs, using nilearn's validated
make_first_level_design_matrix for HRF convolution.

Supersedes the earlier combined-45-predictor approach: CF and AM conditions
partition the SAME underlying tone events (verified: sum(36 CF cols) ==
sum(9 AM cols) exactly, corr=1.0), so combining them into one unconstrained
design is severely ill-conditioned (condition number ~1e9-1e10). Each design
kept separate is full rank / well-conditioned on its own (no such redundancy
within a single PRT family), matching what the earlier GLM analyses already
validated (event_family.py ran CF and AM as separate GLMs).

Does not touch the original PRTs (read-only) or any earlier-generated SDM.
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
TR = G.TR_S


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


def build_sdm(prt_path, order, nvols, out_path, label):
    frame_times = np.arange(nvols) * TR
    events = events_df_from_prt(prt_path)
    dm = make_first_level_design_matrix(
        frame_times, events=events, hrf_model="spm",
        drift_model=None, high_pass=None,
    )
    missing = [c for c in order if c not in dm.columns]
    assert not missing, f"{label}: missing columns {missing}"
    dm = dm[order]

    colors = condition_colors(prt_path)
    n = len(order)
    header = {
        "FileVersion": 1,
        "NrOfPredictors": n,
        "NrOfDataPoints": nvols,
        "IncludesConstant": 0,
        "FirstConfoundPredictor": n + 1,
    }
    data = [{"NameOfPredictor": name,
             "ColorOfPredictor": colors.get(name, [200, 200, 200]),
             "ValuesOfPredictor": dm[name].to_numpy()} for name in order]
    sdmmod.write_sdm(out_path, header, data)

    # verify: round trip + rank/condition number
    _, d2 = sdmmod.read_sdm(out_path)
    X = np.column_stack([c["ValuesOfPredictor"] for c in d2])
    rank = np.linalg.matrix_rank(X)
    cond = np.linalg.cond(X)
    print(f"  {label}: wrote {out_path}  shape={X.shape} rank={rank}/{n} cond={cond:.2f}")
    return out_path


def run(r):
    cf_prt = f"{PRT_DIR}/sub01_run-{r}_tone_events_by_frequency_timestamp.prt"
    am_prt = f"{PRT_DIR}/sub01_run-{r}_tone_events_by_AM_timestamp.prt"
    vtc_path = f"{ROOT}/derivatives/sub-01/reg/sub-01_run-{r}_preproc_coreg.vtc"
    nvols = G.read_vtc_header(vtc_path)["DimT"]

    order_cf = [f"Freq_{i:02d}" for i in range(1, 37)]
    order_am = [f"AM_{i}" for i in range(1, 10)]

    print(f"run-{r}:")
    p1 = build_sdm(cf_prt, order_cf, nvols,
                    f"{OUT_DIR}/sub-01_run-{r}_CF_only.sdm", "CF")
    p2 = build_sdm(am_prt, order_am, nvols,
                    f"{OUT_DIR}/sub-01_run-{r}_AM_only.sdm", "AM")
    return p1, p2


if __name__ == "__main__":
    outs = []
    for r in G.RUNS:
        outs += list(run(r))
    print("DONE")
    for o in outs:
        print(o)
