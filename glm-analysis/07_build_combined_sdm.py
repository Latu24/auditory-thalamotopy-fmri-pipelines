"""Build combined SDM files (36 carrier-frequency + 9 amplitude-modulation
predictors = 45 columns) per run, for multi-condition GLM setup in
BrainVoyager.

This does not merge or relabel the PRT files. The original by-frequency and
by-AM PRTs are read as-is (read-only, untouched) and their true condition
names (Freq_01..Freq_36, AM_1..AM_9) are kept. It is the SDM — the actual
numeric design matrix the GLM operates on — that needs all 45 predictor
columns together, not the PRT.

Predictors are HRF-convolved box-cars, using the same two-gamma HRF and
construction as the earlier GLM analyses (glmlib.py), for consistency.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
import bvbabel.sdm as sdmmod
import bvbabel.prt as prtmod
import glmlib as G

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ROOT = PROJECT_ROOT
PRT_DIR = os.path.join(PROJECT_ROOT, "PRTs")
OUT_DIR = f"{ROOT}/derivatives/sub-01/analysis/sdm"
os.makedirs(OUT_DIR, exist_ok=True)

hrf = G.two_gamma_hrf()


def condition_colors(prt_path):
    _, data = prtmod.read_prt(prt_path)
    return {c["NameOfCondition"]: list(int(v) for v in c["Color"]) for c in data}


def run(r):
    cf_prt = f"{PRT_DIR}/sub01_run-{r}_tone_events_by_frequency_timestamp.prt"
    am_prt = f"{PRT_DIR}/sub01_run-{r}_tone_events_by_AM_timestamp.prt"

    vtc_path = f"{ROOT}/derivatives/sub-01/reg/sub-01_run-{r}_preproc_coreg.vtc"
    h = G.read_vtc_header(vtc_path)
    nvols = h["DimT"]

    X_cf, order_cf = G.build_run_design(cf_prt, nvols, hrf)
    X_am, order_am = G.build_run_design(am_prt, nvols, hrf)
    assert len(order_cf) == 36, f"expected 36 CF conditions, got {len(order_cf)}"
    assert len(order_am) == 9, f"expected 9 AM conditions, got {len(order_am)}"

    cf_colors = condition_colors(cf_prt)
    am_colors = condition_colors(am_prt)

    header = {
        "FileVersion": 1,
        "NrOfPredictors": 45,
        "NrOfDataPoints": nvols,
        "IncludesConstant": 0,
        "FirstConfoundPredictor": 46,  # all 45 are real task predictors, no confounds in this file
    }

    data = []
    for i, name in enumerate(order_cf):
        data.append({
            "NameOfPredictor": name,
            "ColorOfPredictor": cf_colors.get(name, [200, 200, 200]),
            "ValuesOfPredictor": X_cf[:, i],
        })
    for i, name in enumerate(order_am):
        data.append({
            "NameOfPredictor": name,
            "ColorOfPredictor": am_colors.get(name, [200, 200, 200]),
            "ValuesOfPredictor": X_am[:, i],
        })

    out_path = f"{OUT_DIR}/sub-01_run-{r}_CF-AM_combined.sdm"
    sdmmod.write_sdm(out_path, header, data)

    # round-trip verify
    h2, d2 = sdmmod.read_sdm(out_path)
    names = [c["NameOfPredictor"] for c in d2]
    assert h2["NrOfPredictors"] == 45 and len(d2) == 45
    assert names[:36] == order_cf and names[36:] == order_am
    print(f"run-{r}: wrote {out_path}  (45 predictors x {nvols} timepoints, round-trip OK, "
          f"names[0]={names[0]} names[35]={names[35]} names[36]={names[36]} names[44]={names[44]})")
    return out_path


if __name__ == "__main__":
    outs = [run(r) for r in G.RUNS]
    print("DONE")
    for o in outs:
        print(o)
