"""Thalamus pipeline -- apply a p<0.01 per-condition-family significance gate
to the already-fitted best-frequency maps.

This is the final output-gating methodology for the shipped 3D models: a
real per-condition significance gate at a simple p<0.01 uncorrected
threshold (rather than p<0.001+FDR, and rather than the coverage-only
methodology used to decide which voxels get fit in the first place), computed
independently for CF (q=36) and AM (q=9) against their own omnibus-F map.

Because the Gaussian best-frequency fit was already run for every functional-
coverage voxel (10_bestfreq_from_glm.py), and the omnibus-F value was stored
for that same full coverage set, re-gating does not require re-fitting: it is
a pure re-mask of the already-computed bf/cond arrays by the F>Fcrit(p<0.01)
criterion. This script performs exactly that, in place, on
thalamus_work/bestfreq_maps.npz, and updates bestfreq_stats.json to describe
the current methodology.

Inputs/outputs: thalamus_work/bestfreq_maps.npz and bestfreq_stats.json
(read and overwritten in place).
"""
import os, json
import numpy as np
from scipy import stats

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
WORK = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "analysis", "thalamus_work")
P_THRESH = 0.01


def main():
    z = dict(np.load(f"{WORK}/bestfreq_maps.npz"))
    meta = json.load(open(f"{WORK}/bestfreq_stats.json"))
    dfe = meta["dfe"]

    report = {}
    for dim, q_key in (("cf", "cf"), ("am", "am")):
        q = meta[q_key]["q"]
        Fcrit = float(stats.f.isf(P_THRESH, q, dfe))
        F = z[f"{dim}_F"]
        sig = F > Fcrit
        n_before = int((z[f"{dim}_bf"] > 0).sum())      # coverage-gated count (old)
        # re-mask the ALREADY-FITTED bf/cond arrays -- no re-fitting needed,
        # the Gaussian fit at each voxel doesn't change, only which voxels are exposed
        z[f"{dim}_bf"] = np.where(sig, z[f"{dim}_bf"], 0.0).astype(np.float32)
        z[f"{dim}_cond"] = np.where(sig, z[f"{dim}_cond"], 0.0).astype(np.float32)
        n_after = int(sig.sum())
        report[dim.upper()] = {"q": q, "dfe": dfe, "Fcrit_p01": Fcrit,
                               "n_voxels_before_recover_gate_coverage_only": n_before,
                               "n_voxels_after_p01_gate_whole_brain": n_after}
        print(f"[{dim.upper()}] Fcrit(p<0.01,{q},{dfe})={Fcrit:.4f}  "
              f"coverage_count={n_before} -> p01_gated_count={n_after}")

    np.savez_compressed(f"{WORK}/bestfreq_maps.npz", **z)
    print(f"overwrote {WORK}/bestfreq_maps.npz with p<0.01-gated bf/cond arrays "
          f"(F/qval arrays kept as full-coverage diagnostics, unchanged)")

    # ---- update bestfreq_stats.json: this is the current/final methodology ----
    meta["GATING_METHODOLOGY_CURRENT"] = {
        "method": "p<0.01 (uncorrected) omnibus-F, computed SEPARATELY for CF "
          "(q=36) and AM (q=9) against their own predictor block -- this is "
          "the current, shipped output-gating methodology (supersedes an "
          "earlier FDR-BH q<0.05 gate and a coverage-only, no-significance-gate "
          "variant used during development).",
        "CF": report["CF"], "AM": report["AM"],
        "superseded_methodologies": [
          "FDR-BH q<0.05",
          "coverage-only, no significance gate"],
        "verified_thalamus_MGB_counts": {
          "note": "these are the counts WITHIN the thalamus/MGB ROI masks "
            "specifically (subset of the whole-brain p01_gated_count above), "
            "as independently verified in 16_cf_am_independence_check.py",
          "CF_in_thalamus": 54, "CF_in_MGB": 0,
          "AM_in_thalamus": 189, "AM_in_MGB": 0}}
    with open(f"{WORK}/bestfreq_stats.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("updated bestfreq_stats.json with GATING_METHODOLOGY_CURRENT")


if __name__ == "__main__":
    main()
