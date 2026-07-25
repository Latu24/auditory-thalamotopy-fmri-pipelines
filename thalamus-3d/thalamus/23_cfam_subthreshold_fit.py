"""Thalamus pipeline -- sub-threshold Gaussian-fit preferred frequencies for
the CF-only and AM-only voxels.

The per-modality voxel tables from 18_cfam_conjunction_smoothed.py only
contain a Gaussian-fit preferred frequency for voxels that were significant
in that modality (p<0.01 uncorrected omnibus-F). A CF-only voxel
(significant for CF, not for AM) therefore has no AM preferred-frequency
entry anywhere -- not because the fit can't be done, but because 18's
per-modality loop only ever ran the fit on that modality's own active-voxel
set. Since the Gaussian fit procedure has no inherent "acceptance rate" (it
can be run on any voxel's condition profile to discover its best-fitting
frequency, regardless of whether that profile reached significance), the
natural extension is to run that same fit procedure
(bestfreq_block_no_gate(), identical winsorization convention) on the
CF-only voxels' AM condition-beta profiles (and the AM-only voxels' CF
condition-beta profiles) too. This reports a descriptive "preferred
frequency" for the other dimension without claiming it is a significant
tuning result.

Method (per voxel, per "other" modality):
  1. Re-read that modality's own GLM + .ctr (18's own validated parse_ctr /
     verify_ctr_matches_predictors) -- same as scripts 19/20.
  2. Build the native<-raw inverse index (20's build_raw_index_native,
     round-trip-verified via 20's verify_native_to_raw_roundtrip) to map the
     already-known native (i,j,k) back to that GLM's own (z,x,y).
  3. Pull the q-condition beta profile at that voxel, winsorize it (18/20's
     median +/- 5*MAD convention), and fit (18's bestfreq_block_no_gate).
  4. Cross-check: also recompute that voxel's full-volume omnibus F for the
     "other" modality (19's compute_full_F_native) and assert it is in fact
     <= that modality's own Fcrit -- these voxels are CF-only/AM-only by
     construction (sub-threshold in the other modality), so this must hold;
     if it doesn't, something upstream is inconsistent and this script stops.

Output: sub-01_CFAM_subthreshold_fits_SMOOTHED.json -- one entry per CF-only
voxel (with a new, sub-threshold AM preferred frequency) and one per AM-only
voxel (with a new, sub-threshold CF preferred frequency). Does not touch any
of scripts 18/19/20's own already-shipped files.
"""
import os, json, time, importlib.util
import numpy as np

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

ROOT = PROJECT_ROOT
ANA = f"{ROOT}/derivatives/sub-01/analysis"
SG = f"{ANA}/smoothed_glm"

_spec18 = importlib.util.spec_from_file_location(
    "cfam18", f"{ROOT}/scripts/thalamus/18_cfam_conjunction_smoothed.py")
_bf18 = importlib.util.module_from_spec(_spec18)
_spec18.loader.exec_module(_bf18)

_spec20 = importlib.util.spec_from_file_location(
    "cfam20", f"{ROOT}/scripts/thalamus/20_cfam_beta_at_preferred.py")
_bf20 = importlib.util.module_from_spec(_spec20)
_spec20.loader.exec_module(_bf20)

MODALITIES = _bf18.MODALITIES
CF_HZ, AM_HZ = _bf18.CF_HZ, _bf18.AM_HZ
P_THRESH = _bf18.P_THRESH


def fit_other_modality_at_voxels(other_tag, ijk_list):
    """For a list of native (i,j,k) voxels that are significant in the OTHER
    modality (not other_tag), re-read other_tag's own GLM/.ctr, recompute its
    full native-space F volume (19's method, for the sub-threshold-F
    cross-check), pull + winsorize each voxel's other_tag condition profile,
    and Gaussian-fit it (18's bestfreq_block_no_gate, unmodified)."""
    cfg = MODALITIES[other_tag]
    h, R2, SS, beta, SSXiY, meantc, ARlag = _bf18.glmmod.read_glm(cfg["glm"])
    del SSXiY, ARlag, meantc
    shapeZXY = beta.shape[:3]
    dof = h["Nr time points"] - h["Nr all predictors"]
    assert dof == cfg["dof_expected"], f"[{other_tag}] dof mismatch"
    bbox = {k: int(h[k]) for k in ("XStart", "XEnd", "YStart", "YEnd", "ZStart", "ZEnd")}
    names = [p["Name (custom)"] for p in h["Predictor info"]]
    ctr = _bf18.parse_ctr(cfg["ctr"])
    cols, named = _bf18.verify_ctr_matches_predictors(ctr, names)
    q = len(cols)
    assert q == cfg["q"]
    log(f"[{other_tag}] re-verified .ctr columns {cols[0]}..{cols[-1]} (q={q})")

    ok, _, idx_native = _bf20.verify_native_to_raw_roundtrip(shapeZXY, bbox, f"{other_tag}_subthresh")
    assert ok, f"[{other_tag}] native<-raw round-trip check FAILED -- stopping"

    # ---- Fcrit + full native F volume, for the sub-threshold cross-check ----
    invXX = np.asarray(h["Inverted X'X matrix"], dtype=np.float64)
    Minv = np.linalg.inv(invXX[np.ix_(cols, cols)])
    bc = beta[..., cols].astype(np.float64)
    quad = np.einsum("...i,ij,...j->...", bc, Minv, bc)
    sig2 = SS.astype(np.float64) * (1.0 - R2.astype(np.float64) ** 2) / dof
    with np.errstate(divide="ignore", invalid="ignore"):
        F = quad / (q * sig2)
    F[~np.isfinite(F)] = 0.0
    Fcrit = float(_bf18.stats.f.isf(P_THRESH, q, dof))

    profiles = np.zeros((len(ijk_list), q), dtype=np.float64)
    Fvals = np.zeros(len(ijk_list), dtype=np.float64)
    for row_i, (i, j, k) in enumerate(ijk_list):
        flat = idx_native[i, j, k]
        assert flat >= 0, f"[{other_tag}] voxel ({i},{j},{k}) has no GLM source index (unexpected)"
        z, x, y = np.unravel_index(flat, shapeZXY)
        profiles[row_i] = beta[z, x, y, cols]
        Fvals[row_i] = F[z, x, y]

    n_above_Fcrit = int((Fvals > Fcrit).sum())
    assert n_above_Fcrit == 0, (
        f"[{other_tag}] {n_above_Fcrit} of {len(ijk_list)} voxels are ABOVE this modality's own "
        f"Fcrit -- these voxels should be sub-threshold by construction, inputs are inconsistent")
    log(f"[{other_tag}] sub-threshold cross-check OK: all {len(ijk_list)} voxels have "
        f"F <= Fcrit={Fcrit:.4f} (max F here = {Fvals.max():.4f})")

    med = np.median(profiles, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(profiles - med), axis=1, keepdims=True)
    profiles_w = np.clip(profiles, med - 5.0 * mad, med + 5.0 * mad)
    bf_hz, cond_idx, fit_converged, fit_r2_diag = _bf18.bestfreq_block_no_gate(profiles_w, cfg["hz"])

    return dict(bf_hz=bf_hz, cond_idx=cond_idx, fit_converged=fit_converged,
                fit_r2=fit_r2_diag, F_value=Fvals, Fcrit=Fcrit)


def main():
    log("loading already-shipped voxel tables (CF/AM per-modality + conjunction) ...")
    cf_table = json.load(open(f"{SG}/sub-01_CF_voxeltable_SMOOTHED.json"))
    am_table = json.load(open(f"{SG}/sub-01_AM_voxeltable_SMOOTHED.json"))
    conj = json.load(open(f"{SG}/sub-01_CFAM_conjunction_SMOOTHED.json"))

    both_ijk = set(tuple(r["voxel_ijk_native"]) for r in conj["rows"])
    cf_only_rows = [r for r in cf_table["rows"] if tuple(r["voxel_ijk_native"]) not in both_ijk]
    am_only_rows = [r for r in am_table["rows"] if tuple(r["voxel_ijk_native"]) not in both_ijk]
    assert (len(cf_only_rows), len(am_only_rows)) == (436, 1030), (
        f"CF-only/AM-only counts ({len(cf_only_rows)},{len(am_only_rows)}) != anchors (436,1030)")
    log(f"CF-only n={len(cf_only_rows)}  AM-only n={len(am_only_rows)}")

    # ---- CF-only voxels: fit AM's Gaussian (sub-threshold) ----
    log("fitting AM Gaussian at CF-only voxels (sub-threshold) ...")
    cf_only_ijk = [tuple(r["voxel_ijk_native"]) for r in cf_only_rows]
    am_fit = fit_other_modality_at_voxels("AM", cf_only_ijk)
    am_rate = float(am_fit["fit_converged"].mean())
    log(f"[AM@CF-only] curve_fit converged {int(am_fit['fit_converged'].sum())}/{len(cf_only_ijk)} "
        f"({100*am_rate:.1f}%)")

    # ---- AM-only voxels: fit CF's Gaussian (sub-threshold) ----
    log("fitting CF Gaussian at AM-only voxels (sub-threshold) ...")
    am_only_ijk = [tuple(r["voxel_ijk_native"]) for r in am_only_rows]
    cf_fit = fit_other_modality_at_voxels("CF", am_only_ijk)
    cf_rate = float(cf_fit["fit_converged"].mean())
    log(f"[CF@AM-only] curve_fit converged {int(cf_fit['fit_converged'].sum())}/{len(am_only_ijk)} "
        f"({100*cf_rate:.1f}%)")

    cf_only_out = []
    for row_i, r in enumerate(cf_only_rows):
        cf_only_out.append(dict(
            voxel_ijk_native=r["voxel_ijk_native"],
            world_xyz_mm=r["world_xyz_mm"], hemisphere=r["hemisphere"], in_MGB=r["in_MGB"],
            CF_best_fit_freq_hz_continuous=r["best_fit_freq_hz_continuous"],
            CF_F_value=r["F_value"],
            AM_best_fit_freq_hz_continuous_subthreshold=round(float(am_fit["bf_hz"][row_i]), 2),
            AM_F_value_subthreshold=round(float(am_fit["F_value"][row_i]), 3),
            AM_fit_converged=bool(am_fit["fit_converged"][row_i]),
            AM_fit_r2=(round(float(am_fit["fit_r2"][row_i]), 4)
                       if np.isfinite(am_fit["fit_r2"][row_i]) else None),
        ))
    am_only_out = []
    for row_i, r in enumerate(am_only_rows):
        am_only_out.append(dict(
            voxel_ijk_native=r["voxel_ijk_native"],
            world_xyz_mm=r["world_xyz_mm"], hemisphere=r["hemisphere"], in_MGB=r["in_MGB"],
            AM_best_fit_freq_hz_continuous=r["best_fit_freq_hz_continuous"],
            AM_F_value=r["F_value"],
            CF_best_fit_freq_hz_continuous_subthreshold=round(float(cf_fit["bf_hz"][row_i]), 2),
            CF_F_value_subthreshold=round(float(cf_fit["F_value"][row_i]), 3),
            CF_fit_converged=bool(cf_fit["fit_converged"][row_i]),
            CF_fit_r2=(round(float(cf_fit["fit_r2"][row_i]), 4)
                       if np.isfinite(cf_fit["fit_r2"][row_i]) else None),
        ))

    result = {
        "AM_Fcrit_p01": am_fit["Fcrit"], "CF_Fcrit_p01": cf_fit["Fcrit"],
        "CF_only_n": len(cf_only_out), "AM_only_n": len(am_only_out),
        "AM_fit_convergence_rate_at_CF_only_pct": round(100 * am_rate, 2),
        "CF_fit_convergence_rate_at_AM_only_pct": round(100 * cf_rate, 2),
        "CF_only_rows": cf_only_out,
        "AM_only_rows": am_only_out,
    }
    out_path = f"{SG}/sub-01_CFAM_subthreshold_fits_SMOOTHED.json"
    json.dump(result, open(out_path, "w"), indent=2)
    log(f"wrote {out_path}")
    log("DONE")
    return {k: v for k, v in result.items() if k not in ("CF_only_rows", "AM_only_rows")}


if __name__ == "__main__":
    res = main()
    print(json.dumps(res, indent=2))
