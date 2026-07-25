"""Cortex pipeline -- per-voxel best-frequency tables for CF and AM within
auditory cortex, mirroring the thalamus pipeline's
18_cfam_conjunction_smoothed.py masking -> omnibus-F gate -> Gaussian-fit ->
table pipeline exactly (same GLMs, same .ctr cross-verification, same p<0.01
uncorrected omnibus-F test computed separately per modality against its own
dof, same winsorize + Gaussian-fit-in-log10(Hz) procedure), but intersecting
active voxels with the native auditory-cortex mask
(01_segment_auditory_cortex.py's output) instead of the thalamus mask.

Outputs match the existing thalamus CSVs' 8-column schema
(voxel_i,voxel_j,voxel_k,world_x_mm,world_y_mm,world_z_mm,hemisphere,
best_frequency_hz) so downstream analyses (median split, chi-square, etc.)
work unchanged:
  sub-01_CF_voxel_bestfreq_table_CORTEX_SMOOTHED.csv
  sub-01_AM_voxel_bestfreq_table_CORTEX_SMOOTHED.csv
plus JSON voxel tables (richer, same fields as the thalamus JSON tables minus
the thalamus-only in_MGB flag) and a methodology sidecar, for parity with the
existing project conventions.

Inputs: two single-condition-family BrainVoyager .glm files (CF-only,
AM-only) and their .ctr contrast files, the native-space auditory-cortex
mask, and the anatomical NIfTI. Outputs: per-modality voxel tables (JSON +
CSV) and a methodology sidecar (JSON).
"""
import os, csv, json, re, time, importlib.util
import numpy as np
import nibabel as nib
import bvbabel.glm as glmmod
from scipy import stats, optimize

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

ROOT = PROJECT_ROOT
ANA = f"{ROOT}/derivatives/sub-01/analysis"
SG = f"{ANA}/smoothed_glm"
THALWORK = f"{ANA}/thalamus_work"
CTXWORK = f"{ANA}/cortex_work"

# ---- reuse the project's own Gaussian model + R^2 diagnostic, and the
#      corrected (no-R^2-gate) fit/fallback logic, exactly as the thalamus
#      pipeline's script 18 does -- imported, not reimplemented. ----
_spec10 = importlib.util.spec_from_file_location(
    "bestfreq10", f"{ROOT}/scripts/thalamus/10_bestfreq_from_glm.py")
_bf10 = importlib.util.module_from_spec(_spec10)
_spec10.loader.exec_module(_bf10)
gauss = _bf10.gauss
fit_r2 = _bf10.fit_r2

_spec18 = importlib.util.spec_from_file_location(
    "conj18", f"{ROOT}/scripts/thalamus/18_cfam_conjunction_smoothed.py")
_c18 = importlib.util.module_from_spec(_spec18)
_spec18.loader.exec_module(_c18)
bestfreq_block_no_gate = _c18.bestfreq_block_no_gate
parse_ctr = _c18.parse_ctr
verify_ctr_matches_predictors = _c18.verify_ctr_matches_predictors
to_native = _c18.to_native

CF_HZ = _c18.CF_HZ
AM_HZ = _c18.AM_HZ
P_THRESH = _c18.P_THRESH
ANAT_V2_Z_OFFSET = _c18.ANAT_V2_Z_OFFSET

ANAT_NII = f"{ROOT}/derivatives/sub-01/anat/sub-01_desc-UNIdenoisedN4.nii.gz"
CTX_NII = f"{CTXWORK}/sub-01_auditorycortex_native.nii.gz"

MODALITIES = {
    "CF": dict(glm=f"{SG}/sub-01_CFonly_SMOOTHED.glm", ctr=f"{SG}/CF_SMOTHING.ctr",
               hz=CF_HZ, q=36, dof_expected=1326, name_fmt=lambda i: f"Freq_{i+1:02d}",
               peakF_anchor=32.68),
    "AM": dict(glm=f"{SG}/sub-01_AMonly_SMOOTHED.glm", ctr=f"{SG}/AM_SMOTHING.ctr",
               hz=AM_HZ, q=9, dof_expected=1353, name_fmt=lambda i: f"AM_{i+1}",
               peakF_anchor=103.22),
}


def verify_native_transform(R2, bbox, ctx_mask, anat_arr):
    cov_native = to_native((R2 > 0).astype(np.float32), bbox) > 0
    anat_brain = anat_arr > np.percentile(anat_arr[anat_arr > 0], 55)
    precision_in_brain = float((cov_native & anat_brain).sum() / max(cov_native.sum(), 1))
    ctx_total = int(ctx_mask.sum())
    ctx_covered = int((cov_native & ctx_mask).sum())
    return dict(coverage_voxels_native=int(cov_native.sum()),
                precision_functional_in_anat_brain=round(precision_in_brain, 4),
                auditory_cortex_total_voxels=ctx_total,
                auditory_cortex_voxels_covered_by_glm=ctx_covered,
                fraction_auditory_cortex_covered=round(ctx_covered / max(ctx_total, 1), 4))


def process_modality(tag, cfg, ctx_mask, anat_arr, aff):
    log(f"[{tag}] reading GLM {cfg['glm']} ...")
    h, R2, SS, beta, SSXiY, meantc, ARlag = glmmod.read_glm(cfg["glm"])
    del SSXiY, ARlag, meantc
    shapeZXY = beta.shape[:3]
    npred = beta.shape[3]
    assert npred == h["Nr all predictors"]
    dof = h["Nr time points"] - h["Nr all predictors"]
    assert dof == cfg["dof_expected"], f"[{tag}] dof mismatch: got {dof}, expected {cfg['dof_expected']}"
    names = [p["Name (custom)"] for p in h["Predictor info"]]
    log(f"[{tag}] beta shape (Z,X,Y,P)={beta.shape} Nr time points={h['Nr time points']} dof={dof}")

    bbox = {k: int(h[k]) for k in ("XStart", "XEnd", "YStart", "YEnd", "ZStart", "ZEnd")}

    native_check = verify_native_transform(R2, bbox, ctx_mask, anat_arr)
    log(f"[{tag}] native-transform check: {native_check}")
    assert native_check["precision_functional_in_anat_brain"] > 0.80, (
        f"[{tag}] native transform failed sanity check -- STOPPING")

    ctr = parse_ctr(cfg["ctr"])
    cols, named = verify_ctr_matches_predictors(ctr, names)
    expected_names = [cfg["name_fmt"](i) for i in range(cfg["q"])]
    assert named == expected_names, (
        f"[{tag}] .ctr named predictors {named} != expected {expected_names}")
    q = len(cols)
    log(f"[{tag}] .ctr verified columns {cols[0]}..{cols[-1]} (q={q})")

    invXX = np.asarray(h["Inverted X'X matrix"], dtype=np.float64)
    Minv = np.linalg.inv(invXX[np.ix_(cols, cols)])
    bc = beta[..., cols].astype(np.float64)
    quad = np.einsum("...i,ij,...j->...", bc, Minv, bc)
    sig2 = SS.astype(np.float64) * (1.0 - R2.astype(np.float64) ** 2) / dof
    with np.errstate(divide="ignore", invalid="ignore"):
        F = quad / (q * sig2)
    F[~np.isfinite(F)] = 0.0
    Fcrit = float(stats.f.isf(P_THRESH, q, dof))
    peakF = float(F.max())
    rel_diff = abs(peakF - cfg["peakF_anchor"]) / cfg["peakF_anchor"]
    log(f"[{tag}] Fcrit(p<{P_THRESH},{q},{dof})={Fcrit:.4f}  peakF={peakF:.4f}  "
        f"(sanity anchor ~{cfg['peakF_anchor']}, rel_diff={rel_diff:.4f})")
    assert rel_diff < 0.10, (
        f"[{tag}] recomputed peak F ({peakF:.2f}) is >10% off the sanity anchor -- STOPPING")

    active_zxy = (F > Fcrit)
    n_active_wholebrain = int(active_zxy.sum())

    active_native = to_native(active_zxy.astype(np.uint8), bbox, fill_value=0) > 0
    idx_zxy = np.arange(np.prod(shapeZXY), dtype=np.int64).reshape(shapeZXY)
    idx_native = to_native(idx_zxy, bbox, fill_value=-1)

    ctx_active_native = active_native & ctx_mask
    n_in_ctx = int(ctx_active_native.sum())
    log(f"[{tag}] p<0.01 active: whole-GLM-volume={n_active_wholebrain}  in-auditory-cortex={n_in_ctx}")

    ijk_list = np.argwhere(ctx_active_native)
    profiles = np.zeros((len(ijk_list), q), dtype=np.float64)
    Fvals = np.zeros(len(ijk_list), dtype=np.float64)
    for row_i, (i, j, k) in enumerate(ijk_list):
        flat = idx_native[i, j, k]
        assert flat >= 0, f"[{tag}] active-in-cortex voxel ({i},{j},{k}) has no GLM source index"
        z, x, y = np.unravel_index(flat, shapeZXY)
        profiles[row_i] = beta[z, x, y, cols]
        Fvals[row_i] = F[z, x, y]

    if len(profiles):
        med = np.median(profiles, axis=1, keepdims=True)
        mad = 1.4826 * np.median(np.abs(profiles - med), axis=1, keepdims=True)
        lo_c = med - 5.0 * mad
        hi_c = med + 5.0 * mad
        profiles_w = np.clip(profiles, lo_c, hi_c)
        bf_hz, cond_idx, fit_converged, fit_r2_diag = bestfreq_block_no_gate(profiles_w, cfg["hz"])
    else:
        bf_hz, cond_idx, fit_converged, fit_r2_diag = (
            np.zeros(0), np.zeros(0, int), np.zeros(0, bool), np.zeros(0))
    fit_convergence_rate = float(fit_converged.mean()) if len(fit_converged) else float("nan")
    log(f"[{tag}] Gaussian fit: curve_fit converged {int(fit_converged.sum())}/{len(fit_converged)} "
        f"({100*fit_convergence_rate:.1f}%)" if len(fit_converged) else f"[{tag}] Gaussian fit: n=0")

    rows = []
    for row_i, (i, j, k) in enumerate(ijk_list):
        i, j, k = int(i), int(j), int(k)
        world = aff @ np.array([i, j, k, 1.0])
        x, y, z_mm = float(world[0]), float(world[1]), float(world[2])
        hemi = "L" if x < 0 else "R"
        ci = int(cond_idx[row_i])
        r2v = fit_r2_diag[row_i]
        rows.append(dict(
            voxel_ijk_native=[i, j, k],
            world_xyz_mm=[round(x, 1), round(y, 1), round(z_mm, 1)],
            hemisphere=hemi,
            condition_index=ci,
            preferred_freq_hz_exact=float(cfg["hz"][ci - 1]),
            best_fit_freq_hz_continuous=round(float(bf_hz[row_i]), 2),
            F_value=round(float(Fvals[row_i]), 3),
            fit_converged=bool(fit_converged[row_i]),
            fit_r2=(round(float(r2v), 4) if np.isfinite(r2v) else None),
        ))

    stats_out = dict(
        q=q, dof=dof, Fcrit_p01=Fcrit, peakF_wholevolume=peakF,
        peakF_sanity_anchor=cfg["peakF_anchor"], peakF_relative_diff_from_anchor=round(rel_diff, 4),
        n_active_wholevolume_p01=n_active_wholebrain, n_active_in_auditory_cortex_p01=n_in_ctx,
        fit_convergence_rate=fit_convergence_rate,
        native_transform_check=native_check,
        ctr_file=cfg["ctr"], ctr_columns_verified=cols, ctr_names_verified=named,
    )
    return rows, stats_out, Fcrit


def write_json_table(path, rows, Fcrit):
    payload = {"n": len(rows), "Fcrit_p01": Fcrit, "rows": rows}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"wrote {path}  (n={len(rows)})")


def write_bestfreq_csv(path, rows):
    """Exact same 8-column schema as the existing thalamus CSVs."""
    fields = ["voxel_i", "voxel_j", "voxel_k", "world_x_mm", "world_y_mm",
              "world_z_mm", "hemisphere", "best_frequency_hz"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow([r["voxel_ijk_native"][0], r["voxel_ijk_native"][1], r["voxel_ijk_native"][2],
                        r["world_xyz_mm"][0], r["world_xyz_mm"][1], r["world_xyz_mm"][2],
                        r["hemisphere"], r["best_fit_freq_hz_continuous"]])
    log(f"wrote {path}  (n={len(rows)})")


def main():
    log("loading native-space auditory-cortex mask + anatomical ...")
    anat_nib = nib.load(ANAT_NII)
    anat_arr = np.asarray(anat_nib.dataobj, dtype=np.float32)
    aff = anat_nib.affine
    ctx_nib = nib.load(CTX_NII)
    assert np.allclose(ctx_nib.affine, aff), "affine mismatch between anat / auditory-cortex mask"
    ctx_mask = np.asarray(ctx_nib.dataobj) > 0
    log(f"auditory-cortex mask nnz={int(ctx_mask.sum())}")

    cf_rows, cf_stats, cf_Fcrit = process_modality("CF", MODALITIES["CF"], ctx_mask, anat_arr, aff)
    am_rows, am_stats, am_Fcrit = process_modality("AM", MODALITIES["AM"], ctx_mask, anat_arr, aff)

    write_json_table(f"{SG}/sub-01_CF_voxeltable_CORTEX_SMOOTHED.json", cf_rows, cf_Fcrit)
    write_json_table(f"{SG}/sub-01_AM_voxeltable_CORTEX_SMOOTHED.json", am_rows, am_Fcrit)

    write_bestfreq_csv(f"{SG}/sub-01_CF_voxel_bestfreq_table_CORTEX_SMOOTHED.csv", cf_rows)
    write_bestfreq_csv(f"{SG}/sub-01_AM_voxel_bestfreq_table_CORTEX_SMOOTHED.csv", am_rows)

    methodology = {
        "generated_by": "scripts/cortex/02_bestfreq_tables_cortex.py",
        "mirrors": "scripts/thalamus/18_cfam_conjunction_smoothed.py (identical GLM read, "
            ".ctr cross-verification, omnibus-F gate, native-space transform, and "
            "Gaussian-fit-in-log10(Hz) logic -- imported, not reimplemented), with the "
            "thalamus mask swapped for the native auditory-cortex mask built by "
            "scripts/cortex/01_segment_auditory_cortex.py.",
        "roi_definition": "Harvard-Oxford cortical maxprob atlas, labels for Heschl's Gyrus + "
            "Planum Temporale + Planum Polare + Superior Temporal Gyrus (anterior + "
            "posterior divisions), warped to native space via FSL MNI152_T1_1mm -> native "
            "SyNRA (same registration recipe as the thalamus pipeline's Reg B, re-run "
            "fresh for the cortical mask).",
        "inputs": {
            "CF_glm": MODALITIES["CF"]["glm"], "AM_glm": MODALITIES["AM"]["glm"],
            "CF_ctr": MODALITIES["CF"]["ctr"], "AM_ctr": MODALITIES["AM"]["ctr"],
            "auditory_cortex_mask": CTX_NII, "anat_for_world_coords": ANAT_NII,
        },
        "significance_gate": "p<0.01 UNCORRECTED omnibus-F, computed SEPARATELY for CF "
            "(q=36, dof=1326) and AM (q=9, dof=1353) -- identical gate to the thalamus "
            "tables, applied here to the same GLMs restricted to the auditory-cortex mask.",
        "CF": {"n_active_in_auditory_cortex": len(cf_rows), **{k: v for k, v in cf_stats.items()
                                                                 if k not in ("ctr_columns_verified",)}},
        "AM": {"n_active_in_auditory_cortex": len(am_rows), **{k: v for k, v in am_stats.items()
                                                                 if k not in ("ctr_columns_verified",)}},
        "outputs": [
            f"{SG}/sub-01_CF_voxeltable_CORTEX_SMOOTHED.json",
            f"{SG}/sub-01_AM_voxeltable_CORTEX_SMOOTHED.json",
            f"{SG}/sub-01_CF_voxel_bestfreq_table_CORTEX_SMOOTHED.csv",
            f"{SG}/sub-01_AM_voxel_bestfreq_table_CORTEX_SMOOTHED.csv",
        ],
    }
    with open(f"{SG}/sub-01_CFAM_cortex_methodology.json", "w") as f:
        json.dump(methodology, f, indent=2)
    log(f"wrote {SG}/sub-01_CFAM_cortex_methodology.json")
    log("DONE")


if __name__ == "__main__":
    main()
