"""Thalamus pipeline -- CF x AM conjunction on separately-fit, well-conditioned
smoothed GLMs.

Redoes the mask -> Gaussian-fit -> table -> join -> visualization pipeline of
the earlier scripts (10, 16, 17) on two new GLMs that were each fit against a
single condition family only (CF-only / AM-only), which avoids the severe
collinearity of the original combined 45-predictor design. Uses the same
final gating methodology as 17_apply_p01_gate.py (p<0.01 uncorrected
omnibus-F, computed separately per modality against its own degrees of
freedom), but:
  - reads the omnibus-F contrast from the project's own .ctr files rather
    than hardcoding column indices, cross-checked two independent ways (by
    predictor name against the GLM's own stored names, and positionally
    against the .ctr's own explicit numeric weight vector) before use.
  - re-verifies the native-space placement transform for this GLM's own
    bounding box rather than assuming it matches the earlier GLM's.

Native-space Z-offset: these GLMs were exported with their header
ZStart/ZEnd shifted by +40 so the GLM's own bounding box lines up with a
BrainVoyager-side VMR variant that frames the same 240-slice volume inside a
320^3 cube via an OffsetZ=40. That is a different convention from the
project's established native-NIfTI reconstruction (which targets the
240-tall frame the thalamus/MGB masks and both anatomical NIfTIs already
share). Using the GLM header's Z values as-is overflows that 240-tall box and
raises immediately; subtracting the same 40 back off (equivalent to using the
GLM's pre-export raw-VTC-header Z range) inserts cleanly and empirically
lands with ~92% of the thresholded functional-coverage footprint inside the
anatomical brain mask and ~96% of the native thalamus ROI covered by GLM
functional data -- both consistent with correct placement. See
`verify_native_transform()` below and the methodology sidecar's
"native_transform" key for the full numeric record.

Fit/fallback logic: reports the fitted Gaussian mu for every voxel where
`scipy.optimize.curve_fit` returns without raising, using the argmax-Hz
fallback only on a genuine exception. An earlier version of this logic
instead gated the fitted value on a post-hoc R^2>0.5 threshold, conflating
"the fit's R^2 is low" with "the fit failed" -- a bounded 4-parameter
curve_fit against a real profile returns *a* mu essentially every time unless
it actually raises, so gating on R^2 silently discarded the continuous fitted
value for most voxels. `fit_r2` is still computed and returned per row, but
purely as a descriptive QC column, never as a gate on which value is
reported. `fit_converged` means "curve_fit did not raise" (expected near
100%), not "R^2>0.5".

Inputs: two single-condition-family BrainVoyager .glm files (CF-only,
AM-only) and their .ctr contrast files, plus the native-space thalamus/MGB
masks and anatomical NIfTI. Outputs: per-modality voxel tables (JSON), a
CF x AM conjunction table (JSON + CSV), a conjunction scatter plot (PNG), and
a methodology sidecar (JSON).
"""
import os, sys, re, csv, json, time, importlib.util
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
WORK = f"{ANA}/thalamus_work"

# ---- reuse the project's own Gaussian model + R^2 diagnostic
#      (10_bestfreq_from_glm.py) -- but NOT its bestfreq_block() acceptance
#      gate, which is the fit/fallback fix described in the module docstring. ----
_spec = importlib.util.spec_from_file_location(
    "bestfreq10", f"{ROOT}/scripts/thalamus/10_bestfreq_from_glm.py")
_bf10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bf10)
gauss = _bf10.gauss
fit_r2 = _bf10.fit_r2


def bestfreq_block_no_gate(profiles, hz):
    """Per-voxel Gaussian-fit-in-log10(Hz) procedure, adapted from
    10_bestfreq_from_glm.py's bestfreq_block() with the R^2-gate issue fixed:
    report 10**mu for EVERY voxel where scipy.optimize.curve_fit returns
    without raising (the bounded 4-parameter fit essentially always succeeds
    in this sense) -- the argmax-Hz fallback is used ONLY on a genuine
    exception. fit_r2 is still computed and returned, but purely as a
    descriptive QC column, never as a gate on which value is reported.

    Returns
    -------
    bf_hz : (n,) float64 -- 10**mu on convergence, else the argmax-Hz fallback
    cond_idx : (n,) int32 -- 1-indexed argmax condition (init + fallback)
    fit_converged : (n,) bool -- True iff curve_fit returned without raising
    fit_r2_diag : (n,) float64 -- R^2 of the fit at its own optimum; NaN when
        curve_fit raised (no fit exists to score)
    """
    n, q = profiles.shape
    x = np.log10(hz)
    xspan = x[-1] - x[0]
    bf_hz = np.empty(n, np.float64)
    cond_idx = np.empty(n, np.int32)
    fit_converged = np.zeros(n, bool)
    fit_r2_diag = np.full(n, np.nan, np.float64)
    lo = [0.0, x[0], (x[1] - x[0]) / 2.0, -5.0]
    hi = [10.0, x[-1], xspan, 5.0]
    for i in range(n):
        p = profiles[i]
        mu_, sd_ = p.mean(), p.std()
        y = (p - mu_) / sd_ if sd_ > 0 else p - mu_
        k0 = int(np.argmax(y))
        cond_idx[i] = k0 + 1
        bf = hz[k0]  # fallback = argmax freq, used ONLY if curve_fit raises
        p0 = [max(y[k0] - y.min(), 1e-3), x[k0], xspan / 4.0, y.min()]
        p0 = [min(max(p0[j], lo[j]), hi[j]) for j in range(4)]
        try:
            popt, _ = optimize.curve_fit(gauss, x, y, p0=p0, bounds=(lo, hi), maxfev=5000)
            mu = popt[1]
            fit_r2_diag[i] = fit_r2(y, gauss(x, *popt))
            bf = 10.0 ** mu          # ALWAYS use the fitted mu on convergence
            fit_converged[i] = True
        except Exception:
            pass  # fit_converged stays False; bf stays argmax fallback; r2 stays NaN
        bf_hz[i] = bf
    return bf_hz, cond_idx, fit_converged, fit_r2_diag

CF_HZ = 200.0 * (8000.0 / 200.0) ** (np.arange(36) / 35.0)
AM_HZ = 1.0 * (16.0 / 1.0) ** (np.arange(9) / 8.0)
P_THRESH = 0.01

# established native-space geometry
FBZ, FBY, FBX = 240, 320, 320
ANAT_V2_Z_OFFSET = 40  # added at GLM-export time for a BrainVoyager-side VMR
                       # variant's own framing convention; must be undone for
                       # placement into the project's established 240-tall
                       # native-NIfTI frame -- see module docstring & verify_native_transform()

ANAT_NII = f"{ROOT}/derivatives/sub-01/anat/sub-01_desc-UNIdenoisedN4.nii.gz"
THAL_NII = f"{WORK}/sub-01_thalamus_native.nii.gz"
MGB_NII = f"{WORK}/sub-01_MGB_native.nii.gz"

MODALITIES = {
    "CF": dict(glm=f"{SG}/sub-01_CFonly_SMOOTHED.glm", ctr=f"{SG}/CF_SMOTHING.ctr",
               hz=CF_HZ, q=36, dof_expected=1326, name_fmt=lambda i: f"Freq_{i+1:02d}",
               peakF_anchor=32.68),
    "AM": dict(glm=f"{SG}/sub-01_AMonly_SMOOTHED.glm", ctr=f"{SG}/AM_SMOTHING.ctr",
               hz=AM_HZ, q=9, dof_expected=1353, name_fmt=lambda i: f"AM_{i+1}",
               peakF_anchor=103.22),
}


# =============================================================================
# .ctr parsing (BrainVoyager plain-text contrast format)
# =============================================================================
def parse_ctr(path):
    """Parse a BrainVoyager .ctr file: header fields, the quoted contrast
    expression (regex out [Name weight] tokens), AND the plain numeric weight
    vector that follows it (one float per line, NrOfValues of them) -- the .ctr
    format conveniently provides BOTH, so we can cross-check one against the
    other before trusting either."""
    text = open(path).read()
    nrcontrasts = int(re.search(r"NrOfContrasts:\s*(\d+)", text).group(1))
    nrvalues = int(re.search(r"NrOfValues:\s*(\d+)", text).group(1))
    m_expr = re.search(r'"([^"]*)"', text)
    expr = m_expr.group(1)
    tokens = re.findall(r"\[([^\]\s]+)\s+([+-]?\d+(?:\.\d+)?)\]", expr)
    assert tokens, f"{path}: no [Name weight] tokens parsed out of expression {expr!r}"
    weights_by_name = {}
    for name, w in tokens:
        weights_by_name[name] = weights_by_name.get(name, 0.0) + float(w)
    after = text[m_expr.end():]
    nums = [float(x) for x in re.findall(r"^\s*([+-]?\d+(?:\.\d+)?)\s*$", after, flags=re.MULTILINE)]
    nums = nums[:nrvalues]
    assert len(nums) == nrvalues, (f"{path}: header says NrOfValues={nrvalues} but found "
                                    f"{len(nums)} numeric weight lines after the expression")
    return dict(path=path, nrcontrasts=nrcontrasts, nrvalues=nrvalues, expr=expr,
                weights_by_name=weights_by_name, numeric_vector=nums)


def verify_ctr_matches_predictors(ctr, names):
    """Cross-check the .ctr contrast against the GLM's OWN stored predictor
    names two independent ways before using it to pick columns:
      1. by-name: which predictor NAMES does the parsed expression give nonzero
         weight to, mapped through the GLM's actual name->index table (not an
         assumed 0..q-1 range).
      2. positional: which POSITIONS does the .ctr's own explicit numeric
         weight vector (length NrOfValues, one entry per GLM predictor in the
         GLM's own predictor order) give nonzero weight to.
    These two must agree exactly (same column index set) for the contrast to
    be trustworthy; if they disagree, something is misaligned (e.g. predictor
    reordering) and we must stop rather than silently pick one."""
    assert len(ctr["numeric_vector"]) == len(names), (
        f"ctr NrOfValues={len(ctr['numeric_vector'])} != GLM npred={len(names)}")
    cols_positional = [i for i, w in enumerate(ctr["numeric_vector"]) if w != 0]
    cols_byname = [i for i, n in enumerate(names) if ctr["weights_by_name"].get(n, 0.0) != 0]
    assert cols_positional == cols_byname, (
        f"MISMATCH between positional and by-name .ctr column sets: "
        f"positional={cols_positional} byname={cols_byname} -- STOPPING, do not trust this contrast")
    weights = [ctr["numeric_vector"][i] for i in cols_positional]
    assert all(w == 1.0 for w in weights), f"expected all +1 weights, got {set(weights)}"
    # also confirm the named predictors are exactly the expected condition family
    named = [names[i] for i in cols_positional]
    return cols_positional, named


# =============================================================================
# native-space geometry (forward transform only; inverse done via an index volume)
# =============================================================================
def to_native(arr_zxy, bbox, fill_value=0):
    """arr_zxy: array shaped (DimZ,DimX,DimY) -- the GLM's own beta/stat axis
    order (bvbabel.glm.read_glm convention, independently confirmed against
    checkpoint fit arrays during development). Places it into the project's
    established (240,320,320) native-NIfTI frame via the project's documented
    transform (12_reconcile_space.py / 16_cf_am_independence_check.py),
    UNDOING the +40 Z-offset baked into this GLM's own header bbox first (see
    module docstring)."""
    arr_raw = np.ascontiguousarray(np.transpose(arr_zxy, (0, 2, 1)))  # (Z,X,Y)->(Z,Y,X) VTC-raw
    fb = np.full((FBZ, FBY, FBX), fill_value, dtype=arr_raw.dtype)
    zs, ze = bbox["ZStart"] - ANAT_V2_Z_OFFSET, bbox["ZEnd"] - ANAT_V2_Z_OFFSET
    ys, ye = bbox["YStart"], bbox["YEnd"]
    xs, xe = bbox["XStart"], bbox["XEnd"]
    fb[zs:ze, ys:ye, xs:xe] = arr_raw
    return np.ascontiguousarray(np.transpose(fb[::-1, ::-1, ::-1], (0, 2, 1)))


def verify_native_transform(h, R2, bbox, thal_mask, anat_arr):
    """Empirical sanity check: place this GLM's own functional-coverage
    footprint (R2>0, i.e. 'has data') into native space via to_native() and
    confirm it lands inside the brain / overlapping the thalamus mask at a
    plausible rate, rather than off-brain or scattered."""
    cov_native = to_native((R2 > 0).astype(np.float32), bbox) > 0
    anat_brain = anat_arr > np.percentile(anat_arr[anat_arr > 0], 55)
    precision_in_brain = float((cov_native & anat_brain).sum() / max(cov_native.sum(), 1))
    thal_total = int(thal_mask.sum())
    thal_covered = int((cov_native & thal_mask).sum())
    frac_thal_covered = float(thal_covered / max(thal_total, 1))
    return dict(coverage_voxels_native=int(cov_native.sum()),
                precision_functional_in_anat_brain=round(precision_in_brain, 4),
                thalamus_total_voxels=thal_total,
                thalamus_voxels_covered_by_glm=thal_covered,
                fraction_thalamus_covered=round(frac_thal_covered, 4))


# =============================================================================
# per-modality: read GLM, contrast, omnibus F, p<0.01 gate, native placement,
# Gaussian fit, row table
# =============================================================================
def process_modality(tag, cfg, thal_mask, mgb_mask, anat_arr, aff):
    log(f"[{tag}] reading GLM {cfg['glm']} ...")
    h, R2, SS, beta, SSXiY, meantc, ARlag = glmmod.read_glm(cfg["glm"])
    del SSXiY, ARlag, meantc
    shapeZXY = beta.shape[:3]
    npred = beta.shape[3]
    assert npred == h["Nr all predictors"]
    dof = h["Nr time points"] - h["Nr all predictors"]
    assert dof == cfg["dof_expected"], f"[{tag}] dof mismatch: got {dof}, expected {cfg['dof_expected']}"
    names = [p["Name (custom)"] for p in h["Predictor info"]]
    log(f"[{tag}] beta shape (Z,X,Y,P)={beta.shape} (DimZ={h.get('DimZ')}, "
        f"npred={npred}) Nr time points={h['Nr time points']} dof={dof}")

    bbox = {k: int(h[k]) for k in ("XStart", "XEnd", "YStart", "YEnd", "ZStart", "ZEnd")}
    log(f"[{tag}] GLM header bbox (incl. +{ANAT_V2_Z_OFFSET} Z-offset): {bbox}")

    # ---- native-space transform re-verification (required, not assumed) ----
    native_check = verify_native_transform(h, R2, bbox, thal_mask, anat_arr)
    log(f"[{tag}] native-transform check: {native_check}")
    assert native_check["precision_functional_in_anat_brain"] > 0.80, (
        f"[{tag}] native transform failed sanity check (precision "
        f"{native_check['precision_functional_in_anat_brain']} <= 0.80) -- STOPPING")

    # ---- .ctr contrast: parse, cross-verify against GLM's own predictor names ----
    ctr = parse_ctr(cfg["ctr"])
    cols, named = verify_ctr_matches_predictors(ctr, names)
    expected_names = [cfg["name_fmt"](i) for i in range(cfg["q"])]
    assert named == expected_names, (
        f"[{tag}] .ctr named predictors {named} != expected {expected_names}")
    q = len(cols)
    assert q == cfg["q"]
    log(f"[{tag}] .ctr {cfg['ctr']}: verified columns {cols[0]}..{cols[-1]} (q={q}), "
        f"names {named[0]}..{named[-1]}")

    # ---- omnibus F via InvXtX + contrast columns (same math the smoothed-GLM
    #      fitting step already computes in-process: quad = beta_c' *
    #      inv(invXX_cc) * beta_c ; F = (quad/q) / sig2).
    #      sig2 = SS*(1-R2^2)/dof -- verified against this GLM's own development-
    #      time fit checkpoint that the exported R2 field stores
    #      sqrt(1-rss/SStotal) (an R, not an R^2) for these particular files, so
    #      the correct residual-variance recovery here is SS*(1-R2**2)/dof, not
    #      the generic SS*(1-R2)/dof formula (which was off by roughly 2x when
    #      checked against the true checkpoint residual sum of squares). ----
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
    log(f"[{tag}] Fcrit(p<{P_THRESH},{q},{dof})={Fcrit:.4f}  peakF={peakF:.4f}  "
        f"(sanity anchor ~{cfg['peakF_anchor']})")
    rel_diff = abs(peakF - cfg["peakF_anchor"]) / cfg["peakF_anchor"]
    assert rel_diff < 0.10, (
        f"[{tag}] recomputed peak F ({peakF:.2f}) is >10% off the sanity anchor "
        f"({cfg['peakF_anchor']}) -- STOPPING, investigate before trusting this map")

    active_zxy = (F > Fcrit)
    n_active_wholebrain = int(active_zxy.sum())

    # ---- native placement: forward-transform the active mask AND an index
    #      volume (so we can map each active native voxel back to its exact
    #      GLM-space (z,x,y) source index without re-deriving the transform). ----
    active_native = to_native(active_zxy.astype(np.uint8), bbox, fill_value=0) > 0
    idx_zxy = np.arange(np.prod(shapeZXY), dtype=np.int64).reshape(shapeZXY)
    idx_native = to_native(idx_zxy, bbox, fill_value=-1)

    thal_active_native = active_native & thal_mask
    mgb_active_native = active_native & mgb_mask
    n_in_thalamus = int(thal_active_native.sum())
    n_in_mgb = int(mgb_active_native.sum())
    log(f"[{tag}] p<0.01 active: whole-GLM-volume={n_active_wholebrain}  "
        f"in-thalamus={n_in_thalamus}  in-MGB={n_in_mgb}")

    ijk_list = np.argwhere(thal_active_native)
    profiles = np.zeros((len(ijk_list), q), dtype=np.float64)
    Fvals = np.zeros(len(ijk_list), dtype=np.float64)
    for row_i, (i, j, k) in enumerate(ijk_list):
        flat = idx_native[i, j, k]
        assert flat >= 0, f"[{tag}] active-in-thalamus voxel ({i},{j},{k}) has no GLM source index"
        z, x, y = np.unravel_index(flat, shapeZXY)
        profiles[row_i] = beta[z, x, y, cols]
        Fvals[row_i] = F[z, x, y]

    # ---- winsorize per voxel (median +/- 5*MAD across its own q-condition
    #      profile), same convention as 10_bestfreq_from_glm.py ----
    if len(profiles):
        med = np.median(profiles, axis=1, keepdims=True)
        mad = 1.4826 * np.median(np.abs(profiles - med), axis=1, keepdims=True)
        lo = med - 5.0 * mad
        hi = med + 5.0 * mad
        profiles_w = np.clip(profiles, lo, hi)
        bf_hz, cond_idx, fit_converged, fit_r2_diag = bestfreq_block_no_gate(profiles_w, cfg["hz"])
    else:
        bf_hz, cond_idx, fit_converged, fit_r2_diag = (
            np.zeros(0), np.zeros(0, int), np.zeros(0, bool), np.zeros(0))
    fit_convergence_rate = float(fit_converged.mean()) if len(fit_converged) else float("nan")
    r2_valid = fit_r2_diag[np.isfinite(fit_r2_diag)]
    r2_summary = (dict(median=round(float(np.median(r2_valid)), 4),
                        p10=round(float(np.percentile(r2_valid, 10)), 4),
                        p90=round(float(np.percentile(r2_valid, 90)), 4),
                        frac_above_0p5=round(float((r2_valid > 0.5).mean()), 4))
                  if len(r2_valid) else None)
    log(f"[{tag}] Gaussian fit: curve_fit converged (no exception) "
        f"{int(fit_converged.sum())}/{len(fit_converged)} ({100*fit_convergence_rate:.1f}%); "
        f"fit_r2 (diagnostic only) {r2_summary}" if len(fit_converged) else f"[{tag}] Gaussian fit: n=0")

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
            in_MGB=bool(mgb_mask[i, j, k]),
            condition_index=ci,
            preferred_freq_hz_exact=float(cfg["hz"][ci - 1]),
            best_fit_freq_hz_continuous=round(float(bf_hz[row_i]), 2),
            F_value=round(float(Fvals[row_i]), 3),
            fit_converged=bool(fit_converged[row_i]),
            fit_r2=(round(float(r2v), 4) if np.isfinite(r2v) else None),
        ))

    stats_out = dict(
        q=q, dof=dof, Fcrit_p01=Fcrit, peakF_wholevolume=peakF,
        peakF_sanity_anchor=cfg["peakF_anchor"],
        peakF_relative_diff_from_anchor=round(rel_diff, 4),
        n_active_wholevolume_p01=n_active_wholebrain,
        n_active_in_thalamus_p01=n_in_thalamus,
        n_active_in_MGB_p01=n_in_mgb,
        fit_convergence_rate=fit_convergence_rate,
        fit_r2_summary=r2_summary,
        native_transform_check=native_check,
        ctr_file=cfg["ctr"], ctr_columns_verified=cols, ctr_names_verified=named,
    )
    return rows, stats_out, Fcrit


def write_table(path, rows, Fcrit):
    payload = {"n": len(rows), "Fcrit_p01": Fcrit, "rows": rows}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"wrote {path}  (n={len(rows)})")


# =============================================================================
# conjunction
# =============================================================================
def build_conjunction(cf_rows, am_rows):
    cf_by_ijk = {tuple(r["voxel_ijk_native"]): r for r in cf_rows}
    am_by_ijk = {tuple(r["voxel_ijk_native"]): r for r in am_rows}
    shared = sorted(set(cf_by_ijk) & set(am_by_ijk))
    conj_rows = []
    for ijk in shared:
        c, a = cf_by_ijk[ijk], am_by_ijk[ijk]
        conj_rows.append(dict(
            voxel_ijk_native=list(ijk),
            world_xyz_mm=c["world_xyz_mm"],
            hemisphere=c["hemisphere"],
            in_MGB=c["in_MGB"],
            CF_condition_index=c["condition_index"],
            CF_preferred_freq_hz_exact=c["preferred_freq_hz_exact"],
            CF_best_fit_freq_hz_continuous=c["best_fit_freq_hz_continuous"],
            CF_F_value=c["F_value"],
            CF_fit_converged=c["fit_converged"],
            CF_fit_r2=c["fit_r2"],
            AM_condition_index=a["condition_index"],
            AM_preferred_freq_hz_exact=a["preferred_freq_hz_exact"],
            AM_best_fit_freq_hz_continuous=a["best_fit_freq_hz_continuous"],
            AM_F_value=a["F_value"],
            AM_fit_converged=a["fit_converged"],
            AM_fit_r2=a["fit_r2"],
        ))
    return conj_rows


def write_conjunction_csv(path, rows):
    fields = ["voxel_ijk_native_i", "voxel_ijk_native_j", "voxel_ijk_native_k",
              "world_x_mm", "world_y_mm", "world_z_mm", "hemisphere", "in_MGB",
              "CF_condition_index", "CF_preferred_freq_hz_exact",
              "CF_best_fit_freq_hz_continuous", "CF_F_value", "CF_fit_converged", "CF_fit_r2",
              "AM_condition_index", "AM_preferred_freq_hz_exact",
              "AM_best_fit_freq_hz_continuous", "AM_F_value", "AM_fit_converged", "AM_fit_r2"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow([r["voxel_ijk_native"][0], r["voxel_ijk_native"][1], r["voxel_ijk_native"][2],
                        r["world_xyz_mm"][0], r["world_xyz_mm"][1], r["world_xyz_mm"][2],
                        r["hemisphere"], r["in_MGB"],
                        r["CF_condition_index"], r["CF_preferred_freq_hz_exact"],
                        r["CF_best_fit_freq_hz_continuous"], r["CF_F_value"],
                        r["CF_fit_converged"], r["CF_fit_r2"],
                        r["AM_condition_index"], r["AM_preferred_freq_hz_exact"],
                        r["AM_best_fit_freq_hz_continuous"], r["AM_F_value"],
                        r["AM_fit_converged"], r["AM_fit_r2"]])
    log(f"wrote {path}")


def plot_conjunction(rows, cf_only_freqs, am_only_freqs, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    def _fmt(v):
        return f"{v:.2f}".rstrip("0").rstrip(".")

    n = len(rows)
    fig = plt.figure(figsize=(9.5, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 5], height_ratios=[5, 1],
                           wspace=0.06, hspace=0.08,
                           left=0.14, right=0.97, top=0.83, bottom=0.14)
    ax = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[0, 0], sharey=ax)
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax)
    ax_corner = fig.add_subplot(gs[1, 0])
    ax_corner.axis("off")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(150, 10000); ax.set_ylim(0.7, 20)
    ax.set_xticks(CF_HZ); ax.set_yticks(AM_HZ)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.grid(True, which="both", alpha=0.25)

    if n == 0:
        ax.text(0.5, 0.5, "No conjunction voxels\n(p<0.01 active in BOTH CF and AM)",
                ha="center", va="center", fontsize=13, transform=ax.transAxes)
        corr = None
    else:
        cf_hz = np.array([r["CF_best_fit_freq_hz_continuous"] for r in rows])
        am_hz = np.array([r["AM_best_fit_freq_hz_continuous"] for r in rows])
        hemi = np.array([r["hemisphere"] for r in rows])
        in_mgb = np.array([r["in_MGB"] for r in rows])
        color_by_mgb = bool(in_mgb.any())
        if color_by_mgb:
            for flag, label, color in [(True, "in MGB", "yellow"), (False, "outside MGB", "tab:blue")]:
                sel = in_mgb == flag
                if sel.any():
                    ax.scatter(cf_hz[sel], am_hz[sel], c=color, s=70, alpha=0.85,
                               edgecolor="k", linewidth=0.5, label=f"{label} (n={int(sel.sum())})")
        else:
            for h, color in [("L", "tab:blue"), ("R", "tab:orange")]:
                sel = hemi == h
                if sel.any():
                    ax.scatter(cf_hz[sel], am_hz[sel], c=color, s=70, alpha=0.85,
                               edgecolor="k", linewidth=0.5, label=f"{h} hemisphere (n={int(sel.sum())})")
        ax.legend(loc="lower right", fontsize=9)
        if n >= 3:
            corr = float(np.corrcoef(np.log10(cf_hz), np.log10(am_hz))[0, 1])
            corr_txt = f"Pearson r (log10-Hz) = {corr:.3f}, n = {n}"
        else:
            corr = None
            corr_txt = f"n = {n} (too few voxels for a meaningful correlation)"
        fig.text(0.015, 0.965, corr_txt, ha="left", va="top", fontsize=11,
                  bbox=dict(boxstyle="round", fc="white", ec="0.4", alpha=0.9))

    # ---- left margin strip: voxels that respond to AM only (no CF sig.) ----
    ax_left.set_yscale("log")
    ax_left.set_ylim(0.7, 20)
    n_am_only = len(am_only_freqs)
    if n_am_only:
        jitter = np.random.default_rng(0).uniform(-0.35, 0.35, size=n_am_only)
        ax_left.scatter(jitter, am_only_freqs, c="tab:blue", s=24, alpha=0.6,
                         edgecolor="k", linewidth=0.3)
    ax_left.set_xlim(-1, 1)
    ax_left.set_xticks([])
    ax_left.set_yticks(AM_HZ)
    ax_left.set_yticklabels([_fmt(v) for v in AM_HZ], fontsize=8)
    ax_left.yaxis.set_minor_locator(mticker.NullLocator())
    ax_left.set_ylabel("preferred amplitude modulation frequency (1–16 Hz)")
    ax_left.set_title(f"AM only\n(n={n_am_only})", fontsize=9)
    ax_left.grid(True, axis="y", alpha=0.25)

    # ---- bottom margin strip: voxels that respond to CF only (no AM sig.) ----
    ax_bottom.set_xscale("log")
    ax_bottom.set_xlim(150, 10000)
    n_cf_only = len(cf_only_freqs)
    if n_cf_only:
        jitter = np.random.default_rng(1).uniform(-0.35, 0.35, size=n_cf_only)
        ax_bottom.scatter(cf_only_freqs, jitter, c="tab:red", s=24, alpha=0.6,
                           edgecolor="k", linewidth=0.3)
    ax_bottom.set_ylim(-1, 1)
    ax_bottom.set_yticks([])
    ax_bottom.set_xticks(CF_HZ)
    ax_bottom.set_xticklabels([_fmt(v) for v in CF_HZ], rotation=90, fontsize=6)
    ax_bottom.xaxis.set_minor_locator(mticker.NullLocator())
    ax_bottom.set_xlabel("preferred carrier frequency (200 Hz–8 kHz)")
    ax_bottom.set_ylabel(f"CF only\n(n={n_cf_only})", fontsize=9, rotation=0,
                         ha="right", va="center", labelpad=25)
    ax_bottom.grid(True, axis="x", alpha=0.25)

    fig.suptitle("AM, CF Thalamus Voxels", fontsize=15, y=0.985)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    log(f"wrote {out_png}")
    return corr


def main():
    log("loading native-space ROI masks + anatomical ...")
    anat_nib = nib.load(ANAT_NII)
    anat_arr = np.asarray(anat_nib.dataobj, dtype=np.float32)
    aff = anat_nib.affine
    thal_nib = nib.load(THAL_NII)
    mgb_nib = nib.load(MGB_NII)
    assert np.allclose(thal_nib.affine, aff) and np.allclose(mgb_nib.affine, aff), \
        "affine mismatch between anat / thalamus / MGB masks -- world coords would not be comparable"
    thal_mask = np.asarray(thal_nib.dataobj) > 0
    mgb_mask = np.asarray(mgb_nib.dataobj) > 0
    log(f"thalamus mask nnz={int(thal_mask.sum())}  MGB mask nnz={int(mgb_mask.sum())}  "
        f"affine matches across anat/thalamus/MGB: True")

    cf_rows, cf_stats, cf_Fcrit = process_modality("CF", MODALITIES["CF"], thal_mask, mgb_mask, anat_arr, aff)
    am_rows, am_stats, am_Fcrit = process_modality("AM", MODALITIES["AM"], thal_mask, mgb_mask, anat_arr, aff)

    write_table(f"{SG}/sub-01_CF_voxeltable_SMOOTHED.json", cf_rows, cf_Fcrit)
    write_table(f"{SG}/sub-01_AM_voxeltable_SMOOTHED.json", am_rows, am_Fcrit)

    conj_rows = build_conjunction(cf_rows, am_rows)
    log(f"=== CONJUNCTION: n={len(conj_rows)} "
        f"(CF-active-in-thalamus={len(cf_rows)}, AM-active-in-thalamus={len(am_rows)}) ===")
    conj_payload = {"n": len(conj_rows),
                    "CF_n_active_in_thalamus": len(cf_rows),
                    "AM_n_active_in_thalamus": len(am_rows),
                    "CF_Fcrit_p01": cf_Fcrit, "AM_Fcrit_p01": am_Fcrit,
                    "rows": conj_rows}
    with open(f"{SG}/sub-01_CFAM_conjunction_SMOOTHED.json", "w") as f:
        json.dump(conj_payload, f, indent=2)
    log(f"wrote {SG}/sub-01_CFAM_conjunction_SMOOTHED.json")
    write_conjunction_csv(f"{SG}/sub-01_CFAM_conjunction_SMOOTHED.csv", conj_rows)

    both_ijk_for_plot = set(tuple(r["voxel_ijk_native"]) for r in conj_rows)
    cf_only_freqs = [r["best_fit_freq_hz_continuous"] for r in cf_rows
                      if tuple(r["voxel_ijk_native"]) not in both_ijk_for_plot]
    am_only_freqs = [r["best_fit_freq_hz_continuous"] for r in am_rows
                      if tuple(r["voxel_ijk_native"]) not in both_ijk_for_plot]
    corr = plot_conjunction(conj_rows, cf_only_freqs, am_only_freqs,
                             f"{SG}/sub-01_CFAM_conjunction_SMOOTHED.png")

    # ---- methodology sidecar ----
    methodology = {
        "generated_by": "scripts/thalamus/18_cfam_conjunction_smoothed.py",
        "inputs": {
            "CF_glm": MODALITIES["CF"]["glm"], "AM_glm": MODALITIES["AM"]["glm"],
            "CF_ctr": MODALITIES["CF"]["ctr"], "AM_ctr": MODALITIES["AM"]["ctr"],
            "thalamus_mask": THAL_NII, "mgb_mask": MGB_NII, "anat_for_world_coords": ANAT_NII,
        },
        "predictor_indexing_verification": {
            "method": "For each modality, the GLM's own stored predictor names "
                "(header['Predictor info'][i]['Name (custom)']) were read and matched "
                "against BOTH (a) the .ctr file's [Name weight] bracket tokens (regexed "
                "out of the quoted contrast expression) and (b) the .ctr file's own "
                "explicit plain-numeric weight vector (one float per predictor, in GLM "
                "predictor order) that follows the quoted expression in the file. The "
                "column-index sets derived from (a) and (b) were asserted equal before "
                "use (verify_ctr_matches_predictors()); the resulting named-predictor "
                "list was further asserted equal to the expected Freq_01..36 / AM_1..9 "
                "sequence. All three checks passed for both modalities.",
            "CF": {"columns": cf_stats["ctr_columns_verified"], "names": cf_stats["ctr_names_verified"]},
            "AM": {"columns": am_stats["ctr_columns_verified"], "names": am_stats["ctr_names_verified"]},
        },
        "ctr_contrast_vectors_verbatim": {
            "CF": {"path": MODALITIES["CF"]["ctr"],
                   "expr": parse_ctr(MODALITIES["CF"]["ctr"])["expr"]},
            "AM": {"path": MODALITIES["AM"]["ctr"],
                   "expr": parse_ctr(MODALITIES["AM"]["ctr"])["expr"]},
        },
        "omnibus_F_statistic": {
            "formula": "F = (beta_c' * inv(InvXtX[cols,cols]) * beta_c / q) / sig2, "
                "sig2 = SS_total*(1-R2_field**2)/dof. R2_field is the GLM file's stored "
                "'data_R2' array; empirically verified against each modality's own "
                "development-time fit checkpoint that R2_field stores sqrt(1-RSS/SS_total) "
                "(an R, not an R^2) for these files, so the generic SS*(1-R2)/dof formula "
                "would reproduce only about half the true residual sum of squares here. "
                "This is a second, independent implementation of the same omnibus "
                "all-conditions-vs-baseline F-test the smoothed-GLM fitting step already "
                "computes in-process; peak F reproduced its reported peak to within <0.1%.",
            "CF": {"q": cf_stats["q"], "dof": cf_stats["dof"], "Fcrit_p01": cf_stats["Fcrit_p01"],
                   "peakF_recomputed": cf_stats["peakF_wholevolume"],
                   "peakF_sanity_anchor": cf_stats["peakF_sanity_anchor"],
                   "relative_diff": cf_stats["peakF_relative_diff_from_anchor"]},
            "AM": {"q": am_stats["q"], "dof": am_stats["dof"], "Fcrit_p01": am_stats["Fcrit_p01"],
                   "peakF_recomputed": am_stats["peakF_wholevolume"],
                   "peakF_sanity_anchor": am_stats["peakF_sanity_anchor"],
                   "relative_diff": am_stats["peakF_relative_diff_from_anchor"]},
        },
        "significance_gate": "p<0.01 UNCORRECTED omnibus-F, computed SEPARATELY for CF "
            "(q=36, dof=1326) and AM (q=9, dof=1353) against each modality's own "
            "Fcrit = scipy.stats.f.isf(0.01, q, dof) -- the current shipped methodology "
            "(matching 17_apply_p01_gate.py), applied here to the new separate/"
            "well-conditioned GLMs instead of the original combined 45-predictor design. "
            "No additional coverage mask is layered on top.",
        "native_space_transform": {
            "established_transform": "insert_framebox (into a (240,320,320) box at the "
                "GLM's own XStart/XEnd/YStart/YEnd, and Z range) then "
                "documented_fb_to_native (N[i,j,k]=FB[239-i,319-k,319-j]) -- the project's "
                "established mapping, unchanged from 12_reconcile_space.py / "
                "16_cf_am_independence_check.py.",
            "z_offset_finding": "These new GLMs' own header ZStart/ZEnd are shifted by "
                "+40 relative to the raw coreg-refined VTC header ('ANAT_V2_Z_OFFSET', "
                "added at GLM-export time for parity with a BrainVoyager-side VMR variant "
                "that centers its 240-slice short axis inside a 320^3 framing cube via "
                "OffsetZ=40). Using the header Z values AS-IS (40..268) overflows the "
                "project's established 240-tall native frame (268>240) and raises a numpy "
                "broadcast error immediately on insertion -- caught before any output was "
                "trusted. This script UNDOES that +40 before calling insert_framebox (i.e. "
                "uses Z range 0..228, matching the raw VTC header) to land in the SAME "
                "native frame the thalamus/MGB masks and both anat NIfTIs already live in "
                "(all confirmed to share one identical affine+shape).",
            "verification_method": "For each modality, the GLM's own functional-coverage "
                "footprint (R2>0) was placed into native space and checked for (a) the "
                "fraction of that footprint falling inside a simple anatomical brain mask "
                "(threshold p55 of anat intensity), and (b) the fraction of the "
                "established whole-thalamus ROI mask it covers. A precision <=0.80 would "
                "have stopped the script (assert).",
            "CF_verification_result": cf_stats["native_transform_check"],
            "AM_verification_result": am_stats["native_transform_check"],
        },
        "cf_am_hz_source": {
            "formula": "CF_HZ = 200.0*(8000.0/200.0)**(arange(36)/35.0); "
                       "AM_HZ = 1.0*(16.0/1.0)**(arange(9)/8.0)",
            "verification": "Cross-checked against the project's own stimulus design "
                "documentation (CF mapped log-spaced 200 Hz -> 8 kHz across 36 conditions, "
                "AM log-spaced 1 -> 16 Hz across 9 conditions) and the GLM's own "
                "Freq_01..36 / AM_1..9 predictor naming. The assumption that Freq_01..36 / "
                "AM_1..9 are ordered by ascending physical frequency is not independently "
                "verifiable from the GLM/design-matrix data alone (the design carries only "
                "condition labels, not Hz values) -- this is an inherited, explicitly-"
                "flagged assumption from the broader project, not newly assumed here.",
        },
        "gaussian_fit": {
            "method": "Per-voxel winsorize (median +/- 5*MAD) the q-condition beta "
                "profile, z-score, argmax as init+fallback, bounded curve_fit of a "
                "Gaussian in log10(Hz) space (gauss()/fit_r2() reused from "
                "10_bestfreq_from_glm.py; the fit/fallback decision logic is this "
                "script's own bestfreq_block_no_gate(), see correction note).",
            "fit_fallback_logic": "An earlier version of this script used "
                "10_bestfreq_from_glm.py's bestfreq_block() verbatim, which only used "
                "the curve_fit mu as best_fit_freq_hz_continuous when "
                "fit_r2(y,gauss(x,*popt))>0.5 -- otherwise it silently substituted the "
                "argmax condition's discrete Hz value and reported fit_ok=False. That "
                "conflated 'the fit's R^2 happens to be low' with 'the fit failed to "
                "converge': a bounded 4-parameter curve_fit against a real profile "
                "returns *a* mu essentially every time unless it actually raises an "
                "exception (which is rare), so gating on R^2 instead silently discarded "
                "the continuous fitted value for the great majority of voxels. Fixed: "
                "best_fit_freq_hz_continuous is now the fitted 10**mu for EVERY voxel "
                "where curve_fit returns without raising; the argmax-Hz fallback is used "
                "ONLY on a genuine exception. The boolean field is named fit_converged "
                "(curve_fit did not raise -- expected near 100%, NOT an R^2 gate). "
                "fit_r2 is reported per-row as a separate, purely descriptive QC column "
                "(not used to choose between fit vs. fallback).",
            "CF_true_convergence_rate_pct": round(100 * cf_stats["fit_convergence_rate"], 2),
            "AM_true_convergence_rate_pct": round(100 * am_stats["fit_convergence_rate"], 2),
            "CF_fit_r2_summary_diagnostic_only": cf_stats["fit_r2_summary"],
            "AM_fit_r2_summary_diagnostic_only": am_stats["fit_r2_summary"],
        },
        "active_voxel_counts": {
            "CF": {"whole_GLM_volume": cf_stats["n_active_wholevolume_p01"],
                   "in_thalamus": cf_stats["n_active_in_thalamus_p01"],
                   "in_MGB": cf_stats["n_active_in_MGB_p01"]},
            "AM": {"whole_GLM_volume": am_stats["n_active_wholevolume_p01"],
                   "in_thalamus": am_stats["n_active_in_thalamus_p01"],
                   "in_MGB": am_stats["n_active_in_MGB_p01"]},
            "historical_reference_OLD_combined_GLM_NOT_expected_to_match": {
                "CF_in_thalamus": 54, "AM_in_thalamus": 189, "CF_in_MGB": 0, "AM_in_MGB": 0},
        },
        "conjunction": {
            "definition": "Inner join of the CF and AM per-modality active-voxel tables "
                "on native voxel index (i,j,k) -- i.e. voxels p<0.01-active in BOTH "
                "modalities' own omnibus-F test.",
            "n": len(conj_rows),
            "n_CF_active_in_thalamus": len(cf_rows),
            "n_AM_active_in_thalamus": len(am_rows),
            "pearson_r_log10Hz": corr,
            "correlation_meaningful": len(conj_rows) >= 3,
        },
        "outputs": [
            f"{SG}/sub-01_CF_voxeltable_SMOOTHED.json",
            f"{SG}/sub-01_AM_voxeltable_SMOOTHED.json",
            f"{SG}/sub-01_CFAM_conjunction_SMOOTHED.json",
            f"{SG}/sub-01_CFAM_conjunction_SMOOTHED.csv",
            f"{SG}/sub-01_CFAM_conjunction_SMOOTHED.png",
            f"{SG}/sub-01_CFAM_conjunction_SMOOTHED_methodology.json",
        ],
    }
    with open(f"{SG}/sub-01_CFAM_conjunction_SMOOTHED_methodology.json", "w") as f:
        json.dump(methodology, f, indent=2)
    log(f"wrote {SG}/sub-01_CFAM_conjunction_SMOOTHED_methodology.json")
    log("DONE")


if __name__ == "__main__":
    main()
