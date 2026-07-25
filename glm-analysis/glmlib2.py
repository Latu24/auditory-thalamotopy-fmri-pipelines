"""Refined-VTC / SDM-based / ridge-capable GLM support.

This module is additive to `glmlib.py` — it imports and reuses that
module's low-level primitives (VTC header/memmap reading, AR(2) whitening,
VMP writing) but does not modify glmlib.py or any of the earlier driver
scripts, so those deliverables remain fully reproducible untouched.

Extensions in this module:
- Predictors are read directly from precomputed BrainVoyager SDM files
  (built via nilearn's make_first_level_design_matrix, hrf_model="spm") via
  bvbabel.sdm, instead of rebuilding box-car+HRF convolution from scratch —
  this guarantees the exact predictors already validated elsewhere in the
  pipeline (condition numbers checked at build time).
- The GLM can run on refined VTCs (larger bounding box, additional
  nonlinear registration refinement on top of distortion correction and
  rigid coregistration).
- Per-condition BETA sub-maps (one per condition, raw parameter estimate)
  are produced in addition to the omnibus-F / best-condition summary maps.
- Ridge (L2-penalized) least squares with a quick GCV grid search is
  available for designs that are severely ill-conditioned — e.g. a combined
  carrier-frequency + amplitude-modulation design where both partitions
  describe the same underlying tone events, producing near-exact linear
  dependency between the two predictor blocks.
"""
import os
import numpy as np
import bvbabel.sdm

import glmlib as G  # reused read-only for shared low-level primitives


# ---------------------------------------------------------------------------
# SDM-based predictor construction
# ---------------------------------------------------------------------------
def read_sdm_predictors(sdm_path, condition_order=None):
    """Return (X, names, nvols) from a BrainVoyager SDM file."""
    h, d = bvbabel.sdm.read_sdm(sdm_path)
    by_name = {c["NameOfPredictor"]: c["ValuesOfPredictor"] for c in d}
    names = condition_order or [c["NameOfPredictor"] for c in d]
    X = np.column_stack([by_name[n] for n in names])
    return X, names, h["NrOfDataPoints"]


def combined_design_from_sdms(cf_sdm_path, am_sdm_path, cf_order, am_order):
    """Concatenate CF (36) + AM (9) predictor columns -> 45-column design,
    for one run. Both SDMs describe the SAME underlying tone events
    (verified: identical row-sum), just grouped two different ways."""
    Xcf, _, nv1 = read_sdm_predictors(cf_sdm_path, cf_order)
    Xam, _, nv2 = read_sdm_predictors(am_sdm_path, am_order)
    assert nv1 == nv2, f"CF/AM SDM volume-count mismatch: {nv1} vs {nv2}"
    Xcomb = np.column_stack([Xcf, Xam])
    return Xcomb, cf_order + am_order, nv1


# ---------------------------------------------------------------------------
# Ridge-capable, memory-safe, slab-wise-accumulating multi-run GLM
# ---------------------------------------------------------------------------
def _run_slabs(dimz, zchunk):
    z = 0
    while z < dimz:
        yield z, min(z + zchunk, dimz)
        z += zchunk


def prep_multirun_design(vtc_paths, run_task_designs, log=print,
                         ar_sample=4000, mask_thresh=None):
    """Phase 1: build the deterministic multi-run design, estimate global
    AR(2), whiten it, and gather a voxel sample -- everything needed to run
    a quick GCV lambda search WITHOUT touching the full per-voxel
    accumulation. VTC headers/dims are read fresh from disk (refined VTCs
    can have a different, larger bounding box than the original ones).
    """
    headers = [G.read_vtc_header(p) for p in vtc_paths]
    h0 = headers[0]
    DimZ, DimY, DimX = h0["DimZ"], h0["DimY"], h0["DimX"]
    for h in headers[1:]:
        assert (h["DimZ"], h["DimY"], h["DimX"]) == (DimZ, DimY, DimX), \
            "refined VTCs do not share an identical bounding box"
    n_cond = run_task_designs[0].shape[1]
    nrun = len(vtc_paths)
    n_conf = 2  # const + linear per run

    run_nvols = [d.shape[0] for d in run_task_designs]
    Ttotal = sum(run_nvols)
    P = n_cond + nrun * n_conf
    X = np.zeros((Ttotal, P))
    row = 0
    for r in range(nrun):
        nv = run_nvols[r]
        X[row:row + nv, :n_cond] = run_task_designs[r]
        c0 = n_cond + r * n_conf
        X[row:row + nv, c0:c0 + n_conf] = G.confounds(nv)
        row += nv
    run_row_bounds = np.cumsum([0] + run_nvols)

    # ---- global AR(2) from a voxel sample (plain pinv OLS: fitted values /
    # residuals from a minimum-norm solution are correct regardless of
    # collinearity, so this step needs no ridge even for the combined design)
    log("  estimating global AR(2) from voxel sample ...")
    rng = np.random.default_rng(42)
    mm0 = G.vtc_memmap(vtc_paths[0], headers[0])
    tmean = np.asarray(mm0[:, :, :, ::8].mean(-1), dtype=np.float32)
    if mask_thresh is None:
        pos = tmean[tmean > 0]
        mask_thresh = float(np.percentile(pos, 40)) if pos.size else 0.0
    mask = tmean > mask_thresh
    vox_idx = np.argwhere(mask)
    if vox_idx.shape[0] > ar_sample:
        sel = rng.choice(vox_idx.shape[0], ar_sample, replace=False)
        vox_idx = vox_idx[sel]
    Ysamp = np.zeros((Ttotal, vox_idx.shape[0]), dtype=np.float64)
    for r in range(nrun):
        mm = G.vtc_memmap(vtc_paths[r], headers[r])
        a, b = run_row_bounds[r], run_row_bounds[r + 1]
        Ysamp[a:b] = np.asarray(
            mm[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2], :].T,
            dtype=np.float64)
        del mm
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta_s = XtX_inv @ (X.T @ Ysamp)
    resid = Ysamp - X @ beta_s
    phis = []
    for r in range(nrun):
        a, b = run_row_bounds[r], run_row_bounds[r + 1]
        phis.append(G.estimate_ar2(resid[a:b]))
    phi1 = float(np.mean([p[0] for p in phis]))
    phi2 = float(np.mean([p[1] for p in phis]))
    log(f"  global AR(2): phi1={phi1:.4f} phi2={phi2:.4f} "
        f"(per-run {[(round(p[0],3),round(p[1],3)) for p in phis]})")

    # ---- whiten design block-wise ----
    Xw = X.copy()
    for r in range(nrun):
        a, b = run_row_bounds[r], run_row_bounds[r + 1]
        Xw[a:b] = G.ar2_whiten(X[a:b], phi1, phi2)

    return {
        "vtc_paths": vtc_paths, "headers": headers,
        "DimZ": DimZ, "DimY": DimY, "DimX": DimX,
        "n_cond": n_cond, "nrun": nrun, "n_conf": n_conf, "P": P,
        "run_row_bounds": run_row_bounds, "Ttotal": Ttotal,
        "Xw": Xw, "Ysamp": Ysamp, "phi": (phi1, phi2),
        "mask": mask, "mask_thresh": mask_thresh,
    }


def fit_from_prep(prep, log=print, zchunk=6, ridge_lambda=0.0,
                  accum_dtype=np.float32):
    """Phase 2: given a prep dict from prep_multirun_design, run the
    memory-safe slab-wise accumulation and final (OLS or ridge) solve."""
    vtc_paths = prep["vtc_paths"]; headers = prep["headers"]
    DimZ, DimY, DimX = prep["DimZ"], prep["DimY"], prep["DimX"]
    n_cond, nrun, n_conf, P = (prep["n_cond"], prep["nrun"], prep["n_conf"],
                               prep["P"])
    run_row_bounds = prep["run_row_bounds"]
    Xw = prep["Xw"]
    phi1, phi2 = prep["phi"]
    mask = prep["mask"]
    mask_thresh = prep["mask_thresh"]
    Ttotal = prep["Ttotal"]

    # ---- optional column scaling for ridge (task columns only) ----
    col_scale = np.ones(P)
    if ridge_lambda > 0:
        task_norms = np.linalg.norm(Xw[:, :n_cond], axis=0)
        task_norms[task_norms == 0] = 1.0
        col_scale[:n_cond] = task_norms
    Xws = Xw / col_scale[None, :]

    XtXws = Xws.T @ Xws
    D = np.zeros((P, P))
    if ridge_lambda > 0:
        D[np.arange(n_cond), np.arange(n_cond)] = 1.0
    XtXws_reg = XtXws + ridge_lambda * D
    XtXws_reg_inv = np.linalg.pinv(XtXws_reg)

    # ---- accumulate XtY / sumYsq slab-wise, one run-slab at a time ----
    log(f"  fitting voxel-wise (ridge_lambda={ridge_lambda:g}, "
        f"accum_dtype={np.dtype(accum_dtype).name}) ...")
    Nvox = DimZ * DimY * DimX
    XtY = np.zeros((P, Nvox), dtype=accum_dtype)
    sumYsq = np.zeros(Nvox, dtype=accum_dtype)
    for r in range(nrun):
        mm = G.vtc_memmap(vtc_paths[r], headers[r])
        a, b = run_row_bounds[r], run_row_bounds[r + 1]
        Xwr = Xws[a:b]
        for z0, z1 in _run_slabs(DimZ, zchunk):
            slab = np.asarray(mm[z0:z1], dtype=np.float32)
            zc = z1 - z0
            Y = slab.reshape(-1, slab.shape[-1]).T
            del slab
            Yw = G.ar2_whiten(Y, phi1, phi2)
            del Y
            off = z0 * DimY * DimX
            n_sl = zc * DimY * DimX
            XtY[:, off:off + n_sl] += (Xwr.T @ Yw).astype(accum_dtype)
            sumYsq[off:off + n_sl] += np.einsum(
                "tn,tn->n", Yw, Yw).astype(accum_dtype)
            del Yw
        del mm
        log(f"    run {r + 1} accumulated")

    betas_scaled = (XtXws_reg_inv.astype(accum_dtype) @ XtY)   # (P, Nvox)
    rss = sumYsq.astype(np.float64) - np.einsum(
        "pn,pn->n", betas_scaled.astype(np.float64),
        XtY.astype(np.float64))
    rss = np.clip(rss, 0, None)
    # un-scale betas back to original predictor units
    betas = (betas_scaled.astype(np.float64) / col_scale[:, None])

    # effective residual dof: for OLS this is Ttotal-P; for ridge the
    # classical dof formula does not apply cleanly to the penalized
    # coefficients, so it is reported only for informational purposes
    # where ridge_lambda>0 (see driver-script docstrings for the caveat).
    if ridge_lambda > 0:
        eff_df = float(np.trace(XtXws_reg_inv @ XtXws))
        dof = Ttotal - eff_df
    else:
        dof = Ttotal - P

    shape3 = (DimZ, DimY, DimX)
    return {
        "betas": betas.reshape((P,) + shape3),
        "rss": rss.reshape(shape3),
        "dof": dof,
        "XtXws_reg_inv": XtXws_reg_inv,
        "col_scale": col_scale,
        "P": P, "n_cond": n_cond, "nrun": nrun, "n_conf": n_conf,
        "ridge_lambda": ridge_lambda,
        "phi": (phi1, phi2),
        "mask": mask,
        "mask_thresh": mask_thresh,
        "headers": headers,
        "Ttotal": Ttotal,
        "shape3": shape3,
    }


def fit_multirun_glm_v2(vtc_paths, run_task_designs, log=print, zchunk=6,
                        ar_sample=4000, mask_thresh=None,
                        ridge_lambda=0.0, accum_dtype=np.float32):
    """Convenience one-shot wrapper: prep_multirun_design + fit_from_prep.
    Used by the well-conditioned CF-only / AM-only drivers (ridge_lambda=0).
    The combined (ill-conditioned) driver calls the two phases separately so
    it can run a GCV lambda search on the prep before the full voxel-wise
    accumulation."""
    prep = prep_multirun_design(vtc_paths, run_task_designs, log=log,
                                ar_sample=ar_sample, mask_thresh=mask_thresh)
    return fit_from_prep(prep, log=log, zchunk=zchunk,
                         ridge_lambda=ridge_lambda, accum_dtype=accum_dtype)


def t_contrast_v2(fit, c):
    """t-map for contrast vector c. Only statistically valid when
    ridge_lambda==0 (classical OLS t-distribution assumptions)."""
    c = np.asarray(c, dtype=np.float64)
    betas = fit["betas"]
    cb = np.tensordot(c, betas, axes=([0], [0]))
    # unscaled variance: c' (X'X)^-1 c in ORIGINAL units requires rescaling
    # back through col_scale twice (once per side of the bilinear form)
    cs = c / fit["col_scale"]
    var_unit = float(cs @ fit["XtXws_reg_inv"] @ cs)
    s2 = fit["rss"] / fit["dof"]
    denom = np.sqrt(np.clip(var_unit * s2, 1e-20, None))
    t = cb / denom
    t[~fit["mask"]] = 0.0
    t[~np.isfinite(t)] = 0.0
    return t


def f_omnibus_v2(fit, cond_cols=None):
    """Omnibus F. Only statistically valid when ridge_lambda==0."""
    from numpy.linalg import pinv
    if cond_cols is None:
        cond_cols = list(range(fit["n_cond"]))
    q = len(cond_cols)
    C = np.zeros((q, fit["P"]))
    for i, col in enumerate(cond_cols):
        C[i, col] = 1.0
    betas = fit["betas"]
    Cb = np.tensordot(C, betas, axes=([1], [0]))
    Cs = C / fit["col_scale"][None, :]
    M = pinv(Cs @ fit["XtXws_reg_inv"] @ Cs.T)
    shape3 = fit["shape3"]
    Cb2 = Cb.reshape(q, -1)
    num = np.einsum("in,ij,jn->n", Cb2, M, Cb2).reshape(shape3)
    s2 = fit["rss"] / fit["dof"]
    with np.errstate(invalid="ignore", divide="ignore"):
        F = (num / q) / np.clip(s2, 1e-20, None)
    F[~fit["mask"]] = 0.0
    F[~np.isfinite(F)] = 0.0
    return F


# ---------------------------------------------------------------------------
# Quick GCV grid search for ridge lambda (uses only the deterministic design
# + the small voxel sample already needed for AR(2) estimation -- cheap)
# ---------------------------------------------------------------------------
def gcv_select_lambda(Xw, run_row_bounds, Ysamp, n_cond, lambda_grid,
                      phi1, phi2, log=print):
    """Quick generalized cross-validation over a lambda grid, using the
    voxel sample. Task columns standardized to unit norm (ridge convention).
    Returns (best_lambda, diagnostics list of dicts)."""
    nrun = len(run_row_bounds) - 1
    Ttotal = Xw.shape[0]
    P = Xw.shape[1]
    task_norms = np.linalg.norm(Xw[:, :n_cond], axis=0)
    task_norms[task_norms == 0] = 1.0
    col_scale = np.ones(P)
    col_scale[:n_cond] = task_norms
    Xws = Xw / col_scale[None, :]
    XtXws = Xws.T @ Xws
    D = np.zeros((P, P))
    D[np.arange(n_cond), np.arange(n_cond)] = 1.0

    # whiten the sample residuals per run (Ysamp already raw; whiten here)
    Yw = Ysamp.copy()
    for r in range(nrun):
        a, b = run_row_bounds[r], run_row_bounds[r + 1]
        Yw[a:b] = G.ar2_whiten(Ysamp[a:b], phi1, phi2)

    XtY = Xws.T @ Yw   # (P, nvox_sample)
    diagnostics = []
    best = None
    for lam in lambda_grid:
        M = XtXws + lam * D
        Minv = np.linalg.pinv(M)
        H_trace_ratio = Minv @ XtXws          # (P,P) -- cheap
        eff_df = float(np.trace(H_trace_ratio))
        betas = Minv @ XtY                     # (P, nvox_sample)
        fitted = Xws @ betas                   # (Ttotal, nvox_sample)
        rss = float(np.sum((Yw - fitted) ** 2))
        n = Ttotal * Yw.shape[1]
        denom = (1.0 - eff_df / Ttotal) ** 2
        gcv = (rss / n) / max(denom, 1e-8)
        diagnostics.append({"lambda": float(lam), "eff_df": eff_df,
                            "rss_per_obs": rss / n, "gcv": gcv})
        if best is None or gcv < best[1]:
            best = (lam, gcv)
    best_lambda = best[0]
    log(f"  GCV lambda search: best={best_lambda:g} (gcv={best[1]:.6g})")
    for d in diagnostics:
        log(f"    lambda={d['lambda']:>10.4g}  eff_df={d['eff_df']:>7.2f}  "
            f"gcv={d['gcv']:.6g}")
    return best_lambda, diagnostics
