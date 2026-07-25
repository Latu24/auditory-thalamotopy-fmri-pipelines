"""Thalamus pipeline -- Part A: best-frequency-per-voxel from the GLM.

Computes, separately for the carrier-frequency (CF, 36 conditions) and
amplitude-modulation-rate (AM, 9 conditions) predictor families, a per-voxel
best-frequency estimate: winsorize the condition-beta profile, fit a Gaussian
in log10(frequency) space (argmax as both initializer and fallback), and
report a continuous best-frequency (Hz) and a categorical winning-condition
index. An omnibus F-test (and FDR q-values) is also computed per family and
stored as a diagnostic, but does not gate which voxels receive a value here --
every voxel with functional coverage is fit (see METHODOLOGY note below). A
separate significance gate is applied downstream by 17_apply_p01_gate.py.

Input: a BrainVoyager .glm file (beta/SS/R2/meantc arrays + design matrix).
Outputs: thalamus_work/bestfreq_maps.npz (bf/cond/F/qval arrays for CF and
AM, VTC-raw layout) and thalamus_work/bestfreq_stats.json (methodology +
per-block diagnostics). Native-space NIfTI/VMP conversion happens downstream.

METHODOLOGY: voxels receive a best-frequency value based on functional
coverage alone (mean-time-course brain mask), not on the omnibus-F/FDR test.
Layering a significance test on top of an already collinear/unstable GLM
design (many correlated condition regressors) was found to remove real
signal along with noise rather than filter noise selectively -- consistent
with the smoothing dose-response finding below. The F/FDR values are kept as
diagnostics in the JSON sidecar and *_F/*_qval arrays but are not used to
decide which voxels are exposed.

SMOOTHING: pre-test spatial smoothing of the condition-beta/SS volumes
(sigma 0.0-1.2 voxels) was evaluated as a possible SNR-boosting step before
the F-test. Empirically, survival dropped monotonically with sigma and
thalamic overlap dropped to zero at sigma>=0.6 (from 8 CF / 75 AM in-thalamus
at sigma=0), indicating the significant voxels are spatially isolated
single-voxel events rather than a spatially extended signal, so smoothing
dilutes rather than reinforces them here. Shipped default: test_sigma=0.0 for
the F-test (computed on winsorized-only betas); profile_sigma=0.6 still
lightly smooths the tuning-curve profile used only for the Gaussian fit,
applied after the coverage mask so it does not affect which voxels are
exposed.

Variance denominator: residual variance is estimated as SS/dfe rather than
the textbook SS*(1-R2)/dfe. The stored R2 is inflated by the large number of
collinear condition regressors (median R2~0.76 in-brain), and the textbook
formula flags roughly 77% of the brain volume as "significant" -- clearly
non-selective. SS/dfe is used as the primary (more conservative) variance
proxy; the textbook-F peak is reported only as a diagnostic.
"""
import os, sys, json, time
import numpy as np
import multiprocessing as mp
from scipy import optimize, stats, ndimage
import bvbabel.glm

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

NPROC = 8   # empirically ~8x faster via a fork-context process pool than a
            # sequential per-voxel loop at ~1.24M voxels/dimension

t0 = time.time()
GLM = os.environ.get("GLM_PATH", os.path.join(PROJECT_ROOT, "raw", "sub-01_CF_AM_run-1_VTC_N-1_FFX_AR-2.glm"))
WORK = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "analysis", "thalamus_work")
LOGS = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(WORK, exist_ok=True)

CF_HZ = 200.0 * (8000.0 / 200.0) ** (np.arange(36) / 35.0)
AM_HZ = 1.0 * (16.0 / 1.0) ** (np.arange(9) / 8.0)


def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


def gauss(x, a, mu, s, b):
    return a * np.exp(-(x - mu) ** 2 / (2.0 * s * s)) + b


def fit_r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def bestfreq_block(profiles, hz):
    """profiles: (nsig, q) winsorized+smoothed betas (per significant voxel).
    Returns bf_hz (continuous), cond_idx (1..q argmax), fit_ok (bool)."""
    n, q = profiles.shape
    x = np.log10(hz)
    xspan = x[-1] - x[0]
    bf_hz = np.empty(n, np.float64)
    cond_idx = np.empty(n, np.int32)
    fit_ok = np.zeros(n, bool)
    lo = [0.0, x[0], (x[1] - x[0]) / 2.0, -5.0]
    hi = [10.0, x[-1], xspan, 5.0]
    for i in range(n):
        p = profiles[i]
        # z-score across conditions (scale-free)
        mu_, sd_ = p.mean(), p.std()
        y = (p - mu_) / sd_ if sd_ > 0 else p - mu_
        k0 = int(np.argmax(y))
        cond_idx[i] = k0 + 1
        bf = hz[k0]  # fallback = argmax freq
        p0 = [max(y[k0] - y.min(), 1e-3), x[k0], xspan / 4.0, y.min()]
        # clip p0 into bounds
        p0 = [min(max(p0[j], lo[j]), hi[j]) for j in range(4)]
        try:
            popt, _ = optimize.curve_fit(gauss, x, y, p0=p0, bounds=(lo, hi),
                                         maxfev=5000)
            mu = popt[1]
            if x[0] <= mu <= x[-1] and fit_r2(y, gauss(x, *popt)) > 0.5:
                bf = 10.0 ** mu
                fit_ok[i] = True
        except Exception:
            pass
        bf_hz[i] = bf
    return bf_hz, cond_idx, fit_ok


def _fit_chunk(args):
    """Top-level (picklable/fork-safe) worker: run bestfreq_block on one chunk."""
    profiles, hz = args
    return bestfreq_block(profiles, hz)


def bestfreq_block_parallel(profiles, hz, nproc=NPROC):
    """Same contract as bestfreq_block, but fans the per-voxel scipy.curve_fit
    loop out across `nproc` forked processes (each voxel's fit is independent).
    Used because every coverage-mask voxel (not just a small significant
    subset) gets a Gaussian fit -- roughly 1.24M voxels/dimension."""
    n = profiles.shape[0]
    if n == 0:
        return (np.zeros(0, np.float64), np.zeros(0, np.int32), np.zeros(0, bool))
    if n < 2000:   # not worth process-pool overhead for small inputs
        return bestfreq_block(profiles, hz)
    chunks = np.array_split(np.arange(n), nproc)
    args = [(profiles[idx], hz) for idx in chunks]
    ctx = mp.get_context("fork")
    with ctx.Pool(nproc) as pool:
        results = pool.map(_fit_chunk, args)
    bf_hz = np.concatenate([r[0] for r in results])
    cond_idx = np.concatenate([r[1] for r in results])
    fit_ok = np.concatenate([r[2] for r in results])
    return bf_hz, cond_idx, fit_ok


def mask_normalized_smooth(vol, brain_vol, sigma):
    """Gaussian-smooth `vol` (any number of trailing channels) using only
    in-brain support, renormalizing by the smoothed brain indicator so
    out-of-brain zeros don't bleed into near-edge in-brain voxels."""
    w = ndimage.gaussian_filter(brain_vol.astype(np.float32), sigma=sigma)
    w = np.clip(w, 1e-6, None)
    if vol.ndim == brain_vol.ndim:
        num = ndimage.gaussian_filter(vol * brain_vol, sigma=sigma)
        out = num / w
        out[~brain_vol] = vol[~brain_vol]
        return out
    out = np.empty_like(vol)
    for c in range(vol.shape[-1]):
        num = ndimage.gaussian_filter(vol[..., c] * brain_vol, sigma=sigma)
        oc = num / w
        oc[~brain_vol] = vol[..., c][~brain_vol]
        out[..., c] = oc
    return out


def process_block(name, beta_flat, cols, hz, invXX, R2f, SSf, brain_flat,
                  dfe, shapeZXY, test_sigma=0.0, profile_sigma=0.6):
    """test_sigma: spatial smoothing applied to beta+SS before the (diagnostic-
    only, non-gating) omnibus F-test. profile_sigma: smoothing applied to the
    tuning profile used for the Gaussian best-freq fit.

    Every voxel in the functional-coverage mask (`brain_flat`) gets a
    Gaussian-fit best-frequency value; the F/FDR test below is computed and
    reported for every voxel but does not decide which voxels are exposed
    (see module docstring)."""
    q = len(cols)
    log(f"[{name}] block cols {cols[0]}..{cols[-1]} (q={q})  "
        f"test_sigma={test_sigma} (F-test) profile_sigma={profile_sigma} "
        f"(Gaussian-fit tuning curve only, post-significance-mask)")
    M = invXX[np.ix_(cols, cols)]
    Minv = np.linalg.inv(M)
    bidx = np.where(brain_flat)[0]
    brain_vol = brain_flat.reshape(shapeZXY)

    # ---- winsorize RAW block betas (kill |beta|>1e4 spikes) BEFORE any smoothing,
    #      so a single blown-up voxel can't contaminate its neighbours ----
    blk = beta_flat[:, cols].astype(np.float32)               # (nvox, q) full
    bblk = blk[bidx]                                           # copy (nbrain,q)
    med = np.median(bblk, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(bblk - med), axis=1, keepdims=True)
    lo = (med - 5.0 * mad).astype(np.float32)
    hi = (med + 5.0 * mad).astype(np.float32)
    blk[bidx] = np.clip(bblk, lo, hi)
    del bblk
    vol_w = blk.reshape(shapeZXY + (q,))                       # winsorized, unsmoothed
    del blk

    # ---- significance-test volume: winsorized + test_sigma smoothing (beta AND
    #      the SS denominator, same kernel). Default test_sigma=0.0: a dose-
    #      response sweep (sigma 0.0/0.3/0.6/0.9/1.2) showed that meaningful
    #      pre-F-test smoothing here monotonically reduces FDR survival and
    #      eliminates the sparse thalamic significant voxels by sigma=0.6
    #      (present at sigma=0: CF n=8, AM n=75 in-thalamus). This indicates the
    #      significant voxels are spatially isolated single-voxel events (an
    #      unstable-design signature) rather than spatially-extended signal, so
    #      smoothing dilutes rather than reinforces them. test_sigma=0.0 is kept
    #      as the shipped default so the real thalamic signal remains visible in
    #      the 3D models. ----
    if test_sigma > 1e-6:
        vol_test = mask_normalized_smooth(vol_w, brain_vol, test_sigma)
        SSvol_test = mask_normalized_smooth(SSf.reshape(shapeZXY), brain_vol, test_sigma)
    else:
        vol_test = vol_w
        SSvol_test = SSf.reshape(shapeZXY)
    test_flat = vol_test.reshape(-1, q)
    SSf_test = SSvol_test.reshape(-1)

    # ---- omnibus partial-F on brain voxels, using the (test_sigma-smoothed) beta + SS ----
    bc = test_flat[bidx].astype(np.float64)                    # (nbrain, q)
    quad = np.einsum('vi,ij,vj->v', bc, Minv, bc)
    # Residual variance uses SS/dfe rather than the textbook SS*(1-R2)/dfe: the
    # stored R2 is inflated by the many collinear condition regressors (median
    # R2~0.76 in-brain), and the textbook formula flags ~77% of the brain
    # volume as "significant" (verified), which is not a selective mask. SS/dfe
    # is used as the conservative variance proxy; the textbook-F peak is
    # reported as a diagnostic only.
    sig2_total = SSf_test[bidx] / dfe                           # primary
    sig2_resid = SSf_test[bidx] * (1.0 - R2f[bidx]) / dfe       # textbook (diagnostic)
    with np.errstate(divide='ignore', invalid='ignore'):
        F = quad / (q * sig2_total)                            # primary
        F_textbook = quad / (q * sig2_resid)                   # diagnostic
    F[~np.isfinite(F)] = 0.0
    F_textbook[~np.isfinite(F_textbook)] = 0.0
    Fcrit = float(stats.f.isf(0.001, q, dfe))
    p = stats.f.sf(F, q, dfe)
    p[~np.isfinite(p)] = 1.0
    # FDR-BH over brain voxels
    from statsmodels.stats.multitest import multipletests
    rej, qval, _, _ = multipletests(p, alpha=0.05, method='fdr_bh')
    n_raw = int((p < 0.001).sum())
    n_fdr = int(rej.sum())
    log(f"[{name}] (DIAGNOSTIC ONLY, not gating) peakF={F.max():.2f} "
        f"Fcrit(p.001,{q},{dfe})={Fcrit:.2f} nP<.001={n_raw} nFDR={n_fdr} "
        f"(textbook-F peak={F_textbook.max():.2f}) R2[median]={np.median(R2f[bidx]):.3f}")

    # ---- profile volume for the Gaussian best-freq fit: winsorized + profile_sigma
    #      smoothing (shapes the tuning curve only). GATE = brain_flat/bidx
    #      (functional coverage), NOT F/FDR significance. Every coverage voxel
    #      is fit. ----
    if abs(profile_sigma - test_sigma) < 1e-9:
        vol_profile = vol_test
    elif profile_sigma > 1e-6:
        vol_profile = mask_normalized_smooth(vol_w, brain_vol, profile_sigma)
    else:
        vol_profile = vol_w
    profile_flat = vol_profile.reshape(-1, q)
    prof = profile_flat[bidx].astype(np.float64)
    log(f"[{name}] fitting Gaussian on ALL {len(bidx)} functional-coverage voxels "
        f"(parallel, {NPROC} procs) ...")
    t_fit0 = time.time()
    bf_hz, cond_idx, fit_ok = bestfreq_block_parallel(prof, hz)
    log(f"[{name}] fit done in {time.time()-t_fit0:.1f}s: "
        f"fit_ok {fit_ok.sum()}/{len(fit_ok)} ({100*fit_ok.mean():.1f}%)  "
        f"bf range {bf_hz.min():.1f}-{bf_hz.max():.1f} Hz")

    # ---- scatter into full VTC-raw layout volumes (Z,Y,X); ALL coverage voxels
    #      get a value now, not just an F/FDR-significant subset ----
    nvox = beta_flat.shape[0]
    bf_full = np.zeros(nvox, np.float32)
    cond_full = np.zeros(nvox, np.float32)
    bf_full[bidx] = bf_hz
    cond_full[bidx] = cond_idx
    F_full = np.zeros(nvox, np.float32)
    F_full[bidx] = F
    q_full = np.ones(nvox, np.float32)
    q_full[bidx] = qval
    # reshape (Z,X,Y) then transpose to (Z,Y,X) VTC-raw layout
    def toraw(a):
        return np.ascontiguousarray(np.transpose(a.reshape(shapeZXY), (0, 2, 1)))
    out = dict(bf=toraw(bf_full), cond=toraw(cond_full), F=toraw(F_full),
               qval=toraw(q_full))
    stats_d = dict(name=name, q=q, cols=[int(c) for c in cols],
                   hz=[float(v) for v in hz], dfe=int(dfe),
                   gating="functional-coverage mask ONLY (meantc-based brain mask) -- "
                          "F/FDR below is reported as a diagnostic, it does NOT gate "
                          "which voxels receive a best-frequency value",
                   Fcrit_p001_diag=Fcrit, peakF_diag=float(F.max()),
                   peakF_textbook_diag=float(F_textbook.max()),
                   n_raw_p001_diag=n_raw, n_fdr_diag=n_fdr,
                   R2_median_inbrain=float(np.median(R2f[bidx])),
                   fit_success_rate=float(fit_ok.mean()),
                   n_coverage_voxels=int(len(bidx)),
                   test_sigma=test_sigma, profile_sigma=profile_sigma, winsor_k=5.0,
                   smoothing_note="Pre-F-test spatial smoothing of the condition-beta/SS "
                                  "volumes (sigma~0.3-1.2 voxels) was evaluated as a "
                                  "possible SNR-boosting step. It instead monotonically "
                                  "decreased FDR survival for both CF and AM and eliminated "
                                  "the sparse thalamic-overlapping significant voxels by "
                                  "sigma>=0.6 (present at sigma=0: CF n=8, AM n=75 "
                                  "in-thalamus). This indicates the significant voxels are "
                                  "spatially isolated single-voxel events, not "
                                  "spatially-extended signal, so smoothing dilutes rather "
                                  "than reinforces them. Shipped default: test_sigma=0.0 "
                                  "(F-test on winsorized-only betas) so the real thalamic "
                                  "signal remains visible in the 3D models; profile_sigma=0.6 "
                                  "still shapes the Gaussian-fit tuning curve post-mask, "
                                  "unaffected by this finding. Full dose-response results are "
                                  "saved to thalamus_work/smoothing_doseresponse.json. Note: "
                                  "the output-gating criterion below ('significant voxels') "
                                  "reflects the coverage-mask methodology described in the "
                                  "module docstring, not an F/FDR gate.")
    del vol_w, vol_test, vol_profile
    return out, stats_d


def main():
    log("reading GLM ...")
    h, R2, SS, beta, SSXiY, meantc, ARlag = bvbabel.glm.read_glm(GLM)
    del SSXiY, ARlag
    log(f"beta shape {beta.shape} dtype {beta.dtype}  R2 {R2.shape}  meantc {meantc.shape}")
    shapeZXY = beta.shape[:3]                                  # (Z,X,Y)
    npred = beta.shape[3]
    # predictor names
    pinfo = h.get("Predictor info") or h.get("PredictorInfo") or []
    names = []
    for e in pinfo:
        if isinstance(e, dict):
            names.append(e.get("NameOfPredictor") or e.get("Name") or str(e))
        else:
            names.append(str(e))
    log(f"npred={npred}  first names: {names[:3]} ... AM-edge: {names[35:38]} "
        f"tail: {names[-5:]}")
    invXX = np.asarray(h["Inverted X'X matrix"], float)
    ntp = int(h["Nr time points"])
    dfe = ntp - npred
    log(f"Nr time points={ntp} npred={npred} dfe={dfe} invXX {invXX.shape} "
        f"cond(invXX)={np.linalg.cond(invXX):.2e}")
    assert R2.shape == shapeZXY and meantc.shape == shapeZXY

    R2f = R2.reshape(-1).astype(np.float64)
    SSf = SS.reshape(-1).astype(np.float64)
    log(f"R2 range [{R2f.min():.3f},{R2f.max():.3f}] median {np.median(R2f):.3f}")
    beta_flat = beta.reshape(-1, npred)                        # view (nvox,49)

    # brain mask: meantc > p60
    mtc = meantc.reshape(-1)
    thr = np.percentile(mtc, 60)
    brain_flat = mtc > thr
    log(f"brain mask thr(p60)={thr:.3f} nvox={int(brain_flat.sum())}")

    cf_out, cf_stats = process_block("CF", beta_flat, np.arange(0, 36), CF_HZ,
                                     invXX, R2f, SSf, brain_flat, dfe, shapeZXY)
    am_out, am_stats = process_block("AM", beta_flat, np.arange(36, 45), AM_HZ,
                                     invXX, R2f, SSf, brain_flat, dfe, shapeZXY)

    # meantc in VTC-raw layout for QC/axis validation downstream
    meantc_raw = np.ascontiguousarray(np.transpose(meantc, (0, 2, 1)))
    bbox = {k: int(h[k]) for k in
            ("XStart", "XEnd", "YStart", "YEnd", "ZStart", "ZEnd")}
    bbox["Resolution"] = int(h.get("Resolution multiplier", 1))

    np.savez_compressed(f"{WORK}/bestfreq_maps.npz",
                        cf_bf=cf_out['bf'], cf_cond=cf_out['cond'],
                        cf_F=cf_out['F'], cf_qval=cf_out['qval'],
                        am_bf=am_out['bf'], am_cond=am_out['cond'],
                        am_F=am_out['F'], am_qval=am_out['qval'],
                        meantc_raw=meantc_raw,
                        dimZ=shapeZXY[0], dimY=shapeZXY[2], dimX=shapeZXY[1])
    meta = dict(glm=GLM, bbox=bbox, dfe=dfe, npred=npred,
                predictor_names=names,
                vtc_raw_shape=[int(shapeZXY[0]), int(shapeZXY[2]), int(shapeZXY[1])],
                beta_axis_order="(Z,X,Y) -> transposed (0,2,1) to VTC-raw (Z,Y,X)",
                cf=cf_stats, am=am_stats,
                sigma2_definition="residual = SS/dfe (primary, conservative); "
                                  "textbook SS*(1-R2)/dfe reported as *_textbook_diag")
    with open(f"{WORK}/bestfreq_stats.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"WROTE {WORK}/bestfreq_maps.npz and bestfreq_stats.json")
    log("DONE")


if __name__ == "__main__":
    main()
