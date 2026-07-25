"""Diagnostic ablation (not a deliverable) -- isolate why pre-F-test spatial
smoothing decreases FDR survival instead of increasing it. Loads the GLM once
and runs the CF/AM omnibus-F test under four conditions: (a) fully raw
(baseline), (b) winsorize only, (c) winsorize + smooth sigma=0.6, (d)
winsorize + smooth sigma=1.2.

Input: a BrainVoyager .glm file. Output: printed per-condition F/FDR summary
(no files written); feeds the smoothing-dose-response finding documented in
10_bestfreq_from_glm.py.
"""
import os, time
import numpy as np
from scipy import stats, ndimage
from statsmodels.stats.multitest import multipletests
import bvbabel.glm

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

GLM = os.environ.get("GLM_PATH", os.path.join(PROJECT_ROOT, "raw", "sub-01_CF_AM_run-1_VTC_N-1_FFX_AR-2.glm"))

log("loading GLM ...")
h, R2, SS, beta, SSXiY, meantc, ARlag = bvbabel.glm.read_glm(GLM)
del SSXiY, ARlag
shapeZXY = beta.shape[:3]
npred = beta.shape[3]
invXX = np.asarray(h["Inverted X'X matrix"], float)
dfe = int(h["Nr time points"]) - npred
R2f = R2.reshape(-1).astype(np.float64)
SSf = SS.reshape(-1).astype(np.float64)
beta_flat = beta.reshape(-1, npred)
mtc = meantc.reshape(-1)
brain_flat = mtc > np.percentile(mtc, 60)
brain_vol = brain_flat.reshape(shapeZXY)
bidx = np.where(brain_flat)[0]
log(f"loaded. dfe={dfe} nbrain={len(bidx)}")


def mnorm_smooth(vol, sigma):
    if sigma < 1e-3:
        return vol
    w = np.clip(ndimage.gaussian_filter(brain_vol.astype(np.float32), sigma=sigma), 1e-6, None)
    if vol.ndim == brain_vol.ndim:
        num = ndimage.gaussian_filter(vol * brain_vol, sigma=sigma)
        out = num / w; out[~brain_vol] = vol[~brain_vol]; return out
    out = np.empty_like(vol)
    for c in range(vol.shape[-1]):
        num = ndimage.gaussian_filter(vol[..., c] * brain_vol, sigma=sigma)
        oc = num / w; oc[~brain_vol] = vol[..., c][~brain_vol]; out[..., c] = oc
    return out


def run(name, cols, hz, winsorize, sigma):
    q = len(cols)
    M = invXX[np.ix_(cols, cols)]; Minv = np.linalg.inv(M)
    blk = beta_flat[:, cols].astype(np.float32).copy()
    if winsorize:
        bblk = blk[bidx]
        med = np.median(bblk, axis=1, keepdims=True)
        mad = 1.4826 * np.median(np.abs(bblk - med), axis=1, keepdims=True)
        blk[bidx] = np.clip(bblk, (med - 5*mad).astype(np.float32), (med + 5*mad).astype(np.float32))
    vol = blk.reshape(shapeZXY + (q,))
    if sigma > 1e-3:
        vol = mnorm_smooth(vol, sigma)
    smoothed_flat = vol.reshape(-1, q)
    SSvol = SSf.reshape(shapeZXY)
    SSvol_s = mnorm_smooth(SSvol, sigma) if sigma > 1e-3 else SSvol
    SSf_s = SSvol_s.reshape(-1)
    bc = smoothed_flat[bidx].astype(np.float64)
    quad = np.einsum('vi,ij,vj->v', bc, Minv, bc)
    sig2 = SSf_s[bidx] / dfe
    with np.errstate(divide='ignore', invalid='ignore'):
        F = quad / (q * sig2)
    F[~np.isfinite(F)] = 0.0
    p = stats.f.sf(F, q, dfe); p[~np.isfinite(p)] = 1.0
    rej, qv, _, _ = multipletests(p, alpha=0.05, method='fdr_bh')
    print(f"  {name:35s} winsor={str(winsorize):5s} sigma={sigma:.2f}  "
          f"peakF={F.max():7.2f}  n_p001={int((p<0.001).sum()):7d}  n_fdr={int(rej.sum()):7d}")


for cols, hz, tag in [(np.arange(0,36), None, "CF"), (np.arange(36,45), None, "AM")]:
    print(f"=== {tag} ===")
    run("(a) raw (baseline repro)", cols, hz, winsorize=False, sigma=0.0)
    run("(b) winsorize only", cols, hz, winsorize=True, sigma=0.0)
    run("(c) winsorize+smooth 0.6", cols, hz, winsorize=True, sigma=0.6)
    run("(d) winsorize+smooth 1.2", cols, hz, winsorize=True, sigma=1.2)
    run("(e) smooth 1.2 only (no winsorize)", cols, hz, winsorize=False, sigma=1.2)
log("DONE ablation")
