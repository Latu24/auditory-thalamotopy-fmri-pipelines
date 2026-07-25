"""Diagnostic (not a deliverable) -- for each smoothing sigma, report FDR-
significant voxel counts overall and specifically inside the (already
validated) native whole-thalamus / MGB masks, using the same validated
framebox->native transform as 12_reconcile_space.py. Answers: does any
smoothing sigma preserve visible thalamic signal for 3D-model coloring?

Inputs: a BrainVoyager .glm file, the native anatomical NIfTI, and the
native-space thalamus/MGB masks from 11_segment_thalamus.py. Output:
thalamus_work/smoothing_doseresponse.json.
"""
import os, time, json
import numpy as np
import nibabel as nib
from scipy import stats, ndimage
from statsmodels.stats.multitest import multipletests
import bvbabel.glm

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

ROOT = PROJECT_ROOT
GLM = os.environ.get("GLM_PATH", os.path.join(PROJECT_ROOT, "raw", "sub-01_CF_AM_run-1_VTC_N-1_FFX_AR-2.glm"))
WORK = f"{ROOT}/derivatives/sub-01/analysis/thalamus_work"
ANAT = f"{ROOT}/derivatives/sub-01/anat/sub-01_desc-UNIdenoisedN4.nii.gz"

log("loading GLM ...")
h, R2, SS, beta, SSXiY, meantc, ARlag = bvbabel.glm.read_glm(GLM)
del SSXiY, ARlag
shapeZXY = beta.shape[:3]
npred = beta.shape[3]
invXX = np.asarray(h["Inverted X'X matrix"], float)
dfe = int(h["Nr time points"]) - npred
SSf = SS.reshape(-1).astype(np.float64)
beta_flat = beta.reshape(-1, npred)
mtc = meantc.reshape(-1)
brain_flat = mtc > np.percentile(mtc, 60)
brain_vol = brain_flat.reshape(shapeZXY)
bidx = np.where(brain_flat)[0]
bbox = {k: int(h[k]) for k in ("XStart","XEnd","YStart","YEnd","ZStart","ZEnd")}
log(f"loaded. dfe={dfe}")

anat = nib.load(ANAT)
thal = np.asarray(nib.load(f"{WORK}/sub-01_thalamus_native.nii.gz").dataobj) > 0
mgb = np.asarray(nib.load(f"{WORK}/sub-01_MGB_native.nii.gz").dataobj) > 0
FBZ, FBY, FBX = 240, 320, 320


def to_native_from_ZXYflat(flat_ZXY):
    """flat boolean/float over (Z,X,Y) brain-flat-index space -> full (Z,X,Y) vol
    -> VTC-raw (Z,Y,X) -> framebox -> native, using the validated transform."""
    vol = flat_ZXY.reshape(shapeZXY)                      # (Z,X,Y)
    raw = np.ascontiguousarray(np.transpose(vol, (0, 2, 1)))   # (Z,Y,X) VTC-raw
    fb = np.zeros((FBZ, FBY, FBX), np.float32)
    fb[bbox["ZStart"]:bbox["ZEnd"], bbox["YStart"]:bbox["YEnd"],
       bbox["XStart"]:bbox["XEnd"]] = raw
    native = np.ascontiguousarray(np.transpose(fb[::-1, ::-1, ::-1], (0, 2, 1)))
    return native


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


def run(cols, sigma):
    q = len(cols)
    M = invXX[np.ix_(cols, cols)]; Minv = np.linalg.inv(M)
    blk = beta_flat[:, cols].astype(np.float32).copy()
    bblk = blk[bidx]
    med = np.median(bblk, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(bblk - med), axis=1, keepdims=True)
    blk[bidx] = np.clip(bblk, (med-5*mad).astype(np.float32), (med+5*mad).astype(np.float32))
    vol = blk.reshape(shapeZXY + (q,))
    vol = mnorm_smooth(vol, sigma)
    smoothed_flat = vol.reshape(-1, q)
    SSvol = SSf.reshape(shapeZXY)
    SSvol_s = mnorm_smooth(SSvol, sigma)
    SSf_s = SSvol_s.reshape(-1)
    bc = smoothed_flat[bidx].astype(np.float64)
    quad = np.einsum('vi,ij,vj->v', bc, Minv, bc)
    sig2 = SSf_s[bidx] / dfe
    with np.errstate(divide='ignore', invalid='ignore'):
        F = quad / (q * sig2)
    F[~np.isfinite(F)] = 0.0
    p = stats.f.sf(F, q, dfe); p[~np.isfinite(p)] = 1.0
    rej, qv, _, _ = multipletests(p, alpha=0.05, method='fdr_bh')
    sig_full = np.zeros(beta_flat.shape[0], bool)
    sig_full[bidx] = rej
    native_sig = to_native_from_ZXYflat(sig_full.astype(np.float32)) > 0.5
    n_thal = int((native_sig & thal).sum())
    n_mgb = int((native_sig & mgb).sum())
    return dict(sigma=sigma, n_fdr=int(rej.sum()), peakF=float(F.max()),
               n_in_thalamus=n_thal, n_in_MGB=n_mgb)


results = {}
for tag, cols in [("CF", np.arange(0, 36)), ("AM", np.arange(36, 45))]:
    print(f"=== {tag} ===")
    results[tag] = []
    for sigma in (0.0, 0.3, 0.6, 0.9, 1.2):
        r = run(cols, sigma)
        results[tag].append(r)
        print(f"  sigma={sigma:.1f}  n_fdr={r['n_fdr']:7d}  peakF={r['peakF']:7.2f}  "
              f"n_in_thalamus={r['n_in_thalamus']:5d}  n_in_MGB={r['n_in_MGB']}")

json.dump(results, open(f"{WORK}/smoothing_doseresponse.json", "w"), indent=2)
log("DONE dose-response")
