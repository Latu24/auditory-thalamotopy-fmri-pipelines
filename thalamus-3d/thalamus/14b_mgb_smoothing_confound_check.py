"""Control diagnostic (not a deliverable) -- does the MGB neighbor spatial
coherence found by 14_mgb_appearance_check.py reflect real signal, or is it a
mechanical artifact of profile_sigma=0.6 smoothing (which mixes each voxel's
input profile with its neighbors' before the Gaussian fit, and would induce
correlation between adjacent voxels' fitted frequencies regardless of whether
there is real underlying tuning)?

Re-fits only the same MGB-covered voxel positions from raw (winsorized-only,
no spatial smoothing) profiles, and reruns the identical neighbor-roughness
permutation test. If the coherence disappears without smoothing, it was a
smoothing artifact rather than evidence of real tuning.

Inputs: a BrainVoyager .glm file and the native-space MGB mask + best-
frequency NIfTIs. Output: thalamus_work/mgb_smoothing_confound_check.json.
"""
import os, json
import numpy as np
import nibabel as nib
import bvbabel.glm
from scipy import optimize

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ROOT = PROJECT_ROOT
ANA = f"{ROOT}/derivatives/sub-01/analysis"
WORK = f"{ANA}/thalamus_work"
GLM = os.environ.get("GLM_PATH", os.path.join(PROJECT_ROOT, "raw", "sub-01_CF_AM_run-1_VTC_N-1_FFX_AR-2.glm"))

CF_HZ = 200.0 * (8000.0 / 200.0) ** (np.arange(36) / 35.0)
AM_HZ = 1.0 * (16.0 / 1.0) ** (np.arange(9) / 8.0)
rng = np.random.default_rng(1)


def gauss(x, a, mu, s, b):
    return a * np.exp(-(x - mu) ** 2 / (2.0 * s * s)) + b


def fit_r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2); ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def fit_one(p, hz):
    x = np.log10(hz); xspan = x[-1] - x[0]
    mu_, sd_ = p.mean(), p.std()
    y = (p - mu_) / sd_ if sd_ > 0 else p - mu_
    k0 = int(np.argmax(y))
    bf = hz[k0]
    lo = [0.0, x[0], (x[1]-x[0])/2.0, -5.0]; hi = [10.0, x[-1], xspan, 5.0]
    p0 = [max(y[k0]-y.min(),1e-3), x[k0], xspan/4.0, y.min()]
    p0 = [min(max(p0[j], lo[j]), hi[j]) for j in range(4)]
    try:
        popt, _ = optimize.curve_fit(gauss, x, y, p0=p0, bounds=(lo, hi), maxfev=5000)
        mu = popt[1]
        if x[0] <= mu <= x[-1] and fit_r2(y, gauss(x, *popt)) > 0.5:
            bf = 10.0 ** mu
    except Exception:
        pass
    return bf


def neighbor_pairs(mask):
    idx = np.argwhere(mask)
    pos_to_i = {tuple(v): i for i, v in enumerate(idx)}
    pairs = []
    for i, (z, y, x) in enumerate(idx):
        for dz, dy, dx in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            nb = (z + dz, y + dy, x + dx)
            j = pos_to_i.get(nb)
            if j is not None:
                pairs.append((i, j))
    return idx, np.array(pairs, dtype=int) if pairs else np.zeros((0, 2), int)


def roughness(vals_log, pairs):
    if len(pairs) == 0:
        return np.nan
    return float(np.mean(np.abs(vals_log[pairs[:, 0]] - vals_log[pairs[:, 1]])))


def permutation_test(vals_log, pairs, n_perm=2000):
    obs = roughness(vals_log, pairs)
    null = np.empty(n_perm)
    for p in range(n_perm):
        null[p] = roughness(rng.permutation(vals_log), pairs)
    z = (obs - null.mean()) / (null.std() + 1e-12)
    p_lower = float(np.mean(null <= obs))
    return obs, null.mean(), null.std(), z, p_lower


def main():
    print("loading GLM ...")
    h, R2, SS, beta, SSXiY, meantc, ARlag = bvbabel.glm.read_glm(GLM)
    del SSXiY, ARlag, R2, SS, meantc
    shapeZXY = beta.shape[:3]
    bbox = {k: int(h[k]) for k in ("XStart","XEnd","YStart","YEnd","ZStart","ZEnd")}
    FBZ, FBY, FBX = 240, 320, 320

    def native_to_glm_flat_idx(native_ijk):
        """Inverts the validated native<-framebox<-VTCraw<-GLM(Z,X,Y) transform
        for a single voxel index triple, to recover the GLM flat-array index."""
        i, j, k = native_ijk
        # native = FB[::-1,::-1,::-1].transpose(0,2,1) -> invert:
        z_fb = FBZ - 1 - i
        y_fb = FBY - 1 - k
        x_fb = FBX - 1 - j
        # framebox -> VTC-raw (Z,Y,X) via bbox offset
        z_r = z_fb - bbox["ZStart"]; y_r = y_fb - bbox["YStart"]; x_r = x_fb - bbox["XStart"]
        # VTC-raw (Z,Y,X) -> GLM (Z,X,Y): raw = transpose(vol_ZXY,(0,2,1)) -> invert
        z_g, x_g, y_g = z_r, x_r, y_r  # transpose(0,2,1) is its own inverse pairing (Y<->X swap)
        return z_g, x_g, y_g

    mgb = np.asarray(nib.load(f"{WORK}/sub-01_MGB_native.nii.gz").dataobj) > 0

    results = {}
    for dim, hz, fname, cols in [
        ("CF", CF_HZ, "sub-01_CF_bestfreq_fromGLM.nii.gz", np.arange(0, 36)),
        ("AM", AM_HZ, "sub-01_AM_bestfreq_fromGLM.nii.gz", np.arange(36, 45)),
    ]:
        bf_smoothed = np.asarray(nib.load(f"{ANA}/{fname}").dataobj)
        covered = mgb & (bf_smoothed > 0)
        idx, pairs = neighbor_pairs(covered)
        print(f"[{dim}] n_covered={len(idx)} n_pairs={len(pairs)}")

        raw_bf = np.empty(len(idx))
        for n, native_ijk in enumerate(idx):
            zg, xg, yg = native_to_glm_flat_idx(native_ijk)
            if not (0 <= zg < shapeZXY[0] and 0 <= xg < shapeZXY[1] and 0 <= yg < shapeZXY[2]):
                raw_bf[n] = np.nan
                continue
            profile = beta[zg, xg, yg, cols].astype(np.float64)
            # winsorize this single voxel's profile against ITS OWN median/MAD
            # (matches 10_bestfreq_from_glm.py's per-voxel winsorize; no spatial smoothing)
            med = np.median(profile)
            mad = 1.4826 * np.median(np.abs(profile - med))
            profile = np.clip(profile, med - 5*mad, med + 5*mad)
            raw_bf[n] = fit_one(profile, hz)

        valid = np.isfinite(raw_bf)
        vals_log = np.log10(raw_bf[valid])
        # remap pairs to the valid-only subset
        keep_idx = np.where(valid)[0]
        remap = {old: new for new, old in enumerate(keep_idx)}
        pairs_v = np.array([[remap[a], remap[b]] for a, b in pairs
                            if a in remap and b in remap], dtype=int)
        obs, nmean, nstd, z, p = permutation_test(vals_log, pairs_v)
        results[dim] = dict(n_covered=int(len(idx)), n_valid=int(valid.sum()),
                            n_pairs=int(len(pairs_v)),
                            observed_roughness_raw=obs, null_mean_raw=nmean,
                            null_std_raw=nstd, z_raw=z, p_raw=p,
                            interpretation="SURVIVES without smoothing (real structure "
                            "independent of the smoothing kernel)" if p < 0.05 else
                            "DOES NOT survive without smoothing -- the coherence found "
                            "with profile_sigma=0.6 is a MECHANICAL ARTIFACT of "
                            "smoothing neighboring voxels' input profiles together, "
                            "not evidence of real spatial tuning")
        print(f"[{dim}] RAW (unsmoothed) obs={obs:.3f} null={nmean:.3f}+-{nstd:.3f} "
              f"z={z:.2f} p={p:.3f}")
        print(f"  -> {results[dim]['interpretation']}")

    with open(f"{WORK}/mgb_smoothing_confound_check.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote thalamus_work/mgb_smoothing_confound_check.json")


if __name__ == "__main__":
    main()
