#!/usr/bin/env python3
"""Temporal high-pass filtering of the motion-corrected data, using a
BrainVoyager "GLM-Fourier"-style regression against a low-frequency
sine/cosine basis (NCYCLES=4). Input: motion-corrected NIfTI per run.
Output: high-pass-filtered NIfTI per run plus a QC figure and JSON summary.
"""
import os, json, numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FUNC = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "func")
RAW = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "rawdata_nifti")
LOG = os.path.join(PROJECT_ROOT, "logs")
RUNS = [1, 2, 3, 4]
NCYCLES = 4
TAG = "v2"

def hp_regressors(T, ncycles):
    t = np.linspace(0, 1, T, endpoint=False)
    cols = [np.ones(T), (t - t.mean())]
    for k in range(1, ncycles + 1):
        cols.append(np.sin(2 * np.pi * k * t))
        cols.append(np.cos(2 * np.pi * k * t))
    return np.column_stack(cols)

def main():
    qc = {}
    for r in RUNS:
        j = json.load(open(f"{RAW}/sub-01_task-asta_run-{r}_bold.json"))
        tr = j["RepetitionTime"]   # each run's own TR (same 1.6s for all, but read per-run for consistency)
        img = nib.load(f"{FUNC}/sub-01_run-{r}_{TAG}_stage-04mc.nii.gz")
        data = np.asarray(img.dataobj, dtype=np.float32)
        X, Y, S, T = data.shape
        cutoff_hz = NCYCLES / (T * tr)
        Xr = hp_regressors(T, NCYCLES)
        M = Xr @ np.linalg.pinv(Xr)
        flat = data.reshape(-1, T)
        mean = flat.mean(1, keepdims=True)
        resid = flat - (M @ flat.T).T
        out = (resid + mean).reshape(X, Y, S, T).astype(np.float32)
        out_path = f"{FUNC}/sub-01_run-{r}_{TAG}_stage-05hpf.nii.gz"
        nib.save(nib.Nifti1Image(out, img.affine, img.header), out_path)

        mask = data.mean(-1) > np.percentile(data.mean(-1), 70)
        idx = np.argwhere(mask)
        rng = np.random.default_rng(0)
        picks = idx[rng.choice(len(idx), 4, replace=False)]
        fig, ax = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
        for a, (xi, yi, zi) in zip(ax, picks):
            a.plot(data[xi, yi, zi], lw=0.6, label="pre-HPF", color="tab:gray")
            a.plot(out[xi, yi, zi], lw=0.7, label="post-HPF", color="tab:blue")
            a.set_ylabel(f"({xi},{yi},{zi})", fontsize=7)
        ax[0].legend(fontsize=7)
        ax[0].set_title(f"run{r} HPF {NCYCLES}cyc (cutoff {cutoff_hz*1000:.2f} mHz)")
        ax[-1].set_xlabel("volume")
        fig.tight_layout(); fig.savefig(f"{LOG}/qc_run-{r}_{TAG}_hpf_timecourses.png", dpi=90)
        plt.close(fig)
        qc[r] = dict(nvol=int(T), ncycles=NCYCLES, cutoff_hz=float(cutoff_hz),
                     cutoff_period_s=float(1/cutoff_hz), out_path=out_path)
        print(f"[run{r}] HPF {NCYCLES}cyc cutoff={cutoff_hz*1000:.3f}mHz "
              f"(period {1/cutoff_hz:.0f}s) T={T} -> {out_path}")
    json.dump(qc, open(f"{LOG}/qc_highpass_{TAG}.json", "w"), indent=2)
    print(f"wrote logs/qc_highpass_{TAG}.json + qc_run-*_{TAG}_hpf_timecourses.png")

if __name__ == "__main__":
    main()
