#!/usr/bin/env python3
"""Slice-timing correction using each run's own JSON sidecar SliceTiming
array (CMRR multiband, MB2, interleaved acquisition). Applies a Fourier
(sinc) phase shift per slice, shifting each slice's time series by
-slice_time so every slice is realigned to the start of the TR (t=0
reference). Input: trimmed/thresholded BOLD NIfTI per run (stage-02cut).
Output: slice-time-corrected NIfTI per run plus temporal (global-signal and
DVARS) QC figures and a JSON summary.
"""
import os, json, numpy as np, nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FUNC = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "func")
RAW = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "rawdata_nifti")
LOG = os.path.join(PROJECT_ROOT, "logs")
RUNS = [1, 2, 3, 4]
TAG = "v2"

# Each run's SliceTiming array differs slightly under this CMRR interleaved
# MB2 protocol, so slice timing is read from each run's own JSON sidecar
# rather than assumed identical across runs.

def fourier_time_shift(ts, shift_samples):
    T = ts.shape[-1]
    freqs = np.fft.fftfreq(T)
    phase = np.exp(-2j * np.pi * freqs * shift_samples)
    F = np.fft.fft(ts, axis=-1)
    out = np.fft.ifft(F * phase, axis=-1).real
    return out

def main():
    qc = {}
    for r in RUNS:
        j = json.load(open(f"{RAW}/sub-01_task-asta_run-{r}_bold.json"))
        tr = j["RepetitionTime"]
        slice_times = np.array(j["SliceTiming"], dtype=np.float64)
        print(f"[run{r}] own JSON: TR={tr}s nslices={len(slice_times)} "
              f"slicetime range [{slice_times.min()},{slice_times.max()}]s")

        img = nib.load(f"{FUNC}/sub-01_run-{r}_stage-02cut.nii.gz")
        data = np.asarray(img.dataobj, dtype=np.float32)  # (X,Y,S,T)
        X, Y, S, T = data.shape
        assert S == len(slice_times), f"slice count mismatch {S} vs {len(slice_times)}"

        gmean = data.reshape(-1, T).mean(0)
        dvars = np.sqrt((np.diff(data, axis=-1) ** 2).reshape(-1, T - 1).mean(0))

        out = np.empty_like(data)
        for s in range(S):
            shift = -slice_times[s] / tr
            out[:, :, s, :] = fourier_time_shift(data[:, :, s, :], shift)

        out_path = f"{FUNC}/sub-01_run-{r}_{TAG}_stage-03stc.nii.gz"
        nib.save(nib.Nifti1Image(out.astype(np.float32), img.affine, img.header), out_path)

        med = np.median(dvars); iqr = np.subtract(*np.percentile(dvars, [75, 25]))
        spike_thr = med + 4 * iqr
        spikes = np.where(dvars > spike_thr)[0] + 1
        qc[r] = dict(nvol=int(T), tr=float(tr),
                     slicetiming_min_s=float(slice_times.min()),
                     slicetiming_max_s=float(slice_times.max()),
                     gmean_mean=float(gmean.mean()),
                     gmean_cv=float(gmean.std() / gmean.mean()),
                     dvars_med=float(med), dvars_max=float(dvars.max()),
                     n_spikes=int(len(spikes)),
                     spike_vols=spikes.tolist()[:20],
                     out_path=out_path)
        print(f"[run{r}] STC done T={T}  gmean_CV={qc[r]['gmean_cv']:.4f} "
              f"DVARS med={med:.1f} max={dvars.max():.1f} spikes={len(spikes)} -> {out_path}")

        fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        ax[0].plot(gmean, lw=0.7); ax[0].set_ylabel("global mean")
        ax[0].set_title(f"run{r} temporal QC (pre-STC global signal & DVARS, own JSON)")
        ax[1].plot(np.arange(1, T), dvars, lw=0.7, color="tab:red")
        ax[1].axhline(spike_thr, ls="--", color="k", lw=0.6, label="spike thr")
        ax[1].set_ylabel("DVARS"); ax[1].set_xlabel("volume"); ax[1].legend()
        fig.tight_layout(); fig.savefig(f"{LOG}/qc_run-{r}_{TAG}_temporal.png", dpi=90)
        plt.close(fig)

    json.dump(qc, open(f"{LOG}/qc_slicetiming_{TAG}.json", "w"), indent=2)
    print(f"wrote logs/qc_slicetiming_{TAG}.json and qc_run-*_{TAG}_temporal.png")

if __name__ == "__main__":
    main()
