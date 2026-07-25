# fMRI Preprocessing Pipeline

Raw-to-analysis-ready preprocessing pipeline for a 7T fMRI tone-listening
study of tonotopic (best-frequency) and amplitude-modulation-rate mapping in
the human auditory thalamus and cortex. The pipeline had the results tested against a full BrainVoyager pipeline and this one takes raw multiband EPI
and MP2RAGE DICOMs and produces BrainVoyager-native FMR (functional), VMR
(anatomical), and VTC (co-registered functional-in-anatomical-space) files,
entirely programmatically via [`bvbabel`](https://github.com/ofgulban/bvbabel)
— this pipeline aims for Maastricht students that have no access to BrainVoyager but are still required to submit Brainvoyager File types.
- No BrainVoyager GUI is used or required.

The pipeline is organized into three stages that run in sequence:

1. **`functional/`** — DICOM verification, slice-timing correction, motion
   correction, high-pass filtering, and susceptibility-distortion correction
   (FSL topup and/or ANTs SyN), producing a final preprocessed FMR per run.
2. **`anatomical/`** — MP2RAGE UNI denoising, N4 bias-field correction, and
   VMR/V16 construction, producing a QC'd anatomical volume.
3. **`registration/`** — functional-to-anatomical coregistration, distortion-
   residual QC, and VTC construction, producing the final co-registered
   functional volumes used for downstream analysis (e.g. GLM / tonotopic
   mapping).

Every stage writes new, uniquely named output files rather than overwriting
its inputs, and most stages emit a JSON QC report alongside the data.

## Functional

Scripts in [`functional/`](functional/), run in order:

| Script | Description |
|---|---|
| `00_verify_dicoms.py` | Verifies raw multiband EPI and MP2RAGE DICOM series against the expected acquisition protocol (geometry, timing, phase-encoding); read-only. |
| `01_trim_cut_topupinputs.py` | Builds topup fieldmap inputs (first 5 AP/PA volumes) and trims/thresholds each BOLD run, writing an initial FMR. |
| `02_slicetiming.py` | Slice-timing correction via a per-slice Fourier (sinc) phase shift, using each run's own JSON sidecar `SliceTiming` array. |
| `03_motion.sh` | Motion correction via FSL `mcflirt`, registered to a common run-1 reference volume. |
| `03b_motion_qc.py` | Motion QC from the `mcflirt` `.par` files: framewise displacement, flagged high-motion volumes, between-run jumps. |
| `04_highpass.py` | Temporal high-pass filtering via a BrainVoyager-style Fourier sine/cosine regression. |
| `05_topup.sh` | Estimates and applies FSL `topup` susceptibility-distortion correction. |
| `05a_topup_estimate.sh` | Runs the (slow) `topup` field-estimation step alone, parallelizable with motion correction. |
| `05b_applytopup.sh` | Applies a previously estimated `topup` field to the high-pass-filtered run. |
| `06_ants_distortion.py` | ANTs SyN-based distortion correction (blip-up/blip-down midpoint scheme), evaluated as an alternative to `topup`. |
| `07_final_fmr.py` | Writes the final preprocessed FMR per run, selecting `topup` or ANTs correction per run from the distortion QC. |
| `08_distortion_qc.py` | Compares AP/PA b0 similarity (NCC) before/after `topup`/ANTs correction and determines the better method per run. |
| `09_anat_overlay_qc.py` | Robust, multi-seed EPI-to-anatomy rigid-registration QC with a documented pass/warn/fail gate. |
| `topup_config_7T.cnf` | FSL `topup` configuration file tuned for 7T fieldmap estimation. |

## Anatomical

Scripts in [`anatomical/`](anatomical/), run in order:

| Script | Description |
|---|---|
| `01_denoise_uni.py` | MP2RAGE UNI background-noise removal (O'Brien 2014 / Marques robust combination). |
| `02_n4_bias_correct.py` | N4 bias-field correction of the denoised UNI image, computed in a single pass (bias field, then corrected = uni / bias). |
| `03_make_vmr.py` | Builds the BrainVoyager VMR/V16 anatomical volume from the N4-corrected NIfTI, with generic framing-cube centering and a verified axis-mapping. |
| `04_qc_final.py` | Quantitative VMR QC: 8-bit clipping/saturation, white-matter intensity uniformity, and a brain-restricted, multi-line Gibbs-ringing gate. |
| `qc_montage.py` | Reusable tri-planar slice montage / before-after comparison helper (matplotlib). |

## Registration

Scripts in [`registration/`](registration/), run in order:

| Script | Description |
|---|---|
| `00_verify_inputs.py` | Verifies VMR/V16/FMR geometry and cross-run grid consistency before registration; read-only. |
| `01_meanfunc.py` | Computes per-run mean functional and mean b0 (uncorrected / topup-corrected) images used as registration targets. |
| `02_coregister.py` | Two-stage (Initial Alignment/translation -> Fine Alignment/rigid, Mattes MI) functional-to-anatomical coregistration, with an independent run-4 cross-check. |
| `02b_run4_crosscheck_warmstart.py` | Warm-started re-registration of run-4 to test whether the earlier cross-check divergence was an optimizer convergence failure. |
| `03_distortion_residual_recheck.py` | Re-checks the distortion-correction QC with a proper staged rigid fit and Mattes MI / edge-gradient-correlation metrics. |
| `03b_distortion_residual_recheck_warmstart.py` | Repeats the distortion-residual recheck with warm-started registration for more reliable convergence. |
| `03c_visual_qc_run3.py` | Visual QC figures (side-by-side comparison, edge overlay, difference map) for the distortion-residual recheck. |
| `04_make_vtc.py` | Builds the final analysis-ready VTC per run by resampling the FMR through the functional-to-anatomical rigid transform. |
| `05_vtc_qc.py` | VTC QC: cross-run dimension/resolution consistency, NaN/Inf check, and a VMR/VTC overlay figure. |
| `06_lead1_ap_pa_motion_check.py` | Independently quantifies real AP/PA head motion via direct rigid registration, compared against `topup`'s internal movement model. |
| `07_lead2_syn_refinement.py` | ANTs SyN nonlinear refinement on top of the rigid functional-to-anatomical transform, computed per run. |
| `08_lead2_visual_qc.py` | Visual QC for the SyN refinement (rigid-only vs. SyN-refined comparison figures). |
| `09_lead2_make_refined_vtc.py` | Builds refined VTCs using the combined rigid + SyN transform chain. |

## Inputs / Outputs

- **Input**: multiband EPI BOLD DICOMs (AP phase-encoding, 4 runs, plus a
  short PA spin-echo pair for fieldmap estimation) and MP2RAGE DICOMs
  (UNI, INV1, INV2, T1 map), converted to NIfTI + JSON sidecars (e.g. via
  `dcm2niix`) upstream of this pipeline.
- **Intermediate**: per-stage NIfTI volumes (trimmed, slice-time-corrected,
  motion-corrected, high-pass-filtered, distortion-corrected) and an initial
  FMR per run; a denoised, bias-corrected anatomical NIfTI.
- **Output**: a final preprocessed FMR per functional run, an anatomical
  VMR/V16 pair, and a final co-registered VTC per run — all BrainVoyager-
  native formats, readable/writable without the BrainVoyager GUI via
  `bvbabel`. Every stage also writes a JSON QC report and, where relevant,
  PNG QC figures.

## Requirements

Python packages actually imported across these scripts (see
[`requirements.txt`](requirements.txt)):

- [`numpy`](https://numpy.org/) — array processing
- [`scipy`](https://scipy.org/) — image filtering (`scipy.ndimage.sobel`)
- [`nibabel`](https://nipy.org/nibabel/) — NIfTI I/O
- [`pydicom`](https://pydicom.github.io/) — DICOM header inspection
- [`antspyx`](https://github.com/ANTsX/ANTsPy) (imported as `ants`) — image
  registration (rigid, SyN) and N4 bias-field correction
- [`bvbabel`](https://github.com/ofgulban/bvbabel) — BrainVoyager FMR/VMR/
  V16/VTC read/write
- [`matplotlib`](https://matplotlib.org/) — QC figures
- [`scikit-learn`](https://scikit-learn.org/) — k-means tissue clustering
  in the anatomical QC step

Non-Python, external dependencies:

- **FSL** — `topup`, `applytopup`, `mcflirt`, and supporting FSL command-line
  tools, invoked from the `.sh` scripts. Requires a working FSL installation
  with `$FSLDIR` set.
- **dcm2niix** — for the raw DICOM -> NIfTI conversion step that precedes
  `01_trim_cut_topupinputs.py` (not itself included in this repository).

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## How to run

All scripts resolve their data paths relative to a `PROJECT_ROOT`
environment variable (defaulting to the parent of the `scripts/` directory
containing the running script if unset), so the pipeline is portable across
machines:

```python
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
```

Shell scripts use the equivalent convention:

```bash
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
```

Set `PROJECT_ROOT` explicitly to point at your data root, e.g.:

```bash
export PROJECT_ROOT=/path/to/my/dataset
export FSLDIR=/path/to/fsl
```

Scripts expect (and write into) a `derivatives/sub-01/` directory tree under
`PROJECT_ROOT`, laid out generically as:

```
PROJECT_ROOT/
├── sourcedata/
│   ├── dicom/            # raw DICOM series
│   └── prt/               # stimulus protocol (PRT) files
├── derivatives/
│   └── sub-01/
│       ├── rawdata_nifti/ # DICOM -> NIfTI conversion output + JSON sidecars
│       ├── func/          # functional/ stage outputs (per-run NIfTI + FMR)
│       ├── anat/          # anatomical/ stage outputs (VMR, V16, QC)
│       └── reg/           # registration/ stage outputs (VTC, transforms, QC)
├── logs/                  # JSON QC reports and QC figures from every stage
└── scripts/
    ├── functional/
    ├── anatomical/
    └── registration/
```

Run each stage's scripts in the numeric order listed in the tables above;
later scripts assume the outputs of earlier ones are already present under
`derivatives/sub-01/`.

---

Extracted from a larger 7T fMRI research pipeline for portfolio purposes.
Paths and the participant identifier have been genericized; no raw data,
DICOM, or participant data is included in this repository.
