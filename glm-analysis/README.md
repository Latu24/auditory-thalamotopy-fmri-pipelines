# GLM Analysis Pipeline — 7T Auditory Thalamotopy fMRI

## Overview

This is the general linear model (GLM) analysis pipeline for a single-subject
7T fMRI study of tonotopic (best-frequency) and amplitude-modulation-rate
mapping in the human auditory thalamus and cortex. It is built entirely
without the BrainVoyager GUI: it reads BrainVoyager VTC (functional data) and
PRT (protocol / condition timing) files directly via
[`bvbabel`](https://github.com/ofgulban/bvbabel), constructs design matrices
(HRF-convolved box-car regressors — both a hand-rolled two-gamma
implementation and an SPM-canonical / nilearn-based construction, plus custom
BrainVoyager SDM file generation), runs voxelwise GLM regression with AR(2)
prewhitening, and writes statistical results back out as native BrainVoyager
VMP (map) and GLM files so they can be inspected directly in BrainVoyager.

The scripts trace a progression of analyses: a first-pass sanity-check GLM,
a sound-vs-silence localizer, carrier-frequency (tonotopy) and
amplitude-modulation-rate mapping, a combined/ridge-regularized design, a
crossed 2x2 (frequency x AM-rate) interaction design with proper statistical
thresholding, and finally a set of spatially/temporally smoothed re-analyses
and refits used to compare against unsmoothed results. Every stage reads
VTC/PRT/SDM inputs and writes derived statistical maps, beta tables, or
BrainVoyager-native files — no raw or intermediate data is modified in
place.

## Pipeline stages

| Script | Description |
|---|---|
| `01_sanity_soundon.py` | First-pass sanity-check GLM: a single SoundOn block regressor on run 1 only, confirming a plausible sound-vs-baseline response before building the full designs. Writes a JSON summary and a QC overlay PNG. |
| `02_sound_vs_silence.py` | Per-run and multi-run fixed-effects GLM for SoundOn vs. baseline; writes a 5-sub-map VMP (fixed-effects + 4 individual runs). |
| `03_carrier_frequency.py` | Fixed-effects GLM over 36 carrier-frequency conditions (via `event_family.py`): omnibus F, all-tones response, voxel-level best-frequency map, ordinal frequency contrast. |
| `04_amplitude_modulation.py` | Same design as above for the 9 amplitude-modulation-rate conditions. |
| `05_beta_tables.py` | Refits the CF and AM GLMs at full-volume resolution to recover per-voxel betas (not persisted by the map-only scripts); exports plain-text beta tables for the activated cluster and for FDR-significant (BH, q<0.05) voxels. |
| `05_cf_refined.py` | Reruns the carrier-frequency GLM on refined (additionally registered) VTCs with SDM-sourced predictors; writes 36 per-condition beta maps plus omnibus-F / all-tones / best-frequency summary maps. |
| `06_am_refined.py` | Same treatment as `05_cf_refined.py` for the 9 AM conditions. |
| `06_cluster_extent_filter.py` | Applies 26-connectivity spatial cluster-extent filtering (minimum 10 contiguous voxels) on top of the FDR-significant beta tables from `05_beta_tables.py`. |
| `07_build_combined_sdm.py` | Builds a combined 45-predictor (36 CF + 9 AM) BrainVoyager SDM per run using hand-rolled HRF convolution. |
| `07_combined_ridge.py` | Fits the combined 45-condition design with ridge regression (GCV-selected lambda), required because the CF and AM predictor blocks are severely collinear; produces descriptive-only beta/summary maps (no t/F inference, since ridge estimates are biased). |
| `08_build_sdm_nilearn.py` | Rebuilds the combined 45-predictor SDM using nilearn's validated `make_first_level_design_matrix` HRF convolution instead of the hand-rolled version, with a correlation cross-check against it. |
| `09_build_separate_sdm.py` | Builds separate, well-conditioned CF-only and AM-only SDMs via nilearn — supersedes the combined-SDM approach for designs that don't need ridge regularization. |
| `10_build_sdm_crossed_2x2.py` | Builds SDMs for a crossed CF x AM 2x2 design (4 cells: CFlo_AMlo, CFlo_AMhi, CFhi_AMlo, CFhi_AMhi), enabling a genuine statistical interaction test that the marginal 45-predictor design cannot express. |
| `11_glm_crossed_2x2_interaction.py` | Fits the well-conditioned crossed 2x2 GLM (plain OLS) and computes the CF x AM interaction contrast t-map, restricted to/around a native-space thalamus mask; checkpoints the fit for reuse. |
| `12_threshold_interaction.py` | Applies three significance thresholds to the interaction t-map within the thalamus mask: uncorrected p<0.05, FDR q<0.05 (Benjamini-Hochberg), and Monte Carlo cluster-extent correction (AFNI-3dClustSim-style). |
| `13_threshold_interaction_wholebrain.py` | Same three threshold methods as above, applied to the entire functional-coverage volume rather than just the thalamus. |
| `14_cf_am_maineffects_from_crossed.py` | Computes CF-main and AM-main effect contrasts from the same crossed 2x2 fit (an independent replication check against the marginal-design results), applying the same threshold methods across both thalamus and whole-brain scopes. |
| `15_smoothed_reanalysis.py` | Diagnostic: applies post-hoc spatial smoothing (assumed FWHM=4mm) directly to already-fitted t-maps, to estimate how much smoothing alone could broaden activation patterns without a full refit. |
| `16_full_smoothed_refit.py` | Full VTC-level spatial + temporal smoothing followed by a complete GLM refit of the crossed 2x2 design (not just a post-hoc shortcut); AR(2) is re-estimated on the smoothed data. |
| `17_export_smoothed_crossed_glm.py` | Exports the smoothed crossed-2x2 GLM fit as a native BrainVoyager `.glm` file plus VMP contrast maps (with proper FDR tables), reconstructing R2/SS algebraically from the checkpointed fit. |
| `18_marginal_45cond_smoothed_sanity.py` | Refits the earlier marginal 45-condition design with matching spatial/temporal smoothing and ridge regularization, as a smoothed sanity-test comparison point. |
| `19_export_smoothed_45cond_glm.py` | Exports the smoothed 45-condition ridge GLM checkpoint as a native BrainVoyager `.glm` file, handling the ridge-adjusted SS/R2 reconstruction (which differs from the plain-OLS case). |
| `20_reexport_45cond_vmp_fixed.py` | Re-exports the 45-condition smoothed ridge VMP from the existing checkpoint with a corrected bounding-box Z-offset and separate CF/AM color ceilings. |
| `21_cf_smoothed_glm.py` | CF-only smoothed GLM (well-conditioned, no ridge needed); produces valid per-condition beta/t-maps and an omnibus F-map, plus a BrainVoyager `.glm` export. |
| `22_am_smoothed_glm.py` | Same treatment as `21_cf_smoothed_glm.py` for the 9 AM-only conditions. |

### Shared library modules

| Module | Provides |
|---|---|
| `common.py` | Shared PROJECT_ROOT-relative paths, condition orderings (`FREQ_ORDER`, `AM_ORDER`), QC t-map overlay plotting, and JSON logging helpers used by the early driver scripts. |
| `common2.py` | Additional paths for the refined-VTC / SDM-based analysis round (refined VTC paths, corrected VMR path, CF/AM SDM paths) — additive on top of `common.py`. |
| `event_family.py` | Shared GLM runner for the carrier-frequency and amplitude-modulation "family" analyses: fits the multi-run GLM and produces omnibus F, all-tones, best-condition, and ordinal parametric contrast maps. |
| `glm_writer.py` | Reverse-engineered BrainVoyager `.glm` file writer — `bvbabel` can only read `.glm` files, so this implements the inverse of its byte layout, matching real BrainVoyager GLM file conventions. |
| `glmlib.py` | Core GLM engine: VTC header/memmap reading, two-gamma HRF construction, box-car design matrices, AR(2) prewhitening, memory-safe slab-wise multi-run GLM fitting, t/F contrasts, and VMP writing. |
| `glmlib2.py` | Extension of `glmlib.py` adding SDM-based predictor reading, refined-VTC support, and ridge-regularized (GCV-tuned) GLM fitting for ill-conditioned designs. |
| `vmp_fdr_table.py` | Computes BrainVoyager-style FDR significance tables (Benjamini-Hochberg "critical std" and Benjamini-Yekutieli "critical conservative" thresholds per q-level) for embedding in exported VMP/GLM files. |

## Inputs / Outputs

**Inputs:**
- BrainVoyager VTC files — coregistered functional (BOLD) data, one per run.
- BrainVoyager PRT files — condition/event timing protocols (tone onsets
  grouped by carrier frequency, AM rate, or crossed CF x AM cell).
- BrainVoyager SDM files — precomputed design-matrix predictor files (built
  by the `0*_build_sdm_*.py` scripts) for some later analyses.

**Outputs:**
- BrainVoyager VMP files — voxelwise statistical maps (beta, t, F, omnibus F)
  aligned to the anatomical VMR, with proper FDR tables and cluster-extent
  metadata where applicable.
- BrainVoyager GLM files — full native-format GLM fit objects (design
  matrix, betas, R2, AR terms) that can be opened directly in BrainVoyager.
- Plain-text beta tables and JSON logs — per-voxel beta values and analysis
  metadata (degrees of freedom, AR(2) parameters, thresholds, voxel counts)
  for downstream inspection and reporting.
- QC overlay PNGs — quick-look t-map overlays on an anatomical background.

## Requirements

Third-party packages actually imported by these scripts:

```
numpy
scipy
pandas
matplotlib
nilearn
nibabel
bvbabel
```

Install with:

```bash
pip install -r requirements.txt
```

## How to run

All path configuration goes through a `PROJECT_ROOT` environment variable
convention, defined identically at the top of every script:

```python
import os
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
```

Set `PROJECT_ROOT` to point at a directory containing the expected
`derivatives/sub-01/...`, `PRTs/`, and `docs/` layout, then run any script
directly, e.g.:

```bash
export PROJECT_ROOT=/path/to/your/project
python 03_carrier_frequency.py
```

If `PROJECT_ROOT` is not set, it defaults to the parent of the directory the
script lives in. Scripts are numbered in the order they were developed and
generally build on outputs from earlier scripts (e.g. the `0*_build_sdm_*.py`
scripts must run before the analyses that read their SDM output); they are
not required to run end-to-end as a single automated pipeline.

## Note

Extracted from a larger 7T fMRI research pipeline for portfolio purposes.
Paths and the participant identifier have been genericized; no raw data or
participant data is included in this repository.
