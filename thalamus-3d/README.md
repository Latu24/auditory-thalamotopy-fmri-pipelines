# Auditory Thalamotopy: GLM to Interactive 3D Model

Analysis pipeline for a 7T fMRI study of tonotopic (best-frequency) and
amplitude-modulation-rate mapping in the human auditory thalamus and
auditory cortex (single subject).

## Overview

This pipeline goes from voxelwise GLM beta/contrast maps to (a) a
best-frequency-per-voxel map for both carrier-frequency (tonotopy) and
amplitude-modulation-rate tuning, computed via a winsorized Gaussian fit in
log-frequency space; (b) atlas-based segmentation of the thalamus — with
special handling for the medial geniculate body (MGB), the auditory thalamic
relay nucleus — and of auditory cortex, using ANTs SyN registration to
standard atlases rather than a full FreeSurfer pipeline; (c) statistical
characterization of how carrier-frequency (CF) and amplitude-modulation
(AM) tuning relate and spatially overlap, including conjunction analysis,
contingency-table and rank-correlation tests, and artifact checks; and (d)
an interactive 3D model of the thalamus, colored by best frequency, exported
both as a self-contained browser HTML file (Plotly) and as a USDZ file for
native AR viewing in macOS/iOS Preview and Quick Look.

The pipeline includes a number of methodological checks that are unusual to
see shipped alongside the "main" analysis: a smoothing dose-response sweep
that found pre-test spatial smoothing *reduces* rather than increases the
number of significant voxels (consistent with a collinear GLM design
producing sparse, spatially-isolated signal rather than a smooth spatial
map); a spatial-coherence permutation test distinguishing real tuning
structure from noise in a sparsely-covered ROI; independence proofs for the
two tuning dimensions; and a boundary-artifact check for curve-fitting
saturation at the edges of the tested frequency range. These are kept as
separate diagnostic scripts alongside the main deliverables rather than
folded in silently.

## Thalamus

Ordered list of scripts in `thalamus/`:

1. **`10_bestfreq_from_glm.py`** — Reads the GLM once and computes, per voxel, a winsorized-and-Gaussian-fit best frequency (continuous Hz + categorical condition index) for the carrier-frequency (36 conditions) and amplitude-modulation-rate (9 conditions) predictor families, gated by functional coverage; also computes a diagnostic omnibus F-test and FDR q-values per family.
2. **`10b_smoothing_ablation.py`** — Diagnostic: compares raw / winsorized / spatially-smoothed (sigma 0.6, 1.2) omnibus-F significance counts to isolate why pre-test smoothing reduces (not increases) the number of significant voxels.
3. **`10c_smoothing_thalamus_doseresponse.py`** — Diagnostic: sweeps smoothing sigma (0.0–1.2) and reports FDR-significant voxel counts specifically inside the native thalamus/MGB masks, to check whether any smoothing level preserves visible thalamic signal.
4. **`11_segment_thalamus.py`** — Segments the whole thalamus and the MGB in native space via two ANTs SyN registrations (ICBM152→native for a Sitek in-vivo MGB atlas; FSL MNI152→native for the Harvard-Oxford subcortical atlas), with connected-component cleanup and hole-filling.
5. **`12_reconcile_space.py`** — Empirically validates the GLM-framebox-to-native-NIfTI axis transform (brute-force permutation/flip search maximizing functional-in-brain precision), then applies it to write native-space best-frequency NIfTIs, a BrainVoyager VMP, and QC overlay PNGs.
6. **`13_build_3d_models.py`** — Builds two self-contained, offline-viewable interactive HTML 3D models of the thalamus (CF and AM best-frequency) using marching cubes, watertightness verification, and Taubin mesh smoothing, with per-vertex color sampled from the best-frequency volume.
7. **`14_mgb_appearance_check.py`** — Diagnostic: tests whether the sparse, functionally-covered MGB voxels' best-frequency values show real spatial coherence (neighbors agree more than a shuffled null) or are indistinguishable from noise.
8. **`14b_mgb_smoothing_confound_check.py`** — Control diagnostic: re-fits the same MGB voxels from unsmoothed profiles to check whether the spatial coherence found in script 14 is a genuine effect or a mechanical artifact of pre-fit smoothing.
9. **`15_export_usdz.py`** — Exports the same thalamus/MGB meshes as USDZ files (vertex-baked color, meters, Z-up) for native rotate/zoom viewing in macOS/iOS Preview and Quick Look, alongside the HTML models.
10. **`16_cf_am_independence_check.py`** — Proves the CF and AM best-frequency computations are genuinely independent (disjoint predictor columns, non-identical F maps, differing native-space significant-voxel counts), addressing a coincidental matching voxel count in the coverage mask.
11. **`17_apply_p01_gate.py`** — Applies the final output-gating criterion — a p<0.01 uncorrected omnibus-F test, computed separately per condition family — to the already-fitted best-frequency maps as an in-place re-mask (no re-fitting).
12. **`18_cfam_conjunction_smoothed.py`** — Recomputes the CF×AM conjunction on two new, separately-fit (non-collinear) smoothed GLMs, with `.ctr`-contrast cross-verification and re-validated native-space placement, producing per-modality and conjunction voxel tables plus a scatter plot.
13. **`19_cfam_conjunction_graphics.py`** — Builds three additional conjunction figures from script 18's tables: a tri-color native-space slice montage, a bar chart of CF-only/AM-only/both voxel counts, and a CF-F vs AM-F scatter over the full union of active voxels.
14. **`20_cfam_beta_at_preferred.py`** — Adds a "beta value at the voxel's own preferred condition" column to the existing voxel tables, using a round-trip-verified inverse of the native-space transform to look the value up fresh from the GLM.
15. **`21_cfam_conjunction_stats.py`** — Inferential statistics on the conjunction: Fisher's exact / chi-square overlap enrichment test, Pearson/Spearman/Kendall correlation between preferred CF and AM values, and a median-split quadrant analysis.
16. **`22_cfam_freq_quadrant_plot.py`** — Scatter plot of preferred CF frequency vs. preferred AM rate for the conjunction voxels only.
17. **`23_cfam_subthreshold_fit.py`** — Computes descriptive, sub-threshold (non-significant) Gaussian fits for the "other" modality at CF-only and AM-only voxels, with a cross-check that these fits are indeed below their own significance threshold.
18. **`24_cfam_combined_quadrant_stats.py`** — Extends the quadrant analysis to the full CF-only/AM-only/conjunction population using the sub-threshold fits from script 23, reporting results by category for transparency.
19. **`25_cfam_combined_quadrant_plot.py`** — Plots the combined-population quadrant analysis from script 24, with per-quadrant voxel-count annotations.
20. **`26_cfam_highfreq_overlap_stats.py`** — Tests whether "prefers high CF" spatially co-occurs with "prefers high AM," using only each modality's own real significant voxels (no sub-threshold fits), via 2×2 contingency tables.
21. **`27_cfam_highfreq_overlap_plot.py`** — Bar chart comparing script 26's four High/Low overlap odds ratios against the baseline CF-active × AM-active enrichment odds ratio from script 21.
22. **`28_cfam_high_low_group_comparison.py`** — Direct two-group distributional comparison (Mann-Whitney U, t-test) of AM values by CF group and vice versa, as an alternative to the contingency-table framing.
23. **`29_cfam_edge_effect_check.py`** — Chi-square goodness-of-fit and binomial tests for whether preferred-condition indices pile up at the edges of the tested frequency range, a signature of bounded curve-fit saturation.

## Cortex

Ordered list of scripts in `cortex/`:

1. **`01_segment_auditory_cortex.py`** — Segments auditory cortex (Heschl's Gyrus, Planum Temporale, Planum Polare, and anterior/posterior Superior Temporal Gyrus) in native space via an ANTs SyN warp of the Harvard-Oxford cortical atlas, with hemisphere assignment from world-space X sign and connected-component cleanup.
2. **`02_bestfreq_tables_cortex.py`** — Mirrors the thalamus pipeline's masking → omnibus-F gate → Gaussian-fit → table logic exactly (imported, not reimplemented) but restricted to the native auditory-cortex mask, producing CF/AM best-frequency voxel tables in JSON and CSV.
3. **`03_cfam_freq_quadrant_plot_cortex.py`** — Scatter plot of preferred CF frequency vs. preferred AM rate for the auditory-cortex CF/AM voxel tables, mirroring the thalamus conjunction quadrant plot's style.

## Inputs / Outputs

**Inputs** (not included in this repository):
- BrainVoyager GLM files (`.glm`) containing per-voxel condition betas, sum-of-squares, R², and the design matrix's inverted X'X matrix, for the carrier-frequency and amplitude-modulation-rate predictor families.
- BrainVoyager contrast files (`.ctr`) defining the omnibus-F contrast for each predictor family.
- An anatomical volume (`.vmr`/NIfTI) in native space, used for registration targets and world-coordinate reporting.
- Standard-space MNI/ICBM templates and the Harvard-Oxford and Sitek in-vivo atlases, for ANTs-based ROI segmentation.

**Outputs**:
- Best-frequency-per-voxel maps (continuous Hz + categorical condition index), in native NIfTI and BrainVoyager VMP formats.
- Binary segmentation masks for the whole thalamus, the MGB, and auditory cortex (native space, per-hemisphere and merged).
- Per-voxel result tables (JSON and CSV) and statistical test results (contingency tables, correlations, group comparisons) for the CF/AM conjunction analysis.
- Interactive 3D models of the thalamus, colored by best frequency: self-contained HTML (any browser, via Plotly) and USDZ (native AR viewing in macOS/iOS Preview and Quick Look).
- QC overlay images and JSON methodology sidecars documenting each processing decision.

No raw or derived participant data is included in this repository — only the processing code.

## Requirements

Third-party Python packages used across these scripts (see `requirements.txt`):

- `numpy`, `scipy` — array computation, curve fitting, statistics, image filtering
- `nibabel` — NIfTI I/O
- `bvbabel` — BrainVoyager `.glm`/`.vmp` file I/O
- `statsmodels` — FDR (Benjamini-Hochberg) correction
- `antspyx` (imported as `ants`) — SyN registration for atlas-based segmentation
- `scikit-image` (imported as `skimage`) — marching cubes surface extraction
- `matplotlib` — static figures and colormaps
- `plotly` — interactive browser-based 3D HTML models
- `usd-core` (imported as `pxr`) — USDZ export for AR/Quick Look viewing
- `pandas` — tabular I/O for the cortex CSV pipeline

Also requires local installations of FSL (for the MNI152 template and
Harvard-Oxford atlases, referenced via `~/fsl/...`) and, for USDZ
verification, the macOS `usdcat` and `qlmanage` command-line tools.

Install with:

```bash
pip install -r requirements.txt
```

## How to run

Every script resolves its paths from a `PROJECT_ROOT` environment variable
(defaulting to the parent directory of the `scripts/` folder if unset):

```python
import os
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
```

Set `PROJECT_ROOT` to a directory containing your own `raw/`, `derivatives/`,
and (for atlas-based segmentation) FSL-style `atlas` data before running any
script, e.g.:

```bash
export PROJECT_ROOT=/path/to/your/project
export GLM_PATH=/path/to/your/subject.glm   # a few scripts also read this
python3 thalamus/10_bestfreq_from_glm.py
```

Scripts are numbered in pipeline order and are generally run sequentially
within each subfolder (thalamus scripts 10→29, then cortex scripts 01→03,
which depend on several thalamus-pipeline outputs). Several scripts import
functions directly from earlier numbered scripts via `importlib` rather than
duplicating logic — see each script's module docstring for its specific
upstream dependencies.

---

Extracted from a larger 7T fMRI research pipeline for portfolio purposes.
Paths and the participant identifier have been genericized; no raw data or
participant data is included in this repository.
