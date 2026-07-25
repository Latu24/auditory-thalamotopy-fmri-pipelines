# Thalamus CF/AM Statistical Testing Suite (R)

## Overview

This is the R statistical-testing suite for a 7T fMRI single-subject study of tonotopic (characteristic-frequency, CF) and amplitude-modulation-rate (AM) tuning in the human auditory thalamus. Each voxel in the thalamus mask that showed significant tuning was assigned a best-fit CF and/or AM frequency from a GLM analysis; the question addressed here is whether CF tuning and AM tuning are organized **independently** or **jointly** within the thalamus.

The suite approaches this question from three complementary angles: (1) median-split group comparisons and chi-square/Fisher's exact tests of independence on the subset of voxels significantly tuned to both CF and AM ("conjunction" voxels), (2) full-population spatial overlap tests between High/Low CF and High/Low AM voxel groups against the entire thalamus mask, and (3) a spatial nearest-neighbor permutation test that asks whether high-preference voxels of each modality cluster together in physical space beyond the baseline CF/AM spatial co-localization already present in the thalamus. Companion scripts generate the corresponding visualizations (scatter plots, histograms, bar charts) for each test.

## Scripts

1. **`30_cfam_median_split_chisq_test.R`** — Ranks all CF-significant and AM-significant thalamic voxels by best-fit frequency, splits each at its own median, then runs a chi-square test of independence and Fisher's exact test on the resulting 2x2 High/Low CF x AM table for the CF/AM conjunction voxels.
2. **`31_cfam_median_split_chisq_plot.R`** — Scatter plot of conjunction voxels' CF vs. AM frequency, with median-split crosshairs and quadrant coloring, visualizing the classification used by the chi-square test.
3. **`32_cfam_fullthalamus_median_split_plot.R`** — Strip plots of the full CF-significant and AM-significant voxel populations (not limited to conjunction voxels), each colored by its own High/Low median split.
4. **`33_cfam_fullthalamus_overlap_test.R`** — Tests spatial overlap between High/Low CF and High/Low AM voxel groups against the full thalamus mask, using chi-square and Fisher's exact tests on 2x2 voxel-count contingency tables for all four group pairings; also produces an odds-ratio bar chart.
5. **`34_cfam_median_split_histogram.R`** — Histograms of the full CF and AM frequency distributions, each colored by its own Low/High median split.
6. **`35_cfam_extremegroups_scatter.R`** — Extreme-groups scatter of CF vs. AM for conjunction voxels, shading the lower/upper tail of CF values (33% and 27% cuts) and comparing mean AM between the extreme groups via t-test.
7. **`36_cfam_fullthalamus_overlap_plots.R`** — Re-plots the full-thalamus overlap test results as a cleaned-up odds-ratio bar chart and a chi-square statistic bar chart with a significance-threshold reference line.
8. **`37_cfam_highfreq_spatial_nn_test.R`** — Permutation-based spatial nearest-neighbor test (5000 permutations) asking whether High-CF/High-AM (and other pairing) voxels cluster more tightly in physical space than expected from label-shuffled nulls, without using conjunction voxels.
9. **`38_cfam_extremegroups_scatter_noconj.R`** — Repeats the extreme-groups scatter using CF-only and AM-only voxels (no conjunction voxels), pairing each voxel's real frequency value with the corresponding sub-threshold fit on the other axis.
10. **`39_cfam_extremegroups_scatter_separate.R`** — Corrected version of script 38 that keeps the CF-only and AM-only populations, split variables, and t-tests fully separate rather than pooling them into a single sample.

## Inputs / Outputs

- **Inputs**: per-voxel CF and AM best-fit-frequency tables (JSON), a CF/AM conjunction voxel table (JSON), a sub-threshold CF/AM fit table (JSON), and a binary thalamus mask (NIfTI). These are derived from an upstream GLM fitting pipeline (not included in this repository).
- **Outputs**: JSON summaries of test statistics (chi-square, Fisher's exact, t-test, permutation p-values) and PNG figures (scatter plots, histograms, bar charts) visualizing each test.

## Requirements

R packages used across these scripts:

- [`jsonlite`](https://cran.r-project.org/package=jsonlite) — reading/writing JSON input and output files
- [`RNifti`](https://cran.r-project.org/package=RNifti) — reading the NIfTI thalamus mask (script 33 only)

Statistical tests and plotting rely entirely on base R (`stats`: `chisq.test`, `fisher.test`, `t.test`, `lm`, `median`, `quantile`; `graphics`: `plot`, `barplot`, `hist`, `png`) — no `ggplot2`, `dplyr`, or other tidyverse packages are used.

Install the two external dependencies with:

```r
install.packages(c("jsonlite", "RNifti"))
```

## How to run

Each script resolves its data paths relative to a `PROJECT_ROOT` environment variable (defaulting to `..`, i.e. the parent of this scripts directory), so no machine-specific paths are hardcoded. Set `PROJECT_ROOT` to point at a directory containing a `derivatives/sub-01/analysis/...` layout matching the inputs described above, then run any script with `Rscript`:

```bash
export PROJECT_ROOT=/path/to/project/root
Rscript 30_cfam_median_split_chisq_test.R
```

Scripts that consume another script's output (e.g. `31` reads `30`'s JSON result, `36` reads `33`'s JSON result) should be run in numeric order.

## Note

Extracted from a larger 7T fMRI research pipeline for portfolio purposes. Paths and the participant identifier have been genericized; no raw data or participant data is included in this repository.
