# Thalamus vs. Cortex Tonotopy Statistics (R)

## Overview

This R suite compares how characteristic-frequency (CF) tuning and amplitude-modulation-rate (AM) tuning are spatially organized in the human auditory thalamus versus auditory cortex, using per-voxel best-frequency estimates from the same subject's 7T fMRI GLM results. Each analysis is implemented once for the thalamus and once more for auditory cortex (the `_cortex` suffix), so that identical statistical tests and plots can be compared side by side across the two regions.

The suite covers chi-square tests of independence between "high" and "low" CF/AM voxel groups (with a fixed-threshold sensitivity-analysis variant alongside the primary data-driven median-split version), correspondence and correlation scatterplots relating CF to AM best frequency, several bar-chart views of voxel counts and overlap, and two dedicated scripts that directly contrast thalamus and cortex statistics (significance rate, overlap rate, distribution shape, and CF-AM correlation strength) in a single report.

## Scripts

### Chi-square independence tests
Tests whether "high" vs. "low" CF membership is statistically independent of "high" vs. "low" AM membership, via a 2x2 contingency table and `chisq.test`.

| Script | Description |
|---|---|
| `cf_am_chisq_independence.R` | Thalamus: independence test using data-driven median-split High/Low groups; reports the contingency table, standardized residuals, and conditional proportions to a text file. |
| `cf_am_chisq_independence_cortex.R` | Auditory cortex counterpart of the above. |
| `cf_am_chisq_independence_fixedthreshold.R` | Thalamus: same test, but High/Low groups are defined by fixed absolute cutoffs (CF ≥ 4000 Hz, AM ≥ 8 Hz) instead of the median split — a sensitivity check on the thresholding method. |
| `cf_am_chisq_independence_fixedthreshold_cortex.R` | Auditory cortex counterpart of the fixed-threshold test. |

### Barcharts
Visualize voxel counts and group sizes underlying the independence tests.

| Script | Description |
|---|---|
| `cf_am_clustered_barchart.R` | Thalamus: clustered bar chart of voxel counts by CF group, split by AM group (median-split), annotated with the chi-square statistic. |
| `cf_am_clustered_barchart_cortex.R` | Auditory cortex counterpart. |
| `cf_am_clustered_barchart_fixedthreshold.R` | Thalamus: clustered bar chart using the fixed-threshold groups. |
| `cf_am_clustered_barchart_fixedthreshold_cortex.R` | Auditory cortex counterpart of the fixed-threshold clustered bar chart. |
| `cf_am_fixedthreshold_barchart.R` | Thalamus: voxel counts for High/Low CF and High/Low AM shown as four separate bars, fixed-threshold grouping. |
| `cf_am_fixedthreshold_barchart_cortex.R` | Auditory cortex counterpart. |
| `cf_am_median_split_barchart.R` | Thalamus: voxel counts for High/Low CF and High/Low AM shown as four separate bars, median-split grouping. |
| `cf_am_median_split_barchart_cortex.R` | Auditory cortex counterpart. |
| `cf_am_overlap_barchart.R` | Thalamus: bar chart of voxels significant for CF only, AM only, or both, quantifying overlap between the two tonotopic maps. |
| `cf_am_overlap_barchart_cortex.R` | Auditory cortex counterpart. |
| `cf_am_percentage_barchart.R` | Thalamus: CF and AM voxel counts (overall and by High/Low group) expressed as a percentage of all significant voxels (union of the CF and AM maps). |
| `cf_am_percentage_barchart_cortex.R` | Auditory cortex counterpart. |

### Scatter / correlation
Relate CF and AM best-frequency values voxel-by-voxel.

| Script | Description |
|---|---|
| `cf_am_group_correspondence_scatter.R` | Thalamus: CF-vs-AM scatterplot (log-log) colored by CF/AM quadrant (High/Low x High/Low) with median-split reference lines, testing whether high-CF voxels co-occur with high-AM voxels. |
| `cf_am_group_correspondence_scatter_cortex.R` | Auditory cortex counterpart. |
| `cf_am_scatter_correlation.R` | Thalamus: CF-vs-AM scatterplot with a linear fit; reports Pearson correlation (log10-transformed frequencies) and Spearman correlation (raw frequencies). |
| `cf_am_scatter_correlation_cortex.R` | Auditory cortex counterpart. |
| `median_split_scatterplots.R` | Thalamus: spatial scatterplots of the CF and AM median-split groups plotted in in-plane world coordinates, showing the anatomical layout of each frequency group. |
| `median_split_scatterplots_cortex.R` | Auditory cortex counterpart. |

### Direct thalamus-vs-cortex comparison
Statistically contrast the two regions directly rather than analyzing each in isolation.

| Script | Description |
|---|---|
| `thalamus_vs_cortex_comparison.R` | Four-part comparison: (1) proportion test of significance rate relative to ROI size, (2) proportion test of the CF/AM dual-significance (overlap) rate, (3) Kolmogorov-Smirnov test comparing best-frequency distribution shape (log10 Hz), (4) Fisher r-to-z test comparing CF-AM correlation strength between regions. Writes a combined text report. |
| `thalamus_vs_cortex_correlation_comparison.R` | Companion figures for the comparison above: (A) faceted CF-vs-AM scatterplots with linear fits for each region, (B) a point-range plot of the two regions' Pearson correlation estimates with 95% confidence intervals, both annotated with the Fisher r-to-z statistic. |

## Inputs / Outputs

**Inputs:** per-voxel CF and AM best-frequency tables (one row per significant voxel, with voxel index, world coordinates, best frequency in Hz, and a median-split High/Low group label), provided separately for the thalamus ROI and the auditory-cortex ROI.

**Outputs:** console/text-file summaries of each statistical test (contingency tables, chi-square/correlation/proportion/KS test results) and PNG figures (bar charts and scatterplots) visualizing the same comparisons.

No raw or derived data files are included in this repository — only the analysis code.

## Requirements

- R (base installation: `stats` functions such as `chisq.test`, `cor.test`, `prop.test`, `ks.test`, and `lm` are used throughout)
- [`ggplot2`](https://ggplot2.tidyverse.org/) — the only external package loaded via `library()`, used in all plotting scripts

```r
install.packages("ggplot2")
```

## How to run

Each script resolves its input/output paths relative to a `PROJECT_ROOT` environment variable, defaulting to `..` (i.e., the parent of this directory) if unset:

```r
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "..")
```

Input tables are expected under `PROJECT_ROOT/derivatives/sub-01/analysis/smoothed_glm/`, and figure outputs are written under `PROJECT_ROOT/outputs/`. To point the scripts at your own data layout, set `PROJECT_ROOT` before running, e.g.:

```bash
PROJECT_ROOT=/path/to/your/project Rscript cf_am_chisq_independence.R
```

or from within R:

```r
Sys.setenv(PROJECT_ROOT = "/path/to/your/project")
source("cf_am_chisq_independence.R")
```

## Note

Extracted from a larger 7T fMRI research pipeline for portfolio purposes. Paths and the participant identifier have been genericized; no raw data or participant data is included in this repository.
