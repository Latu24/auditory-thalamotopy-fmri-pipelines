# Auditory Thalamotopy — 7T fMRI Analysis Pipelines, when you have no access to BrainVoyager
— this pipeline aims to help Maastricht students that have no access to BrainVoyager but are still required to submit Brainvoyager File types.

Code from a 7T fMRI study mapping how the human auditory thalamus and
auditory cortex represent two independent dimensions of sound: **carrier frequency**
(tonotopy — "which pitch") and **amplitude-modulation rate** ("how fast the sound
flutters"). The central scientific question is whether these two dimensions are encoded
independently or jointly at the level of individual voxels, and whether the
thalamus and cortex organize them the same way.

raw DICOMs are turned into
BrainVoyager-native FMR/VMR/VTC files programmatically via [`bvbabel`](https://github.com/ofgulban/bvbabel),
distortion correction uses FSL `topup` and ANTs (`antspyx`), and the GLM (design matrix,
regression, contrasts) is implemented from scratch.

## What's here

This repo is organized as five independent-but-sequential pipelines, each a self-contained
folder with its own README, requirements, and numbered scripts:

| Folder | What it does |
|---|---|
| [`preprocessing/`](preprocessing/) | Raw DICOM → analysis-ready data: functional preprocessing (slice-timing, motion correction, distortion correction), anatomical preprocessing (MP2RAGE denoising, bias correction), and functional-to-anatomical registration producing final VTC files. |
| [`glm-analysis/`](glm-analysis/) | Builds design matrices and runs the voxelwise GLM: sanity checks, sound-vs-silence, carrier-frequency, amplitude-modulation, and a crossed 2×2 (frequency × AM) interaction model, plus smoothed re-analyses. |
| [`thalamus-3d/`](thalamus-3d/) | From GLM betas to best-frequency-per-voxel maps, atlas-based thalamus/MGB and auditory-cortex segmentation, CF×AM conjunction statistics, and an interactive 3D model (browser HTML + USDZ for AR viewing). |
| [`r-stats-thalamus-cf-am/`](r-stats-thalamus-cf-am/) | R statistical suite testing whether CF and AM tuning are independently or jointly organized within the thalamus (median-split comparisons, chi-square independence tests, spatial nearest-neighbor overlap tests). |
| [`r-stats-thalamus-vs-cortex/`](r-stats-thalamus-vs-cortex/) | R suite directly comparing CF/AM tuning organization between thalamus and cortex — paired statistical tests, correspondence scatterplots, and barcharts for both regions side by side. |

The intended flow is `preprocessing → glm-analysis → thalamus-3d`, with the two R folders
providing the statistical follow-up once best-frequency maps exist for both regions.

## Requirements

Python 3.9+, R 4.x, and (for `preprocessing/`) FSL and `dcm2niix` as external non-Python
dependencies. See each folder's `requirements.txt` / README for exact package lists.

## License

MIT — see [LICENSE](LICENSE).
