# Spatial overlap test between High/Low CF and High/Low AM voxel groups
# across the full thalamus mask: using the same full-thalamus CF (n=1075) and
# AM (n=1669) populations and median splits as the full-thalamus median-split
# plot script, this tests whether High-CF and High-AM voxels co-occur in the
# thalamus more than chance (and likewise for the other three High/Low
# pairings). Each pairing is evaluated with a chi-square test of independence
# and Fisher's exact test on a 2x2 voxel-count contingency table built
# against the full N-voxel thalamus mask.
#
# Inputs: per-voxel CF and AM best-fit-frequency tables (JSON) with native-
# space voxel indices, and the binary thalamus mask (NIfTI).
# Outputs: JSON summary of the four 2x2 overlap tests, and a bar chart (PNG)
# of the resulting odds ratios.
#
# Run: Rscript 33_cfam_fullthalamus_overlap_test.R

library(jsonlite)
library(RNifti)

PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "..")
ROOT <- PROJECT_ROOT
SG <- file.path(ROOT, "derivatives/sub-01/analysis/smoothed_glm")
THAL_NII <- file.path(ROOT, "derivatives/sub-01/analysis/thalamus_work/sub-01_thalamus_native.nii.gz")

cf_table <- fromJSON(file.path(SG, "sub-01_CF_voxeltable_SMOOTHED.json"))
am_table <- fromJSON(file.path(SG, "sub-01_AM_voxeltable_SMOOTHED.json"))

cf_hz <- cf_table$rows$best_fit_freq_hz_continuous
am_hz <- am_table$rows$best_fit_freq_hz_continuous
stopifnot(length(cf_hz) == 1075, length(am_hz) == 1669)

# voxel_ijk_native is a list-column of length-3 integer vectors -- collapse
# each to a single string key for set membership / intersection tests.
ijk_key <- function(ijk_list) sapply(ijk_list, function(v) paste(v, collapse = "_"))
cf_ijk <- ijk_key(cf_table$rows$voxel_ijk_native)
am_ijk <- ijk_key(am_table$rows$voxel_ijk_native)

cf_med_hz <- 10 ^ median(log10(cf_hz))
am_med_hz <- 10 ^ median(log10(am_hz))
cat(sprintf("CF median (n=1075) = %.1f Hz   AM median (n=1669) = %.2f Hz\n", cf_med_hz, am_med_hz))

high_cf_ijk <- cf_ijk[cf_hz >= cf_med_hz]
low_cf_ijk  <- cf_ijk[cf_hz <  cf_med_hz]
high_am_ijk <- am_ijk[am_hz >= am_med_hz]
low_am_ijk  <- am_ijk[am_hz <  am_med_hz]
cat(sprintf("High-CF n=%d  Low-CF n=%d  High-AM n=%d  Low-AM n=%d\n",
            length(high_cf_ijk), length(low_cf_ijk), length(high_am_ijk), length(low_am_ijk)))

thal_mask <- readNifti(THAL_NII)
N <- sum(thal_mask > 0)
cat(sprintf("N thalamus voxels (read fresh from %s) = %d\n", basename(THAL_NII), N))

overlap_test <- function(name_a, ijk_a, name_b, ijk_b, N) {
  both <- length(intersect(ijk_a, ijk_b))
  only_a <- length(ijk_a) - both
  only_b <- length(ijk_b) - both
  neither <- N - both - only_a - only_b
  tbl <- matrix(c(both, only_b, only_a, neither), nrow = 2)
  chisq_res <- chisq.test(tbl, correct = TRUE)
  fisher_res <- fisher.test(tbl)
  cat(sprintf("[%s x %s] both=%d %s-only=%d %s-only=%d neither=%d\n",
              name_a, name_b, both, name_a, only_a, name_b, only_b, neither))
  cat(sprintf("[%s x %s] chi2=%.3f p=%.3e   Fisher OR=%.4f p=%.3e  (expected both=%.1f)\n",
              name_a, name_b, unname(chisq_res$statistic), chisq_res$p.value,
              unname(fisher_res$estimate), fisher_res$p.value, chisq_res$expected[1, 1]))
  list(both = both, only_a = only_a, only_b = only_b, neither = neither,
       chi2 = unname(chisq_res$statistic), chi2_p = chisq_res$p.value,
       odds_ratio = unname(fisher_res$estimate), fisher_p = fisher_res$p.value,
       expected_both = chisq_res$expected[1, 1])
}

cat("\n--- primary: High-CF x High-AM ---\n")
hh <- overlap_test("HighCF", high_cf_ijk, "HighAM", high_am_ijk, N)
cat("\n--- concordant: Low-CF x Low-AM ---\n")
ll <- overlap_test("LowCF", low_cf_ijk, "LowAM", low_am_ijk, N)
cat("\n--- cross: High-CF x Low-AM ---\n")
hl <- overlap_test("HighCF", high_cf_ijk, "LowAM", low_am_ijk, N)
cat("\n--- cross: Low-CF x High-AM ---\n")
lh <- overlap_test("LowCF", low_cf_ijk, "HighAM", high_am_ijk, N)

result <- list(N_thalamus_voxels = N, CF_median_hz = cf_med_hz, AM_median_hz = am_med_hz,
               HighCF_x_HighAM = hh, LowCF_x_LowAM = ll, HighCF_x_LowAM = hl, LowCF_x_HighAM = lh)
out_json <- file.path(SG, "sub-01_CFAM_fullthalamus_overlap_SMOOTHED_R.json")
write(toJSON(result, auto_unbox = TRUE, pretty = TRUE), out_json)
cat(sprintf("\nwrote %s\n", out_json))

# =========================================================================
# graph: odds ratios for all four pairings, base R barplot
# =========================================================================
out_png <- file.path(SG, "sub-01_CFAM_fullthalamus_overlap_SMOOTHED_R.png")
png(out_png, width = 1200, height = 800, res = 130)
par(mar = c(7, 5, 5, 2))

labels <- c("High-CF x\nHigh-AM", "Low-CF x\nLow-AM", "High-CF x\nLow-AM", "Low-CF x\nHigh-AM")
ors <- c(hh$odds_ratio, ll$odds_ratio, hl$odds_ratio, lh$odds_ratio)
ps <- c(hh$fisher_p, ll$fisher_p, hl$fisher_p, lh$fisher_p)
cols <- c("#9467bd", "#2ca02c", "#ff7f0e", "#ff7f0e")

bp <- barplot(ors, names.arg = labels, col = cols, ylim = c(0, max(ors) * 1.55),
              ylab = "Fisher's exact odds ratio (spatial overlap vs. chance)",
              main = "Full-Thalamus High/Low CF x AM Spatial Overlap (R)")
text(bp, ors + max(ors) * 0.08, labels = sprintf("OR=%.1f\np=%.1e", ors, ps), cex = 0.85)
abline(h = 1, lty = 3)
text(mean(bp), max(ors) * 1.45,
     "all four pairings land in a similar range -- reflects the baseline CF/AM\nspatial conjunction, not frequency-specific high-with-high matching",
     cex = 0.85, font = 3)

dev.off()
cat(sprintf("wrote %s\n", out_png))
cat("DONE\n")
