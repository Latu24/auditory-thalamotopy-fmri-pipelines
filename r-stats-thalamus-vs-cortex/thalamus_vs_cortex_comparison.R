# Four-part statistical comparison of CF/AM tonotopic organization between
# thalamus and auditory cortex: (1) proportion test of significance rate
# relative to ROI size, (2) proportion test of the CF/AM dual-significance
# (overlap) rate, (3) Kolmogorov-Smirnov test comparing best-frequency
# distribution shape (log10 Hz), and (4) Fisher r-to-z test comparing CF-AM
# correlation strength between regions. Results are written to a text file.

Sys.setlocale("LC_CTYPE", "en_US.UTF-8")

PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "..")
dir <- file.path(PROJECT_ROOT, "derivatives", "sub-01", "analysis", "smoothed_glm")

cf_th <- read.csv(file.path(dir, "sub-01_CF_voxel_bestfreq_table_SMOOTHED_mediansplit.csv"))
am_th <- read.csv(file.path(dir, "sub-01_AM_voxel_bestfreq_table_SMOOTHED_mediansplit.csv"))
cf_cx <- read.csv(file.path(dir, "sub-01_CF_voxel_bestfreq_table_CORTEX_SMOOTHED_mediansplit.csv"))
am_cx <- read.csv(file.path(dir, "sub-01_AM_voxel_bestfreq_table_CORTEX_SMOOTHED_mediansplit.csv"))

ROI_N_TH <- 40217   # total voxel count in the whole-thalamus segmentation mask (both hemispheres)
ROI_N_CX <- 74064   # total voxel count in the auditory-cortex segmentation mask

cat("=====================================================================\n")
cat("1) SIGNIFICANCE RATE: thalamus vs cortex (voxels reaching p<0.01 / ROI size)\n")
cat("=====================================================================\n")
cf_tbl <- matrix(c(nrow(cf_cx), ROI_N_CX - nrow(cf_cx), nrow(cf_th), ROI_N_TH - nrow(cf_th)),
                  nrow = 2, dimnames = list(c("significant", "not significant"), c("cortex", "thalamus")))
am_tbl <- matrix(c(nrow(am_cx), ROI_N_CX - nrow(am_cx), nrow(am_th), ROI_N_TH - nrow(am_th)),
                  nrow = 2, dimnames = list(c("significant", "not significant"), c("cortex", "thalamus")))
cat("\n-- CF significance rate --\n"); print(cf_tbl)
cf_prop <- prop.test(c(nrow(cf_cx), nrow(cf_th)), c(ROI_N_CX, ROI_N_TH))
print(cf_prop)
cat("\n-- AM significance rate --\n"); print(am_tbl)
am_prop <- prop.test(c(nrow(am_cx), nrow(am_th)), c(ROI_N_CX, ROI_N_TH))
print(am_prop)

cat("\n=====================================================================\n")
cat("2) DUAL-SIGNIFICANCE (OVERLAP) RATE: thalamus vs cortex\n")
cat("   (of all voxels significant for CF and/or AM, what fraction are BOTH?)\n")
cat("=====================================================================\n")
cf_th$vid <- paste(cf_th$voxel_i, cf_th$voxel_j, cf_th$voxel_k, sep = "_")
am_th$vid <- paste(am_th$voxel_i, am_th$voxel_j, am_th$voxel_k, sep = "_")
cf_cx$vid <- paste(cf_cx$voxel_i, cf_cx$voxel_j, cf_cx$voxel_k, sep = "_")
am_cx$vid <- paste(am_cx$voxel_i, am_cx$voxel_j, am_cx$voxel_k, sep = "_")

union_th <- length(union(cf_th$vid, am_th$vid)); both_th <- length(intersect(cf_th$vid, am_th$vid))
union_cx <- length(union(cf_cx$vid, am_cx$vid)); both_cx <- length(intersect(cf_cx$vid, am_cx$vid))
overlap_tbl <- matrix(c(both_cx, union_cx - both_cx, both_th, union_th - both_th),
                       nrow = 2, dimnames = list(c("both", "one only"), c("cortex", "thalamus")))
print(overlap_tbl)
overlap_prop <- prop.test(c(both_cx, both_th), c(union_cx, union_th))
print(overlap_prop)

cat("\n=====================================================================\n")
cat("3) BEST-FREQUENCY DISTRIBUTION SHAPE: thalamus vs cortex (Kolmogorov-Smirnov)\n")
cat("   (does WHERE in the frequency range voxels prefer differ by region?\n")
cat("    compared on log10(Hz) so CF's wide range doesn't dominate)\n")
cat("=====================================================================\n")
ks_cf <- ks.test(log10(cf_cx$best_frequency_hz), log10(cf_th$best_frequency_hz))
cat("\n-- CF: cortex vs thalamus --\n"); print(ks_cf)
ks_am <- ks.test(log10(am_cx$best_frequency_hz), log10(am_th$best_frequency_hz))
cat("\n-- AM: cortex vs thalamus --\n"); print(ks_am)
cat(sprintf("\nMedians (Hz): CF cortex=%.1f thalamus=%.1f | AM cortex=%.2f thalamus=%.2f\n",
            median(cf_cx$best_frequency_hz), median(cf_th$best_frequency_hz),
            median(am_cx$best_frequency_hz), median(am_th$best_frequency_hz)))

cat("\n=====================================================================\n")
cat("4) STRENGTH OF THE CF<->AM RELATIONSHIP ITSELF: thalamus vs cortex\n")
cat("   (Fisher r-to-z test comparing two independent Pearson correlations)\n")
cat("=====================================================================\n")
merged_th <- merge(cf_th, am_th, by = c("voxel_i", "voxel_j", "voxel_k"), suffixes = c("_cf", "_am"))
merged_cx <- merge(cf_cx, am_cx, by = c("voxel_i", "voxel_j", "voxel_k"), suffixes = c("_cf", "_am"))
r_th <- cor(log10(merged_th$best_frequency_hz_cf), log10(merged_th$best_frequency_hz_am))
r_cx <- cor(log10(merged_cx$best_frequency_hz_cf), log10(merged_cx$best_frequency_hz_am))
n_th <- nrow(merged_th); n_cx <- nrow(merged_cx)

fisher_r_to_z_test <- function(r1, n1, r2, n2) {
  z1 <- atanh(r1); z2 <- atanh(r2)
  se <- sqrt(1/(n1 - 3) + 1/(n2 - 3))
  z <- (z1 - z2) / se
  p <- 2 * pnorm(-abs(z))
  list(z = z, p = p)
}
ftest <- fisher_r_to_z_test(r_cx, n_cx, r_th, n_th)
cat(sprintf("\nPearson r (log10 CF vs log10 AM): cortex = %.4f (n=%d), thalamus = %.4f (n=%d)\n",
            r_cx, n_cx, r_th, n_th))
cat(sprintf("Fisher r-to-z: z = %.3f, p = %.3g\n", ftest$z, ftest$p))

sink_path <- file.path(dir, "sub-01_thalamus_vs_cortex_comparison.txt")
sink(sink_path)
cat("Thalamus vs auditory-cortex comparison (sub-01)\n\n")
cat("1) Significance rate (voxels p<0.01 / ROI size)\n")
cat(sprintf("   CF: cortex %d/%d (%.2f%%) vs thalamus %d/%d (%.2f%%)  prop.test p=%.3g\n",
            nrow(cf_cx), ROI_N_CX, 100*nrow(cf_cx)/ROI_N_CX, nrow(cf_th), ROI_N_TH, 100*nrow(cf_th)/ROI_N_TH, cf_prop$p.value))
cat(sprintf("   AM: cortex %d/%d (%.2f%%) vs thalamus %d/%d (%.2f%%)  prop.test p=%.3g\n",
            nrow(am_cx), ROI_N_CX, 100*nrow(am_cx)/ROI_N_CX, nrow(am_th), ROI_N_TH, 100*nrow(am_th)/ROI_N_TH, am_prop$p.value))
cat(sprintf("\n2) Dual-significance/overlap rate: cortex %d/%d (%.1f%%) vs thalamus %d/%d (%.1f%%)  prop.test p=%.3g\n",
            both_cx, union_cx, 100*both_cx/union_cx, both_th, union_th, 100*both_th/union_th, overlap_prop$p.value))
cat(sprintf("\n3) Best-frequency distribution (KS test, log10 Hz)\n   CF: D=%.4f, p=%.3g\n   AM: D=%.4f, p=%.3g\n",
            ks_cf$statistic, ks_cf$p.value, ks_am$statistic, ks_am$p.value))
cat(sprintf("\n4) CF<->AM correlation strength: cortex r=%.4f (n=%d) vs thalamus r=%.4f (n=%d)\n   Fisher r-to-z: z=%.3f, p=%.3g\n",
            r_cx, n_cx, r_th, n_th, ftest$z, ftest$p))
sink()
message("Saved: ", sink_path)
