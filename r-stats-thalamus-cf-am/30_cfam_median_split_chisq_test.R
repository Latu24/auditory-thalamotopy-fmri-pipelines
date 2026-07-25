# Chi-square test of independence between characteristic-frequency (CF) and
# amplitude-modulation-rate (AM) tuning preference, based on a median split of
# each modality's per-voxel best-fit frequency:
#
#   1. Rank ALL CF-significant thalamic voxels (n=1075, real Gaussian-fit
#      best CF) by best CF frequency; split at the median into
#      Low-CF-preference / High-CF-preference.
#   2. Independently rank ALL AM-significant thalamic voxels (n=1669, real
#      best AM) by best AM frequency; split at the median.
#   3. For the 639 CF/AM conjunction voxels (the only ones with BOTH a real
#      CF and a real AM label), cross the two labels into a 2x2 table and run
#      a chi-square test of independence plus Fisher's exact test.
#
# Inputs: per-voxel CF and AM best-fit-frequency tables (JSON) and the CF/AM
# conjunction voxel table (JSON).
# Output: JSON summary of group sizes, contingency table, and test statistics.
#
# Run: Rscript 30_cfam_median_split_chisq_test.R

library(jsonlite)

PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "..")
SG <- file.path(PROJECT_ROOT, "derivatives", "sub-01", "analysis", "smoothed_glm")

cat("loading CF-only GLM table (1075), AM-only GLM table (1669), conjunction table (639) ...\n")
cf_table <- fromJSON(file.path(SG, "sub-01_CF_voxeltable_SMOOTHED.json"))
am_table <- fromJSON(file.path(SG, "sub-01_AM_voxeltable_SMOOTHED.json"))
conj <- fromJSON(file.path(SG, "sub-01_CFAM_conjunction_SMOOTHED.json"))

cf_all <- cf_table$rows$best_fit_freq_hz_continuous
am_all <- am_table$rows$best_fit_freq_hz_continuous
stopifnot(length(cf_all) == 1075, length(am_all) == 1669)

cf_med_log <- median(log10(cf_all))
am_med_log <- median(log10(am_all))
cf_med_hz <- 10 ^ cf_med_log
am_med_hz <- 10 ^ am_med_log
cat(sprintf("CF median (ranking all 1075 CF-significant voxels) = %.1f Hz\n", cf_med_hz))
cat(sprintf("AM median (ranking all 1669 AM-significant voxels) = %.2f Hz\n", am_med_hz))

cf_conj <- conj$rows$CF_best_fit_freq_hz_continuous
am_conj <- conj$rows$AM_best_fit_freq_hz_continuous
n <- nrow(conj$rows)
stopifnot(n == 639)

high_cf <- log10(cf_conj) >= cf_med_log
high_am <- log10(am_conj) >= am_med_log
cat(sprintf("within the %d conjunction voxels: High-CF n=%d Low-CF n=%d  High-AM n=%d Low-AM n=%d\n",
            n, sum(high_cf), sum(!high_cf), sum(high_am), sum(!high_am)))

n_hh <- sum(high_cf & high_am)
n_hl <- sum(high_cf & !high_am)
n_lh <- sum(!high_cf & high_am)
n_ll <- sum(!high_cf & !high_am)

tbl <- matrix(c(n_hh, n_lh, n_hl, n_ll), nrow = 2,
              dimnames = list(CF = c("High-CF", "Low-CF"), AM = c("High-AM", "Low-AM")))
cat("\ncontingency table (rows=CF, cols=AM):\n")
print(tbl)

# Yates' continuity correction (correct = TRUE) is used, matching the
# standard default for 2x2 chi-square tests; without it (correct = FALSE)
# the result is chi2=0.1878, p=0.6647 instead. Both are legitimate, but the
# corrected version is reported here for consistency with the field default.
chisq_res <- chisq.test(tbl, correct = TRUE)
fisher_res <- fisher.test(tbl)

cat(sprintf("\nchi-square test of independence: chi2=%.4f df=%d p=%.4f  (expected High-CF/High-AM=%.1f)\n",
            chisq_res$statistic, chisq_res$parameter, chisq_res$p.value, chisq_res$expected[1, 1]))
cat(sprintf("Fisher's exact test: odds ratio=%.4f p=%.4f\n", fisher_res$estimate, fisher_res$p.value))

result <- list(
  n_conjunction_voxels = n,
  CF_median_hz_ranked_over_all_1075_CF_significant = cf_med_hz,
  AM_median_hz_ranked_over_all_1669_AM_significant = am_med_hz,
  within_conjunction_group_sizes = list(
    n_high_CF = sum(high_cf), n_low_CF = sum(!high_cf),
    n_high_AM = sum(high_am), n_low_AM = sum(!high_am)
  ),
  contingency_table = list(
    high_CF_high_AM = n_hh, high_CF_low_AM = n_hl,
    low_CF_high_AM = n_lh, low_CF_low_AM = n_ll
  ),
  chi_square_independence = list(
    chi2 = unname(chisq_res$statistic), df = unname(chisq_res$parameter),
    p = chisq_res$p.value, expected_highCF_highAM = chisq_res$expected[1, 1]
  ),
  fisher_exact = list(
    odds_ratio = unname(fisher_res$estimate), p = fisher_res$p.value
  )
)

out_path <- file.path(SG, "sub-01_CFAM_median_split_chisq_SMOOTHED_R.json")
write(toJSON(result, auto_unbox = TRUE, pretty = TRUE), out_path)
cat(sprintf("\nwrote %s\n", out_path))
cat("DONE\n")
