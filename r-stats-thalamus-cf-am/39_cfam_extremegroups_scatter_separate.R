# Corrected extreme-groups figure, keeping the CF-only and AM-only voxel
# populations fully separate throughout (never pooled into one sample or one
# t-test), since CF means "real" for one population and "sub-threshold" for
# the other and mixing them would confound the comparison:
#
#   Panel set A -- CF-only voxels (n=436) ONLY: split by their own REAL CF
#     tercile, compare their SUB-THRESHOLD AM value between the extreme
#     groups. Every point in this comparison is a CF-only voxel; no AM-only
#     voxel ever enters this test.
#   Panel set B -- AM-only voxels (n=1030) ONLY: split by their own REAL AM
#     tercile, compare their SUB-THRESHOLD CF value between the extreme
#     groups. Every point here is an AM-only voxel; no CF-only voxel ever
#     enters this test.
#
# The two populations, two split variables, and two t-tests are completely
# separate throughout.
#
# Inputs: sub-threshold CF/AM fit table (JSON), containing CF-only and
# AM-only voxel rows.
# Output: four-panel extreme-groups scatter plot (PNG).
#
# Run: Rscript 39_cfam_extremegroups_scatter_separate.R

library(jsonlite)

PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "..")
SG <- file.path(PROJECT_ROOT, "derivatives", "sub-01", "analysis", "smoothed_glm")
sub <- fromJSON(file.path(SG, "sub-01_CFAM_subthreshold_fits_SMOOTHED.json"))

CF_HZ <- 200.0 * (8000.0 / 200.0) ^ (seq(0, 35) / 35.0)
AM_HZ <- 1.0 * (16.0 / 1.0) ^ (seq(0, 8) / 8.0)
fmt <- function(v) sub("\\.$", "", sub("0+$", "", sprintf("%.2f", v)))

# split_x: the REAL variable this population is split by (log10)
# other_y: the SUB-THRESHOLD variable being compared between extreme groups (log10)
extreme_panel_single_pop <- function(split_x, other_y, frac, pop_label, split_name, other_name,
                                       split_ticks, other_ticks, n) {
  q_lo <- quantile(split_x, frac)
  q_hi <- quantile(split_x, 1 - frac)
  lower_grp <- split_x <= q_lo
  upper_grp <- split_x >= q_hi

  plot(split_x, other_y, pch = 16, col = "black", cex = 0.8,
       xlim = range(split_x), ylim = range(other_y), xaxt = "n", yaxt = "n",
       xlab = sprintf("real %s (Hz)", split_name), ylab = sprintf("sub-threshold %s (Hz)", other_name),
       main = sprintf("%s ONLY (n=%d): lower/upper %d%% of real %s", pop_label, n, round(frac * 100), split_name))
  axis(1, at = log10(split_ticks), labels = sapply(split_ticks, fmt),
       las = 2, cex.axis = if (length(split_ticks) > 15) 0.55 else 0.75)
  axis(2, at = log10(other_ticks), labels = sapply(other_ticks, fmt),
       cex.axis = if (length(other_ticks) > 15) 0.55 else 0.75)

  usr <- par("usr")
  rect(usr[1], usr[3], q_lo, usr[4], col = rgb(0.5, 0.5, 0.5, 0.25), border = NA)
  rect(q_hi, usr[3], usr[2], usr[4], col = rgb(0.5, 0.5, 0.5, 0.25), border = NA)
  abline(v = c(q_lo, q_hi), col = "gray30", lty = 1)
  points(split_x, other_y, pch = 16, cex = 0.8)
  segments(usr[1], mean(other_y[lower_grp]), q_lo, mean(other_y[lower_grp]), lwd = 2, col = "#1f77b4")
  segments(q_hi, mean(other_y[upper_grp]), usr[2], mean(other_y[upper_grp]), lwd = 2, col = "#d62728")
  abline(lm(other_y ~ split_x), lwd = 1.5, lty = 2)

  t_res <- t.test(other_y[upper_grp], other_y[lower_grp])
  legend("bottomright",
         legend = sprintf("mean %s: lower=%.2f Hz, upper=%.2f Hz\nt-test p=%.3f (sub-threshold outcome)",
                           other_name, 10 ^ mean(other_y[lower_grp]), 10 ^ mean(other_y[upper_grp]), t_res$p.value),
         bty = "o", bg = "white", cex = 0.68)
  cat(sprintf("[%s only, %d%% cut] mean %s: lower=%.2f upper=%.2f  t-test p=%.4f\n",
              pop_label, round(frac * 100), other_name,
              10 ^ mean(other_y[lower_grp]), 10 ^ mean(other_y[upper_grp]), t_res$p.value))
  invisible(t_res)
}

cf_only_split <- log10(sub$CF_only_rows$CF_best_fit_freq_hz_continuous)
cf_only_other <- log10(sub$CF_only_rows$AM_best_fit_freq_hz_continuous_subthreshold)
am_only_split <- log10(sub$AM_only_rows$AM_best_fit_freq_hz_continuous)
am_only_other <- log10(sub$AM_only_rows$CF_best_fit_freq_hz_continuous_subthreshold)

out_path <- file.path(SG, "sub-01_CFAM_extremegroups_scatter_separate_SMOOTHED_R.png")
png(out_path, width = 1300, height = 1300, res = 130)
par(mfrow = c(2, 2), mar = c(6, 5, 4, 2))

extreme_panel_single_pop(cf_only_split, cf_only_other, 1/3, "CF-only", "CF", "AM", CF_HZ, AM_HZ,
                          nrow(sub$CF_only_rows))
extreme_panel_single_pop(cf_only_split, cf_only_other, 0.27, "CF-only", "CF", "AM", CF_HZ, AM_HZ,
                          nrow(sub$CF_only_rows))
extreme_panel_single_pop(am_only_split, am_only_other, 1/3, "AM-only", "AM", "CF", AM_HZ, CF_HZ,
                          nrow(sub$AM_only_rows))
extreme_panel_single_pop(am_only_split, am_only_other, 0.27, "AM-only", "AM", "CF", AM_HZ, CF_HZ,
                          nrow(sub$AM_only_rows))

dev.off()
cat(sprintf("wrote %s\n", out_path))
