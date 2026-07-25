# Median-split histograms for CF and AM (full-thalamus populations, same
# data and medians as the full-thalamus median-split plot script): each
# panel shows one continuous frequency distribution, colored Low/High at its
# own median -- CF and AM are each a single distribution split by their own
# median, not two separately-shifted groups.
#
# Inputs: per-voxel CF and AM best-fit-frequency tables (JSON).
# Output: two-panel histogram of CF and AM median splits (PNG).
#
# Run: Rscript 34_cfam_median_split_histogram.R

library(jsonlite)

PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "..")
SG <- file.path(PROJECT_ROOT, "derivatives", "sub-01", "analysis", "smoothed_glm")

cf_table <- fromJSON(file.path(SG, "sub-01_CF_voxeltable_SMOOTHED.json"))
am_table <- fromJSON(file.path(SG, "sub-01_AM_voxeltable_SMOOTHED.json"))
cf_hz <- cf_table$rows$best_fit_freq_hz_continuous
am_hz <- am_table$rows$best_fit_freq_hz_continuous

LOW_COL <- "#AEC7E8"   # light blue: Low group
HIGH_COL <- "#F5E67A"  # light yellow: High group

CF_HZ <- 200.0 * (8000.0 / 200.0) ^ (seq(0, 35) / 35.0)
AM_HZ <- 1.0 * (16.0 / 1.0) ^ (seq(0, 8) / 8.0)
fmt <- function(v) sub("\\.$", "", sub("0+$", "", sprintf("%.2f", v)))

split_hist <- function(x, med_log, n_breaks, label, xlab, hz_ticks) {
  lx <- log10(x)
  breaks <- seq(min(lx), max(lx), length.out = n_breaks)
  h <- hist(lx, breaks = breaks, plot = FALSE)
  mids <- h$mids
  cols <- ifelse(mids < med_log, LOW_COL, HIGH_COL)
  plot(h, col = cols, border = "black", main = "", xlab = "", ylab = "voxel count",
       xaxt = "n", xlim = range(lx))
  axis(1, at = log10(hz_ticks), labels = sapply(hz_ticks, fmt), las = 2,
       cex.axis = if (length(hz_ticks) > 15) 0.55 else 0.8)
  mtext(xlab, side = 1, line = if (length(hz_ticks) > 15) 5.5 else 3.2)
  abline(v = med_log, lty = 2, lwd = 1.5)
  legend("topright", legend = c("Low group", "High group"),
         fill = c(LOW_COL, HIGH_COL), bty = "o", bg = "white", cex = 0.85)
  text(par("usr")[1], par("usr")[4] * 0.95, label, cex = 1.6, font = 2, adj = c(-0.3, 1))
}

cf_med_log <- median(log10(cf_hz))
am_med_log <- median(log10(am_hz))

out_path <- file.path(SG, "sub-01_CFAM_median_split_histogram_SMOOTHED_R.png")
png(out_path, width = 950, height = 950, res = 130)
par(mfrow = c(2, 1), mar = c(6.5, 5, 2, 2))

split_hist(cf_hz, cf_med_log, 31, "A", "preferred carrier frequency (Hz)", CF_HZ)
split_hist(am_hz, am_med_log, 21, "B", "preferred amplitude modulation frequency (Hz)", AM_HZ)

dev.off()
cat(sprintf("wrote %s\n", out_path))
