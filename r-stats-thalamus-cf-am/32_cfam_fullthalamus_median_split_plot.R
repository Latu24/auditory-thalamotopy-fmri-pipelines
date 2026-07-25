# Visualization of the median split on the FULL per-modality thalamus
# populations (not restricted to CF/AM conjunction voxels): all 1075
# CF-significant voxels ranked/split by their own preferred CF frequency, and
# all 1669 AM-significant voxels ranked/split by their own preferred AM
# frequency -- independently, using the same two rankings that classify the
# conjunction voxels in the chi-square test script. Base R graphics only.
#
# Inputs: per-voxel CF and AM best-fit-frequency tables (JSON).
# Output: two-panel strip plot of CF and AM median splits (PNG).
#
# Run: Rscript 32_cfam_fullthalamus_median_split_plot.R

library(jsonlite)

PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "..")
SG <- file.path(PROJECT_ROOT, "derivatives", "sub-01", "analysis", "smoothed_glm")

cf_table <- fromJSON(file.path(SG, "sub-01_CF_voxeltable_SMOOTHED.json"))
am_table <- fromJSON(file.path(SG, "sub-01_AM_voxeltable_SMOOTHED.json"))

cf_hz <- cf_table$rows$best_fit_freq_hz_continuous
am_hz <- am_table$rows$best_fit_freq_hz_continuous
n_cf <- length(cf_hz); n_am <- length(am_hz)
stopifnot(n_cf == 1075, n_am == 1669)

cf_med_hz <- 10 ^ median(log10(cf_hz))
am_med_hz <- 10 ^ median(log10(am_hz))
high_cf <- cf_hz >= cf_med_hz
high_am <- am_hz >= am_med_hz
cat(sprintf("CF: n=%d, median=%.1f Hz, High-CF n=%d, Low-CF n=%d\n",
            n_cf, cf_med_hz, sum(high_cf), sum(!high_cf)))
cat(sprintf("AM: n=%d, median=%.2f Hz, High-AM n=%d, Low-AM n=%d\n",
            n_am, am_med_hz, sum(high_am), sum(!high_am)))

CF_HZ <- 200.0 * (8000.0 / 200.0) ^ (seq(0, 35) / 35.0)
AM_HZ <- 1.0 * (16.0 / 1.0) ^ (seq(0, 8) / 8.0)
fmt <- function(v) sub("\\.$", "", sub("0+$", "", sprintf("%.2f", v)))

set.seed(0)
out_path <- file.path(SG, "sub-01_CFAM_fullthalamus_median_split_SMOOTHED_R.png")
png(out_path, width = 1300, height = 900, res = 130)
par(mfrow = c(2, 1), mar = c(5, 5, 4, 2))

# ---- CF: all 1075 CF-significant thalamus voxels ----
y_cf <- runif(n_cf, -1, 1)
plot(cf_hz, y_cf, log = "x", pch = 21,
     bg = ifelse(high_cf, "#d62728", "#1f77b4"), col = "black", cex = 1.1,
     xlim = c(150, 10000), ylim = c(-1.3, 1.3), yaxt = "n", xaxt = "n",
     xlab = "preferred carrier frequency (Hz)", ylab = "",
     main = sprintf("Full-thalamus CF median split (n=%d CF-significant voxels)", n_cf))
axis(1, at = CF_HZ, labels = sapply(CF_HZ, fmt), las = 2, cex.axis = 0.55)
abline(v = cf_med_hz, lty = 2, lwd = 2)
text(cf_med_hz, 1.25, sprintf("median = %.1f Hz", cf_med_hz), pos = 4, cex = 0.85)
legend("topright", legend = c(sprintf("Low-CF (n=%d)", sum(!high_cf)),
                               sprintf("High-CF (n=%d)", sum(high_cf))),
       pt.bg = c("#1f77b4", "#d62728"), pch = 21, pt.cex = 1.2, bty = "o", bg = "white", cex = 0.85)

# ---- AM: all 1669 AM-significant thalamus voxels ----
y_am <- runif(n_am, -1, 1)
plot(am_hz, y_am, log = "x", pch = 21,
     bg = ifelse(high_am, "#d62728", "#1f77b4"), col = "black", cex = 1.1,
     xlim = c(0.7, 20), ylim = c(-1.3, 1.3), yaxt = "n", xaxt = "n",
     xlab = "preferred amplitude modulation frequency (Hz)", ylab = "",
     main = sprintf("Full-thalamus AM median split (n=%d AM-significant voxels)", n_am))
axis(1, at = AM_HZ, labels = sapply(AM_HZ, fmt), cex.axis = 0.8)
abline(v = am_med_hz, lty = 2, lwd = 2)
text(am_med_hz, 1.25, sprintf("median = %.2f Hz", am_med_hz), pos = 4, cex = 0.85)
legend("topright", legend = c(sprintf("Low-AM (n=%d)", sum(!high_am)),
                               sprintf("High-AM (n=%d)", sum(high_am))),
       pt.bg = c("#1f77b4", "#d62728"), pch = 21, pt.cex = 1.2, bty = "o", bg = "white", cex = 0.85)

dev.off()
cat(sprintf("wrote %s\n", out_path))
