# Visualization of the median-split classification underlying the CF/AM
# chi-square test: scatter of the 639 CF/AM conjunction voxels' real CF (Hz)
# vs. real AM (Hz), with the two median thresholds drawn as crosshair lines
# directly on the data, colored by which of the four High/Low CF x AM
# quadrants each voxel falls into. Base R graphics only (no ggplot2).
#
# Inputs: chi-square test result JSON (from the median-split test script) and
# the CF/AM conjunction voxel table (JSON).
# Output: scatter plot (PNG).
#
# Run: Rscript 31_cfam_median_split_chisq_plot.R

library(jsonlite)

PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "..")
SG <- file.path(PROJECT_ROOT, "derivatives", "sub-01", "analysis", "smoothed_glm")

res <- fromJSON(file.path(SG, "sub-01_CFAM_median_split_chisq_SMOOTHED_R.json"))
conj <- fromJSON(file.path(SG, "sub-01_CFAM_conjunction_SMOOTHED.json"))

cf_hz <- conj$rows$CF_best_fit_freq_hz_continuous
am_hz <- conj$rows$AM_best_fit_freq_hz_continuous
n <- length(cf_hz)
stopifnot(n == res$n_conjunction_voxels)

cf_med_hz <- res$CF_median_hz_ranked_over_all_1075_CF_significant
am_med_hz <- res$AM_median_hz_ranked_over_all_1669_AM_significant

high_cf <- cf_hz >= cf_med_hz
high_am <- am_hz >= am_med_hz

col <- ifelse(high_cf & high_am, "#9467bd",
        ifelse(high_cf & !high_am, "#1f77b4",
        ifelse(!high_cf & high_am, "#d62728", "#7f7f7f")))

CF_HZ <- 200.0 * (8000.0 / 200.0) ^ (seq(0, 35) / 35.0)
AM_HZ <- 1.0 * (16.0 / 1.0) ^ (seq(0, 8) / 8.0)
fmt <- function(v) sub("\\.$", "", sub("0+$", "", sprintf("%.2f", v)))

out_path <- file.path(SG, "sub-01_CFAM_median_split_scatter_SMOOTHED_R.png")
png(out_path, width = 1100, height = 950, res = 130)
par(mar = c(8, 5, 4, 2))

plot(cf_hz, am_hz, log = "xy", pch = 21, bg = col, col = "black", cex = 1.3,
     xlim = c(150, 10000), ylim = c(0.7, 20), xaxt = "n", yaxt = "n",
     xlab = "", ylab = "preferred amplitude modulation frequency (Hz)",
     main = sprintf("Median Split, Shown on the Data (R, n=%d conjunction voxels)", n))
mtext("preferred carrier frequency (Hz)", side = 1, line = 6.5)

axis(1, at = CF_HZ, labels = sapply(CF_HZ, fmt), las = 2, cex.axis = 0.55)
axis(2, at = AM_HZ, labels = sapply(AM_HZ, fmt), las = 1, cex.axis = 0.8)

abline(v = cf_med_hz, lty = 2, lwd = 2)
abline(h = am_med_hz, lty = 2, lwd = 2)
mtext(sprintf("CF median = %.1f Hz", cf_med_hz), side = 3, line = 0.3, at = cf_med_hz, cex = 0.7)

n_hh <- sum(high_cf & high_am); n_hl <- sum(high_cf & !high_am)
n_lh <- sum(!high_cf & high_am); n_ll <- sum(!high_cf & !high_am)
legend("bottomright",
       legend = c(sprintf("High-CF/High-AM (n=%d)", n_hh), sprintf("High-CF/Low-AM (n=%d)", n_hl),
                  sprintf("Low-CF/High-AM (n=%d)", n_lh), sprintf("Low-CF/Low-AM (n=%d)", n_ll)),
       pt.bg = c("#9467bd", "#1f77b4", "#d62728", "#7f7f7f"), pch = 21, pt.cex = 1.3,
       bty = "o", bg = "white", cex = 0.8)

legend("topleft",
       legend = sprintf("chi2=%.3f, p=%.3f (Fisher OR=%.3f, p=%.3f)",
                         res$chi_square_independence$chi2, res$chi_square_independence$p,
                         res$fisher_exact$odds_ratio, res$fisher_exact$p),
       bty = "o", bg = "white", cex = 0.85)

dev.off()
cat(sprintf("wrote %s\n", out_path))
