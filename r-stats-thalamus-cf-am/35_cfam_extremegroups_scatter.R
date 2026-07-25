# "Extreme groups" scatter of CF vs. AM preference for the 639 CF/AM
# conjunction voxels -- the only voxels with a real, paired (CF, AM) value on
# both axes. The lower/upper X% of x-values (preferred CF) are shaded, with a
# horizontal segment marking each shaded group's own mean y (preferred AM),
# alongside the overall regression line. Two panels compare a 33% and a 27%
# extreme-group cut.
#
# Inputs: CF/AM conjunction voxel table (JSON).
# Output: two-panel extreme-groups scatter plot (PNG).
#
# Run: Rscript 35_cfam_extremegroups_scatter.R

library(jsonlite)

PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "..")
SG <- file.path(PROJECT_ROOT, "derivatives", "sub-01", "analysis", "smoothed_glm")
conj <- fromJSON(file.path(SG, "sub-01_CFAM_conjunction_SMOOTHED.json"))

cf_hz <- conj$rows$CF_best_fit_freq_hz_continuous
am_hz <- conj$rows$AM_best_fit_freq_hz_continuous
n <- length(cf_hz)
lx <- log10(cf_hz)
ly <- log10(am_hz)

CF_HZ <- 200.0 * (8000.0 / 200.0) ^ (seq(0, 35) / 35.0)
AM_HZ <- 1.0 * (16.0 / 1.0) ^ (seq(0, 8) / 8.0)
fmt <- function(v) sub("\\.$", "", sub("0+$", "", sprintf("%.2f", v)))

extreme_panel <- function(frac, label) {
  q_lo <- quantile(lx, frac)
  q_hi <- quantile(lx, 1 - frac)
  lower_grp <- lx <= q_lo
  upper_grp <- lx >= q_hi

  plot(lx, ly, pch = 16, col = "black", cex = 0.9,
       xlim = range(lx), ylim = range(ly), xaxt = "n", yaxt = "n",
       xlab = "preferred carrier frequency (Hz)",
       ylab = "preferred amplitude modulation frequency (Hz)",
       main = sprintf("%s: lower/upper %d%% of CF-values (n=%d conjunction voxels)",
                       label, round(frac * 100), n))
  axis(1, at = log10(CF_HZ)[seq(1, 36, by = 2)], labels = sapply(CF_HZ[seq(1, 36, by = 2)], fmt),
       las = 2, cex.axis = 0.6)
  axis(2, at = log10(AM_HZ), labels = sapply(AM_HZ, fmt), cex.axis = 0.75)

  usr <- par("usr")
  rect(usr[1], usr[3], q_lo, usr[4], col = rgb(0.5, 0.5, 0.5, 0.25), border = NA)
  rect(q_hi, usr[3], usr[2], usr[4], col = rgb(0.5, 0.5, 0.5, 0.25), border = NA)
  abline(v = c(q_lo, q_hi), col = "gray30", lty = 1)
  text(usr[1] + 0.02 * diff(usr[1:2]), usr[4] - 0.05 * diff(usr[3:4]),
       sprintf("lower %d%%\nof CF-values", round(frac * 100)), adj = c(0, 1), cex = 0.8)
  text(usr[2] - 0.02 * diff(usr[1:2]), usr[3] + 0.05 * diff(usr[3:4]),
       sprintf("upper %d%%\nof CF-values", round(frac * 100)), adj = c(1, 0), cex = 0.8)

  points(lx, ly, pch = 16, cex = 0.9)
  segments(usr[1], mean(ly[lower_grp]), q_lo, mean(ly[lower_grp]), lwd = 2, col = "#1f77b4")
  segments(q_hi, mean(ly[upper_grp]), usr[2], mean(ly[upper_grp]), lwd = 2, col = "#d62728")
  fit <- lm(ly ~ lx)
  abline(fit, lwd = 1.5, lty = 2)

  t_res <- t.test(ly[upper_grp], ly[lower_grp])
  legend("bottomright",
         legend = sprintf("mean AM: lower=%.2f Hz, upper=%.2f Hz\nt-test p=%.3f",
                           10 ^ mean(ly[lower_grp]), 10 ^ mean(ly[upper_grp]), t_res$p.value),
         bty = "o", bg = "white", cex = 0.75)
  invisible(t_res)
}

out_path <- file.path(SG, "sub-01_CFAM_extremegroups_scatter_SMOOTHED_R.png")
png(out_path, width = 1000, height = 1100, res = 130)
par(mfrow = c(2, 1), mar = c(5, 5, 4, 2))
r33 <- extreme_panel(1/3, "A")
r27 <- extreme_panel(0.27, "B")
dev.off()
cat(sprintf("wrote %s\n", out_path))
cat(sprintf("33%% cut: mean AM lower=%.2f Hz upper=%.2f Hz t-test p=%.4f\n",
            10 ^ mean(ly[lx <= quantile(lx, 1/3)]), 10 ^ mean(ly[lx >= quantile(lx, 2/3)]), r33$p.value))
cat(sprintf("27%% cut: mean AM lower=%.2f Hz upper=%.2f Hz t-test p=%.4f\n",
            10 ^ mean(ly[lx <= quantile(lx, 0.27)]), 10 ^ mean(ly[lx >= quantile(lx, 0.73)]), r27$p.value))
