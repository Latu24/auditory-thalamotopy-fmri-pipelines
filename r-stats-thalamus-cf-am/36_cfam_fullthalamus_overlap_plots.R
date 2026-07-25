# Two additional summary plots built from the full-thalamus overlap test's
# saved JSON result (does not rerun or modify the underlying test):
#   1. The same odds-ratio bar chart as the overlap test script, without the
#      caveat annotation.
#   2. The same four High/Low CF x AM pairings, plotting the chi-square
#      statistic instead of the Fisher odds ratio, with a dashed reference
#      line at the chi-square critical value for p=0.05 at dof=1 (3.841) --
#      the chi-square analogue of the odds-ratio-vs-1 "no association" line,
#      since chi-square has no fixed "no association" value the way OR=1
#      does (0 would be that value, but sits off the bottom of any realistic
#      chi-square plot here).
#
# Input: full-thalamus CF/AM overlap test result (JSON).
# Outputs: odds-ratio bar chart (PNG) and chi-square bar chart (PNG).
#
# Run: Rscript 36_cfam_fullthalamus_overlap_plots.R

library(jsonlite)

PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "..")
SG <- file.path(PROJECT_ROOT, "derivatives", "sub-01", "analysis", "smoothed_glm")
res <- fromJSON(file.path(SG, "sub-01_CFAM_fullthalamus_overlap_SMOOTHED_R.json"))

labels <- c("High-CF x\nHigh-AM", "Low-CF x\nLow-AM", "High-CF x\nLow-AM", "Low-CF x\nHigh-AM")
cols <- c("#9467bd", "#2ca02c", "#ff7f0e", "#ff7f0e")

ors <- c(res$HighCF_x_HighAM$odds_ratio, res$LowCF_x_LowAM$odds_ratio,
         res$HighCF_x_LowAM$odds_ratio, res$LowCF_x_HighAM$odds_ratio)
fisher_ps <- c(res$HighCF_x_HighAM$fisher_p, res$LowCF_x_LowAM$fisher_p,
               res$HighCF_x_LowAM$fisher_p, res$LowCF_x_HighAM$fisher_p)
chi2s <- c(res$HighCF_x_HighAM$chi2, res$LowCF_x_LowAM$chi2,
           res$HighCF_x_LowAM$chi2, res$LowCF_x_HighAM$chi2)
chi2_ps <- c(res$HighCF_x_HighAM$chi2_p, res$LowCF_x_LowAM$chi2_p,
             res$HighCF_x_LowAM$chi2_p, res$LowCF_x_HighAM$chi2_p)

# ---- (1) odds-ratio bar chart, caveat text removed ----
out_png1 <- file.path(SG, "sub-01_CFAM_fullthalamus_overlap_SMOOTHED_R_v2.png")
png(out_png1, width = 1200, height = 800, res = 130)
par(mar = c(7, 5, 5, 2))
bp <- barplot(ors, names.arg = labels, col = cols, ylim = c(0, max(ors) * 1.3),
              ylab = "Fisher's exact odds ratio (spatial overlap vs. chance)",
              main = "Full-Thalamus High/Low CF x AM Spatial Overlap (R)")
text(bp, ors + max(ors) * 0.05, labels = sprintf("OR=%.1f\np=%.1e", ors, fisher_ps), cex = 0.85)
abline(h = 1, lty = 3)
dev.off()
cat(sprintf("wrote %s\n", out_png1))

# ---- (2) chi-square statistic bar chart ----
crit_chi2 <- qchisq(0.95, df = 1)  # 3.841, the p=0.05 significance threshold at dof=1
out_png2 <- file.path(SG, "sub-01_CFAM_fullthalamus_overlap_chisq_SMOOTHED_R.png")
png(out_png2, width = 1200, height = 800, res = 130)
par(mar = c(7, 5, 5, 2))
bp2 <- barplot(chi2s, names.arg = labels, col = cols, ylim = c(0, max(chi2s) * 1.3),
               ylab = "Chi-square statistic (dof=1)",
               main = "Full-Thalamus High/Low CF x AM Spatial Overlap: Chi-square (R)")
text(bp2, chi2s + max(chi2s) * 0.05, labels = sprintf("chi2=%.0f\np=%.1e", chi2s, chi2_ps), cex = 0.85)
abline(h = crit_chi2, lty = 2)
text(bp2[length(bp2)], crit_chi2, sprintf("  p=0.05 threshold (chi2=%.2f, dof=1)", crit_chi2),
     pos = 3, cex = 0.75)
dev.off()
cat(sprintf("wrote %s\n", out_png2))
