# Companion figure script to thalamus_vs_cortex_comparison.R: (A) side-by-
# side CF-vs-AM scatterplots with linear fits, faceted by region, and
# (B) a point-range plot comparing the two regions' Pearson correlation
# estimates with 95% confidence intervals - both panels annotated with the
# Fisher r-to-z comparison statistic.

Sys.setlocale("LC_CTYPE", "en_US.UTF-8")
library(ggplot2)

PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "..")
dir <- file.path(PROJECT_ROOT, "derivatives", "sub-01", "analysis", "smoothed_glm")
outdir <- file.path(PROJECT_ROOT, "outputs", "figures_cortex")

cf_th <- read.csv(file.path(dir, "sub-01_CF_voxel_bestfreq_table_SMOOTHED_mediansplit.csv"))
am_th <- read.csv(file.path(dir, "sub-01_AM_voxel_bestfreq_table_SMOOTHED_mediansplit.csv"))
cf_cx <- read.csv(file.path(dir, "sub-01_CF_voxel_bestfreq_table_CORTEX_SMOOTHED_mediansplit.csv"))
am_cx <- read.csv(file.path(dir, "sub-01_AM_voxel_bestfreq_table_CORTEX_SMOOTHED_mediansplit.csv"))

merged_th <- merge(cf_th, am_th, by = c("voxel_i", "voxel_j", "voxel_k"), suffixes = c("_cf", "_am"))
merged_cx <- merge(cf_cx, am_cx, by = c("voxel_i", "voxel_j", "voxel_k"), suffixes = c("_cf", "_am"))
merged_th$region <- "Thalamus"
merged_cx$region <- "Auditory cortex"

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

# ---- Panel A: two-panel scatter with fitted line, side by side ----
combined <- rbind(
  data.frame(cf = merged_th$best_frequency_hz_cf, am = merged_th$best_frequency_hz_am,
             region = sprintf("Thalamus (r=%.3f, n=%d)", r_th, n_th)),
  data.frame(cf = merged_cx$best_frequency_hz_cf, am = merged_cx$best_frequency_hz_am,
             region = sprintf("Auditory cortex (r=%.3f, n=%d)", r_cx, n_cx))
)
combined$region <- factor(combined$region, levels = unique(combined$region)[c(
  which(grepl("Thalamus", unique(combined$region))), which(grepl("cortex", unique(combined$region))))])

p1 <- ggplot(combined, aes(x = cf, y = am)) +
  geom_point(size = 1.1, alpha = 0.4, color = "#2C7BB6") +
  geom_smooth(method = "lm", se = TRUE, color = "#D7191C", linewidth = 1) +
  scale_x_log10() +
  scale_y_log10() +
  facet_wrap(~region, scales = "free_x") +
  labs(
    title = "CF <-> AM relationship: thalamus vs auditory cortex (sub-01)",
    subtitle = sprintf("Fisher r-to-z comparing the two correlations: z = %.2f, p = %.3f", ftest$z, ftest$p),
    x = "CF best frequency (Hz, log scale)",
    y = "AM best frequency (Hz, log scale)"
  ) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"), strip.text = element_text(face = "bold", size = 11))

out1 <- file.path(outdir, "sub-01_thalamus_vs_cortex_CFAM_scatter_comparison.png")
ggsave(out1, p1, width = 11, height = 5.5, dpi = 300, type = "cairo")
message("Saved: ", out1)

# ---- Panel B: point-range comparison of the two r estimates with 95% CI ----
ci_from_r <- function(r, n, conf = 0.95) {
  z <- atanh(r); se <- 1 / sqrt(n - 3)
  crit <- qnorm(1 - (1 - conf) / 2)
  c(lo = tanh(z - crit * se), hi = tanh(z + crit * se))
}
ci_th <- ci_from_r(r_th, n_th)
ci_cx <- ci_from_r(r_cx, n_cx)

est <- data.frame(
  region = factor(c("Thalamus", "Auditory cortex"), levels = c("Thalamus", "Auditory cortex")),
  r = c(r_th, r_cx),
  lo = c(ci_th["lo"], ci_cx["lo"]),
  hi = c(ci_th["hi"], ci_cx["hi"])
)

p2 <- ggplot(est, aes(x = region, y = r, color = region)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  geom_pointrange(aes(ymin = lo, ymax = hi), size = 1, linewidth = 1.1) +
  geom_text(aes(label = sprintf("r = %.3f", r)), vjust = -1.6, size = 4.5, show.legend = FALSE) +
  scale_color_manual(values = c("Thalamus" = "#7B3294", "Auditory cortex" = "#008837"), guide = "none") +
  labs(
    title = "CF<->AM correlation strength: thalamus vs auditory cortex",
    subtitle = sprintf("Pearson r (log10 Hz) with 95%% CI | Fisher r-to-z: z = %.2f, p = %.3f", ftest$z, ftest$p),
    x = NULL,
    y = "Pearson correlation (CF vs AM, log10 Hz)"
  ) +
  ylim(min(est$lo) - 0.03, max(est$hi) + 0.03) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

out2 <- file.path(outdir, "sub-01_thalamus_vs_cortex_correlation_CI_comparison.png")
ggsave(out2, p2, width = 7, height = 6, dpi = 300, type = "cairo")
message("Saved: ", out2)
