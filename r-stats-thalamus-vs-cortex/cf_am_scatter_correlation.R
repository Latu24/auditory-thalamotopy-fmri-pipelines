# Scatterplot of CF vs AM best frequency (log-log axes) with a linear fit,
# for thalamus voxels. Reports Pearson correlation on log10-transformed
# frequencies and Spearman correlation on raw frequencies, to characterize
# the CF-AM relationship from two complementary angles (linear-on-log-scale
# vs rank-based).

Sys.setlocale("LC_CTYPE", "en_US.UTF-8")
library(ggplot2)

PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "..")
dir <- file.path(PROJECT_ROOT, "derivatives", "sub-01", "analysis", "smoothed_glm")

cf <- read.csv(file.path(dir, "sub-01_CF_voxel_bestfreq_table_SMOOTHED_mediansplit.csv"))
am <- read.csv(file.path(dir, "sub-01_AM_voxel_bestfreq_table_SMOOTHED_mediansplit.csv"))

merged <- merge(
  cf, am,
  by = c("voxel_i", "voxel_j", "voxel_k"),
  suffixes = c("_cf", "_am")
)

merged$log_cf <- log10(merged$best_frequency_hz_cf)
merged$log_am <- log10(merged$best_frequency_hz_am)

pear <- cor.test(merged$log_cf, merged$log_am, method = "pearson")
spear <- cor.test(merged$best_frequency_hz_cf, merged$best_frequency_hz_am, method = "spearman")

cat("Pearson correlation (log10 CF vs log10 AM):\n")
print(pear)
cat("\nSpearman correlation (raw CF vs raw AM):\n")
print(spear)

subtitle_txt <- sprintf(
  "n = %d voxels | Pearson r = %.3f, p = %.3f | Spearman rho = %.3f, p = %.3f",
  nrow(merged), pear$estimate, pear$p.value, spear$estimate, spear$p.value
)

p <- ggplot(merged, aes(x = best_frequency_hz_cf, y = best_frequency_hz_am)) +
  geom_point(size = 2, alpha = 0.7, color = "#2C7BB6") +
  geom_smooth(method = "lm", se = TRUE, color = "#D7191C", linewidth = 1) +
  scale_x_log10() +
  scale_y_log10() +
  labs(
    title = "Scatter Plot and Correlation: CF x AM (sub-01)",
    subtitle = subtitle_txt,
    x = "CF best frequency (Hz, log scale)",
    y = "AM best frequency (Hz, log scale)"
  ) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

out_path <- file.path(dir, "sub-01_CFAM_scatter_correlation.png")
ggsave(out_path, p, width = 7.5, height = 6, dpi = 300, type = "cairo")
message("Saved: ", out_path)
