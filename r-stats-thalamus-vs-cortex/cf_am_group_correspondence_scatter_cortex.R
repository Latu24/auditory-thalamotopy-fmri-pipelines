# Cortical counterpart to cf_am_group_correspondence_scatter.R: CF vs AM
# scatterplot (log-log axes) colored by CF/AM quadrant, with median-split
# reference lines, for auditory-cortex voxels.

Sys.setlocale("LC_CTYPE", "en_US.UTF-8")
library(ggplot2)

PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "..")
dir <- file.path(PROJECT_ROOT, "derivatives", "sub-01", "analysis", "smoothed_glm")
outdir <- file.path(PROJECT_ROOT, "outputs", "figures_cortex")

cf <- read.csv(file.path(dir, "sub-01_CF_voxel_bestfreq_table_CORTEX_SMOOTHED_mediansplit.csv"))
am <- read.csv(file.path(dir, "sub-01_AM_voxel_bestfreq_table_CORTEX_SMOOTHED_mediansplit.csv"))

merged <- merge(
  cf, am,
  by = c("voxel_i", "voxel_j", "voxel_k"),
  suffixes = c("_cf", "_am")
)

cf_median <- median(cf$best_frequency_hz)
am_median <- median(am$best_frequency_hz)

merged$freq_group_cf <- factor(merged$freq_group_cf, levels = c("Low", "High"))
merged$freq_group_am <- factor(merged$freq_group_am, levels = c("Low", "High"))
merged$quadrant <- paste0(merged$freq_group_cf, " CF / ", merged$freq_group_am, " AM")

tbl <- table(merged$freq_group_cf, merged$freq_group_am)
chisq <- chisq.test(tbl)

cat("Contingency table (rows = CF group, cols = AM group), auditory cortex:\n")
print(tbl)
cat("\nChi-squared test:\n")
print(chisq)

subtitle_txt <- sprintf(
  "n = %d voxels significant in both maps | chi-sq = %.2f, df = %d, p = %.3g",
  nrow(merged), chisq$statistic, chisq$parameter, chisq$p.value
)

quad_counts <- table(merged$quadrant)
merged$quadrant <- paste0(merged$quadrant, " (n=", quad_counts[merged$quadrant], ")")

quad_colors <- setNames(
  c("#2C7BB6", "#D7191C", "#ABD9E9", "#FDAE61"),
  paste0(
    c("Low CF / Low AM", "High CF / High AM", "Low CF / High AM", "High CF / Low AM"),
    " (n=", quad_counts[c("Low CF / Low AM", "High CF / High AM", "Low CF / High AM", "High CF / Low AM")], ")"
  )
)

p <- ggplot(merged, aes(x = best_frequency_hz_cf, y = best_frequency_hz_am, color = quadrant)) +
  geom_point(size = 1.3, alpha = 0.6) +
  geom_vline(xintercept = cf_median, linetype = "dashed", color = "grey40") +
  geom_hline(yintercept = am_median, linetype = "dashed", color = "grey40") +
  scale_x_log10() +
  scale_y_log10() +
  scale_color_manual(values = quad_colors, name = "Group") +
  labs(
    title = "Does high CF go with high AM? - auditory cortex (sub-01)",
    subtitle = subtitle_txt,
    x = "CF best frequency (Hz, log scale)",
    y = "AM best frequency (Hz, log scale)"
  ) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

out_path <- file.path(outdir, "sub-01_CFAM_group_correspondence_scatter_cortex.png")
ggsave(out_path, p, width = 7.5, height = 6, dpi = 300, type = "cairo")
message("Saved: ", out_path)
