# Cortical counterpart to cf_am_median_split_barchart.R: median-split voxel
# counts for CF and AM groups in auditory cortex, shown as four separate
# bars.

Sys.setlocale("LC_CTYPE", "en_US.UTF-8")
library(ggplot2)

PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "..")
dir <- file.path(PROJECT_ROOT, "derivatives", "sub-01", "analysis", "smoothed_glm")
outdir <- file.path(PROJECT_ROOT, "outputs", "figures_cortex")

cf <- read.csv(file.path(dir, "sub-01_CF_voxel_bestfreq_table_CORTEX_SMOOTHED_mediansplit.csv"))
am <- read.csv(file.path(dir, "sub-01_AM_voxel_bestfreq_table_CORTEX_SMOOTHED_mediansplit.csv"))

counts <- data.frame(
  group = factor(c("High CF", "Low CF", "High AM", "Low AM"),
                 levels = c("High CF", "Low CF", "High AM", "Low AM")),
  n = c(
    sum(cf$freq_group == "High"),
    sum(cf$freq_group == "Low"),
    sum(am$freq_group == "High"),
    sum(am$freq_group == "Low")
  )
)

print(counts)

p <- ggplot(counts, aes(x = group, y = n, fill = group)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = n), vjust = -0.4, size = 5) +
  scale_fill_manual(
    values = c("High CF" = "#D7191C", "Low CF" = "#FDAE61",
               "High AM" = "#2C7BB6", "Low AM" = "#ABD9E9"),
    guide = "none"
  ) +
  labs(
    title = "Median split results: CF and AM",
    subtitle = "sub-01, auditory cortex",
    x = NULL,
    y = "Number of voxels"
  ) +
  ylim(0, max(counts$n) * 1.15) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

out_path <- file.path(outdir, "sub-01_CFAM_mediansplit_barchart_cortex.png")
ggsave(out_path, p, width = 7, height = 6, dpi = 300, type = "cairo")
message("Saved: ", out_path)
