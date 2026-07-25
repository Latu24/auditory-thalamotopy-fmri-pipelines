# Cortical counterpart to median_split_scatterplots.R: spatial scatterplots
# of CF and AM median-split groups (High/Low) plotted in in-plane world
# coordinates, for auditory-cortex voxels.

Sys.setlocale("LC_CTYPE", "en_US.UTF-8")
library(ggplot2)

PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "..")
dir <- file.path(PROJECT_ROOT, "derivatives", "sub-01", "analysis", "smoothed_glm")
outdir <- file.path(PROJECT_ROOT, "outputs", "figures_cortex")

make_scatter <- function(csv_path, title, out_path) {
  df <- read.csv(csv_path)
  df$freq_group <- factor(df$freq_group, levels = c("Low", "High"))

  p <- ggplot(df, aes(x = world_x_mm, y = world_y_mm, color = freq_group)) +
    geom_point(size = 1.2, alpha = 0.6) +
    scale_color_manual(values = c("Low" = "#2C7BB6", "High" = "#D7191C"), name = "Group") +
    labs(
      title = title,
      x = "world_x_mm (medial <-> lateral)",
      y = "world_y_mm (posterior <-> anterior)"
    ) +
    coord_fixed() +
    theme_minimal(base_size = 13) +
    theme(plot.title = element_text(face = "bold"))

  ggsave(out_path, p, width = 7, height = 6, dpi = 300, type = "cairo")
  message("Saved: ", out_path)
}

make_scatter(
  file.path(dir, "sub-01_CF_voxel_bestfreq_table_CORTEX_SMOOTHED_mediansplit.csv"),
  "CF median split - auditory cortex (sub-01)",
  file.path(outdir, "sub-01_CF_mediansplit_scatter_cortex.png")
)

make_scatter(
  file.path(dir, "sub-01_AM_voxel_bestfreq_table_CORTEX_SMOOTHED_mediansplit.csv"),
  "AM median split - auditory cortex (sub-01)",
  file.path(outdir, "sub-01_AM_mediansplit_scatter_cortex.png")
)
