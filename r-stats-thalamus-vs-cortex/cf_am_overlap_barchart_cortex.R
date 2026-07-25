# Cortical counterpart to cf_am_overlap_barchart.R: counts of auditory-cortex
# voxels significant only for CF, only for AM, or for both.

Sys.setlocale("LC_CTYPE", "en_US.UTF-8")
library(ggplot2)

PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "..")
dir <- file.path(PROJECT_ROOT, "derivatives", "sub-01", "analysis", "smoothed_glm")
outdir <- file.path(PROJECT_ROOT, "outputs", "figures_cortex")

cf <- read.csv(file.path(dir, "sub-01_CF_voxel_bestfreq_table_CORTEX_SMOOTHED_mediansplit.csv"))
am <- read.csv(file.path(dir, "sub-01_AM_voxel_bestfreq_table_CORTEX_SMOOTHED_mediansplit.csv"))

cf$vox_id <- paste(cf$voxel_i, cf$voxel_j, cf$voxel_k, sep = "_")
am$vox_id <- paste(am$voxel_i, am$voxel_j, am$voxel_k, sep = "_")

only_cf <- setdiff(cf$vox_id, am$vox_id)
only_am <- setdiff(am$vox_id, cf$vox_id)
both <- intersect(cf$vox_id, am$vox_id)

counts <- data.frame(
  group = factor(c("CF", "AM", "Both"), levels = c("CF", "AM", "Both")),
  n = c(length(only_cf), length(only_am), length(both))
)

cat("Only CF:", length(only_cf), "\n")
cat("Only AM:", length(only_am), "\n")
cat("Both:   ", length(both), "\n")
cat("Total (union):", length(only_cf) + length(only_am) + length(both), "\n")

p <- ggplot(counts, aes(x = group, y = n, fill = group)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = n), vjust = -0.4, size = 5) +
  scale_fill_manual(values = c("CF" = "#D7191C", "AM" = "#2C7BB6", "Both" = "#7B3294"), guide = "none") +
  labs(
    title = "Significant voxels: CF only, AM only, or both",
    subtitle = "sub-01, auditory cortex",
    x = NULL,
    y = "Number of voxels"
  ) +
  ylim(0, max(counts$n) * 1.15) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

out_path <- file.path(outdir, "sub-01_CFAM_overlap_barchart_cortex.png")
ggsave(out_path, p, width = 7, height = 6, dpi = 300, type = "cairo")
message("Saved: ", out_path)
