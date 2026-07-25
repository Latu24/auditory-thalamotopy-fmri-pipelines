# Bar chart expressing CF and AM voxel counts (overall and by High/Low
# group) as a percentage of all significant thalamus voxels (union of the
# CF and AM maps), giving a normalized view of relative prevalence.

Sys.setlocale("LC_CTYPE", "en_US.UTF-8")
library(ggplot2)

PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "..")
dir <- file.path(PROJECT_ROOT, "derivatives", "sub-01", "analysis", "smoothed_glm")
outdir <- file.path(PROJECT_ROOT, "outputs", "figures")

cf <- read.csv(file.path(dir, "sub-01_CF_voxel_bestfreq_table_SMOOTHED_mediansplit.csv"))
am <- read.csv(file.path(dir, "sub-01_AM_voxel_bestfreq_table_SMOOTHED_mediansplit.csv"))

cf$vox_id <- paste(cf$voxel_i, cf$voxel_j, cf$voxel_k, sep = "_")
am$vox_id <- paste(am$voxel_i, am$voxel_j, am$voxel_k, sep = "_")
union_n <- length(union(cf$vox_id, am$vox_id))

counts <- data.frame(
  group = factor(
    c("CF overall", "CF High", "CF Low", "AM overall", "AM High", "AM Low"),
    levels = c("CF overall", "CF High", "CF Low", "AM overall", "AM High", "AM Low")
  ),
  n = c(
    nrow(cf), sum(cf$freq_group == "High"), sum(cf$freq_group == "Low"),
    nrow(am), sum(am$freq_group == "High"), sum(am$freq_group == "Low")
  )
)
counts$pct <- counts$n / union_n * 100

print(counts)

p <- ggplot(counts, aes(x = group, y = pct, fill = group)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = sprintf("%.1f%%\n(n=%d)", pct, n)), vjust = -0.3, size = 4) +
  scale_fill_manual(
    values = c("CF overall" = "#7B3294", "CF High" = "#D7191C", "CF Low" = "#FDAE61",
               "AM overall" = "#008837", "AM High" = "#2C7BB6", "AM Low" = "#ABD9E9"),
    guide = "none"
  ) +
  labs(
    title = "CF and AM prevalence as % of all significant voxels (sub-01)",
    subtitle = sprintf("n = %d voxels total (union of CF and AM maps)", union_n),
    x = NULL,
    y = "% of all significant voxels"
  ) +
  ylim(0, max(counts$pct) * 1.2) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

out_path <- file.path(outdir, "sub-01_CFAM_percentage_barchart.png")
ggsave(out_path, p, width = 8.5, height = 6, dpi = 300, type = "cairo")
message("Saved: ", out_path)
