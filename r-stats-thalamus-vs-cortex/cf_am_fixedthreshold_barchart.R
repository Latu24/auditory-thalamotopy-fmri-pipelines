# Bar chart of thalamus voxel counts in each fixed-threshold group (High/Low
# CF, High/Low AM), shown as four separate bars rather than a 2x2
# cross-tabulation, using the same cutoffs as the fixed-threshold chi-square
# test.

Sys.setlocale("LC_CTYPE", "en_US.UTF-8")
library(ggplot2)

PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "..")
dir <- file.path(PROJECT_ROOT, "derivatives", "sub-01", "analysis", "smoothed_glm")
outdir <- file.path(PROJECT_ROOT, "outputs", "figures")

cf <- read.csv(file.path(dir, "sub-01_CF_voxel_bestfreq_table_SMOOTHED_mediansplit.csv"))
am <- read.csv(file.path(dir, "sub-01_AM_voxel_bestfreq_table_SMOOTHED_mediansplit.csv"))

CF_CUTOFF <- 4000
AM_CUTOFF <- 8

cf_high_n <- sum(cf$best_frequency_hz >= CF_CUTOFF)
cf_low_n  <- sum(cf$best_frequency_hz <  CF_CUTOFF)
am_high_n <- sum(am$best_frequency_hz >= AM_CUTOFF)
am_low_n  <- sum(am$best_frequency_hz <  AM_CUTOFF)

labels <- c(
  "High CF" = sprintf("High CF\n(≥ %d Hz)", CF_CUTOFF),
  "Low CF"  = sprintf("Low CF\n(< %d Hz)", CF_CUTOFF),
  "High AM" = sprintf("High AM\n(≥ %d Hz)", AM_CUTOFF),
  "Low AM"  = sprintf("Low AM\n(< %d Hz)", AM_CUTOFF)
)

counts <- data.frame(
  group = factor(c("High CF", "Low CF", "High AM", "Low AM"),
                 levels = c("High CF", "Low CF", "High AM", "Low AM")),
  n = c(cf_high_n, cf_low_n, am_high_n, am_low_n)
)

print(counts)

p <- ggplot(counts, aes(x = group, y = n, fill = group)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = n), vjust = -0.4, size = 5) +
  scale_x_discrete(labels = labels) +
  scale_fill_manual(
    values = c("High CF" = "#D7191C", "Low CF" = "#FDAE61",
               "High AM" = "#2C7BB6", "Low AM" = "#ABD9E9"),
    guide = "none"
  ) +
  labs(
    title = "Fixed-threshold split results: CF and AM (sub-01)",
    subtitle = sprintf("High CF = %d Hz | High AM = %d Hz", CF_CUTOFF, AM_CUTOFF),
    x = NULL,
    y = "Number of voxels"
  ) +
  ylim(0, max(counts$n) * 1.15) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

out_path <- file.path(outdir, "sub-01_CFAM_fixedthreshold_barchart.png")
ggsave(out_path, p, width = 7, height = 6, dpi = 300, type = "cairo")
message("Saved: ", out_path)
