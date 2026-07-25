# Cortical counterpart to cf_am_clustered_barchart_fixedthreshold.R:
# fixed-threshold (CF >= 4000 Hz, AM >= 8 Hz) clustered bar chart of CF x AM
# voxel counts in auditory cortex.

Sys.setlocale("LC_CTYPE", "en_US.UTF-8")
library(ggplot2)

PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "..")
dir <- file.path(PROJECT_ROOT, "derivatives", "sub-01", "analysis", "smoothed_glm")
outdir <- file.path(PROJECT_ROOT, "outputs", "figures_cortex")

cf <- read.csv(file.path(dir, "sub-01_CF_voxel_bestfreq_table_CORTEX_SMOOTHED_mediansplit.csv"))
am <- read.csv(file.path(dir, "sub-01_AM_voxel_bestfreq_table_CORTEX_SMOOTHED_mediansplit.csv"))

CF_CUTOFF <- 4000
AM_CUTOFF <- 8

cf$vox_id <- paste(cf$voxel_i, cf$voxel_j, cf$voxel_k, sep = "_")
am$vox_id <- paste(am$voxel_i, am$voxel_j, am$voxel_k, sep = "_")

all_ids <- union(cf$vox_id, am$vox_id)
high_cf_ids <- cf$vox_id[cf$best_frequency_hz >= CF_CUTOFF]
high_am_ids <- am$vox_id[am$best_frequency_hz >= AM_CUTOFF]

df <- data.frame(vox_id = all_ids)
df$High_CF <- factor(ifelse(df$vox_id %in% high_cf_ids, "High CF", "Low CF"), levels = c("High CF", "Low CF"))
df$High_AM <- factor(ifelse(df$vox_id %in% high_am_ids, "High AM", "Low AM"), levels = c("High AM", "Low AM"))

tbl <- table(df$High_CF, df$High_AM)
chisq <- chisq.test(tbl)

counts <- as.data.frame(tbl)
names(counts) <- c("High_CF", "High_AM", "n")

subtitle_txt <- sprintf(
  "n = %d voxels | High CF %d Hz and above; High AM %d Hz and above\nchi-sq = %.2f, df = %d, p = %.3g",
  sum(counts$n), CF_CUTOFF, AM_CUTOFF, chisq$statistic, chisq$parameter, chisq$p.value
)

p <- ggplot(counts, aes(x = High_CF, y = n, fill = High_AM)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_text(aes(label = n), position = position_dodge(width = 0.8), vjust = -0.4, size = 4.5) +
  scale_fill_manual(values = c("High AM" = "#D7191C", "Low AM" = "#2C7BB6"), name = "AM group") +
  labs(
    title = "Significant activation: High CF x High AM\n(sub-01, auditory cortex, fixed threshold)",
    subtitle = subtitle_txt,
    x = "CF group",
    y = "Number of voxels"
  ) +
  ylim(0, max(counts$n) * 1.15) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

out_path <- file.path(outdir, "sub-01_CFAM_clustered_barchart_fixedthreshold_cortex.png")
ggsave(out_path, p, width = 8, height = 6.5, dpi = 300, type = "cairo")
message("Saved: ", out_path)
