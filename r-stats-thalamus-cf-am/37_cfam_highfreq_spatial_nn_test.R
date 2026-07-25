# Spatial nearest-neighbor permutation test for High/Low CF x High/Low AM
# co-localization in the thalamus, without requiring any voxel to be
# significant in both modalities (no conjunction voxels are used anywhere in
# this script). This asks a sharper question than a simple overlap test: is
# there EXTRA spatial clustering specifically between the HIGH-preference
# voxels of each modality (or LOW-with-LOW), beyond whatever baseline CF/AM
# spatial proximity already exists in the thalamus?
#
# Method:
#   - Two independent point clouds in scanner-RAS mm space: all 1075
#     CF-significant voxels (world_xyz_mm) and all 1669 AM-significant
#     voxels, each already labeled High/Low by that modality's own median.
#   - Observed statistic: symmetric mean nearest-neighbor distance between a
#     chosen pair of subsets (e.g. High-CF voxels <-> High-AM voxels) -- mean
#     over every point in each set of its distance to the CLOSEST point in
#     the other set, averaged across both directions.
#   - Null distribution (5000 permutations): reshuffle the High/Low label
#     WITHIN the CF-significant set (538/537 split preserved) and WITHIN the
#     AM-significant set (836/833 split preserved) independently each
#     iteration. This keeps the two point clouds' actual positions (and
#     hence the baseline CF/AM proximity) completely fixed and only
#     randomizes which specific voxels carry the "High" vs "Low" frequency
#     label, isolating whether frequency preference itself adds spatial
#     structure beyond the general CF/AM spatial co-localization.
#
# Inputs: per-voxel CF and AM best-fit-frequency tables (JSON) with
# scanner-space voxel coordinates.
# Outputs: JSON summary of observed distances and permutation p-values, a
# cached RDS of the null distributions, and a four-panel histogram (PNG).
#
# Run: Rscript 37_cfam_highfreq_spatial_nn_test.R

library(jsonlite)

set.seed(1)
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "..")
ROOT <- PROJECT_ROOT
SG <- file.path(ROOT, "derivatives/sub-01/analysis/smoothed_glm")

cf_table <- fromJSON(file.path(SG, "sub-01_CF_voxeltable_SMOOTHED.json"))
am_table <- fromJSON(file.path(SG, "sub-01_AM_voxeltable_SMOOTHED.json"))

cf_xyz <- as.matrix(do.call(rbind, cf_table$rows$world_xyz_mm))
am_xyz <- as.matrix(do.call(rbind, am_table$rows$world_xyz_mm))
cf_hz <- cf_table$rows$best_fit_freq_hz_continuous
am_hz <- am_table$rows$best_fit_freq_hz_continuous
n_cf <- nrow(cf_xyz); n_am <- nrow(am_xyz)
stopifnot(n_cf == 1075, n_am == 1669)

cf_med <- 10 ^ median(log10(cf_hz))
am_med <- 10 ^ median(log10(am_hz))
high_cf0 <- cf_hz >= cf_med
high_am0 <- am_hz >= am_med
cat(sprintf("CF median=%.1f Hz (High n=%d, Low n=%d)  AM median=%.2f Hz (High n=%d, Low n=%d)\n",
            cf_med, sum(high_cf0), sum(!high_cf0), am_med, sum(high_am0), sum(!high_am0)))

# symmetric mean nearest-neighbor distance between point sets A (from cf_xyz)
# and B (from am_xyz), given logical index vectors into cf_xyz/am_xyz
sym_nn_dist <- function(idxA, idxB) {
  A <- cf_xyz[idxA, , drop = FALSE]
  B <- am_xyz[idxB, , drop = FALSE]
  d2 <- outer(rowSums(A^2), rowSums(B^2), "+") - 2 * A %*% t(B)
  d2[d2 < 0] <- 0
  d <- sqrt(d2)
  mean_a_to_b <- mean(apply(d, 1, min))
  mean_b_to_a <- mean(apply(d, 2, min))
  (mean_a_to_b + mean_b_to_a) / 2
}

pairings <- list(
  HighCF_HighAM = list(a = high_cf0, b = high_am0),
  LowCF_LowAM   = list(a = !high_cf0, b = !high_am0),
  HighCF_LowAM  = list(a = high_cf0, b = !high_am0),
  LowCF_HighAM  = list(a = !high_cf0, b = high_am0)
)

observed <- sapply(pairings, function(p) sym_nn_dist(p$a, p$b))
cat("\nobserved symmetric mean NN distance (mm):\n")
print(round(observed, 4))

N_PERM <- 5000
null_mat <- matrix(NA_real_, nrow = N_PERM, ncol = 4, dimnames = list(NULL, names(pairings)))
n_high_cf <- sum(high_cf0); n_high_am <- sum(high_am0)

cat(sprintf("\nrunning %d permutations (reshuffling High/Low labels within each modality's own set) ...\n", N_PERM))
t0 <- Sys.time()
for (i in seq_len(N_PERM)) {
  perm_high_cf <- sample(c(rep(TRUE, n_high_cf), rep(FALSE, n_cf - n_high_cf)))
  perm_high_am <- sample(c(rep(TRUE, n_high_am), rep(FALSE, n_am - n_high_am)))
  null_mat[i, "HighCF_HighAM"] <- sym_nn_dist(perm_high_cf, perm_high_am)
  null_mat[i, "LowCF_LowAM"]   <- sym_nn_dist(!perm_high_cf, !perm_high_am)
  null_mat[i, "HighCF_LowAM"]  <- sym_nn_dist(perm_high_cf, !perm_high_am)
  null_mat[i, "LowCF_HighAM"]  <- sym_nn_dist(!perm_high_cf, perm_high_am)
}
cat(sprintf("done in %.1fs\n\n", as.numeric(Sys.time() - t0, units = "secs")))

results <- list()
for (nm in names(pairings)) {
  null_vals <- null_mat[, nm]
  # one-sided: is the OBSERVED distance smaller (tighter clustering) than the null?
  p_val <- (sum(null_vals <= observed[nm]) + 1) / (N_PERM + 1)
  results[[nm]] <- list(observed_mm = unname(observed[nm]),
                          null_mean_mm = mean(null_vals), null_sd_mm = sd(null_vals),
                          p_tighter_than_null = p_val)
  cat(sprintf("[%s] observed=%.4f mm   null=%.4f +/- %.4f mm   p(tighter)=%.4f\n",
              nm, observed[nm], mean(null_vals), sd(null_vals), p_val))
}

out_json <- file.path(SG, "sub-01_CFAM_highfreq_spatial_nn_SMOOTHED_R.json")
write(toJSON(list(CF_median_hz = cf_med, AM_median_hz = am_med, n_permutations = N_PERM,
                   results = results), auto_unbox = TRUE, pretty = TRUE), out_json)
cat(sprintf("\nwrote %s\n", out_json))

# cache the full null distributions + observed values so the plot can be
# redrawn/tweaked without rerunning the 5000-permutation loop
out_rds <- file.path(SG, "sub-01_CFAM_highfreq_spatial_nn_SMOOTHED_R.rds")
saveRDS(list(null_mat = null_mat, observed = observed), out_rds)
cat(sprintf("wrote %s\n", out_rds))

# =========================================================================
# graph: observed distance vs. null distribution, one panel per pairing
# =========================================================================
out_png <- file.path(SG, "sub-01_CFAM_highfreq_spatial_nn_SMOOTHED_R.png")
png(out_png, width = 1300, height = 1000, res = 130)
par(mfrow = c(2, 2), mar = c(4.5, 4.5, 3.5, 1.5), oma = c(0, 0, 3, 0))
titles <- c(HighCF_HighAM = "High-CF <-> High-AM", LowCF_LowAM = "Low-CF <-> Low-AM",
            HighCF_LowAM = "High-CF <-> Low-AM", LowCF_HighAM = "Low-CF <-> High-AM")
for (nm in names(pairings)) {
  null_vals <- null_mat[, nm]
  # pad the range so the observed line + its label never sit flush against
  # the plot border
  xr <- extendrange(c(null_vals, observed[nm]), f = 0.1)
  hist(null_vals, breaks = 40, col = "#cccccc", border = "white", xlim = xr,
       main = sprintf("%s\np(tighter)=%.4f", titles[nm], results[[nm]]$p_tighter_than_null),
       xlab = "symmetric mean NN distance (mm)")
  abline(v = observed[nm], col = "#d62728", lwd = 2)
  legend("topright", legend = "observed", col = "#d62728", lwd = 2, bty = "n", cex = 0.8)
}
mtext("High/Low CF x AM Spatial Nearest-Neighbor Permutation Test (R, no conjunction voxels used)",
      side = 3, line = 0.5, outer = TRUE, cex = 1.05, font = 2)
dev.off()
cat(sprintf("wrote %s\n", out_png))
cat("DONE\n")
