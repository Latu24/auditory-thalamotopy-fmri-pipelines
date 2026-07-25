# Sensitivity-analysis variant of cf_am_chisq_independence.R: classifies
# thalamus voxels as High/Low CF and High/Low AM using fixed absolute
# frequency cutoffs (CF >= 4000 Hz, AM >= 8 Hz) instead of a data-driven
# median split, to check whether the independence result is robust to the
# choice of thresholding method.

Sys.setlocale("LC_CTYPE", "en_US.UTF-8")

PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "..")
dir <- file.path(PROJECT_ROOT, "derivatives", "sub-01", "analysis", "smoothed_glm")

cf <- read.csv(file.path(dir, "sub-01_CF_voxel_bestfreq_table_SMOOTHED_mediansplit.csv"))
am <- read.csv(file.path(dir, "sub-01_AM_voxel_bestfreq_table_SMOOTHED_mediansplit.csv"))

CF_CUTOFF <- 4000
AM_CUTOFF <- 8

cf$vox_id <- paste(cf$voxel_i, cf$voxel_j, cf$voxel_k, sep = "_")
am$vox_id <- paste(am$voxel_i, am$voxel_j, am$voxel_k, sep = "_")

all_ids <- union(cf$vox_id, am$vox_id)

high_cf_ids <- cf$vox_id[cf$best_frequency_hz >= CF_CUTOFF]
high_am_ids <- am$vox_id[am$best_frequency_hz >= AM_CUTOFF]

df <- data.frame(vox_id = all_ids)
df$High_CF <- factor(ifelse(df$vox_id %in% high_cf_ids, "Sim", "Não"), levels = c("Sim", "Não"))
df$High_AM <- factor(ifelse(df$vox_id %in% high_am_ids, "Sim", "Não"), levels = c("Sim", "Não"))

tbl <- table(df$High_CF, df$High_AM)
dimnames(tbl) <- list("High CF" = rownames(tbl), "High AM" = colnames(tbl))

cat("Fixed thresholds: High CF >=", CF_CUTOFF, "Hz | High AM >=", AM_CUTOFF, "Hz\n\n")
cat("Universo: uniao de todos os voxels significativos em CF e/ou AM\n")
cat("n total de voxels:", length(all_ids), "\n\n")

cat("Tabela de contingencia 2x2 (linhas = High CF, colunas = High AM):\n")
print(tbl)
cat("\n")

chisq <- chisq.test(tbl)
print(chisq)

cat("\nResiduos padronizados (observado vs esperado):\n")
print(round(chisq$residuals, 2))

cat("\nProporcao de High AM dentro de cada grupo de High CF:\n")
print(round(prop.table(tbl, margin = 1), 3))

sink_path <- file.path(dir, "sub-01_CFAM_chisq_independence_fixedthreshold_2x2.txt")
sink(sink_path)
cat("Teste do Qui-Quadrado de Independencia: High CF x High AM (sub-01)\n")
cat("Fixed thresholds: High CF >=", CF_CUTOFF, "Hz | High AM >=", AM_CUTOFF, "Hz\n\n")
cat("Universo: uniao de todos os voxels significativos em CF e/ou AM (n =", length(all_ids), ")\n\n")
cat("Tabela de contingencia 2x2 (linhas = High CF, colunas = High AM):\n")
print(tbl)
cat("\n")
print(chisq)
cat("\nProporcao de High AM dentro de cada grupo de High CF:\n")
print(round(prop.table(tbl, margin = 1), 3))
sink()
message("Saved: ", sink_path)
