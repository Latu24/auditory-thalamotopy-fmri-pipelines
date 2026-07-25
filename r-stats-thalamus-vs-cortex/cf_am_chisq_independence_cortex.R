# Chi-square test of independence between high/low CF and high/low AM voxel
# membership in auditory cortex, using data-driven median-split groups.
# Cortical counterpart to cf_am_chisq_independence.R: builds a 2x2
# contingency table, reports standardized residuals and conditional
# proportions, and writes the full results to a text file.

Sys.setlocale("LC_CTYPE", "en_US.UTF-8")

PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "..")
dir <- file.path(PROJECT_ROOT, "derivatives", "sub-01", "analysis", "smoothed_glm")

cf <- read.csv(file.path(dir, "sub-01_CF_voxel_bestfreq_table_CORTEX_SMOOTHED_mediansplit.csv"))
am <- read.csv(file.path(dir, "sub-01_AM_voxel_bestfreq_table_CORTEX_SMOOTHED_mediansplit.csv"))

cf$vox_id <- paste(cf$voxel_i, cf$voxel_j, cf$voxel_k, sep = "_")
am$vox_id <- paste(am$voxel_i, am$voxel_j, am$voxel_k, sep = "_")

all_ids <- union(cf$vox_id, am$vox_id)

high_cf_ids <- cf$vox_id[cf$freq_group == "High"]
high_am_ids <- am$vox_id[am$freq_group == "High"]

df <- data.frame(vox_id = all_ids)
df$High_CF <- factor(ifelse(df$vox_id %in% high_cf_ids, "Sim", "Não"), levels = c("Sim", "Não"))
df$High_AM <- factor(ifelse(df$vox_id %in% high_am_ids, "Sim", "Não"), levels = c("Sim", "Não"))

tbl <- table(df$High_CF, df$High_AM)
dimnames(tbl) <- list("High CF" = rownames(tbl), "High AM" = colnames(tbl))

cat("Auditory cortex - union of all voxels significant in CF and/or AM\n")
cat("n total de voxels:", length(all_ids), "\n")
cat("  significativos so em CF:", length(setdiff(cf$vox_id, am$vox_id)), "\n")
cat("  significativos so em AM:", length(setdiff(am$vox_id, cf$vox_id)), "\n")
cat("  significativos em ambos:", length(intersect(cf$vox_id, am$vox_id)), "\n\n")

cat("Tabela de contingencia 2x2 (linhas = High CF, colunas = High AM):\n")
print(tbl)
cat("\n")

chisq <- chisq.test(tbl)
print(chisq)

cat("\nResiduos padronizados (observado vs esperado):\n")
print(round(chisq$residuals, 2))

cat("\nProporcao de High AM dentro de cada grupo de High CF:\n")
print(round(prop.table(tbl, margin = 1), 3))

sink_path <- file.path(dir, "sub-01_CFAM_chisq_independence_2x2_cortex.txt")
sink(sink_path)
cat("Teste do Qui-Quadrado de Independencia: High CF x High AM (sub-01, auditory cortex)\n\n")
cat("Universo: uniao de todos os voxels significativos em CF e/ou AM (n =", length(all_ids), ")\n\n")
cat("Tabela de contingencia 2x2 (linhas = High CF, colunas = High AM):\n")
print(tbl)
cat("\n")
print(chisq)
cat("\nProporcao de High AM dentro de cada grupo de High CF:\n")
print(round(prop.table(tbl, margin = 1), 3))
sink()
message("Saved: ", sink_path)
