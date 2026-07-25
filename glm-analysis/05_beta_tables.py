"""Refit the carrier-frequency and amplitude-modulation GLMs at full-volume
resolution to recover per-voxel betas (not persisted by the earlier scripts,
which only kept the derived statistical maps), then export beta tables for
(a) the activated cluster (uncorrected F threshold matching the VMP header)
and (b) FDR-significant voxels (Benjamini-Hochberg, q<0.05) for each design,
as plain-text tables.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import stats
import glmlib as G

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ROOT = PROJECT_ROOT
PRT_DIR = os.path.join(PROJECT_ROOT, "PRTs")
OUT_DIR = f"{ROOT}/derivatives/sub-01/analysis/beta_tables"
os.makedirs(OUT_DIR, exist_ok=True)

vtc_paths = [f"{ROOT}/derivatives/sub-01/reg/sub-01_run-{r}_preproc_coreg.vtc" for r in G.RUNS]
hrf = G.two_gamma_hrf()


def build_designs(prt_glob):
    headers = [G.read_vtc_header(p) for p in vtc_paths]
    designs = []
    order = None
    for r, h in zip(G.RUNS, headers):
        nvols = h["DimT"]
        Xr, order = G.build_run_design(prt_glob.format(r=r), nvols, hrf, condition_order=order)
        designs.append(Xr)
    return designs, order


def fdr_bh(pvals, q=0.05):
    """Benjamini-Hochberg FDR, operating on a flat array of p-values
    (only over in-mask voxels). Returns boolean significance mask (same shape)."""
    flat = pvals.ravel()
    n = flat.size
    order = np.argsort(flat)
    ranked = flat[order]
    thresh_line = (np.arange(1, n + 1) / n) * q
    below = ranked <= thresh_line
    if not np.any(below):
        return np.zeros_like(pvals, dtype=bool)
    kmax = np.max(np.where(below)[0])
    cutoff = ranked[kmax]
    return (pvals <= cutoff)


def write_table(path, header_cols, rows_iter, title):
    with open(path, "w") as f:
        f.write(title + "\n")
        f.write("=" * len(title) + "\n\n")
        f.write("\t".join(header_cols) + "\n")
        n = 0
        for row in rows_iter:
            f.write("\t".join(row) + "\n")
            n += 1
        f.write(f"\n[{n} voxel rows]\n")
    return n


def run(label, prt_glob, cond_prefix):
    print(f"=== {label} ===")
    designs, cond_order = build_designs(prt_glob)
    fit = G.fit_multirun_glm(vtc_paths, designs, hrf, log=print)
    F = G.f_omnibus(fit)  # shape3 raw (DimZ,DimY,DimX)
    mask = fit["mask"]

    # p-values from F distribution using this map's DF1/DF2
    df1, df2 = fit["n_cond"], fit["dof"]
    with np.errstate(invalid="ignore"):
        p = stats.f.sf(F, df1, df2)
    p[~mask] = 1.0

    # -- activated cluster: pipeline's own uncorrected threshold (matches VMP header) --
    crit_uncorr = stats.f.isf(0.05, df1, df2)
    cluster_mask = mask & (F >= crit_uncorr)
    print(f"  uncorrected F-crit(p<.05, df1={df1},df2={df2}) = {crit_uncorr:.3f}  "
          f"-> {cluster_mask.sum()} voxels")

    # -- FDR-significant (BH, q<0.05) restricted to in-mask voxels --
    p_masked = np.where(mask, p, 1.0)
    fdr_mask = fdr_bh(p_masked[mask], q=0.05)
    full_fdr_mask = np.zeros_like(mask)
    full_fdr_mask[mask] = fdr_mask
    print(f"  FDR q<.05 -> {full_fdr_mask.sum()} voxels")

    betas = fit["betas"][:fit["n_cond"]]  # (n_cond, Z, Y, X)
    h0 = fit["headers"][0]

    def rows_for(vmask):
        idx = np.argwhere(vmask)
        # sort by F descending for readability
        fvals = F[vmask]
        order_ = np.argsort(-fvals)
        idx = idx[order_]
        for (z, y, x) in idx:
            bx = h0["XStart"] + x * h0["VTC resolution"]
            by = h0["YStart"] + y * h0["VTC resolution"]
            bz = h0["ZStart"] + z * h0["VTC resolution"]
            row = [f"{bx}", f"{by}", f"{bz}", f"{F[z,y,x]:.3f}", f"{p[z,y,x]:.2e}"]
            row += [f"{betas[c, z, y, x]:.2f}" for c in range(betas.shape[0])]
            yield row

    cols = ["VMR_X", "VMR_Y", "VMR_Z", "F", "p"] + cond_order

    p1 = f"{OUT_DIR}/{label}_cluster_betas.txt"
    n1 = write_table(p1, cols, rows_for(cluster_mask),
                      f"{label} — per-voxel betas, activated cluster (F >= {crit_uncorr:.3f}, uncorrected p<.05, df={df1},{df2})")
    print(f"  wrote {p1} ({n1} rows)")

    p2 = f"{OUT_DIR}/{label}_FDR_betas.txt"
    n2 = write_table(p2, cols, rows_for(full_fdr_mask),
                      f"{label} — per-voxel betas, FDR-significant voxels (Benjamini-Hochberg q<0.05)")
    print(f"  wrote {p2} ({n2} rows)")
    return p1, p2


if __name__ == "__main__":
    outs = []
    outs += list(run("CF", PRT_DIR + "/sub01_run-{r}_tone_events_by_frequency_timestamp.prt", "Freq"))
    outs += list(run("AM", PRT_DIR + "/sub01_run-{r}_tone_events_by_AM_timestamp.prt", "AM"))
    print("DONE")
    for o in outs:
        print(o)
