"""Compute a BrainVoyager-style FDR table (8 rows: q=[0.10,0.05,0.04,0.03,
0.02,0.01,0.005,0.001], each with a "critical std" (Benjamini-Hochberg,
independent/positive-dependence) and "critical conservative"
(Benjamini-Yekutieli, arbitrary-dependence, harmonic-sum-corrected) critical
|t| value) — matching the structure found in genuine BrainVoyager-produced
VMP files. Statistical maps exported without this table (SizeOfFDRTable=0,
FDRTableInfo=[]) render in BrainVoyager's significance-based coloring as
extremes-only rather than as a graded map, since the color lookup indexes
into an empty table.
"""
import numpy as np
from scipy import stats

Q_LEVELS = [0.10, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001]


def compute_fdr_table(t_abs_values, dof):
    """t_abs_values: 1D array of |t| at all in-mask voxels (two-tailed test).
    Returns (8,3) array: [q, critical_t_std (BH), critical_t_conservative (BY)]."""
    p = 2 * stats.t.sf(t_abs_values, dof)
    p_sorted = np.sort(p)
    m = p_sorted.size
    ranks = np.arange(1, m + 1)
    c_m = np.log(m) + 0.5772156649 + 1.0 / (2 * m)  # harmonic-sum approx (Euler-Mascheroni)

    rows = []
    for q in Q_LEVELS:
        bh_crit = ranks / m * q
        below = p_sorted <= bh_crit
        p_crit_std = p_sorted[np.max(np.where(below)[0])] if below.any() else p_sorted[0]

        by_crit = ranks / (m * c_m) * q
        below_by = p_sorted <= by_crit
        p_crit_cons = p_sorted[np.max(np.where(below_by)[0])] if below_by.any() else p_sorted[0]

        t_std = float(stats.t.isf(np.clip(p_crit_std, 1e-300, 1) / 2, dof))
        t_cons = float(stats.t.isf(np.clip(p_crit_cons, 1e-300, 1) / 2, dof))
        rows.append([q, t_std, t_cons])

    return np.array(rows, dtype=np.float64)
