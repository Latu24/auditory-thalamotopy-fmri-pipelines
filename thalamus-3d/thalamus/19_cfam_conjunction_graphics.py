"""Thalamus pipeline -- three additional CF x AM conjunction graphics, built
on top of 18_cfam_conjunction_smoothed.py's already-finalized voxel tables
(that script and its outputs are untouched by this one):

  A) tri-color (CF-only / AM-only / both) native-space anatomical slice
     montage, cropped tightly to the thalamus bounding box.
  B) bar chart of CF-only / AM-only / both voxel counts.
  C) CF-F vs AM-F scatter over the union of active-in-thalamus voxels (not
     just the conjunction), with Fcrit threshold lines and a
     conjunction-zone (upper-right quadrant) cross-check against the
     already-established conjunction count.

This script imports (does not copy/redefine) script 18's own GLM-reading,
.ctr-parsing, omnibus-F formula, and native-space-transform code
(parse_ctr / verify_ctr_matches_predictors / to_native / MODALITIES / P_THRESH)
via importlib, exactly as 18 itself imports gauss/fit_r2 from
10_bestfreq_from_glm.py -- so the F-values/Fcrit/native placement used here
are guaranteed numerically identical to 18's, not a second, independently
drifting implementation. The one thing 18's process_modality() computes
internally but does not persist to disk is the full native-space F volume
(all thalamus voxels, not just the p<0.01-active ones) -- panel (C) needs
that (a CF-only voxel needs its own, sub-threshold AM F-value too), so this
script recomputes it here via the identical imported formula/functions.

Inputs: the per-modality and conjunction voxel tables written by script 18,
plus the native-space thalamus mask and anatomical NIfTI. Outputs: three PNG
figures (tri-color slice montage, bar chart, F-value union scatter).
"""
import os, json, time, importlib.util
import numpy as np
import nibabel as nib

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

ROOT = PROJECT_ROOT
ANA = f"{ROOT}/derivatives/sub-01/analysis"
SG = f"{ANA}/smoothed_glm"
WORK = f"{ANA}/thalamus_work"

# ---- import (do not duplicate) script 18's already-validated GLM/.ctr/native-
#      transform code, exactly the way 18 itself imports from script 10 ----
_spec = importlib.util.spec_from_file_location(
    "cfam18", f"{ROOT}/scripts/thalamus/18_cfam_conjunction_smoothed.py")
_bf18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bf18)

ANAT_NII = _bf18.ANAT_NII
THAL_NII = _bf18.THAL_NII
MGB_NII = _bf18.MGB_NII
MODALITIES = _bf18.MODALITIES


def compute_full_F_native(tag, cfg):
    """Recompute the FULL native-space omnibus-F volume (every voxel, not just
    p<0.01-active ones) for one modality, using 18's own imported .ctr-
    verification + omnibus-F formula + native-space transform verbatim (same
    contrast columns, same sig2=SS*(1-R2**2)/dof, same to_native()). Returns
    (F_native (240,320,320) float32, Fcrit, dof, q)."""
    h, R2, SS, beta, SSXiY, meantc, ARlag = _bf18.glmmod.read_glm(cfg["glm"])
    del SSXiY, ARlag, meantc
    dof = h["Nr time points"] - h["Nr all predictors"]
    assert dof == cfg["dof_expected"], f"[{tag}] dof mismatch"
    names = [p["Name (custom)"] for p in h["Predictor info"]]
    bbox = {k: int(h[k]) for k in ("XStart", "XEnd", "YStart", "YEnd", "ZStart", "ZEnd")}
    ctr = _bf18.parse_ctr(cfg["ctr"])
    cols, named = _bf18.verify_ctr_matches_predictors(ctr, names)
    q = len(cols)
    assert q == cfg["q"]
    invXX = np.asarray(h["Inverted X'X matrix"], dtype=np.float64)
    Minv = np.linalg.inv(invXX[np.ix_(cols, cols)])
    bc = beta[..., cols].astype(np.float64)
    quad = np.einsum("...i,ij,...j->...", bc, Minv, bc)
    sig2 = SS.astype(np.float64) * (1.0 - R2.astype(np.float64) ** 2) / dof
    with np.errstate(divide="ignore", invalid="ignore"):
        F = quad / (q * sig2)
    F[~np.isfinite(F)] = 0.0
    Fcrit = float(_bf18.stats.f.isf(_bf18.P_THRESH, q, dof))
    F_native = _bf18.to_native(F.astype(np.float32), bbox, fill_value=0.0)
    log(f"[{tag}] recomputed full native F volume: peakF={F.max():.4f} "
        f"(vs. 18's own recorded peakF -- see cross-check below) Fcrit={Fcrit:.4f}")
    return F_native, Fcrit, dof, q


def main():
    log("loading previously-written (untouched) per-modality/conjunction tables + masks ...")
    cf_table = json.load(open(f"{SG}/sub-01_CF_voxeltable_SMOOTHED.json"))
    am_table = json.load(open(f"{SG}/sub-01_AM_voxeltable_SMOOTHED.json"))
    conj = json.load(open(f"{SG}/sub-01_CFAM_conjunction_SMOOTHED.json"))

    anat_nib = nib.load(ANAT_NII)
    anat_arr = np.asarray(anat_nib.dataobj, dtype=np.float32)
    aff = anat_nib.affine
    thal_mask = np.asarray(nib.load(THAL_NII).dataobj) > 0

    cf_ijk = set(tuple(r["voxel_ijk_native"]) for r in cf_table["rows"])
    am_ijk = set(tuple(r["voxel_ijk_native"]) for r in am_table["rows"])
    both_ijk_from_conjfile = set(tuple(r["voxel_ijk_native"]) for r in conj["rows"])
    both_ijk = cf_ijk & am_ijk
    assert both_ijk == both_ijk_from_conjfile, (
        "conjunction file's voxel set != (CF table set) & (AM table set) -- inconsistent inputs")
    cf_only_ijk = cf_ijk - both_ijk
    am_only_ijk = am_ijk - both_ijk
    n_cf, n_am, n_both = len(cf_ijk), len(am_ijk), len(both_ijk)
    n_cf_only, n_am_only = len(cf_only_ijk), len(am_only_ijk)
    log(f"CF active-in-thalamus n={n_cf}  AM active-in-thalamus n={n_am}  both n={n_both}")
    log(f"CF-only n={n_cf_only}  AM-only n={n_am_only}")
    assert (n_cf, n_am, n_both) == (1075, 1669, 639), (
        f"active-voxel counts changed from the already-shipped run "
        f"({n_cf},{n_am},{n_both}) != (1075,1669,639) -- inputs drifted, investigate")
    assert (n_cf_only, n_am_only) == (436, 1030), (
        f"CF-only/AM-only counts ({n_cf_only},{n_am_only}) != anchors (436,1030)")

    # =========================================================================
    # (A) tri-color native-space anatomical slice montage, thalamus-cropped
    # =========================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    COLOR_CF_ONLY = "#d62728"   # red
    COLOR_AM_ONLY = "#1f77b4"   # blue
    COLOR_BOTH = "#9467bd"      # purple

    idx = np.argwhere(thal_mask)
    (imin, jmin, kmin), (imax, jmax, kmax) = idx.min(0), idx.max(0)
    MARGIN = 8
    i0, i1 = max(0, imin - MARGIN), min(anat_arr.shape[0], imax + MARGIN + 1)
    k0, k1 = max(0, kmin - MARGIN), min(anat_arr.shape[2], kmax + MARGIN + 1)
    log(f"thalamus bbox: i[{imin},{imax}] j[{jmin},{jmax}] k[{kmin},{kmax}]  "
        f"crop (margin={MARGIN}): i[{i0},{i1}) k[{k0},{k1})")

    # coronal slices (project convention: axis1=j="coronal", per
    # 12_reconcile_space.py's three_plane_overlay) spanning the thalamus's own
    # anterior-posterior (j) extent -- representative subset, one panel per slice.
    N_SLICES = 12
    j_values = sorted(set(int(v) for v in np.linspace(jmin, jmax, N_SLICES).round()))
    ncols = 4
    nrows = int(np.ceil(len(j_values) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.3 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    def cat_for(ijk):
        if ijk in both_ijk:
            return COLOR_BOTH
        if ijk in cf_only_ijk:
            return COLOR_CF_ONLY
        if ijk in am_only_ijk:
            return COLOR_AM_ONLY
        return None

    all_active_ijk = cf_only_ijk | am_only_ijk | both_ijk
    for ax_i, jv in enumerate(j_values):
        ax = axes[ax_i]
        bg = anat_arr[i0:i1, jv, k0:k1]
        ax.imshow(bg.T, cmap="gray", origin="lower", vmin=0,
                  vmax=np.percentile(anat_arr[anat_arr > 0], 99.5))
        pts = [p for p in all_active_ijk if p[1] == jv]
        if pts:
            xs = [p[0] - i0 for p in pts]
            ys = [p[2] - k0 for p in pts]
            cs = [cat_for(p) for p in pts]
            ax.scatter(xs, ys, c=cs, s=26, marker="s", linewidths=0.3, edgecolors="k")
        world = aff @ np.array([ (imin+imax)/2.0, jv, (kmin+kmax)/2.0, 1.0 ])
        ax.set_title(f"coronal j={jv}  (y={world[1]:.1f} mm)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for ax_i in range(len(j_values), len(axes)):
        axes[ax_i].axis("off")

    legend_elems = [
        Patch(facecolor=COLOR_CF_ONLY, edgecolor="k", label=f"CF only (n={n_cf_only})"),
        Patch(facecolor=COLOR_AM_ONLY, edgecolor="k", label=f"AM only (n={n_am_only})"),
        Patch(facecolor=COLOR_BOTH, edgecolor="k", label=f"Both / conjunction (n={n_both})"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("sub-01 thalamus: CF-only / AM-only / conjunction voxels "
                 "(p<0.01 uncorrected omnibus-F), native space, coronal montage",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_a = f"{SG}/sub-01_CFAM_conjunctionmap_SMOOTHED.png"
    fig.savefig(out_a, dpi=140)
    plt.close(fig)
    log(f"wrote {out_a}")

    # =========================================================================
    # (B) bar chart of counts
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(5.5, 5))
    labels = ["CF only", "AM only", "Both"]
    counts = [n_cf_only, n_am_only, n_both]
    colors = [COLOR_CF_ONLY, COLOR_AM_ONLY, COLOR_BOTH]
    bars = ax2.bar(labels, counts, color=colors, edgecolor="k")
    for b, c in zip(bars, counts):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + max(counts) * 0.01,
                  str(c), ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax2.set_ylabel("voxel count (native-space thalamus)")
    ax2.set_title("sub-01 CF x AM: p<0.01 active-voxel counts\n(thalamus, "
                  "CF q=36 dof=1326; AM q=9 dof=1353)")
    ax2.set_ylim(0, max(counts) * 1.15)
    fig2.tight_layout()
    out_b = f"{SG}/sub-01_CFAM_barcounts_SMOOTHED.png"
    fig2.savefig(out_b, dpi=150)
    plt.close(fig2)
    log(f"wrote {out_b}  counts CF-only={n_cf_only} AM-only={n_am_only} both={n_both}")

    # =========================================================================
    # (C) CF-F vs AM-F scatter, UNION of active voxels, full native F volumes
    # =========================================================================
    log("recomputing full native-space F volumes (CF, AM) for the union-set scatter ...")
    F_cf_native, Fcrit_cf, dof_cf, q_cf = compute_full_F_native("CF", MODALITIES["CF"])
    F_am_native, Fcrit_am, dof_am, q_am = compute_full_F_native("AM", MODALITIES["AM"])
    assert abs(Fcrit_cf - cf_table["Fcrit_p01"]) < 1e-6
    assert abs(Fcrit_am - am_table["Fcrit_p01"]) < 1e-6

    # cross-check the recomputed full arrays against 18's own recorded,
    # rounded per-voxel F_value at every already-active voxel (both tables)
    max_abs_diff_cf = max(abs(F_cf_native[tuple(r["voxel_ijk_native"])] - r["F_value"])
                          for r in cf_table["rows"])
    max_abs_diff_am = max(abs(F_am_native[tuple(r["voxel_ijk_native"])] - r["F_value"])
                          for r in am_table["rows"])
    log(f"cross-check vs. 18's stored F_value (rounding-only diff expected): "
        f"CF max|diff|={max_abs_diff_cf:.4f}  AM max|diff|={max_abs_diff_am:.4f}")
    assert max_abs_diff_cf < 5e-3 and max_abs_diff_am < 5e-3, (
        "recomputed full F volume disagrees with 18's own stored F_value beyond "
        "rounding -- STOPPING, the two implementations are not consistent")

    union_ijk = sorted(cf_only_ijk | am_only_ijk | both_ijk)
    cf_F_union = np.array([F_cf_native[p] for p in union_ijk])
    am_F_union = np.array([F_am_native[p] for p in union_ijk])
    cat_union = np.array([("both" if p in both_ijk else
                           ("cf_only" if p in cf_only_ijk else "am_only"))
                          for p in union_ijk])

    n_upper_right = int(((cf_F_union > Fcrit_cf) & (am_F_union > Fcrit_am)).sum())
    log(f"UNION-SET CROSS-CHECK: points with CF_F>{Fcrit_cf:.4f} AND "
        f"AM_F>{Fcrit_am:.4f} = {n_upper_right}  (must equal conjunction n={n_both})")
    cross_check_ok = (n_upper_right == n_both)
    if not cross_check_ok:
        log(f"*** CROSS-CHECK FAILED: {n_upper_right} != {n_both} ***")
    assert cross_check_ok, (
        f"union-set upper-right-quadrant count ({n_upper_right}) != established "
        f"conjunction n ({n_both}) -- STOPPING, do not ship an inconsistent plot")

    fig3, ax3 = plt.subplots(figsize=(7.5, 6.5))
    for cat, color, label in [("cf_only", COLOR_CF_ONLY, f"CF only (n={n_cf_only})"),
                               ("am_only", COLOR_AM_ONLY, f"AM only (n={n_am_only})"),
                               ("both", COLOR_BOTH, f"Both / conjunction (n={n_both})")]:
        sel = cat_union == cat
        ax3.scatter(cf_F_union[sel], am_F_union[sel], c=color, s=32, alpha=0.7,
                    edgecolor="k", linewidth=0.3, label=label)
    ax3.axvline(Fcrit_cf, color="k", linestyle="--", linewidth=1.2)
    ax3.axhline(Fcrit_am, color="k", linestyle="--", linewidth=1.2)
    ax3.text(Fcrit_cf * 1.02, ax3.get_ylim()[1] if False else am_F_union.max() * 0.98,
             f"CF Fcrit(p<0.01,q={q_cf},dof={dof_cf})={Fcrit_cf:.3f}",
             rotation=90, va="top", ha="left", fontsize=8.5)
    ax3.text(cf_F_union.max() * 0.55, Fcrit_am * 1.03,
             f"AM Fcrit(p<0.01,q={q_am},dof={dof_am})={Fcrit_am:.3f}",
             va="bottom", ha="left", fontsize=8.5)
    ax3.annotate(f"conjunction zone\n(CF_F>{Fcrit_cf:.2f} AND AM_F>{Fcrit_am:.2f})\n"
                 f"n={n_upper_right} (== established conjunction n={n_both}: "
                 f"{'OK' if cross_check_ok else 'MISMATCH'})",
                 xy=(0.72, 0.90), xycoords="axes fraction", fontsize=9.5, ha="center",
                 va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.9, ec=COLOR_BOTH))
    ax3.set_xlabel(f"CF omnibus F-value (q={q_cf}, dof={dof_cf})")
    ax3.set_ylabel(f"AM omnibus F-value (q={q_am}, dof={dof_am})")
    ax3.set_title("sub-01 thalamus: CF-F vs AM-F, UNION of p<0.01-active voxels\n"
                  "(not just the conjunction subset)")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(True, alpha=0.25)
    fig3.tight_layout()
    out_c = f"{SG}/sub-01_CFAM_scatter_Fvalue_SMOOTHED.png"
    fig3.savefig(out_c, dpi=150)
    plt.close(fig3)
    log(f"wrote {out_c}")

    log("DONE")
    return dict(n_cf_only=n_cf_only, n_am_only=n_am_only, n_both=n_both,
                cross_check_upper_right=n_upper_right, cross_check_ok=cross_check_ok,
                outputs=[out_a, out_b, out_c])


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))
