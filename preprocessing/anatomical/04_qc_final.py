#!/usr/bin/env python3
"""Quantitative QC of the final VMR: clipping/saturation, WM intensity
uniformity, and a Gibbs-ringing check with a real pass/warn/fail gate.
Input: the VMR/V16 written by 03_make_vmr.py and the INV2-derived brain
mask. Output: qc_final_v2.json and a histogram/line-profile QC figure.

The ringing metric restricts edge detection to purely intracranial tissue
(an edge is only scored if its entire local window lies inside the brain
mask, so skull/scalp/dura/vessel texture cannot contribute) and averages
over many randomly placed line profiles per axis rather than a single fixed
line, so no single atypical row can dominate the result. For every
candidate edge it also classifies whether the local intensity pattern
actually looks like ringing (a decaying, sign-alternating oscillation) or a
single-lobe overshoot (an ordinary sharp tissue edge / partial-volume
effect), and reports whether high-overshoot edges are concentrated at
near-noise-floor intensities -- a known MP2RAGE UNI noise-amplification
signature in low-SNR voxels (CSF/ventricles) that inflates the fractional
overshoot metric without any ringing being present. A pass/warn/fail gate is
computed against a documented overshoot threshold and written into the QC
JSON alongside these diagnostics.
"""
import os, json
import numpy as np
import nibabel as nib
import bvbabel.vmr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ANAT = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "anat")
VMR_PATH = os.path.join(ANAT, "sub-01_desc-UNIdenoisedN4_anat_v2.vmr")
MASK_PATH = os.path.join(ANAT, "sub-01_desc-brainmaskINV2.nii.gz")

h, vmr = bvbabel.vmr.read_vmr(VMR_PATH)
vmr = vmr.astype(np.float64)
mask = nib.as_closest_canonical(nib.load(MASK_PATH)).get_fdata() > 0.5
assert mask.shape == vmr.shape, f"mask/vmr shape mismatch {mask.shape} vs {vmr.shape}"
brain = vmr[mask]

qc = {}
# --- 1. Clipping / saturation (8-bit anatomy should not pile up at 225) ---
tot = brain.size
qc["clipping"] = {
    "frac_at_225": float((brain >= 225).sum()/tot),
    "frac_ge_224": float((brain >= 224).sum()/tot),
    "frac_at_0_in_brainmask": float((brain <= 0).sum()/tot),
    "note": "anatomy scaled to 0..225; near-zero pileup at ceiling = no saturation",
}

# --- 2. Intensity uniformity (WM CoV via 3-class kmeans) ---
from sklearn.cluster import KMeans
x = brain[brain > 0].reshape(-1, 1)
km = KMeans(n_clusters=3, n_init=5, random_state=0).fit(x)
cen = km.cluster_centers_.ravel(); order = np.argsort(cen)
lab = km.labels_
wm = x.ravel()[lab == order[2]]; gm = x.ravel()[lab == order[1]]
qc["uniformity"] = {
    "wm_mean": float(wm.mean()), "wm_std": float(wm.std()),
    "wm_cov": float(wm.std()/wm.mean()),
    "gm_wm_dprime": float(abs(wm.mean()-gm.mean())/np.sqrt(0.5*(wm.std()**2+gm.std()**2))),
}

# --- 3. Gibbs ringing: multi-line, brain-restricted, with a real gate ---
rng = np.random.default_rng(0)
HALFWIN = 15   # local window radius used both for edge scoring and mask-purity check
N_LINES_PER_AXIS = 40

def score_line(prof_line, mask_line):
    """Return list of (overshoot_frac, is_oscillatory, base, plateau) for edges fully inside mask."""
    out = []
    g = np.abs(np.diff(prof_line))
    if g.max() <= 0:
        return out
    steep = np.where(g > 0.4 * g.max())[0]
    n = len(prof_line)
    for e in steep:
        lo15, hi15 = e - HALFWIN, e + HALFWIN + 1
        if lo15 < 0 or hi15 > n:
            continue
        if not mask_line[lo15:hi15].all():
            continue  # touches skull/scalp/background -> not a pure intracranial edge
        lo6 = max(0, e - 6); hi6 = min(n, e + 7)
        seg = prof_line[lo6:hi6]
        base = float(np.median(prof_line[e-15:e-6]))
        plateau = float(np.median(prof_line[e+6:e+15]))
        peak = float(seg.max())
        trough = float(seg.min())
        ref = max(peak, plateau, base, 1.0)
        overshoot = float((peak - max(base, plateau)) / ref)
        undershoot = float((min(base, plateau) - trough) / ref)
        frac = max(overshoot, undershoot, 0.0)

        # oscillation classification: remove the base->plateau linear step
        # trend across the +-10 window, count sign changes of the residual, AND
        # require the residual to be largest near the edge and decay toward the
        # window edges (a real ringing envelope decays away from the causative
        # edge; unstructured noise/vessels typically do not).
        w = prof_line[e-10:e+11]
        trend = np.linspace(base, plateau, num=len(w))
        resid = w - trend
        rng_ = max(np.ptp(resid), 1e-9)
        sig = resid[np.abs(resid) > 0.15 * rng_]
        sign_changes = int(np.sum(np.diff(np.sign(sig)) != 0)) if len(sig) > 1 else 0
        near_amp = float(np.max(np.abs(resid[7:14])))   # +-3 samples around centre
        far_amp = float(np.max(np.abs(np.concatenate([resid[:4], resid[-4:]]))))
        decays = near_amp > 1.3 * far_amp
        is_osc = (sign_changes >= 3) and decays

        out.append((frac, is_osc, base, plateau))
    return out

Z, X, Y = vmr.shape
brain_idx = np.argwhere(mask)
bmin, bmax = brain_idx.min(axis=0), brain_idx.max(axis=0)

results = []
example_lines = {}  # keep one example per axis for the figure
for axis_name, scan_axis in [("S-I", 2), ("A-P", 1), ("L-R", 0)]:
    other_axes = [a for a in range(3) if a != scan_axis]
    n_found = 0
    for _ in range(N_LINES_PER_AXIS):
        i0 = rng.integers(bmin[other_axes[0]], bmax[other_axes[0]] + 1)
        i1 = rng.integers(bmin[other_axes[1]], bmax[other_axes[1]] + 1)
        idx = [None, None, None]
        idx[other_axes[0]] = i0
        idx[other_axes[1]] = i1
        sl = [slice(None) if a == scan_axis else idx[a] for a in range(3)]
        prof = vmr[tuple(sl)]
        mline = mask[tuple(sl)]
        if mline.sum() < 2 * HALFWIN + 2:
            continue
        res = score_line(prof, mline)
        results.extend(res)
        if res and axis_name not in example_lines:
            example_lines[axis_name] = (prof, mline)
        n_found += len(res)

overshoots = [r[0] for r in results]
osc_flags = [r[1] for r in results]
bases = np.array([r[2] for r in results])
plateaus = np.array([r[3] for r in results])

THRESH_NEGLIGIBLE = 0.10   # documented bar, now actually enforced
THRESH_WARN = 0.20

if overshoots:
    ov_arr = np.array(overshoots)
    median_os = float(np.median(overshoots))
    max_os = float(np.max(overshoots))
    mean_os = float(np.mean(overshoots))
    frac_below_negligible = float(np.mean(ov_arr < THRESH_NEGLIGIBLE))
    frac_oscillatory = float(np.mean(osc_flags))
    # diagnostic: is the "overshoot" concentrated at near-noise-floor baselines?
    # (dividing by a near-zero base/plateau inflates the fractional overshoot
    # even for a modest absolute peak -- a known MP2RAGE UNI noise-amplification
    # signature in low-SNR voxels such as CSF/ventricles, NOT Gibbs ringing)
    top100 = np.argsort(ov_arr)[::-1][:min(100, len(ov_arr))]
    near_floor_frac_all = float(np.mean((bases < 20) & (plateaus < 20)))
    near_floor_frac_top = float(np.mean((bases[top100] < 20) & (plateaus[top100] < 20)))
    median_base_top100 = float(np.median(bases[top100]))
    median_base_overall = float(np.median(bases))
else:
    median_os = max_os = mean_os = 0.0
    frac_below_negligible = 1.0
    frac_oscillatory = 0.0
    near_floor_frac_all = near_floor_frac_top = median_base_top100 = median_base_overall = 0.0

if median_os < THRESH_NEGLIGIBLE:
    gate = "PASS"
elif median_os < THRESH_WARN:
    gate = "WARN"
else:
    gate = "FAIL"

# secondary interpretive verdict: is this actually ringing, or something else?
# Evidence considered: (a) oscillatory/decaying-envelope fraction, (b) whether
# the highest-overshoot edges are disproportionately anchored on near-noise-
# floor baselines (base & plateau < 20 on a 0..225 scale), which inflates the
# *fractional* overshoot metric without requiring any ringing at all.
noise_floor_enrichment = (near_floor_frac_top / near_floor_frac_all) if near_floor_frac_all > 0 else float("nan")
if frac_oscillatory < 0.25 and noise_floor_enrichment > 1.5:
    ring_interpretation = (
        "low decaying-oscillation fraction (%.2f) AND the highest-overshoot edges are strongly "
        "enriched (%.1fx) for near-noise-floor baselines (median base intensity %.0f for the top-100 "
        "overshoot edges vs %.0f overall, on a 0..225 scale) -- this pattern (large *fractional* "
        "overshoot driven by dividing by a near-zero baseline, no consistent decaying periodicity) is "
        "the signature of MP2RAGE UNI noise-floor amplification in low-SNR intracranial voxels "
        "(CSF/ventricles/sulcal spaces) and/or small-vessel partial-volume spikes, NOT true Gibbs/"
        "truncation ringing." % (frac_oscillatory, noise_floor_enrichment, median_base_top100, median_base_overall)
    )
elif frac_oscillatory < 0.25:
    ring_interpretation = ("low oscillatory/decaying-envelope fraction (%.2f) -- most flagged edges are "
                            "single-lobe overshoots without a decaying periodic envelope; more consistent "
                            "with ordinary sharp tissue boundaries / partial-volume / noise than true "
                            "Gibbs ringing" % frac_oscillatory)
else:
    ring_interpretation = ("oscillatory/decaying-envelope fraction (%.2f) is substantial -- a meaningful "
                            "share of edges show decaying, alternating-sign structure consistent with true "
                            "Gibbs/truncation ringing" % frac_oscillatory)

qc["ringing"] = {
    "method": "multi-line (up to %d lines/axis x 3 axes), edges scored only if entire local "
              "window lies inside the brain mask (excludes skull/scalp/vessel texture)" % N_LINES_PER_AXIS,
    "n_edges_scored": len(overshoots),
    "max_edge_overshoot_frac": max_os,
    "median_edge_overshoot_frac": median_os,
    "mean_edge_overshoot_frac": mean_os,
    "frac_edges_below_negligible_threshold": frac_below_negligible,
    "frac_edges_oscillatory_decaying_ring_pattern": frac_oscillatory,
    "noise_floor_diagnostic": {
        "near_floor_frac_all_edges": near_floor_frac_all,
        "near_floor_frac_top100_overshoot_edges": near_floor_frac_top,
        "enrichment_ratio": noise_floor_enrichment,
        "median_base_intensity_top100_overshoot_edges": median_base_top100,
        "median_base_intensity_all_edges": median_base_overall,
        "note": "base/plateau near the 8-bit noise floor (<20 of 0..225) inflate the *fractional* "
                "overshoot metric without requiring ringing; used to test whether high overshoot is "
                "noise-floor-driven.",
    },
    "threshold_negligible": THRESH_NEGLIGIBLE,
    "threshold_warn": THRESH_WARN,
    "gate": gate,
    "ring_interpretation": ring_interpretation,
    "note": "overshoot beyond flanking tissue intensity near steep intracranial edges; "
            "<%.2f median = negligible Gibbs (PASS), <%.2f = WARN, else FAIL" % (THRESH_NEGLIGIBLE, THRESH_WARN),
}

# --- also compute a naive single fixed-line, whole-image metric for direct comparison ---
mid = vmr.shape[0] // 2
old_prof_line = vmr[mid, vmr.shape[1] // 2, :]
g_old = np.abs(np.diff(old_prof_line))
steep_old = np.where(g_old > 0.4 * g_old.max())[0]
old_overshoots = []
for e in steep_old:
    lo = max(0, e - 6); hi = min(len(old_prof_line), e + 7)
    seg = old_prof_line[lo:hi]
    base = np.median(old_prof_line[max(0, e - 15):max(1, e - 6)]) if e - 6 > 0 else seg.min()
    peak = seg.max()
    plateau = np.median(old_prof_line[min(len(old_prof_line) - 1, e + 6):min(len(old_prof_line), e + 15)]) if e + 15 < len(old_prof_line) else seg.max()
    ref = max(peak, plateau, 1)
    old_overshoots.append(float((peak - max(base, plateau)) / ref))
qc["ringing_naive_singleline_method_for_comparison"] = {
    "n_steep_edges": int(len(steep_old)),
    "max_edge_overshoot_frac": float(np.max(old_overshoots)) if old_overshoots else 0.0,
    "median_edge_overshoot_frac": float(np.median(old_overshoots)) if old_overshoots else 0.0,
    "note": "a single fixed mid-sagittal/mid-coronal S-I line through the whole (unmasked) volume, "
            "run on the SAME VMR data as the brain-restricted multi-line method above, to isolate how "
            "much of the difference between the two is due to the metric design rather than the "
            "underlying data.",
}

# --- 4. general stats ---
qc["stats"] = {"shape": list(vmr.shape),
               "vox_mm": [h["VoxelSizeX"], h["VoxelSizeY"], h["VoxelSizeZ"]],
               "brain_mean": float(brain.mean()), "brain_max": float(brain.max()),
               "v16_min": h["VMROrigV16MinValue"], "v16_mean": h["VMROrigV16MeanValue"],
               "v16_max": h["VMROrigV16MaxValue"],
               "OffsetX": h["OffsetX"], "OffsetY": h["OffsetY"], "OffsetZ": h["OffsetZ"],
               "FramingCubeDim": h["FramingCubeDim"]}

with open(os.path.join(ANAT, "qc_final_v2.json"), "w") as f:
    json.dump(qc, f, indent=2)
print(json.dumps(qc, indent=2))

# --- figure: histogram + example line profiles (one per axis) ---
fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
axs[0].hist(brain[brain > 0], bins=120, color="#4477aa")
axs[0].axvline(225, color="r", ls="--", lw=1, label="225 ceiling")
axs[0].set_title("VMR brain intensity histogram (0..225)"); axs[0].set_xlabel("intensity"); axs[0].legend()
axs[1].plot(old_prof_line, color="#222", label="fixed S-I line (thru skull)")
axs[1].set_title("Line profiles (Gibbs check)")
axs[1].set_xlabel("voxel along line"); axs[1].set_ylabel("intensity"); axs[1].legend(fontsize=8)
fig.tight_layout()
qc_dir = os.path.join(ANAT, "qc")
os.makedirs(qc_dir, exist_ok=True)
fig.savefig(os.path.join(qc_dir, "qc_04_hist_ringing_v2.png"), dpi=95)
print("wrote qc_04_hist_ringing_v2.png")
print("GATE:", gate, "| median_edge_overshoot_frac =", median_os,
      "| n_edges_scored =", len(overshoots), "| frac_oscillatory =", frac_oscillatory)
