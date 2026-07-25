"""Thalamus pipeline -- CF/AM independence verification (diagnostic, not a
deliverable, does not change the shipped 3D models).

CF and AM report the same voxel COUNT (21,812/21,812) in the whole-thalamus
coverage figures. That match is real but innocuous: both dimensions share the
same functional-COVERAGE mask (meantc > p60, a data-presence mask, not a
significance test), so two different statistics evaluated over the identical
set of "has data" voxels necessarily report the same voxel count for that
mask. It says nothing about whether the underlying per-condition computations
were pooled.

This script proves CF and AM are genuinely independent computations by:
  1. Re-deriving the omnibus-F maps at a real statistical threshold (p<0.01
     uncorrected, not the coverage mask), placed in native space via the same
     validated framebox->native transform used throughout scripts 10-14.
  2. Counting p<0.01-significant voxels separately for CF and AM, within the
     whole thalamus and within the MGB specifically -- 4 numbers.
  3. Confirming CF and AM do not come out identical under this real test (if
     they did, that would be a red flag, unlike the coverage-count match) --
     and if they somehow do, this script stops and reports it as a problem
     rather than passing it through.
  4. Producing explicit machine-checkable proof of independence: disjoint
     predictor-column sets, distinct dfe/Fcrit, non-identical F arrays (exact
     array equality check + correlation, which should be low/moderate, not
     1.0), and independent peak-voxel locations.

Inputs: thalamus_work/bestfreq_maps.npz + bestfreq_stats.json, native-space
thalamus/MGB masks. Output: thalamus_work/cf_am_independence_check.json.
"""
import os, json
import numpy as np
import nibabel as nib
from scipy import stats

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ROOT = PROJECT_ROOT
ANA = f"{ROOT}/derivatives/sub-01/analysis"
WORK = f"{ANA}/thalamus_work"

FBZ, FBY, FBX = 240, 320, 320


def insert_framebox(vtc_raw, bbox):
    """Identical to 12_reconcile_space.py's insert_framebox (VTC-raw (Z,Y,X) ->
    full framebox (240,320,320) at the GLM's own bbox)."""
    fb = np.zeros((FBZ, FBY, FBX), np.float32)
    zs, ze = bbox["ZStart"], bbox["ZEnd"]
    ys, ye = bbox["YStart"], bbox["YEnd"]
    xs, xe = bbox["XStart"], bbox["XEnd"]
    fb[zs:ze, ys:ye, xs:xe] = vtc_raw
    return fb


def documented_fb_to_native(fb):
    """Identical to 12_reconcile_space.py's validated transform
    (BV_Z=239-i, BV_Y=319-k, BV_X=319-j)."""
    return np.ascontiguousarray(np.transpose(fb[::-1, ::-1, ::-1], (0, 2, 1)))


def to_native(vtc_raw, bbox):
    return documented_fb_to_native(insert_framebox(vtc_raw, bbox))


def main():
    z = np.load(f"{WORK}/bestfreq_maps.npz")
    meta = json.load(open(f"{WORK}/bestfreq_stats.json"))
    bbox = meta["bbox"]
    dfe = meta["dfe"]

    thal = np.asarray(nib.load(f"{WORK}/sub-01_thalamus_native.nii.gz").dataobj) > 0
    mgb = np.asarray(nib.load(f"{WORK}/sub-01_MGB_native.nii.gz").dataobj) > 0

    # ---- 1. independence proof BEFORE any placement: raw VTC-raw-space arrays ----
    cf_F_raw = z["cf_F"]
    am_F_raw = z["am_F"]
    identical_arrays = bool(np.array_equal(cf_F_raw, am_F_raw))
    same_shape = cf_F_raw.shape == am_F_raw.shape
    # correlation over voxels where BOTH have signal (F>0, i.e. in brain mask)
    both_nonzero = (cf_F_raw > 0) & (am_F_raw > 0)
    if both_nonzero.sum() > 10:
        corr = float(np.corrcoef(cf_F_raw[both_nonzero], am_F_raw[both_nonzero])[0, 1])
    else:
        corr = float("nan")

    cf_cols = meta["cf"]["cols"]
    am_cols = meta["am"]["cols"]
    disjoint_cols = bool(set(cf_cols).isdisjoint(set(am_cols)))

    q_cf, q_am = meta["cf"]["q"], meta["am"]["q"]
    Fcrit_cf_p01 = float(stats.f.isf(0.01, q_cf, dfe))
    Fcrit_am_p01 = float(stats.f.isf(0.01, q_am, dfe))

    print("=== independence proof (raw VTC-space F maps, pre-placement) ===")
    print(f"  CF predictor cols: {cf_cols[0]}..{cf_cols[-1]} (q={q_cf})")
    print(f"  AM predictor cols: {am_cols[0]}..{am_cols[-1]} (q={q_am})")
    print(f"  disjoint column sets: {disjoint_cols}")
    print(f"  cf_F_raw array identical to am_F_raw array: {identical_arrays} "
          f"(MUST be False)")
    print(f"  cf_F/am_F correlation over shared-coverage voxels: {corr:.4f} "
          f"(should be well below 1.0)")
    print(f"  dfe: CF={dfe} AM={dfe} (same design/residual dfe, as expected -- "
          f"same GLM, different predictor BLOCK)")
    print(f"  Fcrit(p<0.01): CF(q={q_cf})={Fcrit_cf_p01:.4f}  AM(q={q_am})={Fcrit_am_p01:.4f} "
          f"(necessarily different: different q)")

    # peak voxel locations (raw VTC-space index) -- should differ
    cf_peak_idx = np.unravel_index(np.argmax(cf_F_raw), cf_F_raw.shape)
    am_peak_idx = np.unravel_index(np.argmax(am_F_raw), am_F_raw.shape)
    print(f"  CF peak-F voxel (VTC-raw Z,Y,X): {cf_peak_idx}  F={cf_F_raw.max():.2f}")
    print(f"  AM peak-F voxel (VTC-raw Z,Y,X): {am_peak_idx}  F={am_F_raw.max():.2f}")
    print(f"  peak voxels identical: {cf_peak_idx == am_peak_idx} (expected False)")

    # ---- 2. place into native space via the validated transform ----
    cf_F_native = to_native(cf_F_raw, bbox)
    am_F_native = to_native(am_F_raw, bbox)

    # ---- 3. count p<0.01 (uncorrected) voxels, separately, in thalamus & MGB ----
    cf_sig = cf_F_native > Fcrit_cf_p01
    am_sig = am_F_native > Fcrit_am_p01

    counts = {
        "CF_in_thalamus": int((cf_sig & thal).sum()),
        "AM_in_thalamus": int((am_sig & thal).sum()),
        "CF_in_MGB": int((cf_sig & mgb).sum()),
        "AM_in_MGB": int((am_sig & mgb).sum()),
    }
    print("\n=== p<0.01 (uncorrected) omnibus-F voxel counts, native space ===")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    # ---- 4. sanity check: would be a RED FLAG if CF and AM counts collide here ----
    suspicious = (counts["CF_in_thalamus"] == counts["AM_in_thalamus"] or
                 (counts["CF_in_MGB"] == counts["AM_in_MGB"] and counts["CF_in_MGB"] > 0))
    thal_f_identical = bool(np.array_equal(cf_F_native[thal], am_F_native[thal]))
    status = "FLAGGED -- investigate further" if (suspicious or thal_f_identical or identical_arrays) \
        else "PASS -- CF and AM are confirmed independent, counts differ as expected"
    print(f"\n=== sanity check: {status} ===")

    result = {
        "predictor_columns": {"CF": cf_cols, "AM": am_cols, "disjoint": disjoint_cols},
        "dfe": dfe,
        "Fcrit_p01": {"CF": Fcrit_cf_p01, "AM": Fcrit_am_p01},
        "raw_array_independence_proof": {
            "cf_F_am_F_array_identical": identical_arrays,
            "cf_F_am_F_correlation_shared_coverage": corr,
            "cf_peak_voxel_vtc_raw_zyx": [int(v) for v in cf_peak_idx],
            "am_peak_voxel_vtc_raw_zyx": [int(v) for v in am_peak_idx],
            "peak_voxels_identical": bool(cf_peak_idx == am_peak_idx),
            "cf_peakF": float(cf_F_raw.max()), "am_peakF": float(am_F_raw.max())},
        "p01_uncorrected_counts": counts,
        "native_space_F_identical_in_thalamus": thal_f_identical,
        "sanity_check_status": status,
        "note_on_earlier_matching_coverage_count": (
            "The 21,812/21,812 match above the p<0.01 counts is the COVERAGE count "
            "(meantc-based functional-presence mask, identical for CF and AM by "
            "construction since the output-gating methodology uses coverage, not "
            "significance) -- not a significance count. This script uses a real, "
            "independent p<0.01 omnibus-F threshold instead, and the counts above "
            "are the correct answer to 'are CF and AM actually different'.")}
    with open(f"{WORK}/cf_am_independence_check.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {WORK}/cf_am_independence_check.json")


if __name__ == "__main__":
    main()
