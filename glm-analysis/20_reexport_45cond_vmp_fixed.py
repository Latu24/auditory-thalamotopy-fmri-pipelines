"""Re-export the 45-condition smoothed ridge VMP from the existing checkpoint
(_ckpt_smoothed_45cond_ridge_fit.npz) -- no VTC re-read, no re-fit -- fixing
two things flagged after visual inspection in BrainVoyager:

1. Bbox Z-offset: the anatomical VMR centers its 240-slice short axis
   (Z=left-right) inside BrainVoyager's 320^3 framing cube via OffsetZ=40;
   our bbox was computed against the old, uncentered convention. +40 fixes
   it, same as 17/19_export_*.py.
2. Color-scale saturation: beta_absmax was the 99th percentile of |beta|
   (~207), but this design's known collinearity instability gives it a
   heavy tail (max~1283) -- 1% of voxels were clamped to the same extreme
   color. Using the 99.9th percentile (~376) instead of 99th cuts the
   saturated fraction 10x while still not letting a handful of pathological
   voxels wash out the rest of the scale.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import glmlib as G
import common2 as C2

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ANA = f"{PROJECT_ROOT}/derivatives/sub-01/analysis"
WORK = f"{ANA}/thalamus_work"
OUT_DIR = f"{ANA}/smoothed_glm"
CKPT = f"{WORK}/_ckpt_smoothed_45cond_ridge_fit.npz"
ANAT_V2_Z_OFFSET = 40


def main():
    ck = np.load(CKPT)
    betas = ck["betas"]
    mask = ck["mask"]
    n_cond = int(ck["n_cond"])
    bbox = {"XStart": int(ck["bbox_XStart"]), "XEnd": int(ck["bbox_XEnd"]),
            "YStart": int(ck["bbox_YStart"]), "YEnd": int(ck["bbox_YEnd"]),
            "ZStart": int(ck["bbox_ZStart"]) + ANAT_V2_Z_OFFSET,
            "ZEnd": int(ck["bbox_ZEnd"]) + ANAT_V2_Z_OFFSET}
    DimZ, DimY, DimX = betas.shape[1:]
    print(f"Loaded checkpoint: n_cond={n_cond}, shape={betas.shape[1:]}, "
          f"bbox (Z-corrected)={bbox}")

    # checkpointed mask (p40) excludes 63% of the anatomical thalamus (dimmer
    # than surrounding tissue, not outside scan coverage) -- recompute at p10.
    h0 = G.read_vtc_header(C2.VTCS_REFINED[0])
    mm0 = G.vtc_memmap(C2.VTCS_REFINED[0], h0)
    tmean = np.asarray(mm0[:, :, :, ::8].mean(-1), dtype=np.float32)
    pos = tmean[tmean > 0]
    p10 = float(np.percentile(pos, 10))
    mask = tmean > p10
    print(f"recomputed mask at p10={p10:.1f}: {int(mask.sum())} voxels "
          f"(was {int(ck['mask'].sum())} at p40 in checkpoint)")

    cf_order = C2.FREQ_ORDER
    am_order = C2.AM_ORDER
    combined_order = cf_order + am_order
    assert len(combined_order) == n_cond

    betas_cond = betas[:n_cond]
    n_cf = len(cf_order)  # 36

    # SEPARATE color ceilings for CF vs AM -- a single pooled ceiling across
    # all 45 conditions was the bug: CF's raw betas run ~2x larger than AM's
    # (finer 36-way partition of the same underlying events -> ~6 events/
    # condition, noisier estimates, vs AM's coarser 9-way partition -> 16-35
    # events/condition). A ceiling sized for the pooled distribution left CF
    # pinned near/at the saturated extreme while AM (comfortably below it)
    # looked fine -- exactly the reported symptom (CF all-extreme, AM
    # "perfect").
    cf_bv = betas_cond[:n_cf][:, mask]
    am_bv = betas_cond[n_cf:][:, mask]
    cf_absmax = float(np.percentile(np.abs(cf_bv), 99.9))
    am_absmax = float(np.percentile(np.abs(am_bv), 99.9))
    print(f"CF |beta| p99.9={cf_absmax:.1f} (max={np.abs(cf_bv).max():.1f})  "
          f"AM |beta| p99.9={am_absmax:.1f} (max={np.abs(am_bv).max():.1f})  "
          f"-- separate ceilings now, was one pooled ceiling")

    mean_beta = betas_cond.mean(axis=0)
    beta_range = betas_cond.max(axis=0) - betas_cond.min(axis=0)
    best_idx = np.argmax(betas_cond, axis=0).astype(np.float32) + 1.0
    range_thr = float(np.percentile(beta_range[mask], 75))
    desc_sig = (beta_range > range_thr) & mask
    best_map = np.where(desc_sig, best_idx, 0.0).astype(np.float32)
    beta_absmax = max(cf_absmax, am_absmax)  # for the pooled descriptive maps below

    maps = []
    for i, name in enumerate(combined_order):
        this_upper = cf_absmax if i < n_cf else am_absmax
        maps.append({"name": f"beta_{name}_SMOOTHED", "data": betas_cond[i],
                    "type": 1, "df1": 0, "threshold": 0.0, "upper": this_upper,
                    "showposneg": 3})
    maps.append({"name": "combinedCFAM_SMOOTHED mean-beta (descriptive)",
                "data": mean_beta, "type": 1, "df1": 0, "threshold": 0.0,
                "upper": beta_absmax, "showposneg": 3})
    maps.append({"name": "combinedCFAM_SMOOTHED beta-range (descriptive)",
                "data": beta_range, "type": 1, "df1": 0, "threshold": 0.0,
                "upper": float(np.percentile(beta_range[mask], 99.9))})
    maps.append({"name": "combinedCFAM_SMOOTHED best-cond idx (DESCRIPTIVE, top-quartile beta-range)",
                "data": best_map, "type": 1, "df1": 0, "threshold": 0.5,
                "upper": float(n_cond), "showposneg": 1})

    hdr_for_vmp = {"XStart": bbox["XStart"], "XEnd": bbox["XEnd"],
                  "YStart": bbox["YStart"], "YEnd": bbox["YEnd"],
                  "ZStart": bbox["ZStart"], "ZEnd": bbox["ZEnd"],
                  "VTC resolution": 1, "DimX": DimX, "DimY": DimY, "DimZ": DimZ}
    vmp_path = f"{OUT_DIR}/sub-01_combinedCFAM_45cond_SMOOTHED_ridge.vmp"
    G.write_stat_vmp(vmp_path, maps, hdr_for_vmp,
                     vtc_name="sub-01_run-*_preproc_coreg-refined.vtc (smoothed in-memory)",
                     prt_name="combined_CF+AM (45cond, sanity test)")
    print(f"wrote {vmp_path} ({len(maps)} sub-maps)")


if __name__ == "__main__":
    main()
