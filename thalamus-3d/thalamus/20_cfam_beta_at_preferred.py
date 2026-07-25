"""Thalamus pipeline -- add beta-amplitude-at-preferred-condition columns to
the per-modality/conjunction voxel tables from 18_cfam_conjunction_smoothed.py
(18's graphics outputs from 19_cfam_conjunction_graphics.py are not touched
by this script -- this is a pure tabular-column addition).

For every already-active voxel (in either per-modality table), adds
`beta_at_preferred`: the winsorized (median +/- 5*MAD, same per-voxel
convention already used before the Gaussian fit in 18) beta value at that
voxel's own winning/preferred condition index -- i.e.
`profile_winsorized[condition_index - 1]`. This is "how much beta the voxel
produces when stimulated at its own preferred frequency/rate," recomputed
fresh from the (read-only) .glm files rather than reused from any stale
intermediate, since 18 did not persist the per-voxel winsorized profiles to
disk (only the Gaussian-fit outputs derived from them).

Updates in place:
  1. sub-01_CF_voxeltable_SMOOTHED.json / sub-01_AM_voxeltable_SMOOTHED.json
     -- add `beta_at_preferred` to every row.
  2. sub-01_CFAM_conjunction_SMOOTHED.json / .csv -- add
     `CF_beta_at_preferred` / `AM_beta_at_preferred` to every row.
  3. sub-01_CFAM_conjunction_SIMPLE.csv (a simplified summary table) --
     append the same two columns, joined by (voxel_i,voxel_j,voxel_k), in the
     file's existing row order (not re-sorted).
  4. sub-01_CFAM_conjunction_SMOOTHED_methodology.json -- documents what
     beta_at_preferred means and the round-trip verification below.

New plumbing: the inverse of 18's native<-raw forward transform (`to_native`),
needed to go from a voxel's already-known native (i,j,k) back to its GLM's
own (z,x,y) beta-array index. Implemented and explicitly round-trip-verified
below (verify_native_to_raw_roundtrip()) before being trusted for any real
voxel.
"""
import os, csv, json, time, importlib.util
import numpy as np

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

ROOT = PROJECT_ROOT
SG = f"{ROOT}/derivatives/sub-01/analysis/smoothed_glm"

# ---- import (do not duplicate) script 18's validated GLM/.ctr/native-
#      transform code -- same pattern as script 19 ----
_spec = importlib.util.spec_from_file_location(
    "cfam18", f"{ROOT}/scripts/thalamus/18_cfam_conjunction_smoothed.py")
_bf18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bf18)
MODALITIES = _bf18.MODALITIES


def build_raw_index_native(shapeZXY, bbox):
    """The INVERSE of to_native(): a native-space (240,320,320) int64 volume
    where each voxel holds the flat (Z,X,Y)-space index of the GLM-space
    ('raw') voxel it came from, or -1 if that native voxel isn't covered by
    this GLM's bbox at all. Built by forward-transforming an index grid
    through 18's own to_native() -- i.e. the inverse falls out of the SAME
    validated forward transform, not a separately re-derived formula."""
    idx_zxy = np.arange(int(np.prod(shapeZXY)), dtype=np.int64).reshape(shapeZXY)
    return _bf18.to_native(idx_zxy, bbox, fill_value=-1)


def verify_native_to_raw_roundtrip(shapeZXY, bbox, tag, n_test=8, seed=0):
    """Explicit round-trip check: pick random known raw (z,x,y) GLM-space
    coordinates, forward-transform each through to_native() (a single-marker
    volume), confirm it lands at exactly one native voxel, then invert that
    native voxel via build_raw_index_native() and confirm we recover the
    exact original raw coordinate. Every random raw coordinate is guaranteed
    (by construction of insert_framebox/to_native) to land somewhere in the
    native frame, so this should pass 100% if the transform is self-consistent."""
    rng = np.random.default_rng(seed)
    Z, X, Y = shapeZXY
    idx_native = build_raw_index_native(shapeZXY, bbox)
    results = []
    for _ in range(n_test):
        z0, x0, y0 = int(rng.integers(0, Z)), int(rng.integers(0, X)), int(rng.integers(0, Y))
        marker = np.zeros(shapeZXY, dtype=np.uint8)
        marker[z0, x0, y0] = 1
        native_marker = _bf18.to_native(marker, bbox, fill_value=0) > 0
        nz = np.argwhere(native_marker)
        one_hit = len(nz) == 1
        if one_hit:
            i, j, k = (int(v) for v in nz[0])
            flat_back = int(idx_native[i, j, k])
            z1, x1, y1 = (int(v) for v in np.unravel_index(flat_back, shapeZXY))
            recovered_ok = (z1, x1, y1) == (z0, x0, y0)
        else:
            i = j = k = None
            recovered_ok = False
        results.append(dict(raw_zxy=[z0, x0, y0], native_ijk=[i, j, k],
                            single_native_hit=one_hit, recovered_raw_matches=recovered_ok))
    all_ok = all(r["recovered_raw_matches"] for r in results)
    log(f"[{tag}] round-trip check ({n_test} random raw coords): "
        f"{sum(r['recovered_raw_matches'] for r in results)}/{n_test} recovered exactly "
        f"-- {'PASS' if all_ok else 'FAIL'}")
    return all_ok, results, idx_native


def winsorize_profiles(profiles):
    """Identical convention to 18_cfam_conjunction_smoothed.py: per-voxel
    median +/- 5*MAD across that voxel's own q-condition profile."""
    med = np.median(profiles, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(profiles - med), axis=1, keepdims=True)
    lo = med - 5.0 * mad
    hi = med + 5.0 * mad
    return np.clip(profiles, lo, hi)


def compute_beta_at_preferred(tag, cfg, table_rows):
    """Re-read this modality's GLM, re-verify the .ctr-derived condition
    columns, build (and round-trip-verify) the native<-raw inverse index, then
    for every row in the already-shipped voxel table: pull its GLM-space
    condition-beta profile, winsorize it (same convention as the Gaussian
    fit), and read off the value at that row's own condition_index-1. Returns
    a dict {(i,j,k): beta_at_preferred}."""
    h, R2, SS, beta, SSXiY, meantc, ARlag = _bf18.glmmod.read_glm(cfg["glm"])
    del R2, SS, SSXiY, ARlag, meantc
    shapeZXY = beta.shape[:3]
    bbox = {k: int(h[k]) for k in ("XStart", "XEnd", "YStart", "YEnd", "ZStart", "ZEnd")}
    names = [p["Name (custom)"] for p in h["Predictor info"]]
    ctr = _bf18.parse_ctr(cfg["ctr"])
    cols, named = _bf18.verify_ctr_matches_predictors(ctr, names)
    assert len(cols) == cfg["q"]
    log(f"[{tag}] re-verified .ctr columns {cols[0]}..{cols[-1]} (q={len(cols)})")

    ok, _, idx_native = verify_native_to_raw_roundtrip(shapeZXY, bbox, tag)
    assert ok, f"[{tag}] native<-raw round-trip check FAILED -- stopping, do not trust beta lookups"

    ijk_list = [tuple(r["voxel_ijk_native"]) for r in table_rows]
    profiles = np.zeros((len(ijk_list), len(cols)), dtype=np.float64)
    for row_i, (i, j, k) in enumerate(ijk_list):
        flat = idx_native[i, j, k]
        assert flat >= 0, f"[{tag}] table voxel ({i},{j},{k}) has no GLM source index (unexpected)"
        z, x, y = np.unravel_index(flat, shapeZXY)
        profiles[row_i] = beta[z, x, y, cols]
    profiles_w = winsorize_profiles(profiles) if len(profiles) else profiles

    out = {}
    for row_i, r in enumerate(table_rows):
        ci = r["condition_index"]
        out[ijk_list[row_i]] = float(profiles_w[row_i, ci - 1])
    return out


def main():
    log("loading existing (already-shipped) per-modality + conjunction tables ...")
    cf_path = f"{SG}/sub-01_CF_voxeltable_SMOOTHED.json"
    am_path = f"{SG}/sub-01_AM_voxeltable_SMOOTHED.json"
    conj_json_path = f"{SG}/sub-01_CFAM_conjunction_SMOOTHED.json"
    conj_csv_path = f"{SG}/sub-01_CFAM_conjunction_SMOOTHED.csv"
    simple_csv_path = f"{SG}/sub-01_CFAM_conjunction_SIMPLE.csv"
    methodology_path = f"{SG}/sub-01_CFAM_conjunction_SMOOTHED_methodology.json"

    cf_table = json.load(open(cf_path))
    am_table = json.load(open(am_path))
    conj = json.load(open(conj_json_path))

    cf_beta_by_ijk = compute_beta_at_preferred("CF", MODALITIES["CF"], cf_table["rows"])
    am_beta_by_ijk = compute_beta_at_preferred("AM", MODALITIES["AM"], am_table["rows"])

    # ---- (1) per-modality tables: add beta_at_preferred to every row ----
    for r in cf_table["rows"]:
        r["beta_at_preferred"] = round(cf_beta_by_ijk[tuple(r["voxel_ijk_native"])], 4)
    for r in am_table["rows"]:
        r["beta_at_preferred"] = round(am_beta_by_ijk[tuple(r["voxel_ijk_native"])], 4)
    json.dump(cf_table, open(cf_path, "w"), indent=2)
    json.dump(am_table, open(am_path, "w"), indent=2)
    log(f"updated {cf_path} and {am_path} with beta_at_preferred (n={len(cf_table['rows'])}, "
        f"{len(am_table['rows'])})")

    # ---- (2) conjunction JSON + CSV: add CF_/AM_beta_at_preferred ----
    for r in conj["rows"]:
        ijk = tuple(r["voxel_ijk_native"])
        r["CF_beta_at_preferred"] = round(cf_beta_by_ijk[ijk], 4)
        r["AM_beta_at_preferred"] = round(am_beta_by_ijk[ijk], 4)
    json.dump(conj, open(conj_json_path, "w"), indent=2)
    log(f"updated {conj_json_path} (n={conj['n']})")

    fields = ["voxel_ijk_native_i", "voxel_ijk_native_j", "voxel_ijk_native_k",
              "world_x_mm", "world_y_mm", "world_z_mm", "hemisphere", "in_MGB",
              "CF_condition_index", "CF_preferred_freq_hz_exact",
              "CF_best_fit_freq_hz_continuous", "CF_F_value", "CF_fit_converged", "CF_fit_r2",
              "AM_condition_index", "AM_preferred_freq_hz_exact",
              "AM_best_fit_freq_hz_continuous", "AM_F_value", "AM_fit_converged", "AM_fit_r2",
              "CF_beta_at_preferred", "AM_beta_at_preferred"]
    with open(conj_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in conj["rows"]:
            w.writerow([r["voxel_ijk_native"][0], r["voxel_ijk_native"][1], r["voxel_ijk_native"][2],
                        r["world_xyz_mm"][0], r["world_xyz_mm"][1], r["world_xyz_mm"][2],
                        r["hemisphere"], r["in_MGB"],
                        r["CF_condition_index"], r["CF_preferred_freq_hz_exact"],
                        r["CF_best_fit_freq_hz_continuous"], r["CF_F_value"],
                        r["CF_fit_converged"], r["CF_fit_r2"],
                        r["AM_condition_index"], r["AM_preferred_freq_hz_exact"],
                        r["AM_best_fit_freq_hz_continuous"], r["AM_F_value"],
                        r["AM_fit_converged"], r["AM_fit_r2"],
                        r["CF_beta_at_preferred"], r["AM_beta_at_preferred"]])
    log(f"updated {conj_csv_path}")

    # ---- (3) simplified CSV: append the two columns, SAME row order,
    #      joined explicitly by (voxel_i,voxel_j,voxel_k) rather than assumed
    #      identical ordering ----
    with open(simple_csv_path) as f:
        simple_rows = list(csv.DictReader(f))
    conj_by_ijk = {tuple(r["voxel_ijk_native"]): r for r in conj["rows"]}
    n_matched = 0
    for sr in simple_rows:
        ijk = (int(sr["voxel_i"]), int(sr["voxel_j"]), int(sr["voxel_k"]))
        cr = conj_by_ijk.get(ijk)
        assert cr is not None, f"SIMPLE.csv voxel {ijk} not found in conjunction table -- join failed"
        sr["CF_beta_at_preferred"] = cr["CF_beta_at_preferred"]
        sr["AM_beta_at_preferred"] = cr["AM_beta_at_preferred"]
        n_matched += 1
    simple_fields = ["voxel_i", "voxel_j", "voxel_k", "CF_preferred_freq_hz", "AM_preferred_freq_hz",
                      "CF_beta_at_preferred", "AM_beta_at_preferred"]
    with open(simple_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=simple_fields)
        w.writeheader()
        w.writerows(simple_rows)
    log(f"updated {simple_csv_path}  ({n_matched}/{len(simple_rows)} rows joined by voxel ijk)")
    assert n_matched == len(simple_rows) == conj["n"]

    # ---- (4) methodology sidecar: document the addition ----
    methodology = json.load(open(methodology_path))
    methodology["beta_at_preferred_addition"] = {
        "added_by": "scripts/thalamus/20_cfam_beta_at_preferred.py, added to "
                 "sub-01_CF_voxeltable_SMOOTHED.json, sub-01_AM_voxeltable_SMOOTHED.json, "
                 "sub-01_CFAM_conjunction_SMOOTHED.json/.csv, and "
                 "sub-01_CFAM_conjunction_SIMPLE.csv. Does not affect the F-values, "
                 "Fcrit thresholds, active-voxel sets, Gaussian-fit continuous frequencies, "
                 "or any of the graphics (conjunction scatter/map/bar/F-value-scatter PNGs) "
                 "-- purely an added descriptive column.",
        "definition": "beta_at_preferred (per-modality field) / CF_beta_at_preferred, "
                      "AM_beta_at_preferred (conjunction table) = the voxel's own "
                      "WINSORIZED condition-beta profile (median +/- 5*MAD across that "
                      "voxel's own q-condition profile -- IDENTICAL convention to the "
                      "winsorization already applied before the Gaussian fit), evaluated "
                      "at that voxel's own winning/preferred condition index "
                      "(condition_index - 1, 0-indexed). This is NOT the raw/unwinsorized "
                      "beta, and NOT the Gaussian-fit's own model-implied amplitude "
                      "parameter (fit parameter 'a' in gauss(x,a,mu,s,b)) -- it is the "
                      "actual (winsorized) observed beta at the discrete winning condition, "
                      "i.e. 'how much beta this voxel produces when stimulated at its own "
                      "preferred CF frequency / AM rate.'",
        "native_to_raw_inverse_transform": {
            "method": "The forward transform native=to_native(raw) (18's own validated "
                "insert_framebox+documented_fb_to_native) was inverted by forward-"
                "transforming a GLM-space linear-index volume through the SAME to_native() "
                "(build_raw_index_native()) -- i.e. the inverse falls out of the identical, "
                "already-validated forward transform rather than a separately re-derived "
                "closed-form formula.",
            "roundtrip_verification": "8 random raw (z,x,y) GLM-space coordinates were "
                "forward-transformed (single-voxel marker -> to_native), confirmed to "
                "land at exactly one native voxel each, then that native voxel was "
                "inverted back via the index volume and the recovered raw coordinate was "
                "asserted to exactly equal the original for all 8/8 coordinates, both "
                "modalities, before any real voxel lookup was trusted.",
        },
    }
    json.dump(methodology, open(methodology_path, "w"), indent=2)
    log(f"updated {methodology_path}")

    # ---- example rows for a sanity eyeball ----
    examples = sorted(conj["rows"], key=lambda r: -(r["CF_F_value"] + r["AM_F_value"]))[:3]
    log("DONE")
    return examples


if __name__ == "__main__":
    ex = main()
    print(json.dumps(ex, indent=2))
