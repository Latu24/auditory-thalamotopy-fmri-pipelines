"""Thalamus pipeline -- MGB appearance characterization (diagnostic, not a
deliverable).

Because every functionally-covered voxel gets a Gaussian-fit best-frequency
value, the MGB's sparsely-covered voxels will show some color even without a
significance gate. This script asks: does that color show spatial structure
(neighboring voxels tend to agree, plausible tuning), or does it look like
noise (neighboring voxels disagree as much as random shuffles, consistent
with no real signal)?

Method: for each functionally-covered MGB voxel with at least one covered MGB
neighbor (6-connectivity), compute the mean |log10(f_i)-log10(f_j)| over
neighbor pairs (the "observed" spatial roughness). Compare to a null built by
shuffling the same values across the same covered-voxel positions (which
preserves the value distribution and adjacency structure, destroying only
spatial structure), 1000+ times. If observed roughness is not meaningfully
lower than the null distribution, the covered MGB voxels' best-freq
assignment is indistinguishable from noise at this scale.

Inputs: native-space MGB mask and native-space best-frequency NIfTIs.
Output: thalamus_work/mgb_appearance_check.json.
"""
import os, json
import numpy as np
import nibabel as nib
from scipy import ndimage

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ROOT = PROJECT_ROOT
ANA = f"{ROOT}/derivatives/sub-01/analysis"
WORK = f"{ANA}/thalamus_work"
ANAT = f"{ROOT}/derivatives/sub-01/anat/sub-01_desc-UNIdenoisedN4.nii.gz"

rng = np.random.default_rng(1)


def neighbor_pairs(mask):
    """6-connected neighbor voxel-index pairs within `mask` (boolean 3D)."""
    idx = np.argwhere(mask)
    pos_to_i = {tuple(v): i for i, v in enumerate(idx)}
    pairs = []
    for i, (z, y, x) in enumerate(idx):
        for dz, dy, dx in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            nb = (z + dz, y + dy, x + dx)
            j = pos_to_i.get(nb)
            if j is not None:
                pairs.append((i, j))
    return idx, np.array(pairs, dtype=int) if pairs else np.zeros((0, 2), int)


def roughness(vals_log, pairs):
    if len(pairs) == 0:
        return np.nan
    return float(np.mean(np.abs(vals_log[pairs[:, 0]] - vals_log[pairs[:, 1]])))


def analyze(dim, hz_lo, hz_hi, fname):
    mgb = np.asarray(nib.load(f"{WORK}/sub-01_MGB_native.nii.gz").dataobj) > 0
    bf = np.asarray(nib.load(f"{ANA}/{fname}").dataobj)
    covered = mgb & (bf > 0)
    n_covered = int(covered.sum())
    n_mgb = int(mgb.sum())
    result = {"dim": dim, "n_mgb_voxels": n_mgb, "n_covered": n_covered,
             "coverage_frac": round(n_covered / max(n_mgb, 1), 4)}
    if n_covered < 3:
        result["note"] = "too few covered voxels for a spatial-coherence test"
        print(f"[{dim}] n_covered={n_covered} -- too few for coherence test")
        return result

    vals = bf[covered]
    result["value_range_hz"] = [float(vals.min()), float(vals.max())]
    result["value_median_hz"] = float(np.median(vals))
    result["value_std_log10hz"] = float(np.std(np.log10(vals)))

    idx, pairs = neighbor_pairs(covered)
    vals_at_idx = bf[idx[:, 0], idx[:, 1], idx[:, 2]]
    vals_log = np.log10(vals_at_idx)
    obs = roughness(vals_log, pairs)
    result["n_neighbor_pairs"] = int(len(pairs))
    result["observed_roughness_log10hz"] = obs

    if len(pairs) == 0:
        result["note"] = "no adjacent covered-voxel pairs (too sparse/isolated) -- " \
                          "cannot distinguish structure from noise; voxels are " \
                          "spatially isolated singletons"
        print(f"[{dim}] n_covered={n_covered}, 0 neighbor pairs -- voxels are isolated")
        return result

    n_perm = 2000
    null = np.empty(n_perm)
    for p in range(n_perm):
        shuffled = rng.permutation(vals_log)
        null[p] = roughness(shuffled, pairs)
    z = (obs - null.mean()) / (null.std() + 1e-12)
    p_lower = float(np.mean(null <= obs))  # fraction of null <= observed
    result["null_roughness_mean"] = float(null.mean())
    result["null_roughness_std"] = float(null.std())
    result["z_score"] = float(z)
    result["p_observed_le_null"] = p_lower
    result["interpretation"] = (
        "STRUCTURE: observed neighbor roughness is significantly LOWER than "
        "shuffled null (neighbors agree more than chance) -- plausible real "
        "spatial tuning" if p_lower < 0.05 else
        "NOISE-LIKE: observed neighbor roughness is NOT distinguishable from "
        "a random shuffle of the same values -- consistent with no real "
        "spatial structure")
    print(f"[{dim}] n_covered={n_covered} pairs={len(pairs)} obs_roughness={obs:.3f} "
          f"null_mean={null.mean():.3f}+-{null.std():.3f} z={z:.2f} p={p_lower:.3f}")
    print(f"  -> {result['interpretation']}")
    return result


def main():
    out = {}
    out["CF"] = analyze("CF", 200, 8000, "sub-01_CF_bestfreq_fromGLM.nii.gz")
    out["AM"] = analyze("AM", 1, 16, "sub-01_AM_bestfreq_fromGLM.nii.gz")
    with open(f"{WORK}/mgb_appearance_check.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote thalamus_work/mgb_appearance_check.json")


if __name__ == "__main__":
    main()
