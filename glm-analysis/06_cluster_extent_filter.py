"""Apply spatial cluster-extent filtering on top of the already-computed
FDR-significant voxel tables (CF_FDR_betas.txt, AM_FDR_betas.txt).

Does not touch or overwrite those files, or any other existing deliverable —
reads them, and writes new *_FDR_clusterExtent_betas.txt files alongside them.

A voxel only counts as "meaningful" if it (a) survives FDR correction
(already applied) AND (b) belongs to a spatially contiguous cluster of at
least MIN_CLUSTER_VOXELS significant voxels (26-connectivity). This mirrors
BrainVoyager's cluster-size threshold tool, implemented here in Python since
no BrainVoyager GUI is available in this environment.

MIN_CLUSTER_VOXELS=10 is a conventional heuristic minimum, not derived from
a Monte-Carlo/AlphaSim cluster simulation.
"""
import os
import numpy as np
from scipy import ndimage

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ROOT = PROJECT_ROOT
IN_DIR = f"{ROOT}/derivatives/sub-01/analysis/beta_tables"
MIN_CLUSTER_VOXELS = 10

# bounding box grid (from VTC/VMR header, resolution=1)
XSTART, YSTART, ZSTART = 67, 76, 11
DIMX, DIMY, DIMZ = 195, 77, 207


def load_table(path):
    with open(path) as f:
        lines = f.readlines()
    title = lines[0].rstrip("\n")
    header_idx = next(i for i, l in enumerate(lines) if l.startswith("VMR_X"))
    cols = lines[header_idx].rstrip("\n").split("\t")
    rows = []
    for l in lines[header_idx + 1:]:
        l = l.rstrip("\n")
        if not l or l.startswith("["):
            continue
        rows.append(l.split("\t"))
    return title, cols, rows


def run(label):
    in_path = f"{IN_DIR}/{label}_FDR_betas.txt"
    title, cols, rows = load_table(in_path)
    print(f"{label}: loaded {len(rows)} FDR-significant voxels from {in_path}")

    vol = np.zeros((DIMZ, DIMY, DIMX), dtype=bool)
    coord_to_row = {}
    for r in rows:
        x, y, z = int(r[0]), int(r[1]), int(r[2])
        iz, iy, ix = z - ZSTART, y - YSTART, x - XSTART
        vol[iz, iy, ix] = True
        coord_to_row[(iz, iy, ix)] = r

    structure = np.ones((3, 3, 3), dtype=int)  # 26-connectivity
    labels, n_clusters = ndimage.label(vol, structure=structure)
    sizes = ndimage.sum(vol, labels, index=np.arange(1, n_clusters + 1))
    keep_labels = set(np.where(sizes >= MIN_CLUSTER_VOXELS)[0] + 1)
    print(f"  {n_clusters} contiguous clusters found; "
          f"{len(keep_labels)} have >= {MIN_CLUSTER_VOXELS} voxels")

    kept_rows = []
    for (iz, iy, ix), r in coord_to_row.items():
        if labels[iz, iy, ix] in keep_labels:
            f_val = float(r[3])
            kept_rows.append((f_val, r))
    kept_rows.sort(key=lambda t: -t[0])
    print(f"  {len(kept_rows)} voxels survive FDR + cluster-extent (of {len(rows)} FDR-only)")

    out_path = f"{IN_DIR}/{label}_FDR_clusterExtent_betas.txt"
    new_title = (f"{label} — per-voxel betas, FDR q<0.05 AND spatial cluster-extent "
                 f">= {MIN_CLUSTER_VOXELS} contiguous voxels (26-connectivity)")
    with open(out_path, "w") as f:
        f.write(new_title + "\n")
        f.write("=" * len(new_title) + "\n\n")
        f.write("\t".join(cols) + "\n")
        for _, r in kept_rows:
            f.write("\t".join(r) + "\n")
        f.write(f"\n[{len(kept_rows)} voxel rows across {len(keep_labels)} clusters "
                 f"(MIN_CLUSTER_VOXELS={MIN_CLUSTER_VOXELS}, heuristic not "
                 f"simulation-derived); {len(rows)} voxels were FDR-significant "
                 f"before cluster-extent filtering; source file unmodified: "
                 f"{os.path.basename(in_path)}]\n")
    print(f"  wrote {out_path}")
    return out_path


if __name__ == "__main__":
    p1 = run("CF")
    p2 = run("AM")
    print("DONE")
    print(p1)
    print(p2)
