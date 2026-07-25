"""Thalamus pipeline -- Part C addendum: USDZ export for macOS/iOS Preview /
Quick Look (native rotate/zoom, no plugin needed on the viewing end).

Additional output alongside the HTML models (13_build_3d_models.py) -- does
not replace them. HTML works for any recipient on any OS; USDZ is
macOS/iOS-only but opens directly in Preview/Quick Look with no browser.

Reuses the same mesh geometry pipeline as script 13 (marching cubes,
watertightness check, Taubin lambda/mu smoothing, nearest-voxel color
sampling within SAMPLE_RADIUS_VOX, same Turbo colormap on log10(Hz)) so the
USDZ and HTML representations are visually consistent. The geometry helpers
are duplicated here rather than imported (script 13's filename starts with a
digit and isn't importable as a normal module) and kept in sync with it
deliberately.

Vertex coloring is baked as `primvars:displayColor` (color3f[], interpolation
"vertex") -- the standard way to get per-vertex color in USD without a
UV-mapped texture (Quick Look / AR Quick Look render displayColor directly
when no material graph is bound). World-space coordinates are converted from
mm (native NIfTI world) to meters (USD/AR convention) and the stage up-axis
is set to Z (matches native RAS: +z = Superior).

Inputs: native-space thalamus/MGB masks and native-space best-frequency
NIfTIs. Outputs: sub-01_thalamus3D_CFbestfreq.usdz,
sub-01_thalamus3D_AMbestfreq.usdz, and a usdz_export_stats.json sidecar.
"""
import os, sys, json, subprocess, tempfile
import numpy as np
import nibabel as nib
from skimage import measure
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pxr import Usd, UsdGeom, Vt, Gf, Sdf, UsdUtils

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ROOT = PROJECT_ROOT
ANA = f"{ROOT}/derivatives/sub-01/analysis"
WORK = f"{ANA}/thalamus_work"
ANAT = f"{ROOT}/derivatives/sub-01/anat/sub-01_desc-UNIdenoisedN4.nii.gz"

CF_HZ = 200.0 * (8000.0 / 200.0) ** (np.arange(36) / 35.0)
AM_HZ = 1.0 * (16.0 / 1.0) ** (np.arange(9) / 8.0)

GREY = (0.55, 0.55, 0.55)
SAMPLE_RADIUS_VOX = 1.5
SMOOTH_ITERS = 12
TAUBIN_LAMBDA = 0.5
TAUBIN_MU = -0.53
MM_TO_M = 1.0 / 1000.0   # USD/AR convention is meters; native coords are mm


# ---- identical geometry pipeline to script 13 (kept in sync deliberately) ----
def mask_surface(mask, step=1):
    if mask.sum() == 0:
        return None
    pad = np.pad(mask.astype(np.float32), 1)
    verts, faces, _, _ = measure.marching_cubes(pad, level=0.5, step_size=step)
    verts = verts - 1.0
    return verts, faces


def check_watertight(faces):
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    n_bad = int(np.sum(counts != 2))
    return n_bad, int(len(counts))


def taubin_smooth(verts, faces, iterations=SMOOTH_ITERS, lam=TAUBIN_LAMBDA, mu=TAUBIN_MU):
    n = len(verts)
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    A = coo_matrix((np.ones(len(e), np.float32), (e[:, 0], e[:, 1])),
                   shape=(n, n)).tocsr()
    A.data[:] = 1.0
    A.sum_duplicates()
    A.data[:] = 1.0
    deg = np.clip(np.asarray(A.sum(axis=1)).flatten(), 1, None)
    Dinv = (1.0 / deg)[:, None]
    v = verts.astype(np.float64).copy()
    for _ in range(iterations):
        lap = A.dot(v) * Dinv - v
        v = v + lam * lap
        lap2 = A.dot(v) * Dinv - v
        v = v + mu * lap2
    return v


def crop_to(mask_union):
    idx = np.argwhere(mask_union)
    lo = idx.min(0) - 6
    hi = idx.max(0) + 7
    lo = np.maximum(lo, 0)
    sl = tuple(slice(lo[d], hi[d]) for d in range(3))
    return sl, lo


def sample_vertex_colors_rgb(verts_idx, val_coords, val_values, hz_lo, hz_hi, cmap_name="turbo"):
    """Same logic/scale as script 13's sample_vertex_colors, but returns raw
    (N,3) float RGB in [0,1] (USD primvar format) instead of 'rgb(...)' strings."""
    n = len(verts_idx)
    has_data = np.zeros(n, bool)
    vals = np.full(n, np.nan)
    if len(val_coords):
        tree = cKDTree(val_coords)
        d, idx = tree.query(verts_idx, k=1)
        has_data = d <= SAMPLE_RADIUS_VOX
        vals[has_data] = val_values[idx[has_data]]
    norm = mcolors.Normalize(vmin=np.log10(hz_lo), vmax=np.log10(hz_hi))
    cmap = cm.get_cmap(cmap_name)
    rgb = np.tile(np.array(GREY), (n, 1))
    if has_data.any():
        rgba = cmap(norm(np.log10(vals[has_data])))
        rgb[has_data] = rgba[:, :3]
    return rgb.astype(np.float32), has_data


def add_mesh(stage, path, verts_world_m, faces, rgb_per_vertex, opacity=1.0,
            double_sided=True):
    """Create a UsdGeomMesh at `path` with baked vertex coloring."""
    pts = verts_world_m.astype(np.float64).tolist()
    cols = rgb_per_vertex.astype(np.float64).tolist()

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(p[0], p[1], p[2]) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.astype(int).flatten().tolist()))
    mesh.CreateDoubleSidedAttr(double_sided)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)   # keep our own triangles as-is

    primvars_api = UsdGeom.PrimvarsAPI(mesh)
    color_primvar = primvars_api.CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
    color_primvar.Set(Vt.Vec3fArray([Gf.Vec3f(c[0], c[1], c[2]) for c in cols]))

    if opacity < 1.0:
        op_primvar = primvars_api.CreatePrimvar(
            "displayOpacity", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.constant)
        op_primvar.Set(Vt.FloatArray([float(opacity)]))
    return mesh


def build_usdz(dim, hz, out_usdz):
    aff = nib.load(ANAT).affine
    thal = np.asarray(nib.load(f"{WORK}/sub-01_thalamus_native.nii.gz").dataobj) > 0
    mgb = np.asarray(nib.load(f"{WORK}/sub-01_MGB_native.nii.gz").dataobj) > 0
    bf = np.asarray(nib.load(f"{ANA}/sub-01_{dim}_fromGLM.nii.gz").dataobj)

    union = thal | mgb
    sl, lo = crop_to(union)
    aff_c = aff.copy()
    aff_c[:3, 3] = nib.affines.apply_affine(aff, lo[None, :])[0]

    thal_c = thal[sl]
    mgb_c = mgb[sl]
    bfc = bf[sl] * thal_c

    thal_s = mask_surface(thal_c, step=1)
    mgb_s = mask_surface(mgb_c, step=1)

    val_coords = np.argwhere(bfc > 0).astype(float)
    val_values = bfc[bfc > 0]

    tmpdir = tempfile.mkdtemp(prefix="usdz_build_")
    usdc_path = os.path.join(tmpdir, "scene.usdc")
    stage = Usd.Stage.CreateNew(usdc_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)      # native RAS: +z = Superior
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)            # coordinates below already in meters
    root = UsdGeom.Xform.Define(stage, "/ThalamusModel")
    stage.SetDefaultPrim(root.GetPrim())

    report = {"dim": dim, "watertight": {}}
    n_colored_verts = 0
    if thal_s is not None:
        verts_idx, f = thal_s
        n_bad, n_edges = check_watertight(f)
        report["watertight"]["thalamus"] = {"watertight": n_bad == 0,
                                            "bad_edges": n_bad, "n_edges": n_edges}
        verts_sm = taubin_smooth(verts_idx, f)
        rgb, has_data = sample_vertex_colors_rgb(verts_sm, val_coords, val_values,
                                                 hz[0], hz[-1])
        n_colored_verts = int(has_data.sum())
        vw_mm = nib.affines.apply_affine(aff_c, verts_sm)
        vw_m = vw_mm * MM_TO_M
        add_mesh(stage, "/ThalamusModel/Thalamus", vw_m, f, rgb, opacity=1.0)

    if mgb_s is not None:
        verts_idx, f = mgb_s
        n_bad, n_edges = check_watertight(f)
        report["watertight"]["mgb"] = {"watertight": n_bad == 0,
                                       "bad_edges": n_bad, "n_edges": n_edges}
        verts_sm = taubin_smooth(verts_idx, f)
        vw_mm = nib.affines.apply_affine(aff_c, verts_sm)
        vw_m = vw_mm * MM_TO_M
        mgb_rgb = np.tile(np.array([0.08, 0.08, 0.08], np.float32), (len(verts_sm), 1))
        add_mesh(stage, "/ThalamusModel/MGB_boundary", vw_m, f, mgb_rgb, opacity=0.25)

    stage.GetRootLayer().Save()

    # ---- package as .usdz (zipped USD package, no external texture assets) ----
    if os.path.exists(out_usdz):
        os.remove(out_usdz)
    ok = UsdUtils.CreateNewUsdzPackage(usdc_path, out_usdz)
    n_in_mgb = int(((bf > 0) & mgb).sum())
    n_val_total = int((bfc > 0).sum())
    report.update({"usdz_package_created": bool(ok), "n_thal_voxels_with_value": n_val_total,
                   "n_in_mgb": n_in_mgb, "n_colored_surface_vertices": n_colored_verts,
                   "usdz": out_usdz})
    print(f"[{dim}] wrote {out_usdz}  package_ok={ok}  "
          f"colored_verts={n_colored_verts}  watertight={report['watertight']}")
    return report


def verify_opens_in_preview(path):
    """Best-effort automated verification: (1) usdcat can parse the package
    (structural validity), (2) qlmanage can generate a Quick Look thumbnail
    (the same rendering path Preview/Quick Look uses) -- a non-trivial PNG
    means the file actually renders, not just parses. (3) also opens it in
    Preview.app directly."""
    result = {"path": path}
    try:
        out = subprocess.run(["/usr/bin/usdcat", path], capture_output=True,
                             text=True, timeout=30)
        result["usdcat_ok"] = out.returncode == 0
        result["usdcat_stderr"] = out.stderr[:500] if out.returncode != 0 else ""
    except Exception as e:
        result["usdcat_ok"] = False
        result["usdcat_stderr"] = str(e)

    thumb_dir = tempfile.mkdtemp(prefix="ql_thumb_")
    try:
        out = subprocess.run(["qlmanage", "-t", "-s", "800", "-o", thumb_dir, path],
                             capture_output=True, text=True, timeout=60)
        thumbs = [f for f in os.listdir(thumb_dir) if f.lower().endswith(".png")]
        thumb_size = 0
        if thumbs:
            thumb_size = os.path.getsize(os.path.join(thumb_dir, thumbs[0]))
        result["qlmanage_thumbnail_generated"] = len(thumbs) > 0
        result["qlmanage_thumbnail_bytes"] = thumb_size
        result["qlmanage_rc"] = out.returncode
    except Exception as e:
        result["qlmanage_thumbnail_generated"] = False
        result["qlmanage_error"] = str(e)

    try:
        subprocess.run(["open", "-a", "Preview", path], check=True, timeout=15)
        result["opened_in_preview"] = True
    except Exception as e:
        result["opened_in_preview"] = False
        result["open_error"] = str(e)
    return result


def main():
    res = []
    res.append(build_usdz("CF_bestfreq", CF_HZ, f"{ANA}/sub-01_thalamus3D_CFbestfreq.usdz"))
    res.append(build_usdz("AM_bestfreq", AM_HZ, f"{ANA}/sub-01_thalamus3D_AMbestfreq.usdz"))

    print("\n--- verifying (usdcat parse + qlmanage thumbnail + open in Preview) ---")
    verify = []
    for r in res:
        v = verify_opens_in_preview(r["usdz"])
        verify.append(v)
        print(f"[{r['dim']}] usdcat_ok={v.get('usdcat_ok')} "
              f"thumbnail_generated={v.get('qlmanage_thumbnail_generated')} "
              f"thumbnail_bytes={v.get('qlmanage_thumbnail_bytes')} "
              f"opened_in_preview={v.get('opened_in_preview')}")

    out = {"builds": res, "verification": verify}
    json.dump(out, open(f"{WORK}/usdz_export_stats.json", "w"), indent=2)
    print("wrote thalamus_work/usdz_export_stats.json")


if __name__ == "__main__":
    main()
