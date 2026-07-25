"""Thalamus pipeline -- Part C: interactive 3D HTML models of the thalamus,
colored by best frequency.

Builds two self-contained (offline, embedded plotly.js) HTML models --
CF best-frequency and AM best-frequency -- of the whole-thalamus surface,
colored wherever significant data exists (nearest-labeled-voxel sampling
onto every mesh vertex, log-Hz colormap, grey where no surviving data). The
MGB boundary is rendered as a thin dark outline shell so its location stays
identifiable even where it carries no data of its own (see the MGB-coverage
note in 14_mgb_appearance_check.py). Rotatable and zoomable in any browser.

Mesh quality: the raw marching-cubes surface can read as hollow/see-through
if the mesh has small gaps or the lighting is too directional. This is
addressed by (a) verifying watertightness (every mesh edge shared by exactly
2 faces), (b) a more ambient-heavy Plotly lighting model at full opacity so
concave regions don't render near-black, and (c) Taubin (lambda/mu) mesh
smoothing of the geometry itself for a smooth, rounded appearance instead of
blocky marching-cubes terracing.

Voxel gate: "bf > 0" means the voxel survives its own p<0.01 (uncorrected)
omnibus-F test, computed separately for CF (q=36) and AM (q=9) by
17_apply_p01_gate.py. This is a real, sparse per-condition significance
test, not a functional-coverage-only mask -- the sample radius used to color
mesh vertices (SAMPLE_RADIUS_VOX) is kept at 1.5 voxels regardless, so sparse
regions render honestly sparse rather than being visually inflated.

QC note (MGB coverage): the Sitek MGB (labels 7/8) is anatomically correctly
segmented (postero-inferior geniculate) but sits at or below the inferior
edge of the GLM's functional field of view: only a small fraction of MGB
voxels have any functional coverage at all, so most of the MGB surface is
grey (no data) regardless of the significance gate -- a functional-FOV
limitation, not a statistical-threshold artifact.

Inputs: native-space thalamus/MGB masks, native-space best-frequency NIfTIs
(from 12_reconcile_space.py + 17_apply_p01_gate.py). Outputs:
sub-01_thalamus3D_CFbestfreq.html, sub-01_thalamus3D_AMbestfreq.html, and a
model3d_stats.json sidecar.
"""
import os, sys, json
import numpy as np
import nibabel as nib
from skimage import measure
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plotly.graph_objects as go

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ROOT = PROJECT_ROOT
ANA = f"{ROOT}/derivatives/sub-01/analysis"
WORK = f"{ANA}/thalamus_work"
ANAT = f"{ROOT}/derivatives/sub-01/anat/sub-01_desc-UNIdenoisedN4.nii.gz"

CF_HZ = 200.0 * (8000.0 / 200.0) ** (np.arange(36) / 35.0)
AM_HZ = 1.0 * (16.0 / 1.0) ** (np.arange(9) / 8.0)

GREY = (0.55, 0.55, 0.55)
SAMPLE_RADIUS_VOX = 1.5   # nearest labeled voxel within <=1.5 vox

# Taubin (lambda/mu) mesh-smoothing parameters -- classic values that smooth
# without the progressive shrinkage plain Laplacian smoothing causes.
SMOOTH_ITERS = 12
TAUBIN_LAMBDA = 0.5
TAUBIN_MU = -0.53


def mask_surface(mask, step=1):
    """Marching cubes on a padded mask. Returns (verts_index_space, faces) in the
    SAME index frame as the input `mask` array (pad added then subtracted back)."""
    if mask.sum() == 0:
        return None
    pad = np.pad(mask.astype(np.float32), 1)
    verts, faces, _, _ = measure.marching_cubes(pad, level=0.5, step_size=step)
    verts = verts - 1.0  # undo pad -> back in `mask`'s own index space
    return verts, faces


def check_watertight(faces):
    """A closed (watertight) manifold triangle mesh has every edge shared by
    EXACTLY 2 faces. Returns (n_bad_edges, n_edges); watertight iff n_bad==0.
    Non-watertight edges are exactly where a viewer could see "into" the mesh
    since there's a gap in the shell there."""
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    n_bad = int(np.sum(counts != 2))
    return n_bad, int(len(counts))


def taubin_smooth(verts, faces, iterations=SMOOTH_ITERS, lam=TAUBIN_LAMBDA, mu=TAUBIN_MU):
    """Taubin (lambda/mu) mesh smoothing: alternates an umbrella-Laplacian
    smoothing step (lam>0, would shrink the mesh on its own) with a slight
    inverse step (mu<0, |mu|>lam, re-inflates), producing a smooth, rounded
    appearance without plain Laplacian's shrinkage. Moves vertex POSITIONS
    only -- faces/topology (and therefore watertightness) are unchanged."""
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


def sample_vertex_colors(verts_idx, val_coords, val_values, hz_lo, hz_hi, cmap_name="turbo"):
    """For each mesh vertex (index-space coords, post-smoothing), find the
    nearest voxel-with-a-best-freq-value within SAMPLE_RADIUS_VOX; color by
    log10(value) on a fixed [hz_lo,hz_hi] scale; grey if none within radius
    (no functional coverage nearby). Returns (rgb_strings, has_data bool)."""
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
    rgb255 = (rgb * 255).astype(int)
    strs = [f"rgb({r},{g},{b})" for r, g, b in rgb255]
    return strs, has_data


GREY_BAND_FRAC = 0.15   # fraction of the colorbar's total drawn extent reserved
                        # for the "no data" grey band, below the real Hz range


def build_intensity_and_colorscale(verts_idx, val_coords, val_values, hz_lo, hz_hi,
                                   cmap_name="turbo", n_stops=24):
    """Builds a REAL, Plotly-managed colorbar (Mesh3d `intensity` + `colorscale`)
    for a separate, invisible host trace -- see build()'s "colorbar host" trace
    for why this is kept fully separate from the visible mesh's own coloring.

    "No data" vertices render grey without appearing as a real (mis-)colored Hz
    value and without the colorbar's labeled tick range including anything but
    the true Hz range. This is done by extending the trace's [cmin,cmax] domain
    slightly below the true log10(hz_lo) to make room for a solid-grey band
    (GREY_BAND_FRAC of the total colorbar height), assigning every "no data"
    vertex an intensity of exactly cmin (the bottom of that grey band), and
    setting the colorbar's own tickvals/ticktext to only the real Hz ticks --
    so the labeled colorbar reads as a clean Hz scale, with an unlabeled grey
    sliver below it for "no data", like a standard "N/A" legend swatch.

    Returns (intensity_array, has_data, colorscale, cmin, cmax).
    """
    n = len(verts_idx)
    has_data = np.zeros(n, bool)
    vals = np.full(n, np.nan)
    if len(val_coords):
        tree = cKDTree(val_coords)
        d, idx = tree.query(verts_idx, k=1)
        has_data = d <= SAMPLE_RADIUS_VOX
        vals[has_data] = val_values[idx[has_data]]

    true_lo, true_hi = np.log10(hz_lo), np.log10(hz_hi)
    span = true_hi - true_lo
    pad = GREY_BAND_FRAC / (1.0 - GREY_BAND_FRAC) * span   # so grey occupies exactly GREY_BAND_FRAC of [cmin,cmax]
    cmin = true_lo - pad
    cmax = true_hi
    eps = pad / (cmax - cmin)   # fraction of the domain that is the grey band

    intensity = np.full(n, cmin, dtype=np.float64)          # default: "no data" -> bottom of grey band
    intensity[has_data] = np.log10(vals[has_data])

    grey_hex = mcolors.to_hex(GREY)
    cmap = cm.get_cmap(cmap_name)
    turbo_stops = []
    for i in range(n_stops + 1):
        tf = i / n_stops
        pos = eps + tf * (1.0 - eps)
        r, g, b = (np.array(cmap(tf)[:3]) * 255).astype(int)
        turbo_stops.append([pos, f"rgb({r},{g},{b})"])
    colorscale = [[0.0, grey_hex], [eps, grey_hex]] + turbo_stops
    return intensity, has_data, colorscale, cmin, cmax


def build(dim, hz, out_html):
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
    bfc = bf[sl] * thal_c            # best-freq values restricted to thalamus (incl. MGB)

    thal_s = mask_surface(thal_c, step=1)   # full resolution (primary surface)
    mgb_s = mask_surface(mgb_c, step=1)

    val_coords = np.argwhere(bfc > 0).astype(float)
    val_values = bfc[bfc > 0]

    fig = go.Figure()
    watertight_report = {}
    thal_vw = None

    # ---- primary colored whole-thalamus surface (solid, smoothed, fully opaque) ----
    # The VISIBLE surface is colored by per-vertex `vertexcolor` (RGB), not by
    # `intensity`+`colorscale`. With `intensity`, Plotly interpolates the scalar
    # intensity across each triangle and colormaps per fragment, so any face
    # bridging a grey "no data" vertex and a colored vertex ramps through the
    # entire Turbo scale between those two intensity values -- Turbo's wide
    # mid-band is cyan/blue, producing a rainbow/blue halo around every colored
    # patch (confirmed by rendering and inspecting boundary faces). With
    # `vertexcolor`, Plotly Gouraud-interpolates the final RGB colors instead,
    # so a boundary face blends grey directly toward the neighbour's actual
    # color -- the patch fades naturally to grey with no halo.
    #
    # The "Frequency (Hz)" colorbar (incl. its grey "no data" swatch) is hosted
    # on a separate, fully transparent (opacity=0) Mesh3d that shares the same
    # geometry and carries intensity+colorscale+showscale. A colorbar is an
    # SVG-layer element generated from the trace's colorscale/cmin/cmax -- it
    # renders regardless of whether the host geometry draws any pixels -- so
    # this hosts a robust, officially-supported colorbar without tinting the
    # visible surface. See build_intensity_and_colorscale() for the grey-band
    # scheme.
    n_colored_verts = 0
    if thal_s is not None:
        verts_idx, f = thal_s
        n_bad, n_edges = check_watertight(f)
        watertight_report["thalamus"] = {"watertight": n_bad == 0,
                                         "bad_edges": n_bad, "n_edges": n_edges}
        verts_sm = taubin_smooth(verts_idx, f)          # geometry smoothing (inflated look)
        vcolors, has_data = sample_vertex_colors(
            verts_sm, val_coords, val_values, hz[0], hz[-1])            # visible surface (no halo)
        intensity, _, colorscale, cmin, cmax = build_intensity_and_colorscale(
            verts_sm, val_coords, val_values, hz[0], hz[-1])           # colorbar host only
        n_colored_verts = int(has_data.sum())
        thal_vw = nib.affines.apply_affine(aff_c, verts_sm)
        ticks = _hz_ticks(hz)
        solid_lighting = dict(ambient=0.55, diffuse=0.75, specular=0.35,
                              roughness=0.55, fresnel=0.15)
        # visible surface: per-vertex RGB (grey where no data, Turbo where data) --
        # Gouraud color interpolation, so NO blue halo at the grey/colored boundary.
        fig.add_trace(go.Mesh3d(
            x=thal_vw[:, 0], y=thal_vw[:, 1], z=thal_vw[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            vertexcolor=vcolors,
            opacity=1.0, flatshading=False,
            name="whole thalamus (colored by best-frequency; grey = does not survive p<0.01)",
            showscale=False, hoverinfo="name",
            # solid-object lighting: enough ambient that concave regions never
            # render near-black (the "hollow" look), diffuse+specular give it
            # a lit, rounded, physical appearance rather than a flat cutout.
            lighting=solid_lighting,
            lightposition=dict(x=200, y=200, z=250)))
        # colorbar host: a single, fully transparent degenerate triangle (NOT the
        # full geometry -- keeps the file lean) carrying the real "Frequency (Hz)"
        # colorbar (with its grey "no data" swatch) via intensity+colorscale. The
        # colorbar is an SVG-layer element driven only by colorscale/cmin/cmax, so
        # 3 host vertices suffice; cmin/cmax below set the labeled range regardless
        # of the host's actual intensity values.
        hv = thal_vw[f[0]]
        fig.add_trace(go.Mesh3d(
            x=hv[:, 0], y=hv[:, 1], z=hv[:, 2], i=[0], j=[1], k=[2],
            intensity=intensity[f[0]], colorscale=colorscale, cmin=cmin, cmax=cmax,
            opacity=0.0, showscale=True, showlegend=False, hoverinfo="skip",
            name="colorbar", lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
            colorbar=dict(title=dict(text="Frequency (Hz)", side="right"),
                         tickvals=np.log10(ticks), ticktext=[f"{t:g}" for t in ticks],
                         x=1.02, len=0.75, thickness=20)))

    # ---- MGB boundary (no data of its own; location marker only) ----
    if mgb_s is not None:
        verts_idx, f = mgb_s
        n_bad, n_edges = check_watertight(f)
        watertight_report["mgb"] = {"watertight": n_bad == 0,
                                    "bad_edges": n_bad, "n_edges": n_edges}
        verts_sm = taubin_smooth(verts_idx, f)
        vw = nib.affines.apply_affine(aff_c, verts_sm)
        fig.add_trace(go.Mesh3d(x=vw[:, 0], y=vw[:, 1], z=vw[:, 2],
                                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                color="black", opacity=0.25, flatshading=False,
                                name="MGB boundary (location marker; mostly outside "
                                     "the functional FOV, see report)",
                                showscale=False, hoverinfo="name",
                                lighting=dict(ambient=0.75, diffuse=0.35, specular=0.1)))

    n_in_mgb = int(((bf > 0) & mgb).sum())
    n_val_total = int((bfc > 0).sum())
    wt_ok = all(v["watertight"] for v in watertight_report.values())
    title = (f"sub-01 auditory thalamus — {dim.replace('_',' ')} "
             f"(GLM, Gaussian-fit best-freq, p&lt;0.01 uncorrected omnibus-F)<br>"
             f"<sub>{n_val_total} significant voxels (p&lt;0.01)</sub>")
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=13)),
        scene=dict(xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)",
                   aspectmode="data",
                   camera=dict(eye=dict(x=-1.6, y=-1.6, z=1.0))),
        margin=dict(l=0, r=0, t=70, b=0), showlegend=True,
        legend=dict(x=0, y=1))
    fig.write_html(out_html, include_plotlyjs=True, full_html=True)
    print(f"wrote {out_html}  (thalamus surf colored: {n_colored_verts} vertices "
          f"w/ data, MGB boundary kept, {n_val_total} voxels w/ value, {n_in_mgb} in MGB, "
          f"watertight={wt_ok})")
    return {"dim": dim, "n_thal_voxels_with_value": n_val_total,
            "n_in_mgb": n_in_mgb, "n_colored_surface_vertices": n_colored_verts,
            "sample_radius_vox": SAMPLE_RADIUS_VOX, "watertight": watertight_report,
            "smoothing": {"method": "Taubin lambda/mu", "iterations": SMOOTH_ITERS,
                         "lambda": TAUBIN_LAMBDA, "mu": TAUBIN_MU},
            "html": out_html}


def _hz_ticks(hz):
    if hz[-1] > 100:   # CF
        return np.array([200, 500, 1000, 2000, 4000, 8000])
    return np.array([1, 2, 4, 8, 16])


def main():
    res = []
    res.append(build("CF_bestfreq", CF_HZ,
                     f"{ANA}/sub-01_thalamus3D_CFbestfreq.html"))
    res.append(build("AM_bestfreq", AM_HZ,
                     f"{ANA}/sub-01_thalamus3D_AMbestfreq.html"))
    json.dump(res, open(f"{WORK}/model3d_stats.json", "w"), indent=2)
    print("DONE 3D models (full-thalamus vertex coloring)")


if __name__ == "__main__":
    main()
