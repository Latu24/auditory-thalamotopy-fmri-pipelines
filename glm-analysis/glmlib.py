"""Core GLM support library (bvbabel + numpy/scipy).

Builds single-subject GLMs directly on coregistered BrainVoyager VTC
functional data, producing BrainVoyager-native VMP statistical maps aligned
to the anatomical VMR. Implemented without the BrainVoyager GUI:

- GLM implemented mathematically (y = Xb + e): OLS betas, a GLOBAL AR(2)
  pre-whitening step (a documented simplification relative to BrainVoyager's
  voxel-wise AR(2)), contrasts as linear combinations of betas, t/F stats.
- VTCs are large relative to available memory, so a full volume is never
  loaded at once — data is read slab-wise via memmap, and the multi-run GLM
  is accumulated one run-slab at a time (XtX / XtY / sumYsq accumulation).
- VMP bounding box / resolution are copied verbatim from the VTC header so
  the statistical map lands in the identical BrainVoyager-internal frame as
  the VTC (itself coregistered to the VMR), guaranteeing overlay alignment.
"""
import os
import struct
import numpy as np
import bvbabel.prt
import bvbabel.vmp

TR_S = 1.6  # seconds
RUNS = (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# VTC header + memmap (memory-safe, header-only + slab access)
# ---------------------------------------------------------------------------
def _read_vls(f):
    out = b""
    while True:
        c = f.read(1)
        if c == b"\x00" or c == b"":
            break
        out += c
    return out.decode("utf-8", "replace")


def read_vtc_header(path):
    """Parse only the VTC header; return dict incl. data_offset and dims."""
    h = {}
    with open(path, "rb") as f:
        h["File version"] = struct.unpack("<h", f.read(2))[0]
        h["Source FMR name"] = _read_vls(f)
        h["Protocol attached"] = struct.unpack("<h", f.read(2))[0]
        if h["Protocol attached"] > 0:
            h["Protocol name"] = _read_vls(f)
        else:
            h["Protocol name"] = ""
        h["Current protocol index"] = struct.unpack("<h", f.read(2))[0]
        h["Data type"] = struct.unpack("<h", f.read(2))[0]   # 1=short 2=float
        h["Nr time points"] = struct.unpack("<h", f.read(2))[0]
        h["VTC resolution"] = struct.unpack("<h", f.read(2))[0]
        for k in ("XStart", "XEnd", "YStart", "YEnd", "ZStart", "ZEnd"):
            h[k] = struct.unpack("<h", f.read(2))[0]
        h["LR convention"] = struct.unpack("<B", f.read(1))[0]
        h["Reference space"] = struct.unpack("<B", f.read(1))[0]
        h["TR"] = struct.unpack("<f", f.read(4))[0]
        h["data_offset"] = f.tell()
    res = h["VTC resolution"]
    h["DimX"] = (h["XEnd"] - h["XStart"]) // res
    h["DimY"] = (h["YEnd"] - h["YStart"]) // res
    h["DimZ"] = (h["ZEnd"] - h["ZStart"]) // res
    h["DimT"] = h["Nr time points"]
    return h


def vtc_memmap(path, h=None):
    """Return a read-only memmap of shape (DimZ, DimY, DimX, DimT)."""
    h = h or read_vtc_header(path)
    dt = np.dtype("<f4") if h["Data type"] == 2 else np.dtype("<h")
    shape = (h["DimZ"], h["DimY"], h["DimX"], h["DimT"])
    return np.memmap(path, dtype=dt, mode="r", offset=h["data_offset"],
                     shape=shape)


# ---------------------------------------------------------------------------
# HRF and predictors
# ---------------------------------------------------------------------------
def two_gamma_hrf(tr=TR_S, length_s=32.0):
    """BrainVoyager-style canonical two-gamma HRF sampled at TR.

    Peak ~5.4 s, undershoot ~10.8 s, undershoot ratio 6 (BV defaults).
    """
    from scipy.special import gamma
    t = np.arange(0, length_s + tr, tr)
    # BV/SPM canonical parameters
    n1, l1 = 6.0, 1.0        # peak: shape/scale (gamma) -> peak at n1*l1? use SPM form
    # Use SPM double-gamma with BV-like timing
    peak_delay, peak_disp = 5.4, 0.9
    und_delay, und_disp = 10.8, 0.9
    ratio = 6.0

    def _g(t, delay, disp):
        a = (delay / disp)
        # gamma pdf with shape a, scale disp
        return (t ** (a - 1) * np.exp(-t / disp)) / (disp ** a * gamma(a))

    with np.errstate(invalid="ignore"):
        h = _g(t, peak_delay, peak_disp) - _g(t, und_delay, und_disp) / ratio
    h[t == 0] = 0.0
    h = h / np.max(h)
    return h


def convolve_predictor(boxcar, hrf):
    """Convolve a length-T box-car with the HRF, truncate to T."""
    T = boxcar.shape[0]
    c = np.convolve(boxcar, hrf)[:T]
    return c


def boxcar_from_condition(starts, stops, nvols):
    """Box-car in volume units. Event (start==stop) -> single 1; block ->
    filled start..stop inclusive."""
    bc = np.zeros(nvols, dtype=np.float64)
    for s, e in zip(starts, stops):
        s = int(s); e = int(e)
        e = max(e, s)
        e = min(e, nvols - 1)
        if s < 0:
            s = 0
        bc[s:e + 1] = 1.0
    return bc


def read_prt_conditions(path):
    """Return list of (name, starts, stops) in file order."""
    _, data = bvbabel.prt.read_prt(path)
    out = []
    for c in data:
        out.append((c["NameOfCondition"] if "NameOfCondition" in c else
                    c.get("Name of condition", c.get("name", "?")),
                    np.array(c["Time start"], dtype=int),
                    np.array(c["Time stop"], dtype=int)))
    return out


# ---------------------------------------------------------------------------
# Design matrix (per run) — task predictors + per-run confounds
# ---------------------------------------------------------------------------
def build_run_design(prt_path, nvols, hrf, condition_order=None):
    """Return (X_task, names) for one run.

    X_task: (nvols, n_cond) HRF-convolved predictors, one column per condition
    in `condition_order` (default: file order). Missing conditions -> zeros.
    """
    conds = read_prt_conditions(prt_path)
    by_name = {c[0]: c for c in conds}
    if condition_order is None:
        condition_order = [c[0] for c in conds]
    cols = []
    for name in condition_order:
        if name in by_name:
            _, s, e = by_name[name]
            bc = boxcar_from_condition(s, e, nvols)
        else:
            bc = np.zeros(nvols)
        cols.append(convolve_predictor(bc, hrf))
    X = np.column_stack(cols) if cols else np.zeros((nvols, 0))
    return X, list(condition_order)


def confounds(nvols):
    """Per-run confounds: constant + linear trend (z-scored)."""
    const = np.ones(nvols)
    lin = np.linspace(-1, 1, nvols)
    return np.column_stack([const, lin])


# ---------------------------------------------------------------------------
# AR(2) whitening (applied within-run as a causal difference filter)
# ---------------------------------------------------------------------------
def ar2_whiten(A, phi1, phi2):
    """Whiten along axis 0 (time): e_t = a_t - phi1 a_{t-1} - phi2 a_{t-2}."""
    W = A.copy()
    W[2:] = A[2:] - phi1 * A[1:-1] - phi2 * A[:-2]
    W[1] = A[1] - phi1 * A[0]
    # W[0] unchanged
    return W


def estimate_ar2(residuals):
    """Yule-Walker AR(2) from a (T, N) residual array, pooled over columns."""
    R = residuals - residuals.mean(0, keepdims=True)
    r0 = np.mean(np.sum(R * R, 0))
    r1 = np.mean(np.sum(R[1:] * R[:-1], 0))
    r2 = np.mean(np.sum(R[2:] * R[:-2], 0))
    r0 /= R.shape[0]; r1 /= R.shape[0]; r2 /= R.shape[0]
    if r0 <= 0:
        return 0.0, 0.0
    a1 = r1 / r0
    a2 = r2 / r0
    denom = 1 - a1 * a1
    if abs(denom) < 1e-8:
        return 0.0, 0.0
    phi1 = a1 * (1 - a2) / denom
    phi2 = (a2 - a1 * a1) / denom
    # stability clip
    phi1 = float(np.clip(phi1, -0.9, 0.9))
    phi2 = float(np.clip(phi2, -0.9, 0.9))
    return phi1, phi2


# ---------------------------------------------------------------------------
# Multi-run accumulating GLM solver
# ---------------------------------------------------------------------------
def _run_slabs(dimz, zchunk):
    z = 0
    while z < dimz:
        yield z, min(z + zchunk, dimz)
        z += zchunk


def fit_multirun_glm(vtc_paths, run_task_designs, hrf, log=print,
                     zchunk=8, ar_sample=4000, mask_thresh=None):
    """Fit a concatenated multi-run fixed-effects GLM voxel-wise.

    Parameters
    ----------
    vtc_paths : list of run VTC file paths (same bbox across runs).
    run_task_designs : list of (nvols, n_cond) task design arrays, one per run
        (columns aligned to the SAME condition order across runs).
    Returns dict with betas (P, DimZ,DimY,DimX), rss, dof, XtX_inv, design info,
    mask, and per-run baseline column indices.
    """
    headers = [read_vtc_header(p) for p in vtc_paths]
    h0 = headers[0]
    DimZ, DimY, DimX = h0["DimZ"], h0["DimY"], h0["DimX"]
    n_cond = run_task_designs[0].shape[1]
    nrun = len(vtc_paths)
    n_conf = 2  # const + linear per run

    # ---- assemble full design X (Ttotal x P): shared task cols + per-run confounds
    run_nvols = [d.shape[0] for d in run_task_designs]
    Ttotal = sum(run_nvols)
    P = n_cond + nrun * n_conf
    X = np.zeros((Ttotal, P))
    row = 0
    conf_cols = []
    for r in range(nrun):
        nv = run_nvols[r]
        X[row:row + nv, :n_cond] = run_task_designs[r]
        c0 = n_cond + r * n_conf
        X[row:row + nv, c0:c0 + n_conf] = confounds(nv)
        conf_cols.append((c0, c0 + n_conf))
        row += nv
    run_row_bounds = np.cumsum([0] + run_nvols)

    # ---- estimate global AR(2) from a voxel sample (OLS residuals) ----
    log("  estimating global AR(2) from voxel sample ...")
    rng = np.random.default_rng(42)
    # sample in-brain voxels using run1 temporal mean
    mm0 = vtc_memmap(vtc_paths[0], headers[0])
    # cheap mean over a strided subset of time to build mask
    tmean = np.asarray(mm0[:, :, :, ::8].mean(-1), dtype=np.float32)
    if mask_thresh is None:
        pos = tmean[tmean > 0]
        mask_thresh = float(np.percentile(pos, 40)) if pos.size else 0.0
    mask = tmean > mask_thresh
    vox_idx = np.argwhere(mask)
    if vox_idx.shape[0] > ar_sample:
        sel = rng.choice(vox_idx.shape[0], ar_sample, replace=False)
        vox_idx = vox_idx[sel]
    # gather sample timecourses across runs
    Ysamp = np.zeros((Ttotal, vox_idx.shape[0]), dtype=np.float64)
    for r in range(nrun):
        mm = vtc_memmap(vtc_paths[r], headers[r])
        a, b = run_row_bounds[r], run_row_bounds[r + 1]
        Ysamp[a:b] = np.asarray(
            mm[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2], :].T,
            dtype=np.float64)
        del mm
    # OLS on sample, per-run residuals -> AR(2)
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta_s = XtX_inv @ (X.T @ Ysamp)
    resid = Ysamp - X @ beta_s
    phis = []
    for r in range(nrun):
        a, b = run_row_bounds[r], run_row_bounds[r + 1]
        phis.append(estimate_ar2(resid[a:b]))
    phi1 = float(np.mean([p[0] for p in phis]))
    phi2 = float(np.mean([p[1] for p in phis]))
    log(f"  global AR(2): phi1={phi1:.4f} phi2={phi2:.4f} "
        f"(per-run {[(round(p[0],3),round(p[1],3)) for p in phis]})")
    del Ysamp, resid, beta_s

    # ---- whiten design block-wise, recompute (X'X)^-1 ----
    Xw = X.copy()
    for r in range(nrun):
        a, b = run_row_bounds[r], run_row_bounds[r + 1]
        Xw[a:b] = ar2_whiten(X[a:b], phi1, phi2)
    XtXw = Xw.T @ Xw
    XtXw_inv = np.linalg.pinv(XtXw)

    # ---- accumulate XtY and sumYsq slab-wise, one run-slab at a time ----
    log("  fitting voxel-wise (accumulating over run-slabs) ...")
    XtY = np.zeros((P, DimZ * DimY * DimX), dtype=np.float64)
    sumYsq = np.zeros(DimZ * DimY * DimX, dtype=np.float64)
    for r in range(nrun):
        mm = vtc_memmap(vtc_paths[r], headers[r])
        a, b = run_row_bounds[r], run_row_bounds[r + 1]
        Xwr = Xw[a:b]                     # (nv, P)
        for z0, z1 in _run_slabs(DimZ, zchunk):
            slab = np.asarray(mm[z0:z1], dtype=np.float32)      # (zc,Y,X,T)
            zc = z1 - z0
            Y = slab.reshape(-1, slab.shape[-1]).T               # (nv, Nsl)
            del slab
            Yw = ar2_whiten(Y, phi1, phi2)
            del Y
            off = z0 * DimY * DimX
            n_sl = zc * DimY * DimX
            XtY[:, off:off + n_sl] += Xwr.T @ Yw
            sumYsq[off:off + n_sl] += np.einsum("tn,tn->n", Yw, Yw)
            del Yw
        del mm
        log(f"    run {r + 1} accumulated")

    betas = XtXw_inv @ XtY                                        # (P, Nvox)
    rss = sumYsq - np.einsum("pn,pn->n", betas, XtY)
    rss = np.clip(rss, 0, None)
    dof = Ttotal - P

    shape3 = (DimZ, DimY, DimX)
    return {
        "betas": betas.reshape((P,) + shape3),
        "rss": rss.reshape(shape3),
        "dof": dof,
        "XtXw_inv": XtXw_inv,
        "P": P, "n_cond": n_cond, "nrun": nrun, "n_conf": n_conf,
        "conf_cols": conf_cols,
        "phi": (phi1, phi2),
        "mask": mask,
        "mask_thresh": mask_thresh,
        "headers": headers,
        "Ttotal": Ttotal,
        "shape3": shape3,
    }


# ---------------------------------------------------------------------------
# Statistics from a fitted model
# ---------------------------------------------------------------------------
def t_contrast(fit, c):
    """t-map for contrast vector c (length P). Returns (t, shape3)."""
    c = np.asarray(c, dtype=np.float64)
    betas = fit["betas"]
    P = fit["P"]
    cb = np.tensordot(c, betas, axes=([0], [0]))          # (DimZ,DimY,DimX)
    var_unit = float(c @ fit["XtXw_inv"] @ c)
    s2 = fit["rss"] / fit["dof"]
    denom = np.sqrt(np.clip(var_unit * s2, 1e-20, None))
    t = cb / denom
    t[~fit["mask"]] = 0.0
    t[~np.isfinite(t)] = 0.0
    return t


def f_omnibus(fit, cond_cols=None):
    """F-test that the given task columns jointly explain variance.

    Full model = current fit; reduced model drops `cond_cols` (default: all
    task condition columns). Returns F-map (shape3).
    """
    if cond_cols is None:
        cond_cols = list(range(fit["n_cond"]))
    q = len(cond_cols)
    # F = ((RSS_reduced - RSS_full)/q) / (RSS_full/dof)
    # RSS_reduced - RSS_full = (C beta)' [C (X'X)^-1 C']^-1 (C beta)
    C = np.zeros((q, fit["P"]))
    for i, col in enumerate(cond_cols):
        C[i, col] = 1.0
    betas = fit["betas"]
    Cb = np.tensordot(C, betas, axes=([1], [0]))            # (q, DimZ,DimY,DimX)
    M = np.linalg.pinv(C @ fit["XtXw_inv"] @ C.T)           # (q,q)
    # numerator quadratic form per voxel
    shape3 = fit["shape3"]
    Cb2 = Cb.reshape(q, -1)
    num = np.einsum("in,ij,jn->n", Cb2, M, Cb2).reshape(shape3)
    s2 = fit["rss"] / fit["dof"]
    with np.errstate(invalid="ignore", divide="ignore"):
        F = (num / q) / np.clip(s2, 1e-20, None)
    F[~fit["mask"]] = 0.0
    F[~np.isfinite(F)] = 0.0
    return F


# ---------------------------------------------------------------------------
# VMP writing (aligned to the VTC/VMR frame)
# ---------------------------------------------------------------------------
def _raw_to_vmp_frame(raw):
    """Convert a raw BV-internal (DimZ,DimY,DimX) array into the array layout
    that bvbabel.vmp.write_vmp expects (it re-applies transpose+flip so the
    ON-DISK bytes are the raw (DimZ,DimY,DimX) loop order matching the VTC)."""
    a = np.transpose(raw, (0, 2, 1))     # (DimZ, DimX, DimY)
    a = a[::-1, ::-1, ::-1]
    return np.ascontiguousarray(a.astype(np.float32))


def write_stat_vmp(path, maps, vtc_header, vtc_name="", prt_name=""):
    """Write one or more statistical sub-maps to a VMP aligned to the VTC/VMR.

    maps : list of dicts each {name, data(raw DimZ,DimY,DimX), type(1=t,4=F),
           df1, df2, threshold, upper, fdr_table(optional Nx3 array
           [q, critical_std_t, critical_conservative_t]), cluster_size
           (optional int, enables ClusterSizeThreshold if given)}.
    fdr_table/cluster_size default to the prior off/empty behavior if not
    given, so existing callers are unaffected. A real, non-empty fdr_table is
    what genuine BrainVoyager-produced VMPs carry — leaving it empty
    (SizeOfFDRTable=0) makes BrainVoyager's significance-based coloring
    render as pure extremes-only rather than a graded map.
    """
    h = vtc_header
    header, _ = bvbabel.vmp.create_vmp()
    header["NrOfSubMaps"] = np.int32(len(maps))
    header["XStart"] = np.int32(h["XStart"]); header["XEnd"] = np.int32(h["XEnd"])
    header["YStart"] = np.int32(h["YStart"]); header["YEnd"] = np.int32(h["YEnd"])
    header["ZStart"] = np.int32(h["ZStart"]); header["ZEnd"] = np.int32(h["ZEnd"])
    header["Resolution"] = np.int32(h["VTC resolution"])
    header["DimX"] = np.int32(h["DimX"]); header["DimY"] = np.int32(h["DimY"])
    header["DimZ"] = np.int32(h["DimZ"])
    header["NameOfVTCFile"] = vtc_name
    header["NameOfProtocolFile"] = prt_name

    template_map = header["Map"][0]
    header["Map"] = []
    data_stack = []
    for m in maps:
        md = dict(template_map)
        md["TypeOfMap"] = np.int32(m.get("type", 1))
        md["MapName"] = m["name"]
        md["MapThreshold"] = np.float32(m.get("threshold", 2.0))
        md["UpperThreshold"] = np.float32(m.get("upper", 8.0))
        md["DF1"] = np.int32(m.get("df1", 0))
        md["DF2"] = np.int32(m.get("df2", 0))
        md["ShowPosNegValues"] = np.byte(m.get("showposneg", 3))
        fdr_table = m.get("fdr_table")
        if fdr_table is not None:
            fdr_table = np.asarray(fdr_table, dtype=np.float32)
            md["SizeOfFDRTable"] = np.int32(fdr_table.shape[0])
            md["FDRTableInfo"] = fdr_table
            md["UseFDRTableIndex"] = np.int32(m.get("fdr_table_index", 1))
        cluster_size = m.get("cluster_size")
        if cluster_size is not None:
            md["ClusterSizeThreshold"] = np.int32(cluster_size)
            md["EnableClusterSizeThreshold"] = np.byte(1)
        header["Map"].append(md)
        data_stack.append(_raw_to_vmp_frame(m["data"]))

    if len(data_stack) == 1:
        data_img = data_stack[0]
    else:
        data_img = np.stack(data_stack, axis=-1)  # (DimZ,DimX,DimY,NrMaps)
    bvbabel.vmp.write_vmp(path, header, data_img)
    return path


def read_vmp_raw(path):
    """Read a VMP's ON-DISK data straight back to raw (DimZ,DimY,DimX[,maps])
    without bvbabel's transpose/flip — for orientation QC only."""
    h = {}
    with open(path, "rb") as f:
        f.read(4)  # id
        struct.unpack("<h", f.read(2)); struct.unpack("<h", f.read(2))
        nsub = struct.unpack("<i", f.read(4))[0]
        for k in ("NrOfTimePoints", "NrOfComponentParams", "ShowFrom", "ShowTo",
                  "FpFrom", "FpTo"):
            struct.unpack("<i", f.read(4))
        xs = struct.unpack("<i", f.read(4))[0]; xe = struct.unpack("<i", f.read(4))[0]
        ys = struct.unpack("<i", f.read(4))[0]; ye = struct.unpack("<i", f.read(4))[0]
        zs = struct.unpack("<i", f.read(4))[0]; ze = struct.unpack("<i", f.read(4))[0]
        res = struct.unpack("<i", f.read(4))[0]
        dx = struct.unpack("<i", f.read(4))[0]; dy = struct.unpack("<i", f.read(4))[0]
        dz = struct.unpack("<i", f.read(4))[0]
        _read_vls(f); _read_vls(f); _read_vls(f)
        for m in range(nsub):
            struct.unpack("<i", f.read(4))                  # TypeOfMap
            struct.unpack("<f", f.read(4)); struct.unpack("<f", f.read(4))
            _read_vls(f)                                     # MapName
            f.read(3 * 4)                                    # 4 RGB triplets
            f.read(1)                                        # UseVMPColor
            _read_vls(f)                                     # LUTFileName
            struct.unpack("<f", f.read(4))                   # TransparentColorFactor
            struct.unpack("<i", f.read(4))                   # ClusterSizeThreshold
            f.read(1)                                        # EnableClusterSizeThreshold
            struct.unpack("<i", f.read(4))                   # ShowValuesAbove
            struct.unpack("<i", f.read(4)); struct.unpack("<i", f.read(4))  # DF1 DF2
            f.read(1)                                        # ShowPosNegValues
            struct.unpack("<i", f.read(4))                   # NrOfUsedVoxels
            sfdr = struct.unpack("<i", f.read(4))[0]         # SizeOfFDRTable
            f.read(sfdr * 3 * 4)
            struct.unpack("<i", f.read(4))                   # UseFDRTableIndex
        raw = np.fromfile(f, dtype="<f4", count=dz * dy * dx * nsub)
    if nsub == 1:
        return raw.reshape(dz, dy, dx), (xs, xe, ys, ye, zs, ze, res)
    return raw.reshape(dz, dy, dx, nsub), (xs, xe, ys, ye, zs, ze, res)
