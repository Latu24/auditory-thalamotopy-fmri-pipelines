"""Write BrainVoyager-native .glm files. bvbabel can read .glm files but has
no write_glm function — this reverse-engineers the inverse of
bvbabel.glm.read_glm's byte layout, cross-checked against a real GLM file's
header field values/conventions (File version=4, Separate predictors=0,
Time course normalization=0, Cortex-based mask=0 with Nr voxels in mask set
to the full grid size, Nr confounds per study, etc.).

Spatial data layout matches glmlib.py's write_stat_vmp/_raw_to_vmp_frame
convention exactly (the same transpose+flip round-trip already validated by
every VMP this pipeline produces), extended here to the GLM's stacked
value-type axis (R2, SS, betas, SS_XiY, meantc, ARlag) instead of a per-map
stack.

The AR(2) whitening used throughout this pipeline is a single GLOBAL
estimate (not BrainVoyager's native per-voxel AR), so the "Design matrix"
and "Inverted X'X matrix" stored here are the WHITENED design and its own
inverse (not the raw design BrainVoyager would normally store) — this keeps
the file internally self-consistent (recomputing t = c'b / sqrt(s2 *
c'(X'X)^-1 c) from what's stored reproduces the actual fitted t-values), at
the cost of deviating from strict BrainVoyager per-voxel-AR convention.
"""
import struct
import numpy as np
import bvbabel.glm as glmmod


def _raw_to_glm_frame(raw_zyx):
    """(DimZ,DimY,DimX) -> (DimZ,DimX,DimY), TRANSPOSE ONLY, no flip.

    This matches this pipeline's own validated GLM read-side convention
    (toraw = transpose(bvbabel_readglm_output, (0,2,1))) — NOT glmlib.py's
    _raw_to_vmp_frame, which additionally flips. That flip is specific to
    the VMP path; applying it in the GLM writer as well produces a mirrored
    (misaligned-with-VMR) file."""
    a = np.transpose(raw_zyx, (0, 2, 1))
    return np.ascontiguousarray(a.astype(np.float32))


def write_glm(path, header, R2, SS, beta, SS_XiY, meantc, ARlag):
    """Write a standard (non-RFX) VMR-VTC .glm file.

    header must contain: NrTimePoints, NrAllPredictors, NrConfoundPredictors,
    NrStudies, NrConfoundsPerStudy (list), ResolutionMultiplier,
    SerialCorrelation (0/1/2), MeanSerialCorrBefore, MeanSerialCorrAfter,
    XStart/XEnd/YStart/YEnd/ZStart/ZEnd, StudyInfo (list of dicts with
    NrTimePoints/NameOfStudyData/NameOfSDM), PredictorInfo (list of dicts
    with NameInternal/NameCustom/Color), DesignMatrix (N x M), InvXtX (M x M).

    R2, SS, meantc: (DimZ,DimY,DimX) raw arrays.
    beta, SS_XiY: (DimZ,DimY,DimX,P) raw arrays.
    ARlag: (DimZ,DimY,DimX,0/1/2) raw array (2nd dim size matches
    SerialCorrelation: 0 means an (Z,Y,X,0)-shaped placeholder is fine).
    """
    P = header["NrAllPredictors"]
    sc = header["SerialCorrelation"]

    with open(path, "wb") as f:
        f.write(struct.pack("<h", 4))                      # File version
        f.write(struct.pack("<B", 1))                       # Type: VMR-VTC
        f.write(struct.pack("<B", 0))                        # RFX: standard

        f.write(struct.pack("<i", header["NrTimePoints"]))
        f.write(struct.pack("<i", P))
        f.write(struct.pack("<i", header["NrConfoundPredictors"]))
        f.write(struct.pack("<i", header["NrStudies"]))

        if header["NrStudies"] > 1:
            f.write(struct.pack("<i", len(header["NrConfoundsPerStudy"])))
            for n in header["NrConfoundsPerStudy"]:
                f.write(struct.pack("<i", int(n)))

        f.write(struct.pack("<B", 0))   # Separate predictors: 0 (matches reference file)
        f.write(struct.pack("<B", 0))   # Time course normalization: 0 (matches reference file)
        f.write(struct.pack("<h", header["ResolutionMultiplier"]))
        f.write(struct.pack("<B", sc))
        f.write(struct.pack("<f", header["MeanSerialCorrBefore"]))
        f.write(struct.pack("<f", header["MeanSerialCorrAfter"]))

        for k in ("XStart", "XEnd", "YStart", "YEnd", "ZStart", "ZEnd"):
            f.write(struct.pack("<h", header[k]))

        f.write(struct.pack("<B", 0))   # Cortex-based mask: no
        range_X = header["XEnd"] - header["XStart"]
        range_Y = header["YEnd"] - header["YStart"]
        range_Z = header["ZEnd"] - header["ZStart"]
        r = header["ResolutionMultiplier"]
        nr_voxels = (range_X // r) * (range_Y // r) * (range_Z // r)
        f.write(struct.pack("<i", nr_voxels))   # matches reference (full grid, not a real mask)
        glmmod.write_variable_length_string(f, "")   # Name of cortex-based mask

        for si in header["StudyInfo"]:
            f.write(struct.pack("<i", si["NrTimePoints"]))
            glmmod.write_variable_length_string(f, si["NameOfStudyData"])
            glmmod.write_variable_length_string(f, si["NameOfSDM"])

        for pi in header["PredictorInfo"]:
            glmmod.write_variable_length_string(f, pi["NameInternal"])
            glmmod.write_variable_length_string(f, pi["NameCustom"])
            glmmod.write_RGB_bytes(f, pi["Color"])
            f.write(b"\x00" * 9)   # unknown/reserved bytes (read_glm skips these too)

        # Design matrix (N x M), row-major: outer loop N (time), inner loop M (predictors)
        dm = header["DesignMatrix"].astype(np.float32)
        f.write(dm.astype("<f").tobytes(order="C"))

        # Inverted X'X (M x M)
        invxtx = header["InvXtX"].astype(np.float32)
        f.write(invxtx.astype("<f").tobytes(order="C"))

        # ------------------------------------------------------------------
        # Data section: stack value-types along a new last axis in the exact
        # order read_glm expects: R2, SS, beta(P), SS_XiY(P), meantc, ARlag(0/1/2)
        # ------------------------------------------------------------------
        DimZ, DimY, DimX = R2.shape
        n_ar = {0: 0, 1: 1, 2: 2}[sc]
        V = 2 + 2 * P + 1 + n_ar

        frames = []
        frames.append(_raw_to_glm_frame(R2))
        frames.append(_raw_to_glm_frame(SS))
        for i in range(P):
            frames.append(_raw_to_glm_frame(beta[..., i]))
        for i in range(P):
            frames.append(_raw_to_glm_frame(SS_XiY[..., i]))
        frames.append(_raw_to_glm_frame(meantc))
        for i in range(n_ar):
            frames.append(_raw_to_glm_frame(ARlag[..., i]))

        assert len(frames) == V
        data_img = np.stack(frames, axis=-1)   # (DimZ,DimX,DimY,V) matching write_vmp's multi-map convention
        data_img = data_img[::-1, ::-1, ::-1, :]
        data_img = np.transpose(data_img, (3, 0, 2, 1))   # -> (V, DimZ, DimY, DimX)
        f.write(data_img.astype("<f").tobytes(order="C"))

    return path
