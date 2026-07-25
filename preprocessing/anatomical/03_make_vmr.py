#!/usr/bin/env python3
"""Build the BrainVoyager VMR (+ V16) anatomical volume from the N4-corrected
MP2RAGE UNI NIfTI, via bvbabel. Input: the N4-corrected UNI NIfTI and the
INV2-derived brain mask. Output: an 8-bit VMR, a full-precision V16, and a
JSON build/centering report.

Framing-cube centering: after reorientation to closest-canonical RAS the
data is not cubic (shape (240, 320, 320)), but VMR/V16 store data inside a
cubic "framing cube" of side length FramingCubeDim = max(shape). Each axis
is centered inside that cube via Offset{X,Y,Z} = (FramingCubeDim - Dim)//2,
computed generically per axis (not hardcoded), so the short axis is not left
pinned to one corner of the cube -- this matters for any BrainVoyager
operation that expands the data into the full FramingCubeDim^3 volume (e.g.
rotation/resampling views, VOI/mesh tools, Talairach-adjacent geometry).

Axis mapping (verified against bvbabel's write_vmr()/read_vmr() source and a
synthetic round-trip test): the on-disk axis order is (DimZ, DimY, DimX),
and the array passed to write_vmr()/returned by read_vmr() is in
NIfTI-standard order with shape = (DimZ, DimX, DimY), i.e. axis0->DimZ,
axis1->DimX, axis2->DimY.
"""
import os, json
import numpy as np
import nibabel as nib
import bvbabel.vmr, bvbabel.v16

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ANAT = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "anat")
src = os.path.join(ANAT, "sub-01_desc-UNIdenoisedN4_v2.nii.gz")   # N4-corrected UNI (step 02 output)
maskp = os.path.join(ANAT, "sub-01_desc-brainmaskINV2.nii.gz")     # INV2-derived brain mask

img = nib.as_closest_canonical(nib.load(src))
data = np.asarray(img.get_fdata(), dtype=np.float64)
mask = nib.as_closest_canonical(nib.load(maskp)).get_fdata() > 0.5
vox = tuple(float(z) for z in img.header.get_zooms()[:3])
ax = nib.aff2axcodes(img.affine)
print("canonical axcodes", ax, "shape", data.shape, "vox", vox)

# --- 8-bit scaling to 0..225 using robust brain max ---
p999 = float(np.percentile(data[mask], 99.9))
scale = 225.0 / p999
vmr8 = np.clip(np.rint(data * scale), 0, 225).astype(np.uint8)
print(f"robust brain p99.9={p999:.1f} -> scale={scale:.5f}; vmr8 range {vmr8.min()}..{vmr8.max()}")

# --- 16-bit full precision for V16 ---
v16 = np.clip(np.rint(data), 0, 65535).astype(np.uint16)

Z, X, Y = data.shape  # data is (axis0,axis1,axis2) fed as Nifti-standard
FRAME = int(max(data.shape))

# --- centre each axis inside the framing cube ---
offX = (FRAME - X) // 2
offY = (FRAME - Y) // 2
offZ = (FRAME - Z) // 2
print(f"Dim(X,Y,Z)=({X},{Y},{Z}) FramingCubeDim={FRAME} -> Offset(X,Y,Z)=({offX},{offY},{offZ})")

def base_header():
    h, _ = bvbabel.vmr.create_vmr()
    h["File version"] = 4
    h["DimX"], h["DimY"], h["DimZ"] = X, Y, Z
    h["FramingCubeDim"] = FRAME
    h["OffsetX"], h["OffsetY"], h["OffsetZ"] = offX, offY, offZ
    h["NRows"], h["NCols"] = Y, X
    h["FoVRows"] = Y * vox[2]
    h["FoVCols"] = X * vox[1]
    h["VoxelSizeX"], h["VoxelSizeY"], h["VoxelSizeZ"] = vox[1], vox[2], vox[0]
    h["SliceThickness"] = vox[0]
    h["VoxelResolutionVerified"] = 1
    h["VoxelResolutionInTALmm"] = 0     # native space, not Talairach
    h["ReferenceSpaceVMR"] = 0          # native
    h["CoordinateSystem"] = 0
    h["PosInfosVerified"] = 1
    h["LeftRightConvention"] = 1        # radiological flag; verified visually
    # native-centred sagittal slab geometry (slice normal = L-R = axis0/Z)
    halfLR = 0.5 * Z * vox[0]
    h["Slice1CenterX"], h["Slice1CenterY"], h["Slice1CenterZ"] = -halfLR, 0.0, 0.0
    h["SliceNCenterX"], h["SliceNCenterY"], h["SliceNCenterZ"] =  halfLR, 0.0, 0.0
    return h

# --- write V16 (full precision) ---
v16h, _ = bvbabel.v16.create_v16()
v16h["DimX"], v16h["DimY"], v16h["DimZ"] = X, Y, Z
v16_path = os.path.join(ANAT, "sub-01_desc-UNIdenoisedN4_anat_v2.v16")
bvbabel.v16.write_v16(v16_path, v16h, v16)

# --- write VMR (8-bit) with V16 stats recorded ---
h = base_header()
brain16 = v16[mask]
h["VMROrigV16MinValue"] = int(brain16.min())
h["VMROrigV16MeanValue"] = int(brain16.mean())
h["VMROrigV16MaxValue"] = int(brain16.max())
vmr_path = os.path.join(ANAT, "sub-01_desc-UNIdenoisedN4_anat_v2.vmr")
bvbabel.vmr.write_vmr(vmr_path, h, vmr8)

# --- verification: does OffsetZ actually centre the short axis? ---
# The real (Dim) grid occupies index range [OffsetZ, OffsetZ+DimZ) inside the
# FramingCubeDim cube along the BV Z axis (=our data axis0=L-R). Centred iff
# left pad == right pad == (FRAME-Z)/2, i.e. midpoint of that range == FRAME/2.
lo, hi = offZ, offZ + Z
pad_left, pad_right = lo, FRAME - hi
data_midpoint = 0.5 * (lo + hi)
cube_midpoint = 0.5 * FRAME
centering_check = dict(
    axis="Z (BV internal Z = left-right, our NIfTI-canonical data axis0, len=%d)" % Z,
    framing_cube_dim=FRAME,
    offset_used=offZ,
    occupied_index_range=[lo, hi],
    pad_left=pad_left, pad_right=pad_right,
    data_midpoint_index=data_midpoint, cube_midpoint_index=cube_midpoint,
    centred=bool(pad_left == pad_right and abs(data_midpoint - cube_midpoint) < 1e-9),
)
print("centering check:", json.dumps(centering_check, indent=2))

meta = dict(source=src, shape=list(data.shape), voxel_mm=list(vox),
            axcodes="".join(ax), scale_to_225=scale, robust_p999=p999,
            DimX=X, DimY=Y, DimZ=Z, FramingCubeDim=FRAME,
            OffsetX=offX, OffsetY=offY, OffsetZ=offZ,
            centering_check=centering_check,
            vmr=vmr_path, v16=v16_path)
with open(os.path.join(ANAT, "vmr_build_v2.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("wrote", vmr_path)
print("wrote", v16_path)
