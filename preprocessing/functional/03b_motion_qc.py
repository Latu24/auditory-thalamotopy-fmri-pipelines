#!/usr/bin/env python3
"""Motion QC from mcflirt .par files. Flags volumes with excessive
displacement and checks for abrupt jumps between runs.

.par columns are Rx Ry Rz (radians), Tx Ty Tz (mm). Voxel size is 0.9mm
isotropic, so the flag threshold of 3 voxels corresponds to 2.7mm.
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FUNC = os.path.join(PROJECT_ROOT, "derivatives", "sub-01", "func")
LOG = os.path.join(PROJECT_ROOT, "logs")
RUNS = [1, 2, 3, 4]
VOX_MM = 0.9
FLAG_MM = 3 * VOX_MM   # 2.7 mm

def main():
    qc = {}; run_mean_trans = {}
    fig, axes = plt.subplots(len(RUNS), 2, figsize=(12, 9))
    for i, r in enumerate(RUNS):
        par = np.loadtxt(f"{FUNC}/sub-01_run-{r}_stage-04mc.par")
        rot_deg = np.degrees(par[:, 0:3]); trans = par[:, 3:6]
        # framewise displacement (Power): sum |dtrans| + 50mm*|drot(rad)|
        d = np.abs(np.diff(par, axis=0))
        fd = d[:, 3:6].sum(1) + 50.0 * d[:, 0:3].sum(1)
        # absolute displacement magnitude per volume (translation only, vs ref)
        trans_mag = np.sqrt((trans ** 2).sum(1))
        flagged = np.where(trans_mag > FLAG_MM)[0]
        run_mean_trans[r] = trans.mean(0)
        qc[r] = dict(
            nvol=int(par.shape[0]),
            max_abs_trans_mm={"X": float(np.abs(trans[:,0]).max()),
                               "Y": float(np.abs(trans[:,1]).max()),
                               "Z": float(np.abs(trans[:,2]).max())},
            max_abs_rot_deg={"pitch": float(np.abs(rot_deg[:,0]).max()),
                              "yaw": float(np.abs(rot_deg[:,1]).max()),
                              "roll": float(np.abs(rot_deg[:,2]).max())},
            max_trans_mag_mm=float(trans_mag.max()),
            mean_FD_mm=float(fd.mean()), max_FD_mm=float(fd.max()),
            n_vols_over_3vox=int(len(flagged)),
            flagged_vols=flagged.tolist()[:30],
        )
        ax = axes[i, 0]
        ax.plot(trans[:,0], label="X"); ax.plot(trans[:,1], label="Y"); ax.plot(trans[:,2], label="Z")
        ax.axhline(FLAG_MM, ls=":", c="r", lw=0.6); ax.axhline(-FLAG_MM, ls=":", c="r", lw=0.6)
        ax.set_ylabel(f"run{r} trans mm");
        if i == 0: ax.legend(fontsize=7, ncol=3)
        ax2 = axes[i, 1]
        ax2.plot(rot_deg[:,0], label="pitch"); ax2.plot(rot_deg[:,1], label="yaw"); ax2.plot(rot_deg[:,2], label="roll")
        ax2.set_ylabel(f"run{r} rot deg")
        if i == 0: ax2.legend(fontsize=7, ncol=3)
    axes[-1,0].set_xlabel("volume"); axes[-1,1].set_xlabel("volume")
    fig.suptitle("Motion parameters (aligned to run1 vol0); red dotted = 3-voxel (2.7mm) flag")
    fig.tight_layout(); fig.savefig(f"{LOG}/qc_motion_allruns.png", dpi=95); plt.close(fig)

    # between-run jumps: change in mean translation between consecutive runs
    jumps = {}
    for a, b in [(1,2),(2,3),(3,4)]:
        j = np.linalg.norm(run_mean_trans[b] - run_mean_trans[a])
        jumps[f"run{a}->run{b}"] = float(j)
    qc["between_run_mean_trans_jump_mm"] = jumps
    qc["run_mean_trans_mm"] = {r: run_mean_trans[r].tolist() for r in RUNS}
    json.dump(qc, open(f"{LOG}/qc_motion.json", "w"), indent=2)
    for r in RUNS:
        print(f"run{r}: maxTransMag={qc[r]['max_trans_mag_mm']:.2f}mm "
              f"maxFD={qc[r]['max_FD_mm']:.2f}mm >3vox={qc[r]['n_vols_over_3vox']}")
    print("between-run jumps (mm):", jumps)
    print("wrote qc_motion.json + qc_motion_allruns.png")

if __name__ == "__main__":
    main()
