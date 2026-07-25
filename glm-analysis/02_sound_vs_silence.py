"""GLM 2 — Sound vs silence (SoundOn block design).

Produces per-run t-maps and a multi-run fixed-effects (FFX) t-map. FFX is
the appropriate choice here: this is a single subject's 4 runs, so runs are
not a random sample of a population — the analysis is not generalizing
across subjects (that would call for RFX). The concatenated multi-run GLM
with per-run baseline confounds gives the FFX sound>baseline test.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import glmlib as G
import common as C

def run():
    hrf = G.two_gamma_hrf()
    anatf = np.load("/tmp/anatf.npy")
    # ---- build per-run single-condition designs ----
    designs = []
    for prt, vtc in zip(C.SOUNDON, C.VTCS):
        nv = G.read_vtc_header(vtc)["DimT"]
        X, names = G.build_run_design(prt, nv, hrf, condition_order=["SoundOn"])
        designs.append(X)

    log = {"analysis": "sound_vs_silence", "design": "block, 1 condition",
           "effects": "fixed-effects multi-run + per-run", "runs": {}}

    # ---- per-run fits ----
    perrun_t = []
    for i, (vtc, X) in enumerate(zip(C.VTCS, designs), start=1):
        fit = G.fit_multirun_glm([vtc], [X], hrf, log=lambda *_: None)
        c = np.zeros(fit["P"]); c[0] = 1.0
        t = G.t_contrast(fit, c)
        perrun_t.append(t)
        log["runs"][f"run{i}"] = {
            "dof": fit["dof"], "phi_ar2": fit["phi"],
            "peak_t": float(np.max(t)),
            "n_t_gt_3": int((t > 3).sum()),
            "frac_mask_t_gt_3": float((t > 3).sum() / fit["mask"].sum()),
        }
        print(f"run{i}: peak t={np.max(t):.2f}, "
              f"t>3 voxels={int((t>3).sum())}")

    # ---- multi-run FFX fit ----
    fit = G.fit_multirun_glm(C.VTCS, designs, hrf, log=print)
    c = np.zeros(fit["P"]); c[0] = 1.0
    t_ffx = G.t_contrast(fit, c)
    log["ffx"] = {
        "dof": fit["dof"], "phi_ar2": fit["phi"],
        "Ttotal": fit["Ttotal"], "P": fit["P"],
        "peak_t": float(np.max(t_ffx)),
        "peak_voxel_zyx": [int(x) for x in
                           np.unravel_index(np.argmax(t_ffx), t_ffx.shape)],
        "n_t_gt_5": int((t_ffx > 5).sum()),
        "frac_mask_t_gt_5": float((t_ffx > 5).sum() / fit["mask"].sum()),
    }
    print(f"FFX: peak t={np.max(t_ffx):.2f}, dof={fit['dof']}, "
          f"t>5 voxels={int((t_ffx>5).sum())}")

    # ---- write VMP: 5 sub-maps (FFX + 4 runs) ----
    hdr = fit["headers"][0]
    maps = [{"name": "SoundOn>baseline FFX (t)", "data": t_ffx, "type": 1,
             "df1": fit["dof"], "threshold": 5.0, "upper": 15.0}]
    for i, t in enumerate(perrun_t, start=1):
        maps.append({"name": f"SoundOn>baseline run{i} (t)", "data": t,
                     "type": 1, "df1": log["runs"][f"run{i}"]["dof"],
                     "threshold": 3.0, "upper": 12.0})
    out = f"{C.ANA}/sub-01_soundVsSilence.vmp"
    G.write_stat_vmp(out, maps, hdr,
                     vtc_name="sub-01_run-1_preproc_coreg.vtc",
                     prt_name="run1_SoundOn_timestamp.prt")
    log["vmp"] = out
    log["vmp_submaps"] = [m["name"] for m in maps]
    C.save_json(f"{C.LOGS}/analyses_02_sound_vs_silence.json", log)
    C.overlay_tmap(t_ffx, anatf, f"{C.QC}/soundVsSilence_ffx_tmap.png",
                   "SoundOn>base FFX t", thr=5.0, vmax=15)
    print("wrote", out)

if __name__ == "__main__":
    run()
