"""GLM 4 — Amplitude-modulation frequency (9 conditions, AM_1..AM_9).

Fixed-effects multi-run GLM. Outputs an omnibus F, an all-tones response, a
voxel-level best-AM map, and an ordinal AM-rate parametric contrast.
See event_family.py.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import glmlib as G
import common as C
import event_family as E

if __name__ == "__main__":
    hrf = G.two_gamma_hrf()
    E.run_family("amplitudeModulation", C.BYAM, C.AM_ORDER, C.VTCS, hrf,
                 parametric_name="AM-rate")
