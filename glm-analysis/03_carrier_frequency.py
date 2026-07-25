"""GLM 3 — Carrier frequency (36 conditions, Freq_01..Freq_36).

Fixed-effects multi-run GLM. Outputs an omnibus F, an all-tones response, a
voxel-level best-frequency map (a tonotopy proxy, not layer-resolved), and
an ordinal low-to-high frequency parametric contrast. See event_family.py.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import glmlib as G
import common as C
import event_family as E

if __name__ == "__main__":
    hrf = G.two_gamma_hrf()
    E.run_family("carrierFrequency", C.BYFREQ, C.FREQ_ORDER, C.VTCS, hrf,
                 parametric_name="frequency")
