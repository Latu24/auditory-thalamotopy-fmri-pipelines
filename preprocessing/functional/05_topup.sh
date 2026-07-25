#!/bin/bash
# FSL topup + applytopup distortion correction.
# Readout time derived from JSON sidecar: TotalReadoutTime = 0.0729774 s
#   (BandwidthPerPixelPhaseEncode=13.605 Hz/px, ReconMatrixPE=140).
# PE: AP = j- => "0 -1 0 RT" ; PA = j+ => "0 1 0 RT" (from PhaseEncodingDirection).
# imain = AP_first5(cut) ++ PA_first5(cut)  (10 vols). acqparams rows match.
# config: topup_config_7T.cnf (FSL b02b0 config tuned for 7T fieldmap
#   estimation); if it is not present alongside this script, FSL's stock
#   b02b0_7T.cnf is copied in as a fallback.
# applytopup target = full processed AP run.
set -e
: "${FSLDIR:?FSLDIR must be set to your FSL installation directory}"
export PATH="$FSLDIR/bin:$PATH"
source $FSLDIR/etc/fslconf/fsl.sh

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
FUNC="$PROJECT_ROOT/derivatives/sub-01/func"
CFGDIR="$PROJECT_ROOT/scripts/functional"
LOG="$PROJECT_ROOT/logs"
RT=0.0729774
RUN=${1:-1}
APPLY_INPUT=${2:-$FUNC/sub-01_run-${RUN}_stage-05hpf.nii.gz}

# --- acquisition_params.txt (5 AP rows, 5 PA rows) ---
ACQ=$FUNC/acquisition_params.txt
{ for i in 1 2 3 4 5; do echo "0 -1 0 $RT"; done
  for i in 1 2 3 4 5; do echo "0 1 0 $RT"; done ; } > $ACQ
echo "acqparams:"; cat $ACQ

# --- config substitution ---
CFG=$CFGDIR/topup_config_7T.cnf
if [ ! -f "$CFG" ]; then cp $FSLDIR/etc/flirtsch/b02b0_7T.cnf $CFG; fi

# --- merge AP + PA b0 (imain) ---
IMAIN=$FUNC/sub-01_run-${RUN}_AP-PA_b0merged.nii.gz
fslmerge -t $IMAIN \
  $FUNC/sub-01_run-${RUN}_AP_b0-first5_cut.nii.gz \
  $FUNC/sub-01_dir-PA_b0-first5_cut.nii.gz
echo "imain dims:"; fslhd $IMAIN | grep -E "^dim[1-4]"

# --- topup (estimate field) ---
OUT=$FUNC/sub-01_run-${RUN}_topup
echo "=== running topup run${RUN} (may take a while) ==="
topup --imain=$IMAIN --datain=$ACQ --config=$CFG \
      --out=$OUT \
      --fout=${OUT}_field.nii.gz \
      --iout=${OUT}_bunwarped.nii.gz \
      --logout=$LOG/topup_run-${RUN}.log -v 2>&1 | tail -5
echo "TOPUP_ESTIMATE_DONE_run${RUN}"

# --- applytopup to full processed AP run (method=jac) ---
applytopup --imain=$APPLY_INPUT \
           --datain=$ACQ --inindex=1 --topup=$OUT \
           --method=jac --interp=spline \
           --out=$FUNC/sub-01_run-${RUN}_stage-06dc-topup.nii.gz
echo "APPLYTOPUP_DONE_run${RUN}"
