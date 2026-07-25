#!/bin/bash
# Motion correction via FSL mcflirt, run on the per-run slice-time-corrected
# data. Reference: run-1 slice-time-corrected volume 0, used as the common
# target for all 4 runs so they end up on one shared post-motion-correction
# grid.
set -e
: "${FSLDIR:?FSLDIR must be set to your FSL installation directory}"
export PATH="$FSLDIR/bin:$PATH"
source $FSLDIR/etc/fslconf/fsl.sh

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
FUNC="$PROJECT_ROOT/derivatives/sub-01/func"
LOG="$PROJECT_ROOT/logs"

# common reference = run1 STC vol0
fslroi $FUNC/sub-01_run-1_v2_stage-03stc.nii.gz $FUNC/ref_run1_vol0_v2.nii.gz 0 1

for r in 1 2 3 4; do
  echo "=== mcflirt run$r (ref=run1 vol0, sinc final) ==="
  mcflirt -in  $FUNC/sub-01_run-${r}_v2_stage-03stc.nii.gz \
          -out $FUNC/sub-01_run-${r}_v2_stage-04mc \
          -reffile $FUNC/ref_run1_vol0_v2.nii.gz \
          -sinc_final -plots -report -rmsrel -rmsabs
  cp $FUNC/sub-01_run-${r}_v2_stage-04mc.par $LOG/motion_run-${r}_v2.par
done
echo "MOTION_CORRECTION_DONE"
