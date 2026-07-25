#!/bin/bash
# Apply the previously estimated topup field to the high-pass-filtered run.
# The topup field is estimated from the raw AP/PA b0 volumes only, so it is
# unaffected by downstream processing choices upstream of this step and can
# be safely reused; only the applytopup target and output are produced here.
set -e
: "${FSLDIR:?FSLDIR must be set to your FSL installation directory}"
export PATH="$FSLDIR/bin:$PATH"
source $FSLDIR/etc/fslconf/fsl.sh

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
FUNC="$PROJECT_ROOT/derivatives/sub-01/func"
ACQ=$FUNC/acquisition_params.txt
RUN=${1:?run number required}
APPLY_INPUT=$FUNC/sub-01_run-${RUN}_v2_stage-05hpf.nii.gz
OUT=$FUNC/sub-01_run-${RUN}_topup   # reuse the existing topup field estimate

echo "=== applytopup run${RUN} start $(date +%H:%M:%S) ==="
applytopup --imain=$APPLY_INPUT \
           --datain=$ACQ --inindex=1 --topup=$OUT \
           --method=jac --interp=spline \
           --out=$FUNC/sub-01_run-${RUN}_v2_stage-06dc-topup.nii.gz
echo "APPLYTOPUP_DONE_run${RUN} $(date +%H:%M:%S)"
