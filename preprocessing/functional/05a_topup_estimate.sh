#!/bin/bash
# Run topup field estimation only (the slow part). Parallelizable with
# motion correction; applytopup is done later (needs the high-pass-filtered
# output as its target).
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

ACQ=$FUNC/acquisition_params.txt
{ for i in 1 2 3 4 5; do echo "0 -1 0 $RT"; done
  for i in 1 2 3 4 5; do echo "0 1 0 $RT"; done ; } > $ACQ

CFG=$CFGDIR/topup_config_7T.cnf
[ -f "$CFG" ] || cp $FSLDIR/etc/flirtsch/b02b0_7T.cnf $CFG

IMAIN=$FUNC/sub-01_run-${RUN}_AP-PA_b0merged.nii.gz
fslmerge -t $IMAIN \
  $FUNC/sub-01_run-${RUN}_AP_b0-first5_cut.nii.gz \
  $FUNC/sub-01_dir-PA_b0-first5_cut.nii.gz

OUT=$FUNC/sub-01_run-${RUN}_topup
echo "=== topup estimate run${RUN} start $(date +%H:%M:%S) ==="
topup --imain=$IMAIN --datain=$ACQ --config=$CFG \
      --out=$OUT --fout=${OUT}_field.nii.gz --iout=${OUT}_bunwarped.nii.gz \
      --logout=$LOG/topup_run-${RUN}.log
echo "TOPUP_ESTIMATE_DONE_run${RUN} $(date +%H:%M:%S)"
