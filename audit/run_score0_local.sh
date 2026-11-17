#!/bin/bash
# AUDIT: run audit.score0_task locally for every design (largest n first), saving
# to audit/out/score0_local; data regeneration is checked against the stored
# oracle criterion, mismatches are recorded under audit/out/score0_local/mismatch.
source /etc/profile 2>/dev/null; source slurm/env.sh
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8
for spec in "5 0.5 10" "5 1.5 10" "4 0.5 18" "4 1.5 18" "3 0.5 46" "3 1.5 46" \
            "5 0.5 3" "5 0.5 4" "5 0.5 5" "5 0.5 6" "5 0.5 8" "5 1.5 3" "5 1.5 4" "5 1.5 5" "5 1.5 6" "5 1.5 8" \
            "4 0.5 3" "4 0.5 4" "4 0.5 6" "4 0.5 8" "4 0.5 10" "4 0.5 13" "4 1.5 3" "4 1.5 4" "4 1.5 6" "4 1.5 8" "4 1.5 10" "4 1.5 13" \
            "3 0.5 5" "3 0.5 7" "3 0.5 10" "3 0.5 14" "3 0.5 22" "3 0.5 32" "3 0.5 40" \
            "3 1.5 5" "3 1.5 7" "3 1.5 10" "3 1.5 14" "3 1.5 22" "3 1.5 32" "3 1.5 40"; do
  set -- $spec
  python -m audit.score0_task --d $1 --nu0 $2 --G $3 --reps 0:400 --save audit/out/score0_local 2>&1 | grep -v "^\[.*batch" | tail -3
done
echo ALLDONE
