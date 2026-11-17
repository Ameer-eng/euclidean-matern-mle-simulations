#!/bin/bash
# AUDIT: single-thread timing (same code as the 4-thread results/bench JSONs)
source /etc/profile 2>/dev/null; source slurm/env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
for spec in "5 0.5 10" "4 0.5 18" "3 0.5 46"; do set -- $spec
  python -m matern_sim.run_task --d $1 --nu0 $2 --G $3 --bench --out audit/out/bench_1thread 2>&1 | tail -1
done
echo ALLDONE
