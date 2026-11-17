#!/bin/bash
# AUDIT: run the multistart groups locally (output audit/out/ms_local)
source /etc/profile 2>/dev/null; source slurm/env.sh
g=$1; th=$2
export OPENBLAS_NUM_THREADS=$th OMP_NUM_THREADS=$th NUMBA_NUM_THREADS=$th
python -m audit.multistart --group $g --out audit/out/ms_local > audit/out/ms_local/log_$g.txt 2>&1
echo "DONE $?" >> audit/out/ms_local/log_$g.txt
