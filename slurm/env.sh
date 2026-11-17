# Sourced by every Slurm job (and usable interactively): Python env + thread pinning.
module purge >/dev/null 2>&1 || true
module load python/3.12.4
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$PROJECT_DIR/.venv/bin/activate"
NT="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT NUMBA_NUM_THREADS=$NT
# Idle OpenBLAS threads (numpy's and scipy's pools) otherwise spin ~0.1 s after every
# call and starve numba's threads on the job's few cores: measured 3x (d=3, G=46) to
# 180x (d=4, G=8) slowdowns.  2^4 cycles of spinning instead; numerics are unchanged.
export OPENBLAS_THREAD_TIMEOUT=4
# node-local numba cache (avoids races between concurrent jobs on the shared filesystem)
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${USER}_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"
export PYTHONUNBUFFERED=1
cd "$PROJECT_DIR"
