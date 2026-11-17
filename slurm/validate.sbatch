#!/bin/bash
# Dense-vs-blockwise validation (log-likelihood, gradient, Fisher, kurtosis
# matrix, orthogonality of Q) on a compute node with the production software.
#   sbatch slurm/validate.sbatch
#SBATCH --job-name=validate
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR}"
source "$PROJECT_DIR/slurm/env.sh"
python -c "import numpy, scipy, numba, sys; print(sys.version); print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'numba', numba.__version__)"
python -m tests.test_validation
