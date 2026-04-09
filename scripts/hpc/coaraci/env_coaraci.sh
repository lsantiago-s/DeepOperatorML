#!/bin/bash
set -euo pipefail

# Adjust these commands to Coaraci's module/environment stack.
# Example placeholders:
# module purge
# module load cuda/12.1
# module load python/3.11

# Activate your environment
# source "$HOME/miniconda3/etc/profile.d/conda.sh"
# conda activate deepop

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

# Keep cache on local scratch if needed
# export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$PWD/.cache}

python3 -V
