#!/bin/bash
#SBATCH --job-name=deepop-test
#SBATCH --output=output/hpc_logs/%x_%j.out
#SBATCH --error=output/hpc_logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# SBATCH --account=<your-project-account>

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p output/hpc_logs

source scripts/hpc/coaraci/env_coaraci.sh

PROBLEM=${PROBLEM:-ground_vibration}
EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG:-./configs/problems/${PROBLEM}/config_experiment.yaml}
TEST_CONFIG=${TEST_CONFIG:-./configs/problems/${PROBLEM}/config_test.yaml}

python3 main.py \
  --problem "${PROBLEM}" \
  --test \
  --experiment-config-path "${EXPERIMENT_CONFIG}" \
  --test-config-path "${TEST_CONFIG}"
