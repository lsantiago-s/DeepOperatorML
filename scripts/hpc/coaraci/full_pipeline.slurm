#!/bin/bash
#SBATCH --job-name=deepop-pipeline
#SBATCH --output=output/hpc_logs/%x_%j.out
#SBATCH --error=output/hpc_logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# SBATCH --account=<your-project-account>

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p output/hpc_logs

source scripts/hpc/coaraci/env_coaraci.sh

MANIFEST=${MANIFEST:-./configs/benchmarks/paper_ready_benchmark.yaml}
PROBLEM=${PROBLEM:-ground_vibration}
MODEL_TRACK=${MODEL_TRACK:-don}

python3 scripts/benchmark/run_benchmark_pipeline.py \
  --manifest "${MANIFEST}" \
  --problem "${PROBLEM}" \
  --model-track "${MODEL_TRACK}" \
  --stage all \
  --freeze-dataset-version \
  --freeze-experiment-version
