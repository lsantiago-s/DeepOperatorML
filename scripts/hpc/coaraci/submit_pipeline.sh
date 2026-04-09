#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/queue_profiles.sh"

QUEUE="gpu-x"
TIME=""
ACCOUNT=""
QOS=""
JOB_NAME=""
MANIFEST="./configs/benchmarks/paper_ready_benchmark.yaml"
PROBLEM=""
MODEL_TRACK="don"
STAGE="all"
CPUS_PER_TASK="1"
CPUS_PER_TASK_EXPLICIT="0"
NTASKS_PER_NODE=""
GPUS="3"
FREEZE_DATASET="1"
FREEZE_EXPERIMENT="1"
SKIP_LIMIT_CHECK="0"
DRY_RUN="0"

usage() {
  cat <<'EOF'
Usage:
  scripts/hpc/coaraci/submit_pipeline.sh [options]

Options:
  --queue <par48-x|par480-x|gpu-x>   Queue/partition (default: gpu-x)
  --time <HH:MM:SS>                  Walltime (default: queue maximum)
  --account <name>                   Slurm account/project
  --qos <name>                       Slurm QoS
  --job-name <name>                  Job name (default: deepop-pipeline-<queue>)
  --manifest <path>                  Benchmark manifest
  --problem <name>                   Optional single problem (default: all enabled)
  --model-track <don|fno>            Model track (default: don)
  --stage <gen|preprocess|train|test|all>  Stage (default: all)
  --cpus-per-task <int>              CPUs per task (default: 1)
  --ntasks-per-node <int>            Tasks per node (default: 1 on gpu-x, queue profile otherwise)
  --gpus <int>                       GPUs for gpu-x queue (default: 3)
  --no-freeze-dataset                Disable --freeze-dataset-version
  --no-freeze-experiment             Disable --freeze-experiment-version
  --skip-limit-check                 Skip per-queue active-job limit check
  --dry-run                          Print sbatch command and generated job script
  -h, --help                         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --queue) QUEUE="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --problem) PROBLEM="$2"; shift 2 ;;
    --model-track) MODEL_TRACK="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --cpus-per-task) CPUS_PER_TASK="$2"; CPUS_PER_TASK_EXPLICIT="1"; shift 2 ;;
    --ntasks-per-node) NTASKS_PER_NODE="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --no-freeze-dataset) FREEZE_DATASET="0"; shift ;;
    --no-freeze-experiment) FREEZE_EXPERIMENT="0"; shift ;;
    --skip-limit-check) SKIP_LIMIT_CHECK="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

coaraci_set_profile "${QUEUE}"

if [[ -z "${TIME}" ]]; then
  TIME="${COARACI_WALLTIME_MAX}"
fi
coaraci_validate_walltime "${TIME}" "${COARACI_WALLTIME_MAX}"

if [[ "${SKIP_LIMIT_CHECK}" != "1" && "${DRY_RUN}" != "1" ]]; then
  coaraci_check_job_limit "${COARACI_PARTITION}" "${COARACI_MAX_JOBS}"
fi

if [[ -z "${JOB_NAME}" ]]; then
  JOB_NAME="deepop-pipeline-${QUEUE}"
fi

if [[ -z "${NTASKS_PER_NODE}" ]]; then
  if [[ "${COARACI_IS_GPU}" == "1" ]]; then
    NTASKS_PER_NODE="1"
  else
    NTASKS_PER_NODE="${COARACI_NTASKS_PER_NODE}"
  fi
fi

if [[ "${CPUS_PER_TASK_EXPLICIT}" != "1" && "${PROBLEM}" == "ground_vibration" ]]; then
  if [[ "${STAGE}" == "gen" || "${STAGE}" == "all" ]]; then
    CPUS_PER_TASK="16"
  fi
fi

if [[ "${COARACI_IS_GPU}" == "1" ]]; then
  if (( GPUS < 1 || GPUS > COARACI_GPUS_PER_NODE_MAX )); then
    echo "For gpu-x queue, --gpus must be between 1 and ${COARACI_GPUS_PER_NODE_MAX}." >&2
    exit 1
  fi
  if (( NTASKS_PER_NODE * CPUS_PER_TASK > GPUS * 16 )); then
    echo "gpu-x allows at most 16 CPUs per requested GPU. Current request is $((NTASKS_PER_NODE * CPUS_PER_TASK)) CPUs for ${GPUS} GPU(s)." >&2
    echo "Adjust --ntasks-per-node, --cpus-per-task, or --gpus." >&2
    exit 1
  fi
else
  if [[ "${GPUS}" != "0" && "${GPUS}" != "" ]]; then
    echo "Ignoring --gpus for CPU queue ${QUEUE}." >&2
  fi
  GPUS="0"
fi

cd "${REPO_ROOT}"
mkdir -p output/hpc_logs

PIPE_CMD=(
  python3 scripts/benchmark/run_benchmark_pipeline.py
  --manifest "${MANIFEST}"
  --model-track "${MODEL_TRACK}"
  --stage "${STAGE}"
)

if [[ -n "${PROBLEM}" ]]; then
  PIPE_CMD+=(--problem "${PROBLEM}")
fi
if [[ "${FREEZE_DATASET}" == "1" ]]; then
  PIPE_CMD+=(--freeze-dataset-version)
fi
if [[ "${FREEZE_EXPERIMENT}" == "1" ]]; then
  PIPE_CMD+=(--freeze-experiment-version)
fi

JOB_SCRIPT="$(mktemp "/tmp/deepop_pipeline_${QUEUE}_XXXXXX")"
{
  echo '#!/bin/bash'
  echo 'set -euo pipefail'
  echo "cd \"${REPO_ROOT}\""
  echo 'source scripts/hpc/coaraci/env_coaraci.sh'
  printf '%q ' "${PIPE_CMD[@]}"
  echo
} > "${JOB_SCRIPT}"
chmod +x "${JOB_SCRIPT}"

SBATCH_ARGS=(
  --job-name "${JOB_NAME}"
  --output "output/hpc_logs/%x_%j.out"
  --error "output/hpc_logs/%x_%j.err"
  --partition "${COARACI_PARTITION}"
  --nodes "${COARACI_NODES}"
  --ntasks-per-node "${NTASKS_PER_NODE}"
  --cpus-per-task "${CPUS_PER_TASK}"
  --time "${TIME}"
)

if [[ -n "${ACCOUNT}" ]]; then
  SBATCH_ARGS+=(--account "${ACCOUNT}")
fi
if [[ -n "${QOS}" ]]; then
  SBATCH_ARGS+=(--qos "${QOS}")
fi
if [[ "${COARACI_IS_GPU}" == "1" ]]; then
  SBATCH_ARGS+=(--gres "gpu:${GPUS}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "sbatch ${SBATCH_ARGS[*]} ${JOB_SCRIPT}"
  echo "--- job script ---"
  cat "${JOB_SCRIPT}"
  rm -f "${JOB_SCRIPT}"
  exit 0
fi

sbatch "${SBATCH_ARGS[@]}" "${JOB_SCRIPT}"
rm -f "${JOB_SCRIPT}"
