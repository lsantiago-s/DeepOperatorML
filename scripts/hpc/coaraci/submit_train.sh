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
PROBLEM="ground_vibration"
TRAIN_CONFIG="./configs/training/config_don_train.yaml"
EXPERIMENT_CONFIG=""
TEST_CONFIG=""
CPUS_PER_TASK="1"
GPUS="3"
SKIP_LIMIT_CHECK="0"
DRY_RUN="0"

usage() {
  cat <<'EOF'
Usage:
  scripts/hpc/coaraci/submit_train.sh [options]

Options:
  --queue <par48-x|par480-x|gpu-x>   Queue/partition (default: gpu-x)
  --time <HH:MM:SS>                  Walltime (default: queue maximum)
  --account <name>                   Slurm account/project
  --qos <name>                       Slurm QoS
  --job-name <name>                  Job name (default: deepop-train-<queue>)
  --problem <name>                   Problem name (default: ground_vibration)
  --train-config <path>              Training config path
  --experiment-config <path>         Experiment config path (default by problem)
  --test-config <path>               Test config path (default by problem)
  --cpus-per-task <int>              CPUs per task (default: 1)
  --gpus <int>                       GPUs for gpu-x queue (default: 3)
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
    --problem) PROBLEM="$2"; shift 2 ;;
    --train-config) TRAIN_CONFIG="$2"; shift 2 ;;
    --experiment-config) EXPERIMENT_CONFIG="$2"; shift 2 ;;
    --test-config) TEST_CONFIG="$2"; shift 2 ;;
    --cpus-per-task) CPUS_PER_TASK="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
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

if [[ "${SKIP_LIMIT_CHECK}" != "1" ]]; then
  coaraci_check_job_limit "${COARACI_PARTITION}" "${COARACI_MAX_JOBS}"
fi

if [[ -z "${JOB_NAME}" ]]; then
  JOB_NAME="deepop-train-${QUEUE}"
fi

if [[ -z "${EXPERIMENT_CONFIG}" ]]; then
  EXPERIMENT_CONFIG="./configs/problems/${PROBLEM}/config_experiment.yaml"
fi
if [[ -z "${TEST_CONFIG}" ]]; then
  TEST_CONFIG="./configs/problems/${PROBLEM}/config_test.yaml"
fi

if [[ "${COARACI_IS_GPU}" == "1" ]]; then
  if (( GPUS < 1 || GPUS > COARACI_GPUS_PER_NODE_MAX )); then
    echo "For gpu-x queue, --gpus must be between 1 and ${COARACI_GPUS_PER_NODE_MAX}." >&2
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

JOB_SCRIPT="$(mktemp "/tmp/deepop_train_${QUEUE}_XXXXXX")"
cat > "${JOB_SCRIPT}" <<EOF
#!/bin/bash
set -euo pipefail
cd "${REPO_ROOT}"
source scripts/hpc/coaraci/env_coaraci.sh
python3 main.py \
  --problem "${PROBLEM}" \
  --train-config-path "${TRAIN_CONFIG}" \
  --experiment-config-path "${EXPERIMENT_CONFIG}" \
  --test-config-path "${TEST_CONFIG}"
EOF
chmod +x "${JOB_SCRIPT}"

SBATCH_ARGS=(
  --job-name "${JOB_NAME}"
  --output "output/hpc_logs/%x_%j.out"
  --error "output/hpc_logs/%x_%j.err"
  --partition "${COARACI_PARTITION}"
  --nodes "${COARACI_NODES}"
  --ntasks-per-node "${COARACI_NTASKS_PER_NODE}"
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
