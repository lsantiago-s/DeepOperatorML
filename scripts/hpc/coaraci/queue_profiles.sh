#!/bin/bash

# Queue profiles from Coaraci documentation provided by user.
# Exports:
# - COARACI_PARTITION
# - COARACI_NODES
# - COARACI_NTASKS_PER_NODE
# - COARACI_WALLTIME_MAX
# - COARACI_MAX_JOBS
# - COARACI_IS_GPU
# - COARACI_GPUS_PER_NODE_MAX

coaraci_set_profile() {
  local queue="${1:-}"
  case "${queue}" in
    par48-x)
      export COARACI_PARTITION="par48-x"
      export COARACI_NODES=1
      export COARACI_NTASKS_PER_NODE=48
      export COARACI_WALLTIME_MAX="72:00:00"
      export COARACI_MAX_JOBS=3
      export COARACI_IS_GPU=0
      export COARACI_GPUS_PER_NODE_MAX=0
      ;;
    par480-x)
      export COARACI_PARTITION="par480-x"
      export COARACI_NODES=10
      export COARACI_NTASKS_PER_NODE=48
      export COARACI_WALLTIME_MAX="48:00:00"
      export COARACI_MAX_JOBS=2
      export COARACI_IS_GPU=0
      export COARACI_GPUS_PER_NODE_MAX=0
      ;;
    gpu-x)
      export COARACI_PARTITION="gpu-x"
      export COARACI_NODES=1
      export COARACI_NTASKS_PER_NODE=48
      export COARACI_WALLTIME_MAX="168:00:00"
      export COARACI_MAX_JOBS=3
      export COARACI_IS_GPU=1
      export COARACI_GPUS_PER_NODE_MAX=3
      ;;
    *)
      echo "Unsupported queue '${queue}'. Use: par48-x | par480-x | gpu-x" >&2
      return 1
      ;;
  esac
}

coaraci_hms_to_seconds() {
  local hms="${1:-00:00:00}"
  local h m s
  IFS=':' read -r h m s <<< "${hms}"
  h=${h:-0}
  m=${m:-0}
  s=${s:-0}
  echo $((10#${h} * 3600 + 10#${m} * 60 + 10#${s}))
}

coaraci_validate_walltime() {
  local requested="${1:-}"
  local maximum="${2:-}"
  local req_s max_s
  req_s=$(coaraci_hms_to_seconds "${requested}")
  max_s=$(coaraci_hms_to_seconds "${maximum}")
  if (( req_s > max_s )); then
    echo "Requested walltime ${requested} exceeds queue maximum ${maximum}." >&2
    return 1
  fi
}

coaraci_check_job_limit() {
  local partition="${1:-}"
  local max_jobs="${2:-0}"

  if ! command -v squeue >/dev/null 2>&1; then
    echo "Warning: squeue not found. Skipping queue job-limit check." >&2
    return 0
  fi

  local active
  active=$(squeue -h -u "${USER}" -p "${partition}" | wc -l | tr -d ' ')
  if (( active >= max_jobs )); then
    echo "Queue '${partition}' limit reached: ${active}/${max_jobs} active jobs for user ${USER}." >&2
    return 1
  fi
}
