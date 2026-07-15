#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${FASTWAM_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
RUN_NAME="$(basename "${SCRIPT_DIR}")"
RUN_TRAIN="${SCRIPT_DIR}/run_train.sh"

cd "${REPO_ROOT}"

source /data/lmz/miniconda3/etc/profile.d/conda.sh
conda activate /data/lmz/miniconda3/envs/fastwam
export PATH="/data/lmz/miniconda3/envs/fastwam/bin:${PATH}"
export ACCELERATE_BIN="/data/lmz/miniconda3/envs/fastwam/bin/accelerate"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_ROOT}/checkpoints}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-modelscope}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"


export HF_HOME="${HF_HOME:-/data/lmz/hf_home}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

NUM_MACHINES="${WORLD_SIZE:-${NNODES:-1}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MACHINE_RANK="${RANK:-${NODE_RANK:-0}}"
if (( NUM_MACHINES > 1 )); then
  MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR is required for multi-node training}"
else
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
MASTER_PORT="${MASTER_PORT:-23456}"
TOTAL_GPUS=$((NUM_MACHINES * NPROC_PER_NODE))

export NNODES="${NUM_MACHINES}"
export NODE_RANK="${MACHINE_RANK}"
export MASTER_ADDR
export MASTER_PORT
export NPROC_PER_NODE

export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1000}"

EXPLICIT_WANDB_MODE="${WANDB_MODE:-}"
STARGVLA_ENV_FILE="${STARGVLA_ENV_FILE:-/data/lmz/code/starVLA-A/.env}"
if [[ -f "${STARGVLA_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${STARGVLA_ENV_FILE}"
  set +a
fi
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi
if [[ -n "${EXPLICIT_WANDB_MODE}" ]]; then
  export WANDB_MODE="${EXPLICIT_WANDB_MODE}"
else
  export WANDB_MODE="${WANDB_MODE:-offline}"
fi

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STDOUT_LOG="${LOG_DIR}/${RUN_NAME}-multinode-rank${MACHINE_RANK}-${TIMESTAMP}.out"
STDERR_LOG="${LOG_DIR}/${RUN_NAME}-multinode-rank${MACHINE_RANK}-${TIMESTAMP}.err"

exec > >(tee -a "${STDOUT_LOG}") 2> >(tee -a "${STDERR_LOG}" >&2)

echo "Run name: ${RUN_NAME}"
echo "Repo root: ${REPO_ROOT}"
echo "Run script: ${RUN_TRAIN}"
echo "AIHC_JOB_NAME=${AIHC_JOB_NAME:-}"
echo "WORLD_SIZE/NNODES=${NUM_MACHINES}  RANK/NODE_RANK=${MACHINE_RANK}  NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "TOTAL_GPUS=${TOTAL_GPUS}  MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}"
echo "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-}"
echo "NCCL_IB_HCA=${NCCL_IB_HCA:-}"
echo "DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH}"
echo "DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD}"
echo "TRAIN_ZERO_SCRIPT=${TRAIN_ZERO_SCRIPT:-scripts/train_zero2.sh}"
echo "ACCELERATE_CONFIG_FILE=${ACCELERATE_CONFIG_FILE:-scripts/accelerate_configs/accelerate_zero2_ds.yaml}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "WANDB_ENABLED=${WANDB_ENABLED:-true}"
echo "WANDB_PROJECT=${WANDB_PROJECT:-starVLA-RoboTwin-Astribot}"
echo "WANDB_WORKSPACE=${WANDB_WORKSPACE:-${WANDB_ENTITY:-leeibo-beihang-university}}"
echo "Stdout log: ${STDOUT_LOG}"
echo "Stderr log: ${STDERR_LOG}"

bash "${RUN_TRAIN}" "$@"
