#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${FASTWAM_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  NPROC_PER_NODE="$1"
  shift
else
  NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
fi

TASK_NAME="${TASK_NAME:-astribot_uncond_1cam_384_1e-4}"
TRAIN_ZERO_SCRIPT="${TRAIN_ZERO_SCRIPT:-scripts/train_zero2.sh}"
ACTION_DIT_PATH="${ACTION_DIT_PATH:-checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim_astribot18.pt}"
TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-data/text_embeds_cache/astribot}"
MIN_TEXT_EMBED_FILES="${MIN_TEXT_EMBED_FILES:-1470}"
WANDB_WORKSPACE="${WANDB_WORKSPACE:-${WANDB_ENTITY:-leeibo-beihang-university}}"
WANDB_PROJECT="${WANDB_PROJECT:-starVLA-RoboTwin-Astribot}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${AIHC_JOB_NAME:-fastwam_${TASK_NAME}}}"
WANDB_GROUP="${WANDB_GROUP:-fastwam_astribot}"

if [[ ! -f "${ACTION_DIT_PATH}" ]]; then
  echo "Missing ActionDiT backbone: ${ACTION_DIT_PATH}" >&2
  echo "Run scripts/preprocess_action_dit_backbone.py for the Astribot 18-dim config first." >&2
  exit 1
fi

if [[ ! -d "${TEXT_EMBED_CACHE_DIR}" ]]; then
  echo "Missing text embedding cache directory: ${TEXT_EMBED_CACHE_DIR}" >&2
  echo "Run scripts/precompute_text_embeds.py task=${TASK_NAME} first." >&2
  exit 1
fi

TEXT_EMBED_COUNT="$(find "${TEXT_EMBED_CACHE_DIR}" -maxdepth 1 -type f -name '*.pt' | wc -l)"
if (( TEXT_EMBED_COUNT < MIN_TEXT_EMBED_FILES )); then
  echo "Text embedding cache looks incomplete: ${TEXT_EMBED_COUNT} files in ${TEXT_EMBED_CACHE_DIR}, expected at least ${MIN_TEXT_EMBED_FILES}." >&2
  echo "Run scripts/precompute_text_embeds.py task=${TASK_NAME} '+overwrite=false' first." >&2
  exit 1
fi

HYDRA_OVERRIDES=(
  "task=${TASK_NAME}"
  "model.action_dit_pretrained_path=${ACTION_DIT_PATH}"
  "batch_size=${BATCH_SIZE:-1}"
  "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-4}"
  "num_workers=${NUM_WORKERS:-4}"
  "num_epochs=${NUM_EPOCHS:-5}"
  "learning_rate=${LEARNING_RATE:-1e-4}"
  "weight_decay=${WEIGHT_DECAY:-1e-2}"
  "log_every=${LOG_EVERY:-10}"
  "save_every=${SAVE_EVERY:-2500}"
  "eval_every=${EVAL_EVERY:-500}"
  "wandb.enabled=${WANDB_ENABLED:-true}"
  "wandb.workspace=${WANDB_WORKSPACE}"
  "wandb.project=${WANDB_PROJECT}"
  "wandb.name=${WANDB_RUN_NAME}"
  "wandb.group=${WANDB_GROUP}"
  "wandb.mode=${WANDB_MODE:-offline}"
)

if [[ -n "${MAX_STEPS:-}" ]]; then
  HYDRA_OVERRIDES+=("max_steps=${MAX_STEPS}")
fi

if [[ -n "${RESUME:-}" ]]; then
  HYDRA_OVERRIDES+=("resume=${RESUME}")
fi

exec bash "${TRAIN_ZERO_SCRIPT}" "${NPROC_PER_NODE}" "${HYDRA_OVERRIDES[@]}" "$@"
