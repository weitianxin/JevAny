#!/usr/bin/env bash
set -euo pipefail

: "${RL_SUITE:?Set RL_SUITE to the 40,000-record hard-choice JevAny v2 RLCR suite directory}"
: "${SFT:?Set SFT to the selected JevAny SFT checkpoint}"
: "${EVAL_SUITE:?Set EVAL_SUITE to the JevAny v2 SFT suite directory}"
: "${TRANSFER_SUITE:?Set TRANSFER_SUITE to transfer-v9}"

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
OUT=${OUT:-runs/jevany-27b-rlcr}
BASE=${BASE:-Qwen/Qwen3.8-27B}
PYTHON_BIN=${PYTHON_BIN:-python}
EPOCHS=${EPOCHS:-1}
LR=${LR:-2e-6}
HEAD_LR=${HEAD_LR:-$LR}
RLCR_GROUP_SIZE=${RLCR_GROUP_SIZE:-32}
RLCR_SIGMA_START=${RLCR_SIGMA_START:-0.4}
RLCR_SIGMA_END=${RLCR_SIGMA_END:-0.2}
RLCR_POLICY_W=${RLCR_POLICY_W:-0.25}
RLCR_CE_W=${RLCR_CE_W:-0.5}
EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS:-200}
CHECKPOINT_EVERY_STEPS=${CHECKPOINT_EVERY_STEPS:-$EVAL_EVERY_STEPS}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-5}
BASE_LOAD_ARGS=()
if [[ -n "${BASE_LOAD_PATH:-}" ]]; then
  BASE_LOAD_ARGS+=(--base_load_path "$BASE_LOAD_PATH")
fi

if (( NNODES > 1 )) && [[ "$MASTER_ADDR" == 127.0.0.1 ]]; then
  echo "Set MASTER_ADDR to the rank-0 host for multi-node training." >&2
  exit 2
fi

WANDB_ARGS=()
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  WANDB_ARGS+=(--wandb_project "$WANDB_PROJECT" --wandb_name "${WANDB_NAME:-jevany-27b-rlcr}" --wandb_mode "${WANDB_MODE:-online}")
fi

"$PYTHON_BIN" -m torch.distributed.run \
  --nnodes "$NNODES" --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
  --nproc_per_node "$GPUS_PER_NODE" \
  -m jevany.train \
  --base "$BASE" "${BASE_LOAD_ARGS[@]}" \
  --base_revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --suite "$RL_SUITE" --init_from "$SFT" \
  --eval_suite "$EVAL_SUITE" --eval_transfer_suite "$TRANSFER_SUITE" \
  --multimodal \
  --device cuda \
  --epochs "$EPOCHS" --lr "$LR" --head_lr "$HEAD_LR" --weight_decay 0.01 \
  --lora 16 --lora_targets all --head_dim 256 \
  --rlcr --rlcr_group_size "$RLCR_GROUP_SIZE" \
  --rlcr_sigma_start "$RLCR_SIGMA_START" --rlcr_sigma_end "$RLCR_SIGMA_END" \
  --rlcr_policy_w "$RLCR_POLICY_W" --rlcr_ce_w "$RLCR_CE_W" \
  --batch 1 --accum 1 --dtype bf16 --weights_dtype bf16 --checkpointing 1 \
  --p_none 0 --p_none_distract 0 --p_distract 0 --p_none_pair 0 \
  --eval_before_start --eval_every_steps "$EVAL_EVERY_STEPS" \
  --checkpoint_every_steps "$CHECKPOINT_EVERY_STEPS" \
  --early_stop_patience "$EARLY_STOP_PATIENCE" --early_stop_metric transfer_acc \
  --out "$OUT" "${WANDB_ARGS[@]}"
