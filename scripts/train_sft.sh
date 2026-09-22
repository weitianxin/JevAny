#!/usr/bin/env bash
set -euo pipefail

: "${SUITE:?Set SUITE to the decision-v7 directory}"
: "${TRANSFER_SUITE:?Set TRANSFER_SUITE to the transfer-v9 directory}"

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
OUT=${OUT:-runs/jevany-27b-sft}

if (( NNODES > 1 )) && [[ "$MASTER_ADDR" == 127.0.0.1 ]]; then
  echo "Set MASTER_ADDR to the rank-0 host for multi-node training." >&2
  exit 2
fi

WANDB_ARGS=()
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  WANDB_ARGS+=(--wandb_project "$WANDB_PROJECT" --wandb_name "${WANDB_NAME:-jevany-27b-sft}")
fi

torchrun \
  --nnodes "$NNODES" --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
  --nproc_per_node "$GPUS_PER_NODE" \
  -m jevany.train \
  --base Qwen/Qwen3.8-27B \
  --base_revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --suite "$SUITE" --eval_transfer_suite "$TRANSFER_SUITE" \
  --epochs 2 --lr 5e-5 --weight_decay 0.01 \
  --lora 16 --lora_targets all --head_dim 256 \
  --batch 1 --accum 1 --dtype bf16 --weights_dtype bf16 --checkpointing 1 \
  --p_none 0.10 --p_none_distract 0.12 --p_distract 0.15 --p_none_pair 0.25 \
  --eval_before_start --eval_every_steps 786 --checkpoint_every_steps 786 \
  --out "$OUT" "${WANDB_ARGS[@]}"
