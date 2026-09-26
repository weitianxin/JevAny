#!/usr/bin/env bash
# The same recipe runs on a workstation or under a scheduler on multiple hosts.
set -euo pipefail
PYTHON_BIN=${PYTHON_BIN:-python}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
VISIBLE_GPUS=$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')
PROCESSES_PER_HOST=${PROCESSES_PER_HOST:-$VISIBLE_GPUS}

if (( PROCESSES_PER_HOST < 1 || PROCESSES_PER_HOST > VISIBLE_GPUS )); then
  echo "PROCESSES_PER_HOST must be between 1 and $VISIBLE_GPUS visible GPUs." >&2
  exit 2
fi
if (( PROCESSES_PER_HOST < VISIBLE_GPUS )) && [[ -z "${GPU_SHORTFALL_REASON:-}" ]]; then
  echo "Set GPU_SHORTFALL_REASON when using fewer processes than visible GPUs." >&2
  exit 2
fi
if (( NNODES > 1 )) && [[ "$MASTER_ADDR" == 127.0.0.1 ]]; then
  echo "Set MASTER_ADDR to the rank-0 host for multi-node training." >&2
  exit 2
fi
echo "hosts=$NNODES visible_gpus_per_host=$VISIBLE_GPUS processes_per_host=$PROCESSES_PER_HOST world_size=$((NNODES * PROCESSES_PER_HOST)) shortfall_reason=${GPU_SHORTFALL_REASON:-none}"
exec "$PYTHON_BIN" -m torch.distributed.run \
  --nnodes "$NNODES" --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
  --nproc_per_node "$PROCESSES_PER_HOST" \
  -m jevany train "$@"
