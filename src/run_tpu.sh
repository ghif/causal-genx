#!/bin/bash
set -euo pipefail

export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"

exp_name="${1:-tpu_v6e_run_$(date +%Y%m%d_%H%M%S)}"
shift || true

python tpu_launcher.py main.py \
  --accelerator tpu \
  --precision bf16 \
  --exp_name="$exp_name" \
  --data_dir=gs://medical-airnd/causal-gen/datasets/morphomnist \
  --ckpt_dir=gs://medical-airnd/causal-gen/checkpoints \
  --hps morphomnist \
  --parents_x thickness intensity digit \
  --context_dim=12 \
  --concat_pa \
  --lr=0.001 \
  --bs=32 \
  --wd=0.01 \
  --beta=1 \
  --cond_prior \
  --eval_freq=4 \
  "$@"
