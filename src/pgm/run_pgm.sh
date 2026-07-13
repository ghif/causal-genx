#!/bin/bash
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

accelerator="auto"
if [ $# -ge 1 ]; then
  exp_name="$1"
  shift
else
  exp_name="pgm_torch-mps_$(date +%Y%m%d_%H%M%S)"
fi

if [ $# -ge 1 ]; then
  case "$1" in
    cpu|mps|cuda|gpu|auto)
      accelerator="$1"
      shift
      ;;
  esac
fi

if [ "$accelerator" = "gpu" ]; then
  accelerator="cuda"
fi

extra_args=("$@")

run_cmd=(
  python train_pgm.py
  --accelerator "$accelerator"
  --dataset morphomnist
  --data_dir gs://medical-airnd/causal-gen/datasets/morphomnist
  --ckpt_dir gs://medical-airnd/causal-gen/checkpoints
  --exp_name "$exp_name"
  --setup sup_pgm
  --input_res 32
  --pad 4
  --bs 16
  --epochs 1000
  "${extra_args[@]}"
)

if [ "$accelerator" = "mps" ]; then
  PYTORCH_ENABLE_MPS_FALLBACK=1 "${run_cmd[@]}"
else
  "${run_cmd[@]}"
fi
