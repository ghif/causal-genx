#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
export PYTHONPATH="$script_dir/..${PYTHONPATH:+:$PYTHONPATH}"

if [ "${CONDA_DEFAULT_ENV:-}" != "med-jax" ]; then
  conda_base="$(conda info --base 2>/dev/null || true)"
  if [ -z "$conda_base" ] || [ ! -f "$conda_base/etc/profile.d/conda.sh" ]; then
    echo "Unable to locate conda.sh for med-jax activation." >&2
    exit 1
  fi
  . "$conda_base/etc/profile.d/conda.sh"
  conda activate med-jax
fi

accelerator="cpu"
if [ $# -ge 1 ]; then
  exp_name="$1"
  shift
else
  exp_name="pgm_jax-cpu_$(date +%Y%m%d_%H%M%S)"
fi

if [ $# -ge 1 ]; then
  case "$1" in
    cpu|gpu|tpu)
      accelerator="$1"
      shift
      ;;
    cuda)
      accelerator="gpu"
      shift
      ;;
  esac
fi

if [ "$accelerator" = "cpu" ]; then
  export JAX_PLATFORMS=cpu
  export JAX_PLATFORM_NAME=cpu
  export CUDA_VISIBLE_DEVICES=""
else
  unset JAX_PLATFORMS
  unset JAX_PLATFORM_NAME
fi

extra_args=("$@")

run_cmd=(
  python -u train_pgm.py
  --accelerator "$accelerator"
  --precision fp32
  --dataset morphomnist
  --data_dir gs://medical-airnd/causal-gen/datasets/morphomnist
  --ckpt_dir ../../checkpoints
  --remote_ckpt_dir gs://medical-airnd/causal-gen/checkpoints
  --exp_name "$exp_name"
  --setup sup_pgm
  --input_res 32
  --pad 4
  --bs 16
  --epochs 1000
  "${extra_args[@]}"
)

"${run_cmd[@]}"
