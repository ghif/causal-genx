#!/bin/bash
set -eo pipefail

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
precision="auto"
gpu_id="${CUDA_VISIBLE_DEVICES:-0}"
gpu_id="${gpu_id%%,*}"
gpu_memory_fraction="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
exp_name=""
extra_args=()

# Keep the original "experiment-name accelerator" positional interface working.
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
  exp_name="$1"
  shift
fi
if [ $# -gt 0 ]; then
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

while [ $# -gt 0 ]; do
  case "$1" in
    --accelerator)
      accelerator="${2:?missing value for --accelerator}"
      shift 2
      ;;
    --precision)
      precision="${2:?missing value for --precision}"
      shift 2
      ;;
    --gpu_id)
      gpu_id="${2:?missing value for --gpu_id}"
      shift 2
      ;;
    --gpu_memory_fraction)
      gpu_memory_fraction="${2:?missing value for --gpu_memory_fraction}"
      shift 2
      ;;
    --exp_name)
      exp_name="${2:?missing value for --exp_name}"
      shift 2
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

if [ "$accelerator" = "cuda" ]; then
  accelerator="gpu"
fi
if [[ "$accelerator" != "cpu" && "$accelerator" != "gpu" && "$accelerator" != "tpu" ]]; then
  echo "Unsupported accelerator '$accelerator'. Use cpu, gpu (or cuda), or tpu." >&2
  exit 2
fi
if [ "$precision" = "auto" ]; then
  if [ "$accelerator" = "cpu" ]; then
    precision="fp32"
  else
    precision="bf16"
  fi
fi
if [[ "$precision" != "fp32" && "$precision" != "bf16" ]]; then
  echo "Unsupported precision '$precision'. Use auto, fp32, or bf16." >&2
  exit 2
fi
if [ "$accelerator" = "cpu" ] && [ "$precision" != "fp32" ]; then
  echo "CPU predictor training supports fp32 only." >&2
  exit 2
fi
if [ -z "$exp_name" ]; then
  exp_name="sup_aux_jax-${accelerator}_$(date +%Y%m%d_%H%M%S)"
fi

export JAX_ENABLE_COMPILATION_CACHE="${JAX_ENABLE_COMPILATION_CACHE:-true}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/causal-genx-jax-cache}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-1}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

if [ "$accelerator" = "cpu" ]; then
  export JAX_PLATFORMS=cpu
  export JAX_PLATFORM_NAME=cpu
  export CUDA_VISIBLE_DEVICES=""
elif [ "$accelerator" = "gpu" ]; then
  unset JAX_PLATFORMS
  unset JAX_PLATFORM_NAME
  export CUDA_VISIBLE_DEVICES="$gpu_id"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="$gpu_memory_fraction"
  echo "GPU selection: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (single-device JAX)"
else
  unset JAX_PLATFORMS
  unset JAX_PLATFORM_NAME
fi

echo "Compute policy: accelerator=$accelerator precision=$precision master_params=fp32"

run_cmd=(
  python -u train_pgm.py
  --accelerator "$accelerator"
  --gpu_id "$gpu_id"
  --precision "$precision"
  --dataset morphomnist
  --data_dir gs://medical-airnd/causal-gen/datasets/morphomnist
  --ckpt_dir ../../checkpoints
  --remote_ckpt_dir gs://medical-airnd/causal-gen/checkpoints
  --exp_name "$exp_name"
  --setup sup_aux
  --input_res 32
  --pad 4
  --bs 32
  --epochs 1000
)

if [ "${#extra_args[@]}" -gt 0 ]; then
  run_cmd+=("${extra_args[@]}")
fi

"${run_cmd[@]}"
