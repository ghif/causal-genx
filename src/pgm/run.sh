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
exp_name=""
data_dir=""
ckpt_dir=""
pgm_path=""
predictor_path=""
vae_path=""
trust_incomplete_checkpoint="false"
extra_args=()

while [ $# -gt 0 ]; do
  case "$1" in
    --accelerator)
      accelerator="${2:?missing value for --accelerator}"
      shift 2
      ;;
    --exp_name)
      exp_name="${2:?missing value for --exp_name}"
      shift 2
      ;;
    --data_dir)
      data_dir="${2:?missing value for --data_dir}"
      shift 2
      ;;
    --ckpt_dir)
      ckpt_dir="${2:?missing value for --ckpt_dir}"
      shift 2
      ;;
    --pgm_path)
      pgm_path="${2:?missing value for --pgm_path}"
      shift 2
      ;;
    --predictor_path)
      predictor_path="${2:?missing value for --predictor_path}"
      shift 2
      ;;
    --vae_path)
      vae_path="${2:?missing value for --vae_path}"
      shift 2
      ;;
    --trust_incomplete_checkpoint)
      trust_incomplete_checkpoint="true"
      shift
      ;;
    --*)
      if [ $# -ge 2 ] && [[ "${2}" != --* ]]; then
        extra_args+=("$1" "$2")
        shift 2
      else
        extra_args+=("$1")
        shift
      fi
      ;;
    *)
      if [[ "$1" =~ ^[[:space:]]+$ ]]; then
        echo "Unexpected whitespace-only argument. Ensure each line-continuation backslash is the final character on its line." >&2
        exit 2
      fi
      if [ -z "$exp_name" ]; then
        exp_name="$1"
      else
        extra_args+=("$1")
      fi
      shift
      ;;
  esac
done

if [ -z "$exp_name" ]; then
  exp_name="morphomnist_cf_jax_$(date +%Y%m%d_%H%M%S)"
fi

if [ "$accelerator" = "cuda" ]; then
  accelerator="gpu"
fi

if [ -z "$data_dir" ]; then
  data_dir="gs://medical-airnd/causal-gen/datasets/morphomnist"
fi
if [ -z "$ckpt_dir" ]; then
  ckpt_dir="../../checkpoints"
fi
if [ -z "$pgm_path" ]; then
  pgm_path="../../checkpoints/morphomnist/pgm/checkpoints"
fi
if [ -z "$predictor_path" ]; then
  predictor_path="../../checkpoints/morphomnist/run/checkpoints"
fi
if [ -z "$vae_path" ]; then
  vae_path="../../checkpoints/morphomnist/run/checkpoints"
fi

echo "VAE checkpoint: $vae_path"
echo "PGM checkpoint: $pgm_path"
echo "Predictor checkpoint: $predictor_path"

if [ "$accelerator" = "cpu" ]; then
  export JAX_PLATFORMS=cpu
  export JAX_PLATFORM_NAME=cpu
  export CUDA_VISIBLE_DEVICES=""
else
  unset JAX_PLATFORMS
  unset JAX_PLATFORM_NAME
fi

run_cmd=(
  python -u train_cf.py
  --accelerator "$accelerator"
  --precision fp32
  --dataset morphomnist
  --data_dir "$data_dir"
  --ckpt_dir "$ckpt_dir"
  --remote_ckpt_dir gs://medical-airnd/causal-gen/checkpoints
  --exp_name "$exp_name"
  --pgm_path "$pgm_path"
  --predictor_path "$predictor_path"
  --vae_path "$vae_path"
  --do_pa None
  --bs 32
  --lr 1e-4
  --wd 0.1
  --eval_freq 1
  --plot_freq 500
  --alpha 0.1
  --lmbda_init 0
  --damping 100
  --model_validation_batches 1
  --seed 7
)

if [ "${#extra_args[@]}" -gt 0 ]; then
  run_cmd+=("${extra_args[@]}")
fi

if [ "$trust_incomplete_checkpoint" = "true" ]; then
  run_cmd+=(--trust_incomplete_checkpoint)
fi

"${run_cmd[@]}"
