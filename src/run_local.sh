#!/bin/bash
if [ "${CONDA_DEFAULT_ENV:-}" != "med-jax" ]; then
  CONDA_BASE="$(conda info --base 2>/dev/null)"
  if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "Unable to locate conda.sh for med-jax activation." >&2
    exit 1
  fi
  . "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate med-jax
fi

export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""

exp_name="${1:-local_run_$(date +%Y%m%d_%H%M%S)}"
extra_args=()
run_nohup=false
run_smoke=false

for arg in "${@:2}"; do
  if [ "$arg" = "nohup" ]; then
    run_nohup=true
  elif [ "$arg" = "smoke" ]; then
    run_smoke=true
  else
    extra_args+=("$arg")
  fi
done

if [ "$run_smoke" = true ]
then
  extra_args+=(
    --epochs=1
    --viz_freq=1
    --checkpoint_smoke_test
    --checkpoint_smoke_steps=1
    --eval_freq=1000000
  )
fi

run_cmd=(python -u main.py
  --accelerator=cpu
  --precision=fp32
  --exp_name="$exp_name"
  --data_dir=gs://medical-airnd/causal-gen/datasets/morphomnist
  --ckpt_dir=../checkpoints
  --remote_ckpt_dir=gs://medical-airnd/causal-gen/checkpoints
  --hps morphomnist
  --parents_x thickness intensity digit
  --context_dim=12
  --concat_pa
  --lr=0.001
  --bs=32
  --wd=0.01
  --beta=1
  --cond_prior
  --eval_freq=4
  --viz_freq=1
  --viz_batch_size=32
  "${extra_args[@]}")

if [ "$run_nohup" = true ]
then
  nohup "${run_cmd[@]}" > "$exp_name.out" 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  "${run_cmd[@]}"
fi
