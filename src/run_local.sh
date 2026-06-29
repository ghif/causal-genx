#!/bin/bash
if [ "${CONDA_DEFAULT_ENV:-}" != "med-torch" ]; then
  CONDA_BASE="$(conda info --base 2>/dev/null)"
  if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "Unable to locate conda.sh for med-torch activation." >&2
    exit 1
  fi
  . "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate med-torch
fi

exp_name="${1:-local_run_$(date +%Y%m%d_%H%M%S)}"
extra_args=()
run_nohup=false

for arg in "${@:2}"; do
  if [ "$arg" = "nohup" ]; then
    run_nohup=true
  else
    extra_args+=("$arg")
  fi
done

run_cmd=(python main.py
  --exp_name="$exp_name"
  --data_dir=gs://causal-gen/datasets/morphomnist
  --ckpt_dir=gs://causal-gen/checkpoints
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
  "${extra_args[@]}")

# run_cmd="python main.py \
#     --exp_name=$exp_name \
#     --data_dir=/data2/ukbb \
#     --hps ukbb192 \
#     --parents_x mri_seq brain_volume ventricle_volume sex \
#     --context_dim=4 \
#     --concat_pa \
#     --lr=0.001 \
#     --bs=32 \
#     --wd=0.05 \
#     --beta=5 \
#     --x_like=diag_dgauss \
#     --z_max_res=96 \
#     --eval_freq=4"

if [ "$run_nohup" = true ]
then
  nohup "${run_cmd[@]}" > "$exp_name.out" 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  "${run_cmd[@]}"
fi
