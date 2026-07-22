# TPU Optimization Plan for MorphoMNIST

Goal: make the MorphoMNIST training path faster on a v6e-4 TPU while keeping the CPU path working as the default and preserving checkpoint, logging, and experiment semantics.

## Guiding Rules

- Keep CPU as the default backend.
- Add TPU support as an opt-in mode, not a separate codebase.
- Preserve existing CLI flags and checkpoint layout where practical.
- Prefer changes that improve both TPU and CPU behavior.
- Do not make TPU-only assumptions in shared code.

## Current Baseline

The current code is already close to a TPU-friendly JAX stack, but it still has a few CPU-first choices:

- hard CPU environment settings at import time in several modules
- Python/PIL-based per-sample batching and augmentation
- host-side batch assembly on every iteration
- no JIT wrapper around the hot training step
- checkpointing and logging work correctly, but can be trimmed for less host overhead
- mixed precision and gradient accumulation flags exist in the CLI, but are not yet used in the training loop

Relevant files:

- [`src/main.py`](./src/main.py)
- [`src/trainer.py`](./src/trainer.py)
- [`src/datasets.py`](./src/datasets.py)
- [`src/utils.py`](./src/utils.py)
- [`src/hps.py`](./src/hps.py)
- [`src/models.py`](./src/models.py)
- [`src/run_local.sh`](./src/run_local.sh)

## Phase 1: Backend Selection

### 1.1 Make accelerator selection explicit

Update `src/hps.py` so the CLI can distinguish CPU and TPU runs:

- expand `--accelerator` from only `cpu` to `cpu` and `tpu`
- expand `--precision` from only `fp32` to `fp32` and `bf16`
- keep the current defaults so existing CPU launches still work unchanged

### 1.2 Stop forcing CPU at import time

The following files currently force CPU behavior before the program even starts:

- `src/main.py`
- `src/trainer.py`
- `src/models.py`
- `src/utils.py`
- `src/pgm/train_pgm.py`
- `src/pgm/train_cf.py`
- `src/pgm/flow_pgm.py`
- `src/pgm/dscm.py`

Plan:

- move the CPU-specific environment settings into the CPU launcher
- only set them when `args.accelerator == "cpu"`
- keep the shared modules backend-neutral unless there is a strong reason not to

Result:

- CPU remains safe and deterministic
- TPU can run the same code without hidden CPU locks

### 1.3 Split launchers by intent

Keep `src/run_local.sh` as the CPU launcher.

Add a separate TPU launcher that:

- does not set `JAX_PLATFORMS=cpu`
- uses TPU-appropriate environment variables only when needed
- passes `--accelerator tpu`
- can set `--precision bf16` for TPU runs

## Phase 2: Input Pipeline

### 2.1 Remove per-sample Python overhead from batching

The current iterator in `src/utils.py` builds every batch in Python:

- iterates sample by sample
- stacks each key manually
- relies on `numpy.random` and Python loops

This is fine for CPU smoke tests, but it becomes a bottleneck on TPU.

Plan:

- keep the current iterator as a fallback
- add a faster batched iterator for TPU runs
- make batch shapes fixed so TPU compilation stays stable
- drop or pad the final batch instead of changing batch shape mid-epoch

### 2.2 Reduce PIL work during training

In `src/datasets.py`, training augmentation currently uses:

- `PIL.Image`
- `ImageOps.expand`
- random crop in Python
- resize in Python for evaluation

Plan:

- precompute or cache as much as possible at dataset load time
- keep the semantics of padding, crop, and resize
- minimize per-step PIL usage
- make augmentation deterministic when `--deterministic` is set

### 2.3 Keep the MorphoMNIST semantics unchanged

Do not change:

- normalization ranges
- parent-variable concatenation behavior
- train/valid/test split conventions
- image value scaling to `[-1, 1]`

This is important so TPU results remain comparable to existing CPU runs.

## Phase 3: Compile the Hot Path

### 3.1 JIT the training step

The most important function to compile is `train_step` in `src/trainer.py`.

Plan:

- wrap the loss/gradient/update portion in `jax.jit`
- keep the model reconstruction and parameter pytree handling compatible with JIT
- pass only fixed-shape arrays into the compiled function
- keep logging, checkpointing, and image export outside the jitted region

### 3.2 JIT the eval step

Compile `eval_step` separately so validation runs also benefit from TPU execution.

### 3.3 Avoid recompilation churn

Make sure these stay stable:

- batch size
- image size
- parent tensor shape
- dtype

If any of those change during a run, TPU performance drops because the compiler has to re-specialize.

## Phase 4: Precision and Memory

### 4.1 Add bf16 as an opt-in TPU mode

Use `bf16` for TPU activations if profiling shows it helps.

Recommended rules:

- keep checkpoint weights in `fp32`
- cast at model boundaries, not everywhere
- keep numerically sensitive reductions in `fp32` where practical
- leave CPU defaulting to `fp32`

### 4.2 Keep numerical stability safeguards

Preserve existing clipping and stability logic in the model code.

The current model already uses:

- clipping in Gaussian log-scales
- `nan_to_num` guards
- bounded likelihood computations

Those are good to keep on both CPU and TPU.

### 4.3 Use gradient accumulation if needed

The CLI already exposes `accu_steps`, but the loop does not use it yet.

Plan:

- implement accumulation in the training loop
- use it to simulate a larger effective batch on TPU
- keep the CPU path working with `accu_steps=1`

## Phase 5: Scheduling and Stability

The config already defines several knobs that can be used for TPU scaling:

- `lr_warmup_steps`
- `beta_warmup_steps`
- `grad_clip`
- `grad_skip`
- `eval_freq`
- `speed_log_freq`

Plan:

- wire the warmup settings into the actual training step
- keep `grad_clip` and `grad_skip` behavior unchanged unless profiling shows a TPU-specific issue
- reduce evaluation and visualization frequency during large TPU runs if host overhead starts to matter

## Phase 6: Logging and Checkpoints

### 6.1 Reduce host-side logging overhead

In `src/trainer.py`, the training loop currently:

- stores every step output in a Python list
- computes epoch means from that list later
- writes many per-step speed metrics

Plan:

- switch to running averages instead of collecting all outputs
- keep the current speed metrics, but log them less often if needed
- leave TensorBoard intact

### 6.2 Keep Orbax checkpointing

Checkpointing already uses Orbax in `src/utils.py`, which is good.

Plan:

- keep the same checkpoint root layout
- continue saving `params`, `ema_params`, `opt_state`, `step`, `epoch`, and `hparams`
- keep resume semantics unchanged
- make remote sync less frequent or asynchronous if it becomes a TPU bottleneck

### 6.3 Preserve the smoke test

The checkpoint smoke test in `src/trainer.py` should stay in place.

It is a useful guard for:

- restore correctness
- tree structure consistency
- optimizer state round-trip behavior

## Phase 7: Model Code

The model implementation in `src/models.py` is already JAX-native and should mostly stay as-is.

Plan:

- avoid architectural changes until profiling proves they are necessary
- only remove repeated Python-side work if it affects TPU speed
- preserve current tensor shapes and output semantics

Potential later cleanups if profiling points there:

- reduce repeated dict construction in decoder paths
- minimize unnecessary `materialize_nnx()` calls
- consider small shape or layout adjustments only if they help TPU compilation

## Phase 8: Validation

### 8.1 CPU regression check

Before enabling TPU-specific changes:

- run a CPU smoke test
- confirm `--accelerator cpu` still works
- confirm current checkpoint resume still works
- confirm TensorBoard images and scalars still write correctly

### 8.2 TPU smoke check

Then run a tiny TPU job with:

- `--accelerator tpu`
- `--precision bf16` if enabled
- one or two epochs
- checkpoint smoke test enabled

Verify:

- no shape recompiles during the loop
- loss is finite
- checkpoint saves and restores
- sample images still look sane

### 8.3 Compare against the CPU baseline

Check that the TPU run matches the CPU run on:

- output shape
- loss scale
- checkpoint contents
- training semantics

Exact numerical equality is not required, but the runs should be semantically aligned.

## Suggested Implementation Order

1. Add accelerator and precision flags.
2. Remove import-time CPU locking from shared modules.
3. Add a TPU launcher script.
4. Stabilize batch shapes and input pipeline behavior.
5. JIT the training and evaluation steps.
6. Add bf16 as an opt-in path.
7. Implement gradient accumulation and warmup wiring.
8. Reduce logging and checkpoint host overhead.
9. Run CPU regression tests.
10. Run a TPU smoke test and tune from there.

## Expected Outcome

If we follow this plan, we should get:

- a TPU-capable MorphoMNIST training path
- the same CPU default behavior that exists today
- fewer host bottlenecks
- stable checkpointing and resume behavior
- a codebase that can keep growing without splitting into separate CPU and TPU implementations
