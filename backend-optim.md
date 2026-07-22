# Backend Optimization Plan for MorphoMNIST

Goal: keep one codebase that runs well on CPU by default, scales efficiently on an A100 GPU, and can also run on a v6e-4 TPU without forking the project into separate implementations.

## Core Principle

Treat backend choice as a runtime concern, not a codebase split.

- CPU stays the default and the compatibility baseline.
- GPU and TPU are opt-in accelerators.
- Shared model, dataset, and checkpoint logic should stay backend-neutral.
- Backend-specific tuning should live in launch scripts, config flags, and a few narrow runtime branches.

## Where the Current Code Stands

The repository is already in a good place for JAX-based backend portability:

- the model code is JAX/Flax-based
- the optimizer and checkpoint stack are already JAX-native
- the MorphoMNIST path is the main validated path

The main remaining performance issues are:

- hard CPU environment settings at import time
- Python/PIL-heavy per-sample data loading and augmentation
- no explicit compiled hot step
- no mixed precision backend mode exposed through the CLI
- some host-side logging and checkpoint overhead that can grow visible once the device is fast

Relevant files:

- [`src/main.py`](./src/main.py)
- [`src/trainer.py`](./src/trainer.py)
- [`src/datasets.py`](./src/datasets.py)
- [`src/utils.py`](./src/utils.py)
- [`src/hps.py`](./src/hps.py)
- [`src/models.py`](./src/models.py)
- [`src/run_local.sh`](./src/run_local.sh)

## Shared Backend Rules

These rules should hold for CPU, A100 GPU, and TPU:

- keep batch shapes fixed
- keep the train step JIT-compiled
- keep logging and checkpointing on the host
- keep checkpoints in fp32 even if activations use bf16
- keep deterministic mode available
- keep the MorphoMNIST semantics unchanged

That gives us one reliable training contract across devices.

## Phase 1: Runtime Selection

### 1.1 Add explicit accelerator modes

Update `src/hps.py` so `--accelerator` supports:

- `cpu`
- `gpu`
- `tpu`

Keep the default as `cpu`.

### 1.2 Add precision as a runtime choice

Update `src/hps.py` so `--precision` supports:

- `fp32`
- `bf16`

Keep `fp32` as the default.

### 1.3 Remove hard-coded CPU locking from shared imports

These files currently enforce CPU behavior at import time:

- `src/main.py`
- `src/trainer.py`
- `src/models.py`
- `src/utils.py`
- `src/pgm/train_pgm.py`
- `src/pgm/train_cf.py`
- `src/pgm/flow_pgm.py`
- `src/pgm/dscm.py`

Plan:

- move backend selection into launch scripts and config
- only set CPU-only env variables when the user explicitly selects CPU
- keep shared modules backend-neutral

### 1.4 Split launcher scripts by intent

Keep the current `src/run_local.sh` as the CPU launcher.

Add separate launcher scripts or documented command lines for:

- GPU A100
- TPU v6e-4

Each launcher should pass the right `--accelerator` and `--precision` values without affecting the default CPU path.

## Phase 2: Input Pipeline

### 2.1 Remove avoidable Python work from batching

The current iterator in `src/utils.py` builds batches with Python loops and `numpy.stack`.

That is fine for CPU, but it will limit A100 and TPU throughput once the model step gets faster.

Plan:

- keep the existing iterator as a safe fallback
- add a faster batched path for accelerator runs
- preserve fixed batch shapes
- pad or drop the last batch instead of changing shapes mid-epoch

### 2.2 Reduce per-sample PIL overhead

`src/datasets.py` currently does training crop/resize augmentation with PIL per sample.

Plan:

- cache or precompute as much as practical
- keep the same augmentation semantics
- keep normalization and parent-variable construction exactly the same
- preserve deterministic behavior when requested

### 2.3 Preserve MorphoMNIST semantics

Do not change:

- train/valid/test split logic
- normalization ranges
- `pa` concatenation behavior
- input scaling to `[-1, 1]`

The speed-up should not change the experiment.

## Phase 3: Compile the Hot Step

### 3.1 JIT the training update

The hot path is the `train_step` function in `src/trainer.py`.

Plan:

- compile the forward, loss, gradient, and parameter update path
- keep the model reconstruction compatible with JIT
- make sure batch shapes, dtypes, and parameter trees stay stable
- leave logging, image export, and checkpointing outside the compiled region

### 3.2 JIT evaluation too

Compile the validation step separately so validation is also accelerator-friendly.

### 3.3 Avoid recompilation churn

Keep these constant throughout a run:

- batch size
- input resolution
- parent tensor shape
- dtype

That matters on both A100 and TPU.

## Phase 4: Precision Strategy

### 4.1 Default to fp32

CPU stays `fp32`.

### 4.2 Use bf16 on accelerators when stable

Recommended:

- A100: `bf16` first
- TPU: `bf16` first

Keep checkpoints in fp32 so resumes and comparisons remain stable.

### 4.3 Keep stability safeguards

The model already has:

- clipped log-scales
- `nan_to_num` guards
- bounded likelihood logic

Keep these in place while changing precision or batch size.

## Phase 5: Batch Size and Accumulation

### 5.1 Scale batch size with the device

CPU can keep the current batch size.

A100 and TPU can usually take larger batches, especially with bf16.

### 5.2 Use gradient accumulation when needed

The CLI already exposes `accu_steps`, but the loop does not currently use it.

Plan:

- implement accumulation once
- use it on A100 or TPU if memory becomes tight
- leave `accu_steps=1` as the CPU baseline

## Phase 6: Scheduling and Stability

The config already exposes useful knobs:

- `lr_warmup_steps`
- `beta_warmup_steps`
- `grad_clip`
- `grad_skip`
- `eval_freq`
- `speed_log_freq`

Plan:

- wire warmup into the actual training path if it is not already active
- keep gradient clipping behavior unchanged unless profiling suggests otherwise
- reduce eval and visualization frequency on fast accelerators if host overhead becomes visible

## Phase 7: Logging and Checkpointing

### 7.1 Make training loop logging cheaper

The current loop stores every step output and reduces it later.

Plan:

- switch to running averages
- keep scalar logging
- avoid excess device-to-host transfers in the inner loop

### 7.2 Keep Orbax checkpointing

Checkpointing already uses Orbax, which is the right foundation for all three backends.

Plan:

- preserve `params`, `ema_params`, `opt_state`, `step`, `epoch`, and `hparams`
- keep the same checkpoint directory structure
- keep resume behavior unchanged
- if remote sync becomes a bottleneck, decouple it from the hottest part of training

### 7.3 Keep the checkpoint smoke test

The smoke test is useful for all backends.

It verifies:

- restore correctness
- tree structure consistency
- optimizer state round-trip behavior

## Phase 8: Model Code

The model code in `src/models.py` should stay mostly unchanged.

Plan:

- avoid architecture rewrites unless profiling proves them worthwhile
- preserve tensor shapes and output semantics
- only optimize internals if they materially improve device performance

Potential later tweaks:

- reduce repeated Python-side object creation
- minimize repeated materialization/merge calls
- make any shape/layout changes only if they help a specific backend and do not break CPU

## Backend-Specific Focus

### CPU

- keep fp32
- keep the simple launcher
- prioritize correctness and reproducibility
- treat this as the compatibility baseline

### A100 GPU

- use bf16 if stable
- scale batch size up until memory or throughput saturates
- make the input pipeline less Python-heavy
- keep the compiled train step tight and synchronous only where needed

### TPU v6e-4

- use bf16 by default for the accelerator path
- keep batch shapes static
- avoid host-side stalls from the input pipeline
- make recompilation rare by stabilizing dtypes and shapes

## Validation Plan

### 9.1 CPU regression

Before accelerator-specific changes, confirm that:

- the default CPU run still works
- resume still works
- TensorBoard logging still works
- image outputs still look sane

### 9.2 A100 smoke test

Run a short A100 job with:

- `--accelerator gpu`
- `--precision bf16`
- a small epoch count
- checkpoint smoke test enabled

Check:

- the compiled step reuses the same shape
- losses stay finite
- checkpoints save and restore correctly

### 9.3 TPU smoke test

Run the same style of smoke test on TPU with:

- `--accelerator tpu`
- `--precision bf16`
- fixed batch shapes
- checkpoint smoke test enabled

Check:

- no repeated recompiles
- no host input stalls
- sane sample outputs and losses

## Suggested Order of Work

1. Add `gpu` and `tpu` accelerator modes.
2. Add `bf16` as an opt-in precision mode.
3. Remove import-time CPU locks from shared modules.
4. Add backend-specific launch scripts.
5. JIT the train and eval steps.
6. Trim the input pipeline overhead.
7. Implement accumulation and warmup wiring if needed.
8. Reduce logging and sync overhead.
9. Run CPU regression tests.
10. Run A100 and TPU smoke tests.

## Expected Outcome

This should give us:

- one codebase for CPU, A100, and TPU
- faster accelerator training without losing CPU compatibility
- stable checkpoints and resume behavior
- a cleaner path for future device-specific tuning
