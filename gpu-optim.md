# GPU A100 Optimization Plan for MorphoMNIST

Goal: speed up MorphoMNIST training on a single A100 while keeping the CPU path intact and preserving the current experiment, logging, and checkpoint behavior.

## Guiding Rules

- Keep CPU as the default backend.
- Add GPU support as an opt-in runtime mode, not a separate codepath.
- Preserve the existing CLI, checkpoint layout, and artifact names where practical.
- Prefer improvements that benefit both CPU and GPU.
- Keep the code JAX-native and avoid introducing framework-specific branches unless they are truly necessary.

## Current Baseline

The code already has a clean JAX/Flax/Optax structure, which is a good starting point for A100 work. The main areas that still look CPU-first are:

- import-time CPU environment locks
- Python/PIL-driven per-sample batching and augmentation
- no explicit compiled train/eval step boundary
- no mixed precision mode exposed through the CLI
- host-side logging and checkpoint syncing that can become noticeable once the GPU is fast

Relevant files:

- [`src/main.py`](./src/main.py)
- [`src/trainer.py`](./src/trainer.py)
- [`src/datasets.py`](./src/datasets.py)
- [`src/utils.py`](./src/utils.py)
- [`src/hps.py`](./src/hps.py)
- [`src/models.py`](./src/models.py)
- [`src/run_local.sh`](./src/run_local.sh)

## Phase 1: Backend Selection

### 1.1 Make GPU a first-class accelerator option

Update `src/hps.py` so `--accelerator` can represent:

- `cpu`
- `gpu`
- `tpu`

Keep the default as `cpu` so existing launches do not change.

### 1.2 Remove hard CPU locking from shared imports

The shared modules currently force CPU-related environment values at import time.

Plan:

- move backend-specific env settings into launcher scripts
- only enforce CPU-only variables when the user explicitly selects CPU
- keep shared modules backend-neutral so the same code can run on CPU, A100, or TPU

Files to review:

- `src/main.py`
- `src/trainer.py`
- `src/models.py`
- `src/utils.py`
- `src/pgm/train_pgm.py`
- `src/pgm/train_cf.py`
- `src/pgm/flow_pgm.py`
- `src/pgm/dscm.py`

### 1.3 Separate launcher intent

Keep `src/run_local.sh` as the CPU launcher.

Add a GPU launcher that:

- selects `--accelerator gpu`
- does not force CPU-only JAX environment settings
- can optionally set `--precision bf16`
- leaves room for CUDA-specific env tuning if needed

## Phase 2: Input Pipeline

### 2.1 Reduce Python work per batch

The current batch assembly path in `src/utils.py` builds every batch through Python loops and `numpy.stack`.

That is acceptable for CPU, but on an A100 it can become the limiting factor if the model step gets faster than the input pipeline.

Plan:

- keep the current iterator as a safe fallback
- add a faster batched path for GPU runs
- preserve fixed batch shapes across the whole epoch
- avoid a changing last batch shape unless it is padded or dropped

### 2.2 Minimize per-sample augmentation overhead

`src/datasets.py` uses PIL for crop and resize behavior.

Plan:

- cache or precompute as much preprocessing as practical
- keep the same pad/crop/resize semantics
- preserve the normalization and parent-variable behavior
- keep deterministic mode reproducible

### 2.3 Keep data semantics unchanged

Do not alter:

- MorphoMNIST splits
- normalization ranges
- parent-variable concatenation
- image scaling to `[-1, 1]`

We want performance gains, not a different experiment.

## Phase 3: Compile the Hot Path

### 3.1 JIT the training step

The main speedup for A100 should come from compiling the inner update step in `src/trainer.py`.

Plan:

- JIT the forward/loss/grad/update path
- keep logging and checkpointing on the host
- ensure batch and parameter shapes stay stable
- avoid frequent recompiles caused by shape drift or dtype changes

### 3.2 JIT the evaluation step

Compile `eval_step` separately so validation also runs efficiently.

### 3.3 Avoid host-device syncs in the hot loop

Keep these off the critical path:

- frequent Python reductions over per-step outputs
- repeated device-to-host conversions
- image writing during training steps

## Phase 4: Precision and Throughput

### 4.1 Add bf16 as the preferred GPU mode

A100s are very strong on bf16, so this should be the first mixed-precision target.

Recommended rules:

- keep checkpoints in fp32
- use bf16 for model activations when stable
- keep numerically sensitive reductions in fp32 when needed
- leave CPU defaulting to fp32

### 4.2 Keep the existing stability protections

The model already includes useful safeguards in `src/models.py`:

- log-scale clipping
- `nan_to_num` guards
- bounded likelihood logic

Keep those in place while changing precision or batch size.

### 4.3 Increase batch size carefully

Once bf16 and JIT are in place:

- raise `--bs` until throughput or memory saturates
- use `accu_steps` if you want a larger effective batch without exceeding memory
- keep the CPU path working with the original smaller defaults

## Phase 5: Training Schedule

The config already exposes some useful knobs in `src/hps.py`:

- `lr_warmup_steps`
- `beta_warmup_steps`
- `grad_clip`
- `grad_skip`
- `eval_freq`
- `speed_log_freq`

Plan:

- wire in the warmup schedules if they are not already active in the path you use
- keep `grad_clip` behavior the same unless profiling shows a clear need to change it
- reduce validation and visualization frequency if host overhead starts to compete with GPU work

## Phase 6: Logging and Checkpoints

### 6.1 Make logging cheaper

The current training loop stores every step output and computes epoch means later.

Plan:

- switch to running averages instead of collecting a full list of step outputs
- keep scalar logging, but not every metric every step unless needed
- preserve TensorBoard output

### 6.2 Keep Orbax checkpointing

Checkpointing is already on Orbax in `src/utils.py`, which is a good foundation.

Plan:

- preserve `params`, `ema_params`, `opt_state`, `step`, `epoch`, and `hparams`
- keep the checkpoint root layout stable
- keep resume semantics unchanged
- if remote sync becomes a bottleneck, make it less frequent or asynchronous

### 6.3 Preserve smoke tests

The checkpoint smoke test should remain available.

It protects:

- restore correctness
- optimizer tree shape consistency
- EMA round-tripping

## Phase 7: Model Code

The model code in `src/models.py` is already close to what we want for GPU execution.

Plan:

- avoid redesigning the architecture just for A100
- keep the same tensor shapes and output semantics
- only optimize the model internals if profiling shows a real hotspot

Potential later tweaks:

- reduce repeated Python-side object construction
- minimize unnecessary merges/materializations
- consider layout tweaks only if they clearly help the GPU path

## Phase 8: Validation

### 8.1 CPU regression

Before turning on GPU-specific changes:

- confirm the default CPU run still works
- confirm checkpoint resume still works
- confirm sample image writing and TensorBoard logging still work

### 8.2 A100 smoke test

Run a short GPU test with:

- `--accelerator gpu`
- `--precision bf16`
- a small epoch count
- checkpoint smoke test enabled

Check that:

- the step compiles once and then runs without recompiling every iteration
- losses stay finite
- checkpoint saves and restores correctly
- sample outputs remain reasonable

### 8.3 Compare behavior

Compare CPU and A100 runs on:

- output shapes
- loss scale
- checkpoint contents
- training semantics

Bitwise equality is not required, but the runs should be behaviorally aligned.

## Suggested Implementation Order

1. Add `gpu` to the accelerator choices.
2. Remove import-time CPU locking from shared modules.
3. Add a GPU launcher script.
4. JIT the train and eval steps.
5. Add bf16 as an opt-in precision mode.
6. Trim batch pipeline overhead.
7. Wire in accumulation and warmup if needed.
8. Reduce logging and sync overhead.
9. Run CPU regression tests.
10. Run an A100 smoke test and tune batch size.

## Expected Outcome

This should give us:

- a faster MorphoMNIST path on A100
- the same CPU-default behavior as today
- cleaner separation between backend choice and model logic
- a single codebase that can scale from CPU to GPU to TPU without branching into separate implementations
