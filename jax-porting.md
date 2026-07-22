# JAX Porting Plan for `causal-gen`

## Objective

Port `causal-gen` to a **pure JAX** implementation that runs on **Mac CPU** while preserving the repository’s current functionality and behavior as closely as possible.

This means:

- keep the same user-facing training and evaluation capabilities
- keep the same model families: `VAE`, `HVAE`, `FlowPGM`, `DSCM`, and counterfactual training
- keep the same dataset support and checkpointing behavior
- keep the same experiment semantics, hyperparameter names, and output artifacts where practical
- remove PyTorch, TorchVision, Pyro, and Torch XLA dependencies from the runtime path

## What Must Stay Functionally Equivalent

The current repository supports these workflows:

- train the image mechanism with `src/main.py` and `src/trainer.py`
- train the non-image causal mechanisms with `src/pgm/train_pgm.py`
- run counterfactual training with `src/pgm/train_cf.py`
- load and resume checkpoints
- log images, metrics, and TensorBoard summaries
- support Morpho-MNIST, Colour-MNIST, UKBB, and MIMIC-CXR data paths
- support both `simple` and `hierarchical` VAE variants
- preserve the causal graph logic and intervention/counterfactual semantics

The JAX port should preserve those workflows, not simplify them away.

## Suggested Target Stack

Use a small, CPU-friendly JAX stack:

- `jax` and `jaxlib` for array and compilation
- `flax` for module definitions and state handling
- `optax` for optimizers and schedules
- `orbax-checkpoint` for checkpointing
- `tensorboard` or `tensorboardX` for logging
- `chex` for shape and tree assertions
- `distrax` or a tiny local distribution layer for categorical, normal, and transform utilities
- `numpy`, `pandas`, `Pillow`, `imageio`, and `matplotlib` for data and visualization

Prefer local, explicit helpers over heavyweight probabilistic abstractions unless they are needed to preserve behavior.

## High-Level Strategy

Do the port in layers, not all at once:

1. Freeze the existing behavior with a small set of golden tests and reference outputs.
2. Build JAX infrastructure for config, RNG, checkpointing, data loading, and logging.
3. Port the shared math utilities and likelihoods.
4. Port the image models first (`simple_vae.py`, `vae.py`), because they are the core runtime path.
5. Port the structured causal model code (`pgm/*`) and the DSCM composition.
6. Port training loops and CLI entrypoints.
7. Validate parity at every stage against the Torch implementation.

This keeps the migration safe and gives us an early working Mac CPU pipeline.

## Current Code Map to Preserve

| Current file | Role | JAX target |
|---|---|---|
| `src/main.py` | Main training entrypoint | `causal_genx/main.py` or equivalent CLI |
| `src/trainer.py` | Image-model training loop | `causal_genx/trainers/image.py` |
| `src/train_setup.py` | Dataloaders, optimizers, logging, checkpoint dirs | `causal_genx/runtime/*` |
| `src/utils.py` | Seeding, EMA, path helpers, image writing | `causal_genx/utils/*` |
| `src/hps.py` | Hyperparameter registry and CLI args | `causal_genx/config.py` |
| `src/datasets.py` | MorphoMNIST / UKBB dataset loading | `causal_genx/data/*` |
| `src/dmol.py` | Discretized mixture of logistics | `causal_genx/distributions/dmol.py` |
| `src/simple_vae.py` | Lightweight VAE | `causal_genx/models/simple_vae.py` |
| `src/vae.py` | Hierarchical VAE | `causal_genx/models/hvae.py` |
| `src/pgm/layers.py` | Distribution and layer primitives | `causal_genx/models/pgm/layers.py` |
| `src/pgm/resnet.py` | ResNet backbone | `causal_genx/models/pgm/resnet.py` |
| `src/pgm/flow_pgm.py` | Structured-variable SCM | `causal_genx/models/pgm/flow_pgm.py` |
| `src/pgm/dscm.py` | Deep SCM composition | `causal_genx/models/pgm/dscm.py` |
| `src/pgm/train_pgm.py` | PGM training loop | `causal_genx/trainers/pgm.py` |
| `src/pgm/train_cf.py` | Counterfactual training loop | `causal_genx/trainers/counterfactual.py` |
| `src/xla_runtime.py` | Accelerator/XLA glue | remove or replace with CPU-only runtime helpers |

## Phase 0: Behavior Freeze

Before porting code, capture the current behavior from the Torch repository.

### Deliverables

- a tiny, deterministic smoke run for each supported path
- reference checkpoint metadata snapshots
- reference tensor shapes and sample outputs
- reference metrics for a few batches on each dataset path that is available locally

### What to record

- input batch shapes
- output tensor shapes and dtypes
- loss components and their scaling
- checkpoint keys and nesting
- sampled image grids
- counterfactual outputs for a fixed intervention seed

### Why this matters

The Torch code uses a lot of implicit behavior:

- EMA shadow copies
- gradient accumulation
- beta warmup
- resize/crop/flip data augmentation
- checkpoint resume semantics
- causal parent concatenation conventions

These details need to be preserved exactly, or the JAX port will “work” but not actually match the repository.

## Phase 1: JAX Runtime Foundation

Create the JAX-native infrastructure first.

### 1.1 Configuration and CLI

Replace `argparse`-plus-hyperparameter mutation with a structured config layer that can still parse the same flags.

Keep these semantics:

- `--hps` chooses a preset
- CLI overrides still win over preset defaults
- `--resume` reloads hyperparameters from the checkpoint unless explicitly overridden
- the same experiment names and directories remain usable

Recommended implementation:

- keep a thin CLI parser
- move resolved config into a frozen dataclass or `flax.struct.dataclass`
- use a dedicated merge function for checkpoint overrides

### 1.2 RNG and Reproducibility

Replace global Torch seeding with explicit JAX PRNG management.

Requirements:

- derive one master seed from the CLI
- split RNGs deterministically for:
  - initialization
  - dropout
  - data augmentation
  - sampling
  - evaluation
- preserve worker-level determinism where possible

On Mac CPU, run single-host and single-device first.

### 1.3 Checkpointing

Replace `torch.save` / `torch.load` with an Orbax checkpoint directory layout.

Preserve:

- model parameters
- optimizer state
- EMA state
- global step / epoch counters
- hyperparameters
- any resume-specific metadata

Use a versioned checkpoint schema so future changes do not break older runs. The default checkpoint root should be directory-based, with the latest step stored in an Orbax-managed subdirectory rather than a single `.pt` file.

### 1.4 Logging and Artifacts

Keep the same artifact types:

- TensorBoard scalars
- TensorBoard images
- checkpoint files
- sample grids
- evaluation plots

The implementation can change, but the visible outputs should remain familiar.

## Phase 2: Shared Math and Distribution Primitives

Port all reusable math first so model code can stay clean.

### Modules to port

- discretized mixture of logistics from `src/dmol.py`
- sampling and log-prob helpers from `src/simple_vae.py` and `src/vae.py`
- categorical and normal distribution wrappers from `src/pgm/layers.py`
- affine and transform utilities used by the PGM code
- EMA logic from `src/utils.py`

### Requirements

- no in-place mutation
- use `jax.numpy` broadcasting rules intentionally
- implement numerically stable log-sum-exp and clipping behavior
- keep the same event shapes and reduction semantics

### Important parity checks

- log-prob values on fixed toy inputs
- sample shapes
- masking and broadcasting behavior
- edge cases around `NaN`, clamping, and lower/upper bounds

## Phase 3: Data Pipeline Port

The data layer must be JAX-friendly but need not be fancy.

### 3.1 Dataset Readers

Port:

- Morpho-MNIST IDX loading
- UKBB CSV + image loading
- any local preprocessing helpers used by the training code

Maintain:

- same normalization options
- same parent-variable concatenation logic
- same train/validation/test split conventions

### 3.2 Augmentation and Batching

Replace TorchVision transforms with local JAX/PIL-based preprocessing:

- resize
- crop with padding
- horizontal flip
- conversion to `float32`
- normalization to `[-1, 1]`

For CPU training, a simple Python data pipeline is acceptable if it preserves semantics and is not a bottleneck.

### 3.3 DataLoader Replacement

Replace `torch.utils.data.DataLoader` with a lightweight JAX-compatible iterator.

Keep support for:

- deterministic shuffling
- multi-epoch iteration
- batch size semantics
- train/valid/test split iteration

If you later want parallel input pipelines, add them after the single-process CPU path is working.

## Phase 4: Image Models

Port the image mechanism before the structured PGM side.

### 4.1 `simple_vae.py`

This file is the easiest model target and a good smoke-test model.

Port:

- encoder/decoder module structure
- reparameterization
- likelihood evaluation
- sampling and abduction behavior
- latent prior/posterior handling

Use this model to validate:

- training step correctness
- checkpoint save/resume
- generation from a fixed seed

### 4.2 `vae.py`

This is the core port.

Preserve:

- hierarchical latent structure
- conditional prior support
- parent conditioning across resolutions
- abduction logic
- forward pass that returns the same named loss components
- `sample`, `abduct`, and `forward_latents` behavior

Implementation notes:

- use Flax modules with explicit parameter collections
- separate deterministic and stochastic paths clearly
- keep parameter initialization close to the Torch version
- match channel layout conventions carefully

### 4.3 Architectural Fidelity

The code must preserve:

- the same architecture strings and presets from `src/hps.py`
- the same encoder/decoder block scheduling
- the same parent concatenation behavior
- the same bottleneck and latent sizing rules

## Phase 5: Structured Causal Model Port

Once the image model is stable, port the SCM side.

### 5.1 `pgm/resnet.py`

Port the ResNet backbone with the same block counts and activations.

### 5.2 `pgm/layers.py`

Port distributional layers and transforms used by the PGM side.

### 5.3 `pgm/flow_pgm.py`

This file contains the dataset-specific causal mechanisms.

Preserve:

- each dataset’s variable ordering
- interventions
- missing-value handling
- supervised and semi-supervised paths
- anticausal predictors

### 5.4 `pgm/dscm.py`

Port the composition layer that merges the image mechanism with the parent mechanisms.

This must preserve:

- observational sampling
- abduction
- counterfactual generation
- parent preprocessing rules

## Phase 6: Training Loops

Port the training logic after the model code compiles.

### 6.1 Image Training

Recreate the behavior of `src/trainer.py`:

- ELBO computation
- gradient accumulation
- gradient clipping
- beta warmup
- EMA updates
- validation cadence
- visualization cadence
- skip-update behavior for unstable gradients

### 6.2 PGM Training

Recreate `src/pgm/train_pgm.py`:

- train/eval loops
- supervised and semi-supervised modes
- metric reduction
- artifact logging

### 6.3 Counterfactual Training

Recreate `src/pgm/train_cf.py`:

- learned predictor loading
- counterfactual sampling
- constraint / Lagrangian optimization
- evaluation on generated counterfactuals

### 6.4 Optimizers and Schedules

Port the optimizer behavior exactly:

- AdamW settings
- weight decay
- LR warmup
- any schedule semantics used by the current code

## Phase 7: CPU-First Execution Model

Keep the first JAX target intentionally simple.

### Baseline runtime goals

- single-process
- single-device
- CPU only
- no distributed primitives
- no TPU/XLA worker abstractions

### CPU-specific priorities

- minimize compilation churn
- avoid unnecessary `jit` boundaries during development
- favor explicit batching over complex pmap/pjit logic
- make memory usage predictable on Mac CPU

### Later optional improvements

Only after the CPU port is stable:

- add `jit` around hot training steps
- add cached compilation for repeated shapes
- consider `vmap` for batch-heavy helper functions

## Phase 8: Verification Plan

Every major port step should be validated against the Torch version.

### 8.1 Unit Parity

Write small tests for:

- distribution functions
- likelihood functions
- random sampling helpers
- tensor shape transforms
- checkpoint round-trips
- EMA updates

### 8.2 Model Parity

For fixed inputs and fixed seeds, compare:

- output shapes
- loss components
- intermediate latent shapes
- sample images
- counterfactual outputs

Exact floating-point equality is not required, but the JAX outputs should be statistically and semantically aligned with the Torch implementation.

### 8.3 Training Parity

Run short smoke trainings and compare:

- loss curves for the first few hundred steps
- gradient norms
- validation metrics
- artifact creation

### 8.4 Acceptance Criteria

The port is complete when:

- the same main workflows run on Mac CPU without PyTorch
- checkpoints can be saved and resumed
- sample generation and counterfactual generation still work
- the preserved metrics and artifact structure are close enough to the current repository to be useful

## Recommended Implementation Order

1. Create the JAX runtime scaffolding and config system.
2. Port utility math and likelihood functions.
3. Port the dataset pipeline.
4. Port `simple_vae` and verify end-to-end training.
5. Port `vae` and the main training loop.
6. Port `pgm/layers.py` and `pgm/resnet.py`.
7. Port `flow_pgm.py` and `dscm.py`.
8. Port `train_pgm.py` and `train_cf.py`.
9. Add parity tests and documentation.

This order gives the fastest path to a working Mac CPU implementation while limiting risk.

## Risks and Mitigations

### Risk: silent behavior drift

Mitigation:

- keep the Torch repo as reference during porting
- add golden tests before each major rewrite
- preserve file-level semantics, not just model names

### Risk: JAX numerics differ from Torch

Mitigation:

- match clamping and reduction behavior carefully
- test log-probabilities and sampling separately
- keep low-level math helpers small and explicit

### Risk: data pipeline performance on CPU

Mitigation:

- start with correctness first
- use simple prefetching and batching
- only optimize after parity is established

### Risk: checkpoint incompatibility

Mitigation:

- define a new versioned checkpoint schema
- document exactly what gets saved
- provide a one-way migration path only if needed

## Deliverables for the Port

- a pure JAX package under `causal-genx/`
- a CPU-first CLI for training and evaluation
- JAX equivalents of all current model and trainer modules
- parity tests against the Torch reference implementation
- updated usage docs for Mac CPU setup

## Final Note

The safest way to do this port is to keep the Torch code as the behavioral contract and move one subsystem at a time into JAX. That gives us a Mac CPU implementation that is still recognizably the same project, not a rewritten approximation.
