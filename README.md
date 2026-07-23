# Causal-GenX

Causal-GenX is a native JAX/Flax implementation of causal generative image
modelling and counterfactual generation. It is a complete port of the original
PyTorch/Pyro research code—not a fork that depends on the original runtime.

The current reference workflow is MorphoMNIST. It trains a causal SCM, an
image-to-variable predictor, and a conditional VAE/HVAE before fine-tuning the
image mechanism for counterfactual generation. JAX/Flax, Optax, Orbax, and
TensorBoard are used throughout.

## Start here

Create a backend-appropriate environment and install dependencies:

```bash
conda activate med-jax
pip install -r requirements.txt            # CPU
# pip install -r requirements-gpu.txt      # NVIDIA GPU
# pip install -r requirements-tpu.txt      # Google Cloud TPU
```

Check a config before starting a job:

```bash
python scripts/run.py train-image-model \
  --config configs/morphomnist_image_model.yaml --dry-run
```

All researcher-facing operations use one entrypoint, `scripts/run.py`. Configs
are standalone YAML files; append `section.key=value` to override a setting.

## Research workflow

Run the stages in this order for a new MorphoMNIST experiment:

```bash
python scripts/run.py train-scm \
  --config configs/morphomnist_scm.yaml

python scripts/run.py train-predictor \
  --config configs/morphomnist_predictor.yaml

python scripts/run.py train-image-model \
  --config configs/morphomnist_image_model.yaml

python scripts/run.py finetune-counterfactual \
  --config configs/morphomnist_counterfactual.yaml

python scripts/run.py infer \
  --config configs/morphomnist_inference.yaml
```

The counterfactual config names its three upstream artifacts explicitly:
`workflow.scm_checkpoint`, `workflow.predictor_checkpoint`, and
`workflow.image_model_checkpoint`. The inference reference config loads the
verified GCS HVAE run at
`gs://medical-airnd/causal-gen/checkpoints/morphomnist/hvae_jax-cpu_22-07-2026`.

MorphoMNIST defaults to
`gs://medical-airnd/causal-gen/datasets/morphomnist`. Override `dataset.root`
to use a local or alternate GCS dataset location. GPU and TPU image-model
profiles are available in `configs/morphomnist_image_model_gpu.yaml` and
`configs/morphomnist_image_model_tpu_v6e4.yaml`; runtime preflight fails if the
requested topology is unavailable.

## Project structure

```text
scripts/
  run.py                         # only training/inference entrypoint
  morphomnist_visualizer.py      # optional interactive inspection tool
configs/
  morphomnist_{scm,predictor,image_model,counterfactual,inference}.yaml
src/
  data/                          # MorphoMNIST loading and parent encoding
  causal/                        # SCM, predictor, schema-driven DSCM
  models/                        # VAE/HVAE implementations
  training/                      # five stage modules and shared loops
  config.py                      # typed YAML validation and overrides
  runtime.py                     # JAX backend/device setup
  utils.py                       # Orbax, logging, visualization, GCS helpers
tests/
  unit/ contract/ integration/   # configuration, boundary, and runner tests
  test_*_parity.py               # numerical regression tests
```

`src/` contains reusable code only. Do not add another CLI under `src/`; add a
typed config and route a new public operation through `scripts/run.py`.

## Configs and artifacts

Every config specifies dataset, causal schema, model, optimizer, runtime,
artifact location, seed, and workflow inputs. `--dry-run` validates that schema
without reading data or importing a stage implementation.

Runs are stored under:

```text
checkpoints/<dataset>/<run-name>/
```

Training artifacts include Orbax checkpoints, `hparams.json`, TensorBoard
events, logs, and stage-specific PDFs or image previews. If `artifacts.remote_root`
is configured, the run tree is mirrored to GCS. Existing JAX/Orbax artifacts
are supported; `.pt` and `.pkl` checkpoints are intentionally unsupported.

For historical GCS copies that contain a completed payload but no Orbax
completion marker, set the workflow’s `trust_incomplete_checkpoint: true`.
This is explicit in the reference counterfactual and inference configs.

## Benchmark summary: JAX versus PyTorch

Recorded MorphoMNIST benchmarks show substantially higher throughput for the
native JAX implementation than the earlier PyTorch/Pyro implementation.
Throughput is reported as end-to-end or epoch-level `samples/s`, rather than
raw asynchronous dispatch timing.

| Platform | Workload and batch size | PyTorch | Native JAX | Observed ratio | Comparison context |
|---|---|---:|---:|---:|---|
| CPU | Parent PGM, batch size 16 | 3,563 samples/s | 25,685 samples/s | ~7.2x | Same MorphoMNIST hyperparameters |
| CPU | HVAE image training, batch size 32 | 17.1 samples/s | 179.4 samples/s | ~10.5x | JAX late-stage steady state versus PyTorch full loop |
| GPU | HVAE image training | 750.7 samples/s (A100, batch size 128) | 4,769.4 samples/s (G4, batch size 256) | ~6.4x | Throughput indication only; hardware and batch size differ |

The PGM row is the closest controlled framework comparison. The remaining
rows are useful operational signals, not bitwise-identical experiments. For a
new comparison, match dataset split, batch size, precision, model and optimizer
settings, checkpoint/logging cadence, and a post-compilation timing window;
report samples/s alongside loss or ELBO rather than treating throughput alone
as evidence of numerical equivalence.

JAX also accelerates the TPU path: the captured MorphoMNIST HVAE runs reached
about 382 samples/s on TPU v6e-1 at batch size 128 and about 7,418 samples/s
on TPU v6e-4 at batch size 512 after warmup. These are JAX scaling results,
not PyTorch speedups, because no matching PyTorch TPU measurement is available.

## Development

Run the suite from the repository root:

```bash
PYTHONPATH=src pytest -q
```

When changing a stage, run its parity tests and add a unit or integration test.
Keep numerical changes isolated; checkpoint payload fields and run layouts are
part of the research artifact contract.

To add a dataset, implement its provider under `src/data/`, define its schema
and parent encoding, add causal/image mechanisms under `src/causal/` or
`src/models/`, provide a standalone config, and add a small contract fixture.

## Reference

Causal-GenX ports the methods introduced in:

> Fabio De Sousa Ribeiro, Tian Xia, Miguel Monteiro, Nick Pawlowski, and Ben
> Glocker. *High Fidelity Image Counterfactuals with Probabilistic Causal
> Models.* ICML 2023.

- [Paper](https://arxiv.org/abs/2306.15764)
- [Original PyTorch/Pyro code](https://github.com/biomedia-mira/causal-gen)
- [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v202/de-sousa-ribeiro23a.html)
