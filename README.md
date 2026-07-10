## Causal Generative Modelling: Image Counterfactuals, JAX Port
Forked from [ghif/causal-gen](https://github.com/ghif/causal-gen).

### :hugs: Huggingface demos :hugs:

- [Imaginable Imaging](https://huggingface.co/spaces/mira-causality/imaginable-imaging) (**NEW**✨)
- [Counterfactuals](https://huggingface.co/spaces/mira-causality/counterfactuals)

This repository is the **pure JAX** port of the original Torch/Pyro codebase for the ICML 2023 paper:

>[**High Fidelity Image Counterfactuals with Probabilistic Causal Models**](https://arxiv.org/abs/2306.15764)\
>Fabio De Sousa Ribeiro<sup>1</sup>, Tian Xia<sup>1</sup>, Miguel Monteiro<sup>1</sup>, Nick Pawlowski<sup>2</sup>, Ben Glocker<sup>1</sup>\
><sup>1</sup>Imperial College London, <sup>2</sup>Microsoft Research Cambridge, UK

BibTeX:
```bibtex
@InProceedings{pmlr-v202-de-sousa-ribeiro23a,
  title={High Fidelity Image Counterfactuals with Probabilistic Causal Models},
  author={De Sousa Ribeiro, Fabio and Xia, Tian and Monteiro, Miguel and Pawlowski, Nick and Glocker, Ben},
  booktitle={Proceedings of the 40th International Conference on Machine Learning},
  pages={7390--7425},
  year={2023},
  volume={202},
  series={Proceedings of Machine Learning Research},
  month={23--29 Jul},
  url={https://proceedings.mlr.press/v202/de-sousa-ribeiro23a.html}
}
```

### Example Results

The JAX port follows the same high-level goal as the Torch repository: train a causal image model, abduce latent noise from observations, and generate counterfactual images under interventions.

Current development focus is **MorphoMNIST on Mac CPU**, which is the primary acceptance path for the port.

### Project Structure

```text
📦causal-genx
 ┣ 📜README.md                        # this file
 ┣ 📜requirements.txt                # CPU-only JAX environment dependencies
 ┣ 📜requirements-gpu.txt             # NVIDIA GPU / A100 dependencies
 ┣ 📜requirements-tpu.txt             # Google Cloud TPU dependencies
 ┣ 📜jax-porting.md                  # migration and parity plan
 ┗ 📂src                             # main source code directory
   ┣ 📜__init__.py
   ┣ 📜datasets.py                   # MorphoMNIST dataset loading and preprocessing
   ┣ 📜hps.py                        # hyperparameters and CLI arguments
   ┣ 📜main.py                       # main training entrypoint
   ┣ 📜models.py                     # JAX/Flax image models and lightweight SCM pieces
   ┣ 📜run_local.sh                  # example launcher for local CPU training
   ┣ 📜trainer.py                    # training code for the image mechanism
   ┣ 📜utils.py                      # helpers for logging, checkpointing, EMA, plotting
   ┗ 📂pgm
     ┣ 📜__init__.py
     ┣ 📜dscm.py                     # deep structural causal model composition
     ┣ 📜flow_pgm.py                 # MorphoMNIST structured-variable causal model
     ┣ 📜train_cf.py                 # counterfactual training / evaluation entrypoint
     ┗ 📜train_pgm.py                # structured mechanism training entrypoint
```

### Overview

The original repository used Pyro for the structured causal mechanisms and PyTorch for the image mechanism. This port replaces that stack with a **pure JAX** implementation:

1. JAX / Flax / Optax are used for the image mechanism and training loop.
2. The structured-variable side is implemented in JAX for the MorphoMNIST path first.
3. Counterfactual generation keeps the same overall semantics: abduct latent factors from an observation, apply an intervention, then decode a counterfactual image.

The current port is intentionally CPU-first and keeps the model and artifact layout close to the Torch version so it is easy to compare runs.

### Requirements

Create and activate a backend-specific JAX environment, then install the matching requirements file.

```bash
conda activate med-jax
pip install -r requirements.txt
```

For NVIDIA GPU / A100:

```bash
pip install -r requirements-gpu.txt
```

For Google Cloud TPU:

```bash
pip install -r requirements-tpu.txt
```

The port expects:

- `jax[cpu]`
- `jax[cuda13]` for NVIDIA GPU
- `jax[tpu]` for TPU
- `flax`
- `optax`
- `orbax-checkpoint`
- `tensorboard`
- standard scientific Python packages such as `numpy`, `pandas`, `pillow`, `imageio`, and `matplotlib`

### Data

MorphoMNIST is the primary acceptance dataset for this JAX port.

The default data location is the same GCS bucket used by the Torch repository:

```text
gs://medical-airnd/causal-gen/datasets/morphomnist
```

The training code reads MorphoMNIST directly from that bucket. If you want to point to a different dataset location, pass `--data_dir` explicitly.

The structured-variable and counterfactual code paths are currently centered on MorphoMNIST. The broader UK Biobank and MIMIC-CXR functionality from the Torch repository is preserved as a porting target, but MorphoMNIST is the best-supported path right now.

### Run

To launch local CPU training of the JAX image mechanism, run the launcher from inside `src/`:

```bash
cd causal-genx/src
bash run_local.sh my_experiment
```

The launcher accepts extra arguments and forwards them to `main.py`, so you can override the defaults when needed:

```bash
bash run_local.sh my_experiment --bs 32 --epochs 500 --eval_freq 4
```

To run in the background:

```bash
cd causal-genx/src
nohup bash run_local.sh my_experiment nohup > my_experiment.log 2>&1 &

tail -f my_experiment.log
```

You can also call the entrypoint directly:

```bash
cd causal-genx/src
python main.py --exp_name my_experiment --data_dir gs://medical-airnd/causal-gen/datasets/morphomnist
```

For counterfactual or structured-mechanism training, the matching JAX entrypoints live under `src/pgm/`.

### Current Defaults

- `run_local.sh` activates the `med-jax` conda environment
- MorphoMNIST is loaded from `gs://medical-airnd/causal-gen/datasets/morphomnist`
- checkpoints default to a local `../checkpoints` directory from inside `src/`
- CPU is the intended execution target

### Notes

- This port intentionally keeps the Torch repository as the behavioral reference.
- MorphoMNIST is the main acceptance gate for output parity and smooth CPU execution.
- The JAX code is structured to preserve the same training and counterfactual workflow shape as the original repository, even though the implementation stack is different.

### Extending the Port

If you want to add a new dataset or causal mechanism, the rough flow is:

1. Add the dataset loader in `src/datasets.py`.
2. Add or extend the causal model in `src/pgm/flow_pgm.py`.
3. Adjust hyperparameters and defaults in `src/hps.py`.
4. Train the image mechanism with `src/main.py` and the SCM / counterfactual paths with `src/pgm/train_pgm.py` and `src/pgm/train_cf.py`.

### Checkpointing

The JAX port now uses **Orbax checkpoint directories** for persistence. Each run writes its training state under a checkpoint root inside the experiment folder:

```text
<ckpt_dir>/<hps>/<exp_name>/checkpoints/
```

Resume by pointing `--resume` at that checkpoint directory. Orbax keeps the latest step subdirectory and the associated metadata under that root, so the resume path is a folder rather than a single `.pt` file.

If you are migrating an older script, treat `.pt` checkpoints as legacy-only. The new default is always the Orbax directory layout.

By default, local training also mirrors the full experiment tree to GCS under:

```text
gs://medical-airnd/causal-gen/checkpoints/<hps>/<exp_name>/
```

That means the same Orbax checkpoint root is available locally and in the bucket after each save.

### License

See the original repository for licensing details.
