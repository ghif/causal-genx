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

The JAX port follows the same high-level goal as the Torch repository, but the MorphoMNIST workflow is best understood as a **four-stage causal pipeline**:

1. learn the structured causal variables for the dataset
2. train the auxiliary image-to-parent predictor
3. train the image model conditioned on those variables
4. combine all trained pieces to generate counterfactual images under interventions

Current development focus is **MorphoMNIST on Mac CPU**, which is the primary acceptance path for the port and the clearest end-to-end example of that pipeline.

### Project Structure

```text
📦causal-genx
 ┣ 📜README.md                        # this file
 ┣ 📜requirements.txt                # CPU-only JAX environment dependencies
 ┣ 📜requirements-gpu.txt             # NVIDIA GPU / A100 dependencies
 ┣ 📜requirements-tpu.txt             # Google Cloud TPU dependencies
 ┣ 📜jax-porting.md                  # migration and parity plan
 ┣ 📂scripts
 ┃ ┣ 📜run.py                        # sole public runner for the four stages
 ┃ ┗ 📜morphomnist_visualizer.py
 ┣ 📂configs                         # one complete YAML per training stage
 ┃ ┣ 📜morphomnist_scm.yaml
 ┃ ┣ 📜morphomnist_predictor.yaml
 ┃ ┣ 📜morphomnist_image_model.yaml
 ┃ ┗ 📜morphomnist_counterfactual.yaml
 ┗ 📂src                             # reusable lower-level implementation
   ┣ 📂data                          # MorphoMNIST provider and parent conditioning
   ┣ 📂causal                        # SCM, predictor, and deep-SCM mechanisms
   ┣ 📂models                        # VAE/HVAE image model
   ┣ 📂training                      # SCM → predictor → image → CF stage modules
   ┣ 📜artifacts.py                  # metadata and compatibility validation
   ┣ 📜config.py                     # typed stage configuration
   ┗ 📂pgm                           # deprecated compatibility modules
```

### Overview

The original repository used Pyro for the structured causal mechanisms and PyTorch for the image mechanism. This port replaces that stack with a **pure JAX** implementation, but it preserves the same causal workflow:

1. Train the SCM for structured causal variables.
2. Train the auxiliary image-to-parent predictor.
3. Train the image model conditioned on the causal variables.
4. Fine-tune the image model for counterfactual generation.

In other words, the image model is only one stage in the pipeline. The full MorphoMNIST story is:

- parent variables such as digit, thickness, and intensity are learned as causal mechanisms
- the image `x` is then modeled conditional on those parents
- counterfactual images are produced by keeping abducted latent factors fixed while intervening on the parents

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

The primary interface is the single experiment runner. Each file in `configs/`
is a complete, standalone experiment definition; override a value with
`section.key=value` when needed.

```bash
python scripts/run.py train-scm --config configs/morphomnist_scm.yaml
python scripts/run.py train-predictor --config configs/morphomnist_predictor.yaml
python scripts/run.py train-image-model --config configs/morphomnist_image_model.yaml
python scripts/run.py finetune-counterfactual --config configs/morphomnist_counterfactual.yaml
```

Use `--dry-run` to validate a config without accessing data or starting a job.
The older `src/` entrypoints and shell launchers remain available for existing
automation but are deprecated compatibility paths.

For MorphoMNIST, the recommended training order is:

1. train the parent SCM
2. train the auxiliary predictor
3. train the image mechanism
4. run counterfactual composition / evaluation

The scripts below match that order.

#### 1. Train the parent SCM

Run the structured-variable model first:

```bash
python scripts/run.py train-scm --config configs/morphomnist_scm.yaml
```

This learns the causal variables for MorphoMNIST, namely digit, thickness, and intensity.

#### 2. Train the auxiliary predictor

This is a separate image-to-parent prediction job. It does not train or update the SCM.

```bash
python scripts/run.py train-predictor --config configs/morphomnist_predictor.yaml
```

#### 3. Train the image mechanism

```bash
python scripts/run.py train-image-model --config configs/morphomnist_image_model.yaml
```

To launch local CPU training of the JAX image mechanism, run the launcher from inside `src/`:

```bash
cd causal-genx/src
bash run_local.sh my_experiment
```

The launcher accepts extra arguments and forwards them to `main.py`, so you can override the defaults when needed:

```bash
bash run_local.sh my_experiment --bs 32 --epochs 500 --eval_freq 4 --checkpoint_freq 50 --viz_batch_size 32
```

To run in the background:

```bash
cd causal-genx/src
nohup bash run_local.sh my_experiment nohup > my_experiment.log 2>&1 &

tail -f my_experiment.log
```

For NVIDIA GPU / A100, use the GPU launcher:

```bash
cd causal-genx/src
bash run_gpu.sh my_experiment
```

To override GPU defaults:

```bash
cd causal-genx/src
bash run_gpu.sh my_experiment --bs 128 --precision bf16 --eval_freq 4 --checkpoint_freq 4 --viz_batch_size 32
```

For Google Cloud TPU, use the TPU launcher:

```bash
cd causal-genx/src
bash run_tpu.sh my_experiment
```

To override TPU defaults:

```bash
cd causal-genx/src
bash run_tpu.sh my_experiment --bs 32 --precision bf16 --eval_freq 4 --checkpoint_freq 4 --viz_batch_size 32
```

You can also call the entrypoint directly:

```bash
cd causal-genx/src
python main.py --exp_name my_experiment --data_dir gs://medical-airnd/causal-gen/datasets/morphomnist
```

#### 4. Run counterfactual composition

After the SCM, predictor, and image-mechanism checkpoints exist, run:

```bash
cd causal-genx/src
python pgm/train_cf.py --hps morphomnist --exp_name morphomnist_cf
```

Point `--pgm_path` and `--vae_path` at the parent-SCM and image-model checkpoint roots you want to combine. This script does not train a new model from scratch; it loads the two trained pieces and performs counterfactual abduction + intervention.

For counterfactual or structured-mechanism training, the matching JAX entrypoints live under `src/pgm/`.

`run_gpu.sh` is the GPU launcher for the `main.py` image-model training job. It is triggered when you explicitly run it from the shell, for example:

```bash
cd causal-genx/src
bash run_gpu.sh my_experiment
```

It is not called automatically by `main.py`, `train_pgm.py`, or `train_cf.py`.

### Current Defaults

- `run_local.sh` activates the `med-jax` conda environment
- `run_gpu.sh` launches `main.py` with `--accelerator=gpu` and `--precision=bf16`
- `run_tpu.sh` launches `main.py` with `--accelerator=tpu` and `--precision=bf16`
- MorphoMNIST is loaded from `gs://medical-airnd/causal-gen/datasets/morphomnist`
- checkpoints default to a local `../checkpoints` directory from inside `src/`
- `eval_freq` controls how often validation is reported, while `checkpoint_freq` controls how often checkpoint saving is even eligible; a checkpoint is written only when the validation loss improves on that scheduled epoch
- CPU is the intended execution target

### Notes

- This port intentionally keeps the Torch repository as the behavioral reference.
- MorphoMNIST is the main acceptance gate for output parity and smooth CPU execution.
- The JAX code is structured to preserve the same causal workflow shape as the original repository: parent SCM first, image model second, counterfactual composition last.

### Extending the Port

If you want to add a new dataset or causal mechanism, the rough flow is:

1. Add the dataset loader in `src/datasets.py`.
2. Add or extend the causal model in `src/pgm/flow_pgm.py`.
3. Adjust hyperparameters and defaults in `src/hps.py`.
4. Train the parent SCM with `src/pgm/train_pgm.py`.
5. Train the image mechanism with `src/main.py`.
6. Use `src/pgm/train_cf.py` to compose the trained pieces for counterfactual evaluation.

### Checkpointing

The JAX port now uses **Orbax checkpoint directories** for persistence. Each run writes its training state under a checkpoint root inside the experiment run folder:

```text
<ckpt_dir>/<hps>/<exp_name>/checkpoints/
```

Resume by pointing `--resume` at that checkpoint directory. Orbax keeps the latest step subdirectory and the associated metadata under that root, so the resume path is a folder rather than a single `.pt` file.

If you are migrating an older script, treat `.pt` checkpoints as legacy-only. The new default is always the Orbax directory layout.

### Resume Training

To resume from the latest checkpoint, point `--resume` at the checkpoint root directory for the run. Orbax will pick the latest saved step inside that directory automatically.

Local example:

```bash
cd causal-genx/src
bash run_local.sh my_experiment --resume ../checkpoints/morphomnist/my_experiment/checkpoints
```

GPU example:

```bash
cd causal-genx/src
bash run_gpu.sh my_experiment --resume ../checkpoints/morphomnist/my_experiment/checkpoints
```

TPU example:

```bash
cd causal-genx/src
bash run_tpu.sh my_experiment --resume ../checkpoints/morphomnist/my_experiment/checkpoints
```

If you are resuming from the mirrored GCS tree, use the remote checkpoint root instead:

```bash
--resume gs://medical-airnd/causal-gen/checkpoints/morphomnist/my_experiment/checkpoints
```

By default, local training also mirrors the full experiment run tree to GCS under:

```text
gs://medical-airnd/causal-gen/checkpoints/<hps>/<exp_name>/
```

That means the same Orbax checkpoint root is available locally and in the bucket after each save, and the checkpoint data stays under the run's `checkpoints/` subfolder in both places.

Visualization artifacts follow the same training-step convention and are written as:

```text
<save_dir>/viz-step-<training_step>.png
```

For example, the end-of-epoch validation image for global step 400 is saved as `viz-step-400.png`.

### License

See the original repository for licensing details.
