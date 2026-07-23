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
 ┃ ┣ 📜run.py                        # sole public runner for training and inference
 ┃ ┗ 📜morphomnist_visualizer.py
 ┣ 📂configs                         # one complete YAML per training stage
 ┃ ┣ 📜morphomnist_scm.yaml
 ┃ ┣ 📜morphomnist_predictor.yaml
 ┃ ┣ 📜morphomnist_image_model.yaml
 ┃ ┣ 📜morphomnist_counterfactual.yaml
 ┃ ┗ 📜morphomnist_inference.yaml
 ┗ 📂src                             # reusable lower-level implementation
   ┣ 📂data                          # MorphoMNIST provider and parent conditioning
   ┣ 📂causal                        # SCM, predictor, and deep-SCM mechanisms
   ┣ 📂models                        # VAE/HVAE image model
   ┣ 📂training                      # SCM → predictor → image → CF stage modules
   ┣ 📜artifacts.py                  # metadata and compatibility validation
   ┣ 📜config.py                     # typed stage configuration
   ┗ 📂training                      # five named workflow modules
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

For a single-VM TPU v6e-4 image-model run, install the TPU dependencies on
the TPU VM and use the topology-specific config:

```bash
pip install -r requirements-tpu.txt
python scripts/run.py train-image-model \
  --config configs/morphomnist_image_model_tpu_v6e4.yaml
```

The runner performs a hardware preflight before importing the training stage.
The v6e-4 profile requires backend `tpu`, four local/global devices, one JAX
process, BF16 inputs, and replicated data-parallel training with a global batch
of 512 (128 samples per chip). It fails instead of silently falling back to CPU.
Use an explicit override such as `artifacts.run_name=my_v6e4_run`; do not reuse a
completed run name.

Use `--dry-run` to validate a config without accessing data or starting a job.
`scripts/run.py` is the only supported entrypoint.

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

Use YAML overrides for run-specific values, for example
`artifacts.run_name=my_experiment runtime.accelerator=cpu`.

#### 4. Run counterfactual composition

After the SCM, predictor, and image-mechanism checkpoints exist, run:

```bash
python scripts/run.py finetune-counterfactual \
  --config configs/morphomnist_counterfactual.yaml \
  artifacts.run_name=morphomnist_cf
```

Set `workflow.scm_checkpoint`, `workflow.predictor_checkpoint`, and
`workflow.image_model_checkpoint` in the standalone config to the three
upstream checkpoint roots. The stage writes its `cf/` artifact layout,
checkpoint payload, logs, TensorBoard events, and GCS synchronization.

Relative checkpoint paths are resolved locally first, then below
`artifacts.remote_root`; use an explicit `gs://...` path to require GCS. Set
`workflow.trust_incomplete_checkpoint: true` only for a known historical mirror
whose Orbax payload is present but lacks `commit_success.txt`; the run records
that recovery choice in its logs and checkpoint hparams.

#### 5. Run inference

```bash
python scripts/run.py infer --config configs/morphomnist_inference.yaml
```

The reference config reads
`gs://medical-airnd/causal-gen/checkpoints/morphomnist/hvae_jax-cpu_22-07-2026`.

### Current Defaults

- MorphoMNIST is loaded from `gs://medical-airnd/causal-gen/datasets/morphomnist`
- checkpoints default to `checkpoints/` from the repository root
- `eval_freq` controls validation and visualization, while `checkpoint_freq` independently controls periodic training-state saves
- CPU is the intended execution target

### Notes

- This port intentionally keeps the Torch repository as the behavioral reference.
- MorphoMNIST is the main acceptance gate for output parity and smooth CPU execution.
- The JAX code is structured to preserve the same causal workflow shape as the original repository: parent SCM first, image model second, counterfactual composition last.

### Extending the Port

If you want to add a new dataset or causal mechanism, the rough flow is:

1. Add the dataset loader in `src/data/`.
2. Add or extend the causal model in `src/causal/`.
3. Define a complete standalone config in `configs/`.
4. Run the required stages through `scripts/run.py`.

### Checkpointing

The JAX port now uses **Orbax checkpoint directories** for persistence. Each run writes its training state under a checkpoint root inside the experiment run folder:

```text
<checkpoint_root>/<dataset>/<run_name>/checkpoints/
```

Resume paths are configured as `workflow.resume` or
`workflow.resume_checkpoint`. Orbax keeps the latest step and metadata under
that root; `.pt` and `.pkl` files are not supported.

### Resume Training

Set `workflow.resume` (image model) or `workflow.resume_checkpoint`
(counterfactual) to a local or GCS Orbax checkpoint root. The loader selects
the latest valid step.

By default, local training also mirrors the full experiment run tree to GCS under:

```text
gs://medical-airnd/causal-gen/checkpoints/<dataset>/<run_name>/
```

That means the same Orbax checkpoint root is available locally and in the bucket after each save, and the checkpoint data stays under the run's `checkpoints/` subfolder in both places.

Visualization artifacts follow the same training-step convention and are written as:

```text
<save_dir>/viz-<training_step>.png
```

For example, the end-of-epoch validation image for global step 400 is saved as `viz-400.png`.

### License

See the original repository for licensing details.
