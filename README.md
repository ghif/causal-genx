## Causal Generative Modelling: Image Counterfactuals
Forked from [ghif/causal-gen](https://github.com/ghif/causal-gen).
### :hugs: Huggingface demos :hugs::

- [Imaginable Imaging](https://huggingface.co/spaces/mira-causality/imaginable-imaging) (**NEW**✨)
- [Counterfactuals](https://huggingface.co/spaces/mira-causality/counterfactuals)


Code for the **ICML 2023** paper:

>[**High Fidelity Image Counterfactuals with Probabilistic Causal Models**](https://arxiv.org/abs/2306.15764)\
>Fabio De Sousa Ribeiro<sup>1</sup>, Tian Xia<sup>1</sup>, Miguel Monteiro<sup>1</sup>, Nick Pawlowski<sup>2</sup>, Ben Glocker<sup>1</sup>\
><sup>1</sup>Imperial College London, <sup>2</sup>Microsoft Research Cambridge, UK

BibTeX:
```
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
### Example Results:
<img src="imgs/ukbb.png" width="100%" height="100%">
<img src="imgs/chest.png" width="100%" height="100%">
<img src="imgs/morpho.png" width="100%" height="100%">

### Project Structure:

```
📦src                                  # main source code directory
 ┣ 📂pgm                               # graphical models for all SCM mechanisms except the image's
 ┃ ┣ 📜dscm.py                         # deep structural causal model Pytorch module
 ┃ ┣ 📜flow_pgm.py                     # Flow mechanisms in Pyro
 ┃ ┣ 📜layers.py                       # utility modules/layers
 ┃ ┣ 📜resnet.py                       # resnet model definition
 ┃ ┣ 📜run.sh                          # example launch script for counterfactual training (slurm)
 ┃ ┣ 📜train_cf.py                     # counterfactual training code
 ┃ ┣ 📜train_pgm.py                    # SCM mechanisms training code (Pyro)
 ┃ ┗ 📜utils_pgm.py                    # graphical model utilities
 ┣ 📜datasets.py                       # dataset definitions
 ┣ 📜dmol.py                           # discretized mixture of logistics likelihood
 ┣ 📜hps.py                            # hyperparameters for all datasets
 ┣ 📜main.py                           # main file
 ┣ 📜run_local.sh                      # example launch script for HVAE causal mechanism training
 ┣ 📜run_slurm.sh                      # same as above but for slurm jobs
 ┣ 📜simple_vae.py                     # single stochastic layer VAE
 ┣ 📜trainer.py                        # training code for image x's causal mechanism
 ┣ 📜train_setup.py                    # training helpers
 ┣ 📜utils.py                          # utilities for training/plotting
 ┗ 📜vae.py                            # HVAE definition; exogenous prior and latent mediator models 
 ```

### Overview
Our deep structural causal models (SCMs) were designed to be modular: in all instances, the causal mechanism for the structured variable (i.e. image $\mathbf{x}$) is trained separately from the other mechanisms in the associated causal graph.
This enables direct and fair comparisons of different causal mechanisms for $\mathbf{x}$ by holding the remaining mechanisms fixed when making comparisons. 

We use the universal probabilistic programming language (PPL) [Pyro](https://pyro.ai/) for the following:

1. Modelling and training all SCM mechanisms except for the image $\mathbf{x}$'s, see code in `src/pgm`; 
2. The counterfactual inference engine, see `src/pgm/flow_pgm.py`; 
3. Proposed constrained counterfactual training technique, see `src/pgm/train_cf.py`.

Pyro enables flexible and expressive deep probabilistic modeling, for more details refer to the [official site](https://pyro.ai/). 

Our HVAE-based causal mechanisms (`src/vae.py`) are trained outside of Pyro using Pytorch, and all trained mechanisms are subsequently merged into a single Pytorch module to create a DSCM. See `src/pgm/dscm.py` for an example.

### Requirements
To run the code you will need to install the requirements listed in the `requirements.txt` file. For Torch-based runs, use the `med-torch` conda environment. For example:
```
conda activate med-torch
pip install -r requirements.txt
```

### Data
For ease of use, we provide the [Morpho-MNIST](https://github.com/dccastro/Morpho-MNIST) dataset we used in `datasets/morphomnist`. For more details on the associated SCM and data-generating process see the source code [here](https://github.com/biomedia-mira/deepscm) and the original DSCM paper [here](https://arxiv.org/abs/2006.06485). 

The Colour-MNIST dataset we used was generated according to [this paper](https://arxiv.org/abs/2303.01274).

Unfortunately, we are unable to share the UK Biobank brain data or the MIMIC-CXR chest x-ray data. 

If you're interested in gaining access, we recommend you check out the specific documents provided. These resources contain all the necessary details regarding the application process, as well as the eligibility criteria. Application and eligibility criteria for gaining access are detailed [here](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access) and [here](https://physionet.org/content/mimic-cxr/2.0.0/) respectively.

### Run
To launch (local) training of the HVAE mechanism simply run the following script from inside the `src` directory:
```
bash run_local.sh your_experiment_name
```
The launcher accepts extra arguments and forwards them to `main.py`, so you can choose the accelerator explicitly:
```
bash run_local.sh your_experiment_name --accelerator cpu
bash run_local.sh your_experiment_name --accelerator cuda
bash run_local.sh your_experiment_name --accelerator mps
bash run_local.sh your_experiment_name --accelerator auto
```
#### TPU v6e-4

On a single-host TPU v6e-4 VM, switch to the TPU branch and install the matched
PyTorch/XLA environment:

```bash
cd /path/to/causal-gen
git switch tpu-v6e-opt

python -m venv .venv
source .venv/bin/activate
pip install -r requirements-tpu.txt
```

Launch Morpho-MNIST training on all four TPU chips from the `src` directory:

```bash
cd src
bash run_tpu.sh morphomnist_v6e
```

The launcher uses PJRT to create one process per chip. `--bs` is the per-chip
batch size, so the default `--bs=32` produces a global batch size of 128 on a
v6e-4 slice. TPU runs use bf16 by default; pass `--precision fp32` for numerical
debugging.

Additional training arguments are forwarded to `main.py`. For example:

```bash
bash run_tpu.sh morphomnist_v6e \
  --bs 64 \
  --epochs 500 \
  --eval_freq 5
```

To use local data and checkpoint directories:

```bash
bash run_tpu.sh morphomnist_v6e \
  --data_dir /path/to/datasets/morphomnist \
  --ckpt_dir /path/to/checkpoints
```

To run training in the background:

```bash
nohup bash run_tpu.sh morphomnist_v6e \
  > morphomnist_v6e.log 2>&1 &

tail -f morphomnist_v6e.log
```

Run the TPU benchmark before a long training job to validate compilation and
measure steady-state throughput:

```bash
PJRT_DEVICE=TPU python tpu_launcher.py benchmark.py \
  --accelerator tpu \
  --precision bf16 \
  --bs 32
```

If a multi-process run fails, isolate compilation from collective communication
with a single-process benchmark:

```bash
PJRT_DEVICE=TPU python tpu_launcher.py benchmark.py \
  --debug_single_process \
  --accelerator tpu \
  --precision bf16 \
  --bs 32
```

If the single-process command succeeds but the normal benchmark fails, inspect
the startup version/topology diagnostics and verify that `torch`, `torch_xla`,
and `libtpu` come from the same supported release. Do not manually set
`TPU_PROCESS_BOUNDS`, `TPU_PROCESS_ADDRESSES`, or `TPU_WORKER_HOSTNAMES`;
PJRT/libtpu discovers the Cloud TPU topology, and the launcher fills in
`ACCELERATOR_TYPE` only when the environment does not provide it.

The generic launcher also supports the Pyro-based training paths:

```
PJRT_DEVICE=TPU python tpu_launcher.py pgm/train_pgm.py --accelerator tpu ...
PJRT_DEVICE=TPU python tpu_launcher.py pgm/train_cf.py --accelerator tpu ...
```

Only the master process writes TensorBoard events, plots, logs, and checkpoints.
The first training steps will be slower while XLA compiles the graph, so measure
throughput after warmup. GCS dataset and checkpoint paths require the TPU VM
service account to have access to the configured buckets. The PGM paths also
depend on Pyro operator coverage in the installed torch-XLA release; use a small
smoke run before a long experiment.

To run in the background you can append `nohup` to the command: `bash run_local.sh your_experiment_name nohup --accelerator mps`. The script still expects the `med-torch` conda environment for Torch-based runs.

If you want to call the training entrypoint directly, use:
```
python main.py --exp_name your_experiment_name --accelerator mps
```

The same accelerator flag is available in `src/pgm/train_pgm.py` and `src/pgm/train_cf.py` for the PGM/counterfactual training paths. Hyperparameters can be found in `src/hps.py`.

If using [Slurm Workload Manager](https://slurm.schedmd.com/documentation.html), adjust `src/run_slurm.sh` as needed and launch as `bash run_slurm.sh`. You can pass the accelerator there by editing the script or by calling the underlying Python entrypoint directly with `--accelerator`.

Example (loose) steps to add your own dataset and associated SCM:

1. Add dataset class definition to `src/datasets` and setup the dataloader in `src/train_setup.py`
2. Add associated causal graph and mechanism definitions in `src/pgm/flow_pgm.py`
3. Adjust HVAE hyperparameters needed for your dataset (input resolution, architecture, etc) in `src/hps.py`
4. Train the HVAE mechanism as above, and train all other mechanisms (separately) using `src/pgm/train_pgm.py`

Note: `src/pgm/train_cf.py` implements the optional counterfactual training/fine-tuning procedure outlined in Section 3.4 of the paper. This step may not be necessary if the model already performs well enough at counterfactual inference.

If you'd like to make the HVAE more lightweight you can try reducing the number of blocks at each resolution and reducing the block width (hyperparameters `enc_arch`, `dec_arch`, and `width` found in `src/hps.py`). The block `version == "light"` in `src/vae.py` also uses half as much VRAM.

To resume training from a checkpoint simply adjust the argument: `--resume=/path/to/your/checkpoint.pt`.
