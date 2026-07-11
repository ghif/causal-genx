from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Hparams:
    def update(self, values: Dict[str, Any]):
        for k, v in values.items():
            setattr(self, k, v)


HPARAMS_REGISTRY: Dict[str, Hparams] = {}


def _make_hparams(**kwargs) -> Hparams:
    hp = Hparams()
    for k, v in kwargs.items():
        setattr(hp, k, v)
    return hp


HPARAMS_REGISTRY["morphomnist"] = _make_hparams(
    accelerator="cpu",  # default backend for local development
    precision="fp32",  # full precision for parity and stability
    lr=1e-3,  # AdamW learning rate
    bs=32,  # global batch size
    wd=0.01,  # AdamW weight decay
    z_dim=16,  # latent dimensionality
    input_res=32,  # image resolution
    input_channels=1,  # grayscale input channels
    pad=4,  # random crop padding during training
    enc_arch="32b3d2,16b3d2,8b3d2,4b3d4,1b4",  # encoder block layout
    dec_arch="1b4,4b4,8b4,16b4,32b4",  # decoder block layout
    widths=[16, 32, 64, 128, 256],  # channel widths per scale
    bottleneck=4,  # bottleneck reduction factor inside blocks
    parents_x=["thickness", "intensity", "digit"],  # parent variables concatenated to x
    concat_pa=True,  # concatenate parents spatially to feature maps
    context_norm="[-1,1]",  # normalize parent variables to [-1, 1]
    context_dim=12,  # parent/context feature dimension
    data_dir="gs://medical-airnd/causal-gen/datasets/morphomnist",  # MorphoMNIST dataset location
    hps="morphomnist",  # hyperparameter preset name
    vae="hierarchical",  # hierarchical VAE variant
    x_like="diag_dgauss",  # diagonal discretized Gaussian likelihood
    std_init=0.0,  # likelihood log-scale init; 0 keeps random conv init
    q_correction=False,  # disable posterior correction path
    beta=1.0,  # KL weight
    kl_free_bits=0.0,  # free-bits threshold for KL
    beta_warmup_steps=0,  # disable beta warmup in preset
    grad_clip=350.0,  # global grad-norm clipping threshold
    grad_skip=500.0,  # skip update if grad norm exceeds this
    accu_steps=1,  # gradient accumulation steps
    viz_bs=32,  # compatibility alias for legacy visualization batch size
    viz_batch_size=32,  # number of samples used for visualization
    eval_freq=5,  # run validation every N epochs
    viz_freq=10000,  # visualization frequency in epochs
    seed=7,  # random seed
    deterministic=False,  # allow nondeterministic execution
    ema_rate=0.999,  # EMA decay for evaluation weights
    exp_name="",  # experiment name
    ckpt_dir="checkpoints",  # local checkpoint root
    remote_ckpt_dir="gs://medical-airnd/causal-gen/checkpoints",  # mirrored checkpoint root
    resume="",  # checkpoint root to resume from
)


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--accelerator", type=str, default="cpu", choices=["cpu", "gpu", "tpu"])  # backend selector
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16"])  # numeric precision mode
    parser.add_argument("--exp_name", type=str, default="")  # run name used for checkpoints and logs
    parser.add_argument("--data_dir", type=str, default="gs://medical-airnd/causal-gen/datasets/morphomnist")  # dataset path
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")  # local checkpoint output directory
    parser.add_argument("--remote_ckpt_dir", type=str, default="gs://medical-airnd/causal-gen/checkpoints")  # remote mirror for checkpoints
    parser.add_argument("--hps", type=str, default="morphomnist")  # preset name to load
    parser.add_argument("--resume", type=str, default="")  # checkpoint directory to resume from
    parser.add_argument("--seed", type=int, default=7)  # RNG seed
    parser.add_argument("--deterministic", action="store_true", default=False)  # favor deterministic kernels when possible
    parser.add_argument("--epochs", type=int, default=5000)  # total training epochs
    parser.add_argument("--bs", type=int, default=32)  # global batch size
    parser.add_argument("--num_workers", type=int, default=0)  # dataloader worker count
    parser.add_argument("--pin_memory", type=str, default="false")  # pin host memory for data loading
    parser.add_argument("--persistent_workers", type=str, default="false")  # keep dataloader workers alive
    parser.add_argument("--prefetch_factor", type=int, default=-1)  # dataloader prefetch depth
    parser.add_argument("--lr", type=float, default=1e-3)  # optimizer learning rate
    parser.add_argument("--lr_warmup_steps", type=int, default=100)  # warmup duration for learning rate
    parser.add_argument("--wd", type=float, default=0.01)  # optimizer weight decay
    parser.add_argument("--betas", nargs="+", type=float, default=[0.9, 0.9])  # AdamW beta1 and beta2
    parser.add_argument("--ema_rate", type=float, default=0.999)  # EMA decay for eval weights
    parser.add_argument("--input_res", type=int, default=32)  # input image resolution
    parser.add_argument("--input_channels", type=int, default=1)  # number of image channels
    parser.add_argument("--pad", type=int, default=4)  # training crop padding
    parser.add_argument("--hflip", type=float, default=0.5)  # horizontal flip probability
    parser.add_argument("--grad_clip", type=float, default=350.0)  # gradient clipping threshold
    parser.add_argument("--grad_skip", type=float, default=500.0)  # skip updates above this grad norm
    parser.add_argument("--accu_steps", type=int, default=1)  # gradient accumulation steps
    parser.add_argument("--beta", type=float, default=1.0)  # KL multiplier
    parser.add_argument("--beta_warmup_steps", type=int, default=0)  # beta warmup duration
    parser.add_argument("--kl_free_bits", type=float, default=0.0)  # free-bits threshold for KL
    parser.add_argument("--viz_bs", type=int, default=32)  # compatibility alias for legacy visualization batch size
    parser.add_argument("--viz_batch_size", type=int, default=32)  # samples used for visual grids
    parser.add_argument("--speed_log_freq", type=int, default=50)  # step logging frequency
    parser.add_argument("--eval_freq", type=int, default=5)  # validation frequency in epochs
    parser.add_argument("--checkpoint_smoke_test", action="store_true", default=False)  # enable checkpoint smoke test
    parser.add_argument("--checkpoint_smoke_steps", type=int, default=1)  # smoke-test step threshold
    parser.add_argument("--vae", type=str, default="hierarchical", choices=["simple", "hierarchical"])  # VAE variant
    parser.add_argument("--enc_arch", type=str, default="32b3d2,16b3d2,8b3d2,4b3d4,1b4")  # encoder architecture spec
    parser.add_argument("--dec_arch", type=str, default="1b4,4b4,8b4,16b4,32b4")  # decoder architecture spec
    parser.add_argument("--cond_prior", action="store_true", default=False)  # use conditional prior
    parser.add_argument("--widths", nargs="+", type=int, default=[16, 32, 64, 128, 256])  # channel widths by scale
    parser.add_argument("--bottleneck", type=int, default=4)  # block bottleneck divisor
    parser.add_argument("--z_dim", type=int, default=16)  # latent dimension
    parser.add_argument("--z_max_res", type=int, default=192)  # max resolution with stochastic latents
    parser.add_argument("--bias_max_res", type=int, default=64)  # max resolution with learned bias maps
    parser.add_argument("--context_dim", type=int, default=12)  # parent/context feature dimension
    parser.add_argument("--context_norm", type=str, default="[-1,1]")  # parent normalization range
    parser.add_argument("--parents_x", nargs="+", default=["thickness", "intensity", "digit"])  # parent variables
    parser.add_argument("--concat_pa", action="store_true", default=True)  # concatenate parent maps to activations
    parser.add_argument("--x_like", type=str, default="diag_dgauss")  # image likelihood type
    parser.add_argument("--std_init", type=float, default=0.0)  # likelihood std init
    parser.add_argument("--q_correction", action="store_true", default=False)  # enable posterior correction
    parser.add_argument("--dataset", type=str, default="morphomnist")  # dataset identifier
    return parser


def setup_hparams(parser: argparse.ArgumentParser) -> Hparams:
    args = parser.parse_known_args()[0]
    hp = Hparams()
    preset = HPARAMS_REGISTRY.get(args.hps, HPARAMS_REGISTRY["morphomnist"])
    hp.update(preset.__dict__)
    hp.update(vars(args))
    return hp
