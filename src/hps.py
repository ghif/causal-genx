from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List


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
    accelerator="cpu",
    precision="fp32",
    lr=1e-3,
    bs=32,
    wd=0.01,
    z_dim=16,
    input_res=32,
    input_channels=1,
    pad=4,
    enc_arch="32b3d2,16b3d2,8b3d2,4b3d4,1b4",
    dec_arch="1b4,4b4,8b4,16b4,32b4",
    widths=[16, 32, 64, 128, 256],
    bottleneck=4,
    parents_x=["thickness", "intensity", "digit"],
    concat_pa=True,
    context_norm="[-1,1]",
    context_dim=12,
    data_dir="gs://medical-airnd/causal-gen/datasets/morphomnist",
    hps="morphomnist",
    vae="hierarchical",
    x_like="none_dgauss",
    std_init=0.1,
    q_correction=True,
    beta=1.0,
    kl_free_bits=0.0,
    beta_warmup_steps=0,
    grad_clip=350.0,
    grad_skip=500.0,
    accu_steps=1,
    eval_freq=5,
    viz_freq=10000,
    seed=7,
    deterministic=False,
    ema_rate=0.999,
    exp_name="",
    ckpt_dir="checkpoints",
    remote_ckpt_dir="gs://medical-airnd/causal-gen/checkpoints",
    resume="",
)


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--accelerator", type=str, default="cpu", choices=["cpu", "gpu", "tpu"])
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="gs://medical-airnd/causal-gen/datasets/morphomnist")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--remote_ckpt_dir", type=str, default="gs://medical-airnd/causal-gen/checkpoints")
    parser.add_argument("--hps", type=str, default="morphomnist")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=str, default="false")
    parser.add_argument("--persistent_workers", type=str, default="false")
    parser.add_argument("--prefetch_factor", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--betas", nargs="+", type=float, default=[0.9, 0.9])
    parser.add_argument("--ema_rate", type=float, default=0.999)
    parser.add_argument("--input_res", type=int, default=32)
    parser.add_argument("--input_channels", type=int, default=1)
    parser.add_argument("--pad", type=int, default=4)
    parser.add_argument("--hflip", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=350.0)
    parser.add_argument("--grad_skip", type=float, default=500.0)
    parser.add_argument("--accu_steps", type=int, default=1)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--beta_warmup_steps", type=int, default=0)
    parser.add_argument("--kl_free_bits", type=float, default=0.0)
    parser.add_argument("--viz_freq", type=int, default=10000)
    parser.add_argument("--speed_log_freq", type=int, default=50)
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--checkpoint_smoke_test", action="store_true", default=False)
    parser.add_argument("--checkpoint_smoke_steps", type=int, default=1)
    parser.add_argument("--vae", type=str, default="hierarchical", choices=["simple", "hierarchical"])
    parser.add_argument("--enc_arch", type=str, default="32b3d2,16b3d2,8b3d2,4b3d4,1b4")
    parser.add_argument("--dec_arch", type=str, default="1b4,4b4,8b4,16b4,32b4")
    parser.add_argument("--cond_prior", action="store_true", default=False)
    parser.add_argument("--widths", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    parser.add_argument("--bottleneck", type=int, default=4)
    parser.add_argument("--z_dim", type=int, default=16)
    parser.add_argument("--z_max_res", type=int, default=192)
    parser.add_argument("--bias_max_res", type=int, default=192)
    parser.add_argument("--context_dim", type=int, default=12)
    parser.add_argument("--context_norm", type=str, default="[-1,1]")
    parser.add_argument("--parents_x", nargs="+", default=["thickness", "intensity", "digit"])
    parser.add_argument("--concat_pa", action="store_true", default=True)
    parser.add_argument("--x_like", type=str, default="none_dgauss")
    parser.add_argument("--std_init", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="morphomnist")
    return parser


def setup_hparams(parser: argparse.ArgumentParser) -> Hparams:
    args = parser.parse_known_args()[0]
    hp = Hparams()
    preset = HPARAMS_REGISTRY.get(args.hps, HPARAMS_REGISTRY["morphomnist"])
    hp.update(preset.__dict__)
    hp.update(vars(args))
    return hp
