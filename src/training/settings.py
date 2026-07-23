"""Typed runtime settings derived from standalone experiment configs.

Pydantic models validate user-facing YAML. These dataclasses are the mutable
runtime view used by training loops for generated paths and resume metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from config import CounterfactualTrainingConfig, ExperimentConfig, ImageModelTrainingConfig
from data.morphomnist import MORPHOMNIST_SCHEMA


@dataclass
class ImageModelSettings:
    accelerator: str
    precision: str
    dataset: str
    data_dir: str
    ckpt_dir: str
    remote_ckpt_dir: str
    exp_name: str
    seed: int
    epochs: int
    bs: int
    lr: float
    wd: float
    lr_warmup_steps: int
    betas: list[float]
    input_res: int
    pad: int
    hflip: float
    parents_x: list[str]
    context_dim: int
    cond_prior: bool
    enc_arch: str
    dec_arch: str
    widths: list[int]
    bottleneck: int
    z_dim: int
    z_max_res: int
    bias_max_res: int
    x_like: str
    std_init: float
    q_correction: bool
    kl_free_bits: float
    vae: str
    speed_log_freq: int
    viz_batch_size: int
    eval_freq: int
    checkpoint_freq: int
    resume: str
    ema_rate: float
    beta: float
    beta_warmup_steps: int
    grad_clip: float
    grad_skip: float
    accu_steps: int
    checkpoint_smoke_test: bool
    checkpoint_smoke_steps: int
    benchmark_steps: int
    benchmark_warmup_steps: int
    execution_mode: str
    drop_remainder: bool
    input_channels: int = 1
    deterministic: bool = False
    concat_pa: bool = True
    context_norm: str = "[-1,1]"
    dataset_id: str = "morphomnist"
    save_dir: str = ""
    checkpoint_dir: str = ""
    remote_save_dir: str = ""

    def update_from_checkpoint(self, values: dict[str, Any], *, exclude: set[str] = frozenset()) -> None:
        """Apply only known model settings from a resumed artifact's metadata."""
        allowed = {field.name for field in fields(self)} - set(exclude)
        for key, value in values.items():
            if key in allowed:
                setattr(self, key, value)


@dataclass
class CounterfactualSettings(ImageModelSettings):
    gpu_id: str = "0"
    load_path: str = ""
    testing: bool = False
    pgm_path: str = ""
    predictor_path: str = ""
    vae_path: str = ""
    alpha: float = 0.1
    lmbda_init: float = 0.0
    lr_lagrange: float = 1e-2
    damping: float = 100.0
    do_pa: str | None = None
    plot_freq: int = 500
    cf_particles: int = 1
    elbo_constraint: float = 1.841216802597046
    model_validation_batches: int = 1
    trust_incomplete_checkpoint: bool = False


def image_model_settings(config: ExperimentConfig) -> ImageModelSettings:
    workflow = config.workflow
    assert isinstance(workflow, ImageModelTrainingConfig)
    return ImageModelSettings(
        accelerator=config.runtime.accelerator, precision=config.runtime.precision, dataset=config.dataset.name,
        data_dir=config.dataset.root, ckpt_dir=config.artifacts.root, remote_ckpt_dir=config.artifacts.remote_root,
        exp_name=config.artifacts.run_name, seed=config.seed, epochs=workflow.epochs, bs=config.optimizer.batch_size,
        lr=config.optimizer.lr, wd=config.optimizer.weight_decay, lr_warmup_steps=config.optimizer.lr_warmup_steps,
        betas=list(config.optimizer.betas), input_res=config.dataset.input_res, pad=config.dataset.pad,
        hflip=config.dataset.hflip, parents_x=list(MORPHOMNIST_SCHEMA.variable_names),
        context_dim=MORPHOMNIST_SCHEMA.encoded_dim, cond_prior=config.model.cond_prior,
        enc_arch=config.model.enc_arch, dec_arch=config.model.dec_arch, widths=list(config.model.widths),
        bottleneck=config.model.bottleneck, z_dim=config.model.z_dim, z_max_res=config.model.z_max_res,
        bias_max_res=config.model.bias_max_res, x_like=config.model.x_like, std_init=config.model.std_init,
        q_correction=config.model.q_correction, kl_free_bits=config.model.kl_free_bits,
        vae="hierarchical" if config.model.name == "hierarchical_vae" else "simple",
        speed_log_freq=workflow.speed_log_freq, viz_batch_size=workflow.viz_batch_size,
        eval_freq=workflow.eval_freq, checkpoint_freq=workflow.checkpoint_freq, resume=workflow.resume,
        ema_rate=workflow.ema_rate, beta=workflow.beta, beta_warmup_steps=workflow.beta_warmup_steps,
        grad_clip=workflow.grad_clip, grad_skip=workflow.grad_skip, accu_steps=workflow.accu_steps,
        checkpoint_smoke_test=workflow.checkpoint_smoke_test, checkpoint_smoke_steps=workflow.checkpoint_smoke_steps,
        benchmark_steps=workflow.benchmark_steps, benchmark_warmup_steps=workflow.benchmark_warmup_steps,
        execution_mode=workflow.execution_mode, drop_remainder=workflow.drop_remainder,
        context_norm=config.dataset.context_norm, dataset_id=config.dataset.name,
    )


def counterfactual_settings(config: ExperimentConfig) -> CounterfactualSettings:
    workflow = config.workflow
    assert isinstance(workflow, CounterfactualTrainingConfig)
    base = image_model_settings_for_counterfactual(config)
    return CounterfactualSettings(**base, gpu_id=config.runtime.gpu_id or "0", load_path=workflow.resume_checkpoint,
        testing=workflow.testing, pgm_path=workflow.scm_checkpoint, predictor_path=workflow.predictor_checkpoint,
        vae_path=workflow.image_model_checkpoint, alpha=workflow.alpha, lmbda_init=workflow.lmbda_init,
        lr_lagrange=workflow.lr_lagrange, damping=workflow.damping, do_pa=workflow.do_pa,
        cf_particles=workflow.cf_particles, elbo_constraint=workflow.elbo_constraint,
        model_validation_batches=workflow.model_validation_batches,
        trust_incomplete_checkpoint=workflow.trust_incomplete_checkpoint)


def image_model_settings_for_counterfactual(config: ExperimentConfig) -> dict[str, Any]:
    workflow = config.workflow
    assert isinstance(workflow, CounterfactualTrainingConfig)
    return dict(
        accelerator=config.runtime.accelerator, precision=config.runtime.precision, dataset=config.dataset.name,
        data_dir=config.dataset.root, ckpt_dir=config.artifacts.root, remote_ckpt_dir=config.artifacts.remote_root,
        exp_name=config.artifacts.run_name, seed=config.seed, epochs=workflow.epochs, bs=config.optimizer.batch_size,
        lr=config.optimizer.lr, wd=config.optimizer.weight_decay, lr_warmup_steps=config.optimizer.lr_warmup_steps,
        betas=list(config.optimizer.betas), input_res=config.dataset.input_res, pad=config.dataset.pad,
        hflip=config.dataset.hflip, parents_x=list(MORPHOMNIST_SCHEMA.variable_names), context_dim=MORPHOMNIST_SCHEMA.encoded_dim,
        cond_prior=config.model.cond_prior, enc_arch=config.model.enc_arch, dec_arch=config.model.dec_arch,
        widths=list(config.model.widths), bottleneck=config.model.bottleneck, z_dim=config.model.z_dim,
        z_max_res=config.model.z_max_res, bias_max_res=config.model.bias_max_res, x_like=config.model.x_like,
        std_init=config.model.std_init, q_correction=config.model.q_correction, kl_free_bits=config.model.kl_free_bits,
        vae="hierarchical" if config.model.name == "hierarchical_vae" else "simple", speed_log_freq=workflow.speed_log_freq,
        viz_batch_size=config.optimizer.batch_size, eval_freq=workflow.checkpoint_freq, checkpoint_freq=workflow.checkpoint_freq, resume="",
        ema_rate=workflow.ema_rate, beta=1.0, beta_warmup_steps=0, grad_clip=350.0, grad_skip=500.0,
        accu_steps=1, checkpoint_smoke_test=False, checkpoint_smoke_steps=1, benchmark_steps=workflow.benchmark_steps,
        benchmark_warmup_steps=20, execution_mode="single_device", drop_remainder=False,
        context_norm=config.dataset.context_norm, dataset_id=config.dataset.name,
    )
