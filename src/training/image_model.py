"""Stage 3: native conditional VAE/HVAE image-model training."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

import jax
from flax import nnx

from config import ExperimentConfig, ImageModelTrainingConfig
from data.morphomnist import MORPHOMNIST_SCHEMA, morphomnist
from hps import HPARAMS_REGISTRY, Hparams, add_arguments
from models.image_vae import HVAE, SimpleVAE
from utils import (
    SummaryWriter,
    checkpoint_root_dir,
    ensure_dir,
    experiment_run_dir,
    load_checkpoint,
    open_file,
    path_exists,
    seed_all,
    write_images,
)

from .common import legacy_run_dir
from .image_loop import init_state, preprocess_batch, trainer


def output_dir(config: ExperimentConfig) -> Path:
    return legacy_run_dir(config)


def _run_arguments(config: ExperimentConfig) -> Hparams:
    """Build the legacy image-model hparams from one typed stage config."""
    workflow = config.workflow
    assert isinstance(workflow, ImageModelTrainingConfig)
    args = Hparams()
    # Legacy ``setup_hparams`` first applies the dataset preset and then every
    # argparse default. Mirror that merge exactly without parsing run.py's CLI.
    args.update(HPARAMS_REGISTRY["morphomnist"].__dict__)
    parser = add_arguments(argparse.ArgumentParser(add_help=False))
    args.update(vars(parser.parse_args([])))
    args.update({
        "dataset": config.dataset.name,
        "data_dir": config.dataset.root,
        "input_res": config.dataset.input_res,
        "pad": config.dataset.pad,
        "hflip": config.dataset.hflip,
        "parents_x": list(MORPHOMNIST_SCHEMA.variable_names),
        "context_dim": MORPHOMNIST_SCHEMA.encoded_dim,
        "cond_prior": config.model.cond_prior,
        "enc_arch": config.model.enc_arch,
        "dec_arch": config.model.dec_arch,
        "widths": config.model.widths,
        "bottleneck": config.model.bottleneck,
        "z_dim": config.model.z_dim,
        "z_max_res": config.model.z_max_res,
        "bias_max_res": config.model.bias_max_res,
        "x_like": config.model.x_like,
        "std_init": config.model.std_init,
        "q_correction": config.model.q_correction,
        "kl_free_bits": config.model.kl_free_bits,
        "accelerator": config.runtime.accelerator,
        "precision": config.runtime.precision,
        "ckpt_dir": config.artifacts.root,
        "remote_ckpt_dir": config.artifacts.remote_root,
        "exp_name": config.artifacts.run_name,
        "seed": config.seed,
        "epochs": workflow.epochs,
        "bs": config.optimizer.batch_size,
        "lr": config.optimizer.lr,
        "wd": config.optimizer.weight_decay,
        "lr_warmup_steps": config.optimizer.lr_warmup_steps,
        "betas": list(config.optimizer.betas),
        "vae": "hierarchical" if config.model.name == "hierarchical_vae" else "simple",
        "speed_log_freq": workflow.speed_log_freq,
        "viz_batch_size": workflow.viz_batch_size,
        "eval_freq": workflow.eval_freq,
        "checkpoint_freq": workflow.checkpoint_freq,
        "resume": workflow.resume,
        "ema_rate": workflow.ema_rate,
        "beta": workflow.beta,
        "beta_warmup_steps": workflow.beta_warmup_steps,
        "grad_clip": workflow.grad_clip,
        "grad_skip": workflow.grad_skip,
        "accu_steps": workflow.accu_steps,
        "checkpoint_smoke_test": workflow.checkpoint_smoke_test,
        "checkpoint_smoke_steps": workflow.checkpoint_smoke_steps,
        "benchmark_steps": workflow.benchmark_steps,
        "benchmark_warmup_steps": workflow.benchmark_warmup_steps,
        "execution_mode": workflow.execution_mode,
        "drop_remainder": workflow.drop_remainder,
        # Batch sizing is explicit in clean configs; retain auto-scaling only
        # for the deprecated shell launcher.
        "tpu_auto_scale": False,
    })
    return args


def _setup_logging(args: Hparams) -> logging.Logger:
    """Keep terminal logging separate from eval-gated trainlog persistence."""
    ensure_dir(args.save_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger("causal-genx")


def _build_model(args: Hparams):
    model_class = HVAE if args.vae == "hierarchical" else SimpleVAE
    return model_class(
        input_channels=args.input_channels,
        input_res=args.input_res,
        enc_arch=args.enc_arch,
        dec_arch=args.dec_arch,
        widths=args.widths,
        z_dim=args.z_dim,
        context_dim=args.context_dim,
        z_max_res=args.z_max_res,
        bottleneck=args.bottleneck,
        cond_prior=args.cond_prior,
        q_correction=args.q_correction,
        bias_max_res=args.bias_max_res,
        x_like=args.x_like,
        kl_free_bits=args.kl_free_bits,
        std_init=args.std_init,
        hps=args.hps,
        rngs=nnx.Rngs(args.seed),
    )


def _resume_hparams(path: str) -> dict[str, Any]:
    """Read checkpoint hparams without materializing its parameter trees."""
    root = path.rstrip("/")
    if os.path.basename(root).isdigit():
        root = os.path.dirname(root)
    hparams_path = f"{root}/hparams.json"
    if not path_exists(hparams_path):
        return {}
    with open_file(hparams_path, "r") as handle:
        return json.load(handle)


def _run(args: Hparams) -> None:
    """The former ``main.main`` body, kept deliberately outcome-compatible."""
    seed_all(args.seed, args.deterministic)
    if getattr(args, "tpu_auto_scale", False) and args.accelerator == "tpu" and jax.local_device_count() > 1:
        if args.bs == 128:
            args.bs = 512
        args.drop_remainder = True
    has_resume_checkpoint = False
    if args.resume:
        if not path_exists(args.resume):
            raise FileNotFoundError(f"Checkpoint not found at: {args.resume}")
        has_resume_checkpoint = True
        saved = _resume_hparams(args.resume)
        if saved:
            # Preserve the Torch resume contract: checkpoint hparams recreate
            # the trained model, while the active runtime stays explicit.
            saved = {
                key: value
                for key, value in saved.items()
                if key not in {"resume", "accelerator", "device"}
            }
            data_dir, requested_lr, resume_path = args.data_dir, args.lr, args.resume
            args.update(saved)
            if data_dir:
                args.data_dir = data_dir
            if requested_lr < args.lr:
                args.lr = requested_lr
            args.resume = resume_path
    args.save_dir = experiment_run_dir(args.ckpt_dir, args.hps, args.exp_name, "run")
    args.checkpoint_dir = checkpoint_root_dir(args.save_dir)
    args.remote_save_dir = experiment_run_dir(args.remote_ckpt_dir, args.hps, args.exp_name, "run")
    ensure_dir(args.save_dir)
    ensure_dir(args.checkpoint_dir)
    logger = _setup_logging(args)
    logger.info(
        "runtime accelerator=%s backend=%s local_device_count=%d global_device_count=%d process_count=%d process_index=%d jax=%s global_batch_size=%d lr=%g",
        args.accelerator, jax.default_backend(), jax.local_device_count(), jax.device_count(),
        jax.process_count(), jax.process_index(), jax.__version__, args.bs, args.lr,
    )
    logger.info(
        "jax_compilation_cache=%s",
        os.environ.get("JAX_COMPILATION_CACHE_DIR", "disabled"),
    )
    logger.info("loading datasets")
    writer = SummaryWriter(args.save_dir)
    datasets = morphomnist(args)
    logger.info("datasets loaded")
    logger.info("building model")
    model = _build_model(args)
    logger.info("model built")
    graphdef, _ = nnx.split(model, nnx.Param)
    logger.info("initialized model graph")
    sample = datasets["train"][0]
    sample = preprocess_batch(args, {key: value[None] for key, value in sample.items()}, expand_pa=True)
    rng = jax.random.PRNGKey(args.seed)
    logger.info("initializing optimizer state")
    state, tx = init_state(model, args, sample, rng)
    logger.info("optimizer state initialized")
    if has_resume_checkpoint:
        logger.info("restoring checkpoint from %s", args.resume)
        template = {
            "epoch": state.epoch,
            "step": state.step,
            "best_loss": state.best_loss,
            "params": state.params,
            "ema_params": state.ema.params,
            "opt_state": state.opt_state,
        }
        # The metadata restore above occurs before model construction. This
        # templated restore assigns arrays to the live model's sharding.
        checkpoint = load_checkpoint(args.resume, template=template)
        state.params = checkpoint["params"]
        state.ema.params = checkpoint["ema_params"]
        state.opt_state = checkpoint["opt_state"]
        state.step = checkpoint["step"]
        state.epoch = checkpoint["epoch"]
        state.best_loss = checkpoint["best_loss"]
        logger.info("checkpoint restored")
    logger.info("starting training loop")
    try:
        trainer(args, graphdef, state, tx, datasets, writer, logger)
    finally:
        writer.close()


def run(config: ExperimentConfig) -> str:
    """Run image-model training directly from a typed experiment config."""
    _run(_run_arguments(config))
    return str(output_dir(config))


def dry_run_image(config: ExperimentConfig) -> str:
    """Build the configured model and exercise visualization without training."""
    args = _run_arguments(config)
    args.save_dir = experiment_run_dir(args.ckpt_dir, args.hps, args.exp_name, "run")
    args.remote_save_dir = ""
    ensure_dir(args.save_dir)
    model = _build_model(args)
    graphdef, params_state = nnx.split(model, nnx.Param)
    params = params_state.to_pure_dict()
    # Keep this smoke path independent of dataset/pandas availability while
    # preserving the real image and parent tensor shapes.
    batch = {
        "x": np.zeros((args.viz_batch_size, args.input_channels, args.input_res, args.input_res), dtype=np.uint8),
        "pa": np.zeros((args.viz_batch_size, args.context_dim), dtype=np.float32),
    }
    batch = preprocess_batch(args, batch, compact_pa=True)
    path = write_images(
        args,
        graphdef,
        params,
        batch,
        jax.random.PRNGKey(args.seed),
        step=0,
    )
    return f"dry-run-image wrote {path} ({args.viz_batch_size} requested samples)"


def run_legacy_args(args: Any) -> None:
    """Compatibility hook for the deprecated ``src/main.py`` entrypoint."""
    _run(args)
