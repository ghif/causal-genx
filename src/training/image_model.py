"""Stage 3: native conditional VAE/HVAE image-model training."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

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
    seed_all,
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
        "parents_x": list(MORPHOMNIST_SCHEMA.variable_names),
        "context_dim": MORPHOMNIST_SCHEMA.encoded_dim,
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
        "vae": "hierarchical" if config.model.name == "hierarchical_vae" else "simple",
        "speed_log_freq": workflow.speed_log_freq,
        "eval_freq": workflow.eval_freq,
        "checkpoint_freq": workflow.checkpoint_freq,
        "benchmark_steps": workflow.benchmark_steps,
        "benchmark_warmup_steps": workflow.benchmark_warmup_steps,
    })
    return args


def _setup_logging(args: Hparams) -> logging.Logger:
    """Preserve the legacy terminal log format; trainer owns trainlog.txt."""
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


def _run(args: Hparams) -> None:
    """The former ``main.main`` body, kept deliberately outcome-compatible."""
    seed_all(args.seed, args.deterministic)
    if getattr(args, "tpu_auto_scale", False) and args.accelerator == "tpu" and jax.local_device_count() > 1:
        if args.bs == 128:
            args.bs = 512
        args.drop_remainder = True
    args.save_dir = experiment_run_dir(args.ckpt_dir, args.hps, args.exp_name, "run")
    args.checkpoint_dir = checkpoint_root_dir(args.save_dir)
    args.remote_save_dir = experiment_run_dir(args.remote_ckpt_dir, args.hps, args.exp_name, "run")
    ensure_dir(args.save_dir)
    ensure_dir(args.checkpoint_dir)
    logger = _setup_logging(args)
    logger.info(
        "runtime accelerator=%s backend=%s local_device_count=%d jax=%s global_batch_size=%d lr=%g",
        args.accelerator, jax.default_backend(), jax.local_device_count(), jax.__version__, args.bs, args.lr,
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
    if args.resume and os.path.exists(args.resume):
        logger.info("restoring checkpoint from %s", args.resume)
        template = {
            "epoch": state.epoch,
            "step": state.step,
            "best_loss": state.best_loss,
            "params": state.params,
            "ema_params": state.ema.params,
            "opt_state": state.opt_state,
        }
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


def run_legacy_args(args: Any) -> None:
    """Compatibility hook for the deprecated ``src/main.py`` entrypoint."""
    _run(args)
