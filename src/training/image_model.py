"""Stage 3: train a conditional VAE/HVAE image mechanism.

Images are modelled conditional on the encoded causal parents. This stage does
not update the SCM or predictor; it creates the image-model artifact consumed
by counterfactual fine-tuning and standalone inference.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

import jax
from flax import nnx

from config import ExperimentConfig
from data.morphomnist import morphomnist
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

from .common import stage_run_dir
from .image_loop import init_state, preprocess_batch, trainer
from .settings import ImageModelSettings, image_model_settings


def output_dir(config: ExperimentConfig) -> Path:
    return stage_run_dir(config)


def _run_arguments(config: ExperimentConfig) -> ImageModelSettings:
    return image_model_settings(config)


def _setup_logging(args: ImageModelSettings) -> logging.Logger:
    """Keep terminal logging separate from eval-gated trainlog persistence."""
    ensure_dir(args.save_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger("causal-genx")


def _build_model(args: ImageModelSettings):
    """Construct the configured VAE variant before its graph is split for JIT."""
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
        dataset_id=args.dataset_id,
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


def _run(args: ImageModelSettings) -> None:
    """Execute image-model resume → dataset/model setup → generic training loop."""
    seed_all(args.seed, args.deterministic)
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
            args.update_from_checkpoint(saved, exclude={"resume", "accelerator", "data_dir"})
            if data_dir:
                args.data_dir = data_dir
            if requested_lr < args.lr:
                args.lr = requested_lr
            args.resume = resume_path
    # A run owns its logs, previews, and Orbax checkpoint tree in one location.
    args.save_dir = experiment_run_dir(args.ckpt_dir, args.dataset_id, args.exp_name, "run")
    args.checkpoint_dir = checkpoint_root_dir(args.save_dir)
    args.remote_save_dir = experiment_run_dir(args.remote_ckpt_dir, args.dataset_id, args.exp_name, "run")
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
    # Initialize optimizer state from one representative batch; training data is
    # still streamed by ``image_loop.trainer`` below.
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
    args.save_dir = experiment_run_dir(args.ckpt_dir, args.dataset_id, args.exp_name, "run")
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
