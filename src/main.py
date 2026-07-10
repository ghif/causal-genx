from __future__ import annotations

import argparse
import os
import logging

from runtime import configure_backend_from_argv

configure_backend_from_argv()

from flax import nnx

from datasets import morphomnist
from hps import add_arguments, setup_hparams
from models import HVAE, SimpleVAE
from trainer import init_state, trainer
from utils import SummaryWriter, checkpoint_root_dir, ensure_dir, load_checkpoint, seed_all


def setup_logging(args):
    ensure_dir(args.save_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(args.save_dir, "trainlog.txt"))],
        force=True,
    )
    return logging.getLogger("causal-genx")


def setup_tensorboard(args):
    return SummaryWriter(args.save_dir)


def main(args):
    seed_all(args.seed, args.deterministic)
    args.save_dir = os.path.join(args.ckpt_dir, args.hps, args.exp_name or "run")
    args.checkpoint_dir = checkpoint_root_dir(args.save_dir)
    args.remote_save_dir = os.path.join(args.remote_ckpt_dir, args.hps, args.exp_name or "run") if getattr(args, "remote_ckpt_dir", "") else ""
    ensure_dir(args.save_dir)
    ensure_dir(args.checkpoint_dir)
    logger = setup_logging(args)
    logger.info("loading datasets")
    writer = setup_tensorboard(args)
    datasets = morphomnist(args)
    logger.info("datasets loaded")
    rngs = nnx.Rngs(args.seed)
    logger.info("building model")
    model = HVAE(
        input_channels=args.input_channels,
        input_res=args.input_res,
        enc_arch=args.enc_arch,
        dec_arch=args.dec_arch,
        widths=args.widths,
        z_dim=args.z_dim,
        context_dim=args.context_dim,
        bottleneck=args.bottleneck,
        cond_prior=args.cond_prior,
        x_like=args.x_like,
        kl_free_bits=args.kl_free_bits,
        std_init=args.std_init,
        hps=args.hps,
        rngs=rngs,
    ) if args.vae == "hierarchical" else SimpleVAE(
        input_channels=args.input_channels,
        input_res=args.input_res,
        enc_arch=args.enc_arch,
        dec_arch=args.dec_arch,
        widths=args.widths,
        z_dim=args.z_dim,
        context_dim=args.context_dim,
        bottleneck=args.bottleneck,
        cond_prior=args.cond_prior,
        x_like=args.x_like,
        kl_free_bits=args.kl_free_bits,
        std_init=args.std_init,
        hps=args.hps,
        rngs=rngs,
    )
    logger.info("model built")
    graphdef, _ = nnx.split(model, nnx.Param)
    logger.info("initialized model graph")
    sample = datasets["train"][0]
    from trainer import preprocess_batch

    sample = preprocess_batch(args, {k: sample[k][None] for k in sample}, expand_pa=True)
    rng = __import__("jax").random.PRNGKey(args.seed)
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
        ckpt = load_checkpoint(args.resume, template=template)
        state.params = ckpt["params"]
        state.ema.params = ckpt["ema_params"]
        state.opt_state = ckpt["opt_state"]
        state.step = ckpt["step"]
        state.epoch = ckpt["epoch"]
        state.best_loss = ckpt["best_loss"]
        logger.info("checkpoint restored")
    logger.info("starting training loop")
    trainer(args, graphdef, state, tx, datasets, writer, logger)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = setup_hparams(parser)
    main(args)
