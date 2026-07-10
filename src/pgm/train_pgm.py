from __future__ import annotations

import argparse
import os
import logging

from runtime import configure_backend_from_argv

configure_backend_from_argv()

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from datasets import morphomnist
from hps import add_arguments, setup_hparams
from pgm.flow_pgm import MorphoMNISTPGM
from trainer import preprocess_batch
from utils import SummaryWriter, checkpoint_root_dir, ensure_dir, materialize_nnx, save_checkpoint, seed_all, sync_tree


def setup_logging(args):
    ensure_dir(args.save_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(args.save_dir, "trainlog.txt"))],
        force=True,
    )
    return logging.getLogger("causal-genx-pgm")


def unpack_parents(pa):
    thickness = pa[:, 0]
    intensity = pa[:, 1]
    digit = pa[:, 2:]
    digit_idx = jnp.argmax(digit, axis=-1)
    return thickness, intensity, digit, digit_idx


def main(args):
    seed_all(args.seed, args.deterministic)
    args.save_dir = os.path.join(args.ckpt_dir, args.hps, args.exp_name or "pgm")
    args.checkpoint_dir = checkpoint_root_dir(args.save_dir)
    args.remote_save_dir = os.path.join(args.remote_ckpt_dir, args.hps, args.exp_name or "pgm") if getattr(args, "remote_ckpt_dir", "") else ""
    ensure_dir(args.save_dir)
    ensure_dir(args.checkpoint_dir)
    logger = setup_logging(args)
    logger.info("building PGM training writer")
    writer = SummaryWriter(args.save_dir)
    logger.info("writer ready")
    datasets = morphomnist(args)
    rngs = nnx.Rngs(args.seed)
    model = MorphoMNISTPGM(context_dim=args.context_dim, rngs=rngs)
    graphdef, _ = nnx.split(model, nnx.Param)
    sample = preprocess_batch(args, {k: datasets["train"][0][k][None] for k in datasets["train"][0]}, expand_pa=True)
    params = nnx.state(model, nnx.Param).to_pure_dict()
    tx = optax.adamw(args.lr, weight_decay=args.wd)
    opt_state = tx.init(params)

    def loss_fn(p, batch):
        model_ = materialize_nnx(graphdef, p)
        preds = model_(batch["x"])
        pa = batch["pa"][:, 0, 0, :]
        thickness, intensity, digit, digit_idx = unpack_parents(pa)
        digit_loss = optax.softmax_cross_entropy(preds["digit"], digit).mean()
        thickness_loss = jnp.mean((preds["thickness"] - thickness) ** 2)
        intensity_loss = jnp.mean((preds["intensity"] - intensity) ** 2)
        t_mu = p["thickness_mu"][digit_idx]
        t_sigma = jnp.exp(p["thickness_logsigma"][digit_idx])
        i_mu = p["intensity_mu"][digit_idx]
        i_sigma = jnp.exp(p["intensity_logsigma"][digit_idx])
        nll = 0.5 * jnp.mean(((thickness - t_mu) / t_sigma) ** 2 + 2.0 * jnp.log(t_sigma + 1e-8))
        nll = nll + 0.5 * jnp.mean(((intensity - i_mu) / i_sigma) ** 2 + 2.0 * jnp.log(i_sigma + 1e-8))
        return digit_loss + thickness_loss + intensity_loss + nll, {
            "loss": digit_loss + thickness_loss + intensity_loss + nll,
            "digit": digit_loss,
            "thickness": thickness_loss,
            "intensity": intensity_loss,
        }

    train_iter = iter(lambda: None, None)
    def batch_iter():
        while True:
            for idx in range(0, len(datasets["train"]), args.bs):
                batch = [datasets["train"][i] for i in range(idx, min(idx + args.bs, len(datasets["train"])))]
                keys = batch[0].keys()
                out = {}
                for k in keys:
                    out[k] = np.stack([np.asarray(b[k]) for b in batch], axis=0)
                yield preprocess_batch(args, out, expand_pa=True)
    bi = batch_iter()
    for epoch in range(args.epochs):
        losses = []
        for _ in range(max(1, len(datasets["train"]) // args.bs)):
            batch = next(bi)
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            losses.append(metrics)
        mean_loss = float(jnp.mean(jnp.array([m["loss"] for m in losses])))
        writer.add_scalar("train/loss", mean_loss, epoch + 1)
        logger.info(f"epoch={epoch+1} loss={mean_loss:.4f}")
        save_checkpoint(
            {"params": params, "opt_state": opt_state, "hparams": vars(args), "epoch": epoch + 1},
            args.checkpoint_dir,
            step=epoch + 1,
            custom_metadata={"epoch": epoch + 1, "loss": mean_loss},
        )
        if args.remote_save_dir:
            sync_tree(args.save_dir, args.remote_save_dir)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = setup_hparams(parser)
    main(args)
