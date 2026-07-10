from __future__ import annotations

import argparse
import os
import logging

from runtime import configure_backend_from_argv

configure_backend_from_argv()

import jax
import jax.numpy as jnp
from flax import nnx

from datasets import morphomnist
from hps import add_arguments, setup_hparams
from pgm.dscm import DSCM
from pgm.flow_pgm import MorphoMNISTPGM
from models import HVAE
from trainer import preprocess_batch
from utils import SummaryWriter, ensure_dir, load_checkpoint, seed_all


def setup_logging(args):
    ensure_dir(args.save_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(args.save_dir, "trainlog.txt"))],
        force=True,
    )
    return logging.getLogger("causal-genx-cf")


class Bundle:
    def __init__(self, graphdef, params):
        self.graphdef = graphdef
        self.params = params


def main(args):
    seed_all(args.seed, args.deterministic)
    args.save_dir = os.path.join(args.ckpt_dir, args.hps, args.exp_name or "cf")
    ensure_dir(args.save_dir)
    logger = setup_logging(args)
    writer = SummaryWriter(args.save_dir)
    datasets = morphomnist(args)

    rngs = nnx.Rngs(args.seed)
    vae = HVAE(
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
    pgm = MorphoMNISTPGM(context_dim=args.context_dim, rngs=rngs)
    vae_graphdef, _ = nnx.split(vae, nnx.Param)
    pgm_graphdef, _ = nnx.split(pgm, nnx.Param)

    batch = preprocess_batch(args, {k: datasets["valid"][0][k][None] for k in datasets["valid"][0]}, expand_pa=True)
    vae_ckpt = load_checkpoint(args.vae_path)
    pgm_ckpt = load_checkpoint(args.pgm_path)

    dscm = DSCM(Bundle(vae_graphdef, vae_ckpt["params"]), Bundle(pgm_graphdef, pgm_ckpt["params"]))

    obs = {"x": batch["x"], "pa": batch["pa"]}
    intervention = {"digit": jax.nn.one_hot(jnp.array([1]), 10)}
    cf = dscm.counterfactual(obs, intervention, rng=jax.random.PRNGKey(args.seed))
    logger.info(f"counterfactual_x_shape={cf['x'].shape}")
    writer.add_scalar("cf/x_mean", float(jnp.mean(cf["x"])), 1)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    parser.add_argument("--pgm_path", type=str, default="checkpoints/morphomnist/pgm/checkpoints")
    parser.add_argument("--vae_path", type=str, default="checkpoints/morphomnist/run/checkpoints")
    args = setup_hparams(parser)
    main(args)
