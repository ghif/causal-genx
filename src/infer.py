from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from runtime import configure_backend_from_argv

configure_backend_from_argv()

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from hps import Hparams
from models import HVAE, SimpleVAE
from trainer import init_state
from utils import load_checkpoint, materialize_nnx, open_file, postprocess, seed_all


def _parse_parents(text: str | None, context_dim: int) -> jnp.ndarray:
    if not text:
        return jnp.zeros((1, context_dim), dtype=jnp.float32)
    values = np.asarray(json.loads(text), dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[-1] != context_dim:
        raise ValueError(f"Expected parents with final dimension {context_dim}, got {values.shape[-1]}")
    return jnp.asarray(values, dtype=jnp.float32)


def _load_image(path: str, input_res: int) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("L")
    if img.size != (input_res, input_res):
        img = img.resize((input_res, input_res))
    x = np.asarray(img, dtype=np.float32)[None, ..., None]
    if x.max() > 1.5:
        x = (x - 127.5) / 127.5
    return x


def _is_remote_path(path: str) -> bool:
    return path.startswith("gs://")


def _checkpoint_root(path: str) -> str:
    path = path.rstrip("/")
    if path.split("/")[-1].isdigit():
        return path.rsplit("/", 1)[0]
    return path


def _build_model(hparams: Hparams, rngs: nnx.Rngs):
    model_cls = HVAE if hparams.vae == "hierarchical" else SimpleVAE
    return model_cls(
        input_channels=hparams.input_channels,
        input_res=hparams.input_res,
        enc_arch=hparams.enc_arch,
        dec_arch=hparams.dec_arch,
        widths=hparams.widths,
        z_dim=hparams.z_dim,
        context_dim=hparams.context_dim,
        z_max_res=hparams.z_max_res,
        bottleneck=hparams.bottleneck,
        cond_prior=hparams.cond_prior,
        q_correction=hparams.q_correction,
        bias_max_res=hparams.bias_max_res,
        x_like=hparams.x_like,
        kl_free_bits=hparams.kl_free_bits,
        std_init=hparams.std_init,
        hps=hparams.hps,
        rngs=rngs,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Absolute path to the Orbax checkpoint root or a specific step directory.",
    )
    parser.add_argument("--data_dir", default="gs://medical-airnd/causal-gen/datasets/morphomnist")
    parser.add_argument("--image_path", default="", help="Optional grayscale image to run through the model.")
    parser.add_argument(
        "--parents",
        default="",
        help='Optional JSON list of parent values. Example: "[0.0, 0.0, 0.0, ...]"',
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--trust_incomplete_checkpoint",
        action="store_true",
        default=False,
        help="Restore the newest numeric step even if commit_success.txt is missing.",
    )
    args, unknown = parser.parse_known_args()

    seed_all(args.seed, deterministic=True)

    checkpoint_path = args.checkpoint.rstrip("/")
    checkpoint_path = os.path.abspath(checkpoint_path) if not _is_remote_path(checkpoint_path) else checkpoint_path
    checkpoint_root = _checkpoint_root(checkpoint_path)
    target_device = jax.devices()[0]
    fallback_sharding = jax.sharding.SingleDeviceSharding(target_device)

    with open_file(f"{checkpoint_root}/hparams.json", "r") as f:
        hparams_dict = json.load(f)
    hparams = Hparams()
    hparams.update(hparams_dict)

    rngs = nnx.Rngs(args.seed)
    model = _build_model(hparams, rngs)
    graphdef, _ = nnx.split(model, nnx.Param)
    params = nnx.state(model, nnx.Param).to_pure_dict()

    # Build a matching optimizer state so Orbax can restore the full tree.
    sample_args = Hparams()
    sample_args.update(hparams.__dict__)
    sample_args.data_dir = args.data_dir
    _, tx = init_state(model, sample_args, None, jax.random.PRNGKey(args.seed))
    template = {
        "epoch": 0,
        "step": 0,
        "best_loss": float("inf"),
        "params": params,
        "ema_params": params,
        "opt_state": tx.init(params),
    }
    restored = load_checkpoint(
        checkpoint_path,
        template=template,
        fallback_sharding=fallback_sharding,
        allow_incomplete=args.trust_incomplete_checkpoint,
    )

    weights = restored.get("ema_params", restored["params"])
    model = materialize_nnx(graphdef, weights)

    if args.image_path:
        x = _load_image(args.image_path, hparams.input_res)
    else:
        x = np.zeros((1, hparams.input_res, hparams.input_res, hparams.input_channels), dtype=np.float32)

    parents = _parse_parents(args.parents, hparams.context_dim)
    if parents.shape[0] != x.shape[0]:
        parents = jnp.repeat(parents[:1], x.shape[0], axis=0)

    batch = {
        "x": jnp.asarray(x, dtype=jnp.float32),
        "pa": parents,
    }
    out = model(batch["x"], batch["pa"], beta=args.beta, rng=jax.random.PRNGKey(args.seed))
    recon, _ = model.likelihood.sample(
        model.decoder(parents=batch["pa"], rng=jax.random.PRNGKey(args.seed), training=False)[0],
        return_loc=True,
    )

    print(f"restored_step={int(restored['step'])}")
    print(f"loss_elbo={float(out['elbo']):.6f} nll={float(out['nll']):.6f} kl={float(out['kl']):.6f}")
    print(f"input_shape={tuple(batch['x'].shape)} parents_shape={tuple(batch['pa'].shape)}")
    print(f"recon_shape={tuple(recon.shape)} recon_range=({float(recon.min()):.4f}, {float(recon.max()):.4f})")

    # Save a quick visual so you can inspect the forward pass.
    preview_path = Path.cwd() / f"infer-preview-step-{int(restored['step'])}.png"
    from imageio.v2 import imwrite

    preview = postprocess(np.asarray(recon[0]))
    if preview.ndim == 3 and preview.shape[-1] == 1:
        preview = preview[..., 0]
    imwrite(preview_path, preview)
    print(f"preview_path={preview_path}")


if __name__ == "__main__":
    main()
