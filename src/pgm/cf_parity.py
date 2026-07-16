from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp
import optax


def inherit_vae_training_config(args, vae_hparams: Dict[str, Any]) -> None:
    """Mirror the VAE-owned training configuration used by PyTorch train_cf."""
    keys = ("beta", "parents_x", "input_res", "grad_clip", "grad_skip", "wd", "betas")
    for key in keys:
        if key in vae_hparams:
            setattr(args, key, vae_hparams[key])


def damped_lagrangian_loss(aux_loss, lmbda, constraint, damping):
    damp = damping * jax.lax.stop_gradient(constraint)
    return aux_loss - (lmbda - damp) * constraint


def clip_counterfactual_grads(vae_grads, lmbda_grads, max_norm):
    """Apply PyTorch's one global clipping scale to both optimizers."""
    grad_norm = optax.global_norm((vae_grads, lmbda_grads))
    clip_scale = jnp.minimum(1.0, max_norm / (grad_norm + 1e-6))
    vae_grads = jax.tree_util.tree_map(lambda x: x * clip_scale, vae_grads)
    lmbda_grads = jax.tree_util.tree_map(lambda x: x * clip_scale, lmbda_grads)
    return vae_grads, lmbda_grads, grad_norm
