"""Small, pure helpers shared by the counterfactual train and evaluation paths.

Keeping the constrained loss and global gradient clipping here makes the JIT
step readable and ensures the same numerical rule is tested independently.
"""

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp
import optax


def inherit_image_training_config(settings, metadata: Dict[str, Any]) -> None:
    for key in ("beta", "parents_x", "input_res", "grad_clip", "grad_skip", "wd", "betas"):
        if key in metadata:
            setattr(settings, key, metadata[key])


def set_module_training_mode(module, training: bool) -> None:
    (module.train if training else module.eval)()


def batch_progress_kwargs(split: str) -> Dict[str, Any]: return {"desc": split, "leave": False, "mininterval": 0.1}
def epoch_progress_kwargs() -> Dict[str, Any]: return {"desc": "epochs", "leave": True, "mininterval": 0.5}
def intervention_progress_kwargs() -> Dict[str, Any]: return {"desc": "train interventions", "leave": False, "mininterval": 0.1}
def format_run_summary(settings, keys) -> str: return "run | " + ", ".join(f"{key}={getattr(settings, key)}" for key in keys if hasattr(settings, key))


def format_checkpoint_summary(settings) -> str:
    names = ["vae", "pgm", "predictor"] + (["resume"] if getattr(settings, "resolved_resume_path", "") else [])
    return "checkpoint | " + ", ".join(
        f"{name}={getattr(settings, f'resolved_{name}_path', '')}{' (trusted incomplete)' if getattr(settings, f'resolved_{name}_trusted_incomplete', False) else ''}"
        for name in names
    )


def format_checkpoint_validation_summary(stats: Dict[str, float], *, loss_key: str = "loss", extra_keys: tuple[str, ...] = ()) -> str:
    if loss_key not in stats: raise KeyError(f"Missing {loss_key!r} in checkpoint validation stats")
    return "=> eval | " + ", ".join([f"loss: {stats[loss_key]:.4f}"] + [f"{key}: {stats[key]:.4f}" for key in extra_keys if key in stats])


def format_torch_style_eval_progress(stats: Dict[str, float], metric_keys: tuple[str, ...]) -> str:
    return "=> eval | " + ", ".join([f"loss: {stats['loss']:.4f}"] + [f"{key}: {stats[key]:.4f}" for key in metric_keys])


def damped_lagrangian_loss(aux_loss, lmbda, constraint, damping):
    """Apply the detached damping term used to enforce the ELBO constraint."""
    return aux_loss - (lmbda - damping * jax.lax.stop_gradient(constraint)) * constraint


def clip_counterfactual_grads(vae_grads, lmbda_grads, max_norm):
    """Scale both optimizers with one global norm so their updates stay coupled."""
    grad_norm = optax.global_norm((vae_grads, lmbda_grads)); scale = jnp.minimum(1.0, max_norm / (grad_norm + 1e-6))
    return jax.tree_util.tree_map(lambda x: x * scale, vae_grads), jax.tree_util.tree_map(lambda x: x * scale, lmbda_grads), grad_norm
