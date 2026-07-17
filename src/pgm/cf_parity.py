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


def set_module_training_mode(module, training: bool) -> None:
    if training:
        module.train()
    else:
        module.eval()


def batch_progress_kwargs(split: str) -> Dict[str, Any]:
    return {"desc": split, "leave": False, "mininterval": 0.1}


def epoch_progress_kwargs() -> Dict[str, Any]:
    return {"desc": "epochs", "leave": True, "mininterval": 0.5}


def intervention_progress_kwargs() -> Dict[str, Any]:
    return {"desc": "train interventions", "leave": False, "mininterval": 0.1}


def format_run_summary(args, keys) -> str:
    parts = [f"{key}={getattr(args, key)}" for key in keys if hasattr(args, key)]
    return "run | " + ", ".join(parts)


def format_checkpoint_summary(args) -> str:
    names = ["vae", "pgm", "predictor"]
    if getattr(args, "resolved_resume_path", ""):
        names.append("resume")
    parts = []
    for name in names:
        path = getattr(args, f"resolved_{name}_path", "")
        trusted = getattr(args, f"resolved_{name}_trusted_incomplete", False)
        suffix = " (trusted incomplete)" if trusted else ""
        parts.append(f"{name}={path}{suffix}")
    return "checkpoint | " + ", ".join(parts)


def format_checkpoint_validation_summary(
    stats: Dict[str, float],
    *,
    loss_key: str = "loss",
    extra_keys: tuple[str, ...] = (),
) -> str:
    parts = []
    if loss_key not in stats:
        raise KeyError(f"Missing {loss_key!r} in checkpoint validation stats")
    parts.append(f"loss: {stats[loss_key]:.4f}")
    for key in extra_keys:
        if key in stats:
            parts.append(f"{key}: {stats[key]:.4f}")
    return "=> eval | " + ", ".join(parts)


def format_torch_style_eval_progress(stats: Dict[str, float], metric_keys: tuple[str, ...]) -> str:
    parts = [f"loss: {stats['loss']:.4f}"]
    for key in metric_keys:
        parts.append(f"{key}: {stats[key]:.4f}")
    return "=> eval | " + ", ".join(parts)


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
