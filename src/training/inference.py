"""Stage 5: native inference for saved image-model Orbax artifacts.

Inference is intentionally read-only with respect to the source checkpoint:
it restores the EMA image-model parameters, reconstructs one provided (or zero)
image under named parents, and writes a preview plus JSON summary to a new run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from PIL import Image

from config import ExperimentConfig, InferenceConfig
from data.morphomnist import MORPHOMNIST_SCHEMA
from models.image_vae import HVAE, SimpleVAE
from utils import (
    checkpoint_root_dir,
    ensure_dir,
    experiment_run_dir,
    load_checkpoint_with_path,
    open_file,
    path_exists,
    postprocess,
    seed_all,
)


def output_dir(config: ExperimentConfig) -> str:
    return os.path.join(experiment_run_dir(config.artifacts.root, config.dataset.name, config.artifacts.run_name, "inference"), "inference")


def _checkpoint_root(reference: str) -> str:
    """Accept an experiment root, a checkpoint root, or an explicit Orbax step."""
    reference = reference.rstrip("/")
    if reference.rsplit("/", 1)[-1].isdigit() or reference.endswith("/checkpoints"):
        return reference
    nested = f"{reference}/checkpoints"
    return nested if path_exists(f"{nested}/hparams.json") else reference


def _metadata(reference: str) -> tuple[dict[str, Any], str]:
    root = _checkpoint_root(reference)
    hparams_path = f"{root}/hparams.json"
    if not path_exists(hparams_path):
        raise FileNotFoundError(f"Missing image-model metadata: {hparams_path}")
    with open_file(hparams_path, "r") as handle:
        return json.load(handle), root


def _model_from_metadata(metadata: dict[str, Any], seed: int):
    cls = HVAE if metadata.get("vae", "hierarchical") == "hierarchical" else SimpleVAE
    required = ("input_channels", "input_res", "enc_arch", "dec_arch", "widths", "z_dim", "context_dim")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"Checkpoint metadata does not describe an image model: {missing}")
    return cls(
        input_channels=metadata["input_channels"], input_res=metadata["input_res"],
        enc_arch=metadata["enc_arch"], dec_arch=metadata["dec_arch"], widths=metadata["widths"],
        z_dim=metadata["z_dim"], context_dim=metadata["context_dim"],
        z_max_res=metadata.get("z_max_res", 192), bottleneck=metadata.get("bottleneck", 4),
        cond_prior=metadata.get("cond_prior", False), q_correction=metadata.get("q_correction", False),
        bias_max_res=metadata.get("bias_max_res", 64), x_like=metadata.get("x_like", "diag_dgauss"),
        kl_free_bits=metadata.get("kl_free_bits", 0.0), std_init=metadata.get("std_init", 0.0),
        dataset_id=metadata.get("dataset", metadata.get("hps", "morphomnist")), rngs=nnx.Rngs(seed),
    )


def _input_image(path: str, input_res: int, channels: int) -> jax.Array:
    if not path:
        return jnp.zeros((1, input_res, input_res, channels), dtype=jnp.float32)
    image = Image.open(path).convert("L").resize((input_res, input_res))
    values = np.asarray(image, dtype=np.float32)[None, ..., None]
    return jnp.asarray((values - 127.5) / 127.5)


def _parents(values: dict[str, Any], context_dim: int) -> jax.Array:
    """Encode named MorphoMNIST values in the schema's stable parent order."""
    encoded = []
    for variable in MORPHOMNIST_SCHEMA.variables:
        value = values.get(variable.name, 0)
        if variable.name == "digit":
            encoded.append(jax.nn.one_hot(jnp.asarray([int(value)]), variable.encoded_dim, dtype=jnp.float32))
        else:
            encoded.append(jnp.asarray([[float(value)]], dtype=jnp.float32))
    result = jnp.concatenate(encoded, axis=-1)
    if result.shape[-1] != context_dim:
        raise ValueError(f"Parent encoding has dimension {result.shape[-1]}, checkpoint expects {context_dim}")
    return result


def run(config: ExperimentConfig) -> str:
    """Restore EMA weights, run one forward/reconstruction pass, and save outputs."""
    workflow = config.workflow
    assert isinstance(workflow, InferenceConfig)
    seed_all(config.seed, deterministic=True)
    metadata, checkpoint_root = _metadata(workflow.checkpoint)
    model = _model_from_metadata(metadata, config.seed)
    graphdef, params_state = nnx.split(model, nnx.Param)
    template = {"ema_params": params_state.to_pure_dict()}
    # A narrow template restores only EMA weights and maps arrays to the active
    # runtime, allowing a CPU process to inspect GPU- or TPU-authored artifacts.
    checkpoint, resolved = load_checkpoint_with_path(
        checkpoint_root, template=template,
        fallback_sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
        allow_incomplete=workflow.trust_incomplete_checkpoint, partial_restore=True,
    )
    weights = checkpoint.get("ema_params")
    if weights is None:
        raise ValueError(f"Image-model checkpoint at {resolved} has no EMA parameters")
    model = nnx.merge(graphdef, nnx.State(weights)); model.eval()
    x = _input_image(workflow.image_path, metadata["input_res"], metadata["input_channels"])
    parents = _parents(workflow.parents, metadata["context_dim"])
    # ELBO diagnostics use the supplied image; the decoder mean is the preview.
    output = model(x, parents, beta=workflow.beta, rng=jax.random.PRNGKey(config.seed), training=False)
    reconstruction, _ = model.likelihood.sample(
        model.decoder(parents=parents, rng=jax.random.PRNGKey(config.seed), training=False)[0], return_loc=True
    )
    directory = output_dir(config); ensure_dir(directory)
    preview = postprocess(np.asarray(reconstruction[0]));
    if preview.ndim == 3 and preview.shape[-1] == 1: preview = preview[..., 0]
    preview_path = os.path.join(directory, f"preview-step-{resolved.rsplit('/', 1)[-1]}.png")
    imageio.imwrite(preview_path, preview)
    summary = {"checkpoint": checkpoint_root, "resolved_checkpoint": resolved, "preview": preview_path,
               "input_shape": list(x.shape), "parents": workflow.parents,
               "elbo": float(output["elbo"]), "nll": float(output["nll"]), "kl": float(output["kl"])}
    with open(os.path.join(directory, "inference.json"), "w", encoding="utf-8") as handle: json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, sort_keys=True))
    return directory
