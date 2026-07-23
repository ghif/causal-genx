"""Shared process, path, and artifact helpers for named training stages."""

from __future__ import annotations

import json
import os
import posixpath
from pathlib import Path
from typing import Any, Dict, Iterator

import jax
import jax.numpy as jnp
import numpy as np

from config import ExperimentConfig
from utils import is_remote_path, open_file, path_exists


SOURCE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SOURCE_ROOT.parent


def stage_run_dir(config: ExperimentConfig) -> Path:
    return REPOSITORY_ROOT / config.artifacts.root / config.dataset.name / config.artifacts.run_name


def morphomnist_batch(batch: Dict[str, np.ndarray]) -> Dict[str, jax.Array]:
    """Convert a MorphoMNIST provider batch to the shared JAX batch format."""
    image = np.asarray(batch["x"], dtype=np.float32)
    if image.max(initial=0.0) > 1.5:
        image = (image - 127.5) / 127.5
    return {
        "x": jnp.asarray(image),
        "thickness": jnp.asarray(batch["thickness"], dtype=jnp.float32).reshape((-1, 1)),
        "intensity": jnp.asarray(batch["intensity"], dtype=jnp.float32).reshape((-1, 1)),
        "digit": jnp.asarray(batch["digit"], dtype=jnp.float32),
    }


def epoch_batches(dataset: Any, batch_size: int, *, shuffle: bool, drop_last: bool, rng: np.random.Generator) -> Iterator[Dict[str, jax.Array]]:
    """Yield deterministic provider batches for SCM and predictor stages."""
    indices = np.arange(len(dataset), dtype=np.int64)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        if drop_last and batch_indices.size < batch_size:
            continue
        if hasattr(dataset, "make_batch"):
            batch = dataset.make_batch(batch_indices, rng=rng, shuffle=shuffle)
        else:
            batch = {
                key: np.stack([np.asarray(dataset[int(index)][key]) for index in batch_indices])
                for key in dataset[0]
            }
        yield morphomnist_batch(batch)


def resolve_checkpoint_reference(checkpoint: str, remote_root: str = "") -> str:
    """Resolve an artifact reference locally or below the configured GCS root.

    Explicit ``gs://`` paths are authoritative. A relative legacy reference such
    as ``checkpoints/morphomnist/<run>/checkpoints`` uses its local copy when
    present, then falls back to ``remote_root`` with the leading
    ``checkpoints/`` component removed.
    """
    if is_remote_path(checkpoint) or path_exists(checkpoint) or not remote_root:
        return checkpoint
    if os.path.isabs(checkpoint):
        return checkpoint

    relative = checkpoint.replace("\\", "/").lstrip("./")
    root = remote_root.rstrip("/")
    root_name = root.rsplit("/", 1)[-1]
    if relative == root_name:
        relative = ""
    elif relative.startswith(root_name + "/"):
        relative = relative[len(root_name) + 1 :]
    candidate = posixpath.join(root, relative)
    return candidate if path_exists(candidate) else checkpoint


def _read_artifact_hparams(checkpoint: str) -> dict[str, Any]:
    hparams_path = posixpath.join(checkpoint.rstrip("/"), "hparams.json")
    if not path_exists(hparams_path):
        raise ValueError(f"Missing stage metadata: {hparams_path}")
    with open_file(hparams_path, "r") as handle:
        return json.load(handle)


def validate_stage_artifacts(
    scm_checkpoint: str,
    predictor_checkpoint: str,
    image_model_checkpoint: str,
    *,
    remote_root: str = "",
) -> tuple[str, str, str]:
    """Resolve and validate the three counterfactual input artifacts."""
    scm_checkpoint = resolve_checkpoint_reference(scm_checkpoint, remote_root)
    predictor_checkpoint = resolve_checkpoint_reference(predictor_checkpoint, remote_root)
    image_model_checkpoint = resolve_checkpoint_reference(image_model_checkpoint, remote_root)
    expected = ((scm_checkpoint, "sup_pgm"), (predictor_checkpoint, "sup_aux"))
    for checkpoint, setup in expected:
        if _read_artifact_hparams(checkpoint).get("setup") != setup:
            raise ValueError(f"{checkpoint} is not a {setup} artifact")
    if "vae" not in _read_artifact_hparams(image_model_checkpoint):
        raise ValueError(f"{image_model_checkpoint} is not an image-model artifact")
    return scm_checkpoint, predictor_checkpoint, image_model_checkpoint
