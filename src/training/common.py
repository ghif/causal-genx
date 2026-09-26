"""Shared process, path, and artifact helpers for named training stages."""

from __future__ import annotations

import json
import os
import posixpath
from pathlib import Path
from typing import Any, Dict, Iterator, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from config import ExperimentConfig
from utils import (
    checkpoint_is_complete,
    is_remote_path,
    open_file,
    path_exists,
    resolve_checkpoint_path,
)


SOURCE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SOURCE_ROOT.parent


def stage_run_dir(config: ExperimentConfig) -> Path:
    """Return the shared local run root; stages add their own subdirectory if needed."""
    return REPOSITORY_ROOT / config.artifacts.root / config.dataset.name / config.artifacts.run_name


def to_jax_batch(batch: Dict[str, np.ndarray]) -> Dict[str, jax.Array]:
    """Convert a provider batch from any dataset to generic JAX batch format."""
    out: Dict[str, jax.Array] = {}
    for key, value in batch.items():
        if key == "x":
            image = np.asarray(value, dtype=np.float32)
            if image.max(initial=0.0) > 1.5:
                image = (image - 127.5) / 127.5
            out["x"] = jnp.asarray(image)
        else:
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape((-1, 1))
            out[key] = jnp.asarray(arr)
    return out


# Alias for backward compatibility
morphomnist_batch = to_jax_batch


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
        yield to_jax_batch(batch)


def _remote_candidate(checkpoint: str, remote_root: str) -> str:
    relative = checkpoint.replace("\\", "/").lstrip("./")
    root = remote_root.rstrip("/")
    root_name = root.rsplit("/", 1)[-1]
    if relative == root_name:
        relative = ""
    elif relative.startswith(root_name + "/"):
        relative = relative[len(root_name) + 1 :]
    return posixpath.join(root, relative)


def resolve_checkpoint_reference(
    checkpoint: str,
    remote_root: str = "",
    *,
    prefer_remote: bool = False,
    require_remote: bool = False,
) -> str:
    """Resolve an artifact reference locally or below the configured GCS root.

    Explicit ``gs://`` paths are authoritative. A relative legacy reference such
    as ``checkpoints/morphomnist/<run>/checkpoints`` uses its local copy when
    present, then falls back to ``remote_root`` with the leading
    ``checkpoints/`` component removed.

    ``prefer_remote``/``require_remote`` are used by production PadChest Stage 4
    runs so a stale local checkpoint tree cannot silently shadow the published
    GCS artifact selected by the config.
    """
    if is_remote_path(checkpoint) or not remote_root:
        if require_remote and not is_remote_path(checkpoint):
            raise ValueError(f"A remote checkpoint artifact is required, got local path {checkpoint!r}")
        return checkpoint
    if os.path.isabs(checkpoint):
        if require_remote:
            raise ValueError(f"A remote checkpoint artifact is required, got absolute local path {checkpoint!r}")
        return checkpoint

    candidate = _remote_candidate(checkpoint, remote_root)
    if prefer_remote or require_remote or not path_exists(checkpoint):
        if path_exists(candidate):
            return candidate
        if require_remote:
            raise ValueError(
                "Required remote checkpoint artifact was not found; refusing local fallback: "
                f"configured={checkpoint!r} remote_candidate={candidate!r}"
            )
    return candidate if path_exists(candidate) else checkpoint


def _checkpoint_metadata_root(checkpoint: str) -> str:
    root = checkpoint.rstrip("/")
    if root.rsplit("/", 1)[-1].isdigit():
        root = root.rsplit("/", 1)[0]
    return root


def _read_artifact_hparams(checkpoint: str) -> dict[str, Any]:
    hparams_path = posixpath.join(_checkpoint_metadata_root(checkpoint), "hparams.json")
    if not path_exists(hparams_path):
        raise ValueError(f"Missing stage metadata: {hparams_path}")
    with open_file(hparams_path, "r") as handle:
        return json.load(handle)


def _resolve_complete_step(
    name: str,
    checkpoint: str,
    *,
    allow_incomplete: bool,
    require_complete: bool,
) -> str:
    resolved = resolve_checkpoint_path(
        checkpoint,
        allow_incomplete=allow_incomplete and not require_complete,
    )
    if require_complete and not checkpoint_is_complete(resolved):
        raise ValueError(f"{name} checkpoint is not a complete Orbax step: {resolved}")
    return resolved


def _as_sequence(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return [str(value)]
    try:
        return [str(item) for item in value]
    except TypeError:
        return None


def _check_artifact_schema(
    name: str,
    hparams: dict[str, Any],
    *,
    expected_dataset: str | None,
    expected_variables: Sequence[str] | None,
    expected_context_dim: int | None,
    expected_input_res: int | None,
    strict: bool,
) -> None:
    dataset = hparams.get("dataset", hparams.get("dataset_id"))
    if expected_dataset and dataset is not None and dataset != expected_dataset:
        raise ValueError(
            f"{name} checkpoint dataset mismatch: expected {expected_dataset!r}, found {dataset!r}"
        )
    if strict and expected_dataset and dataset is None:
        raise ValueError(f"{name} checkpoint metadata is missing dataset/dataset_id")

    parents = _as_sequence(hparams.get("parents_x"))
    if expected_variables is not None and parents is not None and list(parents) != list(expected_variables):
        raise ValueError(
            f"{name} checkpoint schema variables mismatch: expected {list(expected_variables)!r}, found {parents!r}"
        )
    if strict and expected_variables is not None and parents is None:
        raise ValueError(f"{name} checkpoint metadata is missing parents_x")

    if expected_context_dim is not None and hparams.get("context_dim") is not None:
        found_context_dim = int(hparams["context_dim"])
        if found_context_dim != int(expected_context_dim):
            raise ValueError(
                f"{name} checkpoint context_dim mismatch: expected {expected_context_dim}, found {found_context_dim}"
            )
    if strict and expected_context_dim is not None and hparams.get("context_dim") is None:
        raise ValueError(f"{name} checkpoint metadata is missing context_dim")

    if expected_input_res is not None and hparams.get("input_res") is not None:
        found_input_res = int(hparams["input_res"])
        if found_input_res != int(expected_input_res):
            raise ValueError(
                f"{name} checkpoint input_res mismatch: expected {expected_input_res}, found {found_input_res}"
            )
    if strict and expected_input_res is not None and hparams.get("input_res") is None:
        raise ValueError(f"{name} checkpoint metadata is missing input_res")


def validate_stage_artifacts(
    scm_checkpoint: str,
    predictor_checkpoint: str,
    image_model_checkpoint: str,
    *,
    remote_root: str = "",
    dataset_name: str | None = None,
    expected_variables: Sequence[str] | None = None,
    expected_context_dim: int | None = None,
    expected_image_context_dim: int | None = None,
    expected_input_res: int | None = None,
    prefer_remote: bool = False,
    require_remote: bool = False,
    resolve_steps: bool = False,
    require_complete: bool = False,
    allow_incomplete: bool = False,
    strict_schema: bool = False,
) -> tuple[str, str, str]:
    """Resolve and validate the three frozen inputs required by Stage 4.

    Validation happens before loading any model arrays, preventing an SCM,
    predictor, or image-model artifact from being composed in the wrong role.
    Production PadChest runs additionally require published GCS artifacts and
    resolve checkpoint prefixes to exact complete Orbax step directories.
    """
    scm_checkpoint = resolve_checkpoint_reference(
        scm_checkpoint,
        remote_root,
        prefer_remote=prefer_remote,
        require_remote=require_remote,
    )
    predictor_checkpoint = resolve_checkpoint_reference(
        predictor_checkpoint,
        remote_root,
        prefer_remote=prefer_remote,
        require_remote=require_remote,
    )
    image_model_checkpoint = resolve_checkpoint_reference(
        image_model_checkpoint,
        remote_root,
        prefer_remote=prefer_remote,
        require_remote=require_remote,
    )

    if resolve_steps or require_complete:
        scm_checkpoint = _resolve_complete_step(
            "SCM",
            scm_checkpoint,
            allow_incomplete=allow_incomplete,
            require_complete=require_complete,
        )
        predictor_checkpoint = _resolve_complete_step(
            "predictor",
            predictor_checkpoint,
            allow_incomplete=allow_incomplete,
            require_complete=require_complete,
        )
        image_model_checkpoint = _resolve_complete_step(
            "image-model",
            image_model_checkpoint,
            allow_incomplete=allow_incomplete,
            require_complete=require_complete,
        )

    scm_hparams = _read_artifact_hparams(scm_checkpoint)
    predictor_hparams = _read_artifact_hparams(predictor_checkpoint)
    image_hparams = _read_artifact_hparams(image_model_checkpoint)

    expected = ((scm_checkpoint, scm_hparams, "sup_pgm"), (predictor_checkpoint, predictor_hparams, "sup_aux"))
    for checkpoint, hparams, setup in expected:
        if hparams.get("setup") != setup:
            raise ValueError(f"{checkpoint} is not a {setup} artifact")
    if "vae" not in image_hparams:
        raise ValueError(f"{image_model_checkpoint} is not an image-model artifact")

    for name, hparams, context_dim in (
        ("SCM", scm_hparams, expected_context_dim),
        ("predictor", predictor_hparams, expected_context_dim),
        ("image-model", image_hparams, expected_image_context_dim or expected_context_dim),
    ):
        _check_artifact_schema(
            name,
            hparams,
            expected_dataset=dataset_name,
            expected_variables=expected_variables,
            expected_context_dim=context_dim,
            expected_input_res=expected_input_res,
            strict=strict_schema,
        )
    return scm_checkpoint, predictor_checkpoint, image_model_checkpoint
