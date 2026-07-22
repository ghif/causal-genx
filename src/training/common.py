"""Shared process, path, and artifact helpers for named training stages."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterator

import jax
import jax.numpy as jnp
import numpy as np

from config import ExperimentConfig


SOURCE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SOURCE_ROOT.parent


def run_legacy_module(script: str, arguments: list[str], *, cwd: Path | None = None) -> None:
    cwd = cwd or SOURCE_ROOT
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(SOURCE_ROOT), environment.get("PYTHONPATH", ""))))
    if subprocess.run([sys.executable, "-u", script, *arguments], cwd=cwd, env=environment, check=False).returncode:
        raise RuntimeError(f"Stage implementation {script} failed.")


def legacy_run_dir(config: ExperimentConfig) -> Path:
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


def validate_stage_artifacts(scm_checkpoint: str, predictor_checkpoint: str, image_model_checkpoint: str) -> None:
    """Validate local stage identity before counterfactual fine-tuning."""
    expected = ((scm_checkpoint, "sup_pgm"), (predictor_checkpoint, "sup_aux"))
    for checkpoint, setup in expected:
        hparams = Path(checkpoint) / "hparams.json"
        if not hparams.is_file():
            raise ValueError(f"Missing stage metadata: {hparams}")
        with hparams.open(encoding="utf-8") as handle:
            if json.load(handle).get("setup") != setup:
                raise ValueError(f"{checkpoint} is not a {setup} artifact")
    image_hparams = Path(image_model_checkpoint) / "hparams.json"
    if not image_hparams.is_file():
        raise ValueError(f"Missing image-model metadata: {image_hparams}")
    with image_hparams.open(encoding="utf-8") as handle:
        if "vae" not in json.load(handle):
            raise ValueError(f"{image_model_checkpoint} is not an image-model artifact")
