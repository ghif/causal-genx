from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from config import load_experiment
from contracts import CausalGraphSpec, VariableKind, VariableSpec
from data.padchest import PAD_CHEST_SCHEMA_V2
from training.settings import image_model_settings
from utils import BackgroundArtifactWriter, load_checkpoint, save_checkpoint, to_json_serializable


def test_to_json_serializable_handles_causal_graph_spec():
    spec = CausalGraphSpec(
        dataset_id="padchest",
        variables=(
            VariableSpec("age_group", VariableKind.CATEGORICAL, encoded_dim=5),
            VariableSpec("sex", VariableKind.CATEGORICAL, encoded_dim=2),
            VariableSpec("tb_status", VariableKind.BINARY),
        ),
        edges=(("age_group", "tb_status"), ("sex", "tb_status")),
        version="2",
    )
    serialized = to_json_serializable({"schema": spec, "learning_rate": 0.001, "count": np.int64(42)})
    dumped = json.dumps(serialized, indent=2, sort_keys=True)
    loaded = json.loads(dumped)

    assert loaded["schema"]["dataset_id"] == "padchest"
    assert loaded["schema"]["version"] == "2"
    assert len(loaded["schema"]["variables"]) == 3
    assert loaded["schema"]["variables"][0]["name"] == "age_group"
    assert loaded["schema"]["variables"][0]["kind"] == "categorical"
    assert loaded["schema"]["variables"][0]["encoded_dim"] == 5
    assert loaded["count"] == 42


def test_save_checkpoint_with_causal_graph_spec(tmp_path: Path):
    ckpt_dir = str(tmp_path / "checkpoints")
    payload = {
        "epoch": 1,
        "step": 100,
        "best_loss": 1.23,
        "params": {"kernel": jnp.ones((4, 4), dtype=jnp.float32)},
        "hparams": {
            "schema": PAD_CHEST_SCHEMA_V2,
            "dataset": "padchest",
            "context_dim": 21,
            "lr": 5e-4,
        },
    }

    save_checkpoint(payload, ckpt_dir, step=100)

    hparams_path = tmp_path / "checkpoints" / "hparams.json"
    assert hparams_path.is_file()
    with open(hparams_path, "r", encoding="utf-8") as f:
        saved_hparams = json.load(f)

    assert saved_hparams["dataset"] == "padchest"
    assert saved_hparams["context_dim"] == 21
    assert saved_hparams["schema"]["dataset_id"] == "padchest"
    assert saved_hparams["schema"]["version"] == "2"

    restored = load_checkpoint(str(tmp_path / "checkpoints" / "100"))
    assert "params" in restored
    assert restored["step"] == 100
    assert restored["best_loss"] == 1.23
    np.testing.assert_allclose(restored["params"]["kernel"], np.ones((4, 4)))


def test_background_artifact_writer_checkpoint_with_causal_graph_spec(tmp_path: Path):
    ckpt_dir = str(tmp_path / "bg_checkpoints")
    payload = {
        "epoch": 2,
        "step": 200,
        "best_loss": 0.95,
        "params": {"kernel": jnp.ones((2, 2), dtype=jnp.float32)},
        "hparams": {
            "schema": PAD_CHEST_SCHEMA_V2,
            "parents_x": list(PAD_CHEST_SCHEMA_V2.variable_names),
        },
    }

    writer = BackgroundArtifactWriter()
    try:
        writer.submit_checkpoint(payload, ckpt_dir, step=200)
        writer.flush()
        assert writer._error is None
    finally:
        writer.close()

    assert (tmp_path / "bg_checkpoints" / "hparams.json").is_file()
    assert (tmp_path / "bg_checkpoints" / "200").is_dir()


def test_padchest_image_model_v2_config_settings():
    config = load_experiment("configs/padchest_image_model_v2_tpu_v6e1.yaml")
    settings = image_model_settings(config)

    assert settings.schema_version == "2"
    assert settings.context_dim == 21
    assert settings.parents_x == ["age_group", "sex", "tb_status", "projection", "view_position", "scanner"]
    assert settings.remote_ckpt_dir == "gs://external-cxr-dataset/padchest/checkpoints_v2"
    assert settings.input_stage_dir == "/mnt/data/dataset/padchest/input-stage/images-224"
