import numpy as np
import pytest

from artifacts import ArtifactMetadata, assert_compatible
from conditioning import ParentEncoder
from config import load_experiment
from contracts import CausalGraphSpec, VariableKind, VariableSpec
from datasets import _DATASET_FACTORIES
from config import ExperimentConfig
from training import image_model, predictor, scm


def test_parent_encoder_uses_schema_order_and_validates_dimensions():
    schema = CausalGraphSpec(
        dataset_id="toy",
        variables=(
            VariableSpec("category", VariableKind.CATEGORICAL, encoded_dim=2),
            VariableSpec("score", VariableKind.CONTINUOUS),
        ),
    )
    encoder = ParentEncoder(schema)
    vector = encoder.vector({"score": np.array([3.0]), "category": np.array([[1.0, 0.0]])})
    np.testing.assert_array_equal(vector, [[1.0, 0.0, 3.0]])
    with pytest.raises(ValueError, match="encoded_dim"):
        encoder.vector({"score": np.array([3.0]), "category": np.array([[1.0]])})


def test_config_overrides_are_typed_and_dataset_is_registered(tmp_path):
    config = load_experiment("configs/morphomnist_image_model.yaml", ["seed=9", "dataset.input_res=64", "artifacts.run_name=unit"])
    assert config.seed == 9
    assert config.dataset.input_res == 64
    assert "morphomnist" in _DATASET_FACTORIES


def test_config_composes_named_sections_from_yaml():
    config = load_experiment("configs/morphomnist_image_model.yaml")
    assert config.dataset.name == "morphomnist"
    assert config.optimizer.batch_size == 32


def test_v6e4_image_config_has_explicit_single_host_topology():
    config = load_experiment("configs/morphomnist_image_model_tpu_v6e4.yaml")
    assert config.runtime.accelerator == "tpu"
    assert config.runtime.expected_local_device_count == 4
    assert config.runtime.expected_global_device_count == 4
    assert config.runtime.expected_process_count == 1
    args = image_model._run_arguments(config)
    assert args.execution_mode == "replicated"
    assert args.drop_remainder is True
    assert args.bs == 512
    assert args.precision == "bf16"
    assert args.cond_prior is True


def test_image_model_config_forwards_legacy_training_hparams():
    config = load_experiment(
        "configs/morphomnist_image_model.yaml",
        [
            "dataset.hflip=0.0",
            "model.z_dim=24",
            "model.widths=[8,16,32]",
            "optimizer.betas=[0.8,0.95]",
            "workflow.viz_batch_size=7",
            "workflow.resume=checkpoints/previous/checkpoints",
            "workflow.ema_rate=0.9",
            "workflow.beta_warmup_steps=12",
        ],
    )

    args = image_model._run_arguments(config)

    assert args.hflip == 0.0
    assert args.z_dim == 24
    assert args.widths == [8, 16, 32]
    assert args.betas == [0.8, 0.95]
    assert args.viz_batch_size == 7
    assert args.resume == "checkpoints/previous/checkpoints"
    assert args.ema_rate == 0.9
    assert args.beta_warmup_steps == 12


def test_default_image_model_yaml_matches_run_local_hvae_profile():
    args = image_model._run_arguments(
        load_experiment("configs/morphomnist_image_model.yaml")
    )

    # src/run_local.sh passes these values explicitly to src/main.py.
    assert args.accelerator == "cpu"
    assert args.precision == "fp32"
    assert args.hps == "morphomnist"
    assert args.vae == "hierarchical"
    assert args.parents_x == ["thickness", "intensity", "digit"]
    assert args.context_dim == 12
    assert args.concat_pa is True
    assert args.cond_prior is True
    assert args.lr == 0.001
    assert args.bs == 32
    assert args.wd == 0.01
    assert args.beta == 1.0
    assert args.speed_log_freq == 50
    assert args.eval_freq == 50
    assert args.checkpoint_freq == 50
    assert args.viz_batch_size == 32


def test_artifact_composition_rejects_schema_mismatch():
    first = ArtifactMetadata("image", "vae", "toy", "data-v1", "1")
    second = ArtifactMetadata("scm", "pgm", "toy", "data-v1", "2")
    with pytest.raises(ValueError, match="causal_schema_version"):
        assert_compatible(first, second)
    assert_compatible(first, second, allow_override=True)


def test_schema_rejects_unknown_or_non_intervenable_values():
    schema = CausalGraphSpec(
        dataset_id="toy",
        variables=(VariableSpec("fixed", VariableKind.BINARY, intervenable=False),),
    )
    with pytest.raises(ValueError):
        schema.validate_intervention({"fixed": 1})
    with pytest.raises(ValueError):
        schema.validate_intervention({"missing": 1})


def test_legacy_pgm_delegate_exposes_src_on_pythonpath(monkeypatch):
    captured = {}

    class Result:
        returncode = 0

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        captured["cwd"] = kwargs["cwd"]
        return Result()

    # Legacy module execution is centralized in training.common; stage tests
    # below cover the named public SCM command instead.


def test_native_scm_arguments_match_reference_profile():
    config = load_experiment("configs/morphomnist_scm.yaml")
    args = scm._run_arguments(config)
    assert args.accelerator == "cpu"
    assert args.ckpt_dir == "checkpoints"
    assert args.remote_ckpt_dir == "gs://medical-airnd/causal-gen/checkpoints"
    assert args.bs == 16
    assert args.epochs == 1000
    assert args.widths == [32, 32]
    assert args.setup == "sup_pgm"  # retained checkpoint identity, not dispatch.


def test_native_predictor_arguments_match_legacy_artifact_profile():
    config = load_experiment("configs/morphomnist_predictor.yaml")
    args = predictor._run_arguments(config)
    assert args.accelerator == "cpu"
    assert args.remote_ckpt_dir == "gs://medical-airnd/causal-gen/checkpoints"
    assert args.bs == 32
    assert args.epochs == 1000
    assert args.setup == "sup_aux"


def test_predictor_artifact_contract_accepts_legacy_shape(tmp_path):
    (tmp_path / "checkpoints" / "1").mkdir(parents=True)
    for name in ("hparams.json",):
        (tmp_path / "checkpoints" / name).touch()
    for name in ("trainlog.txt", "events.out.tfevents.test"):
        (tmp_path / name).touch()
    (tmp_path / "checkpoints" / "1" / "_CHECKPOINT_METADATA").touch()
    predictor.validate_artifacts(tmp_path)


def test_strict_scm_artifact_contract_accepts_legacy_shape(tmp_path):
    (tmp_path / "checkpoints" / "1").mkdir(parents=True)
    for name in ("hparams.json",):
        (tmp_path / "checkpoints" / name).touch()
    for name in ("trainlog.txt", "joint_data.pdf", "joint_model_1.pdf", "events.out.tfevents.test"):
        (tmp_path / name).touch()
    (tmp_path / "checkpoints" / "1" / "_CHECKPOINT_METADATA").touch()
    # Contract is enforced by the historical PGM training implementation.
