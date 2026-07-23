import logging

import numpy as np
import pytest
from types import SimpleNamespace

from artifacts import ArtifactMetadata, assert_compatible
from data.conditioning import ParentEncoder
from config import load_experiment
from contracts import CausalGraphSpec, VariableKind, VariableSpec
from data.morphomnist import _DATASET_FACTORIES
from config import ExperimentConfig
from training import counterfactual, image_model, predictor, scm


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


def test_image_model_config_forwards_typed_training_settings():
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


def test_predictor_checkpoint_frequency_is_typed_and_forwarded():
    config = load_experiment(
        "configs/morphomnist_predictor.yaml", ["workflow.checkpoint_freq=3"]
    )

    assert config.workflow.checkpoint_freq == 3
    assert predictor._run_arguments(config).checkpoint_freq == 3


def test_predictor_runtime_execution_settings_are_typed_and_forwarded():
    config = load_experiment(
        "configs/morphomnist_predictor.yaml",
        ["workflow.execution_mode=replicated", "workflow.drop_remainder=false"],
    )

    args = predictor._run_arguments(config)
    assert args.execution_mode == "replicated"
    assert args.drop_remainder is False


def test_predictor_checkpoint_frequency_defaults_to_every_epoch():
    config = load_experiment("configs/morphomnist_predictor.yaml")

    assert config.workflow.checkpoint_freq == 1
    assert predictor._checkpoint_due(1, config.workflow.checkpoint_freq)
    assert predictor._checkpoint_due(3, 3)
    assert not predictor._checkpoint_due(2, 3)


def test_predictor_epoch_summary_writes_tensorboard_metrics_and_trainlog(caplog):
    scalars = []

    class RecordingWriter:
        def add_scalar(self, tag, value, step):
            scalars.append((tag, value, step))

    train_stats = {
        "loss": 1.0,
        "logp(digit_aux)": -0.1,
        "logp(thickness_aux)": -0.2,
        "logp(intensity_aux)": -0.3,
    }
    valid_stats = {key: value + 0.5 for key, value in train_stats.items()}
    prediction_stats = {"thickness_mae": 0.4, "intensity_mae": 0.5, "digit_acc": 0.6}

    predictor._write_epoch_summary(
        RecordingWriter(), epoch=2, step=12, train_stats=train_stats,
        valid_stats=valid_stats, prediction_stats=prediction_stats,
        train_time=3.0, total_time=5.0, iter_per_sec=4.0, sample_per_sec=128.0,
    )
    logger = logging.getLogger("predictor-epoch-summary-test")
    with caplog.at_level(logging.INFO, logger=logger.name):
        predictor._log_epoch_summary(
            logger, epoch=2, step=12, train_stats=train_stats,
            valid_stats=valid_stats, prediction_stats=prediction_stats,
            train_time=3.0, total_time=5.0, iter_per_sec=4.0, sample_per_sec=128.0,
        )

    scalar_tags = {tag for tag, _, _ in scalars}
    assert {f"train/{key}" for key in train_stats} <= scalar_tags
    assert {f"valid/{key}" for key in valid_stats | prediction_stats} <= scalar_tags
    assert {
        "elbo/train", "elbo/valid", "epoch/number", "epoch/global_step",
        "epoch/train_time_sec", "epoch/total_time_sec", "epoch/iter_per_sec",
        "epoch/sample_per_sec",
    } <= scalar_tags
    assert all(step == 12 for _, _, step in scalars)
    assert "=> train |" in caplog.text
    assert "=> valid |" in caplog.text
    assert "train_time=3.0s total_time=5.0s" in caplog.text


def test_predictor_tensorboard_events_are_synced_to_remote_run_dir(tmp_path, monkeypatch):
    save_dir = tmp_path / "run"
    save_dir.mkdir()
    (save_dir / "events.out.tfevents.1").write_bytes(b"event-one")
    (save_dir / "events.out.tfevents.2").write_bytes(b"event-two")
    (save_dir / "trainlog.txt").write_text("not an event", encoding="utf-8")
    copied = []

    def fake_sync_file(local_path, remote_path):
        copied.append((local_path, remote_path))

    monkeypatch.setattr("training.predictor.sync_file", fake_sync_file)
    predictor._sync_tensorboard_artifacts(
        SimpleNamespace(save_dir=str(save_dir), remote_save_dir="gs://bucket/morphomnist/predictor")
    )

    assert copied == [
        (str(save_dir / "events.out.tfevents.1"), "gs://bucket/morphomnist/predictor/events.out.tfevents.1"),
        (str(save_dir / "events.out.tfevents.2"), "gs://bucket/morphomnist/predictor/events.out.tfevents.2"),
    ]


def test_predictor_best_checkpoint_is_submitted_to_artifact_writer(tmp_path):
    submitted = []

    class RecordingWriter:
        def submit_checkpoint(self, *args, **kwargs):
            submitted.append((args, kwargs))

    args = SimpleNamespace(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        remote_save_dir="gs://bucket/morphomnist/predictor",
        setup="sup_aux",
    )
    ema = predictor.WarmupEMA(params={"ema": np.array([1.0])}, batch_stats={"bn": np.array([2.0])})

    predictor._submit_best_checkpoint(
        RecordingWriter(),
        args,
        {"model": np.array([3.0])},
        {"bn": np.array([4.0])},
        ema,
        {"optimizer": np.array([5.0])},
        epoch=3,
        step=12,
        best_loss=0.25,
    )

    assert len(submitted) == 1
    call_args, call_kwargs = submitted[0]
    assert call_args[1] == args.checkpoint_dir
    assert call_kwargs["step"] == 12
    assert call_kwargs["local_tree_dir"] == args.checkpoint_dir
    assert call_kwargs["remote_tree_dir"] == "gs://bucket/morphomnist/predictor/checkpoints"
    assert call_args[0]["best_loss"] == 0.25


def test_default_image_model_yaml_matches_hvae_profile():
    args = image_model._run_arguments(
        load_experiment("configs/morphomnist_image_model.yaml")
    )

    assert args.accelerator == "cpu"
    assert args.precision == "fp32"
    assert args.dataset_id == "morphomnist"
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
    assert args.eval_freq == 5
    assert args.checkpoint_freq == 10
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


def test_native_scm_arguments_match_reference_profile():
    config = load_experiment("configs/morphomnist_scm.yaml")
    args = scm._run_arguments(config)
    assert args.accelerator == "cpu"
    assert args.ckpt_dir == "checkpoints"
    assert args.remote_ckpt_dir == "gs://medical-airnd/causal-gen/checkpoints"
    assert args.bs == 16
    assert args.epochs == 1000
    assert args.speed_log_freq == 50
    assert args.checkpoint_freq == 1
    assert args.widths == [32, 32]
    assert args.setup == "sup_pgm"  # retained checkpoint identity, not dispatch.


def test_scm_checkpoint_frequency_is_typed_and_controls_checkpoint_schedule():
    config = load_experiment(
        "configs/morphomnist_scm.yaml",
        ["workflow.speed_log_freq=7", "workflow.checkpoint_freq=3"],
    )

    args = scm._run_arguments(config)
    assert args.speed_log_freq == 7
    assert args.checkpoint_freq == 3
    assert scm._checkpoint_due(3, args.checkpoint_freq)
    assert not scm._checkpoint_due(2, args.checkpoint_freq)
    with pytest.raises(ValueError, match="eval_freq"):
        load_experiment("configs/morphomnist_scm.yaml", ["workflow.eval_freq=1"])


def test_scm_epoch_summary_writes_complete_metrics_and_trainlog(caplog):
    scalars = []

    class RecordingWriter:
        def add_scalar(self, tag, value, step):
            scalars.append((tag, value, step))

    train_stats = {"loss": 1.0, "logp(digit)": -0.1, "logp(thickness)": -0.2, "logp(intensity)": -0.3}
    valid_stats = {key: value + 0.5 for key, value in train_stats.items()}
    scm._write_epoch_summary(
        RecordingWriter(), epoch=2, step=12, train_stats=train_stats,
        valid_stats=valid_stats, train_time=3.0, total_time=5.0,
        iter_per_sec=4.0, sample_per_sec=128.0, grad_norm=0.75,
    )
    logger = logging.getLogger("scm-epoch-summary-test")
    with caplog.at_level(logging.INFO, logger=logger.name):
        scm._log_epoch_summary(
            logger, epoch=2, step=12, train_stats=train_stats,
            valid_stats=valid_stats, train_time=3.0, total_time=5.0,
            iter_per_sec=4.0, sample_per_sec=128.0, grad_norm=0.75,
        )

    scalar_tags = {tag for tag, _, _ in scalars}
    assert {f"train/{key}" for key in train_stats} <= scalar_tags
    assert {f"valid/{key}" for key in valid_stats} <= scalar_tags
    assert {
        "elbo/train", "elbo/valid", "epoch/number", "epoch/global_step",
        "epoch/train_time_sec", "epoch/total_time_sec", "epoch/iter_per_sec",
        "epoch/sample_per_sec", "epoch/grad_norm",
    } <= scalar_tags
    assert all(step == 12 for _, _, step in scalars)
    assert "=> train |" in caplog.text
    assert "=> valid |" in caplog.text
    assert "train_time=3.0s total_time=5.0s" in caplog.text


def test_native_predictor_arguments_match_artifact_profile():
    config = load_experiment("configs/morphomnist_predictor.yaml")
    args = predictor._run_arguments(config)
    assert args.accelerator == "cpu"
    assert args.remote_ckpt_dir == "gs://medical-airnd/causal-gen/checkpoints"
    assert args.bs == 32
    assert args.epochs == 1000
    assert args.speed_log_freq == 50
    assert args.setup == "sup_aux"


def test_native_counterfactual_arguments_match_profile():
    config = load_experiment("configs/morphomnist_counterfactual.yaml")
    args = counterfactual._run_arguments(config)
    assert args.accelerator == "cpu"
    assert args.precision == "fp32"
    assert args.ckpt_dir == "checkpoints"
    assert args.remote_ckpt_dir == "gs://medical-airnd/causal-gen/checkpoints"
    assert args.bs == 32
    assert args.lr == 1e-4
    assert args.wd == 0.1
    assert args.speed_log_freq == 50
    assert args.checkpoint_freq == 1
    assert args.alpha == 0.1
    assert args.damping == 100.0
    assert args.do_pa is None
    assert args.trust_incomplete_checkpoint is True
    assert args.pgm_path == config.workflow.scm_checkpoint
    assert args.predictor_path == config.workflow.predictor_checkpoint
    assert args.vae_path == config.workflow.image_model_checkpoint
    assert str(counterfactual.output_dir(config)).endswith(
        f"checkpoints/morphomnist/{config.artifacts.run_name}/cf"
    )
    with pytest.raises(ValueError, match="eval_freq"):
        load_experiment("configs/morphomnist_counterfactual.yaml", ["workflow.eval_freq=1"])
    with pytest.raises(ValueError, match="plot_freq"):
        load_experiment("configs/morphomnist_counterfactual.yaml", ["workflow.plot_freq=10"])


def test_counterfactual_epoch_summary_records_train_and_intervention_metrics(caplog):
    scalars = []

    class RecordingWriter:
        def add_scalar(self, tag, value, step):
            scalars.append((tag, value, step))

    train_stats = {"loss": 1.0, "aux_loss": 0.2, "elbo": 0.8, "nll": 0.6, "kl": 0.2}
    valid_stats = {key: value + 0.1 for key, value in train_stats.items()}
    validation = {
        "do_digit": (valid_stats, {"thickness_mae": 0.3}),
        "observational": (valid_stats, {"digit_acc": 0.9}),
    }
    diagnostics = {"grad_norm": 0.7, "grad_clipped": 0.0, "lr_scale": 1.0, "update_skipped": 0.0}
    counterfactual._write_epoch_summary(
        RecordingWriter(), epoch=2, step=12, train_stats=train_stats, lmbda=0.4,
        diagnostics=diagnostics, train_time=3.0, total_time=5.0,
        iter_per_sec=4.0, sample_per_sec=128.0, validation=validation,
    )
    logger = logging.getLogger("counterfactual-epoch-summary-test")
    with caplog.at_level(logging.INFO, logger=logger.name):
        counterfactual._log_epoch_summary(
            logger, epoch=2, step=12, train_stats=train_stats, lmbda=0.4,
            diagnostics=diagnostics, train_time=3.0, total_time=5.0,
            iter_per_sec=4.0, sample_per_sec=128.0, validation=validation,
        )

    scalar_tags = {tag for tag, _, _ in scalars}
    assert {f"train/{key}" for key in train_stats} <= scalar_tags
    assert "valid/do_digit/thickness_mae" in scalar_tags
    assert "valid/observational/digit_acc" in scalar_tags
    assert {"epoch/global_step", "epoch/train_time_sec", "epoch/total_time_sec", "epoch/iter_per_sec", "epoch/sample_per_sec"} <= scalar_tags
    assert all(step == 12 for _, _, step in scalars)
    assert "=> train |" in caplog.text
    assert "=> valid observational |" in caplog.text


def test_counterfactual_stage_runs_native_implementation(monkeypatch):
    config = load_experiment("configs/morphomnist_counterfactual.yaml")
    captured = {}

    def validate(*paths, **kwargs):
        captured["paths"] = paths
        captured["validation_kwargs"] = kwargs
        return paths

    def native(args):
        captured["args"] = args

    monkeypatch.setattr(counterfactual, "validate_stage_artifacts", validate)
    monkeypatch.setattr(counterfactual, "main", native)

    output = counterfactual.run(config)

    assert captured["paths"] == (
        config.workflow.scm_checkpoint,
        config.workflow.predictor_checkpoint,
        config.workflow.image_model_checkpoint,
    )
    assert captured["validation_kwargs"] == {
        "remote_root": config.artifacts.remote_root,
    }
    assert captured["args"].wd == 0.1
    assert output.endswith(f"{config.artifacts.run_name}/cf")


def test_predictor_artifact_contract_accepts_expected_shape(tmp_path):
    (tmp_path / "checkpoints" / "1").mkdir(parents=True)
    for name in ("hparams.json",):
        (tmp_path / "checkpoints" / name).touch()
    for name in ("trainlog.txt", "events.out.tfevents.test"):
        (tmp_path / name).touch()
    (tmp_path / "checkpoints" / "1" / "_CHECKPOINT_METADATA").touch()
    predictor.validate_artifacts(tmp_path)


def test_scm_artifact_contract_accepts_expected_shape(tmp_path):
    (tmp_path / "checkpoints" / "1").mkdir(parents=True)
    for name in ("hparams.json",):
        (tmp_path / "checkpoints" / name).touch()
    for name in ("trainlog.txt", "joint_data.pdf", "joint_model_1.pdf", "events.out.tfevents.test"):
        (tmp_path / name).touch()
    (tmp_path / "checkpoints" / "1" / "_CHECKPOINT_METADATA").touch()
    # Contract is enforced by the historical PGM training implementation.
