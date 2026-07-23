from config import load_experiment
from training.inference import _checkpoint_root, output_dir


def test_inference_config_targets_the_reference_hvae_run():
    config = load_experiment("configs/morphomnist_inference.yaml")
    assert config.workflow.type == "infer"
    assert config.workflow.checkpoint.endswith("morphomnist/hvae_jax-cpu_22-07-2026")
    assert str(output_dir(config)).endswith("hvae_jax-cpu_22-07-2026_inference/inference")


def test_inference_normalizes_a_run_root_to_checkpoint_root(monkeypatch):
    monkeypatch.setattr("training.inference.path_exists", lambda path: path.endswith("/checkpoints/hparams.json"))
    assert _checkpoint_root("gs://bucket/run") == "gs://bucket/run/checkpoints"
    assert _checkpoint_root("gs://bucket/run/checkpoints/150000") == "gs://bucket/run/checkpoints/150000"
