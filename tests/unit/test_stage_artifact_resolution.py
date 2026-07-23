from training import common


def test_relative_checkpoint_falls_back_to_configured_gcs_root(monkeypatch):
    local = "checkpoints/morphomnist/predictor/checkpoints"
    remote_root = "gs://medical-airnd/causal-gen/checkpoints"
    remote = "gs://medical-airnd/causal-gen/checkpoints/morphomnist/predictor/checkpoints"

    monkeypatch.setattr(common, "path_exists", lambda path: path == remote)

    assert common.resolve_checkpoint_reference(local, remote_root) == remote


def test_explicit_gcs_checkpoint_is_not_rewritten(monkeypatch):
    checkpoint = "gs://bucket/checkpoints/morphomnist/run/checkpoints"
    monkeypatch.setattr(common, "path_exists", lambda path: False)

    assert common.resolve_checkpoint_reference(
        checkpoint, "gs://other/checkpoints"
    ) == checkpoint
