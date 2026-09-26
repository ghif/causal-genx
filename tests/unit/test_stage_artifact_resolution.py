from training import common


def test_padchest_requires_remote_checkpoint_when_local_reference_is_given(monkeypatch):
    monkeypatch.setattr(common, "path_exists", lambda path: False)
    try:
        common.resolve_checkpoint_reference(
            "checkpoints/padchest/run/checkpoints",
            "gs://bucket/checkpoints",
            require_remote=True,
        )
    except ValueError as exc:
        assert "refusing local fallback" in str(exc)
    else:
        raise AssertionError("missing remote PadChest artifact should not fall back locally")


def test_relative_checkpoint_falls_back_to_configured_gcs_root(monkeypatch):
    local = "checkpoints/morphomnist/predictor/checkpoints"
    remote_root = "gs://medical-airnd/causal-gen/checkpoints"
    remote = "gs://medical-airnd/causal-gen/checkpoints/morphomnist/predictor/checkpoints"

    monkeypatch.setattr(common, "path_exists", lambda path: path == remote)

    assert common.resolve_checkpoint_reference(local, remote_root) == remote


def test_validate_stage_artifacts_checks_padchest_schema_and_complete_steps(monkeypatch):
    paths = {
        "gs://bucket/scm/checkpoints": "gs://bucket/scm/checkpoints/12",
        "gs://bucket/predictor/checkpoints": "gs://bucket/predictor/checkpoints/34",
        "gs://bucket/image/checkpoints": "gs://bucket/image/checkpoints/56",
    }
    hparams = {
        "gs://bucket/scm/checkpoints": {"setup": "sup_pgm", "dataset": "padchest", "parents_x": ["age", "sex"], "context_dim": 2, "input_res": 128},
        "gs://bucket/predictor/checkpoints": {"setup": "sup_aux", "dataset": "padchest", "parents_x": ["age", "sex"], "context_dim": 2, "input_res": 128},
        "gs://bucket/image/checkpoints": {"vae": "hierarchical", "dataset": "padchest", "parents_x": ["age", "sex"], "context_dim": 2, "input_res": 128},
    }
    monkeypatch.setattr(common, "resolve_checkpoint_path", lambda path, **_: paths[path])
    monkeypatch.setattr(common, "checkpoint_is_complete", lambda path: True)
    monkeypatch.setattr(common, "path_exists", lambda path: path in paths or path.endswith("hparams.json"))
    import contextlib
    import io
    import json
    monkeypatch.setattr(
        common,
        "open_file",
        lambda path, mode: contextlib.nullcontext(
            io.StringIO(json.dumps(hparams[path.rsplit("/", 1)[0]]))
        ),
    )
    resolved = common.validate_stage_artifacts(
        *paths.keys(),
        dataset_name="padchest",
        expected_variables=["age", "sex"],
        expected_context_dim=2,
        expected_input_res=128,
        require_complete=True,
        resolve_steps=True,
        strict_schema=True,
    )
    assert resolved == tuple(paths.values())


def test_explicit_gcs_checkpoint_is_not_rewritten(monkeypatch):
    checkpoint = "gs://bucket/checkpoints/morphomnist/run/checkpoints"
    monkeypatch.setattr(common, "path_exists", lambda path: False)

    assert common.resolve_checkpoint_reference(
        checkpoint, "gs://other/checkpoints"
    ) == checkpoint
