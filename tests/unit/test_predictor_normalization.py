import logging
from io import BytesIO
import numpy as np
from types import SimpleNamespace
import pytest

from training import predictor
from training.predictor import _log_input_normalization


class DummyDataset:
    def __init__(self):
        self.min_max = {"thickness": [0.8, 6.2]}
        self.samples = {
            "thickness": np.array([-1.0, 0.0, 1.0], dtype=np.float32),
            "digit": np.eye(10, dtype=np.float32)[[0, 1, 2]],
        }

    def make_batch(self, indices, rng=None, shuffle=False):
        return {"x": np.ones((len(indices), 1, 32, 32), dtype=np.float32) * 128.0}


def test_log_input_normalization(caplog):
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test_norm")
    dataset = DummyDataset()
    args = SimpleNamespace(
        dataset="morphomnist",
        context_norm="[-1,1]",
        input_res=32,
        pad=4,
        parents_x=["thickness", "digit"],
    )

    stats = _log_input_normalization(logger, dataset, dataset, args)

    assert "thickness" in stats
    assert stats["thickness"]["kind"] == "continuous"
    assert np.isclose(stats["thickness"]["norm_min"], -1.0)
    assert np.isclose(stats["thickness"]["norm_max"], 1.0)
    assert "Input Normalization Check" in caplog.text
    assert "Image 'x'" in caplog.text
    assert "Variable 'thickness'" in caplog.text


def test_pretrained_weights_are_materialized_from_gcs(tmp_path, monkeypatch):
    remote_path = "gs://cxr-rait/checkpoints/pretrained/weights.npz"
    cache_path = tmp_path / "weights.npz"
    monkeypatch.setattr(predictor, "local_staging_path", lambda _: str(cache_path))
    monkeypatch.setattr(predictor, "open_file", lambda path, mode: BytesIO(b"canonical-weights"))

    assert predictor._materialize_pretrained_weights(remote_path) == str(cache_path)
    assert cache_path.read_bytes() == b"canonical-weights"

    with pytest.raises(ValueError, match="gs:// URI"):
        predictor._materialize_pretrained_weights("checkpoints/pretrained/weights.npz")
