import logging
from io import BytesIO
import numpy as np
from types import SimpleNamespace
import pytest

from training import predictor
from training.predictor import _log_input_normalization


class DummyDataset:
    def __init__(self):
        self.batch_calls = 0
        self.min_max = {"thickness": [0.8, 6.2]}
        self.samples = {
            "thickness": np.array([-1.0, 0.0, 1.0], dtype=np.float32),
            "digit": np.eye(10, dtype=np.float32)[[0, 1, 2]],
        }

    def __len__(self):
        return len(self.samples["thickness"])

    def make_batch(self, indices, rng=None, shuffle=False):
        self.batch_calls += 1
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
        normalization_sample_size=0,
    )

    stats = _log_input_normalization(logger, dataset, dataset, args)

    assert "thickness" in stats
    assert stats["thickness"]["kind"] == "continuous"
    assert np.isclose(stats["thickness"]["norm_min"], -1.0)
    assert np.isclose(stats["thickness"]["norm_max"], 1.0)
    assert "Input Normalization Check" in caplog.text
    assert "Image 'x': sample stats skipped" in caplog.text
    assert dataset.batch_calls == 0
    assert "Variable 'thickness'" in caplog.text


def test_log_input_normalization_optional_bounded_image_sample(caplog):
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test_norm_bounded")
    dataset = DummyDataset()
    args = SimpleNamespace(
        dataset="morphomnist",
        context_norm="[-1,1]",
        input_res=32,
        pad=4,
        parents_x=[],
        normalization_sample_size=2,
    )

    _log_input_normalization(logger, dataset, dataset, args)

    assert dataset.batch_calls == 1
    assert "Image 'x': shape=(1, 32, 32)" in caplog.text


class ItemDataset:
    cache_fingerprint = "item-dataset"

    def __init__(self):
        self.calls = {}
        self.samples = {"y": np.arange(4, dtype=np.float32)}
        self.min_max = {}

    def __len__(self):
        return 4

    def __getitem__(self, index):
        self.calls[index] = self.calls.get(index, 0) + 1
        return {
            "x": np.full((1, 4, 4), index, dtype=np.float32),
            "y": np.asarray(index, dtype=np.float32),
        }


def test_cached_item_dataset_reuses_bounded_disk_cache(tmp_path):
    base = ItemDataset()
    cached = predictor._CachedItemDataset(
        base,
        cache_dir=str(tmp_path),
        max_items=2,
        workers=2,
        name="train",
    )

    first = cached.make_batch(np.array([0, 1]))
    second = cached.make_batch(np.array([0, 1]))

    np.testing.assert_array_equal(first["x"], second["x"])
    assert base.calls == {0: 1, 1: 1}
    assert len(list(tmp_path.rglob("*.npz"))) <= 2


def test_predictor_epoch_batches_prefetch_preserves_order():
    base = ItemDataset()
    rng = np.random.default_rng(0)
    batches = list(
        predictor._predictor_epoch_batches(
            base,
            2,
            shuffle=False,
            drop_last=False,
            rng=rng,
            prefetch_batches=2,
        )
    )

    assert [batch["x"].shape[0] for batch in batches] == [2, 2]
    np.testing.assert_array_equal(np.asarray(batches[0]["y"]).reshape(-1), np.array([0.0, 1.0]))
    np.testing.assert_array_equal(np.asarray(batches[1]["y"]).reshape(-1), np.array([2.0, 3.0]))


def test_pretrained_weights_are_materialized_from_gcs(tmp_path, monkeypatch):
    remote_path = "gs://cxr-rait/checkpoints/pretrained/weights.npz"
    cache_path = tmp_path / "weights.npz"
    monkeypatch.setattr(predictor, "local_staging_path", lambda _: str(cache_path))
    monkeypatch.setattr(predictor, "open_file", lambda path, mode: BytesIO(b"canonical-weights"))

    assert predictor._materialize_pretrained_weights(remote_path) == str(cache_path)
    assert cache_path.read_bytes() == b"canonical-weights"

    with pytest.raises(ValueError, match="gs:// URI"):
        predictor._materialize_pretrained_weights("checkpoints/pretrained/weights.npz")
