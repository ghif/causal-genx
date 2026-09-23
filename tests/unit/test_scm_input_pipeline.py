import csv

import numpy as np
import pytest

from data.padchest import PAD_CHEST_SCHEMA, PadChestDataset
from training.scm import epoch_batches


class MetadataOnlyDataset:
    samples = {
        "a": np.arange(6, dtype=np.float32),
        "b": np.eye(2, dtype=np.float32)[[0, 1, 0, 1, 0, 1]],
    }

    def __len__(self):
        return 6

    def make_batch(self, *_args, **_kwargs):
        raise AssertionError("image-backed make_batch should not be called")


def test_epoch_batches_can_use_metadata_samples_without_image_batch():
    batches = list(
        epoch_batches(
            MetadataOnlyDataset(),
            4,
            shuffle=False,
            drop_last=False,
            rng=np.random.default_rng(0),
            variables=("a", "b"),
        )
    )

    assert [batch["a"].shape for batch in batches] == [(4, 1), (2, 1)]
    np.testing.assert_array_equal(np.asarray(batches[0]["a"]).ravel(), np.arange(4, dtype=np.float32))
    assert batches[0]["b"].shape == (4, 2)
    assert "x" not in batches[0]


def test_epoch_batches_falls_back_to_dataset_batch_when_metadata_missing():
    with pytest.raises(AssertionError, match="make_batch"):
        next(
            epoch_batches(
                MetadataOnlyDataset(),
                2,
                shuffle=False,
                drop_last=False,
                rng=np.random.default_rng(0),
                variables=("missing",),
            )
        )


def _write_padchest_metadata(path):
    fieldnames = [
        "PatientID",
        "ImageID",
        "StudyDate_DICOM",
        "PatientBirth",
        "Labels",
        "PatientSex_DICOM",
        "Pediatric",
        "Projection",
        "ViewPosition_DICOM",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(10):
            writer.writerow(
                {
                    "PatientID": f"patient-{index:02d}",
                    "ImageID": f"image-{index:02d}.png",
                    "StudyDate_DICOM": "20140101",
                    "PatientBirth": "1970",
                    "Labels": "['tuberculosis']" if index % 2 else "[]",
                    "PatientSex_DICOM": "M" if index % 2 else "F",
                    "Pediatric": "no",
                    "Projection": "PA",
                    "ViewPosition_DICOM": "POSTEROANTERIOR",
                }
            )


def test_padchest_scm_batches_skip_remote_png_loading(tmp_path, monkeypatch):
    metadata_path = tmp_path / "padchest.csv"
    _write_padchest_metadata(metadata_path)
    dataset = PadChestDataset(
        root=str(tmp_path),
        metadata=str(metadata_path),
        image_prefix="gs://example-bucket/padchest/images-224",
        split="train",
        input_res=128,
        tb_label_mode="tb_or_sequelae",
        seed=7,
    )
    def fail_image_load(_index):
        raise AssertionError("PNG download attempted")

    monkeypatch.setattr(dataset, "_get_image", fail_image_load)

    batch = next(
        epoch_batches(
            dataset,
            3,
            shuffle=False,
            drop_last=False,
            rng=np.random.default_rng(0),
            variables=PAD_CHEST_SCHEMA.variable_names,
        )
    )

    assert "x" not in batch
    assert set(batch) == set(PAD_CHEST_SCHEMA.variable_names)
    assert all(value.shape[0] == 3 for value in batch.values())
