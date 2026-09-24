import csv
import logging
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image

from data.padchest import PadChestDataset, stage_padchest_images
from training.predictor import _CachedItemDataset, _prepare_input_pipeline


def _write_metadata(path, count=10):
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
        for index in range(count):
            writer.writerow(
                {
                    "PatientID": f"patient-{index:02d}",
                    "ImageID": f"image-{index:02d}.png",
                    "StudyDate_DICOM": "20140101",
                    "PatientBirth": "1970",
                    "Labels": "['tuberculosis']",
                    "PatientSex_DICOM": "M",
                    "Pediatric": "no",
                    "Projection": "PA",
                    "ViewPosition_DICOM": "POSTEROANTERIOR",
                }
            )


def _write_pngs(image_dir, count=10):
    image_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        data = np.full((6, 6), index, dtype=np.uint8)
        Image.fromarray(data, mode="L").save(image_dir / f"image-{index:02d}.png")


def _settings(tmp_path, *, max_items=100, max_bytes=1_000_000):
    metadata = tmp_path / "padchest.csv"
    images = tmp_path / "source-images"
    _write_metadata(metadata)
    _write_pngs(images)
    return SimpleNamespace(
        data_dir=str(tmp_path),
        metadata=str(metadata),
        image_prefix=str(images),
        input_res=8,
        pad=0,
        context_norm="[-1,1]",
        tb_label_mode="tb_or_sequelae",
        seed=7,
        input_stage_dir=str(tmp_path / "stage"),
        input_stage_manifest=str(tmp_path / "stage" / "manifest.jsonl"),
        input_stage_max_items=max_items,
        input_stage_max_bytes=max_bytes,
        input_stage_workers=2,
    )


def test_padchest_stage_execute_then_require_uses_local_files(tmp_path):
    settings = _settings(tmp_path)

    summary = stage_padchest_images(settings, splits=("train",), execute=True)
    assert summary["copied"] == summary["required_items"]
    assert not summary["refusal"]

    shutil.rmtree(settings.image_prefix)
    dataset = PadChestDataset(
        root=settings.data_dir,
        metadata=settings.metadata,
        image_prefix=settings.image_prefix,
        split="train",
        input_res=8,
        tb_label_mode="tb_or_sequelae",
        seed=7,
        stage_dir=settings.input_stage_dir,
        stage_manifest=settings.input_stage_manifest,
        stage_mode="require",
    )

    batch = dataset.make_batch([0, 1])
    assert batch["x"].shape == (2, 1, 8, 8)
    assert batch["tb_status"].shape == (2, 1)


def test_padchest_stage_policy_refuses_over_item_budget(tmp_path):
    settings = _settings(tmp_path, max_items=1)

    summary = stage_padchest_images(settings, splits=("train",), execute=False)

    assert summary["required_items"] > 1
    assert "input_stage_max_items" in summary["refusal"]


def test_padchest_require_refuses_missing_manifest(tmp_path):
    settings = _settings(tmp_path)

    try:
        PadChestDataset(
            root=settings.data_dir,
            metadata=settings.metadata,
            image_prefix="gs://example-bucket/padchest/images-224",
            split="train",
            input_res=8,
            tb_label_mode="tb_or_sequelae",
            seed=7,
            stage_dir=settings.input_stage_dir,
            stage_manifest=settings.input_stage_manifest,
            stage_mode="require",
        )
    except RuntimeError as err:
        assert "input_stage_mode=require" in str(err)
    else:
        raise AssertionError("missing PadChest stage manifest should be refused")


class TinyDataset:
    samples = {}
    min_max = {}

    def __len__(self):
        return 2


def test_prepare_input_pipeline_keeps_generic_auto_cache_fallback(tmp_path):
    dataset = TinyDataset()
    args = SimpleNamespace(
        dataset="morphomnist",
        input_cache="auto",
        input_cache_dir=str(tmp_path / "cache"),
        input_cache_max_items=8,
        input_prefetch_workers=2,
        input_prefetch_batches=1,
    )

    datasets, train = _prepare_input_pipeline(logging.getLogger("test"), {"train": dataset}, dataset, args)

    assert train is dataset
    assert datasets["train"] is dataset
    assert not isinstance(train, _CachedItemDataset)
