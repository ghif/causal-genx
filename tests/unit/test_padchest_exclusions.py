import csv
import importlib
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

padchest_module = importlib.import_module("data.padchest")
from data.padchest import PadChestProvider, stage_padchest_images


MISSING_IMAGE = "216840111366964012558082906712009300162151055_00-078-079.png"
MISSING_SOURCE = f"gs://external-cxr-dataset/padchest/unpacked/images-224/images-224/{MISSING_IMAGE}"


def _write_metadata(path, image_ids):
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
        for index, image_id in enumerate(image_ids):
            writer.writerow(
                {
                    "PatientID": f"patient-{index:02d}",
                    "ImageID": image_id,
                    "StudyDate_DICOM": "20140101",
                    "PatientBirth": "1970",
                    "Labels": "['tuberculosis']",
                    "PatientSex_DICOM": "M",
                    "Pediatric": "no",
                    "Projection": "PA",
                    "ViewPosition_DICOM": "POSTEROANTERIOR",
                }
            )


def _write_pngs(image_dir, image_ids):
    image_dir.mkdir(parents=True, exist_ok=True)
    for index, image_id in enumerate(image_ids):
        Image.fromarray(np.full((6, 6), index, dtype=np.uint8), mode="L").save(image_dir / image_id)


def _settings(tmp_path, image_ids, excluded_sources=()):
    metadata = tmp_path / "padchest.csv"
    images = tmp_path / "source-images"
    _write_metadata(metadata, image_ids)
    _write_pngs(images, image_ids)
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
        input_stage_max_items=100,
        input_stage_max_bytes=1_000_000,
        input_stage_workers=2,
        excluded_sources=list(excluded_sources),
    )


def _all_image_ids():
    return [MISSING_IMAGE] + [f"image-{index:02d}.png" for index in range(9)]


def test_padchest_exclusion_exact_match_and_manifest_reporting(tmp_path):
    settings = _settings(
        tmp_path,
        _all_image_ids(),
        excluded_sources=[{"source": f"{tmp_path}/source-images/{MISSING_IMAGE}", "reason": "confirmed absent"}],
    )
    (tmp_path / "source-images" / MISSING_IMAGE).unlink()

    summary = stage_padchest_images(settings, execute=True)

    assert summary["excluded_items"] == 1
    assert summary["required_items"] == 9
    assert summary["copied"] == 9
    excluded = summary["excluded"][0]
    assert excluded["source"].endswith(MISSING_IMAGE)
    assert excluded["reason"] == "confirmed absent"
    assert excluded["split"] in {"train", "valid", "test"}
    assert isinstance(excluded["row_index"], int)

    entries = [json.loads(line) for line in (tmp_path / "stage" / "manifest.jsonl").read_text().splitlines()]
    excluded_entries = [entry for entry in entries if entry.get("kind") == "excluded"]
    assert excluded_entries == [excluded]
    item_sources = {entry.get("source") for entry in entries if entry.get("kind") == "item"}
    assert excluded["source"] not in item_sources


def test_configured_gcs_exclusion_is_not_transferred_and_is_manifested(tmp_path, monkeypatch):
    settings = _settings(
        tmp_path,
        _all_image_ids(),
        excluded_sources=[{"source": MISSING_SOURCE, "reason": "confirmed absent source object"}],
    )
    settings.image_prefix = MISSING_SOURCE.rsplit("/", 1)[0]
    transferred_sources = []

    def fake_bulk_copy(records, _stage_root, _workers):
        transferred_sources.extend(record["source"] for record in records)
        for record in records:
            record["bytes"] = 10
        return list(records), len(records), 0

    monkeypatch.setattr(padchest_module, "_source_size", lambda _source: 10)
    monkeypatch.setattr(padchest_module, "_run_gcloud_bulk_copy", fake_bulk_copy)

    summary = stage_padchest_images(settings, execute=True)

    assert summary["excluded_items"] == 1
    excluded = summary["excluded"][0]
    assert excluded["kind"] == "excluded"
    assert excluded["image_id"] == MISSING_IMAGE
    assert excluded["source"] == MISSING_SOURCE
    assert excluded["reason"] == "confirmed absent source object"
    assert excluded["split"] in {"train", "valid", "test"}
    assert isinstance(excluded["row_index"], int)
    assert MISSING_SOURCE not in transferred_sources
    assert len(transferred_sources) == summary["required_items"] == 9
    entries = [json.loads(line) for line in (tmp_path / "stage" / "manifest.jsonl").read_text().splitlines()]
    assert [entry for entry in entries if entry.get("kind") == "excluded"] == summary["excluded"]
    assert MISSING_SOURCE not in {entry.get("source") for entry in entries if entry.get("kind") == "item"}


def test_padchest_dataset_excludes_same_row_from_predictor_splits(tmp_path):
    excluded_source = f"{tmp_path}/source-images/{MISSING_IMAGE}"
    settings = _settings(
        tmp_path,
        _all_image_ids(),
        excluded_sources=[{"source": excluded_source, "reason": "confirmed absent"}],
    )
    provider = PadChestProvider(
        settings.data_dir,
        input_res=settings.input_res,
        metadata=settings.metadata,
        image_prefix=settings.image_prefix,
        tb_label_mode=settings.tb_label_mode,
        seed=settings.seed,
        excluded_sources=settings.excluded_sources,
    )

    datasets = {split: provider.load_split(split) for split in ("train", "valid", "test")}

    assert sum(len(dataset) for dataset in datasets.values()) == 9
    assert sum(len(dataset.excluded_records) for dataset in datasets.values()) == 1
    assert all(record["source"] != excluded_source for dataset in datasets.values() for record in dataset.image_records())


def test_padchest_stage_refuses_unexpected_missing_object(tmp_path):
    settings = _settings(
        tmp_path,
        _all_image_ids(),
        excluded_sources=[{"source": f"{tmp_path}/source-images/{MISSING_IMAGE}", "reason": "confirmed absent"}],
    )
    unexpected = tmp_path / "source-images" / "image-03.png"
    unexpected.unlink()

    try:
        stage_padchest_images(settings, execute=True)
    except RuntimeError as err:
        assert "Unexpected missing PadChest source object" in str(err)
        assert "image-03.png" in str(err)
    else:
        raise AssertionError("unexpected missing PadChest image should be refused")


def test_configured_padchest_predictor_default_excludes_confirmed_gcs_source():
    from config import load_experiment

    config = load_experiment("configs/padchest_predictor.yaml")
    tpu_config = load_experiment("configs/padchest_predictor_tpu_v6e1.yaml")

    assert config.dataset.name == "padchest"
    assert tpu_config.dataset.name == "padchest"
    assert config.dataset.excluded_sources == [
        {
            "source": MISSING_SOURCE,
            "reason": "confirmed absent source object; gcloud storage ls matches no objects",
        }
    ]
    assert tpu_config.dataset.excluded_sources == config.dataset.excluded_sources
