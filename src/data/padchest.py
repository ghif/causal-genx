"""PadChest PNG/CSV provider with patient-safe metadata conditioning."""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from contracts import Batch, CausalGraphSpec, DatasetSpec, ImageSpec, VariableKind, VariableSpec
from utils import normalize

PAD_CHEST_SCHEMA = CausalGraphSpec(
    dataset_id="padchest",
    version="1",
    variables=(
        VariableSpec("age_at_study", VariableKind.CONTINUOUS, normalization="[-1,1]"),
        VariableSpec("sex", VariableKind.CATEGORICAL, encoded_dim=2),
        VariableSpec("pediatric", VariableKind.BINARY),
        VariableSpec("tb_status", VariableKind.BINARY),
        VariableSpec("projection", VariableKind.CATEGORICAL, encoded_dim=5),
        VariableSpec("view_position", VariableKind.CATEGORICAL, encoded_dim=6),
        VariableSpec("study_year", VariableKind.CONTINUOUS, normalization="[-1,1]"),
    ),
    edges=(("age_at_study", "tb_status"), ("sex", "tb_status"), ("pediatric", "tb_status")),
)

AGE_AT_STUDY_MIN = 0.0
AGE_AT_STUDY_MAX = 110.0
STUDY_YEAR_MIN = 2007.0
STUDY_YEAR_MAX = 2017.0
PROJECTIONS = ("PA", "AP", "AP_horizontal", "L", "COSTAL")
VIEW_POSITIONS = ("POSTEROANTERIOR", "ANTEROPOSTERIOR", "LATERAL", "AP", "PA", "OTHER")

# Backwards-compatible private aliases used by existing derivation code.
_PROJECTIONS = PROJECTIONS
_VIEWS = VIEW_POSITIONS


def _open_binary(path: str):
    if path.startswith("gs://"):
        import fsspec
        return fsspec.open(path, mode="rb").open()
    return open(path, "rb")


def _as_list(value: str) -> list[str]:
    try:
        parsed = ast.literal_eval(value)
        return [str(item).strip().lower() for item in parsed] if isinstance(parsed, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return [item.strip().lower() for item in value.strip("[]").split(",") if item.strip()]


def _category(value: str, choices: tuple[str, ...], fallback: str) -> int:
    value = str(value or "").strip().upper()
    return choices.index(value) if value in choices else choices.index(fallback)


def _read_rows(metadata: str) -> list[dict[str, str]]:
    with _open_binary(metadata) as handle:
        return list(csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8", newline="")))


def _derive(row: Mapping[str, str], tb_label_mode: str) -> dict[str, Any]:
    date = str(row.get("StudyDate_DICOM", ""))
    year = int(date[:4]) if date[:4].isdigit() else 2014
    birth = float(row.get("PatientBirth", "")) if str(row.get("PatientBirth", "")).strip() else float(year - 45)
    age = float(np.clip(year - birth, 0.0, 110.0))
    labels = set(_as_list(row.get("Labels", "")))
    if tb_label_mode == "tb_only":
        tb = "tuberculosis" in labels
    else:
        tb = bool(labels.intersection({"tuberculosis", "tuberculosis sequelae"}))
    sex = 1 if str(row.get("PatientSex_DICOM", "")).upper() == "M" else 0
    return {
        "age_at_study": normalize(np.asarray([age], dtype=np.float32), x_min=AGE_AT_STUDY_MIN, x_max=AGE_AT_STUDY_MAX)[0],
        "sex": np.eye(2, dtype=np.float32)[sex],
        "pediatric": np.asarray([1.0 if str(row.get("Pediatric", "")).lower() == "yes" else 0.0], dtype=np.float32),
        "tb_status": np.asarray([1.0 if tb else 0.0], dtype=np.float32),
        "projection": np.eye(len(_PROJECTIONS), dtype=np.float32)[_category(row.get("Projection", ""), _PROJECTIONS, "PA")],
        "view_position": np.eye(len(_VIEWS), dtype=np.float32)[_category(row.get("ViewPosition_DICOM", ""), _VIEWS, "OTHER")],
        "study_year": normalize(np.asarray([year], dtype=np.float32), x_min=STUDY_YEAR_MIN, x_max=STUDY_YEAR_MAX)[0],
    }


def _split_patients(patients: Sequence[str], seed: int) -> dict[str, list[str]]:
    shuffled = sorted(patients)
    random.Random(seed).shuffle(shuffled)
    n_train, n_valid = int(len(shuffled) * 0.8), int(len(shuffled) * 0.1)
    return {
        "train": shuffled[:n_train],
        "valid": shuffled[n_train:n_train + n_valid],
        "test": shuffled[n_train + n_valid:],
    }


def _patient_digest(patient_ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(patient_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def padchest_split_summary(metadata: str, seed: int = 7) -> dict[str, Any]:
    """Return patient-level split counts and privacy-preserving fingerprints.

    The PadChest provider splits by shuffled patient IDs, then keeps rows that
    have an ``ImageID``.  This summary is the durable, non-PHI fingerprint used
    by post-training validation reports to prove downstream stages are aligned
    to the same split algorithm and seed without checking patient identifiers
    into source control.
    """
    rows = _read_rows(metadata)
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row.get("PatientID", row.get("ImageID", "")), []).append(row)
    split_patients = _split_patients(tuple(groups), seed)
    split_counts = {}
    for split, patients in split_patients.items():
        row_count = sum(1 for patient in patients for row in groups[patient] if row.get("ImageID"))
        split_counts[split] = {
            "patients": len(patients),
            "rows_with_image_id": row_count,
            "patient_sha256": _patient_digest(patients),
        }
    return {
        "seed": seed,
        "total_patients": len(groups),
        "total_rows": len(rows),
        "total_rows_with_image_id": sum(1 for row in rows if row.get("ImageID")),
        "splits": split_counts,
    }


class PadChestDataset:
    def __init__(self, root: str, metadata: str, image_prefix: str, split: str, input_res: int, tb_label_mode: str, seed: int = 7):
        rows = _read_rows(metadata)
        groups: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            groups.setdefault(row.get("PatientID", row.get("ImageID", "")), []).append(row)
        selected = _split_patients(tuple(groups), seed)[split]
        self.rows = [row for patient in selected for row in groups[patient] if row.get("ImageID")]
        self.root, self.image_prefix, self.input_res, self.tb_label_mode = root, image_prefix, input_res, tb_label_mode
        self._metadata = [_derive(row, tb_label_mode) for row in self.rows]
        self.samples = {
            spec.name: (
                np.stack([np.asarray(metadata[spec.name]) for metadata in self._metadata]).astype(np.float32)
                if self._metadata
                else np.empty((0, spec.encoded_dim), dtype=np.float32)
            )
            for spec in PAD_CHEST_SCHEMA.variables
        }

    def __len__(self):
        return len(self.rows)

    def _image_path(self, row: Mapping[str, str]) -> str:
        image_id = row["ImageID"]
        prefix = self.image_prefix.rstrip("/")
        if prefix:
            return f"{prefix}/{image_id}"
        return os.path.join(self.root, image_id)

    def _get_image(self, index: int) -> np.ndarray:
        with _open_binary(self._image_path(self.rows[index])) as handle:
            image = Image.open(handle).convert("L").resize((self.input_res, self.input_res), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32)[None] / 255.0

    def make_batch(self, indices: Sequence[int], **_: Any) -> dict[str, np.ndarray]:
        batch_indices = np.asarray(indices, dtype=np.int64)
        images = np.stack([self._get_image(int(index)) for index in batch_indices])
        variables = {name: np.asarray(values[batch_indices], dtype=np.float32) for name, values in self.samples.items()}
        return {"x": images, **variables}


class PadChestProvider:
    schema = PAD_CHEST_SCHEMA

    def __init__(self, root: str, input_res: int = 128, pad: int = 0, context_norm: str = "[-1,1]", metadata: str = "", image_prefix: str = "", tb_label_mode: str = "tb_or_sequelae", seed: int = 7):
        self.root, self.input_res, self.pad, self.context_norm = root, input_res, pad, context_norm
        self.metadata = metadata or f"{root.rstrip('/')}/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
        self.image_prefix = image_prefix or f"{root.rstrip('/')}/images-224"
        self.tb_label_mode, self.seed = tb_label_mode, seed

    @property
    def spec(self) -> DatasetSpec:
        return DatasetSpec("padchest", self.root, ImageSpec(1, self.input_res, self.input_res), {"train": "train", "valid": "valid", "test": "test"}, self.metadata)

    def load_split(self, split: str) -> PadChestDataset:
        return PadChestDataset(self.root, self.metadata, self.image_prefix, split, self.input_res, self.tb_label_mode, self.seed)

    def make_batch(self, split: str, indices: Sequence[int], *, rng=None, training: bool = False) -> Batch:
        raw = self.load_split(split).make_batch(indices, rng=rng, training=training)
        return Batch(raw.pop("x"), raw)

    def fingerprint(self) -> str:
        value = f"{self.root}|{self.metadata}|{self.image_prefix}|{self.input_res}|{self.tb_label_mode}|{self.schema.version}"
        return hashlib.sha256(value.encode()).hexdigest()[:16]


def padchest(settings) -> dict[str, PadChestDataset]:
    provider = PadChestProvider(settings.data_dir, settings.input_res, settings.pad, settings.context_norm, getattr(settings, "metadata", ""), getattr(settings, "image_prefix", ""), getattr(settings, "tb_label_mode", "tb_or_sequelae"), settings.seed)
    return {split: provider.load_split(split) for split in ("train", "valid", "test")}
