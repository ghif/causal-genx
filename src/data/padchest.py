"""PadChest PNG/CSV provider with patient-safe metadata conditioning."""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

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

# The PadChest SCM and predictor use the 17-dimensional causal encoding above.
# The authoritative HVAE artifact was trained with five additional, reserved
# image-context channels.  They are non-causal and were zero for every image
# batch; keeping them explicit prevents those channels from being confused with
# new SCM variables or intervention targets.
PAD_CHEST_SOURCE_CONTEXT_DIM = PAD_CHEST_SCHEMA.encoded_dim
PAD_CHEST_HVAE_CONTEXT_EXTENSION_DIM = 5
PAD_CHEST_HVAE_CONTEXT_DIM = PAD_CHEST_SOURCE_CONTEXT_DIM + PAD_CHEST_HVAE_CONTEXT_EXTENSION_DIM


def adapt_padchest_hvae_context(pa: np.ndarray, *, target_dim: int) -> np.ndarray:
    """Adapt the causal PadChest vector to the trained HVAE context contract.

    Only the known 17 -> 22 PadChest artifact contract is supported.  The
    extension is deliberately explicit and fixed at zero because those five
    channels are reserved image-context slots, not SCM variables.  Unknown
    dimensions fail instead of being silently padded or truncated.
    """
    array = np.asarray(pa)
    if array.ndim not in (2, 4):
        raise ValueError(f"PadChest context must be [N,C] or [N,H,W,C], got shape {array.shape}")
    source_dim = int(array.shape[-1])
    if source_dim != PAD_CHEST_SOURCE_CONTEXT_DIM:
        raise ValueError(
            "PadChest causal context must have dimension "
            f"{PAD_CHEST_SOURCE_CONTEXT_DIM}, got {source_dim}"
        )
    if int(target_dim) == PAD_CHEST_SOURCE_CONTEXT_DIM:
        return array
    if int(target_dim) != PAD_CHEST_HVAE_CONTEXT_DIM:
        raise ValueError(
            "Unsupported PadChest HVAE context dimension: "
            f"expected {PAD_CHEST_SOURCE_CONTEXT_DIM} or {PAD_CHEST_HVAE_CONTEXT_DIM}, got {target_dim}"
        )
    extension_shape = array.shape[:-1] + (PAD_CHEST_HVAE_CONTEXT_EXTENSION_DIM,)
    extension = np.zeros(extension_shape, dtype=array.dtype)
    return np.concatenate((array, extension), axis=-1)


def _configured_exclusions(excluded_sources: Sequence[Mapping[str, str] | str] | None) -> dict[str, str]:
    """Return an exact source URI/path -> reason map for PadChest-only exclusions."""
    exclusions: dict[str, str] = {}
    for entry in excluded_sources or []:
        if isinstance(entry, str):
            source, reason = entry, "configured PadChest source exclusion"
        else:
            source = str(entry.get("source", ""))
            reason = str(entry.get("reason", "configured PadChest source exclusion"))
        if not source:
            raise ValueError("PadChest excluded_sources entries must include a non-empty source")
        previous = exclusions.setdefault(source, reason)
        if previous != reason:
            raise ValueError(f"PadChest source exclusion {source!r} was configured with multiple reasons")
    return exclusions


@contextmanager
def _open_binary(path: str) -> Iterator[BinaryIO]:
    if not path.startswith("gs://"):
        with open(path, "rb") as handle:
            yield handle
        return
    if shutil.which("gsutil"):
        # gcsfs can fail under user ADC on requester-pays/org-policy buckets while
        # the project-local gsutil installation succeeds. Prefer gsutil so
        # training and staging reuse the same authenticated tooling researchers
        # use at the shell.
        completed = subprocess.run(
            ["gsutil", "cat", path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        yield io.BytesIO(completed.stdout)
        return
    import fsspec
    with fsspec.open(path, mode="rb").open() as handle:
        yield handle


def _default_stage_manifest(stage_dir: str | os.PathLike[str]) -> str:
    return str(Path(stage_dir) / "padchest-stage-manifest.jsonl")


def _safe_stage_relative(source: str, image_id: str) -> str:
    suffix = Path(image_id).suffix or Path(source).suffix or ".png"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return str(Path("images") / digest[:2] / f"{digest}{suffix}")


def _bulk_stage_relative(source: str, image_id: str) -> str:
    """Deterministic basename path used by gcloud's multi-object directory copy."""
    name = PurePosixPath(str(image_id)).name or PurePosixPath(source.split("gs://", 1)[-1]).name
    if not name or name in {".", ".."}:
        name = hashlib.sha256(source.encode("utf-8")).hexdigest() + ".png"
    return str(Path("gcloud-bulk") / name)


def _require_gcloud_storage() -> str:
    gcloud = shutil.which("gcloud")
    if not gcloud:
        raise RuntimeError(
            "PadChest GCS bulk staging requires the Google Cloud CLI (`gcloud`) with `gcloud storage cp`; "
            "install/authenticate gcloud and rerun instead of falling back to per-object downloads."
        )
    return gcloud


def _validate_bulk_destinations(records: Sequence[Mapping[str, Any]]) -> str:
    seen_paths: dict[str, str] = {}
    seen_images: dict[str, str] = {}
    for record in records:
        image_id = str(record["image_id"])
        relative_path = str(record["relative_path"])
        source = str(record["source"])
        previous_path_source = seen_paths.setdefault(relative_path, source)
        if previous_path_source != source:
            return (
                "gcloud bulk staging would map multiple sources to "
                f"{relative_path!r}; choose a non-colliding staging layout before executing"
            )
        previous_image_source = seen_images.setdefault(image_id, source)
        if previous_image_source != source:
            return (
                "PadChest staging would map ImageID "
                f"{image_id!r} to multiple sources; refusing ambiguous manifest"
            )
    return ""


def _run_gcloud_bulk_copy(records: Sequence[dict[str, Any]], stage_root: Path, workers: int) -> tuple[list[dict[str, Any]], int, int]:
    """Copy GCS records with one parallel gcloud invocation and verify deterministic outputs."""
    gcloud = _require_gcloud_storage()
    bulk_root = stage_root / "gcloud-bulk"
    bulk_root.mkdir(parents=True, exist_ok=True)
    to_copy: list[dict[str, Any]] = []
    copied_records: list[dict[str, Any]] = []
    reused = 0
    for record in records:
        destination = stage_root / str(record["relative_path"])
        if destination.is_file():
            record["bytes"] = destination.stat().st_size
            reused += 1
        else:
            to_copy.append(record)
        copied_records.append(record)

    if to_copy:
        shard_count = max(1, min(8, workers, len(to_copy)))
        thread_count = max(1, workers // shard_count)
        command = [
            gcloud,
            "--quiet",
            "storage",
            "cp",
            "--read-paths-from-stdin",
            "--continue-on-error",
            str(bulk_root),
        ]

        def run_shard(shard: Sequence[dict[str, Any]]) -> subprocess.CompletedProcess[str]:
            env = os.environ.copy()
            env["CLOUDSDK_STORAGE_PARALLEL_THREAD_COUNT"] = str(thread_count)
            return subprocess.run(
                command,
                input="\n".join(str(record["source"]) for record in shard) + "\n",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                check=False,
            )

        shards = [to_copy[index::shard_count] for index in range(shard_count)]
        with ThreadPoolExecutor(max_workers=shard_count) as executor:
            completed_shards = list(executor.map(run_shard, shards))
        failures = [completed for completed in completed_shards if completed.returncode != 0]
        if failures:
            failure = failures[0]
            stderr = (failure.stderr or failure.stdout or "").strip().splitlines()[-8:]
            raise RuntimeError(
                "gcloud storage cp failed during PadChest bulk staging; source objects may be missing/inaccessible "
                "or Google Cloud CLI authentication may need configuration. "
                f"command={' '.join(command)!r} exit={failure.returncode} output={' | '.join(stderr)}"
            )

    missing: list[str] = []
    for record in to_copy:
        destination = stage_root / str(record["relative_path"])
        if not destination.is_file():
            missing.append(f"{record['source']} -> {destination}")
        else:
            record["bytes"] = destination.stat().st_size
    if missing:
        preview = "; ".join(missing[:5])
        raise RuntimeError(
            "gcloud storage cp completed but PadChest bulk staging could not verify all expected files: "
            f"missing={len(missing)} examples={preview}"
        )
    return copied_records, len(to_copy), reused


import functools

@functools.lru_cache(maxsize=8)
def _load_manifest_exclusions(manifest: str) -> dict[str, str]:
    if not manifest or not Path(manifest).is_file():
        return {}
    try:
        mtime = Path(manifest).stat().st_mtime
    except OSError:
        return {}
    return _load_manifest_exclusions_mtime_cached(manifest, mtime)


@functools.lru_cache(maxsize=8)
def _load_manifest_exclusions_mtime_cached(manifest: str, mtime: float) -> dict[str, str]:
    if not manifest or not Path(manifest).is_file():
        return {}
    exclusions = {}
    with Path(manifest).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("kind") == "excluded":
                source = str(record.get("source", ""))
                reason = str(record.get("reason", "manifest-recorded exclusion"))
                if source:
                    exclusions[source] = reason
    return exclusions


def _load_stage_manifest_cached(manifest: str, stage_dir: str) -> dict[str, tuple[str, str]]:
    if not manifest or not Path(manifest).is_file():
        return {}
    try:
        mtime = Path(manifest).stat().st_mtime
    except OSError:
        return {}
    return _load_stage_manifest_mtime_cached(manifest, stage_dir, mtime)


@functools.lru_cache(maxsize=8)
def _load_stage_manifest_mtime_cached(manifest: str, stage_dir: str, mtime: float) -> dict[str, tuple[str, str]]:
    """Load a JSONL ImageID -> (source, local file) map without requiring credentials."""
    if not manifest or not Path(manifest).is_file():
        return {}
    root = Path(stage_dir)
    mapping: dict[str, tuple[str, str]] = {}
    with Path(manifest).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("kind") not in {None, "item"}:
                continue
            image_id = str(record.get("image_id", ""))
            source = str(record.get("source", ""))
            local_path = record.get("local_path") or record.get("relative_path")
            if not image_id or not local_path:
                continue
            path = Path(str(local_path))
            if not path.is_absolute():
                path = root / path
            mapping[image_id] = (source, str(path))
    return mapping


def _load_stage_manifest(manifest: str, stage_dir: str, expected_sources: Mapping[str, str] | None = None) -> dict[str, str]:
    """Load a JSONL ImageID -> local file map with optional source validation."""
    cached = _load_stage_manifest_cached(manifest, stage_dir)
    if expected_sources is None:
        return {k: v[1] for k, v in cached.items()}
    mapping: dict[str, str] = {}
    for image_id, (source, local_path) in cached.items():
        if source and source != expected_sources.get(image_id):
            continue
        mapping[image_id] = local_path
    return mapping


def _source_size(path: str) -> int | None:
    try:
        if path.startswith("gs://"):
            if shutil.which("gsutil"):
                completed = subprocess.run(
                    ["gsutil", "du", path],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                return int(completed.stdout.split()[0])
            import fsspec
            fs, fs_path = fsspec.core.url_to_fs(path)
            return int(fs.size(fs_path))
        return int(Path(path).stat().st_size)
    except Exception:
        return None


def _remote_tree_size(prefix: str) -> int | None:
    try:
        if not prefix.startswith("gs://"):
            return None
        if shutil.which("gsutil"):
            completed = subprocess.run(
                ["gsutil", "du", "-s", prefix.rstrip("/")],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return int(completed.stdout.split()[0])
    except Exception:
        return None
    return None


def _as_list(value: str) -> list[str]:
    try:
        parsed = ast.literal_eval(value)
        return [str(item).strip().lower() for item in parsed] if isinstance(parsed, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return [item.strip().lower() for item in value.strip("[]").split(",") if item.strip()]


def _category(value: str, choices: tuple[str, ...], fallback: str) -> int:
    value = str(value or "").strip().upper()
    return choices.index(value) if value in choices else choices.index(fallback)


@functools.lru_cache(maxsize=4)
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
    def __init__(
        self,
        root: str,
        metadata: str,
        image_prefix: str,
        split: str,
        input_res: int,
        tb_label_mode: str,
        seed: int = 7,
        *,
        stage_dir: str = "",
        stage_manifest: str = "",
        stage_mode: str = "auto",
        excluded_sources: Sequence[Mapping[str, str] | str] | None = None,
        concat_pa: bool = True,
    ):
        rows = _read_rows(metadata)
        groups: dict[str, list[tuple[int, dict[str, str]]]] = {}
        for row_index, row in enumerate(rows):
            groups.setdefault(row.get("PatientID", row.get("ImageID", "")), []).append((row_index, row))
        selected = _split_patients(tuple(groups), seed)[split]
        selected_rows = [(row_index, row) for patient in selected for row_index, row in groups[patient] if row.get("ImageID")]
        self.root, self.image_prefix, self.input_res, self.tb_label_mode = root, image_prefix, input_res, tb_label_mode
        self.stage_dir = stage_dir
        self.stage_manifest = stage_manifest or (_default_stage_manifest(stage_dir) if stage_dir else "")
        self.stage_mode = stage_mode
        self.concat_pa = concat_pa
        exclusions = _configured_exclusions(excluded_sources)
        if self.stage_manifest and self.stage_mode != "off":
            manifest_exclusions = _load_manifest_exclusions(self.stage_manifest)
            for src, reason in manifest_exclusions.items():
                exclusions.setdefault(src, reason)
        self.excluded_records: list[dict[str, Any]] = []
        self.rows: list[dict[str, str]] = []
        for row_index, row in selected_rows:
            source = self._source_image_path(row)
            if source in exclusions:
                self.excluded_records.append({
                    "kind": "excluded",
                    "image_id": row["ImageID"],
                    "source": source,
                    "reason": exclusions[source],
                    "split": split,
                    "row_index": row_index,
                })
                continue
            self.rows.append(row)
        expected_sources = {row["ImageID"]: self._source_image_path(row) for row in self.rows}
        self._staged_images = _load_stage_manifest(self.stage_manifest, stage_dir, expected_sources) if stage_dir and stage_mode != "off" else {}
        if stage_mode == "require":
            missing = [row["ImageID"] for row in self.rows if row["ImageID"] not in self._staged_images]
            if missing:
                raise RuntimeError(
                    "input_stage_mode=require but PadChest local staging is incomplete: "
                    f"split={split} missing={len(missing)}/{len(self.rows)} manifest={self.stage_manifest!r} "
                    "Run `python scripts/stage_padchest_inputs.py --config <config> --execute` "
                    "or relax workflow.input_stage_mode."
                )
            if self._staged_images and self.rows:
                first_staged = self._staged_images.get(self.rows[0]["ImageID"])
                if first_staged and not Path(first_staged).is_file():
                    raise RuntimeError(
                        f"input_stage_mode=require but staged image path {first_staged!r} is not a valid file on disk."
                    )
        self._metadata = [_derive(row, tb_label_mode) for row in self.rows]
        self.samples = {
            spec.name: (
                np.stack([np.asarray(metadata[spec.name]) for metadata in self._metadata]).astype(np.float32)
                if self._metadata
                else np.empty((0, spec.encoded_dim), dtype=np.float32)
            )
            for spec in PAD_CHEST_SCHEMA.variables
        }
        self.min_max = {"age_at_study": (AGE_AT_STUDY_MIN, AGE_AT_STUDY_MAX), "study_year": (STUDY_YEAR_MIN, STUDY_YEAR_MAX)}
        self.cache_fingerprint = hashlib.sha256(
            f"padchest|{root}|{metadata}|{image_prefix}|{split}|{input_res}|{tb_label_mode}|{seed}|{json.dumps(exclusions, sort_keys=True)}".encode()
        ).hexdigest()[:16]

    def __len__(self):
        return len(self.rows)

    def _source_image_path(self, row: Mapping[str, str]) -> str:
        image_id = row["ImageID"]
        prefix = self.image_prefix.rstrip("/")
        if prefix:
            return f"{prefix}/{image_id}"
        return os.path.join(self.root, image_id)

    def _image_path(self, row: Mapping[str, str]) -> str:
        image_id = row["ImageID"]
        staged = self._staged_images.get(image_id)
        if staged and Path(staged).is_file():
            return staged
        if self.stage_mode == "require":
            raise FileNotFoundError(
                f"PadChest staged image is missing for ImageID={image_id!r}; manifest={self.stage_manifest!r}"
            )
        return self._source_image_path(row)

    def image_records(self) -> list[dict[str, str]]:
        return [{"image_id": row["ImageID"], "source": self._source_image_path(row)} for row in self.rows]

    def _get_image(self, index: int) -> np.ndarray:
        with _open_binary(self._image_path(self.rows[index])) as handle:
            raw = Image.open(handle)
            arr = np.array(raw)
            if arr.dtype == np.uint16 or getattr(raw, "mode", "").startswith("I"):
                u8 = (arr.astype(np.float32) / 256.0).clip(0, 255).astype(np.uint8)
                image = Image.fromarray(u8, mode="L").resize((self.input_res, self.input_res), Image.Resampling.BILINEAR)
            else:
                image = raw.convert("L").resize((self.input_res, self.input_res), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32)[None] / 255.0

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        batch = self.make_batch([int(index)])
        return {key: np.asarray(value[0]) for key, value in batch.items()}

    def make_batch(self, indices: Sequence[int], **_: Any) -> dict[str, np.ndarray]:
        batch_indices = np.asarray(indices, dtype=np.int64)
        images = np.stack([self._get_image(int(index)) for index in batch_indices])
        variables = {name: np.asarray(values[batch_indices], dtype=np.float32) for name, values in self.samples.items()}
        sample = {"x": images, **variables}
        if self.concat_pa:
            parts = []
            for spec in PAD_CHEST_SCHEMA.variables:
                v = variables[spec.name]
                if v.ndim == 1:
                    v = v[:, None]
                parts.append(v)
            sample["pa"] = np.concatenate(parts, axis=1).astype(np.float32)
        return sample


class PadChestProvider:
    schema = PAD_CHEST_SCHEMA

    def __init__(
        self,
        root: str,
        input_res: int = 128,
        pad: int = 0,
        context_norm: str = "[-1,1]",
        metadata: str = "",
        image_prefix: str = "",
        tb_label_mode: str = "tb_or_sequelae",
        seed: int = 7,
        *,
        stage_dir: str = "",
        stage_manifest: str = "",
        stage_mode: str = "auto",
        excluded_sources: Sequence[Mapping[str, str] | str] | None = None,
        concat_pa: bool = True,
    ):
        self.root, self.input_res, self.pad, self.context_norm = root, input_res, pad, context_norm
        self.metadata = metadata or f"{root.rstrip('/')}/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
        self.image_prefix = image_prefix or f"{root.rstrip('/')}/images-224"
        self.tb_label_mode, self.seed = tb_label_mode, seed
        self.stage_dir, self.stage_manifest, self.stage_mode = stage_dir, stage_manifest, stage_mode
        self.excluded_sources = excluded_sources or []
        self.concat_pa = concat_pa

    @property
    def spec(self) -> DatasetSpec:
        return DatasetSpec("padchest", self.root, ImageSpec(1, self.input_res, self.input_res), {"train": "train", "valid": "valid", "test": "test"}, self.metadata)

    def load_split(self, split: str) -> PadChestDataset:
        return PadChestDataset(
            self.root,
            self.metadata,
            self.image_prefix,
            split,
            self.input_res,
            self.tb_label_mode,
            self.seed,
            stage_dir=self.stage_dir,
            stage_manifest=self.stage_manifest,
            stage_mode=self.stage_mode,
            excluded_sources=self.excluded_sources,
            concat_pa=self.concat_pa,
        )

    def make_batch(self, split: str, indices: Sequence[int], *, rng=None, training: bool = False) -> Batch:
        raw = self.load_split(split).make_batch(indices, rng=rng, training=training)
        return Batch(raw.pop("x"), raw)

    def fingerprint(self) -> str:
        exclusions = _configured_exclusions(self.excluded_sources)
        value = f"{self.root}|{self.metadata}|{self.image_prefix}|{self.input_res}|{self.tb_label_mode}|{self.schema.version}|{json.dumps(exclusions, sort_keys=True)}"
        return hashlib.sha256(value.encode()).hexdigest()[:16]


def stage_padchest_images(
    settings: Any,
    *,
    splits: Sequence[str] = ("train", "valid", "test"),
    execute: bool = False,
    limit: int = 0,
) -> dict[str, Any]:
    """Plan or populate a bounded local PadChest image stage.

    The manifest is append-only and existing staged files are reused.  Budget
    checks happen before copying so an accidental full-dataset stage requires an
    explicit item/byte policy in the workflow config.
    """
    stage_dir = str(getattr(settings, "input_stage_dir", "") or "")
    if not stage_dir:
        raise ValueError("workflow.input_stage_dir is required for PadChest staging")
    stage_manifest = str(getattr(settings, "input_stage_manifest", "") or _default_stage_manifest(stage_dir))
    max_items = int(getattr(settings, "input_stage_max_items", 0) or 0)
    max_bytes = int(getattr(settings, "input_stage_max_bytes", 0) or 0)
    size_sample_items = max(0, int(getattr(settings, "input_stage_size_sample_items", 256) or 0))
    workers = max(1, int(getattr(settings, "input_stage_workers", 8) or 1))

    provider = PadChestProvider(
        settings.data_dir,
        settings.input_res,
        settings.pad,
        getattr(settings, "context_norm", "[-1,1]"),
        getattr(settings, "metadata", ""),
        getattr(settings, "image_prefix", ""),
        getattr(settings, "tb_label_mode", "tb_or_sequelae"),
        settings.seed,
        stage_mode="off",
        excluded_sources=getattr(settings, "excluded_sources", None) or getattr(settings, "input_stage_exclusions", None) or [],
    )
    unique: dict[str, dict[str, Any]] = {}
    split_counts: dict[str, int] = {}
    excluded_split_counts: dict[str, int] = {}
    excluded_records: list[dict[str, Any]] = []
    for split in splits:
        dataset = provider.load_split(split)
        split_counts[split] = len(dataset)
        excluded_split_counts[split] = len(dataset.excluded_records)
        excluded_records.extend(dataset.excluded_records)
        for record in dataset.image_records():
            source = record["source"]
            if source not in unique:
                unique[source] = {
                    "kind": "item",
                    "image_id": record["image_id"],
                    "source": source,
                    "split": split,
                }
    records = list(unique.values())
    if limit > 0:
        records = records[:limit]

    gcs_sources = [record for record in records if str(record["source"]).startswith("gs://")]
    all_gcs_sources = bool(records) and len(gcs_sources) == len(records)
    transfer_method = "gcloud-storage-cp" if all_gcs_sources else "python-copy"
    for record in records:
        source = str(record["source"])
        record["relative_path"] = (
            _bulk_stage_relative(source, str(record["image_id"]))
            if all_gcs_sources
            else _safe_stage_relative(source, str(record["image_id"]))
        )

    existing_map = _load_stage_manifest(stage_manifest, stage_dir)
    if execute and not all_gcs_sources:
        missing_sources = []
        for record in records:
            source = str(record["source"])
            staged = existing_map.get(str(record["image_id"]))
            destination = Path(stage_dir) / str(record["relative_path"])
            if source.startswith("gs://") or Path(source).is_file() or destination.is_file() or (staged and Path(staged).is_file()):
                continue
            missing_sources.append(source)
        if missing_sources:
            preview = "; ".join(missing_sources[:5])
            raise RuntimeError(
                "Unexpected missing PadChest source object: "
                f"missing={len(missing_sources)} examples={preview}"
            )
    total_bytes = 0
    unknown_sizes = 0
    size_basis = "per_item"
    sample_count = len(records)
    if len(records) > 1024 and size_sample_items > 0:
        sample_count = min(len(records), size_sample_items)
        size_basis = f"sampled_{sample_count}_items"
    prefix = provider.image_prefix.rstrip("/")
    if len(records) > 1024 and size_sample_items <= 0 and prefix.startswith("gs://"):
        prefix_bytes = _remote_tree_size(prefix)
        if prefix_bytes is not None:
            sample_count = 0
            total_bytes = int(prefix_bytes)
            size_basis = "source_prefix_upper_bound"
    records_to_size = records[:sample_count]

    def record_size(record: dict[str, Any]) -> int | None:
        local_path = Path(stage_dir) / record["relative_path"]
        if local_path.is_file():
            return local_path.stat().st_size
        staged = existing_map.get(record["image_id"])
        if staged and Path(staged).is_file():
            return Path(staged).stat().st_size
        return _source_size(record["source"])

    if records_to_size:
        with ThreadPoolExecutor(max_workers=min(workers, len(records_to_size))) as executor:
            sizes = list(executor.map(record_size, records_to_size))
        known_sizes = [int(size) for size in sizes if size is not None]
        unknown_sizes = len(sizes) - len(known_sizes)
        for record, size in zip(records_to_size, sizes):
            record["bytes"] = size
        for record in records[sample_count:]:
            record["bytes"] = None
        if size_basis.startswith("sampled_") and known_sizes:
            total_bytes = int((sum(known_sizes) / len(known_sizes)) * len(records))
            unknown_sizes = 0
        else:
            total_bytes = sum(known_sizes)

    refusal = ""
    if max_items <= 0:
        refusal = "workflow.input_stage_max_items must be set to a positive bounded value"
    elif max_bytes <= 0:
        refusal = "workflow.input_stage_max_bytes must be set to a positive bounded value"
    elif gcs_sources and not all_gcs_sources:
        refusal = "PadChest staging refuses mixed local and gs:// image sources"
    elif all_gcs_sources:
        refusal = _validate_bulk_destinations(records)
    if not refusal and max_items > 0 and len(records) > max_items:
        refusal = f"required_items={len(records)} exceeds input_stage_max_items={max_items}"
    elif not refusal and max_bytes > 0 and unknown_sizes:
        refusal = f"cannot enforce input_stage_max_bytes={max_bytes} with {unknown_sizes} unknown object sizes"
    elif not refusal and max_bytes > 0 and total_bytes > max_bytes:
        refusal = f"estimated_bytes={total_bytes} exceeds input_stage_max_bytes={max_bytes}"

    transfer_prerequisite = ""
    if all_gcs_sources:
        transfer_prerequisite = "Google Cloud CLI with authenticated `gcloud storage cp --read-paths-from-stdin`"

    summary = {
        "stage_dir": stage_dir,
        "stage_manifest": stage_manifest,
        "splits": list(splits),
        "split_rows": split_counts,
        "excluded_split_rows": excluded_split_counts,
        "required_items": len(records),
        "excluded_items": len(excluded_records),
        "excluded": excluded_records,
        "estimated_bytes": total_bytes,
        "unknown_sizes": unknown_sizes,
        "size_basis": size_basis,
        "max_items": max_items,
        "max_bytes": max_bytes,
        "size_sample_items": size_sample_items,
        "transfer_method": transfer_method,
        "transfer_prerequisite": transfer_prerequisite,
        "execute": execute,
        "refusal": refusal,
        "copied": 0,
        "reused": 0,
        "elapsed_seconds": 0.0,
        "items_per_second": 0.0,
        "bytes_per_second": 0.0,
    }
    if refusal:
        if execute:
            raise RuntimeError(f"PadChest staging refused: {refusal}")
        return summary
    if not execute:
        return summary

    stage_root = Path(stage_dir)
    stage_root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(stage_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "kind": "header",
                "version": 1,
                "created_at": int(time.time()),
                "metadata": provider.metadata,
                "image_prefix": provider.image_prefix,
                "splits": list(splits),
                "split_rows": split_counts,
                "excluded_split_rows": excluded_split_counts,
                "excluded_items": len(excluded_records),
                "required_items": len(records),
                "max_items": max_items,
                "max_bytes": max_bytes,
            }, sort_keys=True) + "\n")

    def copy_one(record: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        destination = stage_root / record["relative_path"]
        if destination.is_file():
            record["bytes"] = destination.stat().st_size
            return record, False
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=destination.name, suffix=".tmp", dir=str(destination.parent))
        os.close(fd)
        try:
            with _open_binary(record["source"]) as source, open(tmp, "wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            os.replace(tmp, destination)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        record["bytes"] = destination.stat().st_size
        return record, True

    started_at = time.monotonic()
    if all_gcs_sources:
        copied_records, copied, reused = _run_gcloud_bulk_copy(records, stage_root, workers)
        summary["copied"] = copied
        summary["reused"] = reused
    else:
        copied_records = []
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(records)))) as executor:
            for record, copied in executor.map(copy_one, records):
                summary["copied" if copied else "reused"] += 1
                copied_records.append(record)
    elapsed_seconds = max(time.monotonic() - started_at, 0.0)
    with manifest_path.open("a", encoding="utf-8") as handle:
        for record in excluded_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        for record in copied_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    staged_bytes = sum(int(record.get("bytes") or 0) for record in copied_records)
    summary["estimated_bytes"] = staged_bytes
    summary["elapsed_seconds"] = elapsed_seconds
    summary["items_per_second"] = (summary["copied"] / elapsed_seconds) if elapsed_seconds > 0 else 0.0
    summary["bytes_per_second"] = (staged_bytes / elapsed_seconds) if elapsed_seconds > 0 else 0.0
    return summary


def _stage_settings(settings: Any) -> dict[str, Any]:
    return {
        "stage_dir": str(getattr(settings, "input_stage_dir", "") or ""),
        "stage_manifest": str(getattr(settings, "input_stage_manifest", "") or ""),
        "stage_mode": str(getattr(settings, "input_stage_mode", "auto") or "auto"),
        "excluded_sources": getattr(settings, "excluded_sources", None) or getattr(settings, "input_stage_exclusions", None) or [],
    }


def padchest(settings) -> dict[str, PadChestDataset]:
    provider = PadChestProvider(
        settings.data_dir,
        settings.input_res,
        settings.pad,
        settings.context_norm,
        getattr(settings, "metadata", ""),
        getattr(settings, "image_prefix", ""),
        getattr(settings, "tb_label_mode", "tb_or_sequelae"),
        settings.seed,
        concat_pa=getattr(settings, "concat_pa", True),
        **_stage_settings(settings),
    )
    return {split: provider.load_split(split) for split in ("train", "valid", "test")}
