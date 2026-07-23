"""Portable run metadata and artifact compatibility checks."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ArtifactMetadata:
    artifact_type: str
    model_id: str
    dataset_id: str
    dataset_fingerprint: str
    causal_schema_version: str
    parent_encoder_version: str = "1"
    checkpoint_format_version: str = "1"
    config: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    parent_artifacts: Mapping[str, str] = field(default_factory=dict)


class LocalArtifactStore:
    def __init__(self, root: str = "checkpoints") -> None:
        self.root = Path(root)

    def run_dir(self, dataset_id: str, run_name: str) -> str:
        path = self.root / dataset_id / run_name
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def write_metadata(self, run_dir: str, metadata: ArtifactMetadata) -> str:
        target = Path(run_dir) / "artifact.json"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
            json.dump(asdict(metadata), handle, indent=2, sort_keys=True)
            handle.write("\n")
            staged = handle.name
        os.replace(staged, target)
        return str(target)

    def read_metadata(self, run_dir: str) -> ArtifactMetadata:
        with open(Path(run_dir) / "artifact.json", encoding="utf-8") as handle:
            return ArtifactMetadata(**json.load(handle))


def assert_compatible(*items: ArtifactMetadata, allow_override: bool = False) -> None:
    if len(items) < 2:
        return
    fields = ("dataset_id", "dataset_fingerprint", "causal_schema_version", "parent_encoder_version")
    mismatch = [field for field in fields if any(getattr(item, field) != getattr(items[0], field) for item in items[1:])]
    if mismatch and not allow_override:
        raise ValueError(f"Incompatible artifacts: {', '.join(mismatch)}")

