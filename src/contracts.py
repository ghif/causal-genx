"""Small shared contracts for datasets, causal variables, and model boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


class VariableKind(str, Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    BINARY = "binary"


@dataclass(frozen=True)
class VariableSpec:
    name: str
    kind: VariableKind
    encoded_dim: int = 1
    normalization: str | None = None
    observed: bool = True
    intervenable: bool = True

    def __post_init__(self) -> None:
        if not self.name or self.encoded_dim < 1:
            raise ValueError("Variables require a name and positive encoded_dim.")
        if self.kind is VariableKind.BINARY and self.encoded_dim != 1:
            raise ValueError("Binary variables must have encoded_dim=1.")


@dataclass(frozen=True)
class CausalGraphSpec:
    dataset_id: str
    variables: tuple[VariableSpec, ...]
    edges: tuple[tuple[str, str], ...] = ()
    version: str = "1"

    def __post_init__(self) -> None:
        names = self.variable_names
        if not self.dataset_id or not names or len(names) != len(set(names)):
            raise ValueError("Schemas require a dataset ID and uniquely named variables.")
        if any(parent not in names or child not in names for parent, child in self.edges):
            raise ValueError("Causal edges must reference schema variables.")

    @property
    def variable_names(self) -> tuple[str, ...]:
        return tuple(variable.name for variable in self.variables)

    @property
    def encoded_dim(self) -> int:
        return sum(variable.encoded_dim for variable in self.variables)

    def validate_intervention(self, values: Mapping[str, object]) -> None:
        known = {variable.name: variable for variable in self.variables}
        unknown = set(values).difference(known)
        forbidden = [name for name in values if name in known and not known[name].intervenable]
        if unknown or forbidden:
            raise ValueError(f"Invalid intervention; unknown={sorted(unknown)}, forbidden={forbidden}")


@dataclass(frozen=True)
class ImageSpec:
    channels: int
    height: int
    width: int


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    root: str
    image: ImageSpec
    splits: Mapping[str, str]
    metadata_source: str | None = None
    transform: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Batch:
    image: np.ndarray
    variables: Mapping[str, np.ndarray]
    sample_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.image.ndim != 4:
            raise ValueError("Batches use NCHW images before model preprocessing.")
        if any(np.asarray(value).shape[0] != self.image.shape[0] for value in self.variables.values()):
            raise ValueError("Variables must share the image batch dimension.")


@runtime_checkable
class DatasetProvider(Protocol):
    spec: DatasetSpec
    schema: CausalGraphSpec
    def load_split(self, split: str) -> Any: ...
    def make_batch(self, split: str, indices: Sequence[int], *, rng: np.random.Generator | None = None, training: bool = False) -> Batch: ...
    def fingerprint(self) -> str: ...


@runtime_checkable
class StructuralCausalModel(Protocol):
    def counterfactual(self, obs: Mapping[str, Any], intervention: Mapping[str, Any], rng: Any) -> Mapping[str, Any]: ...

