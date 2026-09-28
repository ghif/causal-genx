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
    """Declarative encoding contract for one causal variable.

    ``encoded_dim`` is the width consumed by the image model.  Categorical
    values are represented by a one-hot vector and binary values by one
    scalar; this deliberately keeps the representation independent of the
    model that consumes it.  ``categories`` is optional for legacy schemas,
    but makes arbitrary YAML schemas self-describing.
    """

    name: str
    kind: VariableKind
    encoded_dim: int = 1
    normalization: str | None = None
    observed: bool = True
    intervenable: bool = True
    categories: tuple[str, ...] = ()
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.name or self.encoded_dim < 1:
            raise ValueError("Variables require a name and positive encoded_dim.")
        kind = VariableKind(self.kind)
        object.__setattr__(self, "kind", kind)
        categories = tuple(str(category) for category in self.categories)
        object.__setattr__(self, "categories", categories)
        if kind is VariableKind.BINARY and self.encoded_dim != 1:
            raise ValueError("Binary variables must have encoded_dim=1.")
        if kind is VariableKind.CONTINUOUS and self.encoded_dim != 1:
            raise ValueError("Continuous variables must have encoded_dim=1.")
        if kind is VariableKind.CATEGORICAL and self.encoded_dim < 2:
            raise ValueError("Categorical variables require at least two encoded categories.")
        if categories and len(categories) != self.encoded_dim:
            raise ValueError(
                f"{self.name}: categories must have encoded_dim={self.encoded_dim} entries."
            )
        if len(categories) != len(set(categories)):
            raise ValueError(f"{self.name}: categories must be unique.")


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
        if any(parent == child for parent, child in self.edges):
            raise ValueError("Causal edges cannot contain self-loops.")
        if len(self.edges) != len(set(self.edges)):
            raise ValueError("Causal edges must be unique.")
        # Validate that the graph is a DAG while retaining the declared
        # variable order for encoded parent vectors.
        pending = {name: 0 for name in names}
        children: dict[str, list[str]] = {name: [] for name in names}
        for parent, child in self.edges:
            pending[child] += 1
            children[parent].append(child)
        ready = [name for name in names if pending[name] == 0]
        visited = 0
        while ready:
            name = ready.pop()
            visited += 1
            for child in children[name]:
                pending[child] -= 1
                if pending[child] == 0:
                    ready.append(child)
        if visited != len(names):
            raise ValueError("Causal edges must form a directed acyclic graph.")
    @property
    def variable_names(self) -> tuple[str, ...]:
        return tuple(variable.name for variable in self.variables)

    @property
    def encoded_dim(self) -> int:
        return sum(variable.encoded_dim for variable in self.variables)

    @property
    def topological_order(self) -> tuple[str, ...]:
        """Return a deterministic topological order for factorized SCMs."""
        names = self.variable_names
        parents = {name: set() for name in names}
        children = {name: [] for name in names}
        for parent, child in self.edges:
            parents[child].add(parent)
            children[parent].append(child)
        ready = [name for name in names if not parents[name]]
        order: list[str] = []
        while ready:
            name = ready.pop(0)
            order.append(name)
            for child in children[name]:
                parents[child].remove(name)
                if not parents[child]:
                    ready.append(child)
        return tuple(order)

    def parents_of(self, name: str) -> tuple[str, ...]:
        if name not in self.variable_names:
            raise KeyError(name)
        return tuple(parent for parent, child in self.edges if child == name)

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

