"""Generic row-to-parent encodings for schema-defined metadata."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from contracts import CausalGraphSpec, VariableKind


class MetadataEncoder:
    """Encode mapping-like metadata rows according to a causal schema.

    ``mappings`` may map a variable name to a source column name or a callable
    accepting the row.  Callables are the small hook needed for derived fields
    such as PadChest age groups; categorical values use ``VariableSpec``'s
    declared category order and never silently collapse unknown values.
    """

    def __init__(
        self,
        schema: CausalGraphSpec,
        mappings: Mapping[str, str | Callable[[Mapping[str, Any]], Any]] | None = None,
    ) -> None:
        self.schema = schema
        self.mappings = dict(mappings or {})

    def _raw(self, spec, row: Mapping[str, Any]) -> Any:
        source = self.mappings.get(spec.name, spec.source or spec.name)
        if callable(source):
            return source(row)
        if source not in row:
            raise KeyError(f"Missing metadata source {source!r} for variable {spec.name!r}")
        return row[source]

    def encode(self, row: Mapping[str, Any]) -> dict[str, np.ndarray]:
        encoded: dict[str, np.ndarray] = {}
        for spec in self.schema.variables:
            raw = self._raw(spec, row)
            if spec.kind is VariableKind.CONTINUOUS:
                value = np.asarray([float(raw)], dtype=np.float32)
            elif spec.kind is VariableKind.BINARY:
                if isinstance(raw, str):
                    value = 1.0 if raw.strip().lower() in {"1", "true", "yes", "y", "positive", "+"} else 0.0
                else:
                    value = float(raw)
                if value not in (0.0, 1.0):
                    raise ValueError(f"{spec.name}: binary metadata must be 0/1, got {raw!r}")
                value = np.asarray([value], dtype=np.float32)
            else:
                if not spec.categories:
                    raise ValueError(f"{spec.name}: categorical metadata requires declared categories")
                try:
                    index = spec.categories.index(str(raw))
                except ValueError as exc:
                    raise ValueError(f"{spec.name}: unknown category {raw!r}") from exc
                value = np.eye(spec.encoded_dim, dtype=np.float32)[index]
            encoded[spec.name] = value
        return encoded

    __call__ = encode

    def vector(self, row: Mapping[str, Any]) -> np.ndarray:
        encoded = self.encode(row)
        return np.concatenate([encoded[spec.name].reshape(-1) for spec in self.schema.variables])
