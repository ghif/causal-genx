"""Generic row-to-parent encodings for schema-defined metadata."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import ast
import re
from typing import Any

import numpy as np

from contracts import CausalGraphSpec, VariableKind


_LINEAR_NORMALIZATION = re.compile(
    r"^linear\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)->\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)$"
)


def encode_continuous_value(value: Any, normalization: str | None, *, name: str = "value") -> np.ndarray:
    """Apply a schema-declared linear source-to-model range transform."""
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name}: continuous value must be finite")
    if normalization:
        match = _LINEAR_NORMALIZATION.fullmatch(normalization.strip())
        if match:
            source_low, source_high, target_low, target_high = map(float, match.groups())
            if source_high <= source_low or target_high <= target_low:
                raise ValueError(f"{name}: normalization ranges must be increasing")
            if not source_low <= scalar <= source_high:
                raise ValueError(f"{name}: {scalar} outside normalization range [{source_low}, {source_high}]")
            scalar = target_low + (scalar - source_low) * (target_high - target_low) / (source_high - source_low)
    return np.asarray([scalar], dtype=np.float32)


def encode_binary_value(value: Any, *, name: str = "value") -> np.ndarray:
    """Encode common explicit binary values and reject ambiguous strings."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "positive", "+"}:
            scalar = 1.0
        elif normalized in {"0", "false", "no", "n", "negative", "-"}:
            scalar = 0.0
        else:
            raise ValueError(f"{name}: unknown binary value {value!r}")
    else:
        scalar = float(value)
    if not np.isfinite(scalar) or scalar not in (0.0, 1.0):
        raise ValueError(f"{name}: binary value must be finite 0/1, got {value!r}")
    return np.asarray([scalar], dtype=np.float32)


class MetadataEncoder:
    """Encode mapping-like metadata rows according to a causal schema.

    ``mappings`` may map a variable name to a source column or row callback.
    Declarative ``rules`` cover common transformations without adding a new
    dataset-specific encoder. Categorical values follow the schema's declared
    order and unknown categories are rejected.
    """

    def __init__(
        self,
        schema: CausalGraphSpec,
        mappings: Mapping[str, str | Callable[[Mapping[str, Any]], Any]] | None = None,
        *,
        rules: Mapping[str, Mapping[str, Any]] | None = None,
        label_mode: str = "any_tb",
    ) -> None:
        self.schema = schema
        self.mappings = dict(mappings or {})
        self.rules = {name: dict(rule) for name, rule in (rules or {}).items()}
        self.label_mode = str(label_mode)
        variables = {variable.name: variable for variable in schema.variables}
        unknown = set(self.rules).difference(variables)
        if unknown:
            raise ValueError(f"Metadata rules reference undeclared variables: {sorted(unknown)}")
        required = {
            "column": {"source"},
            "year_difference": {"date_source", "birth_source"},
            "category_map": {"source", "mapping"},
            "contains_any": {"source", "values"},
        }
        for name, rule in self.rules.items():
            operation = str(rule.get("operation", "column"))
            if operation not in required:
                raise ValueError(f"{name}: unsupported metadata rule operation {operation!r}")
            missing = required[operation].difference(rule)
            if missing:
                raise ValueError(f"{name}: metadata rule {operation!r} is missing {sorted(missing)}")
            if operation == "category_map":
                spec = variables[name]
                outputs = {str(value) for value in rule["mapping"].values()}
                if not spec.categories or not outputs.issubset(spec.categories):
                    raise ValueError(f"{name}: category map outputs must belong to the declared categories")

    @staticmethod
    def _labels(value: Any) -> set[str]:
        if isinstance(value, (list, tuple, set)):
            return {str(item).strip().lower() for item in value if str(item).strip()}
        text = str(value or "").strip()
        if not text:
            return set()
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return {str(item).strip().lower() for item in parsed if str(item).strip()}
            if isinstance(parsed, str):
                return {parsed.strip().lower()} if parsed.strip() else set()
        except (ValueError, SyntaxError):
            pass
        return {item.strip().lower() for item in text.split("|") if item.strip()}

    def _apply_rule(self, name: str, row: Mapping[str, Any]) -> Any:
        rule = self.rules[name]
        operation = str(rule.get("operation", "column"))
        if operation == "year_difference":
            date = str(row.get(str(rule["date_source"]), "")).strip()
            birth = str(row.get(str(rule["birth_source"]), "")).strip()
            if len(date) < 4 or not date[:4].isdigit() or not birth:
                raise ValueError(f"{name}: missing/invalid date or birth-year metadata")
            try:
                value = float(int(date[:4]) - float(birth))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name}: invalid birth-year metadata {birth!r}") from exc
            bounds = rule.get("valid_range")
            if bounds is not None and not float(bounds[0]) <= value <= float(bounds[1]):
                raise ValueError(f"{name}: value {value} outside supported range {list(bounds)}")
            return value
        source = str(rule.get("source", ""))
        if source not in row:
            raise KeyError(f"Missing metadata source {source!r} for variable {name!r}")
        raw = row[source]
        if operation == "column":
            return raw
        if operation == "category_map":
            value = str(raw or "").strip()
            mapping = rule.get("mapping", {})
            if value not in mapping:
                raise ValueError(f"{name}: unknown source category {value!r}")
            return mapping[value]
        if operation == "contains_any":
            labels = self._labels(raw)
            positives = [str(item).lower() for item in rule.get("values", [])]
            positives.extend(
                str(item).lower()
                for item in rule.get("additional_values_by_mode", {}).get(self.label_mode, [])
            )
            return float(bool(labels.intersection(positives)))
        raise ValueError(f"{name}: unsupported metadata rule operation {operation!r}")

    def _raw(self, spec, row: Mapping[str, Any]) -> Any:
        if spec.name in self.rules:
            return self._apply_rule(spec.name, row)
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
                value = encode_continuous_value(raw, spec.normalization, name=spec.name)
            elif spec.kind is VariableKind.BINARY:
                value = encode_binary_value(raw, name=spec.name)
            elif spec.kind is VariableKind.ORDINAL and spec.encoded_dim == 1:
                value = encode_continuous_value(raw, None, name=spec.name)
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
