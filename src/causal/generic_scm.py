"""Schema-driven tabular structural causal models.

The legacy dataset models remain in their original modules because their
checkpoint parameter trees are part of the artifact contract.  This module is
used for new typed schemas: every variable gets a head selected from its
kind, and every non-root head receives the encoded values of its declared
parents.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp
from flax import nnx

from contracts import CausalGraphSpec, VariableKind, VariableSpec


def _matrix(value: Any, width: int) -> jax.Array:
    value = jnp.asarray(value, dtype=jnp.float32)
    if value.ndim == 0:
        value = value.reshape(1, 1)
    elif value.ndim == 1:
        value = value[:, None]
    else:
        value = value.reshape(value.shape[0], -1)
    # A scalar categorical index is a convenient input for callers while the
    # model contract itself remains one-hot.
    if width > 1 and value.shape[-1] == 1:
        value = jax.nn.one_hot(value[:, 0].astype(jnp.int32), width)
    if value.shape[-1] != width:
        raise ValueError(f"Expected encoded width {width}, got {value.shape[-1]}")
    return value


def validate_encoded_values(
    values: Mapping[str, Any], schema: CausalGraphSpec, *, atol: float = 1e-5
) -> None:
    """Validate a batch against the schema's one-hot/binary contract."""
    missing = [name for name in schema.variable_names if name not in values]
    if missing:
        raise ValueError(f"Missing schema variables: {missing}")
    for spec in schema.variables:
        value = jnp.asarray(values[spec.name])
        matrix = _matrix(value, spec.encoded_dim)
        if not bool(jnp.all(jnp.isfinite(matrix))):
            raise ValueError(f"{spec.name}: values must be finite")
        if spec.kind is VariableKind.BINARY:
            if not bool(jnp.all((matrix >= -atol) & (matrix <= 1.0 + atol))):
                raise ValueError(f"{spec.name}: binary values must be in [0, 1]")
        elif spec.kind is VariableKind.CATEGORICAL:
            if not bool(jnp.all((matrix >= -atol) & (matrix <= 1.0 + atol))):
                raise ValueError(f"{spec.name}: categorical values must be in [0, 1]")
            if not bool(jnp.allclose(jnp.sum(matrix, axis=-1), 1.0, atol=atol)):
                raise ValueError(f"{spec.name}: categorical values must be one-hot")


class SchemaDrivenSCM(nnx.Module):
    """A factorized SCM generated from a :class:`CausalGraphSpec`.

    Continuous variables use conditional diagonal Gaussian heads, categorical
    variables use softmax heads, and binary variables use Bernoulli heads.
    This is intentionally compact: it supplies a stable generic contract for
    metadata-only SCM training while legacy flow models continue to serve old
    checkpoints.
    """

    def __init__(
        self,
        schema: CausalGraphSpec,
        widths: tuple[int, ...] | list[int] = (32, 32),
        *,
        rngs: Optional[nnx.Rngs] = None,
    ):
        if not isinstance(schema, CausalGraphSpec):
            raise TypeError("schema must be a CausalGraphSpec")
        rngs = rngs or nnx.Rngs(0)
        self.schema = schema
        self.variables = {spec.name: spec.kind.value for spec in schema.variables}
        self.encoded_dims = {spec.name: int(spec.encoded_dim) for spec in schema.variables}
        self.variable_specs = schema.variables
        self.plot_variables = schema.variable_names[:2]
        self.widths = tuple(int(width) for width in widths) or (32,)
        self._parent_names = {name: schema.parents_of(name) for name in schema.variable_names}
        self._head_attrs: dict[str, tuple[str, ...]] = {}
        self._root_attrs: dict[str, tuple[str, ...]] = {}

        for index, spec in enumerate(schema.variables):
            parents = self._parent_names[spec.name]
            output_dim = 2 if spec.kind is VariableKind.CONTINUOUS else (
                1 if spec.kind is VariableKind.BINARY else spec.encoded_dim
            )
            if parents:
                parent_dim = sum(self.encoded_dims[parent] for parent in parents)
                hidden = self.widths[0]
                first = f"head_{index}_hidden"
                second = f"head_{index}_output"
                setattr(self, first, nnx.Linear(parent_dim, hidden, rngs=rngs))
                setattr(self, second, nnx.Linear(hidden, output_dim, rngs=rngs))
                self._head_attrs[spec.name] = (first, second)
            else:
                root = f"root_{index}"
                setattr(self, root, nnx.Param(jnp.zeros((1, output_dim), dtype=jnp.float32)))
                self._root_attrs[spec.name] = (root,)

    def _parents(self, name: str, values: Mapping[str, Any]) -> jax.Array | None:
        parents = self._parent_names[name]
        if not parents:
            return None
        missing = [parent for parent in parents if parent not in values]
        if missing:
            raise KeyError(f"{name!r} is missing parent values {missing}")
        return jnp.concatenate(
            [_matrix(values[parent], self.encoded_dims[parent]) for parent in parents], axis=-1
        )

    def _raw(self, spec: VariableSpec, values: Mapping[str, Any]) -> jax.Array:
        parent_values = self._parents(spec.name, values)
        if parent_values is None:
            return getattr(self, self._root_attrs[spec.name][0])[...]
        first, second = self._head_attrs[spec.name]
        return getattr(self, second)(jax.nn.silu(getattr(self, first)(parent_values)))

    def logits(self, name: str, values: Mapping[str, Any]) -> jax.Array:
        """Return the raw categorical/Bernoulli logits for a variable."""
        spec = next((item for item in self.variable_specs if item.name == name), None)
        if spec is None:
            raise KeyError(name)
        if spec.kind is VariableKind.CONTINUOUS:
            raise ValueError(f"Continuous variable {name!r} has loc/scale, not logits")
        return self._raw(spec, values)

    def _distribution(self, spec: VariableSpec, values: Mapping[str, Any]) -> tuple[jax.Array, jax.Array | None]:
        raw = self._raw(spec, values)
        if spec.kind is VariableKind.CONTINUOUS:
            loc, log_scale = jnp.split(raw, 2, axis=-1)
            return loc, jnp.clip(log_scale, -7.0, 5.0)
        return raw, None

    def log_prob(self, **values: jax.Array) -> dict[str, jax.Array]:
        missing = [name for name in self.schema.variable_names if name not in values]
        if missing:
            raise KeyError(f"Missing schema variables: {missing}")
        terms: dict[str, jax.Array] = {}
        for spec in self.schema.variables:
            raw, log_scale = self._distribution(spec, values)
            target = _matrix(values[spec.name], spec.encoded_dim)
            if spec.kind is VariableKind.CONTINUOUS:
                scale = jnp.exp(log_scale)
                terms[spec.name] = jnp.sum(
                    -0.5 * jnp.square((target - raw) / scale)
                    - jnp.log(scale)
                    - 0.5 * jnp.log(2.0 * jnp.pi), axis=-1
                )
            elif spec.kind is VariableKind.BINARY:
                terms[spec.name] = jnp.sum(
                    target * jax.nn.log_sigmoid(raw)
                    + (1.0 - target) * jax.nn.log_sigmoid(-raw), axis=-1
                )
            else:
                terms[spec.name] = jnp.sum(
                    target * jax.nn.log_softmax(raw, axis=-1), axis=-1
                )
        joint = sum(terms.values())
        return {**terms, "joint": joint}

    def sample(self, n_samples: int = 1, rng: Optional[jax.Array] = None) -> dict[str, jax.Array]:
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        rng = jax.random.PRNGKey(0) if rng is None else rng
        keys = iter(jax.random.split(rng, len(self.schema.variable_names)))
        values: dict[str, jax.Array] = {}
        specs = {spec.name: spec for spec in self.schema.variables}
        for name in self.schema.topological_order:
            spec = specs[name]
            key = next(keys)
            raw, log_scale = self._distribution(spec, values)
            if spec.kind is VariableKind.CONTINUOUS:
                values[spec.name] = raw + jnp.exp(log_scale) * jax.random.normal(key, (n_samples, spec.encoded_dim))
            elif spec.kind is VariableKind.BINARY:
                probability = jax.nn.sigmoid(raw)
                values[spec.name] = jax.random.bernoulli(key, probability, shape=(n_samples, 1)).astype(jnp.float32)
            else:
                index = jax.random.categorical(key, raw, axis=-1, shape=(n_samples,))
                values[spec.name] = jax.nn.one_hot(index, spec.encoded_dim)
        values["pa"] = jnp.concatenate([_matrix(values[name], self.encoded_dims[name]) for name in self.schema.variable_names], axis=-1)
        return values

    def counterfactual(
        self,
        obs: Mapping[str, Any],
        intervention: Mapping[str, Any],
        rng: Optional[jax.Array] = None,
    ) -> dict[str, jax.Array]:
        self.schema.validate_intervention(intervention)
        values: dict[str, jax.Array] = {}
        key = jax.random.PRNGKey(0) if rng is None else rng
        specs = {spec.name: spec for spec in self.schema.variables}
        for index, name in enumerate(self.schema.topological_order):
            spec = specs[name]
            if spec.name in intervention:
                values[spec.name] = _matrix(intervention[spec.name], spec.encoded_dim)
                continue
            if spec.name in obs and not self._parent_names[spec.name]:
                values[spec.name] = _matrix(obs[spec.name], spec.encoded_dim)
                continue
            raw, log_scale = self._distribution(spec, values)
            key, sample_key = jax.random.split(key)
            if spec.kind is VariableKind.CONTINUOUS:
                values[spec.name] = raw + jnp.exp(log_scale) * jax.random.normal(sample_key, raw.shape)
            elif spec.kind is VariableKind.BINARY:
                values[spec.name] = jax.random.bernoulli(sample_key, jax.nn.sigmoid(raw)).astype(jnp.float32)
            else:
                values[spec.name] = jax.nn.one_hot(jax.random.categorical(sample_key, raw), spec.encoded_dim)
        values["pa"] = jnp.concatenate([_matrix(values[name], self.encoded_dims[name]) for name in self.schema.variable_names], axis=-1)
        return values


GenericSCM = SchemaDrivenSCM
GenericCausalModel = SchemaDrivenSCM
SchemaSCM = SchemaDrivenSCM
