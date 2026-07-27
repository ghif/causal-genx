from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from .flow_scm import monotonic_rational_spline, _as_column, _normal_log_prob, _normalize_forward, _normalize_inverse


class CxrRaitPGM(nnx.Module):
    """Structural Causal Model for CXR-RAIT demography dataset.
    
    DAG structure:
      sex (categorical, 2D)
      age (continuous, normalized [-1, 1])
      age -> tb_status (categorical, 2D)
    """

    variables = {
        "age": "continuous",
        "sex": "categorical",
        "tb_status": "categorical",
    }

    def __init__(
        self,
        widths: Sequence[int] = (32, 32),
        num_bins: int = 8,
        bound: float = 3.0,
        compute_dtype: jnp.dtype = jnp.float32,
        rngs: Optional[nnx.Rngs] = None,
    ):
        rngs = rngs or nnx.Rngs(0)
        self.num_bins = int(num_bins)
        self.bound = float(bound)
        self.compute_dtype = compute_dtype

        # Marginal Sex distribution (2 classes: Female=0, Male=1)
        self.sex_logits = nnx.Param(jnp.zeros((1, 2), dtype=jnp.float32))

        # Spline parameters for Continuous Age
        self.age_widths = nnx.Param(jnp.zeros((1, self.num_bins), dtype=jnp.float32))
        self.age_heights = nnx.Param(jnp.zeros((1, self.num_bins), dtype=jnp.float32))
        self.age_derivatives = nnx.Param(jnp.zeros((1, self.num_bins - 1), dtype=jnp.float32))
        self.age_lambdas = nnx.Param(jnp.zeros((1, self.num_bins), dtype=jnp.float32))

        # Conditional TB status logits conditioned on Age: age -> tb_status
        self.tb_fc1 = nnx.Linear(1, widths[0], rngs=rngs)
        self.tb_fc2 = nnx.Linear(widths[0], 2, rngs=rngs)

    def _spline_params(self) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        w = jax.nn.softmax(self.age_widths[...], axis=-1)
        h = jax.nn.softmax(self.age_heights[...], axis=-1)
        d = jax.nn.softplus(self.age_derivatives[...])
        l = jax.nn.sigmoid(self.age_lambdas[...])
        return w, h, d, l

    def age_forward(self, base: jax.Array) -> Tuple[jax.Array, jax.Array]:
        spline, spline_logdet = monotonic_rational_spline(
            base, *self._spline_params(), bound=self.bound
        )
        output, normalize_logdet = _normalize_forward(spline)
        return output, spline_logdet + normalize_logdet

    def age_inverse(self, value: jax.Array) -> Tuple[jax.Array, jax.Array]:
        spline, normalize_logdet = _normalize_inverse(_as_column(value))
        base, spline_logdet = monotonic_rational_spline(
            spline, *self._spline_params(), inverse=True, bound=self.bound
        )
        return base, normalize_logdet + spline_logdet

    def tb_logits(self, age: jax.Array) -> jax.Array:
        h = jax.nn.silu(self.tb_fc1(_as_column(age)))
        return self.tb_fc2(h)

    def log_prob(
        self, age: jax.Array, sex: jax.Array, tb_status: jax.Array
    ) -> Dict[str, jax.Array]:
        age_base, age_logdet = self.age_inverse(age)
        age_log_prob = jnp.sum(_normal_log_prob(age_base) + age_logdet, axis=-1)

        sex_log_prob = jnp.sum(
            jnp.asarray(sex) * jax.nn.log_softmax(self.sex_logits[...], axis=-1),
            axis=-1,
        )

        tb_log = jax.nn.log_softmax(self.tb_logits(age), axis=-1)
        tb_log_prob = jnp.sum(jnp.asarray(tb_status) * tb_log, axis=-1)

        return {
            "age": age_log_prob,
            "sex": sex_log_prob,
            "tb_status": tb_log_prob,
            "joint": age_log_prob + sex_log_prob + tb_log_prob,
        }

    def sample(
        self, n_samples: int = 1, rng: Optional[jax.Array] = None
    ) -> Dict[str, jax.Array]:
        rng = jax.random.PRNGKey(0) if rng is None else rng
        sex_key, age_key, tb_key = jax.random.split(rng, 3)

        # Sample Sex
        sex_idx = jax.random.categorical(sex_key, self.sex_logits[0], shape=(n_samples,))
        sex = jax.nn.one_hot(sex_idx, 2)

        # Sample Age
        age, _ = self.age_forward(jax.random.normal(age_key, (n_samples, 1)))

        # Sample TB status conditioned on Age
        tb_logits = self.tb_logits(age)
        tb_idx = jax.random.categorical(tb_key, tb_logits, axis=-1)
        tb_status = jax.nn.one_hot(tb_idx, 2)

        pa = jnp.concatenate([age, sex, tb_status], axis=-1)
        return {
            "age": age,
            "sex": sex,
            "tb_status": tb_status,
            "pa": pa,
        }

    def infer_exogeneous(self, obs: Dict[str, jax.Array]) -> Dict[str, jax.Array]:
        age_base, _ = self.age_inverse(obs["age"])
        return {"age_base": age_base}

    def counterfactual(
        self,
        obs: Dict[str, jax.Array],
        intervention: Dict[str, jax.Array],
        rng: Optional[jax.Array] = None,
    ) -> Dict[str, jax.Array]:
        exogeneous = self.infer_exogeneous(obs)
        sex = jnp.asarray(intervention.get("sex", obs["sex"]))

        if "age" in intervention:
            age = _as_column(intervention["age"])
        else:
            age, _ = self.age_forward(exogeneous["age_base"])

        if "tb_status" in intervention:
            tb_status = jnp.asarray(intervention["tb_status"])
        else:
            tb_logits = self.tb_logits(age)
            tb_idx = jnp.argmax(tb_logits, axis=-1)
            tb_status = jax.nn.one_hot(tb_idx, 2)

        pa = jnp.concatenate([age, sex, tb_status], axis=-1)
        return {
            "age": age,
            "sex": sex,
            "tb_status": tb_status,
            "pa": pa,
        }
