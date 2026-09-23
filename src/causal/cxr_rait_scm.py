from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from .flow_scm import monotonic_rational_spline, _as_column, _normal_log_prob, _normalize_forward, _normalize_inverse


class CxrRaitPGM(nnx.Module):
    """Structural Causal Model for CXR-RAIT demography dataset.
    
    DAG structure:
      age -> image (X)
      gender -> image (X)
      tb_status -> image (X)
      age -> tb_status
    """


    variables = {
        "age": "continuous",
        "gender": "categorical",
        "tb_status": "categorical",
    }
    plot_variables = ("age", "tb_status")

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

        # Marginal Gender distribution (2 classes: Female=0, Male=1)
        self.gender_logits = nnx.Param(jnp.zeros((1, 2), dtype=jnp.float32))

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
        self, age: jax.Array, gender: jax.Array, tb_status: jax.Array
    ) -> Dict[str, jax.Array]:
        age_base, age_logdet = self.age_inverse(age)
        age_log_prob = jnp.sum(_normal_log_prob(age_base) + age_logdet, axis=-1)

        gender_log_prob = jnp.sum(
            jnp.asarray(gender) * jax.nn.log_softmax(self.gender_logits[...], axis=-1),
            axis=-1,
        )

        tb_log = jax.nn.log_softmax(self.tb_logits(age), axis=-1)
        tb_log_prob = jnp.sum(jnp.asarray(tb_status) * tb_log, axis=-1)

        return {
            "age": age_log_prob,
            "gender": gender_log_prob,
            "tb_status": tb_log_prob,
            "joint": age_log_prob + gender_log_prob + tb_log_prob,
        }

    def sample(
        self, n_samples: int = 1, rng: Optional[jax.Array] = None
    ) -> Dict[str, jax.Array]:
        rng = jax.random.PRNGKey(0) if rng is None else rng
        gender_key, age_key, tb_key = jax.random.split(rng, 3)

        # Sample Gender
        gender_idx = jax.random.categorical(gender_key, self.gender_logits[0], shape=(n_samples,))
        gender = jax.nn.one_hot(gender_idx, 2)

        # Sample Age
        age, _ = self.age_forward(jax.random.normal(age_key, (n_samples, 1)))

        # Sample TB status conditioned on Age
        tb_logits = self.tb_logits(age)
        tb_idx = jax.random.categorical(tb_key, tb_logits, axis=-1)
        tb_status = jax.nn.one_hot(tb_idx, 2)

        pa = jnp.concatenate([age, gender, tb_status], axis=-1)
        return {
            "age": age,
            "gender": gender,
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
        gender = jnp.asarray(intervention.get("gender", obs["gender"]))

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

        pa = jnp.concatenate([age, gender, tb_status], axis=-1)
        return {
            "age": age,
            "gender": gender,
            "tb_status": tb_status,
            "pa": pa,
        }


class PadChestPGM(nnx.Module):
    """Tabular SCM for PadChest metadata variables.

    Continuous marginals use the same spline flow as the CXR-RAIT and
    MorphoMNIST SCMs. Discrete acquisition variables are modeled as categorical
    or Bernoulli marginals, and TB status follows the PadChest schema parents:
    age_at_study, sex, and pediatric.
    """

    variables = {
        "age_at_study": "continuous",
        "sex": "categorical",
        "pediatric": "binary",
        "tb_status": "binary",
        "projection": "categorical",
        "view_position": "categorical",
        "study_year": "continuous",
    }
    plot_variables = ("age_at_study", "tb_status")
    continuous_variables = ("age_at_study", "study_year")

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

        for name in self.continuous_variables:
            setattr(self, f"{name}_widths", nnx.Param(jnp.zeros((1, self.num_bins), dtype=jnp.float32)))
            setattr(self, f"{name}_heights", nnx.Param(jnp.zeros((1, self.num_bins), dtype=jnp.float32)))
            setattr(self, f"{name}_derivatives", nnx.Param(jnp.zeros((1, self.num_bins - 1), dtype=jnp.float32)))
            setattr(self, f"{name}_lambdas", nnx.Param(jnp.zeros((1, self.num_bins), dtype=jnp.float32)))

        self.sex_logits = nnx.Param(jnp.zeros((1, 2), dtype=jnp.float32))
        self.pediatric_logit = nnx.Param(jnp.zeros((1, 1), dtype=jnp.float32))
        self.projection_logits = nnx.Param(jnp.zeros((1, 5), dtype=jnp.float32))
        self.view_position_logits = nnx.Param(jnp.zeros((1, 6), dtype=jnp.float32))

        tb_hidden = int(widths[0]) if widths else 32
        self.tb_fc1 = nnx.Linear(4, tb_hidden, rngs=rngs)
        self.tb_fc2 = nnx.Linear(tb_hidden, 1, rngs=rngs)

    def _spline_params(self, name: str) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        w = jax.nn.softmax(getattr(self, f"{name}_widths")[...], axis=-1)
        h = jax.nn.softmax(getattr(self, f"{name}_heights")[...], axis=-1)
        d = jax.nn.softplus(getattr(self, f"{name}_derivatives")[...])
        l = jax.nn.sigmoid(getattr(self, f"{name}_lambdas")[...])
        return w, h, d, l

    def _continuous_forward(self, name: str, base: jax.Array) -> Tuple[jax.Array, jax.Array]:
        spline, spline_logdet = monotonic_rational_spline(
            base, *self._spline_params(name), bound=self.bound
        )
        output, normalize_logdet = _normalize_forward(spline)
        return output, spline_logdet + normalize_logdet

    def _continuous_inverse(self, name: str, value: jax.Array) -> Tuple[jax.Array, jax.Array]:
        spline, normalize_logdet = _normalize_inverse(_as_column(value))
        base, spline_logdet = monotonic_rational_spline(
            spline, *self._spline_params(name), inverse=True, bound=self.bound
        )
        return base, normalize_logdet + spline_logdet

    def _binary_log_prob(self, value: jax.Array, logits: jax.Array) -> jax.Array:
        value = _as_column(value)
        logits = jnp.broadcast_to(logits, value.shape)
        return jnp.sum(value * jax.nn.log_sigmoid(logits) + (1.0 - value) * jax.nn.log_sigmoid(-logits), axis=-1)

    def tb_logits(self, age_at_study: jax.Array, sex: jax.Array, pediatric: jax.Array) -> jax.Array:
        parents = jnp.concatenate([_as_column(age_at_study), jnp.asarray(sex), _as_column(pediatric)], axis=-1)
        hidden = jax.nn.silu(self.tb_fc1(parents))
        return self.tb_fc2(hidden)

    def log_prob(
        self,
        age_at_study: jax.Array,
        sex: jax.Array,
        pediatric: jax.Array,
        tb_status: jax.Array,
        projection: jax.Array,
        view_position: jax.Array,
        study_year: jax.Array,
    ) -> Dict[str, jax.Array]:
        age_base, age_logdet = self._continuous_inverse("age_at_study", age_at_study)
        year_base, year_logdet = self._continuous_inverse("study_year", study_year)
        age_log_prob = jnp.sum(_normal_log_prob(age_base) + age_logdet, axis=-1)
        year_log_prob = jnp.sum(_normal_log_prob(year_base) + year_logdet, axis=-1)

        sex_log_prob = jnp.sum(jnp.asarray(sex) * jax.nn.log_softmax(self.sex_logits[...], axis=-1), axis=-1)
        pediatric_log_prob = self._binary_log_prob(pediatric, self.pediatric_logit[...])
        tb_log_prob = self._binary_log_prob(tb_status, self.tb_logits(age_at_study, sex, pediatric))
        projection_log_prob = jnp.sum(jnp.asarray(projection) * jax.nn.log_softmax(self.projection_logits[...], axis=-1), axis=-1)
        view_log_prob = jnp.sum(jnp.asarray(view_position) * jax.nn.log_softmax(self.view_position_logits[...], axis=-1), axis=-1)
        joint = age_log_prob + sex_log_prob + pediatric_log_prob + tb_log_prob + projection_log_prob + view_log_prob + year_log_prob
        return {
            "age_at_study": age_log_prob,
            "sex": sex_log_prob,
            "pediatric": pediatric_log_prob,
            "tb_status": tb_log_prob,
            "projection": projection_log_prob,
            "view_position": view_log_prob,
            "study_year": year_log_prob,
            "joint": joint,
        }

    def sample(self, n_samples: int = 1, rng: Optional[jax.Array] = None) -> Dict[str, jax.Array]:
        rng = jax.random.PRNGKey(0) if rng is None else rng
        age_key, sex_key, pediatric_key, tb_key, projection_key, view_key, year_key = jax.random.split(rng, 7)
        age_at_study, _ = self._continuous_forward("age_at_study", jax.random.normal(age_key, (n_samples, 1)))
        study_year, _ = self._continuous_forward("study_year", jax.random.normal(year_key, (n_samples, 1)))
        sex_idx = jax.random.categorical(sex_key, self.sex_logits[0], shape=(n_samples,))
        sex = jax.nn.one_hot(sex_idx, 2)
        pediatric = jax.random.bernoulli(pediatric_key, jax.nn.sigmoid(self.pediatric_logit[0, 0]), shape=(n_samples, 1)).astype(jnp.float32)
        tb_status = jax.random.bernoulli(tb_key, jax.nn.sigmoid(self.tb_logits(age_at_study, sex, pediatric))).astype(jnp.float32)
        projection_idx = jax.random.categorical(projection_key, self.projection_logits[0], shape=(n_samples,))
        projection = jax.nn.one_hot(projection_idx, 5)
        view_idx = jax.random.categorical(view_key, self.view_position_logits[0], shape=(n_samples,))
        view_position = jax.nn.one_hot(view_idx, 6)
        pa = jnp.concatenate([age_at_study, sex, pediatric, tb_status, projection, view_position, study_year], axis=-1)
        return {
            "age_at_study": age_at_study,
            "sex": sex,
            "pediatric": pediatric,
            "tb_status": tb_status,
            "projection": projection,
            "view_position": view_position,
            "study_year": study_year,
            "pa": pa,
        }

    def infer_exogeneous(self, obs: Dict[str, jax.Array]) -> Dict[str, jax.Array]:
        age_base, _ = self._continuous_inverse("age_at_study", obs["age_at_study"])
        year_base, _ = self._continuous_inverse("study_year", obs["study_year"])
        return {"age_at_study_base": age_base, "study_year_base": year_base}

    def counterfactual(
        self,
        obs: Dict[str, jax.Array],
        intervention: Dict[str, jax.Array],
        rng: Optional[jax.Array] = None,
    ) -> Dict[str, jax.Array]:
        del rng
        exogeneous = self.infer_exogeneous(obs)
        age_at_study = _as_column(intervention["age_at_study"]) if "age_at_study" in intervention else self._continuous_forward("age_at_study", exogeneous["age_at_study_base"])[0]
        study_year = _as_column(intervention["study_year"]) if "study_year" in intervention else self._continuous_forward("study_year", exogeneous["study_year_base"])[0]
        sex = jnp.asarray(intervention.get("sex", obs["sex"]))
        pediatric = _as_column(intervention.get("pediatric", obs["pediatric"]))
        tb_status = _as_column(intervention.get("tb_status", obs["tb_status"]))
        projection = jnp.asarray(intervention.get("projection", obs["projection"]))
        view_position = jnp.asarray(intervention.get("view_position", obs["view_position"]))
        pa = jnp.concatenate([age_at_study, sex, pediatric, tb_status, projection, view_position, study_year], axis=-1)
        return {
            "age_at_study": age_at_study,
            "sex": sex,
            "pediatric": pediatric,
            "tb_status": tb_status,
            "projection": projection,
            "view_position": view_position,
            "study_year": study_year,
            "pa": pa,
        }
