from __future__ import annotations

# ruff: noqa: E402 -- backend selection must happen before importing JAX.

from typing import Dict, Optional, Sequence, Tuple

from runtime import configure_backend_from_argv

configure_backend_from_argv()

import jax
import jax.numpy as jnp
from flax import nnx


LOG_2PI = jnp.log(2.0 * jnp.pi)


def _set_variable_value(variable: nnx.Variable, value: jax.Array) -> None:
    if hasattr(variable, "set_value"):
        variable.set_value(value)
    else:  # Flax NNX < 0.12
        variable.value = value


def _as_column(value: jax.Array) -> jax.Array:
    value = jnp.asarray(value)
    return value[..., None] if value.ndim == 1 else value


def _calculate_knots(
    lengths: jax.Array, lower: float, upper: float
) -> Tuple[jax.Array, jax.Array]:
    knots = jnp.cumsum(lengths, axis=-1)
    knots = jnp.pad(knots, ((0, 0), (1, 0)), constant_values=0.0)
    knots = (upper - lower) * knots + lower
    knots = knots.at[..., 0].set(lower)
    knots = knots.at[..., -1].set(upper)
    return knots[..., 1:] - knots[..., :-1], knots


def _select_bins(values: jax.Array, indices: jax.Array) -> jax.Array:
    indices = jnp.clip(indices, 0, values.shape[-1] - 1)
    target_shape = indices.shape[:-1] + (values.shape[-1],)
    values = jnp.broadcast_to(values, target_shape)
    return jnp.take_along_axis(values, indices, axis=-1).squeeze(-1)


def monotonic_rational_spline(
    inputs: jax.Array,
    widths: jax.Array,
    heights: jax.Array,
    derivatives: jax.Array,
    lambdas: jax.Array,
    *,
    inverse: bool = False,
    bound: float = 3.0,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
    min_lambda: float = 0.025,
    eps: float = 1e-6,
) -> Tuple[jax.Array, jax.Array]:
    """Pyro-compatible elementwise linear rational spline and log-Jacobian."""
    inputs = _as_column(inputs)
    num_bins = widths.shape[-1]
    if min_bin_width * num_bins > 1.0 or min_bin_height * num_bins > 1.0:
        raise ValueError("Minimum spline bin size is too large for the bin count")

    inside = (inputs >= -bound) & (inputs <= bound)
    widths = min_bin_width + (1.0 - min_bin_width * num_bins) * widths
    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * heights
    derivatives = min_derivative + derivatives
    widths, cumwidths = _calculate_knots(widths, -bound, bound)
    heights, cumheights = _calculate_knots(heights, -bound, bound)
    derivatives = jnp.pad(
        derivatives, ((0, 0), (1, 1)), constant_values=1.0 - min_derivative
    )

    search_knots = cumheights if inverse else cumwidths
    bin_idx = (
        jnp.sum(inputs[..., None] >= search_knots + eps, axis=-1, keepdims=True) - 1
    )
    input_widths = _select_bins(widths, bin_idx)
    input_cumwidths = _select_bins(cumwidths, bin_idx)
    input_cumheights = _select_bins(cumheights, bin_idx)
    input_delta = _select_bins(heights / widths, bin_idx)
    input_derivatives = _select_bins(derivatives, bin_idx)
    input_derivatives_plus_one = _select_bins(derivatives[..., 1:], bin_idx)
    input_heights = _select_bins(heights, bin_idx)
    input_lambdas = _select_bins(
        (1.0 - 2.0 * min_lambda) * lambdas + min_lambda, bin_idx
    )

    wa = 1.0
    wb = jnp.sqrt(input_derivatives / input_derivatives_plus_one)
    wc = (
        input_lambdas * input_derivatives
        + (1.0 - input_lambdas) * wb * input_derivatives_plus_one
    ) / input_delta
    ya = input_cumheights
    yb = input_heights + input_cumheights
    yc = ((1.0 - input_lambdas) * ya + input_lambdas * wb * yb) / (
        (1.0 - input_lambdas) + input_lambdas * wb
    )

    if inverse:
        lower = inputs <= yc
        numerator_lower = input_lambdas * (ya - inputs)
        numerator_upper = (
            (wc - input_lambdas * wb) * inputs + input_lambdas * wb * yb - wc * yc
        )
        denominator_lower = (wc - wa) * inputs + wa * ya - wc * yc
        denominator_upper = (wc - wb) * inputs + wb * yb - wc * yc
        numerator = jnp.where(lower, numerator_lower, numerator_upper)
        denominator = jnp.where(lower, denominator_lower, denominator_upper)
        theta = numerator / denominator
        outputs = theta * input_widths + input_cumwidths
        derivative_numerator = (
            jnp.where(
                lower,
                wa * wc * input_lambdas * (yc - ya),
                wb * wc * (1.0 - input_lambdas) * (yb - yc),
            )
            * input_widths
        )
        logabsdet = jnp.log(derivative_numerator) - 2.0 * jnp.log(jnp.abs(denominator))
    else:
        theta = (inputs - input_cumwidths) / input_widths
        lower = theta <= input_lambdas
        numerator_lower = wa * ya * (input_lambdas - theta) + wc * yc * theta
        numerator_upper = wc * yc * (1.0 - theta) + wb * yb * (theta - input_lambdas)
        denominator_lower = wa * (input_lambdas - theta) + wc * theta
        denominator_upper = wc * (1.0 - theta) + wb * (theta - input_lambdas)
        numerator = jnp.where(lower, numerator_lower, numerator_upper)
        denominator = jnp.where(lower, denominator_lower, denominator_upper)
        outputs = numerator / denominator
        derivative_numerator = (
            jnp.where(
                lower,
                wa * wc * input_lambdas * (yc - ya),
                wb * wc * (1.0 - input_lambdas) * (yb - yc),
            )
            / input_widths
        )
        logabsdet = jnp.log(derivative_numerator) - 2.0 * jnp.log(jnp.abs(denominator))

    outputs = jnp.where(inside, outputs, inputs)
    logabsdet = jnp.where(inside, logabsdet, 0.0)
    return outputs, logabsdet


def _normalize_forward(value: jax.Array) -> Tuple[jax.Array, jax.Array]:
    output = 2.0 * jax.nn.sigmoid(value) - 1.0
    logabsdet = jnp.log(2.0) - jax.nn.softplus(-value) - jax.nn.softplus(value)
    return output, logabsdet


def _normalize_inverse(value: jax.Array) -> Tuple[jax.Array, jax.Array]:
    finfo = jnp.finfo(jnp.asarray(value).dtype)
    probability = jnp.clip((value + 1.0) / 2.0, finfo.eps, 1.0 - finfo.eps)
    output = jnp.log(probability) - jnp.log1p(-probability)
    _, forward_logabsdet = _normalize_forward(output)
    return output, -forward_logabsdet


def _normal_log_prob(value: jax.Array) -> jax.Array:
    return -0.5 * (jnp.square(value) + LOG_2PI)


def _pytorch_linear_kernel(
    key: jax.Array, shape: Sequence[int], dtype=jnp.float32
) -> jax.Array:
    bound = 1.0 / jnp.sqrt(float(shape[0]))
    return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


def _pytorch_linear_bias(
    key: jax.Array, shape: Sequence[int], dtype=jnp.float32
) -> jax.Array:
    # NNX does not pass fan-in to bias initializers; DenseNN replaces these values below.
    return jax.random.uniform(key, shape, dtype, minval=-1.0, maxval=1.0)


class MorphoMNISTPGM(nnx.Module):
    """Pure-JAX port of the Pyro MorphoMNISTPGM used by ``sup_pgm``."""

    variables = {
        "thickness": "continuous",
        "intensity": "continuous",
        "digit": "categorical",
    }

    def __init__(
        self,
        widths: Sequence[int] = (32, 32),
        count_bins: int = 4,
        bound: float = 3.0,
        rngs: Optional[nnx.Rngs] = None,
        **_: object,
    ):
        rngs = rngs or nnx.Rngs(0)
        self.widths = tuple(int(width) for width in widths)
        self.count_bins = int(count_bins)
        self.bound = float(bound)
        self.digit_logits = nnx.Param(jnp.zeros((1, 10), dtype=jnp.float32))
        self.unnormalized_widths = nnx.Param(
            jax.random.normal(rngs.params(), (1, count_bins))
        )
        self.unnormalized_heights = nnx.Param(
            jax.random.normal(rngs.params(), (1, count_bins))
        )
        self.unnormalized_derivatives = nnx.Param(
            jax.random.normal(rngs.params(), (1, count_bins - 1))
        )
        self.unnormalized_lambdas = nnx.Param(
            jax.random.uniform(rngs.params(), (1, count_bins))
        )

        dims = (1, *self.widths, 2)
        self.context_layer_count = len(dims) - 1
        for index, (fan_in, fan_out) in enumerate(zip(dims[:-1], dims[1:])):
            layer = nnx.Linear(
                fan_in,
                fan_out,
                kernel_init=_pytorch_linear_kernel,
                bias_init=_pytorch_linear_bias,
                rngs=rngs,
            )
            bias_bound = 1.0 / jnp.sqrt(float(fan_in))
            _set_variable_value(
                layer.bias,
                jax.random.uniform(
                    rngs.params(),
                    layer.bias.shape,
                    layer.bias.dtype,
                    minval=-bias_bound,
                    maxval=bias_bound,
                ),
            )
            setattr(self, f"context_layer_{index}", layer)

    def _spline_params(self) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return (
            jax.nn.softmax(self.unnormalized_widths[...], axis=-1),
            jax.nn.softmax(self.unnormalized_heights[...], axis=-1),
            jax.nn.softplus(self.unnormalized_derivatives[...]),
            jax.nn.sigmoid(self.unnormalized_lambdas[...]),
        )

    def _context(self, thickness: jax.Array) -> Tuple[jax.Array, jax.Array]:
        hidden = _as_column(thickness)
        for index in range(self.context_layer_count - 1):
            layer = getattr(self, f"context_layer_{index}")
            hidden = jax.nn.gelu(layer(hidden))
        output_layer = getattr(self, f"context_layer_{self.context_layer_count - 1}")
        loc, log_scale = jnp.split(output_layer(hidden), 2, axis=-1)
        return loc, log_scale

    def thickness_forward(self, base: jax.Array) -> Tuple[jax.Array, jax.Array]:
        spline, spline_logdet = monotonic_rational_spline(
            base, *self._spline_params(), bound=self.bound
        )
        output, normalize_logdet = _normalize_forward(spline)
        return output, spline_logdet + normalize_logdet

    def thickness_inverse(self, value: jax.Array) -> Tuple[jax.Array, jax.Array]:
        spline, normalize_logdet = _normalize_inverse(_as_column(value))
        base, spline_logdet = monotonic_rational_spline(
            spline, *self._spline_params(), inverse=True, bound=self.bound
        )
        return base, normalize_logdet + spline_logdet

    def intensity_forward(
        self, base: jax.Array, thickness: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        loc, log_scale = self._context(thickness)
        affine = _as_column(base) * jnp.exp(log_scale) + loc
        output, normalize_logdet = _normalize_forward(affine)
        return output, log_scale + normalize_logdet

    def intensity_inverse(
        self, value: jax.Array, thickness: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        affine, normalize_logdet = _normalize_inverse(_as_column(value))
        loc, log_scale = self._context(thickness)
        base = (affine - loc) * jnp.exp(-log_scale)
        return base, normalize_logdet - log_scale

    def log_prob(
        self, thickness: jax.Array, intensity: jax.Array, digit: jax.Array
    ) -> Dict[str, jax.Array]:
        thickness_base, thickness_logdet = self.thickness_inverse(thickness)
        intensity_base, intensity_logdet = self.intensity_inverse(intensity, thickness)
        digit_log_prob = jnp.sum(
            jnp.asarray(digit) * jax.nn.log_softmax(self.digit_logits[...], axis=-1),
            axis=-1,
        )
        thickness_log_prob = jnp.sum(
            _normal_log_prob(thickness_base) + thickness_logdet, axis=-1
        )
        intensity_log_prob = jnp.sum(
            _normal_log_prob(intensity_base) + intensity_logdet, axis=-1
        )
        return {
            "digit": digit_log_prob,
            "thickness": thickness_log_prob,
            "intensity": intensity_log_prob,
            "joint": digit_log_prob + thickness_log_prob + intensity_log_prob,
        }

    def sample(
        self, n_samples: int = 1, rng: Optional[jax.Array] = None
    ) -> Dict[str, jax.Array]:
        rng = jax.random.PRNGKey(0) if rng is None else rng
        digit_key, thickness_key, intensity_key = jax.random.split(rng, 3)
        digit_index = jax.random.categorical(
            digit_key, self.digit_logits[0], shape=(n_samples,)
        )
        digit = jax.nn.one_hot(digit_index, 10)
        thickness, _ = self.thickness_forward(
            jax.random.normal(thickness_key, (n_samples, 1))
        )
        intensity, _ = self.intensity_forward(
            jax.random.normal(intensity_key, (n_samples, 1)), thickness
        )
        return {
            "thickness": thickness,
            "intensity": intensity,
            "digit": digit,
            "pa": jnp.concatenate([thickness, intensity, digit], axis=-1),
        }

    def infer_exogeneous(self, obs: Dict[str, jax.Array]) -> Dict[str, jax.Array]:
        thickness_base, _ = self.thickness_inverse(obs["thickness"])
        intensity_base, _ = self.intensity_inverse(obs["intensity"], obs["thickness"])
        return {"thickness_base": thickness_base, "intensity_base": intensity_base}

    def counterfactual(
        self,
        obs: Dict[str, jax.Array],
        intervention: Dict[str, jax.Array],
        rng: Optional[jax.Array] = None,
    ) -> Dict[str, jax.Array]:
        del rng
        exogeneous = self.infer_exogeneous(obs)
        digit = jnp.asarray(intervention.get("digit", obs["digit"]))
        if "thickness" in intervention:
            thickness = _as_column(intervention["thickness"])
        else:
            thickness, _ = self.thickness_forward(exogeneous["thickness_base"])
        if "intensity" in intervention:
            intensity = _as_column(intervention["intensity"])
        else:
            intensity, _ = self.intensity_forward(
                exogeneous["intensity_base"], thickness
            )
        return {
            "thickness": thickness,
            "intensity": intensity,
            "digit": digit,
            "pa": jnp.concatenate([thickness, intensity, digit], axis=-1),
        }
