from __future__ import annotations

# ruff: noqa: E402 -- backend selection must happen before importing JAX.

from typing import Optional

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


def _fan_in(shape) -> int:
    if len(shape) <= 1:
        return int(shape[0])
    if len(shape) == 2:
        return int(shape[0])
    return int(jnp.prod(jnp.asarray(shape[:-1])))


def _torch_linear_kernel(key: jax.Array, shape, dtype=jnp.float32) -> jax.Array:
    bound = 1.0 / jnp.sqrt(float(shape[0]))
    return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


def _torch_linear_bias(
    key: jax.Array, shape, dtype=jnp.float32, *, fan_in: int
) -> jax.Array:
    bound = 1.0 / jnp.sqrt(float(fan_in))
    return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


def _torch_conv_kernel(key: jax.Array, shape, dtype=jnp.float32) -> jax.Array:
    bound = 1.0 / jnp.sqrt(float(_fan_in(shape)))
    return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


def _torch_conv_bias(key: jax.Array, shape, dtype=jnp.float32) -> jax.Array:
    bound = 1.0 / jnp.sqrt(float(_fan_in((shape[0],))))
    return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


def _as_column(value: jax.Array) -> jax.Array:
    value = jnp.asarray(value)
    return value[..., None] if value.ndim == 1 else value


def _normal_log_prob(value: jax.Array) -> jax.Array:
    return -0.5 * (jnp.square(value) + LOG_2PI)


def _positive_scale(raw_scale: jax.Array, fixed_std: float = 0.0) -> jax.Array:
    if fixed_std > 0:
        return jnp.ones_like(raw_scale) * fixed_std
    return jax.nn.softplus(raw_scale)


class CNNEncoder(nnx.Module):
    def __init__(
        self,
        in_shape=(1, 32, 32),
        width: int = 8,
        num_outputs: int = 1,
        context_dim: int = 0,
        compute_dtype: jnp.dtype = jnp.float32,
        rngs: Optional[nnx.Rngs] = None,
    ):
        rngs = rngs or nnx.Rngs(0)
        self.in_shape = tuple(in_shape)
        self.width = int(width)
        self.num_outputs = int(num_outputs)
        self.context_dim = int(context_dim)
        self.compute_dtype = compute_dtype

        def activation(x):
            return jax.nn.leaky_relu(x, negative_slope=0.01)

        self._activation = activation

        self.conv1 = nnx.Conv(
            in_features=self.in_shape[0],
            out_features=self.width,
            kernel_size=(7, 7),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            kernel_init=_torch_conv_kernel,
            rngs=rngs,
        )
        self.bn1 = nnx.BatchNorm(
            num_features=self.width,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=self.width,
            out_features=2 * self.width,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            kernel_init=_torch_conv_kernel,
            rngs=rngs,
        )
        self.bn2 = nnx.BatchNorm(
            num_features=2 * self.width,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.conv3 = nnx.Conv(
            in_features=2 * self.width,
            out_features=2 * self.width,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            kernel_init=_torch_conv_kernel,
            rngs=rngs,
        )
        self.bn3 = nnx.BatchNorm(
            num_features=2 * self.width,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.conv4 = nnx.Conv(
            in_features=2 * self.width,
            out_features=4 * self.width,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            kernel_init=_torch_conv_kernel,
            rngs=rngs,
        )
        self.bn4 = nnx.BatchNorm(
            num_features=4 * self.width,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.conv5 = nnx.Conv(
            in_features=4 * self.width,
            out_features=4 * self.width,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            kernel_init=_torch_conv_kernel,
            rngs=rngs,
        )
        self.bn5 = nnx.BatchNorm(
            num_features=4 * self.width,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.conv6 = nnx.Conv(
            in_features=4 * self.width,
            out_features=8 * self.width,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            kernel_init=_torch_conv_kernel,
            rngs=rngs,
        )
        self.bn6 = nnx.BatchNorm(
            num_features=8 * self.width,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.fc1 = nnx.Linear(
            8 * self.width + self.context_dim,
            8 * self.width,
            use_bias=False,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            kernel_init=_torch_linear_kernel,
            rngs=rngs,
        )
        self.bn_fc = nnx.BatchNorm(
            num_features=8 * self.width,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.fc2 = nnx.Linear(
            8 * self.width,
            self.num_outputs,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            kernel_init=_torch_linear_kernel,
            bias_init=lambda key, shape, dtype=jnp.float32: _torch_linear_bias(
                key, shape, dtype, fan_in=8 * self.width
            ),
            rngs=rngs,
        )

    def __call__(self, x, y=None):
        x = jnp.asarray(x, dtype=self.compute_dtype)
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D image batch, got shape {x.shape}")
        if x.shape[1] in (1, 3):
            x = jnp.transpose(x, (0, 2, 3, 1))

        x = self._activation(self.bn1(self.conv1(x)))
        x = self._activation(self.bn2(self.conv2(x)))
        x = self._activation(self.bn3(self.conv3(x)))
        x = self._activation(self.bn4(self.conv4(x)))
        x = self._activation(self.bn5(self.conv5(x)))
        x = self._activation(self.bn6(self.conv6(x)))
        x = x.mean(axis=(1, 2))
        if y is not None:
            y = jnp.asarray(y, dtype=self.compute_dtype)
            x = jnp.concatenate([x, y], axis=-1)
        x = self.fc1(x)
        x = self._activation(self.bn_fc(x))
        return self.fc2(x)


class MorphoMNISTSupAuxPredictor(nnx.Module):
    variables = {
        "thickness": "continuous",
        "intensity": "continuous",
        "digit": "categorical",
    }

    def __init__(
        self,
        input_channels: int = 1,
        input_res: int = 32,
        width: int = 8,
        std_fixed: float = 0.0,
        compute_dtype: jnp.dtype = jnp.float32,
        rngs: Optional[nnx.Rngs] = None,
    ):
        rngs = rngs or nnx.Rngs(0)
        self.input_channels = int(input_channels)
        self.input_res = int(input_res)
        self.width = int(width)
        self.std_fixed = float(std_fixed)
        self.compute_dtype = compute_dtype
        input_shape = (self.input_channels, self.input_res, self.input_res)
        self.encoder_t = CNNEncoder(
            input_shape,
            width=self.width,
            num_outputs=2,
            context_dim=1,
            compute_dtype=self.compute_dtype,
            rngs=rngs,
        )
        self.encoder_i = CNNEncoder(
            input_shape,
            width=self.width,
            num_outputs=2,
            context_dim=0,
            compute_dtype=self.compute_dtype,
            rngs=rngs,
        )
        self.encoder_y = CNNEncoder(
            input_shape,
            width=self.width,
            num_outputs=10,
            context_dim=0,
            compute_dtype=self.compute_dtype,
            rngs=rngs,
        )

    def _thickness_params(self, x, intensity):
        loc, logscale = jnp.split(self.encoder_t(x, y=intensity), 2, axis=-1)
        return jnp.tanh(loc.astype(jnp.float32)), logscale.astype(jnp.float32)

    def _intensity_params(self, x):
        loc, logscale = jnp.split(self.encoder_i(x), 2, axis=-1)
        return jnp.tanh(loc.astype(jnp.float32)), logscale.astype(jnp.float32)

    def _digit_logits(self, x):
        return self.encoder_y(x).astype(jnp.float32)

    def predict(self, *, x, intensity, **_):
        t_loc, _ = self._thickness_params(x, intensity)
        i_loc, _ = self._intensity_params(x)
        digit_logits = self._digit_logits(x)
        return {
            "thickness": t_loc,
            "intensity": i_loc,
            "digit": jax.nn.softmax(digit_logits, axis=-1),
        }

    def anticausal_log_probs(self, *, x, thickness, intensity, digit, **_):
        t_loc, t_logscale = self._thickness_params(x, intensity)
        i_loc, i_logscale = self._intensity_params(x)
        digit_logits = self._digit_logits(x)

        t_scale = _positive_scale(t_logscale, self.std_fixed)
        i_scale = _positive_scale(i_logscale, self.std_fixed)
        thickness = _as_column(thickness)
        intensity = _as_column(intensity)
        thickness_log_prob = jnp.sum(
            _normal_log_prob((thickness - t_loc) / t_scale) - jnp.log(t_scale),
            axis=-1,
        )
        intensity_log_prob = jnp.sum(
            _normal_log_prob((intensity - i_loc) / i_scale) - jnp.log(i_scale),
            axis=-1,
        )
        digit_log_prob = jnp.sum(
            jnp.asarray(digit, dtype=jnp.float32)
            * jax.nn.log_softmax(digit_logits, axis=-1),
            axis=-1,
        )
        joint = thickness_log_prob + intensity_log_prob + digit_log_prob
        return {
            "thickness_aux": thickness_log_prob,
            "intensity_aux": intensity_log_prob,
            "digit_aux": digit_log_prob,
            "joint": joint,
        }

    def model_anticausal(self, **obs):
        return self.anticausal_log_probs(**obs)

    def svi_model(self, **obs):
        return self.model_anticausal(**obs)

    def guide_pass(self, **obs):
        del obs
        return None
