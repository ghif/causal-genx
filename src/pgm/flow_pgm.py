from __future__ import annotations

import os

from runtime import configure_backend_from_argv

configure_backend_from_argv()

from typing import Dict, Optional

import jax
import jax.numpy as jnp
from flax import nnx


class MorphoMNISTPGM(nnx.Module):
    def __init__(self, context_dim: int = 12, hidden_dim: int = 64, rngs: Optional[nnx.Rngs] = None):
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.digit_logits = nnx.Param(jnp.zeros((10,), dtype=jnp.float32))
        self.thickness_mu = nnx.Param(jnp.zeros((10,), dtype=jnp.float32))
        self.thickness_logsigma = nnx.Param(jnp.full((10,), -0.2, dtype=jnp.float32))
        self.intensity_mu = nnx.Param(jnp.zeros((10,), dtype=jnp.float32))
        self.intensity_logsigma = nnx.Param(jnp.full((10,), -0.2, dtype=jnp.float32))
        self.conv1 = nnx.Conv(in_features=1, out_features=16, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(in_features=16, out_features=32, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv3 = nnx.Conv(in_features=32, out_features=64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.fc = nnx.Linear(in_features=64 * 4 * 4, out_features=self.hidden_dim, rngs=rngs)
        self.digit_head = nnx.Linear(in_features=self.hidden_dim, out_features=10, rngs=rngs)
        self.thickness_head = nnx.Linear(in_features=self.hidden_dim, out_features=1, rngs=rngs)
        self.intensity_head = nnx.Linear(in_features=self.hidden_dim, out_features=1, rngs=rngs)

    def __call__(self, x):
        return self.predict(x)

    def _encode(self, x):
        h = jax.nn.relu(self.conv1(x))
        h = jax.image.resize(h, (h.shape[0], max(1, h.shape[1] // 2), max(1, h.shape[2] // 2), h.shape[-1]), method="linear")
        h = jax.nn.relu(self.conv2(h))
        h = jax.image.resize(h, (h.shape[0], max(1, h.shape[1] // 2), max(1, h.shape[2] // 2), h.shape[-1]), method="linear")
        h = jax.nn.relu(self.conv3(h))
        h = jax.image.resize(h, (h.shape[0], max(1, h.shape[1] // 2), max(1, h.shape[2] // 2), h.shape[-1]), method="linear")
        h = h.reshape((h.shape[0], -1))
        return jax.nn.relu(self.fc(h))

    def sample(self, n_samples: int = 1, rng=None):
        rng = rng if rng is not None else jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(rng, 3)
        digit_idx = jax.random.categorical(k1, self.digit_logits[...], shape=(n_samples,))
        digit = jax.nn.one_hot(digit_idx, 10)
        t_mu = self.thickness_mu[digit_idx]
        t_sigma = jnp.exp(self.thickness_logsigma[digit_idx])
        i_mu = self.intensity_mu[digit_idx]
        i_sigma = jnp.exp(self.intensity_logsigma[digit_idx])
        thickness = t_mu + t_sigma * jax.random.normal(k2, (n_samples,))
        intensity = i_mu + i_sigma * jax.random.normal(k3, (n_samples,))
        pa = jnp.concatenate([thickness[:, None], intensity[:, None], digit], axis=-1)
        return {"digit": digit, "thickness": thickness, "intensity": intensity, "pa": pa}

    def predict(self, x):
        h = self._encode(x)
        return {"digit": self.digit_head(h), "thickness": self.thickness_head(h)[..., 0], "intensity": self.intensity_head(h)[..., 0]}

    def counterfactual(self, obs: Dict[str, jnp.ndarray], intervention: Dict[str, jnp.ndarray], rng=None):
        rng = rng if rng is not None else jax.random.PRNGKey(0)
        obs_digit = obs["digit"]
        if obs_digit.ndim > 1:
            obs_digit_idx = jnp.argmax(obs_digit, axis=-1)
        else:
            obs_digit_idx = obs_digit.astype(jnp.int32)
        digit = intervention.get("digit", obs_digit)
        digit_idx = jnp.argmax(digit, axis=-1) if getattr(digit, "ndim", 0) > 1 else digit.astype(jnp.int32)
        t = intervention.get("thickness")
        i = intervention.get("intensity")
        if t is None:
            t = self.thickness_mu[digit_idx] + jnp.exp(self.thickness_logsigma[digit_idx]) * ((obs["thickness"] - self.thickness_mu[obs_digit_idx]) / jnp.exp(self.thickness_logsigma[obs_digit_idx]))
        if i is None:
            i = self.intensity_mu[digit_idx] + jnp.exp(self.intensity_logsigma[digit_idx]) * ((obs["intensity"] - self.intensity_mu[obs_digit_idx]) / jnp.exp(self.intensity_logsigma[obs_digit_idx]))
        pa = jnp.concatenate([jnp.asarray(t)[..., None], jnp.asarray(i)[..., None], digit], axis=-1)
        return {"digit": digit, "thickness": t, "intensity": i, "pa": pa}
