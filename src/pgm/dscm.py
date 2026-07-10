from __future__ import annotations

import os

from runtime import configure_backend_from_argv

configure_backend_from_argv()

from typing import Dict, Optional

import jax.numpy as jnp
from flax import nnx


class DSCM:
    def __init__(self, vae, pgm):
        self.vae = vae
        self.pgm = pgm

    def _materialize(self, bundle):
        return nnx.merge(bundle.graphdef, nnx.State(bundle.params))

    def counterfactual(self, obs: Dict[str, jnp.ndarray], intervention: Dict[str, jnp.ndarray], rng=None):
        obs_vec = obs["pa"][:, 0, 0, :] if obs["pa"].ndim == 4 else obs["pa"]
        obs_pgm = {
            "thickness": obs_vec[:, 0],
            "intensity": obs_vec[:, 1],
            "digit": obs_vec[:, 2:],
        }
        pgm = self._materialize(self.pgm)
        vae = self._materialize(self.vae)
        cf_pa = pgm.counterfactual(obs=obs_pgm, intervention=intervention, rng=rng)
        latents = vae.abduct(obs["x"], obs["pa"], rng=rng)
        x_cf = vae.forward_latents(
            latents,
            cf_pa["pa"][:, None, None, :].repeat(vae.input_res, axis=1).repeat(vae.input_res, axis=2),
            rng=rng,
        )
        return {"x": x_cf, "pa": cf_pa}
