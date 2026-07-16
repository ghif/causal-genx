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

    def _spatial_parents(self, parents: jnp.ndarray, input_res: int) -> jnp.ndarray:
        if parents.ndim == 4:
            return parents
        return parents[:, None, None, :].repeat(input_res, axis=1).repeat(input_res, axis=2)

    def counterfactual(
        self,
        obs: Dict[str, jnp.ndarray],
        intervention: Dict[str, jnp.ndarray],
        rng=None,
        cf_particles: int = 1,
        t_abduct: float = 1.0,
    ):
        obs_vec = obs["pa"][:, 0, 0, :] if obs["pa"].ndim == 4 else obs["pa"]
        obs_pgm = {
            "thickness": obs_vec[:, 0],
            "intensity": obs_vec[:, 1],
            "digit": obs_vec[:, 2:],
        }
        pgm = self._materialize(self.pgm)
        vae = self._materialize(self.vae)
        obs_pa = self._spatial_parents(obs["pa"], vae.input_res)

        if cf_particles > 1:
            cfs = {"x": jnp.zeros_like(obs["x"]), "x2": jnp.zeros_like(obs["x"])}
        for _ in range(cf_particles):
            cf_pa = pgm.counterfactual(obs=obs_pgm, intervention=intervention, rng=rng)
            cf_pa_spatial = self._spatial_parents(cf_pa["pa"], vae.input_res)
            latents = vae.abduct(obs["x"], obs_pa, t=t_abduct, rng=rng)
            cf_loc, cf_scale = vae.forward_latents(latents, cf_pa_spatial, rng=rng)
            rec_loc, rec_scale = vae.forward_latents(latents, obs_pa, rng=rng)
            u = (obs["x"] - rec_loc) / jnp.clip(rec_scale, min=1e-12)
            x_cf = jnp.clip(cf_loc + cf_scale * u, min=-1, max=1)
            if cf_particles > 1:
                cfs["x"] = cfs["x"] + x_cf
                cfs["x2"] = cfs["x2"] + x_cf**2
            else:
                cfs = {"x": x_cf}

        if cf_particles > 1:
            var_cf_x = (cfs["x2"] - cfs["x"] ** 2 / cf_particles) / cf_particles
            cfs.pop("x2", None)
            cfs["x"] = cfs["x"] / cf_particles
        else:
            var_cf_x = None

        cfs.update(cf_pa)
        if var_cf_x is not None:
            cfs["var_cf_x"] = var_cf_x
        return cfs
