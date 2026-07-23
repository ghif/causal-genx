from __future__ import annotations

import jax.numpy as jnp
from flax import nnx
from data.conditioning import ParentEncoder
from contracts import CausalGraphSpec

class DeepStructuralCausalModel:
    """Schema-driven image and structural causal model composition."""

    def __init__(self, image_model, scm, schema: CausalGraphSpec):
        self.image_model, self.scm, self.schema = image_model, scm, schema
        self.parents = ParentEncoder(schema)

    def counterfactual(self, image, observed, intervention, *, rng, t_abduct: float = 1.0):
        self.schema.validate_intervention(intervention)
        variables = self.scm.counterfactual(obs=observed, intervention=intervention, rng=rng)
        variables = {name: variables[name] for name in self.schema.variable_names}
        height, width = image.shape[1:3]
        observed_pa = jnp.asarray(self.parents.spatial(observed, height, width)).transpose(0, 2, 3, 1)
        cf_pa = jnp.asarray(self.parents.spatial(variables, height, width)).transpose(0, 2, 3, 1)
        latents = self.image_model.abduct(image, observed_pa, t=t_abduct, rng=rng)
        cf_loc, cf_scale = self.image_model.forward_latents(latents, cf_pa, rng=rng)
        rec_loc, rec_scale = self.image_model.forward_latents(latents, observed_pa, rng=rng)
        residual = (image - rec_loc) / jnp.clip(rec_scale, min=1e-12)
        return {"image": jnp.clip(cf_loc + cf_scale * residual, -1, 1), "variables": variables}
