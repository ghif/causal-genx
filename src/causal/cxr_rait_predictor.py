from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from .image_parent_predictor import CNNEncoder, _as_column, _normal_log_prob, _positive_scale


class CxrRaitSupAuxPredictor(nnx.Module):
    """Anticausal image parent predictor for CXR-RAIT chest X-rays.
    
    Predicts parents: age (continuous), sex (categorical 2D), and tb_status (categorical 2D).
    """

    variables = {
        "age": "continuous",
        "sex": "categorical",
        "tb_status": "categorical",
    }

    def __init__(
        self,
        input_channels: int = 1,
        input_res: int = 128,
        width: int = 16,
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

        # Age encoder (predicts mean and log-scale)
        self.encoder_a = CNNEncoder(
            input_shape,
            width=self.width,
            num_outputs=2,
            context_dim=0,
            compute_dtype=self.compute_dtype,
            rngs=rngs,
        )

        # Sex encoder (predicts 2-class logits)
        self.encoder_s = CNNEncoder(
            input_shape,
            width=self.width,
            num_outputs=2,
            context_dim=0,
            compute_dtype=self.compute_dtype,
            rngs=rngs,
        )

        # TB status encoder (predicts 2-class logits)
        self.encoder_tb = CNNEncoder(
            input_shape,
            width=self.width,
            num_outputs=2,
            context_dim=0,
            compute_dtype=self.compute_dtype,
            rngs=rngs,
        )

    def _age_params(self, x):
        loc, logscale = jnp.split(self.encoder_a(x), 2, axis=-1)
        return jnp.tanh(loc.astype(jnp.float32)), logscale.astype(jnp.float32)

    def _sex_logits(self, x):
        return self.encoder_s(x).astype(jnp.float32)

    def _tb_logits(self, x):
        return self.encoder_tb(x).astype(jnp.float32)

    def predict(self, *, x, **_):
        a_loc, _ = self._age_params(x)
        sex_logits = self._sex_logits(x)
        tb_logits = self._tb_logits(x)
        return {
            "age": a_loc,
            "sex": jax.nn.softmax(sex_logits, axis=-1),
            "tb_status": jax.nn.softmax(tb_logits, axis=-1),
        }

    def anticausal_log_probs(self, *, x, age, sex, tb_status, **_):
        a_loc, a_logscale = self._age_params(x)
        sex_logits = self._sex_logits(x)
        tb_logits = self._tb_logits(x)

        a_scale = _positive_scale(a_logscale, self.std_fixed)
        age = _as_column(age)

        age_log_prob = jnp.sum(
            _normal_log_prob((age - a_loc) / a_scale) - jnp.log(a_scale),
            axis=-1,
        )
        sex_log_prob = jnp.sum(
            jnp.asarray(sex, dtype=jnp.float32)
            * jax.nn.log_softmax(sex_logits, axis=-1),
            axis=-1,
        )
        tb_log_prob = jnp.sum(
            jnp.asarray(tb_status, dtype=jnp.float32)
            * jax.nn.log_softmax(tb_logits, axis=-1),
            axis=-1,
        )
        joint = age_log_prob + sex_log_prob + tb_log_prob
        return {
            "age_aux": age_log_prob,
            "sex_aux": sex_log_prob,
            "tb_status_aux": tb_log_prob,
            "joint": joint,
        }

    def model_anticausal(self, **obs):
        return self.anticausal_log_probs(**obs)

    def svi_model(self, **obs):
        return self.model_anticausal(**obs)

    def guide_pass(self, **obs):
        del obs
        return None
