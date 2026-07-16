from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from pgm.cf_parity import (
    clip_counterfactual_grads,
    damped_lagrangian_loss,
    inherit_vae_training_config,
    set_module_training_mode,
)
from utils import EMA


class _ModeRecorder:
    def __init__(self):
        self.mode = None

    def train(self):
        self.mode = "train"

    def eval(self):
        self.mode = "eval"


def test_nnx_training_mode_uses_zero_argument_train_and_eval_methods():
    module = _ModeRecorder()
    set_module_training_mode(module, True)
    assert module.mode == "train"

    set_module_training_mode(module, False)
    assert module.mode == "eval"


def test_damping_coefficient_is_detached_like_pytorch():
    def loss(elbo, lmbda):
        constraint = 4.0 - elbo
        return damped_lagrangian_loss(1.0, lmbda, constraint, damping=5.0)

    elbo_grad, lmbda_grad = jax.grad(loss, argnums=(0, 1))(2.0, 3.0)

    np.testing.assert_allclose(elbo_grad, -7.0)
    np.testing.assert_allclose(lmbda_grad, -2.0)


def test_vae_and_lagrange_gradients_share_one_global_clip_scale():
    vae_grads = {"weight": jnp.asarray([3.0, 4.0])}
    lmbda_grads = jnp.asarray(12.0)

    clipped_vae, clipped_lmbda, norm = clip_counterfactual_grads(
        vae_grads, lmbda_grads, max_norm=6.5
    )

    np.testing.assert_allclose(norm, 13.0)
    np.testing.assert_allclose(clipped_vae["weight"], [1.5, 2.0], rtol=1e-6)
    np.testing.assert_allclose(clipped_lmbda, 6.0, rtol=1e-6)


def test_counterfactual_ema_matches_pytorch_warmup_schedule():
    ema = EMA.init_from({"weight": jnp.asarray(0.0)}, decay=0.999, update_after_step=2)

    for value in (1.0, 2.0, 3.0, 4.0):
        ema.update({"weight": jnp.asarray(value)})
        np.testing.assert_allclose(ema.params["weight"], value)

    ema.update({"weight": jnp.asarray(5.0)})
    np.testing.assert_allclose(ema.params["weight"], 13.0 / 3.0, rtol=1e-6)


def test_counterfactual_training_inherits_vae_optimizer_configuration():
    args = SimpleNamespace(lr=1e-4, wd=0.01, beta=1.0)
    checkpoint_hparams = {
        "beta": 2.0,
        "parents_x": ["thickness", "intensity", "digit"],
        "input_res": 32,
        "grad_clip": 350.0,
        "grad_skip": 500.0,
        "wd": 0.1,
        "betas": [0.9, 0.9],
    }

    inherit_vae_training_config(args, checkpoint_hparams)

    assert args.lr == 1e-4
    assert args.beta == 2.0
    assert args.wd == 0.1
    assert args.betas == [0.9, 0.9]
    assert args.grad_clip == 350.0
    assert args.grad_skip == 500.0
