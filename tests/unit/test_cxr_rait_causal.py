import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from causal.cxr_rait_scm import CxrRaitPGM
from causal.cxr_rait_predictor import CxrRaitSupAuxPredictor


def test_cxr_rait_pgm_shapes_and_sampling():
    model = CxrRaitPGM(widths=(16, 16), rngs=nnx.Rngs(0))
    samples = model.sample(n_samples=4, rng=jax.random.PRNGKey(0))

    assert samples["age"].shape == (4, 1)
    assert samples["sex"].shape == (4, 2)
    assert samples["tb_status"].shape == (4, 2)
    assert samples["pa"].shape == (4, 5)

    log_probs = model.log_prob(samples["age"], samples["sex"], samples["tb_status"])
    assert "joint" in log_probs
    assert log_probs["joint"].shape == (4,)


def test_cxr_rait_pgm_counterfactual():
    model = CxrRaitPGM(widths=(16, 16), rngs=nnx.Rngs(0))
    obs = {
        "age": jnp.array([[0.2]]),
        "sex": jnp.array([[1.0, 0.0]]),
        "tb_status": jnp.array([[1.0, 0.0]]),
    }
    intervention = {"tb_status": jnp.array([[0.0, 1.0]])}
    cf = model.counterfactual(obs, intervention)

    assert jnp.array_equal(cf["tb_status"], intervention["tb_status"])
    assert cf["age"].shape == (1, 1)


def test_cxr_rait_predictor_forward():
    model = CxrRaitSupAuxPredictor(input_channels=1, input_res=128, width=8, rngs=nnx.Rngs(0))
    x = jnp.zeros((2, 1, 128, 128))
    pred = model.predict(x=x)

    assert pred["age"].shape == (2, 1)
    assert pred["sex"].shape == (2, 2)
    assert pred["tb_status"].shape == (2, 2)

    log_probs = model.anticausal_log_probs(
        x=x,
        age=jnp.zeros((2, 1)),
        sex=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
        tb_status=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    assert "joint" in log_probs
    assert log_probs["joint"].shape == (2,)
