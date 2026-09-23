import jax
import jax.numpy as jnp
from flax import nnx

from causal.cxr_rait_scm import CxrRaitPGM, PadChestPGM
from causal.flow_scm import MorphoMNISTPGM
from training.scm import _loss


def _params_and_graph(model):
    graphdef, _ = nnx.split(model, nnx.Param)
    return graphdef, nnx.state(model, nnx.Param).to_pure_dict()


def test_scm_loss_uses_model_declared_morphomnist_variables():
    model = MorphoMNISTPGM(widths=(8, 8), rngs=nnx.Rngs(0))
    graphdef, params = _params_and_graph(model)
    batch = {
        "thickness": jnp.zeros((2, 1)),
        "intensity": jnp.zeros((2, 1)),
        "digit": jax.nn.one_hot(jnp.array([0, 1]), 10),
    }

    loss, metrics = _loss(graphdef, params, batch)

    assert loss.shape == ()
    assert set(metrics) == {"loss", "logp(thickness)", "logp(intensity)", "logp(digit)"}


def test_scm_loss_uses_model_declared_cxr_rait_variables():
    model = CxrRaitPGM(widths=(8, 8), rngs=nnx.Rngs(0))
    graphdef, params = _params_and_graph(model)
    batch = {
        "age": jnp.zeros((2, 1)),
        "gender": jax.nn.one_hot(jnp.array([0, 1]), 2),
        "tb_status": jax.nn.one_hot(jnp.array([1, 0]), 2),
    }

    loss, metrics = _loss(graphdef, params, batch)

    assert loss.shape == ()
    assert set(metrics) == {"loss", "logp(age)", "logp(gender)", "logp(tb_status)"}


def test_scm_loss_uses_model_declared_padchest_variables():
    model = PadChestPGM(widths=(8, 8), rngs=nnx.Rngs(0))
    graphdef, params = _params_and_graph(model)
    batch = {
        "age_at_study": jnp.zeros((2, 1)),
        "sex": jax.nn.one_hot(jnp.array([0, 1]), 2),
        "pediatric": jnp.array([[0.0], [1.0]]),
        "tb_status": jnp.array([[1.0], [0.0]]),
        "projection": jax.nn.one_hot(jnp.array([0, 1]), 5),
        "view_position": jax.nn.one_hot(jnp.array([0, 1]), 6),
        "study_year": jnp.zeros((2, 1)),
    }

    loss, metrics = _loss(graphdef, params, batch)

    assert loss.shape == ()
    assert set(metrics) == {
        "loss",
        "logp(age_at_study)",
        "logp(sex)",
        "logp(pediatric)",
        "logp(tb_status)",
        "logp(projection)",
        "logp(view_position)",
        "logp(study_year)",
    }
