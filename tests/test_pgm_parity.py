import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from pgm.flow_pgm import (
    MorphoMNISTPGM,
    _normal_log_prob,
    _normalize_inverse,
    _set_variable_value,
    monotonic_rational_spline,
)
from pgm.train_pgm import (
    PGMEMA,
    _progress_description,
    epoch_batches,
    make_train_step,
    preprocess,
)
from utils import load_checkpoint, save_checkpoint


def _golden_model():
    model = MorphoMNISTPGM(rngs=nnx.Rngs(0))
    _set_variable_value(model.unnormalized_widths, jnp.array([[-0.4, 0.2, 0.8, -0.1]]))
    _set_variable_value(model.unnormalized_heights, jnp.array([[0.3, -0.5, 0.7, 0.1]]))
    _set_variable_value(model.unnormalized_derivatives, jnp.array([[-0.2, 0.4, 1.0]]))
    _set_variable_value(model.unnormalized_lambdas, jnp.array([[-0.5, 0.0, 0.5, 1.0]]))
    return model


def test_spline_matches_pyro_golden_values():
    model = _golden_model()
    base = jnp.array([[-3.5], [-2.0], [-0.5], [0.3], [2.5], [3.5]])
    expected_output = jnp.array(
        [
            -0.9413755536,
            -0.5707439780,
            -0.2311848998,
            0.1034560204,
            0.8383610249,
            0.9413754940,
        ]
    )
    expected_log_prob = jnp.array(
        [
            -4.1775856018,
            -1.1359809637,
            -0.1776087284,
            -0.0489391088,
            -2.3380627632,
            -4.1775836945,
        ]
    )

    output, _ = model.thickness_forward(base)
    recovered, inverse_logdet = model.thickness_inverse(output)
    actual_log_prob = (
        -0.5 * (recovered**2 + jnp.log(2.0 * jnp.pi)) + inverse_logdet
    ).squeeze(-1)

    np.testing.assert_allclose(
        output.squeeze(-1), expected_output, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(actual_log_prob, expected_log_prob, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(recovered, base, rtol=1e-5, atol=1e-6)


def test_spline_is_identity_outside_bound_and_gradients_are_finite():
    model = _golden_model()
    base = jnp.array([[-4.0], [4.0]])
    spline, logdet = model.thickness_forward(base)
    normalized_expected = 2.0 * jax.nn.sigmoid(base) - 1.0
    np.testing.assert_allclose(spline, normalized_expected, atol=1e-7)
    assert np.isfinite(np.asarray(logdet)).all()

    gradient = jax.grad(
        lambda value: model.thickness_forward(value.reshape(1, 1))[0].sum()
    )(jnp.array(0.25))
    assert np.isfinite(float(gradient))


def test_spline_loss_gradients_match_pyro_golden_values():
    raw = (
        jnp.array([[-0.4, 0.2, 0.8, -0.1]]),
        jnp.array([[0.3, -0.5, 0.7, 0.1]]),
        jnp.array([[-0.2, 0.4, 1.0]]),
        jnp.array([[-0.5, 0.0, 0.5, 1.0]]),
    )
    observed = jnp.array([[-0.8], [-0.2], [0.1], [0.6]])

    def loss_fn(widths, heights, derivatives, lambdas):
        spline_value, normalize_logdet = _normalize_inverse(observed)
        base, spline_logdet = monotonic_rational_spline(
            spline_value,
            jax.nn.softmax(widths, axis=-1),
            jax.nn.softmax(heights, axis=-1),
            jax.nn.softplus(derivatives),
            jax.nn.sigmoid(lambdas),
            inverse=True,
        )
        return -jnp.mean(_normal_log_prob(base) + normalize_logdet + spline_logdet)

    loss, gradients = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3))(*raw)
    expected = (
        [-0.4999862611, 0.3759943247, 0.2576174140, -0.1336254925],
        [0.4485385120, -0.2162882984, -0.3543391228, 0.1220889241],
        [-0.1729327440, 0.0785290301, 0.0974547938],
        [0.0385955535, 0.0, 0.0104345763, 0.0],
    )
    np.testing.assert_allclose(loss, 1.4480490685, rtol=1e-5, atol=1e-6)
    for actual, target in zip(gradients, expected):
        np.testing.assert_allclose(actual.squeeze(0), target, rtol=1e-4, atol=1e-5)


def test_joint_log_prob_is_sum_of_reference_factors():
    model = _golden_model()
    sample = model.sample(8, jax.random.PRNGKey(2))
    log_probs = model.log_prob(
        sample["thickness"], sample["intensity"], sample["digit"]
    )
    np.testing.assert_allclose(
        log_probs["joint"],
        log_probs["digit"] + log_probs["thickness"] + log_probs["intensity"],
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.isfinite(np.asarray(log_probs["joint"])).all()


def test_counterfactual_preserves_abducted_noise_and_dag():
    model = _golden_model()
    sample = model.sample(6, jax.random.PRNGKey(3))
    obs = {key: sample[key] for key in ("thickness", "intensity", "digit")}

    changed_digit = jax.nn.one_hot((jnp.argmax(obs["digit"], axis=-1) + 1) % 10, 10)
    digit_cf = model.counterfactual(obs, {"digit": changed_digit})
    np.testing.assert_allclose(digit_cf["thickness"], obs["thickness"], atol=2e-6)
    np.testing.assert_allclose(digit_cf["intensity"], obs["intensity"], atol=2e-6)

    target_thickness = jnp.zeros_like(obs["thickness"])
    thickness_cf = model.counterfactual(obs, {"thickness": target_thickness})
    observed_noise = model.infer_exogeneous(obs)["intensity_base"]
    counterfactual_noise, _ = model.intensity_inverse(
        thickness_cf["intensity"], thickness_cf["thickness"]
    )
    np.testing.assert_allclose(
        counterfactual_noise, observed_noise, rtol=1e-5, atol=2e-6
    )


def test_reference_ema_warmup_schedule():
    ema = PGMEMA.init_from({"value": jnp.array(0.0)})
    for index in range(101):
        ema.update({"value": jnp.array(float(index))})
    assert ema.step == 101
    assert not ema.initted
    np.testing.assert_allclose(ema.params["value"], 100.0)

    ema.update({"value": jnp.array(1.0)})
    assert ema.initted
    np.testing.assert_allclose(ema.params["value"], 1.0)
    ema.update({"value": jnp.array(3.0)})
    np.testing.assert_allclose(ema.params["value"], 5.0 / 3.0, rtol=1e-6)


class _Dataset:
    def __len__(self):
        return 5

    def make_batch(self, indices, **_):
        return {
            "x": np.zeros((len(indices), 1, 32, 32), dtype=np.float32),
            "thickness": np.asarray(indices, dtype=np.float32),
            "intensity": np.asarray(indices, dtype=np.float32),
            "digit": np.eye(10, dtype=np.float32)[indices],
        }


def test_training_batches_drop_only_incomplete_batch():
    batches = list(
        epoch_batches(
            _Dataset(),
            2,
            shuffle=False,
            drop_last=True,
            rng=np.random.default_rng(0),
        )
    )
    assert len(batches) == 2
    assert all(batch["thickness"].shape == (2, 1) for batch in batches)


def test_progress_description_matches_pytorch_format():
    stats = {
        "loss": 5.6714,
        "logp(digit)": -2.3021,
        "logp(thickness)": -3.0910,
        "logp(intensity)": -0.2784,
    }
    assert _progress_description("train", stats, 6.037) == (
        " => train | loss: 5.6714, logp(digit): -2.3021, "
        "logp(thickness): -3.0910, logp(intensity): -0.2784, grad_norm: 6.037"
    )
    assert _progress_description("eval", stats) == (
        " => eval | loss: 5.6714, logp(digit): -2.3021, "
        "logp(thickness): -3.0910, logp(intensity): -0.2784"
    )


def test_clipped_adamw_training_step_has_reference_metrics():
    model = MorphoMNISTPGM(rngs=nnx.Rngs(7))
    graphdef, _ = nnx.split(model, nnx.Param)
    params = nnx.state(model, nnx.Param).to_pure_dict()
    optimizer = optax.chain(
        optax.clip_by_global_norm(200.0),
        optax.adamw(1e-4, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.1),
    )
    batch = preprocess(
        {
            "x": np.zeros((4, 1, 32, 32), dtype=np.float32),
            "thickness": np.linspace(-0.5, 0.5, 4, dtype=np.float32),
            "intensity": np.linspace(-0.4, 0.4, 4, dtype=np.float32),
            "digit": np.eye(10, dtype=np.float32)[:4],
        }
    )
    updated, _, metrics, grad_norm = make_train_step(graphdef, optimizer)(
        params, optimizer.init(params), batch
    )
    assert float(metrics["loss"]) > 0.0
    assert all(
        float(metrics[key]) < 0.0
        for key in ("logp(digit)", "logp(thickness)", "logp(intensity)")
    )
    assert np.isfinite(float(grad_norm))
    assert any(
        not np.array_equal(np.asarray(before), np.asarray(after))
        for before, after in zip(
            jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(updated)
        )
    )


def test_orbax_checkpoint_round_trip(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    payload = {
        "params": {"value": jnp.array([1.0, 2.0])},
        "ema_params": {"value": jnp.array([1.0, 2.0])},
        "model_params": {"value": jnp.array([1.5, 2.5])},
        "opt_state": {"count": jnp.array(3)},
        "epoch": 2,
        "step": 7,
        "best_loss": 1.25,
        "format_version": 2,
        "hparams": {"widths": [32, 32]},
    }
    save_checkpoint(payload, str(checkpoint_dir), step=7)
    restored = load_checkpoint(str(checkpoint_dir))
    assert restored["format_version"] == 2
    assert restored["step"] == 7
    np.testing.assert_allclose(restored["ema_params"]["value"], [1.0, 2.0])
