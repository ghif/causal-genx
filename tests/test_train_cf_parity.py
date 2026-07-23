import os
import sys
import types
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax

from training.counterfactual_support import (
    batch_progress_kwargs,
    clip_counterfactual_grads,
    damped_lagrangian_loss,
    epoch_progress_kwargs,
    format_checkpoint_summary,
    format_checkpoint_validation_summary,
    format_run_summary,
    inherit_image_training_config,
    intervention_progress_kwargs,
    format_torch_style_eval_progress,
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


def test_progress_bars_match_compact_pytorch_policy():
    assert batch_progress_kwargs("valid") == {
        "desc": "valid",
        "leave": False,
        "mininterval": 0.1,
    }
    assert intervention_progress_kwargs()["leave"] is False
    assert epoch_progress_kwargs() == {
        "desc": "epochs",
        "leave": True,
        "mininterval": 0.5,
    }


def test_compact_run_and_exact_checkpoint_summaries():
    args = SimpleNamespace(
        exp_name="cf-test",
        bs=32,
        resolved_vae_path="gs://bucket/vae/checkpoints/112320",
        resolved_pgm_path="gs://bucket/pgm/checkpoints/637500",
        resolved_predictor_path="gs://bucket/predictor/checkpoints/298125",
        resolved_resume_path="gs://bucket/cf/checkpoints/1875",
        resolved_resume_trusted_incomplete=True,
    )

    assert format_run_summary(args, ["exp_name", "bs", "missing"]) == (
        "run | exp_name=cf-test, bs=32"
    )
    assert format_checkpoint_summary(args) == (
        "checkpoint | vae=gs://bucket/vae/checkpoints/112320, "
        "pgm=gs://bucket/pgm/checkpoints/637500, "
        "predictor=gs://bucket/predictor/checkpoints/298125, "
        "resume=gs://bucket/cf/checkpoints/1875 (trusted incomplete)"
    )
    assert format_checkpoint_validation_summary(
        {"loss": 0.4797, "nll": 0.3667, "kl": 0.1130},
        extra_keys=("nll", "kl"),
    ) == "=> eval | loss: 0.4797, nll: 0.3667, kl: 0.1130"
    assert format_torch_style_eval_progress(
        {
            "loss": -3.9613,
            "logp(thickness_aux)": 1.9663,
            "logp(intensity_aux)": 2.0167,
            "logp(digit_aux)": -0.0217,
        },
        ("logp(thickness_aux)", "logp(intensity_aux)", "logp(digit_aux)"),
    ) == (
        "=> eval | loss: -3.9613, logp(thickness_aux): 1.9663, "
        "logp(intensity_aux): 2.0167, logp(digit_aux): -0.0217"
    )


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


def test_counterfactual_step_rejects_explosive_gradients(monkeypatch):
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.read_csv = lambda *args, **kwargs: None
    pandas_stub.DataFrame = object
    monkeypatch.setitem(sys.modules, "pandas", pandas_stub)

    from training import counterfactual as train_cf

    def loss_fn(vae_params, lmbda, *_):
        loss = 10.0 * (vae_params["weight"] + lmbda)
        return loss, {"loss": loss}

    monkeypatch.setattr(train_cf, "_make_losses", lambda *_: loss_fn)
    args = SimpleNamespace(grad_clip=1.0, grad_skip=0.5, lr_warmup_steps=0)
    optimizer = optax.sgd(0.1)
    lambda_optimizer = optax.sgd(0.1)
    params = {"weight": jnp.asarray(1.0)}
    lmbda = jnp.asarray(1.0)
    step = train_cf._make_train_step(
        args, None, None, None, optimizer, lambda_optimizer
    )

    new_params, _, new_lmbda, _, out = step(
        params,
        optimizer.init(params),
        lmbda,
        lambda_optimizer.init(lmbda),
        None,
        None,
        jax.random.PRNGKey(0),
        jnp.asarray(100),
    )

    assert float(out["grad_norm"]) > args.grad_skip
    assert float(out["grad_clipped"]) == 1.0
    assert float(out["update_skipped"]) == 1.0
    np.testing.assert_allclose(new_params["weight"], 1.0)
    np.testing.assert_allclose(new_lmbda, 1.0)


def test_counterfactual_vae_learning_rate_warmup_uses_global_step():
    from training import counterfactual as train_cf

    np.testing.assert_allclose(train_cf._cf_lr_scale(0, 100), 0.0)
    np.testing.assert_allclose(train_cf._cf_lr_scale(50, 100), 0.5)
    np.testing.assert_allclose(train_cf._cf_lr_scale(100, 100), 1.0)
    np.testing.assert_allclose(train_cf._cf_lr_scale(200, 100), 1.0)
    np.testing.assert_allclose(train_cf._cf_lr_scale(0, 0), 1.0)


def test_counterfactual_execution_mode_selects_single_or_replicated_tpu(monkeypatch):
    from training import counterfactual as train_cf

    monkeypatch.setattr(jax, "local_device_count", lambda: 1)
    assert not train_cf._use_tpu_replication(SimpleNamespace(accelerator="gpu", execution_mode="auto"))
    assert not train_cf._use_tpu_replication(SimpleNamespace(accelerator="tpu", execution_mode="auto"))
    with np.testing.assert_raises_regex(ValueError, "requires accelerator=tpu with multiple local devices"):
        train_cf._use_tpu_replication(SimpleNamespace(accelerator="tpu", execution_mode="replicated"))

    monkeypatch.setattr(jax, "local_device_count", lambda: 4)
    assert train_cf._use_tpu_replication(SimpleNamespace(accelerator="tpu", execution_mode="auto"))
    assert train_cf._use_tpu_replication(SimpleNamespace(accelerator="tpu", execution_mode="replicated"))
    assert not train_cf._use_tpu_replication(SimpleNamespace(accelerator="tpu", execution_mode="single_device"))


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

    inherit_image_training_config(args, checkpoint_hparams)

    assert args.lr == 1e-4
    assert args.beta == 2.0
    assert args.wd == 0.1
    assert args.betas == [0.9, 0.9]
    assert args.grad_clip == 350.0
    assert args.grad_skip == 500.0


class _TinyValidDataset:
    def __len__(self):
        return 2

    def make_batch(self, indices, **_):
        batch_size = len(indices)
        return {
            "x": np.full((batch_size, 1, 32, 32), 255.0, dtype=np.float32),
            "pa": np.zeros((batch_size, 12), dtype=np.float32),
        }


class _DummyTqdm:
    def __init__(self, iterable, **_):
        self._iter = iter(iterable)
        self.descriptions = []

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)

    def set_description(self, description):
        self.descriptions.append(description)


class _DummyPredictor:
    def __init__(self):
        self.calls = 0

    def eval(self):
        return self

    def predict(self, **cfs):
        self.calls += 1
        batch_size = int(np.asarray(cfs["x"]).shape[0])
        digit = jnp.eye(10, dtype=jnp.float32)[jnp.zeros((batch_size,), dtype=jnp.int32)]
        return {
            "thickness": jnp.zeros((batch_size, 1), dtype=jnp.float32),
            "intensity": jnp.zeros((batch_size, 1), dtype=jnp.float32),
            "digit": digit,
        }


def test_eval_split_rejects_nan_batches(monkeypatch):
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.read_csv = lambda *args, **kwargs: None
    pandas_stub.DataFrame = object
    monkeypatch.setitem(sys.modules, "pandas", pandas_stub)

    from training import counterfactual as train_cf

    args = SimpleNamespace(
        bs=1,
        seed=7,
        input_res=32,
        alpha=0.1,
        do_pa=None,
        parents_x=["thickness", "intensity", "digit"],
    )
    datasets = {"valid": _TinyValidDataset()}
    state = {"vae_params": None, "lmbda": jnp.asarray(0.0, dtype=jnp.float32)}
    train_samples = {
        "thickness": np.zeros((2, 1), dtype=np.float32),
        "intensity": np.zeros((2, 1), dtype=np.float32),
        "digit": np.eye(10, dtype=np.float32)[np.zeros((2,), dtype=np.int32)],
    }
    predictor = _DummyPredictor()
    predictor_bundle = SimpleNamespace(materialize=lambda: predictor)

    finite_out = {
        "loss": jnp.asarray(1.0, dtype=jnp.float32),
        "aux_loss": jnp.asarray(2.0, dtype=jnp.float32),
        "elbo": jnp.asarray(3.0, dtype=jnp.float32),
        "nll": jnp.asarray(4.0, dtype=jnp.float32),
        "kl": jnp.asarray(5.0, dtype=jnp.float32),
        "cfs": {
            "x": jnp.zeros((1, 32, 32, 1), dtype=jnp.float32),
            "thickness": jnp.zeros((1, 1), dtype=jnp.float32),
            "intensity": jnp.zeros((1, 1), dtype=jnp.float32),
            "digit": jnp.eye(10, dtype=jnp.float32)[jnp.zeros((1,), dtype=jnp.int32)],
        },
        "var_cf_x": None,
    }
    nan_out = {
        "loss": jnp.asarray(jnp.nan, dtype=jnp.float32),
        "aux_loss": jnp.asarray(jnp.nan, dtype=jnp.float32),
        "elbo": jnp.asarray(jnp.nan, dtype=jnp.float32),
        "nll": jnp.asarray(jnp.nan, dtype=jnp.float32),
        "kl": jnp.asarray(jnp.nan, dtype=jnp.float32),
        "cfs": finite_out["cfs"],
        "var_cf_x": None,
    }
    outputs = iter([finite_out, nan_out])

    monkeypatch.setattr(train_cf, "preprocess_batch", lambda args, raw_batch, compact_pa=True: raw_batch)
    monkeypatch.setattr(train_cf, "_choose_intervention", lambda args, dag_vars: "thickness")
    monkeypatch.setattr(
        train_cf,
        "_make_intervention",
        lambda args, batch, do_k, train_samples, train: {
            "thickness": jnp.zeros((batch["x"].shape[0], 1), dtype=jnp.float32)
        },
    )
    monkeypatch.setattr(train_cf, "tqdm", lambda iterable, **kwargs: _DummyTqdm(iterable, **kwargs))
    monkeypatch.setattr(
        train_cf,
        "_predictor_metrics",
        lambda args, dataset, preds, targets: {"digit_acc": 1.0, "thickness_mae": 0.0},
    )

    def eval_step(*_):
        return next(outputs)

    with np.testing.assert_raises_regex(
        FloatingPointError, "Non-finite values produced during valid evaluation batch 1"
    ):
        train_cf._eval_split(
            args,
            "valid",
            datasets,
            state,
            None,
            None,
            predictor_bundle,
            eval_step,
            train_samples,
            np.random.default_rng(0),
        )


def test_predictor_preflight_normalizes_images_before_likelihood(monkeypatch):
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.read_csv = lambda *args, **kwargs: None
    pandas_stub.DataFrame = object
    monkeypatch.setitem(sys.modules, "pandas", pandas_stub)

    from training import counterfactual as train_cf

    args = SimpleNamespace(
        bs=1,
        seed=7,
        resolved_predictor_path="gs://bucket/predictor/checkpoints/298125",
    )
    dataset = _TinyValidDataset()
    class _FinitePredictor(_DummyPredictor):
        def __init__(self):
            super().__init__()
            self.preflight_calls = 0

        def model_anticausal(self, **batch):
            self.preflight_calls += 1
            x = np.asarray(batch["x"])
            assert x.min() >= -1.0
            assert x.max() <= 1.0
            size = x.shape[0]
            return {
                "joint": jnp.ones((size,), dtype=jnp.float32),
                "thickness_aux": jnp.ones((size,), dtype=jnp.float32) * 2.0,
                "intensity_aux": jnp.ones((size,), dtype=jnp.float32) * 3.0,
                "digit_aux": jnp.ones((size,), dtype=jnp.float32) * 4.0,
            }

    predictor = _FinitePredictor()
    bundle = SimpleNamespace(materialize=lambda: predictor)

    monkeypatch.setattr(train_cf, "tqdm", lambda iterable, **kwargs: _DummyTqdm(iterable, **kwargs))

    metrics_calls = []

    def fake_metrics(args, dataset, preds, targets):
        metrics_calls.append((preds, targets))
        return {"digit_acc": 1.0, "thickness_mae": 0.0, "intensity_mae": 0.0}

    monkeypatch.setattr(train_cf, "_predictor_metrics", fake_metrics)

    train_cf._validate_predictor_checkpoint(args, bundle, dataset)

    assert predictor.preflight_calls == 2
    assert predictor.calls == 2
    assert metrics_calls


def test_dataset_normalization_report_matches_pytorch_format(monkeypatch, capsys):
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.read_csv = lambda *args, **kwargs: None
    pandas_stub.DataFrame = object
    monkeypatch.setitem(sys.modules, "pandas", pandas_stub)

    from training import counterfactual as train_cf

    class _Dataset:
        norm = "[-1,1]"
        min_max = {
            "thickness": [0.87598526, 6.255515],
            "intensity": [66.601204, 254.90317],
        }

        def __init__(self, size):
            self.size = size

        def __len__(self):
            return self.size

    train_cf._print_dataset_normalization(
        {
            "train": _Dataset(60000),
            "valid": _Dataset(10000),
            "test": _Dataset(10000),
        }
    )

    assert capsys.readouterr().out == (
        "thickness normalization: [-1,1]\n"
        "max: 6.255515, min: 0.87598526\n"
        "intensity normalization: [-1,1]\n"
        "max: 254.90317, min: 66.601204\n"
        "#samples: 60000\n\n"
        "thickness normalization: [-1,1]\n"
        "max: 6.255515, min: 0.87598526\n"
        "intensity normalization: [-1,1]\n"
        "max: 254.90317, min: 66.601204\n"
        "#samples: 10000\n\n"
        "thickness normalization: [-1,1]\n"
        "max: 6.255515, min: 0.87598526\n"
        "intensity normalization: [-1,1]\n"
        "max: 254.90317, min: 66.601204\n"
        "#samples: 10000\n\n"
    )
