import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from causal.image_parent_predictor import MorphoMNISTSupAuxPredictor, _set_variable_value
from training.predictor import (
    _IndexedDataset, WarmupEMA as _WarmupEMA, _run as _main_sup_aux,
    _compute_dtype as _configure_sup_aux_compute_policy,
    _assert_compatible_checkpoint as _sup_aux_assert_compatible_checkpoint,
    _checkpoint_payload as _sup_aux_checkpoint_payload,
    _eval_epoch as _sup_aux_eval_epoch, _loss_and_state as _sup_aux_loss_and_state,
    _make_train_step as _sup_aux_make_train_step, _merge as _sup_aux_merge,
    _validate_scope as _setup_sup_aux_scope, _validate_runtime_device as _validate_sup_aux_runtime_device,
)
from utils import load_checkpoint, save_checkpoint


class _SyntheticMorphoDataset:
    def __init__(self, size: int):
        self.size = int(size)
        self.min_max = {
            "thickness": (-1.0, 1.0),
            "intensity": (-1.0, 1.0),
        }
        self.samples = {
            "thickness": np.zeros((self.size,), dtype=np.float32),
            "intensity": np.zeros((self.size,), dtype=np.float32),
            "digit": np.eye(10, dtype=np.float32)[
                np.zeros((self.size,), dtype=np.int64)
            ],
        }

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "x": np.zeros((1, 32, 32), dtype=np.float32),
            "thickness": np.array(0.0, dtype=np.float32),
            "intensity": np.array(0.0, dtype=np.float32),
            "digit": np.eye(10, dtype=np.float32)[0],
        }

    def make_batch(self, indices, rng=None, shuffle=False):
        del rng, shuffle
        batch_size = len(indices)
        return {
            "x": np.zeros((batch_size, 1, 32, 32), dtype=np.float32),
            "thickness": np.zeros((batch_size, 1), dtype=np.float32),
            "intensity": np.zeros((batch_size, 1), dtype=np.float32),
            "digit": np.eye(10, dtype=np.float32)[
                np.zeros((batch_size,), dtype=np.int64)
            ],
        }


def _zero_predictor(model: MorphoMNISTSupAuxPredictor) -> None:
    for head_name in ("encoder_t", "encoder_i", "encoder_y"):
        head = getattr(model, head_name)
        for layer_name in (
            "conv1",
            "conv2",
            "conv3",
            "conv4",
            "conv5",
            "conv6",
            "fc1",
            "fc2",
        ):
            layer = getattr(head, layer_name)
            kernel = getattr(layer, "kernel", None)
            if kernel is not None:
                _set_variable_value(kernel, jnp.zeros_like(kernel.value))
            bias = getattr(layer, "bias", None)
            if bias is not None:
                _set_variable_value(bias, jnp.zeros_like(bias.value))
        for bn_name in ("bn1", "bn2", "bn3", "bn4", "bn5", "bn6", "bn_fc"):
            bn = getattr(head, bn_name)
            if getattr(bn, "scale", None) is not None:
                _set_variable_value(bn.scale, jnp.ones_like(bn.scale.value))
            if getattr(bn, "bias", None) is not None:
                _set_variable_value(bn.bias, jnp.zeros_like(bn.bias.value))
            _set_variable_value(bn.mean, jnp.zeros_like(bn.mean.value))
            _set_variable_value(bn.var, jnp.ones_like(bn.var.value))


def test_sup_aux_zeroed_model_matches_analytical_loss():
    model = MorphoMNISTSupAuxPredictor(rngs=nnx.Rngs(0))
    _zero_predictor(model)
    graphdef, params, batch_stats = nnx.split(model, nnx.Param, nnx.BatchStat)
    params = params.to_pure_dict()
    batch_stats = batch_stats.to_pure_dict()
    batch = {
        "x": np.zeros((3, 1, 32, 32), dtype=np.float32),
        "thickness": np.array([[-0.5], [0.0], [0.5]], dtype=np.float32),
        "intensity": np.array([[-0.25], [0.0], [0.25]], dtype=np.float32),
        "digit": np.eye(10, dtype=np.float32)[np.zeros((3,), dtype=np.int64)],
    }

    loss, metrics, _, _ = _sup_aux_loss_and_state(
        graphdef, params, batch_stats, batch, training=False
    )
    pred = _sup_aux_merge(graphdef, params, batch_stats).predict(
        x=batch["x"], intensity=batch["intensity"]
    )

    expected_scale = np.log1p(np.e**0.0)
    expected_thickness = np.zeros((3, 1), dtype=np.float32)
    expected_digit = np.full((3, 10), 0.1, dtype=np.float32)
    thickness_log_prob = (
        -0.5 * ((batch["thickness"] / expected_scale) ** 2)
        - np.log(expected_scale)
        - 0.5 * np.log(2.0 * np.pi)
    )
    intensity_log_prob = (
        -0.5 * ((batch["intensity"] / expected_scale) ** 2)
        - np.log(expected_scale)
        - 0.5 * np.log(2.0 * np.pi)
    )
    digit_log_prob = np.log(0.1) * np.ones((3,), dtype=np.float32)
    expected_loss = -np.mean(thickness_log_prob + intensity_log_prob + digit_log_prob)

    np.testing.assert_allclose(pred["thickness"], expected_thickness, atol=1e-6)
    np.testing.assert_allclose(pred["intensity"], expected_thickness, atol=1e-6)
    np.testing.assert_allclose(pred["digit"], expected_digit, atol=1e-6)
    np.testing.assert_allclose(float(loss), float(expected_loss), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        float(metrics["logp(thickness_aux)"]),
        float(np.mean(thickness_log_prob)),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        float(metrics["logp(intensity_aux)"]),
        float(np.mean(intensity_log_prob)),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        float(metrics["logp(digit_aux)"]),
        float(np.mean(digit_log_prob)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_sup_aux_train_step_updates_and_clips_gradients():
    model = MorphoMNISTSupAuxPredictor(rngs=nnx.Rngs(1))
    graphdef, params, batch_stats = nnx.split(model, nnx.Param, nnx.BatchStat)
    params = params.to_pure_dict()
    batch_stats = batch_stats.to_pure_dict()
    optimizer = optax.chain(
        optax.clip_by_global_norm(200.0),
        optax.adamw(1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.1),
    )
    train_step = _sup_aux_make_train_step(graphdef, optimizer)
    batch = {
        "x": np.zeros((4, 1, 32, 32), dtype=np.float32),
        "thickness": np.zeros((4, 1), dtype=np.float32),
        "intensity": np.zeros((4, 1), dtype=np.float32),
        "digit": np.eye(10, dtype=np.float32)[np.zeros((4,), dtype=np.int64)],
    }
    updated_params, updated_batch_stats, opt_state, metrics, grad_norm = train_step(
        params, batch_stats, optimizer.init(params), batch
    )

    assert np.isfinite(float(metrics["loss"]))
    assert np.isfinite(float(grad_norm))
    assert any(
        not np.array_equal(np.asarray(before), np.asarray(after))
        for before, after in zip(
            jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(updated_params)
        )
    )
    assert any(
        not np.array_equal(np.asarray(before), np.asarray(after))
        for before, after in zip(
            jax.tree_util.tree_leaves(batch_stats),
            jax.tree_util.tree_leaves(updated_batch_stats),
        )
    )
    assert opt_state is not None


def test_sup_aux_bf16_compute_keeps_fp32_master_params_and_outputs():
    args = SimpleNamespace(accelerator="tpu", precision="bf16")
    compute_dtype = _configure_sup_aux_compute_policy(args)
    model = MorphoMNISTSupAuxPredictor(compute_dtype=compute_dtype, rngs=nnx.Rngs(2))
    params = nnx.state(model, nnx.Param).to_pure_dict()
    assert all(
        value.dtype == jnp.float32 for value in jax.tree_util.tree_leaves(params)
    )
    assert model.encoder_i.conv1.dtype == jnp.bfloat16
    assert model.encoder_i.bn1.dtype == jnp.bfloat16

    pred = model.predict(
        x=jnp.zeros((2, 1, 32, 32), dtype=jnp.float32),
        intensity=jnp.zeros((2, 1), dtype=jnp.float32),
    )
    assert all(value.dtype == jnp.float32 for value in pred.values())


def test_sup_aux_precision_scope_accepts_accelerator_bf16_only():
    base = dict(dataset="morphomnist", setup="sup_aux", input_channels=1, input_res=32, pad=4)
    _setup_sup_aux_scope(SimpleNamespace(**base, accelerator="gpu", precision="bf16"))
    _setup_sup_aux_scope(SimpleNamespace(**base, accelerator="tpu", precision="bf16"))
    with np.testing.assert_raises_regex(ValueError, "CPU predictor training"):
        _setup_sup_aux_scope(
            SimpleNamespace(**base, accelerator="cpu", precision="bf16")
        )


def test_sup_aux_device_preflight_requires_one_gpu_and_accepts_tpu(monkeypatch):
    gpu = SimpleNamespace(platform="gpu")
    monkeypatch.setattr(jax, "devices", lambda: [gpu, gpu])
    with np.testing.assert_raises_regex(RuntimeError, "one visible GPU"):
        _validate_sup_aux_runtime_device(SimpleNamespace(accelerator="gpu"))

    tpu = SimpleNamespace(platform="tpu")
    monkeypatch.setattr(jax, "devices", lambda: [tpu])
    assert _validate_sup_aux_runtime_device(SimpleNamespace(accelerator="tpu")) is tpu


def test_sup_aux_ema_and_checkpoint_round_trip(tmp_path):
    ema = _WarmupEMA.init_from({"value": jnp.array(0.0)}, {"mean": jnp.array(0.0)})
    for index in range(101):
        ema.update(
            {"value": jnp.array(float(index))}, {"mean": jnp.array(float(index))}
        )
    assert ema.step == 101
    assert not ema.initted
    np.testing.assert_allclose(ema.params["value"], 100.0)

    ema.update({"value": jnp.array(1.0)}, {"mean": jnp.array(1.0)})
    assert ema.initted

    payload = _sup_aux_checkpoint_payload(
        SimpleNamespace(exp_name="demo", setup="sup_aux"),
        model_params={"value": jnp.array([1.0, 2.0])},
        batch_stats={"mean": jnp.array([0.0])},
        ema=ema,
        opt_state={"count": jnp.array(3)},
        epoch=2,
        step=7,
        best_loss=1.25,
    )
    checkpoint_dir = tmp_path / "checkpoint"
    save_checkpoint(payload, str(checkpoint_dir), step=7)
    restored = load_checkpoint(str(checkpoint_dir))
    assert restored["format_version"] == 3
    assert restored["hparams"]["setup"] == "sup_aux"
    _sup_aux_assert_compatible_checkpoint(
        restored,
        {"value": jnp.array([1.0, 2.0])},
        {"mean": jnp.array([0.0])},
    )
    assert restored["step"] == 7
    np.testing.assert_allclose(restored["ema_params"]["value"], ema.params["value"])


def test_sup_aux_end_to_end_training_and_test_only(tmp_path, monkeypatch):
    train_full = _SyntheticMorphoDataset(8)
    valid = _SyntheticMorphoDataset(4)
    test = _SyntheticMorphoDataset(4)
    train_subset = _IndexedDataset(train_full, np.arange(4))

    def fake_build_sup_aux_datasets(_args):
        return {"train": train_full, "valid": valid, "test": test}, train_subset

    monkeypatch.setattr(
        "training.predictor._build_datasets", fake_build_sup_aux_datasets
    )

    class _NoOpSummaryWriter:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def add_scalar(self, *args, **kwargs):
            del args, kwargs

        def add_custom_scalars(self, *args, **kwargs):
            del args, kwargs

        def flush(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr("training.predictor.SummaryWriter", _NoOpSummaryWriter)

    args = SimpleNamespace(
        accelerator="cpu",
        gpu_id="0",
        precision="fp32",
        dataset="morphomnist",
        setup="sup_aux",
        data_dir="unused",
        ckpt_dir=str(tmp_path / "checkpoints"),
        remote_ckpt_dir="",
        exp_name="sup_aux_smoke",
        seed=0,
        deterministic=True,
        testing=False,
        epochs=3,
        bs=4,
        lr=1e-2,
        wd=0.0,
        sup_frac=0.5,
        input_res=32,
        input_channels=1,
        pad=4,
        eval_freq=1,
        widths=[32, 32],
        std_fixed=0.0,
        plot_samples=10000,
        load_path="",
    )

    initial_model = MorphoMNISTSupAuxPredictor(rngs=nnx.Rngs(args.seed))
    initial_graphdef, initial_params, initial_batch_stats = nnx.split(
        initial_model, nnx.Param, nnx.BatchStat
    )
    initial_stats = _sup_aux_eval_epoch(
        initial_graphdef,
        initial_params.to_pure_dict(),
        initial_batch_stats.to_pure_dict(),
        valid,
        args.bs,
        np.random.default_rng(args.seed),
    )
    final_stats = _main_sup_aux(args)

    checkpoint_dir = Path(args.ckpt_dir) / "morphomnist" / args.exp_name / "checkpoints"
    restored = load_checkpoint(str(checkpoint_dir))
    assert restored["format_version"] == 3
    assert restored["hparams"]["setup"] == "sup_aux"
    assert np.isfinite(float(final_stats["loss"]))
    assert float(final_stats["loss"]) < float(initial_stats["loss"])

    test_args = SimpleNamespace(**vars(args))
    test_args.testing = True
    test_args.load_path = str(checkpoint_dir)
    test_stats = _main_sup_aux(test_args)
    assert set(test_stats) == {"thickness_mae", "intensity_mae", "digit_acc"}
    assert np.isfinite(float(test_stats["digit_acc"]))
    assert np.isfinite(float(test_stats["thickness_mae"]))
    assert np.isfinite(float(test_stats["intensity_mae"]))
