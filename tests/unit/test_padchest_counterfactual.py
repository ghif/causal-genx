from __future__ import annotations

from types import SimpleNamespace
import yaml
import pytest
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from causal.cxr_rait_predictor import CxrImageParentPredictor, CxrPretrainedImageParentPredictor
from models.image_vae import HVAE
from causal.cxr_rait_scm import PadChestPGM
from config import CounterfactualTrainingConfig, ExperimentConfig
from data.padchest import (
    PAD_CHEST_HVAE_CONTEXT_DIM,
    PAD_CHEST_SCHEMA,
    adapt_padchest_hvae_context,
)
from training.counterfactual import (
    Bundle,
    _cf_forward,
    _bounded_validation_batches,
    _build_pgm_model,
    _build_predictor_model,
    _dag_variables,
    _validate_artifact_contract,
    _eval_due,
    _make_losses,
    _make_train_step,
    _epoch_batches,
    _parent_slices,
    _split_parent_vector,
)
from training.settings import counterfactual_settings


def test_padchest_schema_slicing_and_dag_variables():
    args = SimpleNamespace(dataset="padchest")
    slices = _parent_slices(args)
    assert "age_at_study" in slices
    assert "sex" in slices
    assert "pediatric" in slices
    assert "tb_status" in slices
    assert "projection" in slices
    assert "view_position" in slices
    assert "study_year" in slices

    total_dim = sum(s.stop - s.start for s in slices.values())
    assert total_dim == 17

    dag_vars = _dag_variables(args)
    assert dag_vars == [
        "age_at_study",
        "sex",
        "pediatric",
        "tb_status",
        "projection",
        "view_position",
        "study_year",
    ]

    dummy_pa = jnp.arange(17, dtype=jnp.float32)[None, :]
    split = _split_parent_vector(args, dummy_pa)
    assert split["age_at_study"].shape == (1, 1)
    assert split["sex"].shape == (1, 2)
    assert split["pediatric"].shape == (1, 1)
    assert split["tb_status"].shape == (1, 1)
    assert split["projection"].shape == (1, 5)
    assert split["view_position"].shape == (1, 6)
    assert split["study_year"].shape == (1, 1)


def test_padchest_context_adapter_is_explicit_and_shape_safe():
    source = np.arange(PAD_CHEST_SCHEMA.encoded_dim, dtype=np.float32)[None, :]
    adapted = adapt_padchest_hvae_context(source, target_dim=PAD_CHEST_HVAE_CONTEXT_DIM)
    assert adapted.shape == (1, PAD_CHEST_HVAE_CONTEXT_DIM)
    np.testing.assert_array_equal(adapted[:, : PAD_CHEST_SCHEMA.encoded_dim], source)
    np.testing.assert_array_equal(adapted[:, PAD_CHEST_SCHEMA.encoded_dim :], 0.0)
    with pytest.raises(ValueError, match="Unsupported PadChest HVAE context dimension"):
        adapt_padchest_hvae_context(source, target_dim=21)
    with pytest.raises(ValueError, match="causal context must have dimension"):
        adapt_padchest_hvae_context(np.zeros((1, 16), dtype=np.float32), target_dim=22)


def test_artifact_context_mismatch_is_rejected_before_restore():
    args = SimpleNamespace(dataset="padchest", input_res=128)
    with pytest.raises(ValueError, match="VAE checkpoint context_dim mismatch: expected 22, found 17"):
        _validate_artifact_contract(
            args,
            "VAE",
            {"dataset": "padchest", "parents_x": list(PAD_CHEST_SCHEMA.variable_names), "context_dim": 17, "input_res": 128},
            context_dim=22,
        )


def test_padchest_model_builders():
    args_scm = SimpleNamespace(
        dataset="padchest",
        pgm_widths=(16, 16),
        seed=42,
    )
    pgm = _build_pgm_model(args_scm, {"widths": (16, 16)})
    assert isinstance(pgm, PadChestPGM)
    samples = pgm.sample(n_samples=2, rng=jax.random.PRNGKey(0))
    assert samples["pa"].shape == (2, 17)

    args_pred_pretrained = SimpleNamespace(
        dataset="padchest",
        input_channels=1,
        input_res=128,
        seed=42,
    )
    pred_pretrained = _build_predictor_model(
        args_pred_pretrained,
        {"freeze_backbone": True, "pretrained_weights_path": "", "width": 32},
        load_pretrained_weights=False,
    )
    assert isinstance(pred_pretrained, CxrPretrainedImageParentPredictor)
    assert pred_pretrained.parent_variables == tuple(PAD_CHEST_SCHEMA.variable_names)

    args_pred_standard = SimpleNamespace(
        dataset="padchest",
        input_channels=1,
        input_res=128,
        seed=42,
    )
    pred_standard = _build_predictor_model(
        args_pred_standard,
        {"freeze_backbone": False, "pretrained": False, "width": 16},
        load_pretrained_weights=False,
    )
    assert isinstance(pred_standard, CxrImageParentPredictor)
    assert pred_standard.parent_variables == tuple(PAD_CHEST_SCHEMA.variable_names)


def test_eval_due_and_bounded_validation_batches():
    assert not _eval_due(epoch=1, eval_freq=5)
    assert not _eval_due(epoch=4, eval_freq=5)
    assert _eval_due(epoch=5, eval_freq=5)
    assert _eval_due(epoch=10, eval_freq=5)

    assert _bounded_validation_batches(SimpleNamespace(model_validation_batches=2)) == 2
    assert _bounded_validation_batches(SimpleNamespace(model_validation_batches=None)) is None
    assert _bounded_validation_batches(SimpleNamespace(model_validation_batches=0)) is None


def test_counterfactual_epoch_prefetch_is_bounded_and_ordered():
    class Dataset:
        def __len__(self):
            return 5

        def make_batch(self, indices, **kwargs):
            del kwargs
            return {"x": np.asarray(indices, dtype=np.float32)}

    batches = list(
        _epoch_batches(
            Dataset(),
            2,
            shuffle=False,
            drop_last=False,
            rng=np.random.default_rng(0),
            prefetch_batches=2,
            prefetch_workers=2,
        )
    )
    assert [batch["x"].tolist() for batch in batches] == [[0.0, 1.0], [2.0, 3.0], [4.0]]


def test_padchest_yaml_config_validation():
    config_path = "configs/padchest_counterfactual_tpu_v6e1.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = ExperimentConfig.model_validate(cfg)
    assert isinstance(exp_cfg.workflow, CounterfactualTrainingConfig)
    assert exp_cfg.workflow.eval_freq == 5
    assert exp_cfg.workflow.model_validation_batches == 1
    assert exp_cfg.workflow.benchmark_steps == 0
    assert exp_cfg.workflow.execution_mode == "single_device"
    assert exp_cfg.workflow.input_prefetch_workers == 8
    assert exp_cfg.workflow.input_prefetch_batches == 4
    assert exp_cfg.workflow.input_stage_mode == "require"
    assert exp_cfg.workflow.scm_checkpoint.endswith("/checkpoints")
    assert exp_cfg.workflow.predictor_checkpoint.endswith("/checkpoints")
    assert exp_cfg.workflow.image_model_checkpoint.endswith("/checkpoints")
    assert exp_cfg.model.context_dim == 22

    settings = counterfactual_settings(exp_cfg)
    assert settings.eval_freq == 5
    assert settings.model_validation_batches == 1
    assert settings.benchmark_steps == 0
    assert settings.input_prefetch_workers == 8
    assert settings.input_prefetch_batches == 4
    assert settings.context_dim == 22
    assert settings.parents_x == list(PAD_CHEST_SCHEMA.variable_names)


class _DummyPadChestVAE(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.dense = nnx.Linear(17, 32, rngs=rngs)
        self.out = nnx.Linear(32, 32 * 32 * 1, rngs=rngs)

    def __call__(self, x, pa=None, beta=1.0, rng=None, training=True, **_):
        batch_size = x.shape[0]
        return {
            "loss": jnp.asarray(1.0, dtype=jnp.float32),
            "elbo": jnp.asarray(1.0, dtype=jnp.float32),
            "nll": jnp.asarray(0.5, dtype=jnp.float32),
            "kl": jnp.asarray(0.5, dtype=jnp.float32),
        }

    def abduct(self, x, pa, t=1.0, rng=None):
        return jnp.zeros((x.shape[0], 16), dtype=jnp.float32)

    def forward_latents(self, latents, pa_maps, rng=None):
        batch_size = latents.shape[0]
        loc = jnp.zeros((batch_size, 32, 32, 1), dtype=jnp.float32)
        scale = jnp.ones((batch_size, 32, 32, 1), dtype=jnp.float32)
        return loc, scale


def test_padchest_context_dispatch_rejects_hvae_source_mismatch():
    args = SimpleNamespace(dataset="padchest")
    with pytest.raises(ValueError, match="refusing implicit padding or truncation"):
        _split_parent_vector(args, jnp.zeros((1, PAD_CHEST_HVAE_CONTEXT_DIM)))


def test_padchest_context_adapter_reaches_context_22_hvae_without_changing_scm():
    rngs = nnx.Rngs(0)
    vae = HVAE(
        input_channels=1, input_res=16,
        enc_arch="16b1d2,8b1d2,4b1d2,2b1d2,1b1",
        dec_arch="1b1,2b1,4b1,8b1,16b1", widths=[4, 8, 16, 32, 64],
        z_dim=2, context_dim=PAD_CHEST_HVAE_CONTEXT_DIM, cond_prior=True,
        q_correction=False, bias_max_res=16, rngs=rngs,
    )
    pgm = PadChestPGM(widths=(8, 8), rngs=rngs)
    predictor = CxrImageParentPredictor(
        variable_specs=PAD_CHEST_SCHEMA.variables, input_channels=1, input_res=16,
        width=4, rngs=rngs,
    )

    def bundle(model, *state_types):
        graphdef, *states = nnx.split(model, nnx.Param, *state_types)
        return Bundle(graphdef, *(state.to_pure_dict() for state in states))

    args = SimpleNamespace(dataset="padchest", input_res=16, context_dim=PAD_CHEST_HVAE_CONTEXT_DIM,
                           alpha=0.1, damping=10.0, elbo_constraint=1.0, cf_particles=1)
    source_pa = jnp.zeros((2, PAD_CHEST_SCHEMA.encoded_dim))
    out = _cf_forward(
        args, bundle(vae), bundle(pgm), bundle(predictor, nnx.BatchStat),
        {"x": jnp.zeros((2, 16, 16, 1)), "pa": source_pa},
        {"tb_status": jnp.ones((2, 1))}, jax.random.PRNGKey(0),
        beta=1.0, alpha=0.1, lmbda=jnp.asarray(0.5), cf_particles=1, training=False,
    )
    assert jnp.isfinite(out["loss"])
    assert out["cfs"]["pa"].shape == source_pa.shape


def test_padchest_counterfactual_forward_and_loss():
    rngs = nnx.Rngs(0)
    vae = _DummyPadChestVAE(rngs)
    pgm = PadChestPGM(widths=(16, 16), rngs=rngs)
    predictor = CxrImageParentPredictor(
        variable_specs=PAD_CHEST_SCHEMA.variables,
        input_channels=1,
        input_res=32,
        width=8,
        rngs=rngs,
    )

    args = SimpleNamespace(
        dataset="padchest",
        input_res=32,
        alpha=0.1,
        damping=10.0,
        elbo_constraint=1.0,
        beta=1.0,
        grad_clip=1.0,
        grad_skip=100.0,
        lr_warmup_steps=0,
        cf_particles=1,
    )

    batch_size = 2
    x = jnp.zeros((batch_size, 32, 32, 1), dtype=jnp.float32)
    pa = jnp.zeros((batch_size, 17), dtype=jnp.float32)
    batch = {"x": x, "pa": pa}

    do_pa = {"tb_status": jnp.ones((batch_size, 1), dtype=jnp.float32)}

    vae_graphdef, vae_params = nnx.split(vae, nnx.Param)
    pgm_graphdef, pgm_params = nnx.split(pgm, nnx.Param)
    pred_graphdef, pred_params, pred_batch_stats = nnx.split(
        predictor, nnx.Param, nnx.BatchStat
    )

    vae_bundle = Bundle(vae_graphdef, vae_params)
    pgm_bundle = Bundle(pgm_graphdef, pgm_params)
    pred_bundle = Bundle(pred_graphdef, pred_params, pred_batch_stats)

    loss_fn = _make_losses(args, vae_bundle, pgm_bundle, pred_bundle)

    lmbda = jnp.asarray(0.5, dtype=jnp.float32)
    rng = jax.random.PRNGKey(42)

    total_loss, out = loss_fn(
        vae_params,
        lmbda,
        batch,
        do_pa,
        rng,
    )

    assert jnp.isfinite(total_loss)
    assert "loss" in out
    assert "elbo" in out
    assert "aux_loss" in out
    assert "cfs" in out
    assert out["cfs"]["x"].shape == (batch_size, 32, 32, 1)


def test_padchest_single_train_step_execution():
    rngs = nnx.Rngs(0)
    vae = _DummyPadChestVAE(rngs)
    pgm = PadChestPGM(widths=(16, 16), rngs=rngs)
    predictor = CxrImageParentPredictor(
        variable_specs=PAD_CHEST_SCHEMA.variables,
        input_channels=1,
        input_res=32,
        width=8,
        rngs=rngs,
    )

    args = SimpleNamespace(
        dataset="padchest",
        input_res=32,
        alpha=0.1,
        damping=10.0,
        elbo_constraint=1.0,
        beta=1.0,
        grad_clip=1.0,
        grad_skip=100.0,
        lr_warmup_steps=0,
        cf_particles=1,
    )

    vae_optimizer = optax.adam(1e-4)
    lambda_optimizer = optax.sgd(1e-2)

    vae_graphdef, vae_params = nnx.split(vae, nnx.Param)
    pgm_graphdef, pgm_params = nnx.split(pgm, nnx.Param)
    pred_graphdef, pred_params, pred_batch_stats = nnx.split(
        predictor, nnx.Param, nnx.BatchStat
    )

    vae_bundle = Bundle(vae_graphdef, vae_params)
    pgm_bundle = Bundle(pgm_graphdef, pgm_params)
    pred_bundle = Bundle(pred_graphdef, pred_params, pred_batch_stats)

    step_fn = _make_train_step(
        args,
        vae_bundle,
        pgm_bundle,
        pred_bundle,
        vae_optimizer,
        lambda_optimizer,
    )

    opt_state = vae_optimizer.init(vae_params)
    lmbda = jnp.asarray(0.0, dtype=jnp.float32)
    lmbda_opt_state = lambda_optimizer.init(lmbda)

    batch_size = 2
    batch = {
        "x": jnp.zeros((batch_size, 32, 32, 1), dtype=jnp.float32),
        "pa": jnp.zeros((batch_size, 17), dtype=jnp.float32),
    }
    do_pa = {"tb_status": jnp.ones((batch_size, 1), dtype=jnp.float32)}

    new_params, new_opt_state, new_lmbda, new_lmbda_opt_state, out = step_fn(
        vae_params,
        opt_state,
        lmbda,
        lmbda_opt_state,
        batch,
        do_pa,
        jax.random.PRNGKey(0),
        jnp.asarray(0),
    )

    assert jnp.isfinite(out["loss"])
    assert float(out["update_skipped"]) == 0.0
    assert float(out["grad_norm"]) >= 0.0
