"""Stage 4: counterfactual image-model fine-tuning.

This stage freezes three upstream artifacts: the SCM supplies intervened parent
variables, the predictor supplies an auxiliary image-consistency likelihood,
and the pretrained VAE supplies the initial image mechanism. Only VAE weights
and the Lagrange multiplier are optimized here.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from config import CounterfactualTrainingConfig, ExperimentConfig
from contracts import CausalGraphSpec, VariableKind
from data.cxr_rait import CXR_RAIT_SCHEMA, cxr_rait
from data.morphomnist import MORPHOMNIST_SCHEMA, morphomnist
from data.padchest import (
    PAD_CHEST_HVAE_CONTEXT_DIM,
    PAD_CHEST_HVAE_CONTEXT_EXTENSION_DIM,
    PAD_CHEST_SCHEMA,
    PAD_CHEST_SCHEMA_V2,
    PAD_CHEST_SOURCE_CONTEXT_DIM,
    padchest,
)
from models.image_vae import HVAE
from .counterfactual_support import (
    clip_counterfactual_grads,
    damped_lagrangian_loss,
    format_checkpoint_summary,
    format_checkpoint_validation_summary,
    format_run_summary,
    inherit_image_training_config,
    set_module_training_mode,
)
from causal.cxr_predictor import (
    CxrImageParentPredictor,
    CxrPretrainedImageParentPredictor,
    CxrRaitPretrainedPredictor,
    CxrRaitSupAuxPredictor,
)
from causal.cxr_rait_scm import CxrRaitPGM, PadChestPGM
from causal.generic_scm import SchemaDrivenSCM
from causal.flow_scm import MorphoMNISTPGM
from causal.image_parent_predictor import MorphoMNISTSupAuxPredictor
from training.image_loop import _first_local_replica, _replicate, _shard_batch, _unreplicate, init_state, preprocess_batch
from utils import (
    BackgroundArtifactWriter,
    EMA,
    SummaryWriter,
    checkpoint_is_complete,
    checkpoint_root_dir,
    ensure_dir,
    experiment_run_dir,
    load_checkpoint_with_path,
    open_file,
    seed_all,
    sync_file,
    tree_copy,
)

from .common import stage_run_dir, validate_stage_artifacts
from .settings import CounterfactualSettings, counterfactual_settings


def output_dir(config: ExperimentConfig):
    return stage_run_dir(config) / "cf"


def _run_arguments(config: ExperimentConfig) -> CounterfactualSettings:
    return counterfactual_settings(config)


def setup_logging(args):
    ensure_dir(args.save_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.save_dir, "trainlog.txt"), mode="a"),
        ],
        force=True,
    )
    return logging.getLogger("causal-genx-cf")


def _validate_runtime_device(args) -> None:
    if args.accelerator == "gpu":
        devices = jax.devices()
        gpu_devices = [device for device in devices if device.platform in {"gpu", "cuda"}]
        if not gpu_devices:
            raise RuntimeError(
                "--accelerator gpu requested, but JAX found no CUDA GPU. Install a CUDA-enabled "
                "JAX build compatible with the NVIDIA driver and CUDA runtime."
            )
        if len(gpu_devices) != 1:
            raise RuntimeError(
                f"Counterfactual finetuning requires one visible GPU, found {len(gpu_devices)}. "
                "Set --gpu_id or CUDA_VISIBLE_DEVICES to a single device."
            )
        device = gpu_devices[0]
    elif args.accelerator == "cpu":
        try:
            devices = jax.devices("cpu")
        except Exception:
            devices = jax.devices()
        cpu_devices = [device for device in devices if device.platform == "cpu"]
        if not cpu_devices:
            raise RuntimeError("--accelerator cpu requested, but JAX found no CPU device")
        device = cpu_devices[0]
    else:
        try:
            devices = jax.devices(args.accelerator)
        except Exception:
            devices = jax.devices()
        matching_devices = [device for device in devices if device.platform == args.accelerator]
        if not matching_devices:
            raise RuntimeError(
                f"--accelerator {args.accelerator} requested, but JAX devices are {devices}"
            )
        device = matching_devices[0]
    print(f"JAX device preflight passed: platform={device.platform} device={device}")


def _configure_compute_policy(args) -> None:
    if args.accelerator == "cpu" and args.precision != "fp32":
        raise ValueError("CPU counterfactual finetuning requires --precision fp32")
    if args.accelerator in {"gpu", "tpu"} and args.precision == "bf16":
        matmul_precision = "default"
    else:
        matmul_precision = "highest"
    jax.config.update("jax_default_matmul_precision", matmul_precision)
    print(
        "JAX compute policy: "
        f"precision={args.precision} master_params=fp32 vae_cf_compute=fp32 "
        f"matmul_precision={matmul_precision}"
    )


def _vae_compute_params(args, params):
    del args
    # The damped ELBO objective is too sensitive for BF16 VAE parameters.
    return params


def loginfo(title: str, logger: Any, stats: Dict[str, Any]) -> None:
    logger.info(f"{title} | " + " - ".join(f"{k}: {v:.4f}" for k, v in stats.items()))


def log_run_summary(logger: Any, args, keys: List[str]) -> None:
    logger.info(format_run_summary(args, keys))


def log_checkpoint_summary(logger: Any, args) -> None:
    logger.info(format_checkpoint_summary(args))


_SCHEMAS: dict[str, CausalGraphSpec] = {
    MORPHOMNIST_SCHEMA.dataset_id: MORPHOMNIST_SCHEMA,
    CXR_RAIT_SCHEMA.dataset_id: CXR_RAIT_SCHEMA,
    PAD_CHEST_SCHEMA.dataset_id: PAD_CHEST_SCHEMA,
}


def _schema_for_dataset(dataset: str, configured: CausalGraphSpec | None = None, schema_version: str | int | None = None) -> CausalGraphSpec:
    if isinstance(configured, CausalGraphSpec):
        return configured
    if dataset == "padchest" and (str(schema_version) == "2" or (configured is not None and getattr(configured, "version", "") == "2")):
        return PAD_CHEST_SCHEMA_V2
    try:
        return _SCHEMAS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unsupported counterfactual dataset {dataset!r}; available={sorted(_SCHEMAS)}") from exc


def _schema_for_args(args) -> CausalGraphSpec:
    if isinstance(getattr(args, "schema", None), CausalGraphSpec):
        return args.schema
    return _schema_for_dataset(
        getattr(args, "dataset", getattr(args, "dataset_id", "morphomnist")),
        configured=getattr(args, "schema", None),
        schema_version=getattr(args, "schema_version", getattr(args, "causal_schema_version", None)),
    )


def _parent_slices(args) -> dict[str, slice]:
    schema = _schema_for_args(args)
    dims = {variable.name: variable.encoded_dim for variable in schema.variables}
    out: dict[str, slice] = {}
    offset = 0
    for name in getattr(args, "parents_x", list(schema.variable_names)):
        width = int(dims[name])
        out[name] = slice(offset, offset + width)
        offset += width
    return out


def _split_parent_vector(args, pa: jax.Array) -> dict[str, jax.Array]:
    expected_dim = _schema_for_args(args).encoded_dim
    actual_dim = int(pa.shape[-1])
    if actual_dim != expected_dim:
        raise ValueError(
            f"{args.dataset} causal parent vector has dimension {actual_dim}, "
            f"expected {expected_dim}; refusing implicit padding or truncation"
        )
    return {name: pa[:, slc] for name, slc in _parent_slices(args).items()}


def _dag_variables(args) -> list[str]:
    return list(_schema_for_args(args).variable_names)


def _build_datasets(args):
    if args.dataset == "cxr_rait":
        return cxr_rait(args)
    if args.dataset == "padchest":
        return padchest(args)
    if args.dataset == "morphomnist":
        return morphomnist(args)
    raise ValueError(f"Counterfactual finetuning does not support dataset={args.dataset!r}")


def _build_pgm_model(args, hparams: Dict[str, Any]):
    rngs = nnx.Rngs(args.seed)
    widths = hparams.get("widths", [32, 32])
    if args.dataset == "cxr_rait":
        return CxrRaitPGM(widths=widths, rngs=rngs)
    if args.dataset == "padchest":
        if hparams.get("causal_schema_version") == "2" or str(getattr(args, "schema_version", "")) == "2":
            schema = _schema_for_args(args)
            return SchemaDrivenSCM(schema=schema, widths=widths, rngs=rngs)
        return PadChestPGM(widths=widths, rngs=rngs)
    if args.dataset == "morphomnist":
        return MorphoMNISTPGM(widths=widths, rngs=rngs)
    raise ValueError(f"Counterfactual SCM restore does not support dataset={args.dataset!r}")


def _build_predictor_model(args, hparams: Dict[str, Any], *, load_pretrained_weights: bool = False):
    rngs = nnx.Rngs(args.seed)
    if args.dataset == "cxr_rait":
        uses_pretrained_backbone = bool(
            hparams.get("freeze_backbone", False)
            or hparams.get("type") == "finetune-predictor"
            or hparams.get("pretrained", False)
        )
        if uses_pretrained_backbone:
            return CxrRaitPretrainedPredictor(
                input_channels=hparams.get("input_channels", args.input_channels),
                input_res=hparams.get("input_res", args.input_res),
                width=hparams.get("width", 32),
                std_fixed=hparams.get("std_fixed", 0.0),
                freeze_backbone=hparams.get("freeze_backbone", True),
                weights_path=hparams.get("pretrained_weights_path", ""),
                dropout_rate=hparams.get("dropout_rate", 0.0),
                label_smoothing=hparams.get("label_smoothing", 0.0),
                load_pretrained_weights=load_pretrained_weights,
                rngs=rngs,
            )
        return CxrRaitSupAuxPredictor(
            input_channels=hparams.get("input_channels", args.input_channels),
            input_res=hparams.get("input_res", args.input_res),
            width=hparams.get("width", 16),
            std_fixed=hparams.get("std_fixed", 0.0),
            rngs=rngs,
        )
    if args.dataset == "padchest":
        uses_pretrained_backbone = bool(
            hparams.get("freeze_backbone", False)
            or hparams.get("type") == "finetune-predictor"
            or hparams.get("pretrained", False)
        )
        schema = _schema_for_args(args)
        if uses_pretrained_backbone:
            return CxrPretrainedImageParentPredictor(
                variable_specs=schema.variables,
                input_channels=hparams.get("input_channels", args.input_channels),
                input_res=hparams.get("input_res", args.input_res),
                width=hparams.get("width", 32),
                std_fixed=hparams.get("std_fixed", 0.0),
                freeze_backbone=hparams.get("freeze_backbone", True),
                weights_path=hparams.get("pretrained_weights_path", ""),
                dropout_rate=hparams.get("dropout_rate", 0.0),
                label_smoothing=hparams.get("label_smoothing", 0.0),
                load_pretrained_weights=load_pretrained_weights,
                rngs=rngs,
            )
        return CxrImageParentPredictor(
            variable_specs=schema.variables,
            input_channels=hparams.get("input_channels", args.input_channels),
            input_res=hparams.get("input_res", args.input_res),
            width=hparams.get("width", 16),
            std_fixed=hparams.get("std_fixed", 0.0),
            rngs=rngs,
        )
    if args.dataset == "morphomnist":
        return MorphoMNISTSupAuxPredictor(
            input_channels=hparams.get("input_channels", args.input_channels),
            input_res=hparams.get("input_res", args.input_res),
            width=hparams.get("width", 8),
            std_fixed=hparams.get("std_fixed", 0.0),
            rngs=rngs,
        )
    raise ValueError(f"Counterfactual predictor restore does not support dataset={args.dataset!r}")


@jax.tree_util.register_pytree_node_class
@dataclass
class Bundle:
    graphdef: Any
    params: Any
    batch_stats: Any = None

    def materialize(self):
        states = [self.params]
        if self.batch_stats is not None:
            states.append(self.batch_stats)
        return nnx.merge(self.graphdef, *states)

    def tree_flatten(self):
        children = (self.params, self.batch_stats)
        return children, self.graphdef

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        params, batch_stats = children
        return cls(aux_data, params, batch_stats)


def _use_tpu_replication(args) -> bool:
    """Resolve counterfactual execution mode for the local TPU topology."""
    requested_mode = getattr(args, "execution_mode", "auto")
    multi_tpu_available = args.accelerator == "tpu" and jax.local_device_count() > 1
    if requested_mode == "replicated" and not multi_tpu_available:
        raise ValueError("execution_mode=replicated requires accelerator=tpu with multiple local devices")
    return multi_tpu_available and requested_mode != "single_device"


def _portable_cf_state(state: Dict[str, Any], *, replicated: bool) -> Dict[str, Any]:
    """Return ordinary trees for single-device evaluation and checkpointing."""
    if not replicated:
        return state
    portable = dict(state)
    for key in ("vae_params", "ema_params", "opt_state", "lmbda", "lambda_opt_state"):
        portable[key] = _unreplicate(state[key])
    portable["ema"] = EMA(
        params=_unreplicate(state["ema"].params),
        decay=state["ema"].decay,
        update_after_step=state["ema"].update_after_step,
        step=state["ema"].step,
        initted=state["ema"].initted,
    )
    return portable


def _host_lmbda(state: Dict[str, Any], *, replicated: bool) -> float:
    """Read the synchronized scalar multiplier for host logging."""
    value = _first_local_replica(state["lmbda"]) if replicated else state["lmbda"]
    return float(np.asarray(value))


def _restore_args(args, checkpoint):
    saved = checkpoint.get("hparams", {})
    preserved = {
        "accelerator": args.accelerator,
        "gpu_id": args.gpu_id,
        "precision": args.precision,
        "data_dir": args.data_dir,
        "load_path": args.load_path,
        "testing": args.testing,
        "remote_ckpt_dir": args.remote_ckpt_dir,
        "pgm_path": args.pgm_path,
        "predictor_path": args.predictor_path,
        "vae_path": args.vae_path,
        "trust_incomplete_checkpoint": args.trust_incomplete_checkpoint,
        "model_validation_batches": args.model_validation_batches,
        "input_prefetch_workers": args.input_prefetch_workers,
        "input_prefetch_batches": args.input_prefetch_batches,
        "input_stage_mode": args.input_stage_mode,
        "input_stage_dir": args.input_stage_dir,
        "input_stage_manifest": args.input_stage_manifest,
        "input_stage_max_items": args.input_stage_max_items,
        "input_stage_max_bytes": args.input_stage_max_bytes,
        "input_stage_size_sample_items": args.input_stage_size_sample_items,
        "input_stage_workers": args.input_stage_workers,
        "execution_mode": args.execution_mode,
        "drop_remainder": args.drop_remainder,
    }
    for name in ("vae", "pgm", "predictor", "resume"):
        for suffix in ("path", "trusted_incomplete"):
            key = f"resolved_{name}_{suffix}"
            if hasattr(args, key):
                preserved[key] = getattr(args, key)
    for key, value in saved.items():
        if hasattr(args, key):
            setattr(args, key, value)
    for key, value in preserved.items():
        setattr(args, key, value)


def _assert_tree_compatible(name: str, checkpoint: Dict[str, Any], tree: Any, key: str) -> None:
    if jax.tree_util.tree_structure(checkpoint[key]) != jax.tree_util.tree_structure(tree):
        raise ValueError(f"{name} checkpoint parameter structure does not match the current model")


def _load_runtime_checkpoint(
    args,
    path: str,
    name: str,
    template: Optional[Dict[str, Any]] = None,
    partial_restore: bool = False,
):
    fallback_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    checkpoint, resolved_path = load_checkpoint_with_path(
        path,
        template=template,
        fallback_sharding=fallback_sharding,
        allow_incomplete=args.trust_incomplete_checkpoint,
        partial_restore=partial_restore,
    )
    trusted_incomplete = (
        args.trust_incomplete_checkpoint and not checkpoint_is_complete(resolved_path)
    )
    setattr(args, f"resolved_{name}_path", resolved_path)
    setattr(args, f"resolved_{name}_trusted_incomplete", trusted_incomplete)
    suffix = " (trusted incomplete)" if trusted_incomplete else ""
    print(f"Loaded {name} checkpoint: {resolved_path}{suffix}")
    return checkpoint


def _checkpoint_root(path: str) -> str:
    path = path.rstrip("/")
    if path.split("/")[-1].isdigit():
        return path.rsplit("/", 1)[0]
    return path


def _load_checkpoint_hparams(path: str) -> Dict[str, Any]:
    with open_file(f"{_checkpoint_root(path)}/hparams.json", "r") as f:
        return json.load(f)


def _preprocess_counterfactual_batch(args, raw_batch):
    """Keep SCM/predictor batches causal, adapting only at the HVAE boundary."""
    source_args = copy.copy(args)
    source_args.context_dim = _schema_for_args(args).encoded_dim
    return preprocess_batch(source_args, raw_batch, compact_pa=True)


def _hvae_context(args, pa: jax.Array, target_dim: int) -> jax.Array:
    schema = _schema_for_args(args)
    if args.dataset != "padchest" or str(getattr(args, "schema_version", "")) == "2" or getattr(schema, "version", "") == "2":
        if int(pa.shape[-1]) != int(target_dim):
            raise ValueError(
                "Parent encoding dimension does not match the HVAE context: "
                f"expected {target_dim}, got {pa.shape[-1]}"
            )
        return pa
    if int(pa.shape[-1]) != PAD_CHEST_SOURCE_CONTEXT_DIM:
        raise ValueError(
            f"PadChest causal context must have dimension {PAD_CHEST_SOURCE_CONTEXT_DIM}, got {pa.shape[-1]}"
        )
    if int(target_dim) == PAD_CHEST_SOURCE_CONTEXT_DIM:
        return pa
    if int(target_dim) != PAD_CHEST_HVAE_CONTEXT_DIM:
        raise ValueError(
            "Unsupported PadChest HVAE context dimension: "
            f"expected {PAD_CHEST_SOURCE_CONTEXT_DIM} or {PAD_CHEST_HVAE_CONTEXT_DIM}, got {target_dim}"
        )
    extension = jnp.zeros(
        pa.shape[:-1] + (PAD_CHEST_HVAE_CONTEXT_EXTENSION_DIM,), dtype=pa.dtype
    )
    return jnp.concatenate((pa, extension), axis=-1)


def _expand_parents(pa: jax.Array, input_res: int, *, expected_dim: int | None = None) -> jax.Array:
    if pa.ndim not in (2, 4):
        raise ValueError(f"Parent encoding must be [N,C] or [N,H,W,C], got shape {pa.shape}")
    if expected_dim is not None and int(pa.shape[-1]) != int(expected_dim):
        raise ValueError(
            "Parent encoding dimension does not match the HVAE context: "
            f"expected {expected_dim}, got {pa.shape[-1]}"
        )
    if pa.ndim == 4:
        if pa.shape[1] != input_res or pa.shape[2] != input_res:
            raise ValueError(f"Spatial parent encoding must be {input_res}x{input_res}, got {pa.shape[1:3]}")
        return pa
    return pa[:, None, None, :].repeat(input_res, axis=1).repeat(input_res, axis=2)


def _choose_intervention(args, dag_vars: List[str]) -> str:
    return copy.deepcopy(args.do_pa) if args.do_pa else random.choice(dag_vars)


def _batch_parent(args, batch: Dict[str, jax.Array], name: str) -> jax.Array:
    if name in batch:
        value = batch[name]
        return value[:, None] if value.ndim == 1 else value

    pa = batch["pa"]
    slices = _parent_slices(args)
    if name in slices:
        return pa[:, slices[name]]
    raise KeyError(f"Parent {name!r} is not present in batch or schema for args: {args.dataset}")


def _make_intervention(
    args,
    batch: Dict[str, jax.Array],
    do_k: str,
    train_samples: Dict[str, np.ndarray],
    *,
    train: bool,
) -> Dict[str, jax.Array]:
    """Create one named intervention, shuffled for training and sampled for evaluation."""
    if train:
        parent = _batch_parent(args, batch, do_k)
        permutation = np.random.permutation(parent.shape[0])
        value = parent[permutation]
        return {do_k: value}

    idx = np.random.permutation(train_samples[do_k].shape[0])
    value = np.asarray(train_samples[do_k][idx][: batch["x"].shape[0]], dtype=np.float32)
    if value.ndim == 1:
        value = value[:, None]
    return {do_k: jnp.asarray(value)}


def _validate_artifact_contract(args, name: str, hparams: Dict[str, Any], *, context_dim: int) -> None:
    schema = _schema_for_args(args)
    found_dataset = hparams.get("dataset", hparams.get("dataset_id"))
    if found_dataset is not None and found_dataset != schema.dataset_id:
        raise ValueError(f"{name} checkpoint dataset mismatch: expected {schema.dataset_id!r}, found {found_dataset!r}")
    found_parents = hparams.get("parents_x")
    if found_parents is not None and list(found_parents) != list(schema.variable_names):
        raise ValueError(
            f"{name} checkpoint parent schema mismatch: expected {list(schema.variable_names)!r}, found {list(found_parents)!r}"
        )
    found_context = hparams.get("context_dim")
    if found_context is None or int(found_context) != int(context_dim):
        raise ValueError(
            f"{name} checkpoint context_dim mismatch: expected {context_dim}, found {found_context!r}; "
            "checkpoint and context contract are incompatible"
        )
    found_res = hparams.get("input_res")
    if found_res is not None and int(found_res) != int(args.input_res):
        raise ValueError(f"{name} checkpoint input_res mismatch: expected {args.input_res}, found {found_res}")


def _load_vae_bundle(args):
    """Restore the frozen pretrained image mechanism and its checkpoint metadata."""
    vae_hparams = _load_checkpoint_hparams(args.vae_path)
    _validate_artifact_contract(args, "VAE", vae_hparams, context_dim=int(args.context_dim))
    inherit_image_training_config(args, vae_hparams)
    model_args = {
        key: vae_hparams.get(key, getattr(args, key))
        for key in (
            "input_channels",
            "input_res",
            "enc_arch",
            "dec_arch",
            "widths",
            "z_dim",
            "context_dim",
            "z_max_res",
            "bottleneck",
            "cond_prior",
            "q_correction",
            "bias_max_res",
            "x_like",
            "kl_free_bits",
            "std_init",
        )
    }
    for key, value in model_args.items():
        setattr(args, key, value)

    rngs = nnx.Rngs(args.seed)
    vae = HVAE(**model_args, rngs=rngs)
    graphdef, _ = nnx.split(vae, nnx.Param)
    params = nnx.state(vae, nnx.Param).to_pure_dict()
    _, tx = init_state(vae, args, None, jax.random.PRNGKey(args.seed))
    template = {
        "epoch": 0,
        "step": 0,
        "best_loss": float("inf"),
        "params": params,
        "ema_params": params,
        "opt_state": tx.init(params),
    }
    checkpoint = _load_runtime_checkpoint(args, args.vae_path, "vae", template=template)
    params = checkpoint.get("ema_params", checkpoint.get("params"))
    if params is None:
        raise ValueError(f"VAE checkpoint at {args.vae_path} is missing params")
    _assert_tree_compatible("VAE", checkpoint, nnx.state(vae, nnx.Param).to_pure_dict(), "params")
    return checkpoint, Bundle(graphdef, params)


def _load_pgm_bundle(args):
    """Restore only the SCM EMA inference leaves onto the active device."""
    pgm_hparams = _load_checkpoint_hparams(args.pgm_path)
    _validate_artifact_contract(args, "PGM", pgm_hparams, context_dim=_schema_for_args(args).encoded_dim)
    pgm = _build_pgm_model(args, pgm_hparams)
    params = nnx.state(pgm, nnx.Param).to_pure_dict()
    # The configured PGM can have been trained on a different accelerator.
    # A narrow target template restores only the frozen inference state onto
    # the current device without requiring the saved source topology.
    pgm_ckpt = _load_runtime_checkpoint(
        args,
        args.pgm_path,
        "pgm",
        template={"ema_params": params, "format_version": 0},
        partial_restore=True,
    )
    if pgm_ckpt.get("format_version") != 2 or "ema_params" not in pgm_ckpt:
        raise ValueError(
            "The PGM checkpoint uses the old simplified Gaussian/CNN format. "
            "Retrain it with the clean train-scm stage before running counterfactual finetuning."
        )
    _assert_tree_compatible("PGM", pgm_ckpt, params, "ema_params")
    graphdef, _ = nnx.split(pgm, nnx.Param)
    return pgm_ckpt, Bundle(graphdef, pgm_ckpt["ema_params"])


def _load_predictor_bundle(args):
    """Restore predictor EMA parameters plus BatchNorm statistics for evaluation mode."""
    predictor_hparams = _load_checkpoint_hparams(args.predictor_path)
    _validate_artifact_contract(args, "predictor", predictor_hparams, context_dim=_schema_for_args(args).encoded_dim)
    predictor = _build_predictor_model(args, predictor_hparams, load_pretrained_weights=False)
    graphdef, params_state, batch_stats_state = nnx.split(
        predictor, nnx.Param, nnx.BatchStat
    )
    params = params_state.to_pure_dict()
    batch_stats = batch_stats_state.to_pure_dict()
    predictor_ckpt = _load_runtime_checkpoint(
        args,
        args.predictor_path,
        "predictor",
        template={
            "ema_params": params,
            "ema_batch_stats": batch_stats,
            "format_version": 0,
        },
        partial_restore=True,
    )
    if predictor_ckpt.get("format_version") != 3 or "ema_params" not in predictor_ckpt:
        raise ValueError(
            "The predictor checkpoint uses the old simplified CNN format. "
            "Retrain it with the clean train-predictor stage before running counterfactual finetuning."
        )
    params = predictor_ckpt["ema_params"]
    batch_stats = predictor_ckpt.get("ema_batch_stats", batch_stats)
    _assert_tree_compatible(
        "predictor", predictor_ckpt, params_state.to_pure_dict(), "ema_params"
    )
    if "ema_batch_stats" in predictor_ckpt and predictor_ckpt["ema_batch_stats"]:
        _assert_tree_compatible(
            "predictor", predictor_ckpt, batch_stats_state.to_pure_dict(), "ema_batch_stats"
        )
    return predictor_ckpt, Bundle(graphdef, params, batch_stats)


def _predictor_metrics(args, dataset, preds, targets):
    schema = _schema_for_args(args)
    var_specs = {spec.name: spec for spec in schema.variables}
    stats: Dict[str, float] = {}
    for k in preds.keys():
        if k not in var_specs:
            continue
        spec = var_specs[k]
        pred = np.asarray(preds[k])
        target = np.asarray(targets[k])
        if spec.kind == VariableKind.CATEGORICAL or k == "digit":
            stats[f"{k}_acc"] = float((target.argmax(-1) == pred.argmax(-1)).mean())
        elif spec.kind == VariableKind.BINARY:
            target_bin = (target.squeeze(-1) > 0.5).astype(np.float32) if target.ndim > 1 else (target > 0.5).astype(np.float32)
            pred_bin = (pred.squeeze(-1) > 0.5).astype(np.float32) if pred.ndim > 1 else (pred > 0.5).astype(np.float32)
            stats[f"{k}_acc"] = float((target_bin == pred_bin).mean())
        else:
            if hasattr(dataset, "min_max") and k in dataset.min_max:
                min_val, max_val = dataset.min_max[k]
                pred = ((pred.squeeze(-1) + 1.0) / 2.0) * (max_val - min_val) + min_val
                target = ((target.squeeze(-1) + 1.0) / 2.0) * (max_val - min_val) + min_val
            else:
                pred = pred.squeeze(-1)
                target = target.squeeze(-1)
            stats[f"{k}_mae"] = float(np.mean(np.abs(target - pred)))
    return stats


def _tree_is_finite(tree: Any) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    checks = []
    for leaf in leaves:
        value = jnp.asarray(leaf)
        if jnp.issubdtype(value.dtype, jnp.floating) or jnp.issubdtype(value.dtype, jnp.complexfloating):
            checks.append(jnp.all(jnp.isfinite(value)))
    return jnp.all(jnp.stack(checks)) if checks else jnp.asarray(True)


def _require_finite(tree: Any, *, context: str) -> None:
    if not bool(np.asarray(_tree_is_finite(tree))):
        raise FloatingPointError(f"Non-finite values produced during {context}")


def _nan_like(value: jax.Array) -> jax.Array:
    return jnp.asarray(jnp.nan, dtype=jnp.asarray(value).dtype)


def _cf_lr_scale(step: jax.Array, warmup_steps: int) -> jax.Array:
    if warmup_steps <= 0:
        return jnp.asarray(1.0, dtype=jnp.float32)
    return jnp.minimum(
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(step, dtype=jnp.float32) / jnp.asarray(warmup_steps, dtype=jnp.float32),
    )


def _cf_forward(
    args,
    vae_bundle: Bundle,
    pgm_bundle: Bundle,
    predictor_bundle: Bundle,
    batch: Dict[str, jax.Array],
    do: Dict[str, jax.Array],
    rng: jax.Array,
    *,
    beta: float,
    alpha: float,
    lmbda: jax.Array,
    cf_particles: int,
    t_abduct: float = 1.0,
    training: bool = True,
):
    vae = vae_bundle.materialize()
    pgm = pgm_bundle.materialize()
    predictor = predictor_bundle.materialize()

    set_module_training_mode(vae, training)
    pgm.eval()
    predictor.eval()

    x = batch["x"].astype(jnp.float32)
    pa = batch["pa"].astype(jnp.float32)
    schema_dim = _schema_for_args(args).encoded_dim
    _split_parent_vector(args, pa)  # Validate the source representation before dispatch.
    vae_context_dim = int(getattr(vae, "context_dim", schema_dim))
    vae_pa = _hvae_context(args, pa, vae_context_dim)
    pa_maps = _expand_parents(vae_pa, args.input_res, expected_dim=vae_context_dim)
    vae_rng, counterfactual_rng = jax.random.split(rng)
    vae_out = vae(x, pa_maps, beta=beta, rng=vae_rng, training=training)

    obs_pgm = _split_parent_vector(args, pa)

    if cf_particles > 1:
        cfs = {"x": jnp.zeros_like(batch["x"]), "x2": jnp.zeros_like(batch["x"])}

    particle_keys = jax.random.split(counterfactual_rng, cf_particles)
    for i in range(cf_particles):
        pgm_rng, abduct_rng, cf_rng, rec_rng = jax.random.split(particle_keys[i], 4)
        cf_pa = pgm.counterfactual(obs=obs_pgm, intervention=do, rng=pgm_rng)
        cf_vae_pa = _hvae_context(args, cf_pa["pa"], vae_context_dim)
        cf_pa_maps = _expand_parents(cf_vae_pa, args.input_res, expected_dim=vae_context_dim)
        latents = vae.abduct(x, pa_maps, t=t_abduct, rng=abduct_rng)
        cf_loc, cf_scale = vae.forward_latents(latents, cf_pa_maps, rng=cf_rng)
        rec_loc, rec_scale = vae.forward_latents(latents, pa_maps, rng=rec_rng)
        u = (x - rec_loc) / jnp.clip(rec_scale, min=1e-12)
        cf_x = jnp.clip(cf_loc + cf_scale * u, min=-1, max=1)
        if cf_particles > 1:
            cfs["x"] = cfs["x"] + cf_x
            cfs["x2"] = cfs["x2"] + jax.lax.stop_gradient(cf_x**2)
        else:
            cfs = {"x": cf_x}

    if cf_particles > 1:
        var_cf_x = (cfs["x2"] - cfs["x"] ** 2 / cf_particles) / cf_particles
        cfs.pop("x2", None)
        cfs["x"] = cfs["x"] / cf_particles
    else:
        var_cf_x = None

    cfs.update(cf_pa)
    finite = _tree_is_finite(vae_out) & _tree_is_finite(cfs)
    if var_cf_x is not None:
        finite = finite & _tree_is_finite(var_cf_x)
    log_probs = predictor.model_anticausal(**cfs)
    aux_loss = -jnp.mean(log_probs["joint"])
    constraint = args.elbo_constraint - vae_out["elbo"]
    loss = damped_lagrangian_loss(aux_loss, lmbda, constraint, args.damping)
    loss = jax.lax.select(finite, loss, _nan_like(loss))
    aux_loss = jax.lax.select(finite, aux_loss, _nan_like(aux_loss))
    out = dict(vae_out)
    out.update({"loss": loss, "aux_loss": aux_loss, "cfs": cfs, "var_cf_x": var_cf_x})
    return out


def _make_losses(args, vae_bundle, pgm_bundle, predictor_bundle):
    """Build the constrained objective from frozen SCM/predictor and trainable VAE state."""
    def loss_fn(vae_params, lmbda, batch, do, rng):
        local_vae = Bundle(vae_bundle.graphdef, _vae_compute_params(args, vae_params))
        out = _cf_forward(
            args,
            local_vae,
            pgm_bundle,
            predictor_bundle,
            batch,
            do,
            rng,
            beta=args.beta,
            alpha=args.alpha,
            lmbda=lmbda,
            cf_particles=args.cf_particles,
            training=True,
        )
        return out["loss"], out
    return loss_fn


def _make_train_step(args, vae_bundle, pgm_bundle, predictor_bundle, optimizer, lambda_optimizer):
    """Compile one joint VAE/Lagrange update while keeping SCM and predictor frozen."""
    loss_fn = _make_losses(args, vae_bundle, pgm_bundle, predictor_bundle)

    def step(
        vae_params,
        opt_state,
        lmbda,
        lambda_opt_state,
        batch,
        do,
        rng,
        global_step,
    ):
        (loss, out), grads = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(
            vae_params, lmbda, batch, do, rng
        )
        vae_grads, lmbda_grads = grads
        clipped_vae_grads, clipped_lmbda_grads, grad_norm = clip_counterfactual_grads(
            vae_grads, lmbda_grads, args.grad_clip
        )
        finite = jnp.isfinite(loss) & jnp.isfinite(grad_norm)
        finite = finite & (grad_norm < args.grad_skip)
        lr_scale = _cf_lr_scale(global_step, args.lr_warmup_steps)

        def _apply_updates(values):
            params, opt_state, lmbda_value, lambda_opt_state = values
            updates, opt_state = optimizer.update(clipped_vae_grads, opt_state, params)
            updates = jax.tree_util.tree_map(lambda update: update * lr_scale, updates)
            params = optax.apply_updates(params, updates)
            lambda_updates, lambda_opt_state = lambda_optimizer.update(
                jax.tree_util.tree_map(lambda x: -x, clipped_lmbda_grads),
                lambda_opt_state,
                lmbda_value,
            )
            lmbda_value = optax.apply_updates(lmbda_value, lambda_updates)
            lmbda_value = jnp.clip(lmbda_value, min=0)
            return params, opt_state, lmbda_value, lambda_opt_state

        def _skip_updates(values):
            return values

        vae_params, opt_state, lmbda, lambda_opt_state = jax.lax.cond(
            finite,
            _apply_updates,
            _skip_updates,
            operand=(vae_params, opt_state, lmbda, lambda_opt_state),
        )

        out = dict(out)
        out["grad_norm"] = grad_norm
        out["grad_clipped"] = (grad_norm > args.grad_clip).astype(jnp.float32)
        out["lr_scale"] = lr_scale
        out["update_skipped"] = jnp.logical_not(finite).astype(jnp.float32)
        return vae_params, opt_state, lmbda, lambda_opt_state, out

    return jax.jit(step, donate_argnums=(0, 1, 2, 3))


def _make_pmap_train_step(args, vae_graphdef, optimizer, lambda_optimizer, devices):
    """Compile one synchronized local-TPU counterfactual update."""
    scalar_keys = ("loss", "aux_loss", "elbo", "nll", "kl")

    def step(
        vae_params,
        opt_state,
        lmbda,
        lambda_opt_state,
        batch,
        do,
        rng,
        pgm_bundle,
        predictor_bundle,
        global_step,
    ):
        def loss_fn(current_vae_params, current_lmbda):
            local_vae = Bundle(vae_graphdef, _vae_compute_params(args, current_vae_params))
            out = _cf_forward(
                args,
                local_vae,
                pgm_bundle,
                predictor_bundle,
                batch,
                do,
                rng,
                beta=args.beta,
                alpha=args.alpha,
                lmbda=current_lmbda,
                cf_particles=args.cf_particles,
                training=True,
            )
            return out["loss"], out

        (loss, out), (vae_grads, lmbda_grads) = jax.value_and_grad(
            loss_fn, argnums=(0, 1), has_aux=True
        )(vae_params, lmbda)
        vae_grads = jax.lax.pmean(vae_grads, axis_name="devices")
        lmbda_grads = jax.lax.pmean(lmbda_grads, axis_name="devices")
        clipped_vae_grads, clipped_lmbda_grads, grad_norm = clip_counterfactual_grads(
            vae_grads, lmbda_grads, args.grad_clip
        )
        local_finite = jnp.isfinite(loss) & jnp.isfinite(grad_norm) & (grad_norm < args.grad_skip)
        finite = jax.lax.pmin(local_finite.astype(jnp.int32), axis_name="devices").astype(jnp.bool_)
        lr_scale = _cf_lr_scale(global_step, args.lr_warmup_steps)

        def _apply_updates(values):
            params, current_opt_state, lmbda_value, current_lambda_opt_state = values
            updates, current_opt_state = optimizer.update(clipped_vae_grads, current_opt_state, params)
            updates = jax.tree_util.tree_map(lambda update: update * lr_scale, updates)
            params = optax.apply_updates(params, updates)
            lambda_updates, current_lambda_opt_state = lambda_optimizer.update(
                jax.tree_util.tree_map(lambda value: -value, clipped_lmbda_grads),
                current_lambda_opt_state,
                lmbda_value,
            )
            lmbda_value = jnp.clip(optax.apply_updates(lmbda_value, lambda_updates), min=0)
            return params, current_opt_state, lmbda_value, current_lambda_opt_state

        vae_params, opt_state, lmbda, lambda_opt_state = jax.lax.cond(
            finite, _apply_updates, lambda values: values,
            operand=(vae_params, opt_state, lmbda, lambda_opt_state),
        )
        metrics = {
            key: jax.lax.pmean(out[key], axis_name="devices")
            for key in scalar_keys
        }
        metrics.update(
            grad_norm=jax.lax.pmean(grad_norm, axis_name="devices"),
            grad_clipped=(grad_norm > args.grad_clip).astype(jnp.float32),
            lr_scale=lr_scale,
            update_skipped=jnp.logical_not(finite).astype(jnp.float32),
        )
        return vae_params, opt_state, lmbda, lambda_opt_state, metrics

    return jax.pmap(
        step,
        axis_name="devices",
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None),
        devices=devices,
    )


def _make_eval_step(args, vae_bundle, pgm_bundle, predictor_bundle):
    """Compile the same counterfactual objective without applying updates."""
    def step(vae_params, lmbda, batch, do, rng):
        local_vae = Bundle(vae_bundle.graphdef, _vae_compute_params(args, vae_params))
        out = _cf_forward(
            args,
            local_vae,
            pgm_bundle,
            predictor_bundle,
            batch,
            do,
            rng,
            beta=args.beta,
            alpha=args.alpha,
            lmbda=lmbda,
            cf_particles=args.cf_particles,
            training=False,
        )
        return out

    return jax.jit(step)


def _make_optimizers(args):
    optimizer = optax.adamw(
        learning_rate=args.lr,
        b1=args.betas[0],
        b2=args.betas[1],
        weight_decay=args.wd,
    )
    lambda_optimizer = optax.adamw(
        learning_rate=args.lr_lagrange,
        b1=args.betas[0],
        b2=args.betas[1],
        weight_decay=0.0,
    )
    return optimizer, lambda_optimizer


def _prefetch_kwargs(args) -> Dict[str, int]:
    return {
        "prefetch_batches": max(0, int(getattr(args, "input_prefetch_batches", 0) or 0)),
        "prefetch_workers": max(1, int(getattr(args, "input_prefetch_workers", 1) or 1)),
    }


def _eval_split(
    args,
    split: str,
    datasets,
    state,
    vae_bundle,
    pgm_bundle,
    predictor_bundle,
    eval_step,
    train_samples,
    rng,
    *,
    max_batches: int | None = None,
):
    dag_vars = _dag_variables(args)
    dataset = datasets[split]
    stats = {k: 0.0 for k in ["loss", "aux_loss", "elbo", "nll", "kl", "n"]}
    preds = {k: [] for k in dag_vars}
    targets = {k: [] for k in dag_vars}
    grad_norm = 0.0
    predictor = predictor_bundle.materialize()
    predictor.eval()
    for i, raw_batch in enumerate(
        _epoch_batches(
            dataset,
            args.bs,
            shuffle=(split == "train"),
            drop_last=(split == "train"),
            rng=rng,
            **_prefetch_kwargs(args),
        )
    ):
        if max_batches is not None and i >= max_batches:
            break
        batch = _preprocess_counterfactual_batch(args, raw_batch)
        do_k = _choose_intervention(args, dag_vars)
        do = _make_intervention(args, batch, do_k, train_samples, train=(split == "train"))
        out = eval_step(
            state["vae_params"],
            state["lmbda"],
            batch,
            do,
            jax.random.fold_in(jax.random.PRNGKey(args.seed), i),
        )
        _require_finite(out, context=f"{split} evaluation batch {i}")
        bs = int(batch["x"].shape[0])
        stats["n"] += bs
        stats["loss"] += float(out["loss"]) * bs
        stats["aux_loss"] += float(out["aux_loss"]) * args.alpha * bs
        stats["elbo"] += float(out["elbo"]) * bs
        stats["nll"] += float(out["nll"]) * bs
        stats["kl"] += float(out["kl"]) * bs
        grad_norm = float(out.get("grad_norm", 0.0))
        if split != "train":
            preds_cf = predictor.predict(**out["cfs"])
            for k, v in preds_cf.items():
                if k in preds:
                    preds[k].append(np.asarray(v))
            for k in targets.keys():
                t_k = do[k] if k in do else out["cfs"][k]
                targets[k].append(np.asarray(t_k))
    mean_stats = {k: v / stats["n"] for k, v in stats.items() if k != "n"}
    if split == "train":
        return mean_stats, None

    preds = {k: np.concatenate(v, axis=0) if len(v) > 1 else np.asarray(v[0]) for k, v in preds.items() if v}
    targets = {k: np.concatenate(v, axis=0) if len(v) > 1 else np.asarray(v[0]) for k, v in targets.items() if v}
    return mean_stats, _predictor_metrics(args, dataset, preds, targets)


def _batch_indices_for_epoch(
    dataset,
    batch_size: int,
    *,
    shuffle: bool,
    drop_last: bool,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    indices = np.arange(len(dataset), dtype=np.int64)
    if shuffle:
        rng.shuffle(indices)
    batches: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        if drop_last and batch_idx.size < batch_size:
            continue
        batches.append(batch_idx)
    return batches


def _load_epoch_batch(dataset, batch_idx: np.ndarray, *, rng: np.random.Generator, shuffle: bool):
    if hasattr(dataset, "make_batch"):
        return dataset.make_batch(batch_idx, rng=rng, shuffle=shuffle)
    examples = [dataset[int(i)] for i in batch_idx]
    keys = examples[0].keys()
    batch = {}
    for k in keys:
        values = [np.asarray(item[k]) for item in examples]
        batch[k] = np.stack(values, axis=0)
    return batch


def _epoch_batches(
    dataset,
    batch_size: int,
    *,
    shuffle: bool,
    drop_last: bool,
    rng: np.random.Generator,
    prefetch_batches: int = 0,
    prefetch_workers: int = 1,
) -> Iterator[Dict[str, np.ndarray]]:
    prefetch = max(0, int(prefetch_batches))
    workers = max(1, int(prefetch_workers))
    batches = _batch_indices_for_epoch(
        dataset,
        batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        rng=rng,
    )
    if prefetch <= 0:
        for batch_idx in batches:
            yield _load_epoch_batch(dataset, batch_idx, rng=rng, shuffle=shuffle)
        return

    def submit(executor: ThreadPoolExecutor, batch_idx: np.ndarray) -> Future:
        # Derive per-batch seeds before asynchronous loading to keep any dataset
        # augmentation deterministic despite bounded host-side prefetch.
        seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        return executor.submit(
            _load_epoch_batch,
            dataset,
            batch_idx,
            rng=np.random.default_rng(seed),
            shuffle=shuffle,
        )

    with ThreadPoolExecutor(max_workers=min(workers, prefetch)) as executor:
        in_flight: list[Future] = []
        iterator = iter(batches)
        for _ in range(prefetch):
            try:
                in_flight.append(submit(executor, next(iterator)))
            except StopIteration:
                break
        while in_flight:
            future = in_flight.pop(0)
            try:
                in_flight.append(submit(executor, next(iterator)))
            except StopIteration:
                pass
            yield future.result()


def _model_validation_batches(args, dataset):
    rng = np.random.default_rng(args.seed)
    for index, raw_batch in enumerate(
        _epoch_batches(
            dataset,
            args.bs,
            shuffle=False,
            drop_last=False,
            rng=rng,
            **_prefetch_kwargs(args),
        )
    ):
        if args.model_validation_batches > 0 and index >= args.model_validation_batches:
            break
        yield _preprocess_counterfactual_batch(args, raw_batch)


def _full_model_validation_batches(args, dataset):
    rng = np.random.default_rng(args.seed)
    for raw_batch in _epoch_batches(
        dataset,
        args.bs,
        shuffle=False,
        drop_last=False,
        rng=rng,
        **_prefetch_kwargs(args),
    ):
        yield _preprocess_counterfactual_batch(args, raw_batch)


def _validated_means(name: str, path: str, totals: Dict[str, float], count: int) -> Dict[str, float]:
    if count == 0:
        raise ValueError(f"{name} checkpoint validation found no test samples: {path}")
    means = {key: value / count for key, value in totals.items()}
    if not all(np.isfinite(value) for value in means.values()):
        raise ValueError(f"{name} checkpoint validation produced non-finite metrics at {path}: {means}")
    metrics = " - ".join(f"{key}: {value:.4f}" for key, value in means.items())
    print(f"Validated {name} checkpoint: {path} | samples: {count} | {metrics}")
    return means


def _print_dataset_normalization(datasets: Dict[str, Any]) -> None:
    for split in ("train", "valid", "test"):
        dataset = datasets[split]
        norm = getattr(dataset, "norm", getattr(dataset, "context_norm", "N/A"))
        min_max = getattr(dataset, "min_max", {})
        for variable, bounds in min_max.items():
            min_value, max_value = bounds
            print(f"{variable} normalization: {norm}")
            print(f"max: {max_value}, min: {min_value}")
        print(f"#samples: {len(dataset)}")
        print()


def _validate_vae_checkpoint(args, bundle: Bundle, dataset) -> None:
    model = Bundle(bundle.graphdef, _vae_compute_params(args, bundle.params)).materialize()
    model.eval()
    totals = {key: 0.0 for key in ("elbo", "nll", "kl")}
    count = 0
    for index, batch in enumerate(_full_model_validation_batches(args, dataset)):
        x = batch["x"].astype(jnp.float32)
        source_pa = batch["pa"].astype(jnp.float32)
        parents = _expand_parents(
            _hvae_context(args, source_pa, int(getattr(model, "context_dim", args.context_dim))),
            args.input_res,
            expected_dim=int(getattr(model, "context_dim", args.context_dim)),
        )
        outputs = model(
            x,
            parents,
            beta=args.beta,
            rng=jax.random.PRNGKey(args.seed + index),
            training=False,
        )
        size = int(batch["x"].shape[0])
        for key in totals:
            totals[key] += float(outputs[key]) * size
        count += size
    means = _validated_means("VAE", args.resolved_vae_path, totals, count)
    print(
        format_checkpoint_validation_summary(
            {"loss": means["elbo"], "nll": means["nll"], "kl": means["kl"]},
            extra_keys=("nll", "kl"),
        )
    )


def _validate_pgm_checkpoint(args, bundle: Bundle, dataset) -> None:
    model = bundle.materialize()
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    for batch in _full_model_validation_batches(args, dataset):
        pa = batch["pa"]
        obs = _split_parent_vector(args, pa)
        outputs = model.log_prob(**obs)
        batch_means = {
            "loss": float(-jnp.mean(outputs["joint"])),
            **{f"logp({k})": float(jnp.mean(v)) for k, v in outputs.items() if k != "joint"}
        }
        size = int(batch["x"].shape[0])
        for key, value in batch_means.items():
            totals[key] = totals.get(key, 0.0) + value * size
        count += size
    means = _validated_means("PGM", args.resolved_pgm_path, totals, count)
    extra_keys = tuple(k for k in sorted(means.keys()) if k != "loss")
    print(
        format_checkpoint_validation_summary(
            means,
            extra_keys=extra_keys,
        )
    )


def _validate_predictor_checkpoint(args, bundle: Bundle, dataset) -> None:
    model = bundle.materialize()
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    dag_vars = _dag_variables(args)
    preds = {k: [] for k in dag_vars}
    targets = {k: [] for k in dag_vars}
    for batch_index, batch in enumerate(_full_model_validation_batches(args, dataset)):
        pa = batch["pa"]
        parents = _split_parent_vector(args, pa)
        outputs = model.model_anticausal(x=batch["x"], **parents)
        batch_stats = {
            "loss": float(-jnp.mean(outputs["joint"])),
            **{f"logp({k})": float(jnp.mean(v)) for k, v in outputs.items() if k != "joint" and k.endswith("_aux")},
            **{f"logp({k}_aux)": float(jnp.mean(v)) for k, v in outputs.items() if k != "joint" and not k.endswith("_aux")},
        }
        _require_finite(
            batch_stats,
            context=f"predictor checkpoint validation batch {batch_index}",
        )
        size = int(batch["x"].shape[0])
        for key, value in batch_stats.items():
            totals[key] = totals.get(key, 0.0) + value * size
        count += size
        pred_batch = model.predict(x=batch["x"], **parents)
        for key, value in pred_batch.items():
            if key in preds:
                preds[key].append(np.asarray(value))
        for key, value in parents.items():
            if key in targets:
                targets[key].append(np.asarray(value))
    _validated_means("predictor", args.resolved_predictor_path, totals, count)
    metrics = _predictor_metrics(
        args,
        dataset,
        {k: np.concatenate(v, axis=0) for k, v in preds.items() if v},
        {k: np.concatenate(v, axis=0) for k, v in targets.items() if v},
    )
    print("test | " + " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))


def _cf_checkpoint_payload(args, state, epoch: int) -> Dict[str, Any]:
    return {
        "epoch": epoch,
        "step": state["step"],
        "best_loss": state["best_loss"],
        "vae_params": state["vae_params"],
        "ema_params": state["ema_params"],
        "opt_state": state["opt_state"],
        "lmbda": state["lmbda"],
        "lambda_opt_state": state["lambda_opt_state"],
        "hparams": vars(args),
        "format_version": 1,
    }


def _submit_best_cf_checkpoint(artifact_writer: BackgroundArtifactWriter, args, state, epoch: int) -> None:
    """Snapshot an improved counterfactual state for asynchronous persistence."""
    payload = _cf_checkpoint_payload(args, state, epoch)
    remote_checkpoint_dir = (
        os.path.join(args.remote_save_dir, "checkpoints")
        if getattr(args, "remote_save_dir", "")
        else None
    )
    artifact_writer.submit_checkpoint(
        payload,
        args.checkpoint_dir,
        step=state["step"],
        custom_metadata={"epoch": epoch, "best_loss": float(state["best_loss"])},
        local_tree_dir=args.checkpoint_dir if remote_checkpoint_dir else None,
        remote_tree_dir=remote_checkpoint_dir,
    )


def _sync_tensorboard_artifacts(args) -> None:
    if not getattr(args, "remote_save_dir", ""):
        return
    for event_path in sorted(Path(args.save_dir).glob("events.out.tfevents.*")):
        sync_file(str(event_path), os.path.join(args.remote_save_dir, event_path.name))


def _sync_metric_artifacts(args) -> None:
    """Synchronize flushed counterfactual logs and TensorBoard events to GCS."""
    if not getattr(args, "remote_save_dir", ""):
        return
    sync_file(
        os.path.join(args.save_dir, "trainlog.txt"),
        os.path.join(args.remote_save_dir, "trainlog.txt"),
    )
    _sync_tensorboard_artifacts(args)


def _checkpoint_due(epoch: int, checkpoint_freq: int) -> bool:
    return epoch % max(1, checkpoint_freq) == 0


def _eval_due(epoch: int, eval_freq: int) -> bool:
    return eval_freq > 0 and epoch % eval_freq == 0


def _bounded_validation_batches(args) -> int | None:
    raw = getattr(args, "model_validation_batches", 0)
    if raw is None:
        return None
    max_batches = int(raw)
    return max_batches if max_batches > 0 else None


def _intervention_tag(variable: str | None) -> str:
    return "observational" if variable is None else f"do_{variable}"


def _write_epoch_summary(
    writer: Any,
    *,
    epoch: int,
    step: int,
    train_stats: Dict[str, float],
    lmbda: float,
    diagnostics: Dict[str, float],
    train_time: float,
    total_time: float,
    iter_per_sec: float,
    sample_per_sec: float,
    validation: Dict[str, tuple[Dict[str, float], Dict[str, float]]] | None = None,
) -> None:
    for key, value in train_stats.items():
        writer.add_scalar(f"train/{key}", value, step)
    writer.add_scalar("loss/train", train_stats["loss"], step)
    writer.add_scalar("aux_loss/train", train_stats["aux_loss"], step)
    writer.add_scalar("train/lmbda", lmbda, step)
    for key, value in diagnostics.items():
        writer.add_scalar(f"train/{key}", value, step)
    writer.add_scalar("epoch/number", epoch, step)
    writer.add_scalar("epoch/global_step", step, step)
    writer.add_scalar("epoch/train_time_sec", train_time, step)
    writer.add_scalar("epoch/total_time_sec", total_time, step)
    writer.add_scalar("epoch/iter_per_sec", iter_per_sec, step)
    writer.add_scalar("epoch/sample_per_sec", sample_per_sec, step)
    if validation is None:
        return
    for intervention, (stats, metrics) in validation.items():
        for key, value in stats.items():
            writer.add_scalar(f"valid/{intervention}/{key}", value, step)
        for key, value in metrics.items():
            writer.add_scalar(f"valid/{intervention}/{key}", value, step)
    if "observational" in validation:
        observational_stats, observational_metrics = validation["observational"]
        for key, value in observational_stats.items():
            writer.add_scalar(f"valid/{key}", value, step)
        for key, value in observational_metrics.items():
            writer.add_scalar(f"valid/{key}", value, step)
        writer.add_scalar("loss/valid", observational_stats["loss"], step)
        writer.add_scalar("aux_loss/valid", observational_stats["aux_loss"], step)


def _log_epoch_summary(
    logger: logging.Logger,
    *,
    epoch: int,
    step: int,
    train_stats: Dict[str, float],
    lmbda: float,
    diagnostics: Dict[str, float],
    train_time: float,
    total_time: float,
    iter_per_sec: float,
    sample_per_sec: float,
    validation: Dict[str, tuple[Dict[str, float], Dict[str, float]]] | None = None,
) -> None:
    train_description = " - ".join(f"{key}: {value:.4f}" for key, value in train_stats.items())
    logger.info(
        "=> train | %s - lmbda: %.4f - grad_norm: %.4f - steps: %d - it/s: %.3f - samples/s: %.3f",
        train_description, lmbda, diagnostics.get("grad_norm", 0.0), step, iter_per_sec, sample_per_sec,
    )
    if validation is not None:
        for intervention, (stats, metrics) in validation.items():
            description = " - ".join(
                f"{key}: {value:.4f}" for key, value in {**stats, **metrics}.items()
            )
            logger.info("=> valid %s | %s - steps: %d", intervention, description, step)
    logger.info(
        "epoch=%d train_time=%.1fs total_time=%.1fs epoch_iter/s=%.3f epoch_sample/s=%.3f",
        epoch, train_time, total_time, iter_per_sec, sample_per_sec,
    )


def _evaluate_interventions(
    args,
    datasets,
    eval_state,
    vae_bundle,
    pgm_bundle,
    predictor_bundle,
    eval_step,
    train_samples,
    rng,
    dag_vars: list[str],
    *,
    max_batches: int | None,
) -> Dict[str, tuple[Dict[str, float], Dict[str, float]]]:
    validation: Dict[str, tuple[Dict[str, float], Dict[str, float]]] = {}
    copy_do_pa = copy.deepcopy(args.do_pa)
    try:
        for pa_k in dag_vars + [None]:
            args.do_pa = pa_k
            valid_stats, valid_metrics = _eval_split(
                args,
                "valid",
                datasets,
                eval_state,
                vae_bundle,
                pgm_bundle,
                predictor_bundle,
                eval_step,
                train_samples,
                rng,
                max_batches=max_batches,
            )
            validation[_intervention_tag(pa_k)] = (valid_stats, valid_metrics)
    finally:
        args.do_pa = copy_do_pa
    return validation


def main(args):
    """Run artifact validation, frozen-model preflight, and counterfactual fine-tuning."""
    _validate_runtime_device(args)
    _configure_compute_policy(args)
    seed_all(args.seed, args.deterministic)
    if args.do_pa in {"None", "none", "null", ""}:
        args.do_pa = None
    _schema_for_args(args)

    if not hasattr(args, "elbo_constraint") or args.elbo_constraint is None:
        args.elbo_constraint = 1.841216802597046

    # Load and validate each upstream component independently. This fails early
    # if a checkpoint has the wrong architecture, schema, or device topology.
    vae_ckpt, vae_bundle = _load_vae_bundle(args)
    datasets = _build_datasets(args)

    _print_dataset_normalization(datasets)
    _validate_vae_checkpoint(args, vae_bundle, datasets["test"])

    pgm_ckpt, pgm_bundle = _load_pgm_bundle(args)
    _print_dataset_normalization(datasets)
    _validate_pgm_checkpoint(args, pgm_bundle, datasets["test"])

    predictor_ckpt, predictor_bundle = _load_predictor_bundle(args)
    _print_dataset_normalization(datasets)
    _validate_predictor_checkpoint(args, predictor_bundle, datasets["test"])

    if jax.tree_util.tree_structure(vae_bundle.params) != jax.tree_util.tree_structure(
        vae_ckpt.get("ema_params", vae_ckpt.get("params"))
    ):
        raise ValueError("VAE checkpoint parameter structure is incompatible with HVAE")

    # Start optimization from the pretrained VAE; SCM and predictor remain in
    # immutable Bundles captured by the compiled train/eval functions.
    state = {
        "vae_params": vae_ckpt.get("ema_params", vae_ckpt.get("params")),
        "opt_state": None,
        "lmbda": jnp.asarray(args.lmbda_init, dtype=jnp.float32),
        "lambda_opt_state": None,
        "ema": EMA.init_from(
            vae_ckpt.get("ema_params", vae_ckpt.get("params")),
            args.ema_rate,
            update_after_step=100,
        ),
        "ema_params": tree_copy(vae_ckpt.get("ema_params", vae_ckpt.get("params"))),
        "step": 0,
        "epoch": 0,
        "best_loss": float("inf"),
    }

    optimizer, lambda_optimizer = _make_optimizers(args)
    state["opt_state"] = optimizer.init(state["vae_params"])
    state["lambda_opt_state"] = lambda_optimizer.init(state["lmbda"])

    if args.load_path:
        if os.path.exists(args.load_path) or str(args.load_path).startswith("gs://"):
            template = {
                "epoch": state["epoch"],
                "step": state["step"],
                "best_loss": state["best_loss"],
                "vae_params": state["vae_params"],
                "ema_params": state["ema_params"],
                "opt_state": state["opt_state"],
                "lmbda": state["lmbda"],
                "lambda_opt_state": state["lambda_opt_state"],
            }
            ckpt = _load_runtime_checkpoint(args, args.load_path, "resume", template=template)
            _restore_args(args, ckpt)
            state["vae_params"] = ckpt["vae_params"]
            state["ema_params"] = ckpt["ema_params"]
            state["ema"] = EMA(
                params=tree_copy(ckpt["ema_params"]),
                decay=args.ema_rate,
                update_after_step=100,
            )
            state["opt_state"] = ckpt["opt_state"]
            state["lmbda"] = ckpt["lmbda"]
            state["lambda_opt_state"] = ckpt["lambda_opt_state"]
            state["step"] = int(ckpt.get("step", 0))
            state["epoch"] = int(ckpt.get("epoch", 0))
            state["best_loss"] = float(ckpt.get("best_loss", float("inf")))
            optimizer, lambda_optimizer = _make_optimizers(args)
        else:
            print(f"Checkpoint not found: {args.load_path}")

    args.save_dir = experiment_run_dir(args.ckpt_dir, args.dataset_id, args.exp_name, "cf")
    args.checkpoint_dir = checkpoint_root_dir(args.save_dir)
    args.remote_save_dir = experiment_run_dir(args.remote_ckpt_dir, args.dataset_id, args.exp_name, "cf")
    ensure_dir(args.save_dir)
    ensure_dir(args.checkpoint_dir)
    logger = setup_logging(args)
    writer = SummaryWriter(args.save_dir)
    train_samples = datasets["train"].samples

    log_run_summary(
        logger,
        args,
        [
            "exp_name",
            "accelerator",
            "precision",
            "dataset",
            "bs",
            "epochs",
            "lr",
            "lr_lagrange",
            "ema_rate",
            "alpha",
            "do_pa",
            "speed_log_freq",
            "checkpoint_freq",
            "eval_freq",
            "model_validation_batches",
            "final_eval_full",
            "cf_particles",
            "load_path",
            "pgm_path",
            "predictor_path",
            "vae_path",
        ],
    )
    log_checkpoint_summary(logger, args)

    # JIT closures receive dynamic VAE state and a named intervention; their
    # captured bundles are frozen upstream mechanisms.
    eval_step = _make_eval_step(args, vae_bundle, pgm_bundle, predictor_bundle)
    single_train_step = _make_train_step(
        args, vae_bundle, pgm_bundle, predictor_bundle, optimizer, lambda_optimizer
    )
    dag_vars = _dag_variables(args)
    rng = np.random.default_rng(args.seed)

    if args.testing:
        if not args.load_path:
            raise ValueError("--testing requires --load_path")
        stats, metrics = _eval_split(
            args,
            "test",
            datasets,
            state,
            vae_bundle,
            pgm_bundle,
            predictor_bundle,
            eval_step,
            train_samples,
            rng,
        )
        print("\n[test] " + " - ".join(f"{k}: {v:.4f}" for k, v in stats.items()))
        print("[test] " + " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
        writer.close()
        return

    use_tpu_pmap = _use_tpu_replication(args)
    devices = jax.local_devices() if use_tpu_pmap else []
    device_count = len(devices) if use_tpu_pmap else 1
    if use_tpu_pmap and args.bs % device_count:
        raise ValueError(
            f"Global batch size {args.bs} must be divisible by TPU local device count {device_count}."
        )
    drop_remainder = bool(getattr(args, "drop_remainder", False) or use_tpu_pmap)
    if use_tpu_pmap:
        logger.info(
            "execution_mode=replicated local_device_count=%d global_batch_size=%d per_device_batch_size=%d",
            device_count, args.bs, args.bs // device_count,
        )
        train_step = _make_pmap_train_step(args, vae_bundle.graphdef, optimizer, lambda_optimizer, devices)
        for key in ("vae_params", "opt_state", "lmbda", "lambda_opt_state"):
            state[key] = _replicate(state[key], devices)
        state["ema"] = EMA(
            params=_replicate(state["ema"].params, devices),
            decay=state["ema"].decay,
            update_after_step=state["ema"].update_after_step,
            step=state["ema"].step,
            initted=state["ema"].initted,
        )
        state["ema_params"] = _replicate(state["ema_params"], devices)
        replicated_pgm_bundle = _replicate(pgm_bundle, devices)
        replicated_predictor_bundle = _replicate(predictor_bundle, devices)
    else:
        logger.info(
            "execution_mode=single_device accelerator=%s local_device_count=%d global_batch_size=%d",
            args.accelerator, jax.local_device_count(), args.bs,
        )
        train_step = single_train_step
        replicated_pgm_bundle = replicated_predictor_bundle = None

    benchmark_start_step = state["step"]
    benchmark_done = False
    artifact_writer = BackgroundArtifactWriter()
    metric_artifact_writer = BackgroundArtifactWriter()
    steps_per_epoch = max(1, len(datasets["train"]) // args.bs) if drop_remainder else max(1, (len(datasets["train"]) + args.bs - 1) // args.bs)
    total_train_steps = steps_per_epoch * max(1, args.epochs - state["epoch"])
    try:
      for epoch in range(state["epoch"], args.epochs):
        logger.info("Epoch %d:", epoch + 1)
        totals: Dict[str, float] = {}
        seen = 0
        diagnostics = {"grad_norm": 0.0, "grad_clipped": 0.0, "lr_scale": 1.0, "update_skipped": 0.0}
        epoch_t0 = epoch_step_t0 = speed_window_t0 = time.perf_counter()
        speed_window_step = 0
        speed_window_samples = 0

        for batch_index, raw_batch in enumerate(
            _epoch_batches(
                datasets["train"],
                args.bs,
                shuffle=True,
                drop_last=drop_remainder,
                rng=rng,
                **_prefetch_kwargs(args),
            ),
            start=1,
        ):
            fetch_t0 = time.perf_counter()
            batch = _preprocess_counterfactual_batch(args, raw_batch)
            batch_ready_t0 = time.perf_counter()
            bs = int(batch["x"].shape[0])
            # Randomly intervene on one permitted causal variable each step.
            do_k = _choose_intervention(args, dag_vars)
            do = _make_intervention(args, batch, do_k, train_samples, train=True)
            step_rng = jax.random.PRNGKey(args.seed + state["step"] + batch_index + epoch * 1000)
            if use_tpu_pmap:
                batch = _shard_batch(batch, devices)
                do = _shard_batch(do, devices)
                vae_params, opt_state, lmbda, lambda_opt_state, out = train_step(
                    state["vae_params"], state["opt_state"], state["lmbda"], state["lambda_opt_state"],
                    batch, do, jax.random.split(step_rng, device_count), replicated_pgm_bundle,
                    replicated_predictor_bundle, state["step"],
                )
                out = _unreplicate(out)
            else:
                vae_params, opt_state, lmbda, lambda_opt_state, out = train_step(
                    state["vae_params"], state["opt_state"], state["lmbda"], state["lambda_opt_state"],
                    batch, do, step_rng, state["step"],
                )
            state["vae_params"] = vae_params
            state["opt_state"] = opt_state
            state["lmbda"] = lmbda
            state["lambda_opt_state"] = lambda_opt_state
            if float(out["update_skipped"]) == 0.0:
                state["ema"].update(state["vae_params"])
                state["ema_params"] = tree_copy(state["ema"].params)
            seen += bs
            state["step"] += 1
            totals["loss"] = totals.get("loss", 0.0) + float(out["loss"]) * bs
            totals["aux_loss"] = totals.get("aux_loss", 0.0) + float(out["aux_loss"]) * args.alpha * bs
            totals["elbo"] = totals.get("elbo", 0.0) + float(out["elbo"]) * bs
            totals["nll"] = totals.get("nll", 0.0) + float(out["nll"]) * bs
            totals["kl"] = totals.get("kl", 0.0) + float(out["kl"]) * bs
            diagnostics = {
                "grad_norm": float(out.get("grad_norm", 0.0)),
                "grad_clipped": float(out.get("grad_clipped", 0.0)),
                "lr_scale": float(out.get("lr_scale", 1.0)),
                "update_skipped": float(out.get("update_skipped", 0.0)),
            }
            if batch_index % max(1, args.speed_log_freq) == 0:
                sync_t0 = time.perf_counter()
                window_steps = batch_index - speed_window_step
                step_dt = (sync_t0 - speed_window_t0) / max(1, window_steps)
                data_dt = batch_ready_t0 - fetch_t0
                compute_dt = max(0.0, step_dt - data_dt)
                iter_per_sec = 1.0 / max(step_dt, 1e-12)
                sample_per_sec = (seen - speed_window_samples) / max(sync_t0 - speed_window_t0, 1e-12)
                epoch_elapsed = sync_t0 - epoch_step_t0
                epoch_iter_per_sec = batch_index / max(epoch_elapsed, 1e-12)
                epoch_sample_per_sec = seen / max(epoch_elapsed, 1e-12)
                train_steps_done = (epoch - state["epoch"]) * steps_per_epoch + batch_index
                eta_sec = max(0, total_train_steps - train_steps_done) / max(epoch_iter_per_sec, 1e-12)
                train_stats = {key: value / max(1, seen) for key, value in totals.items()}
                logger.info(
                    "epoch=%d step=%d/%d global_step=%d %s data_time=%.2fs compute_time=%.2fs step_time=%.2fs iter/s=%.3f sample/s=%.3f epoch_iter/s=%.3f epoch_sample/s=%.3f eta=%.1fs",
                    epoch + 1, batch_index, steps_per_epoch, state["step"],
                    " - ".join(f"{key}: {value:.4f}" for key, value in train_stats.items()),
                    data_dt, compute_dt, step_dt, iter_per_sec, sample_per_sec,
                    epoch_iter_per_sec, epoch_sample_per_sec, eta_sec,
                )
                writer.add_scalar("speed/data_time_sec", data_dt, state["step"])
                writer.add_scalar("speed/compute_time_sec", compute_dt, state["step"])
                writer.add_scalar("speed/step_time_sec", step_dt, state["step"])
                writer.add_scalar("speed/iter_per_sec", iter_per_sec, state["step"])
                writer.add_scalar("speed/sample_per_sec", sample_per_sec, state["step"])
                writer.add_scalar("speed/epoch_iter_per_sec", epoch_iter_per_sec, state["step"])
                writer.add_scalar("speed/epoch_sample_per_sec", epoch_sample_per_sec, state["step"])
                writer.add_scalar("speed/eta_sec", eta_sec, state["step"])
                for key, value in diagnostics.items():
                    writer.add_scalar(f"train/{key}", value, state["step"])
                writer.add_scalar("train/lmbda", _host_lmbda(state, replicated=use_tpu_pmap), state["step"])
                speed_window_t0 = sync_t0
                speed_window_step = batch_index
                speed_window_samples = seen

            if args.benchmark_steps > 0 and state["step"] - benchmark_start_step >= args.benchmark_steps:
                benchmark_done = True
                break

        if benchmark_done:
            logger.info("Benchmark completed after %d training step(s).", args.benchmark_steps)
            break

        train_stats = {k: v / max(1, seen) for k, v in totals.items()}
        train_time = time.perf_counter() - epoch_step_t0
        checkpoint_due = _checkpoint_due(epoch + 1, args.checkpoint_freq)
        validation_due = _eval_due(epoch + 1, int(getattr(args, "eval_freq", args.checkpoint_freq)))
        validation: Dict[str, tuple[Dict[str, float], Dict[str, float]]] | None = None
        if validation_due:
            eval_state = _portable_cf_state(state, replicated=use_tpu_pmap)
            validation = _evaluate_interventions(
                args,
                datasets,
                eval_state,
                vae_bundle,
                pgm_bundle,
                predictor_bundle,
                eval_step,
                train_samples,
                rng,
                dag_vars,
                max_batches=_bounded_validation_batches(args),
            )
        epoch_iter_per_sec = steps_per_epoch / max(train_time, 1e-12)
        epoch_sample_per_sec = seen / max(train_time, 1e-12)
        total_time = time.perf_counter() - epoch_t0
        _write_epoch_summary(
            writer, epoch=epoch + 1, step=state["step"], train_stats=train_stats,
            lmbda=_host_lmbda(state, replicated=use_tpu_pmap), diagnostics=diagnostics,
            train_time=train_time, total_time=total_time,
            iter_per_sec=epoch_iter_per_sec, sample_per_sec=epoch_sample_per_sec,
            validation=validation,
        )
        _log_epoch_summary(
            logger, epoch=epoch + 1, step=state["step"], train_stats=train_stats,
            lmbda=_host_lmbda(state, replicated=use_tpu_pmap), diagnostics=diagnostics,
            train_time=train_time, total_time=total_time,
            iter_per_sec=epoch_iter_per_sec, sample_per_sec=epoch_sample_per_sec,
            validation=validation,
        )
        if checkpoint_due and validation is not None and "observational" in validation:
            observational_stats, _ = validation["observational"]
            if observational_stats["loss"] < state["best_loss"]:
                state["best_loss"] = observational_stats["loss"]
                _submit_best_cf_checkpoint(
                    artifact_writer, args,
                    _portable_cf_state(state, replicated=use_tpu_pmap), epoch + 1,
                )
                logger.info("Model checkpoint enqueued: %s queue=%s", args.checkpoint_dir, artifact_writer.stats)
        writer.flush()
        if checkpoint_due and getattr(args, "remote_save_dir", ""):
            metric_artifact_writer.submit(_sync_metric_artifacts, args)
            logger.info("metric_artifacts_enqueued epoch=%d step=%d queue=%s", epoch + 1, state["step"], metric_artifact_writer.stats)
      if getattr(args, "final_eval_full", False) and not benchmark_done:
        logger.info("final_full_validation=started split=valid")
        final_validation = _evaluate_interventions(
            args,
            datasets,
            _portable_cf_state(state, replicated=use_tpu_pmap),
            vae_bundle,
            pgm_bundle,
            predictor_bundle,
            eval_step,
            train_samples,
            rng,
            dag_vars,
            max_batches=None,
        )
        for intervention, (stats, metrics) in final_validation.items():
            description = " - ".join(
                f"{key}: {value:.4f}" for key, value in {**stats, **metrics}.items()
            )
            logger.info("=> final valid %s | %s - steps: %d", intervention, description, state["step"])
    finally:
      try:
        artifact_writer.close()
      finally:
        try:
          metric_artifact_writer.close()
        finally:
          writer.close()


def run(config: ExperimentConfig) -> str:
    """Run counterfactual fine-tuning directly from a standalone config."""
    workflow = config.workflow
    assert isinstance(workflow, CounterfactualTrainingConfig)
    from training.settings import _get_schema
    schema = _get_schema(config.dataset.name, config)
    scm_checkpoint, predictor_checkpoint, image_model_checkpoint = validate_stage_artifacts(
        workflow.scm_checkpoint,
        workflow.predictor_checkpoint,
        workflow.image_model_checkpoint,
        remote_root=config.artifacts.remote_root,
        dataset_name=config.dataset.name,
        expected_variables=list(schema.variable_names),
        expected_context_dim=schema.encoded_dim,
        expected_image_context_dim=config.model.context_dim or schema.encoded_dim,
        expected_input_res=config.dataset.input_res,
        prefer_remote=config.dataset.name == "padchest",
        require_remote=config.dataset.name == "padchest",
        resolve_steps=config.dataset.name == "padchest",
        require_complete=config.dataset.name == "padchest",
        allow_incomplete=workflow.trust_incomplete_checkpoint,
        strict_schema=config.dataset.name == "padchest",
    )
    args = _run_arguments(config)
    args.pgm_path = scm_checkpoint
    args.predictor_path = predictor_checkpoint
    args.vae_path = image_model_checkpoint
    main(args)
    return str(output_dir(config))
