"""Stage 2: train the image-to-causal-variable predictor.

The predictor is trained independently from the SCM. It maps an observed image
to thickness, intensity, and digit distributions, then supplies the auxiliary
counterfactual constraint used by Stage 4.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from contracts import CausalGraphSpec
from causal.image_parent_predictor import MorphoMNISTSupAuxPredictor
from config import ExperimentConfig, PredictorTrainingConfig
from data.morphomnist import morphomnist
from data.padchest import padchest
from utils import (
    BackgroundArtifactWriter, SummaryWriter, checkpoint_root_dir, ensure_dir,
    ensure_parent_dir, experiment_run_dir, load_checkpoint, local_staging_path,
    open_file, seed_all, sync_file, tree_copy,
)

from .common import epoch_batches, stage_run_dir, to_jax_batch


@dataclass
class PredictorRunArguments:
    """Runtime-only predictor settings constructed from the typed config."""

    accelerator: str
    gpu_id: str | None
    precision: str
    exp_name: str
    dataset: str
    data_dir: str
    ckpt_dir: str
    remote_ckpt_dir: str
    seed: int
    epochs: int
    bs: int
    lr: float
    wd: float
    input_res: int
    pad: int
    metadata: str = ""
    image_prefix: str = ""
    input_channels: int = 1
    sup_frac: float = 1.0
    std_fixed: float = 0.0
    eval_freq: int = 1
    speed_log_freq: int = 50
    checkpoint_freq: int = 1
    execution_mode: str = "auto"
    drop_remainder: bool = True
    widths: list[int] | None = None
    plot_samples: int = 10000
    benchmark_steps: int = 0
    load_path: str = ""
    deterministic: bool = False
    testing: bool = False
    predictor_model: str = "morphomnist_image_parent_predictor"
    # Artifact identity only; native code never uses this to select a path.
    setup: str = "sup_aux"
    parents_x: list[str] | None = None
    context_norm: str = ""
    context_dim: int = 0
    concat_pa: bool = False
    freeze_backbone: bool = True
    backbone_lr_scale: float = 1.0
    pretrained_weights_path: str = "gs://cxr-rait/checkpoints/pretrained/torchxrayvision_densenet121_flax.npz"
    warmup_epochs: int = 0
    label_smoothing: float = 0.0
    torchxray_preprocessing: bool = False
    dropout_rate: float = 0.0
    normalization_sample_size: int = 0
    input_cache: str = "auto"
    input_cache_dir: str = ""
    input_cache_max_items: int = 2048
    input_prefetch_workers: int = 8
    input_prefetch_batches: int = 2
    augment: bool = True
    type: str = "train-predictor"
    save_dir: str = ""
    checkpoint_dir: str = ""
    remote_save_dir: str = ""



class _IndexedDataset:
    def __init__(self, dataset: Any, indices: np.ndarray):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.min_max = getattr(dataset, "min_max", {})
        self.samples = getattr(dataset, "samples", {})
        self.cache_fingerprint = f"{getattr(dataset, 'cache_fingerprint', type(dataset).__name__)}-indexed-{_hash_array(self.indices)}"

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int):
        return self.dataset[int(self.indices[int(index)])]

    def make_batch(self, indices: Any, rng=None, shuffle: bool = False):
        return self.dataset.make_batch(self.indices[np.asarray(indices, dtype=np.int64)], rng=rng, shuffle=shuffle)


def _hash_array(values: np.ndarray) -> str:
    digest = hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()
    return digest[:16]


class _CachedItemDataset:
    """Bounded item-level cache/prefetch adapter for image-backed datasets.

    The adapter is intentionally opt-in/auto-selected for known remote image
    datasets. It preserves the generic ``__getitem__`` contract and never embeds
    source paths or credentials in cache filenames.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        cache_dir: str,
        max_items: int,
        workers: int,
        name: str,
    ):
        self.dataset = dataset
        self.min_max = getattr(dataset, "min_max", {})
        self.samples = getattr(dataset, "samples", {})
        self.workers = max(1, int(workers))
        self.max_items = max(0, int(max_items))
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir and self.max_items else None
        self.name = name
        fingerprint = getattr(dataset, "cache_fingerprint", None) or getattr(dataset, "fingerprint", None)
        if callable(fingerprint):
            fingerprint = fingerprint()
        self.fingerprint = hashlib.sha256(f"{name}|{fingerprint or type(dataset).__name__}|{len(dataset)}".encode()).hexdigest()[:16]
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self._get_item(int(index))

    def _cache_path(self, index: int) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / self.fingerprint / f"{index:012d}.npz"

    def _load_cached(self, path: Path) -> Dict[str, np.ndarray] | None:
        try:
            if not path.is_file():
                return None
            with np.load(path, allow_pickle=False) as data:
                sample = {key: np.asarray(data[key]) for key in data.files}
            os.utime(path, None)
            return sample
        except Exception:
            return None

    def _save_cached(self, path: Path, sample: Dict[str, np.ndarray]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
            os.close(fd)
            try:
                with open(tmp, "wb") as handle:
                    np.savez_compressed(handle, **{key: np.asarray(value) for key, value in sample.items()})
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            self._evict_if_needed(path.parent)
        except Exception:
            return

    def _evict_if_needed(self, directory: Path) -> None:
        if self.max_items <= 0:
            return
        files = sorted(directory.glob("*.npz"), key=lambda p: p.stat().st_mtime)
        overflow = len(files) - self.max_items
        for victim in files[:max(0, overflow)]:
            try:
                victim.unlink()
            except OSError:
                pass

    def _get_item(self, index: int) -> Dict[str, np.ndarray]:
        path = self._cache_path(index)
        if path is not None:
            cached = self._load_cached(path)
            if cached is not None:
                return cached
        sample = {key: np.asarray(value) for key, value in self.dataset[index].items()}
        if path is not None:
            self._save_cached(path, sample)
        return sample

    def make_batch(self, indices: Any, rng=None, shuffle: bool = False):
        del rng, shuffle
        batch_indices = np.asarray(indices, dtype=np.int64)
        if self.workers > 1 and batch_indices.size > 1:
            with ThreadPoolExecutor(max_workers=min(self.workers, int(batch_indices.size))) as executor:
                samples = list(executor.map(lambda i: self._get_item(int(i)), batch_indices))
        else:
            samples = [self._get_item(int(i)) for i in batch_indices]
        return {key: np.stack([np.asarray(sample[key]) for sample in samples], axis=0) for key in samples[0]}


@dataclass
class WarmupEMA:
    params: Any
    batch_stats: Any
    step: int = 0
    initted: bool = False
    beta: float = 0.999
    update_after_step: int = 100
    inv_gamma: float = 1.0
    power: float = 2.0 / 3.0
    min_value: float = 0.0

    @classmethod
    def init_from(cls, params: Any, batch_stats: Any) -> "WarmupEMA":
        return cls(params=tree_copy(params), batch_stats=tree_copy(batch_stats), step=0, initted=False)

    def update(self, params: Any, batch_stats: Any) -> None:
        self.step += 1
        decay = float(np.clip(1.0 - (1.0 + self.step / self.inv_gamma) ** (-self.power), self.min_value, self.beta)) if self.step < self.update_after_step else self.beta
        if not self.initted:
            self.params, self.batch_stats, self.initted = tree_copy(params), tree_copy(batch_stats), True
        epoch = max(self.step - self.update_after_step - 1, 0)
        decay = 0.0 if epoch <= 0 else min(max(1.0 - (1.0 + epoch / self.inv_gamma) ** (-self.power), self.min_value), self.beta)
        self.params = jax.tree_util.tree_map(lambda ema, value: ema * decay + value * (1.0 - decay), self.params, params)
        self.batch_stats = jax.tree_util.tree_map(lambda ema, value: ema * decay + value * (1.0 - decay), self.batch_stats, batch_stats)


def output_dir(config: ExperimentConfig) -> Path:
    return stage_run_dir(config)


def validate_artifacts(run_dir: str | Path) -> None:
    root = Path(run_dir)
    required = (root / "checkpoints" / "hparams.json", root / "trainlog.txt")
    missing = [str(path) for path in required if not path.is_file()]
    if not list(root.glob("events.out.tfevents.*")):
        missing.append(f"{root}/events.out.tfevents.*")
    if not list((root / "checkpoints").glob("[0-9]*/_CHECKPOINT_METADATA")):
        missing.append(f"{root}/checkpoints/<step>/_CHECKPOINT_METADATA")
    if missing:
        raise RuntimeError(f"Predictor run is missing required artifacts: {', '.join(missing)}")


def _run_arguments(config: ExperimentConfig) -> PredictorRunArguments:
    workflow = config.workflow
    assert isinstance(workflow, PredictorTrainingConfig)
    freeze_backbone = getattr(workflow, "freeze_backbone", True)
    if workflow.type == "finetune-predictor" and not hasattr(config.workflow, "freeze_backbone"):
        freeze_backbone = False
    backbone_lr_scale = float(getattr(workflow, "backbone_lr_scale", 1.0))
    augment = getattr(config.dataset, "augment", True)
    return PredictorRunArguments(
        accelerator=config.runtime.accelerator, gpu_id=config.runtime.gpu_id,
        precision=config.runtime.precision, exp_name=config.artifacts.run_name,
        dataset=config.dataset.name, data_dir=config.dataset.root,
        metadata=config.dataset.metadata, image_prefix=config.dataset.image_prefix,
        ckpt_dir=config.artifacts.root, remote_ckpt_dir=config.artifacts.remote_root,
        seed=config.seed, epochs=workflow.epochs, bs=config.optimizer.batch_size,
        lr=config.optimizer.lr, wd=config.optimizer.weight_decay,
        input_res=config.dataset.input_res, pad=config.dataset.pad,
        checkpoint_freq=workflow.checkpoint_freq, speed_log_freq=workflow.speed_log_freq,
        execution_mode=workflow.execution_mode, drop_remainder=workflow.drop_remainder,
        widths=[32, 32],
        freeze_backbone=freeze_backbone,
        backbone_lr_scale=backbone_lr_scale,
        pretrained_weights_path=workflow.pretrained_weights_path,
        warmup_epochs=getattr(workflow, "warmup_epochs", 0),
        label_smoothing=getattr(workflow, "label_smoothing", 0.0),
        torchxray_preprocessing=getattr(workflow, "torchxray_preprocessing", False),
        dropout_rate=getattr(workflow, "dropout_rate", 0.0),
        normalization_sample_size=getattr(workflow, "normalization_sample_size", 0),
        input_cache=getattr(workflow, "input_cache", "auto"),
        input_cache_dir=getattr(workflow, "input_cache_dir", ""),
        input_cache_max_items=getattr(workflow, "input_cache_max_items", 2048),
        input_prefetch_workers=getattr(workflow, "input_prefetch_workers", 8),
        input_prefetch_batches=getattr(workflow, "input_prefetch_batches", 2),
        augment=augment,
        type=workflow.type,
        predictor_model=workflow.predictor_model,
    )


def _schema_for_dataset(dataset: str) -> CausalGraphSpec:
    if dataset == "morphomnist":
        from data.morphomnist import MORPHOMNIST_SCHEMA
        return MORPHOMNIST_SCHEMA
    if dataset == "cxr_rait":
        from data.cxr_rait import CXR_RAIT_SCHEMA
        return CXR_RAIT_SCHEMA
    if dataset == "padchest":
        from data.padchest import PAD_CHEST_SCHEMA
        return PAD_CHEST_SCHEMA
    raise ValueError(
        "Predictor training currently supports dataset=morphomnist, dataset=cxr_rait, or dataset=padchest"
    )


def _validate_scope(args: PredictorRunArguments) -> None:
    _schema_for_dataset(args.dataset)
    if args.accelerator == "cpu" and args.precision != "fp32":
        raise ValueError("CPU predictor training requires precision=fp32")


def _configure_dataset_args(args: PredictorRunArguments) -> None:
    schema = _schema_for_dataset(args.dataset)
    args.parents_x = list(schema.variable_names)
    args.context_norm, args.context_dim, args.concat_pa = "[-1,1]", schema.encoded_dim, False


def _build_datasets(args: PredictorRunArguments):
    if args.dataset == "cxr_rait":
        from data.cxr_rait import cxr_rait
        datasets = cxr_rait(args)
    elif args.dataset == "padchest":
        datasets = padchest(args)
    else:
        datasets = morphomnist(args)
    indices = np.arange(len(datasets["train"]))
    rng = np.random.RandomState(1); rng.shuffle(indices)
    return datasets, _IndexedDataset(datasets["train"], indices[:int(args.sup_frac * len(indices))])


def _materialize_pretrained_weights(weights_path: str) -> str:
    """Download the canonical GCS archive into a disposable local read cache."""
    if not weights_path.startswith("gs://"):
        raise ValueError(
            "pretrained_weights_path must be a gs:// URI so GCS remains the source of truth; "
            f"got {weights_path!r}"
        )
    local_path = local_staging_path(weights_path)
    ensure_parent_dir(local_path)
    with open_file(weights_path, "rb") as source, open(local_path, "wb") as destination:
        shutil.copyfileobj(source, destination)
    return local_path


def _validate_runtime_device(args: PredictorRunArguments) -> jax.Device:
    """Keep the legacy accelerator preflight before allocating the predictor."""
    devices = jax.devices()
    if args.accelerator == "gpu":
        matching = [device for device in devices if device.platform in {"gpu", "cuda"}]
        if not matching:
            raise RuntimeError("accelerator=gpu requested, but JAX found no CUDA GPU")
        if len(matching) != 1:
            raise RuntimeError(f"Predictor training requires one visible GPU, found {len(matching)}")
    else:
        matching = [device for device in devices if device.platform == args.accelerator]
        if not matching:
            raise RuntimeError(f"accelerator={args.accelerator} requested, but JAX devices are {devices}")
    device = matching[0]
    print(f"JAX device preflight passed: platform={device.platform} device={device}")
    return device


def _compute_dtype(args: PredictorRunArguments) -> jnp.dtype:
    dtype = jnp.bfloat16 if args.precision == "bf16" and args.accelerator in {"gpu", "tpu"} else jnp.float32
    matmul_precision = "default" if dtype == jnp.bfloat16 else "highest"
    jax.config.update("jax_default_matmul_precision", matmul_precision)
    print(
        "JAX predictor compute policy: "
        f"precision={args.precision} compute_dtype={dtype} "
        f"master_params=fp32 optimizer_state=fp32 matmul_precision={matmul_precision}"
    )
    return dtype


def _first_local_replica(value: Any) -> Any:
    """Read one local pmap replica without a cross-device gather."""
    if isinstance(value, jax.Array) and value.addressable_shards:
        return jnp.squeeze(value.addressable_shards[0].data, axis=0)
    return value[0]


def _unreplicate(tree: Any) -> Any:
    return jax.tree_util.tree_map(_first_local_replica, tree)


def _replicate(tree: Any, devices: list[jax.Device]) -> Any:
    mesh = jax.sharding.Mesh(np.asarray(devices), ("devices",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("devices"))

    def _put(value: Any) -> jax.Array:
        copies = np.stack([np.asarray(value)] * len(devices), axis=0)
        return jax.device_put(copies, sharding)

    return jax.tree_util.tree_map(_put, tree)


def _shard_batch(batch: Dict[str, jax.Array], devices: list[jax.Device]) -> Dict[str, jax.Array]:
    device_count = len(devices)
    mesh = jax.sharding.Mesh(np.asarray(devices), ("devices",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("devices"))

    def _put(value: jax.Array) -> jax.Array:
        value = jnp.asarray(value)
        if value.shape[0] % device_count:
            raise ValueError(
                f"Global batch size {value.shape[0]} must be divisible by TPU local device count {device_count}."
            )
        per_device_batch = value.shape[0] // device_count
        return jax.device_put(value.reshape((device_count, per_device_batch) + value.shape[1:]), sharding)

    return jax.tree_util.tree_map(_put, batch)


def _use_tpu_replication(args: PredictorRunArguments) -> bool:
    """Resolve the predictor execution mode for the current local TPU topology."""
    requested_mode = getattr(args, "execution_mode", "auto")
    local_device_count = jax.local_device_count()
    multi_tpu_available = args.accelerator == "tpu" and local_device_count > 1
    if requested_mode == "replicated" and not multi_tpu_available:
        raise ValueError("execution_mode=replicated requires accelerator=tpu with multiple local devices")
    return multi_tpu_available and requested_mode != "single_device"


def _merge(graphdef: Any, params: Any, batch_stats: Any, rng_state: Any | None = None):
    """Reconstruct a predictor, including mutable NNX Dropout RNG state when present."""
    if rng_state is None:
        return nnx.merge(graphdef, params, batch_stats)
    return nnx.merge(graphdef, params, batch_stats, rng_state)


def _model_parent_variables(model: Any) -> tuple[str, ...]:
    parents = getattr(model, "parent_variables", None)
    if parents is not None:
        return tuple(parents)
    return tuple(getattr(model, "variables", {}).keys())


def _batch_for_model(model: Any, batch: Dict[str, jax.Array]) -> Dict[str, jax.Array]:
    """Return exactly the image and parent variables required by a predictor."""
    required = ("x",) + _model_parent_variables(model)
    missing = [key for key in required if key not in batch]
    if missing:
        available = sorted(batch.keys())
        raise KeyError(
            f"Predictor {type(model).__name__} requires batch keys {missing}; available={available}"
        )
    variables = getattr(model, "variables", {})
    result = {"x": batch["x"]}
    for key in required[1:]:
        value = batch[key]
        if variables.get(key) == "continuous" and getattr(value, "ndim", None) == 1:
            value = value[:, None]
        result[key] = value
    return result


def _loss_and_state(
    graphdef: Any,
    params: Any,
    batch_stats: Any,
    rng_state: Any,
    batch: Dict[str, jax.Array] | None = None,
    *,
    training: bool,
):
    legacy_call = batch is None
    if legacy_call:
        batch = rng_state
        rng_state = None
    model = _merge(graphdef, params, batch_stats, rng_state)
    model.train() if training else model.eval()
    log_probs = model.model_anticausal(**_batch_for_model(model, batch))
    loss = -jnp.mean(log_probs["joint"])
    metrics = {"loss": loss, **{f"logp({key})": jnp.mean(value) for key, value in log_probs.items() if key != "joint"}}
    updated_params = nnx.state(model, nnx.Param).to_pure_dict()
    updated_batch_stats = nnx.state(model, nnx.BatchStat).to_pure_dict()
    if legacy_call:
        return loss, metrics, updated_params, updated_batch_stats
    return (
        loss,
        metrics,
        updated_params,
        updated_batch_stats,
        nnx.state(model, nnx.RngState).to_pure_dict(),
    )


def _scale_backbone_grads(grads: Any, scale: float) -> Any:
    if scale == 1.0:
        return grads
    def _scale(path, val):
        if any("encoder_shared" in str(p) for p in path):
            return val * scale
        return val
    return jax.tree_util.tree_map_with_path(_scale, grads)


def _make_train_step(graphdef: Any, optimizer: optax.GradientTransformation, backbone_lr_scale: float = 1.0):
    """Compile one predictor update, including BatchNorm state evolution."""
    @jax.jit
    def train_step(params, batch_stats, *step_args):
        legacy_call = len(step_args) == 2
        if legacy_call:
            rng_state = None
            opt_state, batch = step_args
        else:
            rng_state, opt_state, batch = step_args

        def loss_fn(current_params):
            if legacy_call:
                loss, metrics, new_params, new_batch_stats = _loss_and_state(
                    graphdef, current_params, batch_stats, batch, training=True
                )
                return loss, (metrics, new_params, new_batch_stats, None)
            loss, metrics, new_params, new_batch_stats, new_rng_state = _loss_and_state(
                graphdef, current_params, batch_stats, rng_state, batch, training=True
            )
            return loss, (metrics, new_params, new_batch_stats, new_rng_state)

        (_, (metrics, _new_params, new_batch_stats, new_rng_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads = _scale_backbone_grads(grads, backbone_lr_scale)
        grad_norm = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        updated_params = optax.apply_updates(params, updates)
        if legacy_call:
            return updated_params, new_batch_stats, opt_state, metrics, grad_norm
        return updated_params, new_batch_stats, new_rng_state, opt_state, metrics, grad_norm
    return train_step


def _make_pmap_train_step(
    graphdef: Any,
    optimizer: optax.GradientTransformation,
    devices: list[jax.Device],
    backbone_lr_scale: float = 1.0,
):
    """Compile a synchronized multi-core TPU predictor update."""
    def train_step(params, batch_stats, rng_state, opt_state, batch):
        def loss_fn(current_params):
            loss, metrics, _new_params, new_batch_stats, new_rng_state = _loss_and_state(
                graphdef, current_params, batch_stats, rng_state, batch, training=True
            )
            return loss, (metrics, new_batch_stats, new_rng_state)

        (_, (metrics, new_batch_stats, new_rng_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads = jax.lax.pmean(grads, axis_name="devices")
        grads = _scale_backbone_grads(grads, backbone_lr_scale)
        metrics = jax.tree_util.tree_map(
            lambda value: jax.lax.pmean(value, axis_name="devices"), metrics
        )
        new_batch_stats = jax.tree_util.tree_map(
            lambda value: jax.lax.pmean(value, axis_name="devices"), new_batch_stats
        )
        grad_norm = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_batch_stats, new_rng_state, opt_state, metrics, grad_norm

    return jax.pmap(
        train_step,
        axis_name="devices",
        in_axes=(0, 0, 0, 0, 0),
        devices=devices,
    )


def _portable_training_state(
    model_params: Any,
    batch_stats: Any,
    rng_state: Any,
    ema: WarmupEMA,
    opt_state: Any,
    *,
    replicated: bool,
) -> tuple[Any, Any, Any, WarmupEMA, Any]:
    """Return ordinary single-device trees for evaluation and persistence."""
    if not replicated:
        return model_params, batch_stats, rng_state, ema, opt_state
    return (
        _unreplicate(model_params),
        _unreplicate(batch_stats),
        _unreplicate(rng_state),
        WarmupEMA(
            params=_unreplicate(ema.params),
            batch_stats=_unreplicate(ema.batch_stats),
            step=ema.step,
            initted=ema.initted,
            beta=ema.beta,
            update_after_step=ema.update_after_step,
            inv_gamma=ema.inv_gamma,
            power=ema.power,
            min_value=ema.min_value,
        ),
        _unreplicate(opt_state),
    )


def _eval_epoch(graphdef: Any, params: Any, batch_stats: Any, rng_state: Any, dataset: Any | None = None, batch_size: int | None = None, rng: np.random.Generator | None = None) -> Dict[str, float]:
    if rng is None:
        rng = batch_size
        batch_size = dataset
        dataset = rng_state
        rng_state = None
    totals: Dict[str, float] = {}; count = 0
    for batch in epoch_batches(dataset, batch_size, shuffle=False, drop_last=False, rng=rng):
        if rng_state is None:
            _, metrics, _, _ = _loss_and_state(graphdef, params, batch_stats, batch, training=False)
        else:
            _, metrics, _, _, _ = _loss_and_state(graphdef, params, batch_stats, rng_state, batch, training=False)
        size = int(next(iter(batch.values())).shape[0])

        for key, value in metrics.items(): totals[key] = totals.get(key, 0.0) + float(value) * size
        count += size
    return {key: value / max(1, count) for key, value in totals.items()}


def _progress_description(mode: str, stats: Dict[str, float], grad_norm: Optional[float] = None) -> str:
    keys = ["loss"] + [k for k in sorted(stats.keys()) if k != "loss" and k.startswith("logp(")]
    description = " => " + mode + " | " + ", ".join(
        f"{key}: {stats[key]:.4f}" for key in keys if key in stats
    )
    return description if grad_norm is None else f"{description}, grad_norm: {grad_norm:.3f}"


def _prediction_description(metrics: Dict[str, float]) -> str:
    return " - ".join(
        f"{key}: {metrics[key]:.4f}" for key in sorted(metrics.keys())
    )


def _checkpoint_due(epoch: int, checkpoint_freq: int) -> bool:
    """Return whether a completed one-based epoch may persist a checkpoint."""
    return epoch % max(1, checkpoint_freq) == 0


def _submit_best_checkpoint(
    artifact_writer: BackgroundArtifactWriter,
    args: PredictorRunArguments,
    model_params: Any,
    batch_stats: Any,
    ema: WarmupEMA,
    opt_state: Any,
    epoch: int,
    step: int,
    best_loss: float,
) -> None:
    """Snapshot an improved predictor state and enqueue its persistence."""
    payload = _checkpoint_payload(
        args, model_params, batch_stats, ema, opt_state, epoch, step, best_loss
    )
    remote_checkpoint_dir = (
        os.path.join(args.remote_save_dir, "checkpoints")
        if args.remote_save_dir
        else None
    )
    artifact_writer.submit_checkpoint(
        payload,
        args.checkpoint_dir,
        step=step,
        custom_metadata={"epoch": epoch, "best_loss": best_loss},
        local_tree_dir=args.checkpoint_dir if remote_checkpoint_dir else None,
        remote_tree_dir=remote_checkpoint_dir,
    )


def _writer_add_custom_scalars(writer: Any) -> None:
    if hasattr(writer, "add_custom_scalars"):
        writer.add_custom_scalars({"elbo": {"elbo": ["Multiline", ["elbo/train", "elbo/valid"]]}})


def _sync_tensorboard_artifacts(args: PredictorRunArguments) -> None:
    """Copy all flushed TensorBoard event files to the configured remote run."""
    if not args.remote_save_dir:
        return
    for event_path in sorted(Path(args.save_dir).glob("events.out.tfevents.*")):
        sync_file(str(event_path), os.path.join(args.remote_save_dir, event_path.name))


def _sync_metric_artifacts(args: PredictorRunArguments) -> None:
    """Synchronize the flushed train log and TensorBoard events for one checkpoint interval."""
    if not args.remote_save_dir:
        return
    sync_file(
        os.path.join(args.save_dir, "trainlog.txt"),
        os.path.join(args.remote_save_dir, "trainlog.txt"),
    )
    _sync_tensorboard_artifacts(args)


def _write_epoch_summary(
    writer: Any,
    *,
    epoch: int,
    step: int,
    train_stats: Dict[str, float],
    valid_stats: Dict[str, float],
    prediction_stats: Dict[str, float],
    train_time: float,
    total_time: float,
    iter_per_sec: float,
    sample_per_sec: float,
    train_prediction_stats: Optional[Dict[str, float]] = None,
) -> None:
    """Persist the complete completed-epoch predictor summary to TensorBoard."""
    for key, value in train_stats.items():
        writer.add_scalar(f"train/{key}", value, step)
    if train_prediction_stats:
        for key, value in train_prediction_stats.items():
            writer.add_scalar(f"train/{key}", value, step)
    for key, value in valid_stats.items():
        writer.add_scalar(f"valid/{key}", value, step)
    for key, value in prediction_stats.items():
        writer.add_scalar(f"valid/{key}", value, step)
    writer.add_scalar("elbo/train", train_stats["loss"], step)
    writer.add_scalar("elbo/valid", valid_stats["loss"], step)
    writer.add_scalar("epoch/number", epoch, step)
    writer.add_scalar("epoch/global_step", step, step)
    writer.add_scalar("epoch/train_time_sec", train_time, step)
    writer.add_scalar("epoch/total_time_sec", total_time, step)
    writer.add_scalar("epoch/iter_per_sec", iter_per_sec, step)
    writer.add_scalar("epoch/sample_per_sec", sample_per_sec, step)


def _log_epoch_summary(
    logger: logging.Logger,
    *,
    epoch: int,
    step: int,
    train_stats: Dict[str, float],
    valid_stats: Dict[str, float],
    prediction_stats: Dict[str, float],
    train_time: float,
    total_time: float,
    iter_per_sec: float,
    sample_per_sec: float,
    train_prediction_stats: Optional[Dict[str, float]] = None,
) -> None:
    """Write the same completed-epoch metrics to the console and train log."""
    train_desc = _progress_description("train", train_stats).strip()
    if train_prediction_stats:
        train_desc = f"{train_desc} - {_prediction_description(train_prediction_stats)}"
    logger.info(
        "%s - steps: %d - it/s: %.3f - samples/s: %.3f",
        train_desc, step, iter_per_sec, sample_per_sec,
    )
    logger.info(
        "%s - %s - steps: %d",
        _progress_description("valid", valid_stats).strip(),
        _prediction_description(prediction_stats),
        step,
    )
    logger.info(
        "epoch=%d train_time=%.1fs total_time=%.1fs epoch_iter/s=%.3f epoch_sample/s=%.3f",
        epoch, train_time, total_time, iter_per_sec, sample_per_sec,
    )


def _prediction_metrics(args: PredictorRunArguments, model: Any, dataset: Any, batch_size: int, rng: np.random.Generator) -> Dict[str, float]:
    model.eval(); predictions = {key: [] for key in model.variables}; targets = {key: [] for key in model.variables}
    for batch in epoch_batches(dataset, batch_size, shuffle=False, drop_last=False, rng=rng):
        model_batch = _batch_for_model(model, batch)
        for key in targets: targets[key].extend(np.asarray(model_batch[key]))
        for key, value in model.predict(**model_batch).items(): predictions[key].extend(np.asarray(value))
    stats: Dict[str, float] = {}
    for key, var_kind in getattr(model, "variables", {}).items():
        target_arr = np.asarray(targets[key])
        prediction_arr = np.asarray(predictions[key])
        if var_kind == "categorical" or key == "digit":
            stats[f"{key}_acc"] = float((target_arr.argmax(-1) == prediction_arr.argmax(-1)).mean())
        elif var_kind == "binary":
            stats[f"{key}_acc"] = float(((prediction_arr.squeeze(-1) >= 0.5) == (target_arr.squeeze(-1) >= 0.5)).mean())
        else:
            if hasattr(dataset, "min_max") and key in dataset.min_max:
                low, high = dataset.min_max[key]
                prediction = ((prediction_arr.squeeze(-1) + 1.0) / 2.0) * (high - low) + low
                target = ((target_arr.squeeze(-1) + 1.0) / 2.0) * (high - low) + low
            else:
                prediction = prediction_arr.squeeze(-1)
                target = target_arr.squeeze(-1)
            stats[f"{key}_mae"] = float(np.mean(np.abs(target - prediction)))
    return stats


def _checkpoint_payload(args: PredictorRunArguments, model_params: Any, batch_stats: Any, ema: WarmupEMA, opt_state: Any, epoch: int, step: int, best_loss: float) -> Dict[str, Any]:
    return {"params": ema.params, "ema_params": ema.params, "model_params": model_params, "batch_stats": batch_stats, "ema_batch_stats": ema.batch_stats, "opt_state": opt_state, "epoch": epoch, "step": step, "best_loss": best_loss, "ema_step": ema.step, "ema_initted": ema.initted, "hparams": vars(args), "format_version": 3}


def _restore_args(args: PredictorRunArguments, checkpoint: Dict[str, Any]) -> None:
    preserved = {key: getattr(args, key) for key in ("accelerator", "precision", "gpu_id", "data_dir", "load_path", "testing", "remote_ckpt_dir")}
    for key, value in checkpoint.get("hparams", {}).items():
        if hasattr(args, key): setattr(args, key, value)
    for key, value in preserved.items(): setattr(args, key, value)


def _assert_compatible_checkpoint(checkpoint: Dict[str, Any], params: Any, batch_stats: Any) -> None:
    if checkpoint.get("format_version") != 3 or checkpoint.get("hparams", {}).get("setup") != "sup_aux":
        raise ValueError("Checkpoint is not a compatible predictor artifact")
    if jax.tree_util.tree_structure(checkpoint["model_params"]) != jax.tree_util.tree_structure(params) or jax.tree_util.tree_structure(checkpoint["batch_stats"]) != jax.tree_util.tree_structure(batch_stats):
        raise ValueError("Checkpoint model structure does not match the predictor")


def _setup_logging(args: PredictorRunArguments) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s, %(message)s", datefmt="%d-%b-%y %H:%M:%S", handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(args.save_dir, "trainlog.txt"), mode="a")], force=True)
    logging.getLogger("orbax").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.WARNING)
    return logging.getLogger(args.exp_name or f"{args.dataset}-predictor")


def _predictor_model_family(args: PredictorRunArguments) -> str:
    requested = (getattr(args, "predictor_model", "") or "").strip()
    if requested in {"", "auto"}:
        return "morphomnist" if args.dataset == "morphomnist" else "cxr"
    if requested == "morphomnist_image_parent_predictor" and args.dataset != "morphomnist":
        # Several legacy CXR configs predate the predictor_model key and therefore carry
        # the Pydantic default.  Resolve those by dataset rather than instantiating the
        # MorphoMNIST-only loss contract for CXR parent variables.
        return "cxr"
    if requested in {"morphomnist_image_parent_predictor", "morphomnist_sup_aux_predictor"}:
        return "morphomnist"
    if requested in {"cxr_rait_image_parent_predictor", "cxr_image_parent_predictor", "padchest_image_parent_predictor"}:
        return "cxr"
    raise ValueError(f"Unknown predictor_model={requested!r}")


def _build_predictor_model(args: PredictorRunArguments, dtype: jnp.dtype, logger: logging.Logger | None = None):
    family = _predictor_model_family(args)
    schema = _schema_for_dataset(args.dataset)
    if family == "morphomnist":
        if args.dataset != "morphomnist":
            raise ValueError("The MorphoMNIST predictor can only be used with dataset=morphomnist")
        return MorphoMNISTSupAuxPredictor(
            input_channels=args.input_channels,
            input_res=args.input_res,
            width=8,
            std_fixed=args.std_fixed,
            compute_dtype=dtype,
            rngs=nnx.Rngs(args.seed),
        )

    use_pretrained = args.type == "finetune-predictor" or getattr(args, "pretrained", False)
    legacy_cxr_rait = args.dataset == "cxr_rait" and tuple(schema.variable_names) == ("age", "gender", "tb_status")
    if use_pretrained:
        from causal.cxr_rait_predictor import CxrPretrainedImageParentPredictor, CxrRaitPretrainedPredictor
        local_weights_path = _materialize_pretrained_weights(args.pretrained_weights_path)
        if logger is not None:
            logger.info(
                "pretrained_weights_source=%s local_cache=%s",
                args.pretrained_weights_path,
                local_weights_path,
            )
        model_cls = CxrRaitPretrainedPredictor if legacy_cxr_rait else CxrPretrainedImageParentPredictor
        kwargs = {} if legacy_cxr_rait else {"variable_specs": schema.variables}
        return model_cls(
            **kwargs,
            input_channels=args.input_channels,
            input_res=args.input_res,
            width=32,
            std_fixed=args.std_fixed,
            freeze_backbone=getattr(args, "freeze_backbone", True),
            weights_path=local_weights_path,
            dropout_rate=getattr(args, "dropout_rate", 0.0),
            label_smoothing=getattr(args, "label_smoothing", 0.0),
            compute_dtype=dtype,
            rngs=nnx.Rngs(args.seed),
        )

    from causal.cxr_rait_predictor import CxrImageParentPredictor, CxrRaitSupAuxPredictor
    if legacy_cxr_rait:
        return CxrRaitSupAuxPredictor(
            input_channels=args.input_channels,
            input_res=args.input_res,
            width=16,
            std_fixed=args.std_fixed,
            compute_dtype=dtype,
            rngs=nnx.Rngs(args.seed),
        )
    return CxrImageParentPredictor(
        variable_specs=schema.variables,
        input_channels=args.input_channels,
        input_res=args.input_res,
        width=16,
        std_fixed=args.std_fixed,
        compute_dtype=dtype,
        rngs=nnx.Rngs(args.seed),
    )



def _default_input_cache_dir(args: Any) -> str:
    return os.path.join(tempfile.gettempdir(), "causal-genx-input-cache", str(getattr(args, "dataset", "dataset")))


def _input_cache_enabled(args: Any) -> bool:
    mode = str(getattr(args, "input_cache", "auto"))
    if mode == "on":
        return True
    if mode == "off":
        return False
    return str(getattr(args, "dataset", "")) == "padchest"


def _prepare_input_pipeline(logger: logging.Logger, datasets: Dict[str, Any], train_dataset: Any, args: Any):
    """Apply bounded, configurable adapters for image-backed predictor input."""
    workers = max(1, int(getattr(args, "input_prefetch_workers", 1)))
    max_items = max(0, int(getattr(args, "input_cache_max_items", 0)))
    cache_dir = str(getattr(args, "input_cache_dir", "") or _default_input_cache_dir(args))
    if _input_cache_enabled(args) and max_items > 0:
        train_dataset = _CachedItemDataset(train_dataset, cache_dir=cache_dir, max_items=max_items, workers=workers, name="train")
        for split in ("valid", "test"):
            if split in datasets:
                datasets[split] = _CachedItemDataset(datasets[split], cache_dir=cache_dir, max_items=max_items, workers=workers, name=split)
        logger.info(
            "input_pipeline cache=on cache_dir=%s max_items_per_split=%d workers=%d prefetch_batches=%d",
            cache_dir, max_items, workers, int(getattr(args, "input_prefetch_batches", 0)),
        )
    else:
        logger.info(
            "input_pipeline cache=off workers=%d prefetch_batches=%d",
            workers, int(getattr(args, "input_prefetch_batches", 0)),
        )
    return datasets, train_dataset


def _batch_indices_for_epoch(dataset: Any, batch_size: int, *, shuffle: bool, drop_last: bool, rng: np.random.Generator) -> list[np.ndarray]:
    indices = np.arange(len(dataset), dtype=np.int64)
    if shuffle:
        rng.shuffle(indices)
    batches = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        if drop_last and batch_indices.size < batch_size:
            continue
        batches.append(batch_indices)
    return batches


def _load_predictor_batch(dataset: Any, batch_indices: np.ndarray, *, rng: np.random.Generator, shuffle: bool) -> Dict[str, Any]:
    if hasattr(dataset, "make_batch"):
        return dataset.make_batch(batch_indices, rng=rng, shuffle=shuffle)
    return {
        key: np.stack([np.asarray(dataset[int(index)][key]) for index in batch_indices])
        for key in dataset[0]
    }


def _predictor_epoch_batches(
    dataset: Any,
    batch_size: int,
    *,
    shuffle: bool,
    drop_last: bool,
    rng: np.random.Generator,
    prefetch_batches: int = 0,
) -> Iterator[Dict[str, jax.Array]]:
    """Yield deterministic batches with optional bounded background prefetch."""
    prefetch = max(0, int(prefetch_batches))
    if prefetch <= 0:
        yield from epoch_batches(dataset, batch_size, shuffle=shuffle, drop_last=drop_last, rng=rng)
        return
    batches = _batch_indices_for_epoch(dataset, batch_size, shuffle=shuffle, drop_last=drop_last, rng=rng)

    def submit(executor: ThreadPoolExecutor, batch_indices: np.ndarray) -> Future:
        # Derive seeds sequentially before the asynchronous load to keep sampling deterministic.
        seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        return executor.submit(
            _load_predictor_batch,
            dataset,
            batch_indices,
            rng=np.random.default_rng(seed),
            shuffle=shuffle,
        )

    with ThreadPoolExecutor(max_workers=prefetch) as executor:
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
            yield to_jax_batch(future.result())


def _log_input_normalization(
    logger: logging.Logger,
    dataset: Any,
    train_dataset: Any,
    args: Any,
) -> Dict[str, Any]:
    """Display and log input preprocessing and attribute normalization statistics."""
    logger.info(
        "Input Normalization Check (dataset=%s, context_norm=%s, input_res=%d, pad=%d):",
        args.dataset, getattr(args, "context_norm", "N/A"), args.input_res, args.pad,
    )

    sample_size = max(0, int(getattr(args, "normalization_sample_size", 0)))
    if sample_size:
        try:
            sample_batch = train_dataset.make_batch(np.arange(min(len(train_dataset), sample_size)))
            if "x" in sample_batch:
                x = sample_batch["x"]
                logger.info(
                    "  Image 'x': shape=%s, min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                    tuple(x.shape[1:]), float(np.min(x)), float(np.max(x)), float(np.mean(x)), float(np.std(x)),
                )
        except Exception as err:
            logger.warning("  Image 'x': could not compute bounded sample stats (%s)", err)
    else:
        logger.info("  Image 'x': sample stats skipped (normalization_sample_size=0)")

    min_max = getattr(dataset, "min_max", {})
    samples = getattr(dataset, "samples", {})
    stats_summary = {}

    for var in getattr(args, "parents_x", []) or []:
        if var in samples:
            data = samples[var]
            is_one_hot = (data.ndim == 2 and data.shape[1] > 1) or (var == "digit")
            if is_one_hot:
                dim = data.shape[1] if data.ndim == 2 else 10
                logger.info(
                    "  Variable '%s' (categorical): dim=%d, min=%.4f, max=%.4f",
                    var, dim, float(np.min(data)), float(np.max(data)),
                )
                stats_summary[var] = {"kind": "categorical", "dim": dim}
            else:
                raw_range_str = f"[{min_max[var][0]:.4f}, {min_max[var][1]:.4f}]" if var in min_max else "N/A"
                norm_min, norm_max = float(np.min(data)), float(np.max(data))
                norm_mean, norm_std = float(np.mean(data)), float(np.std(data))
                logger.info(
                    "  Variable '%s' (continuous): raw_min_max=%s, norm_min=%.4f, norm_max=%.4f, norm_mean=%.4f, norm_std=%.4f",
                    var, raw_range_str, norm_min, norm_max, norm_mean, norm_std,
                )
                stats_summary[var] = {
                    "kind": "continuous",
                    "raw_min_max": min_max.get(var),
                    "norm_min": norm_min,
                    "norm_max": norm_max,
                    "norm_mean": norm_mean,
                    "norm_std": norm_std,
                }
    return stats_summary


def _run(args: PredictorRunArguments) -> Dict[str, float]:
    """Execute predictor resume → supervised training → EMA validation → save."""
    checkpoint: Optional[Dict[str, Any]] = load_checkpoint(args.load_path) if args.load_path else None
    if checkpoint is not None: _restore_args(args, checkpoint)
    _validate_scope(args); _validate_runtime_device(args); dtype = _compute_dtype(args); _configure_dataset_args(args); seed_all(args.seed, args.deterministic)
    args.save_dir = experiment_run_dir(args.ckpt_dir, args.dataset, args.exp_name, "pgm")
    args.checkpoint_dir = checkpoint_root_dir(args.save_dir); args.remote_save_dir = experiment_run_dir(args.remote_ckpt_dir, args.dataset, args.exp_name, "pgm")

    ensure_dir(args.save_dir); ensure_dir(args.checkpoint_dir)
    logger = _setup_logging(args); writer = SummaryWriter(args.save_dir); datasets, train_dataset = _build_datasets(args)
    datasets, train_dataset = _prepare_input_pipeline(logger, datasets, train_dataset, args)
    valid_dataset = datasets["valid"]
    _log_input_normalization(logger, datasets["train"], train_dataset, args)
    model = _build_predictor_model(args, dtype, logger)

    graphdef, params_state, batch_stats_state, rng_state = nnx.split(
        model, nnx.Param, nnx.BatchStat, nnx.RngState
    )
    model_params = params_state.to_pure_dict()
    model_batch_stats = batch_stats_state.to_pure_dict()
    model_rng_state = rng_state.to_pure_dict()
    optimizer = optax.chain(optax.clip_by_global_norm(200.0), optax.adamw(args.lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=args.wd)); opt_state = optimizer.init(model_params); ema = WarmupEMA.init_from(model_params, model_batch_stats)
    start_epoch = step = 0; best_loss = float("inf")
    if checkpoint is not None:
        _assert_compatible_checkpoint(checkpoint, model_params, model_batch_stats); model_params, model_batch_stats, opt_state = checkpoint["model_params"], checkpoint["batch_stats"], checkpoint["opt_state"]
        ema = WarmupEMA(params=checkpoint.get("ema_params", checkpoint["params"]), batch_stats=checkpoint["ema_batch_stats"], step=int(checkpoint.get("ema_step", checkpoint.get("step", 0))), initted=bool(checkpoint.get("ema_initted", True)))
        start_epoch, step, best_loss = int(checkpoint.get("epoch", 0)), int(checkpoint.get("step", 0)), float(checkpoint.get("best_loss", float("inf")))
    rng = np.random.default_rng(args.seed)
    if args.testing:
        if checkpoint is None: raise ValueError("testing requires load_path")
        stats = _prediction_metrics(
            args, _merge(graphdef, ema.params, ema.batch_stats, model_rng_state),
            datasets["test"], args.bs, rng,
        ); logger.info("test | %s", _prediction_description(stats)); writer.close(); return stats
    for key in sorted(vars(args)):
        logger.info("--%s=%s", key, getattr(args, key))
    logger.info("Data splits: #labelled: %d - #unlabelled: %d", len(train_dataset), len(datasets["train"]) - len(train_dataset))
    use_tpu_pmap = _use_tpu_replication(args)
    devices = jax.local_devices() if use_tpu_pmap else []
    device_count = len(devices) if use_tpu_pmap else 1
    if use_tpu_pmap and args.bs % device_count:
        raise ValueError(
            f"Global batch size {args.bs} must be divisible by TPU local device count {device_count}."
        )
    drop_remainder = bool(getattr(args, "drop_remainder", True) or use_tpu_pmap)
    batches_per_epoch = len(train_dataset) // args.bs if drop_remainder else (len(train_dataset) + args.bs - 1) // args.bs
    total_train_steps = max(1, batches_per_epoch) * max(1, args.epochs - start_epoch)
    if use_tpu_pmap:
        logger.info(
            "execution_mode=replicated local_device_count=%d global_batch_size=%d per_device_batch_size=%d",
            device_count, args.bs, args.bs // device_count,
        )
        backbone_lr_scale = float(getattr(args, "backbone_lr_scale", 1.0))
        train_step = _make_pmap_train_step(graphdef, optimizer, devices, backbone_lr_scale=backbone_lr_scale)
        model_params = _replicate(model_params, devices)
        model_batch_stats = _replicate(model_batch_stats, devices)
        model_rng_state = _replicate(model_rng_state, devices)
        opt_state = _replicate(opt_state, devices)
        ema = WarmupEMA(
            params=_replicate(ema.params, devices), batch_stats=_replicate(ema.batch_stats, devices),
            step=ema.step, initted=ema.initted, beta=ema.beta,
            update_after_step=ema.update_after_step, inv_gamma=ema.inv_gamma,
            power=ema.power, min_value=ema.min_value,
        )
    else:
        logger.info(
            "execution_mode=single_device accelerator=%s local_device_count=%d global_batch_size=%d",
            args.accelerator, jax.local_device_count(), args.bs,
        )
        backbone_lr_scale = float(getattr(args, "backbone_lr_scale", 1.0))
        train_step = _make_train_step(graphdef, optimizer, backbone_lr_scale=backbone_lr_scale)
    final_stats: Dict[str, float] = {}
    artifact_writer = BackgroundArtifactWriter()
    metric_artifact_writer = BackgroundArtifactWriter()
    try:
        for epoch in range(start_epoch, args.epochs):
            logger.info("Epoch %d:", epoch + 1)
            totals: Dict[str, float] = {}; seen = 0
            total_batches = batches_per_epoch
            epoch_t0 = epoch_step_t0 = speed_window_t0 = time.perf_counter()
            speed_window_step = 0
            speed_window_samples = 0
            prefetch_batches = int(getattr(args, "input_prefetch_batches", 0)) if _input_cache_enabled(args) else 0
            for batch_index, batch in enumerate(
                _predictor_epoch_batches(
                    train_dataset,
                    args.bs,
                    shuffle=True,
                    drop_last=drop_remainder,
                    rng=rng,
                    prefetch_batches=prefetch_batches,
                ), start=1
            ):
                if use_tpu_pmap:
                    batch = _shard_batch(batch, devices)
                model_params, model_batch_stats, model_rng_state, opt_state, metrics, grad_norm = train_step(
                    model_params, model_batch_stats, model_rng_state, opt_state, batch
                ); ema.update(model_params, model_batch_stats); size = int(next(iter(batch.values())).shape[0])
                if use_tpu_pmap:
                    metrics = _unreplicate(metrics)
                    grad_norm = _first_local_replica(grad_norm)
                    size *= int(next(iter(batch.values())).shape[1])

                for key, value in metrics.items(): totals[key] = totals.get(key, 0.0) + float(value) * size
                seen += size; step += 1
                if batch_index % max(1, getattr(args, "speed_log_freq", 50)) == 0:
                    sync_t0 = time.perf_counter()
                    window_steps = batch_index - speed_window_step
                    step_dt = (sync_t0 - speed_window_t0) / max(1, window_steps)
                    iter_per_sec = 1.0 / max(step_dt, 1e-12)
                    sample_per_sec = (seen - speed_window_samples) / max(sync_t0 - speed_window_t0, 1e-12)
                    epoch_elapsed = sync_t0 - epoch_step_t0
                    epoch_iter_per_sec = batch_index / max(epoch_elapsed, 1e-12)
                    epoch_sample_per_sec = seen / max(epoch_elapsed, 1e-12)
                    train_steps_done = (epoch - start_epoch) * total_batches + batch_index
                    eta_sec = max(0, total_train_steps - train_steps_done) / max(epoch_iter_per_sec, 1e-12)
                    current_stats = {key: value / max(1, seen) for key, value in totals.items()}
                    logger.info(
                        "epoch=%d step=%d/%d global_step=%d %s step_time=%.2fs iter/s=%.3f sample/s=%.3f epoch_iter/s=%.3f epoch_sample/s=%.3f eta=%.1fs",
                        epoch + 1, batch_index, total_batches, step,
                        _progress_description("train", current_stats, float(grad_norm)).removeprefix(" => train | "),
                        step_dt, iter_per_sec, sample_per_sec, epoch_iter_per_sec,
                        epoch_sample_per_sec, eta_sec,
                    )
                    if hasattr(writer, "add_scalar"):
                        writer.add_scalar("speed/step_time_sec", step_dt, step)
                        writer.add_scalar("speed/iter_per_sec", iter_per_sec, step)
                        writer.add_scalar("speed/sample_per_sec", sample_per_sec, step)
                        writer.add_scalar("speed/epoch_iter_per_sec", epoch_iter_per_sec, step)
                        writer.add_scalar("speed/epoch_sample_per_sec", epoch_sample_per_sec, step)
                        writer.add_scalar("speed/eta_sec", eta_sec, step)
                        writer.add_scalar("train/grad_norm", float(grad_norm), step)
                    speed_window_t0 = sync_t0
                    speed_window_step = batch_index
                    speed_window_samples = seen
            # Evaluate with EMA parameters and EMA BatchNorm statistics, not the live model.
            train_stats = {key: value / max(1, seen) for key, value in totals.items()}
            portable_params, portable_batch_stats, portable_rng_state, portable_ema, portable_opt_state = _portable_training_state(
                model_params, model_batch_stats, model_rng_state, ema, opt_state, replicated=use_tpu_pmap,
            )
            valid_stats = _eval_epoch(
                graphdef, portable_ema.params, portable_ema.batch_stats, portable_rng_state,
                valid_dataset, args.bs, rng,
            ); final_stats = valid_stats
            train_time = time.perf_counter() - epoch_step_t0
            eval_model = _merge(graphdef, portable_ema.params, portable_ema.batch_stats, portable_rng_state)
            train_prediction_stats = _prediction_metrics(args, eval_model, train_dataset, args.bs, rng)
            prediction_stats = _prediction_metrics(args, eval_model, valid_dataset, args.bs, rng)
            epoch_iter_per_sec = total_batches / max(train_time, 1e-12)
            epoch_sample_per_sec = seen / max(train_time, 1e-12)
            total_time = time.perf_counter() - epoch_t0
            _write_epoch_summary(
                writer, epoch=epoch + 1, step=step, train_stats=train_stats,
                valid_stats=valid_stats, prediction_stats=prediction_stats,
                train_time=train_time, total_time=total_time,
                iter_per_sec=epoch_iter_per_sec, sample_per_sec=epoch_sample_per_sec,
                train_prediction_stats=train_prediction_stats,
            )
            _writer_add_custom_scalars(writer)
            _log_epoch_summary(
                logger, epoch=epoch + 1, step=step, train_stats=train_stats,
                valid_stats=valid_stats, prediction_stats=prediction_stats,
                train_time=train_time, total_time=total_time,
                iter_per_sec=epoch_iter_per_sec, sample_per_sec=epoch_sample_per_sec,
                train_prediction_stats=train_prediction_stats,
            )
            checkpoint_due = _checkpoint_due(epoch + 1, getattr(args, "checkpoint_freq", 1))
            if checkpoint_due and valid_stats["loss"] < best_loss:
                best_loss = valid_stats["loss"]
                _submit_best_checkpoint(
                    artifact_writer, args, portable_params, portable_batch_stats, portable_ema,
                    portable_opt_state, epoch + 1, step, best_loss,
                )
                logger.info("Model checkpoint enqueued: %s queue=%s", args.checkpoint_dir, artifact_writer.stats)
            writer.flush()
            if checkpoint_due and args.remote_save_dir:
                metric_artifact_writer.submit(_sync_metric_artifacts, args)
                logger.info(
                    "metric_artifacts_enqueued epoch=%d step=%d queue=%s",
                    epoch + 1, step, metric_artifact_writer.stats,
                )
    finally:
        try:
            artifact_writer.close()
        finally:
            try:
                metric_artifact_writer.close()
            finally:
                writer.close()
    return final_stats


def run(config: ExperimentConfig) -> str:
    """Run predictor training directly from a typed experiment configuration."""
    run_dir = output_dir(config)
    _run(_run_arguments(config))
    validate_artifacts(run_dir)
    return str(run_dir)
