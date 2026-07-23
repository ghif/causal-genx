"""Stage 2: train the image-to-causal-variable predictor.

The predictor is trained independently from the SCM. It maps an observed image
to thickness, intensity, and digit distributions, then supplies the auxiliary
counterfactual constraint used by Stage 4.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from causal.image_parent_predictor import MorphoMNISTSupAuxPredictor
from config import ExperimentConfig, PredictorTrainingConfig
from data.morphomnist import morphomnist
from utils import (
    BackgroundArtifactWriter, SummaryWriter, checkpoint_root_dir, ensure_dir,
    experiment_run_dir, load_checkpoint, seed_all, sync_file, tree_copy,
)

from .common import epoch_batches, stage_run_dir


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
    # Artifact identity only; native code never uses this to select a path.
    setup: str = "sup_aux"
    parents_x: list[str] | None = None
    context_norm: str = ""
    context_dim: int = 0
    concat_pa: bool = False
    save_dir: str = ""
    checkpoint_dir: str = ""
    remote_save_dir: str = ""


class _IndexedDataset:
    def __init__(self, dataset: Any, indices: np.ndarray):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.min_max = getattr(dataset, "min_max", {})
        self.samples = getattr(dataset, "samples", {})

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int):
        return self.dataset[int(self.indices[int(index)])]

    def make_batch(self, indices: Any, rng=None, shuffle: bool = False):
        return self.dataset.make_batch(self.indices[np.asarray(indices, dtype=np.int64)], rng=rng, shuffle=shuffle)


@dataclass
class WarmupEMA:
    params: Any
    batch_stats: Any
    step: int = 0
    initted: bool = False
    beta: float = 0.999
    update_after_step: int = 100
    inv_gamma: float = 1.0
    power: float = 1.0
    min_value: float = 0.0

    @classmethod
    def init_from(cls, params: Any, batch_stats: Any) -> "WarmupEMA":
        return cls(params=tree_copy(params), batch_stats=tree_copy(batch_stats))

    def update(self, params: Any, batch_stats: Any) -> None:
        current_step = self.step
        self.step += 1
        if current_step <= self.update_after_step:
            self.params, self.batch_stats = tree_copy(params), tree_copy(batch_stats)
            return
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
    return PredictorRunArguments(
        accelerator=config.runtime.accelerator, gpu_id=config.runtime.gpu_id,
        precision=config.runtime.precision, exp_name=config.artifacts.run_name,
        dataset=config.dataset.name, data_dir=config.dataset.root,
        ckpt_dir=config.artifacts.root, remote_ckpt_dir=config.artifacts.remote_root,
        seed=config.seed, epochs=workflow.epochs, bs=config.optimizer.batch_size,
        lr=config.optimizer.lr, wd=config.optimizer.weight_decay,
        input_res=config.dataset.input_res, pad=config.dataset.pad,
        checkpoint_freq=workflow.checkpoint_freq, speed_log_freq=workflow.speed_log_freq,
        execution_mode=workflow.execution_mode, drop_remainder=workflow.drop_remainder,
        widths=[32, 32],
    )


def _validate_scope(args: PredictorRunArguments) -> None:
    if args.dataset != "morphomnist":
        raise ValueError("Predictor training currently supports only dataset=morphomnist")
    if args.accelerator == "cpu" and args.precision != "fp32":
        raise ValueError("CPU predictor training requires precision=fp32")
    if args.input_channels != 1 or args.input_res != 32 or args.pad != 4:
        raise ValueError("MorphoMNIST predictor requires input_channels=1, input_res=32, and pad=4")


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


def _configure_dataset_args(args: PredictorRunArguments) -> None:
    args.parents_x = ["thickness", "intensity", "digit"]
    args.context_norm, args.context_dim, args.concat_pa = "[-1,1]", 12, False


def _build_datasets(args: PredictorRunArguments):
    datasets = morphomnist(args)
    indices = np.arange(len(datasets["train"]))
    rng = np.random.RandomState(1); rng.shuffle(indices)
    return datasets, _IndexedDataset(datasets["train"], indices[:int(args.sup_frac * len(indices))])


def _merge(graphdef: Any, params: Any, batch_stats: Any):
    return nnx.merge(graphdef, params, batch_stats)


def _loss_and_state(graphdef: Any, params: Any, batch_stats: Any, batch: Dict[str, jax.Array], *, training: bool):
    model = _merge(graphdef, params, batch_stats)
    model.train() if training else model.eval()
    log_probs = model.model_anticausal(**batch)
    loss = -jnp.mean(log_probs["joint"])
    metrics = {"loss": loss, **{f"logp({key})": jnp.mean(log_probs[key]) for key in ("thickness_aux", "intensity_aux", "digit_aux")}}
    return loss, metrics, nnx.state(model, nnx.Param).to_pure_dict(), nnx.state(model, nnx.BatchStat).to_pure_dict()


def _make_train_step(graphdef: Any, optimizer: optax.GradientTransformation):
    """Compile one predictor update, including BatchNorm state evolution."""
    @jax.jit
    def train_step(params, batch_stats, opt_state, batch):
        def loss_fn(current_params):
            loss, metrics, new_params, new_batch_stats = _loss_and_state(graphdef, current_params, batch_stats, batch, training=True)
            return loss, (metrics, new_params, new_batch_stats)
        (_, (metrics, _new_params, new_batch_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grad_norm = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_batch_stats, opt_state, metrics, grad_norm
    return train_step


def _make_pmap_train_step(
    graphdef: Any,
    optimizer: optax.GradientTransformation,
    devices: list[jax.Device],
):
    """Compile a synchronized multi-core TPU predictor update."""
    def train_step(params, batch_stats, opt_state, batch):
        def loss_fn(current_params):
            loss, metrics, _new_params, new_batch_stats = _loss_and_state(
                graphdef, current_params, batch_stats, batch, training=True
            )
            return loss, (metrics, new_batch_stats)

        (_, (metrics, new_batch_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads = jax.lax.pmean(grads, axis_name="devices")
        metrics = jax.tree_util.tree_map(
            lambda value: jax.lax.pmean(value, axis_name="devices"), metrics
        )
        new_batch_stats = jax.tree_util.tree_map(
            lambda value: jax.lax.pmean(value, axis_name="devices"), new_batch_stats
        )
        grad_norm = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_batch_stats, opt_state, metrics, grad_norm

    return jax.pmap(
        train_step,
        axis_name="devices",
        in_axes=(0, 0, 0, 0),
        devices=devices,
    )


def _portable_training_state(
    model_params: Any,
    batch_stats: Any,
    ema: WarmupEMA,
    opt_state: Any,
    *,
    replicated: bool,
) -> tuple[Any, Any, WarmupEMA, Any]:
    """Return ordinary single-device trees for evaluation and persistence."""
    if not replicated:
        return model_params, batch_stats, ema, opt_state
    return (
        _unreplicate(model_params),
        _unreplicate(batch_stats),
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


def _eval_epoch(graphdef: Any, params: Any, batch_stats: Any, dataset: Any, batch_size: int, rng: np.random.Generator) -> Dict[str, float]:
    totals: Dict[str, float] = {}; count = 0
    for batch in epoch_batches(dataset, batch_size, shuffle=False, drop_last=False, rng=rng):
        _, metrics, _, _ = _loss_and_state(graphdef, params, batch_stats, batch, training=False)
        size = int(batch["digit"].shape[0])
        for key, value in metrics.items(): totals[key] = totals.get(key, 0.0) + float(value) * size
        count += size
    return {key: value / max(1, count) for key, value in totals.items()}


def _progress_description(mode: str, stats: Dict[str, float], grad_norm: Optional[float] = None) -> str:
    ordered = ("loss", "logp(digit_aux)", "logp(thickness_aux)", "logp(intensity_aux)")
    description = " => " + mode + " | " + ", ".join(
        f"{key}: {stats[key]:.4f}" for key in ordered if key in stats
    )
    return description if grad_norm is None else f"{description}, grad_norm: {grad_norm:.3f}"


def _prediction_description(metrics: Dict[str, float]) -> str:
    return " - ".join(
        f"{key}: {metrics[key]:.4f}"
        for key in ("thickness_mae", "intensity_mae", "digit_acc")
        if key in metrics
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
) -> None:
    """Persist the complete completed-epoch predictor summary to TensorBoard."""
    for key, value in train_stats.items():
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
) -> None:
    """Write the same completed-epoch metrics to the console and train log."""
    logger.info(
        "%s - steps: %d - it/s: %.3f - samples/s: %.3f",
        _progress_description("train", train_stats).strip(), step, iter_per_sec, sample_per_sec,
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
        for key in targets: targets[key].extend(np.asarray(batch[key]))
        for key, value in model.predict(**batch).items(): predictions[key].extend(np.asarray(value))
    stats: Dict[str, float] = {}
    for key in model.variables:
        if key == "digit":
            stats["digit_acc"] = float((np.asarray(targets[key]).argmax(-1) == np.asarray(predictions[key]).argmax(-1)).mean())
        else:
            low, high = dataset.min_max[key]
            prediction = ((np.asarray(predictions[key]).squeeze(-1) + 1.0) / 2.0) * (high - low) + low
            target = ((np.asarray(targets[key]).squeeze(-1) + 1.0) / 2.0) * (high - low) + low
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
    return logging.getLogger(args.exp_name or "morphomnist-predictor")


def _run(args: PredictorRunArguments) -> Dict[str, float]:
    """Execute predictor resume → supervised training → EMA validation → save."""
    checkpoint: Optional[Dict[str, Any]] = load_checkpoint(args.load_path) if args.load_path else None
    if checkpoint is not None: _restore_args(args, checkpoint)
    _validate_scope(args); _validate_runtime_device(args); dtype = _compute_dtype(args); _configure_dataset_args(args); seed_all(args.seed, args.deterministic)
    args.save_dir = experiment_run_dir(args.ckpt_dir, "morphomnist", args.exp_name, "pgm")
    args.checkpoint_dir = checkpoint_root_dir(args.save_dir); args.remote_save_dir = experiment_run_dir(args.remote_ckpt_dir, "morphomnist", args.exp_name, "pgm")
    ensure_dir(args.save_dir); ensure_dir(args.checkpoint_dir)
    # The labelled subset is deterministic, making partial-supervision runs reproducible.
    logger = _setup_logging(args); writer = SummaryWriter(args.save_dir); datasets, train_dataset = _build_datasets(args); valid_dataset = datasets["valid"]
    model = MorphoMNISTSupAuxPredictor(input_channels=args.input_channels, input_res=args.input_res, width=8, std_fixed=args.std_fixed, compute_dtype=dtype, rngs=nnx.Rngs(args.seed))
    graphdef, params_state, batch_stats_state = nnx.split(model, nnx.Param, nnx.BatchStat); model_params, model_batch_stats = params_state.to_pure_dict(), batch_stats_state.to_pure_dict()
    optimizer = optax.chain(optax.clip_by_global_norm(200.0), optax.adamw(args.lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=args.wd)); opt_state = optimizer.init(model_params); ema = WarmupEMA.init_from(model_params, model_batch_stats)
    start_epoch = step = 0; best_loss = float("inf")
    if checkpoint is not None:
        _assert_compatible_checkpoint(checkpoint, model_params, model_batch_stats); model_params, model_batch_stats, opt_state = checkpoint["model_params"], checkpoint["batch_stats"], checkpoint["opt_state"]
        ema = WarmupEMA(params=checkpoint.get("ema_params", checkpoint["params"]), batch_stats=checkpoint["ema_batch_stats"], step=int(checkpoint.get("ema_step", checkpoint.get("step", 0))), initted=bool(checkpoint.get("ema_initted", True)))
        start_epoch, step, best_loss = int(checkpoint.get("epoch", 0)), int(checkpoint.get("step", 0)), float(checkpoint.get("best_loss", float("inf")))
    rng = np.random.default_rng(args.seed)
    if args.testing:
        if checkpoint is None: raise ValueError("testing requires load_path")
        stats = _prediction_metrics(args, _merge(graphdef, ema.params, ema.batch_stats), datasets["test"], args.bs, rng); logger.info("test | %s", _prediction_description(stats)); writer.close(); return stats
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
        train_step = _make_pmap_train_step(graphdef, optimizer, devices)
        model_params = _replicate(model_params, devices)
        model_batch_stats = _replicate(model_batch_stats, devices)
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
        train_step = _make_train_step(graphdef, optimizer)
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
            for batch_index, batch in enumerate(
                epoch_batches(train_dataset, args.bs, shuffle=True, drop_last=drop_remainder, rng=rng), start=1
            ):
                if use_tpu_pmap:
                    batch = _shard_batch(batch, devices)
                model_params, model_batch_stats, opt_state, metrics, grad_norm = train_step(model_params, model_batch_stats, opt_state, batch); ema.update(model_params, model_batch_stats); size = int(batch["digit"].shape[0])
                if use_tpu_pmap:
                    metrics = _unreplicate(metrics)
                    grad_norm = _first_local_replica(grad_norm)
                    size *= int(batch["digit"].shape[1])
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
            portable_params, portable_batch_stats, portable_ema, portable_opt_state = _portable_training_state(
                model_params, model_batch_stats, ema, opt_state, replicated=use_tpu_pmap,
            )
            valid_stats = _eval_epoch(graphdef, portable_ema.params, portable_ema.batch_stats, valid_dataset, args.bs, rng); final_stats = valid_stats
            train_time = time.perf_counter() - epoch_step_t0
            prediction_stats = _prediction_metrics(args, _merge(graphdef, portable_ema.params, portable_ema.batch_stats), valid_dataset, args.bs, rng)
            epoch_iter_per_sec = total_batches / max(train_time, 1e-12)
            epoch_sample_per_sec = seen / max(train_time, 1e-12)
            total_time = time.perf_counter() - epoch_t0
            _write_epoch_summary(
                writer, epoch=epoch + 1, step=step, train_stats=train_stats,
                valid_stats=valid_stats, prediction_stats=prediction_stats,
                train_time=train_time, total_time=total_time,
                iter_per_sec=epoch_iter_per_sec, sample_per_sec=epoch_sample_per_sec,
            )
            _writer_add_custom_scalars(writer)
            _log_epoch_summary(
                logger, epoch=epoch + 1, step=step, train_stats=train_stats,
                valid_stats=valid_stats, prediction_stats=prediction_stats,
                train_time=train_time, total_time=total_time,
                iter_per_sec=epoch_iter_per_sec, sample_per_sec=epoch_sample_per_sec,
            )
            checkpoint_due = _checkpoint_due(epoch + 1, args.checkpoint_freq)
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
