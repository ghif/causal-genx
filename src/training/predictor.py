"""Stage 2: native image-to-causal-variable predictor training."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from tqdm import tqdm

from causal.image_parent_predictor import MorphoMNISTSupAuxPredictor
from config import ExperimentConfig, PredictorTrainingConfig
from data.morphomnist import morphomnist
from utils import (
    SummaryWriter, checkpoint_root_dir, ensure_dir, experiment_run_dir,
    load_checkpoint, save_checkpoint, seed_all, sync_file, sync_tree, tree_copy,
)

from .common import epoch_batches, legacy_run_dir


@dataclass
class PredictorRunArguments:
    """Legacy-compatible predictor settings constructed from the typed config."""

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
    return legacy_run_dir(config)


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
        input_res=config.dataset.input_res, pad=config.dataset.pad, widths=[32, 32],
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
    @jax.jit
    def train_step(params, batch_stats, opt_state, batch):
        def loss_fn(current_params):
            loss, metrics, new_params, new_batch_stats = _loss_and_state(graphdef, current_params, batch_stats, batch, training=True)
            return loss, (metrics, new_params, new_batch_stats)
        (_, (metrics, _new_params, new_batch_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grad_norm = optax.tree.norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_batch_stats, opt_state, metrics, grad_norm
    return train_step


def _eval_epoch(graphdef: Any, params: Any, batch_stats: Any, dataset: Any, batch_size: int, rng: np.random.Generator) -> Dict[str, float]:
    totals: Dict[str, float] = {}; count = 0
    progress = tqdm(epoch_batches(dataset, batch_size, shuffle=False, drop_last=False, rng=rng), total=(len(dataset) + batch_size - 1) // batch_size, miniters=max(1, len(dataset) // batch_size // 100), mininterval=5)
    for batch in progress:
        _, metrics, _, _ = _loss_and_state(graphdef, params, batch_stats, batch, training=False)
        size = int(batch["digit"].shape[0])
        for key, value in metrics.items(): totals[key] = totals.get(key, 0.0) + float(value) * size
        count += size
        progress.set_description(_progress_description("eval", {key: value / max(1, count) for key, value in totals.items()}))
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


def _writer_add_custom_scalars(writer: Any) -> None:
    if hasattr(writer, "add_custom_scalars"):
        writer.add_custom_scalars({"elbo": {"elbo": ["Multiline", ["elbo/train", "elbo/valid"]]}})


def _prediction_metrics(args: PredictorRunArguments, model: Any, dataset: Any, batch_size: int, rng: np.random.Generator) -> Dict[str, float]:
    model.eval(); predictions = {key: [] for key in model.variables}; targets = {key: [] for key in model.variables}
    for batch in tqdm(epoch_batches(dataset, batch_size, shuffle=False, drop_last=False, rng=rng), total=(len(dataset) + batch_size - 1) // batch_size, mininterval=5):
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
    checkpoint: Optional[Dict[str, Any]] = load_checkpoint(args.load_path) if args.load_path else None
    if checkpoint is not None: _restore_args(args, checkpoint)
    _validate_scope(args); _validate_runtime_device(args); dtype = _compute_dtype(args); _configure_dataset_args(args); seed_all(args.seed, args.deterministic)
    args.save_dir = experiment_run_dir(args.ckpt_dir, "morphomnist", args.exp_name, "pgm")
    args.checkpoint_dir = checkpoint_root_dir(args.save_dir); args.remote_save_dir = experiment_run_dir(args.remote_ckpt_dir, "morphomnist", args.exp_name, "pgm")
    ensure_dir(args.save_dir); ensure_dir(args.checkpoint_dir)
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
    train_step = _make_train_step(graphdef, optimizer); final_stats: Dict[str, float] = {}
    for epoch in range(start_epoch, args.epochs):
        logger.info("Epoch %d:", epoch + 1)
        totals: Dict[str, float] = {}; seen = 0; total_batches = len(train_dataset) // args.bs
        progress = tqdm(epoch_batches(train_dataset, args.bs, shuffle=True, drop_last=True, rng=rng), total=total_batches, miniters=max(1, total_batches // 100), mininterval=5)
        for batch in progress:
            model_params, model_batch_stats, opt_state, metrics, grad_norm = train_step(model_params, model_batch_stats, opt_state, batch); ema.update(model_params, model_batch_stats); size = int(batch["digit"].shape[0])
            for key, value in metrics.items(): totals[key] = totals.get(key, 0.0) + float(value) * size
            seen += size; step += 1
            progress.set_description(_progress_description("train", {key: value / max(1, seen) for key, value in totals.items()}, float(grad_norm)))
        train_stats = {key: value / max(1, seen) for key, value in totals.items()}; valid_stats = _eval_epoch(graphdef, ema.params, ema.batch_stats, valid_dataset, args.bs, rng); final_stats = valid_stats
        logger.info("loss | train: %.4f - valid: %.4f - steps: %d", train_stats["loss"], valid_stats["loss"], step)
        for key in train_stats: writer.add_scalar(f"train/{key}", train_stats[key], step); writer.add_scalar(f"valid/{key}", valid_stats[key], step)
        _writer_add_custom_scalars(writer)
        writer.add_scalar("elbo/train", train_stats["loss"], step); writer.add_scalar("elbo/valid", valid_stats["loss"], step)
        metrics = _prediction_metrics(args, _merge(graphdef, ema.params, ema.batch_stats), valid_dataset, args.bs, rng); logger.info("valid | %s", _prediction_description(metrics))
        if valid_stats["loss"] < best_loss:
            best_loss = valid_stats["loss"]; save_checkpoint(_checkpoint_payload(args, model_params, model_batch_stats, ema, opt_state, epoch + 1, step, best_loss), args.checkpoint_dir, step=step, custom_metadata={"epoch": epoch + 1, "best_loss": best_loss})
            if args.remote_save_dir: sync_tree(args.checkpoint_dir, os.path.join(args.remote_save_dir, "checkpoints"))
            logger.info("Model saved: %s", args.checkpoint_dir)
        writer.flush()
        if args.remote_save_dir: sync_file(os.path.join(args.save_dir, "trainlog.txt"), os.path.join(args.remote_save_dir, "trainlog.txt"))
    writer.close(); return final_stats


def run(config: ExperimentConfig) -> str:
    """Run predictor training directly from a typed experiment configuration."""
    run_dir = output_dir(config)
    _run(_run_arguments(config))
    validate_artifacts(run_dir)
    return str(run_dir)


def run_legacy_args(legacy_args: Any) -> Dict[str, float]:
    """Compatibility adapter for ``pgm/train_pgm.py --setup sup_aux``."""
    fields = PredictorRunArguments.__dataclass_fields__
    args = PredictorRunArguments(**{name: getattr(legacy_args, name, field.default) for name, field in fields.items() if field.init})
    return _run(args)
