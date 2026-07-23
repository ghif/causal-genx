"""Stage 1: native MorphoMNIST structural causal model training.

This module owns ``scripts/run.py train-scm``. Its process is:

1. load named causal variables from MorphoMNIST;
2. fit the flow SCM by maximizing their joint log probability;
3. evaluate the EMA parameters, plot the learned thickness/intensity joint,
   and checkpoint only when validation improves.

The checkpoint payload deliberately records both live and EMA parameters:
the latter is the stable inference artifact consumed by counterfactual training.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx

from causal.flow_scm import MorphoMNISTPGM
from config import ExperimentConfig, ScmTrainingConfig
from data.morphomnist import morphomnist
from utils import (
    BackgroundArtifactWriter,
    SummaryWriter,
    checkpoint_root_dir,
    ensure_dir,
    experiment_run_dir,
    load_checkpoint,
    materialize_nnx,
    seed_all,
    sync_file,
    tree_copy,
)

from .common import stage_run_dir


@dataclass
class ScmRunArguments:
    """Runtime-only settings derived from the typed SCM YAML configuration."""

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
    speed_log_freq: int
    checkpoint_freq: int
    plot_samples: int
    widths: list[int]
    benchmark_steps: int = 0
    input_channels: int = 1
    load_path: str = ""
    deterministic: bool = False
    testing: bool = False
    # ``setup`` is metadata only: preserving it keeps legacy consumers able to
    # identify an SCM artifact; the native runner never dispatches on it.
    setup: str = "sup_pgm"
    sup_frac: float = 1.0
    std_fixed: float = 0.0
    parents_x: list[str] | None = None
    context_norm: str = ""
    context_dim: int = 0
    concat_pa: bool = False
    save_dir: str = ""
    checkpoint_dir: str = ""
    remote_save_dir: str = ""


def output_dir(config: ExperimentConfig) -> Path:
    return stage_run_dir(config)


def validate_artifacts(run_dir: str | Path) -> None:
    root = Path(run_dir)
    required = (root / "checkpoints" / "hparams.json", root / "trainlog.txt", root / "joint_data.pdf")
    missing = [str(path) for path in required if not path.is_file()]
    if not list(root.glob("events.out.tfevents.*")):
        missing.append(f"{root}/events.out.tfevents.*")
    if not list(root.glob("joint_model_*.pdf")):
        missing.append(f"{root}/joint_model_*.pdf")
    if not list((root / "checkpoints").glob("[0-9]*/_CHECKPOINT_METADATA")):
        missing.append(f"{root}/checkpoints/<step>/_CHECKPOINT_METADATA")
    if missing:
        raise RuntimeError(f"SCM run is missing required artifacts: {', '.join(missing)}")


def _run_arguments(config: ExperimentConfig) -> ScmRunArguments:
    workflow = config.workflow
    assert isinstance(workflow, ScmTrainingConfig)
    return ScmRunArguments(
        accelerator=config.runtime.accelerator,
        gpu_id=config.runtime.gpu_id,
        precision=config.runtime.precision,
        exp_name=config.artifacts.run_name,
        dataset=config.dataset.name,
        data_dir=config.dataset.root,
        ckpt_dir=config.artifacts.root,
        remote_ckpt_dir=config.artifacts.remote_root,
        seed=config.seed,
        epochs=workflow.epochs,
        bs=config.optimizer.batch_size,
        lr=config.optimizer.lr,
        wd=config.optimizer.weight_decay,
        input_res=config.dataset.input_res,
        pad=config.dataset.pad,
        speed_log_freq=workflow.speed_log_freq,
        checkpoint_freq=workflow.checkpoint_freq,
        plot_samples=workflow.plot_samples,
        widths=list(workflow.widths),
        benchmark_steps=workflow.benchmark_steps,
    )


def _validate_scope(args: ScmRunArguments) -> None:
    if args.dataset != "morphomnist":
        raise ValueError("SCM training currently supports only dataset=morphomnist")
    if args.precision != "fp32":
        raise ValueError("MorphoMNIST SCM parity requires precision=fp32")
    if args.input_channels != 1 or args.input_res != 32 or args.pad != 4:
        raise ValueError("MorphoMNIST SCM requires input_channels=1, input_res=32, and pad=4")


def _configure_dataset_args(args: ScmRunArguments) -> None:
    args.parents_x = ["thickness", "intensity", "digit"]
    args.context_norm = "[-1,1]"
    args.context_dim = 12
    args.concat_pa = False


def _setup_logging(args: ScmRunArguments) -> logging.Logger:
    ensure_dir(args.save_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s, %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(args.save_dir, "trainlog.txt"), mode="a")],
        force=True,
    )
    return logging.getLogger(args.exp_name or "morphomnist-pgm")


def preprocess(batch: Dict[str, np.ndarray]) -> Dict[str, jax.Array]:
    x = np.asarray(batch["x"], dtype=np.float32)
    if x.max(initial=0.0) > 1.5:
        x = (x - 127.5) / 127.5
    return {
        "x": jnp.asarray(x),
        "thickness": jnp.asarray(batch["thickness"], dtype=jnp.float32).reshape((-1, 1)),
        "intensity": jnp.asarray(batch["intensity"], dtype=jnp.float32).reshape((-1, 1)),
        "digit": jnp.asarray(batch["digit"], dtype=jnp.float32),
    }


def epoch_batches(dataset: Any, batch_size: int, *, shuffle: bool, drop_last: bool, rng: np.random.Generator) -> Iterator[Dict[str, jax.Array]]:
    indices = np.arange(len(dataset), dtype=np.int64)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        if drop_last and batch_indices.size < batch_size:
            continue
        batch = dataset.make_batch(batch_indices, rng=rng, shuffle=shuffle) if hasattr(dataset, "make_batch") else {
            key: np.stack([np.asarray(dataset[int(index)][key]) for index in batch_indices]) for key in dataset[0]
        }
        yield preprocess(batch)


def _loss(graphdef: Any, params: Any, batch: Dict[str, jax.Array]):
    model = materialize_nnx(graphdef, params)
    log_probs = model.log_prob(batch["thickness"], batch["intensity"], batch["digit"])
    loss = -jnp.mean(log_probs["joint"])
    return loss, {"loss": loss, **{f"logp({name})": jnp.mean(log_probs[name]) for name in ("digit", "thickness", "intensity")}}


def _make_train_step(graphdef: Any, optimizer: optax.GradientTransformation):
    """Compile one parameter update; host logging and EMA updates stay outside JIT."""
    @jax.jit
    def train_step(params: Any, opt_state: Any, batch: Dict[str, jax.Array]):
        (loss, metrics), grads = jax.value_and_grad(_loss, argnums=1, has_aux=True)(graphdef, params, batch)
        grad_norm = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, metrics, grad_norm
    return train_step


@dataclass
class PGMEMA:
    params: Any
    step: int = 0
    initted: bool = False
    beta: float = 0.999
    update_after_step: int = 100

    @classmethod
    def init_from(cls, params: Any) -> "PGMEMA":
        return cls(params=tree_copy(params))

    def update(self, params: Any) -> None:
        current_step = self.step
        self.step += 1
        if current_step <= self.update_after_step:
            self.params = tree_copy(params)
            return
        if not self.initted:
            self.params, self.initted = tree_copy(params), True
        epoch = max(self.step - self.update_after_step - 1, 0)
        decay = 0.0 if epoch <= 0 else min(1.0 - (1.0 + epoch) ** -1.0, self.beta)
        self.params = jax.tree_util.tree_map(lambda ema, value: ema * decay + value * (1.0 - decay), self.params, params)


def _mean_metrics(totals: Dict[str, float], count: int) -> Dict[str, float]:
    return {key: value / max(1, count) for key, value in totals.items()}


def _progress_description(mode: str, stats: Dict[str, float], grad_norm: Optional[float] = None) -> str:
    description = f" => {mode} | " + ", ".join(f"{key}: {value:.4f}" for key, value in stats.items())
    return description if grad_norm is None else f"{description}, grad_norm: {grad_norm:.3f}"


def _eval_epoch(graphdef: Any, params: Any, dataset: Any, batch_size: int, rng: np.random.Generator) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    count = 0
    for batch in epoch_batches(dataset, batch_size, shuffle=False, drop_last=False, rng=rng):
        _, metrics = _loss(graphdef, params, batch)
        size = int(batch["digit"].shape[0])
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * size
        count += size
    return _mean_metrics(totals, count)


def _joint_figure(x: np.ndarray, y: np.ndarray, title: str, path: str) -> None:
    figure = plt.figure(figsize=(6, 6))
    grid = figure.add_gridspec(4, 4, hspace=0.05, wspace=0.05)
    joint = figure.add_subplot(grid[1:, :3]); top = figure.add_subplot(grid[0, :3], sharex=joint); right = figure.add_subplot(grid[1:, 3], sharey=joint)
    joint.scatter(x, y, s=2, alpha=0.2); top.hist(x, bins=50); right.hist(y, bins=50, orientation="horizontal")
    joint.set_xlabel("thickness"); joint.set_ylabel("intensity"); figure.suptitle(title)
    figure.savefig(path, bbox_inches="tight"); plt.close(figure)


def _plot_joint(args: ScmRunArguments, graphdef: Any, params: Any, dataset: Any, step: int) -> None:
    data_path = os.path.join(args.save_dir, "joint_data.pdf")
    if not os.path.exists(data_path):
        _joint_figure(np.asarray(dataset.samples["thickness"]), np.asarray(dataset.samples["intensity"]), "Data Joint", data_path)
    samples = materialize_nnx(graphdef, params).sample(args.plot_samples, jax.random.PRNGKey(args.seed + step))
    _joint_figure(np.asarray(samples["thickness"]).squeeze(), np.asarray(samples["intensity"]).squeeze(), f"Model Joint (step {step})", os.path.join(args.save_dir, f"joint_model_{step}.pdf"))


def _sync_pdf_artifacts(args: ScmRunArguments) -> None:
    if args.remote_save_dir:
        for path in sorted(Path(args.save_dir).glob("*.pdf")):
            sync_file(str(path), os.path.join(args.remote_save_dir, path.name))


def _checkpoint_due(epoch: int, checkpoint_freq: int) -> bool:
    """Return whether a completed one-based epoch may validate and persist artifacts."""
    return epoch % max(1, checkpoint_freq) == 0


def _sync_tensorboard_artifacts(args: ScmRunArguments) -> None:
    if not args.remote_save_dir:
        return
    for event_path in sorted(Path(args.save_dir).glob("events.out.tfevents.*")):
        sync_file(str(event_path), os.path.join(args.remote_save_dir, event_path.name))


def _sync_metric_artifacts(args: ScmRunArguments) -> None:
    """Synchronize flushed logs, TensorBoard events, and SCM plots to GCS."""
    if not args.remote_save_dir:
        return
    sync_file(
        os.path.join(args.save_dir, "trainlog.txt"),
        os.path.join(args.remote_save_dir, "trainlog.txt"),
    )
    _sync_tensorboard_artifacts(args)
    _sync_pdf_artifacts(args)


def _submit_best_checkpoint(
    artifact_writer: BackgroundArtifactWriter,
    args: ScmRunArguments,
    params: Any,
    ema: PGMEMA,
    opt_state: Any,
    epoch: int,
    step: int,
    best_loss: float,
) -> None:
    payload = _checkpoint_payload(args, params, ema, opt_state, epoch, step, best_loss)
    remote_checkpoint_dir = (
        os.path.join(args.remote_save_dir, "checkpoints") if args.remote_save_dir else None
    )
    artifact_writer.submit_checkpoint(
        payload,
        args.checkpoint_dir,
        step=step,
        custom_metadata={"epoch": epoch, "best_loss": best_loss},
        local_tree_dir=args.checkpoint_dir if remote_checkpoint_dir else None,
        remote_tree_dir=remote_checkpoint_dir,
    )


def _write_epoch_summary(
    writer: Any,
    *,
    epoch: int,
    step: int,
    train_stats: Dict[str, float],
    train_time: float,
    total_time: float,
    iter_per_sec: float,
    sample_per_sec: float,
    grad_norm: float,
    valid_stats: Dict[str, float] | None = None,
) -> None:
    """Persist a completed SCM epoch, with validation metrics when checkpoint-due."""
    for key, value in train_stats.items():
        writer.add_scalar(f"train/{key}", value, step)
    if valid_stats is not None:
        for key, value in valid_stats.items():
            writer.add_scalar(f"valid/{key}", value, step)
        writer.add_scalar("elbo/valid", valid_stats["loss"], step)
    writer.add_scalar("elbo/train", train_stats["loss"], step)
    writer.add_scalar("epoch/number", epoch, step)
    writer.add_scalar("epoch/global_step", step, step)
    writer.add_scalar("epoch/train_time_sec", train_time, step)
    writer.add_scalar("epoch/total_time_sec", total_time, step)
    writer.add_scalar("epoch/iter_per_sec", iter_per_sec, step)
    writer.add_scalar("epoch/sample_per_sec", sample_per_sec, step)
    writer.add_scalar("epoch/grad_norm", grad_norm, step)


def _log_epoch_summary(
    logger: logging.Logger,
    *,
    epoch: int,
    step: int,
    train_stats: Dict[str, float],
    train_time: float,
    total_time: float,
    iter_per_sec: float,
    sample_per_sec: float,
    grad_norm: float,
    valid_stats: Dict[str, float] | None = None,
) -> None:
    logger.info(
        "%s - steps: %d - it/s: %.3f - samples/s: %.3f",
        _progress_description("train", train_stats, grad_norm).strip(), step, iter_per_sec, sample_per_sec,
    )
    if valid_stats is not None:
        logger.info("%s - steps: %d", _progress_description("valid", valid_stats).strip(), step)
    logger.info(
        "epoch=%d train_time=%.1fs total_time=%.1fs epoch_iter/s=%.3f epoch_sample/s=%.3f",
        epoch, train_time, total_time, iter_per_sec, sample_per_sec,
    )


def _checkpoint_payload(args: ScmRunArguments, params: Any, ema: PGMEMA, opt_state: Any, epoch: int, step: int, best_loss: float) -> Dict[str, Any]:
    return {"params": ema.params, "ema_params": ema.params, "model_params": params, "opt_state": opt_state, "epoch": epoch, "step": step, "best_loss": best_loss, "ema_step": ema.step, "ema_initted": ema.initted, "hparams": vars(args), "format_version": 2}


def _assert_compatible_checkpoint(checkpoint: Dict[str, Any], params: Any) -> None:
    if checkpoint.get("format_version") != 2 or "model_params" not in checkpoint:
        raise ValueError("Checkpoint is not a compatible JAX SCM artifact")
    if jax.tree_util.tree_structure(checkpoint["model_params"]) != jax.tree_util.tree_structure(params):
        raise ValueError("Checkpoint parameter structure does not match the SCM")


def _restore_args(args: ScmRunArguments, checkpoint: Dict[str, Any]) -> None:
    saved = checkpoint.get("hparams", {})
    preserved = {key: getattr(args, key) for key in ("accelerator", "data_dir", "load_path", "testing", "remote_ckpt_dir")}
    for key, value in saved.items():
        if hasattr(args, key):
            setattr(args, key, value)
    for key, value in preserved.items():
        setattr(args, key, value)


def _run(args: ScmRunArguments) -> Dict[str, float]:
    """Execute load/resume → train → validate/plot → checkpoint for one SCM run."""
    _validate_scope(args)
    checkpoint: Optional[Dict[str, Any]] = None
    if args.load_path:
        checkpoint = load_checkpoint(args.load_path); _restore_args(args, checkpoint); _validate_scope(args)
    _configure_dataset_args(args); seed_all(args.seed, args.deterministic)
    args.save_dir = experiment_run_dir(args.ckpt_dir, "morphomnist", args.exp_name, "pgm")
    args.checkpoint_dir = checkpoint_root_dir(args.save_dir)
    args.remote_save_dir = experiment_run_dir(args.remote_ckpt_dir, "morphomnist", args.exp_name, "pgm")
    ensure_dir(args.save_dir); ensure_dir(args.checkpoint_dir)
    # Build data and model only after resume metadata has restored model settings.
    logger = _setup_logging(args); writer = SummaryWriter(args.save_dir); datasets = morphomnist(args)
    model = MorphoMNISTPGM(widths=args.widths, rngs=nnx.Rngs(args.seed))
    graphdef, _ = nnx.split(model, nnx.Param); params = nnx.state(model, nnx.Param).to_pure_dict()
    optimizer = optax.chain(optax.clip_by_global_norm(200.0), optax.adamw(args.lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=args.wd))
    opt_state = optimizer.init(params); ema = PGMEMA.init_from(params); start_epoch = step = 0; best_loss = float("inf")
    if checkpoint is not None:
        # Resume the trainable state while retaining the artifact's EMA state.
        _assert_compatible_checkpoint(checkpoint, params)
        params, opt_state = checkpoint["model_params"], checkpoint["opt_state"]
        ema = PGMEMA(params=checkpoint.get("ema_params", checkpoint["params"]), step=int(checkpoint.get("ema_step", checkpoint.get("step", 0))), initted=bool(checkpoint.get("ema_initted", True)))
        start_epoch, step, best_loss = int(checkpoint.get("epoch", 0)), int(checkpoint.get("step", 0)), float(checkpoint.get("best_loss", float("inf")))
        logger.info("Loaded checkpoint %s at epoch=%d step=%d", args.load_path, start_epoch, step)
    rng = np.random.default_rng(args.seed)
    if args.testing:
        if checkpoint is None: raise ValueError("testing requires load_path")
        stats = _eval_epoch(graphdef, ema.params, datasets["test"], args.bs, rng); logger.info("test | %s", stats)
        _plot_joint(args, graphdef, ema.params, datasets["test"], 0); _sync_pdf_artifacts(args); writer.close(); return stats
    for key in sorted(vars(args)): logger.info("--%s=%s", key, getattr(args, key))
    train_step = _make_train_step(graphdef, optimizer); final_stats: Dict[str, float] = {}
    artifact_writer = BackgroundArtifactWriter()
    metric_artifact_writer = BackgroundArtifactWriter()
    total_batches = len(datasets["train"]) // args.bs
    total_train_steps = max(1, total_batches) * max(1, args.epochs - start_epoch)
    try:
        for epoch in range(start_epoch, args.epochs):
            logger.info("Epoch %d:", epoch + 1)
            totals: Dict[str, float] = {}; seen = 0; last_grad_norm = 0.0
            epoch_t0 = epoch_step_t0 = speed_window_t0 = time.perf_counter()
            speed_window_step = 0
            speed_window_samples = 0
            for batch_index, batch in enumerate(
                epoch_batches(datasets["train"], args.bs, shuffle=True, drop_last=True, rng=rng), start=1
            ):
                params, opt_state, metrics, grad_norm = train_step(params, opt_state, batch)
                ema.update(params)
                size = int(batch["digit"].shape[0])
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + float(value) * size
                seen += size; step += 1; last_grad_norm = float(grad_norm)
                if batch_index % max(1, args.speed_log_freq) == 0:
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
                    logger.info(
                        "epoch=%d step=%d/%d global_step=%d %s step_time=%.2fs iter/s=%.3f sample/s=%.3f epoch_iter/s=%.3f epoch_sample/s=%.3f eta=%.1fs",
                        epoch + 1, batch_index, total_batches, step,
                        _progress_description("train", _mean_metrics(totals, seen), last_grad_norm).removeprefix(" => train | "),
                        step_dt, iter_per_sec, sample_per_sec, epoch_iter_per_sec,
                        epoch_sample_per_sec, eta_sec,
                    )
                    writer.add_scalar("speed/step_time_sec", step_dt, step)
                    writer.add_scalar("speed/iter_per_sec", iter_per_sec, step)
                    writer.add_scalar("speed/sample_per_sec", sample_per_sec, step)
                    writer.add_scalar("speed/epoch_iter_per_sec", epoch_iter_per_sec, step)
                    writer.add_scalar("speed/epoch_sample_per_sec", epoch_sample_per_sec, step)
                    writer.add_scalar("speed/eta_sec", eta_sec, step)
                    writer.add_scalar("train/grad_norm", last_grad_norm, step)
                    speed_window_t0 = sync_t0
                    speed_window_step = batch_index
                    speed_window_samples = seen
                if args.benchmark_steps and step >= args.benchmark_steps:
                    break
            train_stats = _mean_metrics(totals, seen)
            train_time = time.perf_counter() - epoch_step_t0
            checkpoint_due = _checkpoint_due(epoch + 1, args.checkpoint_freq)
            valid_stats: Dict[str, float] | None = None
            if checkpoint_due:
                # Validation and plots use EMA weights so saved samples match inference.
                valid_stats = _eval_epoch(graphdef, ema.params, datasets["valid"], args.bs, rng)
                final_stats = valid_stats
                _plot_joint(args, graphdef, ema.params, datasets["train"], step)
            epoch_iter_per_sec = total_batches / max(train_time, 1e-12)
            epoch_sample_per_sec = seen / max(train_time, 1e-12)
            total_time = time.perf_counter() - epoch_t0
            _write_epoch_summary(
                writer, epoch=epoch + 1, step=step, train_stats=train_stats,
                valid_stats=valid_stats, train_time=train_time, total_time=total_time,
                iter_per_sec=epoch_iter_per_sec, sample_per_sec=epoch_sample_per_sec,
                grad_norm=last_grad_norm,
            )
            _log_epoch_summary(
                logger, epoch=epoch + 1, step=step, train_stats=train_stats,
                valid_stats=valid_stats, train_time=train_time, total_time=total_time,
                iter_per_sec=epoch_iter_per_sec, sample_per_sec=epoch_sample_per_sec,
                grad_norm=last_grad_norm,
            )
            if checkpoint_due and valid_stats is not None and valid_stats["loss"] < best_loss:
                best_loss = valid_stats["loss"]
                _submit_best_checkpoint(
                    artifact_writer, args, params, ema, opt_state,
                    epoch + 1, step, best_loss,
                )
                logger.info("Model checkpoint enqueued: %s queue=%s", args.checkpoint_dir, artifact_writer.stats)
            writer.flush()
            if checkpoint_due and args.remote_save_dir:
                metric_artifact_writer.submit(_sync_metric_artifacts, args)
                logger.info(
                    "metric_artifacts_enqueued epoch=%d step=%d queue=%s",
                    epoch + 1, step, metric_artifact_writer.stats,
                )
            if args.benchmark_steps and step >= args.benchmark_steps:
                break
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
    """Run SCM training directly from a typed experiment configuration."""
    run_dir = output_dir(config)
    _run(_run_arguments(config))
    validate_artifacts(run_dir)
    return str(run_dir)
