"""Stage 1: native MorphoMNIST structural causal model training.

This module owns the SCM training loop used by ``scripts/run.py train-scm``.
It deliberately retains the established checkpoint payload and run-directory
contract, so existing analysis tools and checkpoints remain usable.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx
from tqdm import tqdm

from causal.flow_scm import MorphoMNISTPGM
from config import ExperimentConfig, ScmTrainingConfig
from data.morphomnist import morphomnist
from utils import (
    SummaryWriter,
    checkpoint_root_dir,
    ensure_dir,
    experiment_run_dir,
    load_checkpoint,
    materialize_nnx,
    save_checkpoint,
    seed_all,
    sync_file,
    sync_tree,
    tree_copy,
)

from .common import stage_run_dir


@dataclass
class ScmRunArguments:
    """Legacy-compatible run settings, constructed from the typed YAML config."""

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
    eval_freq: int
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
        eval_freq=workflow.eval_freq,
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
    progress = tqdm(epoch_batches(dataset, batch_size, shuffle=False, drop_last=False, rng=rng), total=(len(dataset) + batch_size - 1) // batch_size, miniters=max(1, len(dataset) // batch_size // 100), mininterval=5)
    for batch in progress:
        _, metrics = _loss(graphdef, params, batch)
        size = int(batch["digit"].shape[0])
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * size
        count += size
        progress.set_description(_progress_description("eval", _mean_metrics(totals, count)))
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
        for path in sorted(glob(os.path.join(args.save_dir, "*.pdf"))):
            sync_file(path, os.path.join(args.remote_save_dir, os.path.basename(path)))


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
    _validate_scope(args)
    checkpoint: Optional[Dict[str, Any]] = None
    if args.load_path:
        checkpoint = load_checkpoint(args.load_path); _restore_args(args, checkpoint); _validate_scope(args)
    _configure_dataset_args(args); seed_all(args.seed, args.deterministic)
    args.save_dir = experiment_run_dir(args.ckpt_dir, "morphomnist", args.exp_name, "pgm")
    args.checkpoint_dir = checkpoint_root_dir(args.save_dir)
    args.remote_save_dir = experiment_run_dir(args.remote_ckpt_dir, "morphomnist", args.exp_name, "pgm")
    ensure_dir(args.save_dir); ensure_dir(args.checkpoint_dir)
    logger = _setup_logging(args); writer = SummaryWriter(args.save_dir); datasets = morphomnist(args)
    model = MorphoMNISTPGM(widths=args.widths, rngs=nnx.Rngs(args.seed))
    graphdef, _ = nnx.split(model, nnx.Param); params = nnx.state(model, nnx.Param).to_pure_dict()
    optimizer = optax.chain(optax.clip_by_global_norm(200.0), optax.adamw(args.lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=args.wd))
    opt_state = optimizer.init(params); ema = PGMEMA.init_from(params); start_epoch = step = 0; best_loss = float("inf")
    if checkpoint is not None:
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
    for epoch in range(start_epoch, args.epochs):
        logger.info("Epoch %d:", epoch + 1); totals: Dict[str, float] = {}; seen = 0; last_grad_norm = 0.0
        total_batches = len(datasets["train"]) // args.bs
        progress = tqdm(epoch_batches(datasets["train"], args.bs, shuffle=True, drop_last=True, rng=rng), total=total_batches, miniters=max(1, total_batches // 100), mininterval=5)
        for batch in progress:
            params, opt_state, metrics, grad_norm = train_step(params, opt_state, batch); ema.update(params); size = int(batch["digit"].shape[0])
            for key, value in metrics.items(): totals[key] = totals.get(key, 0.0) + float(value) * size
            seen += size; step += 1; last_grad_norm = float(grad_norm)
            progress.set_description(_progress_description("train", _mean_metrics(totals, seen), last_grad_norm))
            if args.benchmark_steps and step >= args.benchmark_steps: break
        train_stats = _mean_metrics(totals, seen)
        if epoch % args.eval_freq != 0:
            if args.benchmark_steps and step >= args.benchmark_steps: break
            continue
        valid_stats = _eval_epoch(graphdef, ema.params, datasets["valid"], args.bs, rng); final_stats = valid_stats
        logger.info("loss | train: %.4f - valid: %.4f - steps: %d", train_stats["loss"], valid_stats["loss"], step)
        for key in train_stats: writer.add_scalar(f"train/{key}", train_stats[key], step); writer.add_scalar(f"valid/{key}", valid_stats[key], step)
        writer.add_scalar("elbo/train", train_stats["loss"], step); writer.add_scalar("elbo/valid", valid_stats["loss"], step)
        _plot_joint(args, graphdef, ema.params, datasets["train"], step); _sync_pdf_artifacts(args)
        if valid_stats["loss"] < best_loss:
            best_loss = valid_stats["loss"]; save_checkpoint(_checkpoint_payload(args, params, ema, opt_state, epoch + 1, step, best_loss), args.checkpoint_dir, step=step, custom_metadata={"epoch": epoch + 1, "best_loss": best_loss})
            if args.remote_save_dir: sync_tree(args.checkpoint_dir, os.path.join(args.remote_save_dir, "checkpoints"))
            logger.info("Model saved: %s", args.checkpoint_dir)
        writer.flush()
        if args.remote_save_dir:
            sync_file(os.path.join(args.save_dir, "trainlog.txt"), os.path.join(args.remote_save_dir, "trainlog.txt")); _sync_pdf_artifacts(args)
        if args.benchmark_steps and step >= args.benchmark_steps: break
    writer.close(); return final_stats


def run(config: ExperimentConfig) -> str:
    """Run SCM training directly from a typed experiment configuration."""
    run_dir = output_dir(config)
    _run(_run_arguments(config))
    validate_artifacts(run_dir)
    return str(run_dir)
