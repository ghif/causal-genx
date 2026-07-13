from __future__ import annotations

# ruff: noqa: E402 -- backend selection must happen before importing JAX.

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

from runtime import configure_backend_from_argv

configure_backend_from_argv()

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx

from pgm.flow_pgm import MorphoMNISTPGM
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


def setup_logging(args: argparse.Namespace) -> logging.Logger:
    ensure_dir(args.save_dir)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    handlers.append(
        logging.FileHandler(os.path.join(args.save_dir, "trainlog.txt"), mode="a")
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s, %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(args.exp_name or "morphomnist-pgm")


def _validate_scope(args: argparse.Namespace) -> None:
    if args.dataset != "morphomnist":
        raise ValueError("The pure-JAX parity port supports only --dataset morphomnist")
    if args.setup != "sup_pgm":
        raise ValueError("The pure-JAX parity port supports only --setup sup_pgm")
    if args.precision != "fp32":
        raise ValueError("PGM parity requires --precision fp32")


def _configure_dataset_args(args: argparse.Namespace) -> None:
    if args.input_channels != 1 or args.input_res != 32 or args.pad != 4:
        raise ValueError(
            "Morpho-MNIST PGM parity requires input_channels=1, input_res=32, and pad=4"
        )
    args.parents_x = ["thickness", "intensity", "digit"]
    args.context_norm = "[-1,1]"
    args.context_dim = 12
    args.concat_pa = False


def preprocess(batch: Dict[str, np.ndarray]) -> Dict[str, jax.Array]:
    """Convert a Morpho-MNIST PGM batch to the shapes used by the Pyro model."""
    x = np.asarray(batch["x"], dtype=np.float32)
    if x.max(initial=0.0) > 1.5:
        x = (x - 127.5) / 127.5
    return {
        "x": jnp.asarray(x),
        "thickness": jnp.asarray(batch["thickness"], dtype=jnp.float32).reshape(
            (-1, 1)
        ),
        "intensity": jnp.asarray(batch["intensity"], dtype=jnp.float32).reshape(
            (-1, 1)
        ),
        "digit": jnp.asarray(batch["digit"], dtype=jnp.float32),
    }


def epoch_batches(
    dataset: Any,
    batch_size: int,
    *,
    shuffle: bool,
    drop_last: bool,
    rng: np.random.Generator,
) -> Iterator[Dict[str, jax.Array]]:
    indices = np.arange(len(dataset), dtype=np.int64)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        if drop_last and batch_indices.size < batch_size:
            continue
        if hasattr(dataset, "make_batch"):
            batch = dataset.make_batch(batch_indices, rng=rng, shuffle=shuffle)
        else:
            examples = [dataset[int(index)] for index in batch_indices]
            batch = {
                key: np.stack([np.asarray(item[key]) for item in examples])
                for key in examples[0]
            }
        yield preprocess(batch)


def _loss(graphdef: Any, params: Any, batch: Dict[str, jax.Array]):
    model = materialize_nnx(graphdef, params)
    log_probs = model.log_prob(batch["thickness"], batch["intensity"], batch["digit"])
    loss = -jnp.mean(log_probs["joint"])
    metrics = {"loss": loss}
    metrics.update(
        {
            f"logp({name})": jnp.mean(log_probs[name])
            for name in ("digit", "thickness", "intensity")
        }
    )
    return loss, metrics


def make_train_step(graphdef: Any, optimizer: optax.GradientTransformation):
    @jax.jit
    def train_step(params: Any, opt_state: Any, batch: Dict[str, jax.Array]):
        (loss, metrics), grads = jax.value_and_grad(_loss, argnums=1, has_aux=True)(
            graphdef, params, batch
        )
        grad_norm = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, metrics, grad_norm

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
            self.params = tree_copy(params)
            self.initted = True
        epoch = max(self.step - self.update_after_step - 1, 0)
        decay = 0.0 if epoch <= 0 else min(1.0 - (1.0 + epoch) ** -1.0, self.beta)
        self.params = jax.tree_util.tree_map(
            lambda ema_value, value: ema_value * decay + value * (1.0 - decay),
            self.params,
            params,
        )


def _mean_metrics(totals: Dict[str, float], count: int) -> Dict[str, float]:
    return {key: value / max(1, count) for key, value in totals.items()}


def eval_epoch(
    graphdef: Any,
    params: Any,
    dataset: Any,
    batch_size: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    count = 0
    for batch in epoch_batches(
        dataset, batch_size, shuffle=False, drop_last=False, rng=rng
    ):
        _, metrics = _loss(graphdef, params, batch)
        size = int(batch["digit"].shape[0])
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * size
        count += size
    return _mean_metrics(totals, count)


def _plot_joint(
    args: argparse.Namespace, graphdef: Any, params: Any, dataset: Any, step: int
) -> None:
    data_path = os.path.join(args.save_dir, "joint_data.pdf")
    if not os.path.exists(data_path):
        _joint_figure(
            np.asarray(dataset.samples["thickness"]),
            np.asarray(dataset.samples["intensity"]),
            "Data Joint",
            data_path,
        )
    model = materialize_nnx(graphdef, params)
    samples = model.sample(args.plot_samples, jax.random.PRNGKey(args.seed + step))
    _joint_figure(
        np.asarray(samples["thickness"]).squeeze(),
        np.asarray(samples["intensity"]).squeeze(),
        f"Model Joint (step {step})",
        os.path.join(args.save_dir, f"joint_model_{step}.pdf"),
    )


def _joint_figure(x: np.ndarray, y: np.ndarray, title: str, path: str) -> None:
    figure = plt.figure(figsize=(6, 6))
    grid = figure.add_gridspec(4, 4, hspace=0.05, wspace=0.05)
    joint = figure.add_subplot(grid[1:, :3])
    top = figure.add_subplot(grid[0, :3], sharex=joint)
    right = figure.add_subplot(grid[1:, 3], sharey=joint)
    joint.scatter(x, y, s=2, alpha=0.2)
    top.hist(x, bins=50)
    right.hist(y, bins=50, orientation="horizontal")
    joint.set_xlabel("thickness")
    joint.set_ylabel("intensity")
    figure.suptitle(title)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _checkpoint_payload(
    args: argparse.Namespace,
    params: Any,
    ema: PGMEMA,
    opt_state: Any,
    epoch: int,
    step: int,
    best_loss: float,
) -> Dict[str, Any]:
    return {
        "params": ema.params,
        "ema_params": ema.params,
        "model_params": params,
        "opt_state": opt_state,
        "epoch": epoch,
        "step": step,
        "best_loss": best_loss,
        "ema_step": ema.step,
        "ema_initted": ema.initted,
        "hparams": vars(args),
        "format_version": 2,
    }


def _restore_args(args: argparse.Namespace, checkpoint: Dict[str, Any]) -> None:
    saved = checkpoint.get("hparams", {})
    preserved = {
        "accelerator": args.accelerator,
        "data_dir": args.data_dir,
        "load_path": args.load_path,
        "testing": args.testing,
        "remote_ckpt_dir": args.remote_ckpt_dir,
    }
    for key, value in saved.items():
        if hasattr(args, key):
            setattr(args, key, value)
    for key, value in preserved.items():
        setattr(args, key, value)


def _assert_compatible_checkpoint(checkpoint: Dict[str, Any], params: Any) -> None:
    if checkpoint.get("format_version") != 2 or "model_params" not in checkpoint:
        raise ValueError(
            "This checkpoint belongs to the old simplified Gaussian/CNN PGM. "
            "Retrain it with the flow-parity train_pgm.py implementation."
        )
    if jax.tree_util.tree_structure(
        checkpoint["model_params"]
    ) != jax.tree_util.tree_structure(params):
        raise ValueError(
            "Checkpoint parameter structure does not match the Pyro-compatible JAX PGM"
        )


def main(args: argparse.Namespace) -> Dict[str, float]:
    from datasets import morphomnist

    _validate_scope(args)
    checkpoint: Optional[Dict[str, Any]] = None
    if args.load_path:
        checkpoint = load_checkpoint(args.load_path)
        _restore_args(args, checkpoint)
        _validate_scope(args)
    _configure_dataset_args(args)
    seed_all(args.seed, args.deterministic)

    args.save_dir = experiment_run_dir(args.ckpt_dir, "t_i_d", args.exp_name, "pgm")
    args.checkpoint_dir = checkpoint_root_dir(args.save_dir)
    args.remote_save_dir = experiment_run_dir(
        args.remote_ckpt_dir, "t_i_d", args.exp_name, "pgm"
    )
    ensure_dir(args.save_dir)
    ensure_dir(args.checkpoint_dir)
    logger = setup_logging(args)
    writer = SummaryWriter(args.save_dir)
    datasets = morphomnist(args)

    model = MorphoMNISTPGM(widths=args.widths, rngs=nnx.Rngs(args.seed))
    graphdef, _ = nnx.split(model, nnx.Param)
    params = nnx.state(model, nnx.Param).to_pure_dict()
    optimizer = optax.chain(
        optax.clip_by_global_norm(200.0),
        optax.adamw(args.lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=args.wd),
    )
    opt_state = optimizer.init(params)
    ema = PGMEMA.init_from(params)
    start_epoch = 0
    step = 0
    best_loss = float("inf")

    if checkpoint is not None:
        _assert_compatible_checkpoint(checkpoint, params)
        params = checkpoint["model_params"]
        opt_state = checkpoint["opt_state"]
        ema = PGMEMA(
            params=checkpoint.get("ema_params", checkpoint["params"]),
            step=int(checkpoint.get("ema_step", checkpoint.get("step", 0))),
            initted=bool(checkpoint.get("ema_initted", True)),
        )
        start_epoch = int(checkpoint.get("epoch", 0))
        step = int(checkpoint.get("step", 0))
        best_loss = float(checkpoint.get("best_loss", float("inf")))
        logger.info(
            "Loaded checkpoint %s at epoch=%d step=%d",
            args.load_path,
            start_epoch,
            step,
        )

    rng = np.random.default_rng(args.seed)
    if args.testing:
        if checkpoint is None:
            raise ValueError("--testing requires --load_path")
        stats = eval_epoch(graphdef, ema.params, datasets["test"], args.bs, rng)
        logger.info(
            "test | %s",
            " - ".join(f"{key}: {value:.4f}" for key, value in stats.items()),
        )
        _plot_joint(args, graphdef, ema.params, datasets["test"], 0)
        writer.close()
        return stats

    for key in sorted(vars(args)):
        logger.info("--%s=%s", key, getattr(args, key))
    train_step = make_train_step(graphdef, optimizer)
    final_stats: Dict[str, float] = {}
    for epoch in range(start_epoch, args.epochs):
        totals: Dict[str, float] = {}
        seen = 0
        last_grad_norm = 0.0
        for batch in epoch_batches(
            datasets["train"], args.bs, shuffle=True, drop_last=True, rng=rng
        ):
            params, opt_state, metrics, grad_norm = train_step(params, opt_state, batch)
            ema.update(params)
            size = int(batch["digit"].shape[0])
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value) * size
            seen += size
            step += 1
            last_grad_norm = float(grad_norm)
            if args.benchmark_steps and step >= args.benchmark_steps:
                break
        train_stats = _mean_metrics(totals, seen)
        logger.info(
            "Epoch %d: train loss=%.4f grad_norm=%.3f steps=%d",
            epoch + 1,
            train_stats["loss"],
            last_grad_norm,
            step,
        )

        if epoch % args.eval_freq != 0:
            if args.benchmark_steps and step >= args.benchmark_steps:
                break
            continue
        valid_stats = eval_epoch(graphdef, ema.params, datasets["valid"], args.bs, rng)
        final_stats = valid_stats
        logger.info(
            "loss | train: %.4f - valid: %.4f - steps: %d",
            train_stats["loss"],
            valid_stats["loss"],
            step,
        )
        for key in train_stats:
            writer.add_scalar(f"train/{key}", train_stats[key], step)
            writer.add_scalar(f"valid/{key}", valid_stats[key], step)
        writer.add_scalar("elbo/train", train_stats["loss"], step)
        writer.add_scalar("elbo/valid", valid_stats["loss"], step)
        _plot_joint(args, graphdef, ema.params, datasets["train"], step)

        if valid_stats["loss"] < best_loss:
            best_loss = valid_stats["loss"]
            payload = _checkpoint_payload(
                args, params, ema, opt_state, epoch + 1, step, best_loss
            )
            save_checkpoint(
                payload,
                args.checkpoint_dir,
                step=step,
                custom_metadata={"epoch": epoch + 1, "best_loss": best_loss},
            )
            if args.remote_save_dir:
                sync_tree(
                    args.checkpoint_dir,
                    os.path.join(args.remote_save_dir, "checkpoints"),
                )
            logger.info("Model saved: %s", args.checkpoint_dir)
        writer.flush()
        if args.remote_save_dir:
            sync_file(
                os.path.join(args.save_dir, "trainlog.txt"),
                os.path.join(args.remote_save_dir, "trainlog.txt"),
            )

        if args.benchmark_steps and step >= args.benchmark_steps:
            break
    writer.close()
    return final_stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", default="cpu", choices=["cpu", "gpu", "tpu"])
    parser.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--exp_name", default="")
    parser.add_argument("--dataset", default="morphomnist")
    parser.add_argument(
        "--data_dir", default="gs://medical-airnd/causal-gen/datasets/morphomnist"
    )
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--remote_ckpt_dir", default="")
    parser.add_argument("--load_path", default="")
    parser.add_argument("--resume", dest="load_path", help=argparse.SUPPRESS)
    parser.add_argument("--setup", default="sup_pgm")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--input_res", type=int, default=32)
    parser.add_argument("--input_channels", type=int, default=1)
    parser.add_argument("--pad", type=int, default=4)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--widths", nargs="+", type=int, default=[32, 32])
    parser.add_argument("--plot_samples", type=int, default=10000)
    parser.add_argument("--benchmark_steps", type=int, default=0)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
