from __future__ import annotations

# ruff: noqa: E402 -- backend selection must happen before importing JAX.

import argparse
import logging
import os
import warnings
from glob import glob
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
from tqdm import tqdm

from pgm.flow_pgm import MorphoMNISTPGM
from pgm.sup_aux_pgm import MorphoMNISTSupAuxPredictor
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
        grad_norm = optax.tree.norm(grads)
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


def _progress_description(
    mode: str, stats: Dict[str, float], grad_norm: Optional[float] = None
) -> str:
    description = f" => {mode} | " + ", ".join(
        f"{key}: {value:.4f}" for key, value in stats.items()
    )
    if grad_norm is not None:
        description += f", grad_norm: {grad_norm:.3f}"
    return description


def _writer_add_custom_scalars(writer: Any) -> None:
    if hasattr(writer, "add_custom_scalars"):
        writer.add_custom_scalars(
            {"elbo": {"elbo": ["Multiline", ["elbo/train", "elbo/valid"]]}}
        )


def _sup_aux_progress_description(
    mode: str, stats: Dict[str, float], grad_norm: Optional[float] = None
) -> str:
    ordered_keys = [
        "loss",
        "logp(digit_aux)",
        "logp(thickness_aux)",
        "logp(intensity_aux)",
    ]
    parts = [f"{key}: {stats[key]:.4f}" for key in ordered_keys if key in stats]
    description = f" => {mode} | " + ", ".join(parts)
    if grad_norm is not None:
        description += f", grad_norm: {grad_norm:.3f}"
    return description


def _sup_aux_prediction_description(metrics: Dict[str, float]) -> str:
    ordered_keys = ["thickness_mae", "intensity_mae", "digit_acc"]
    parts = [f"{key}: {metrics[key]:.4f}" for key in ordered_keys if key in metrics]
    return " - ".join(parts)


def eval_epoch(
    graphdef: Any,
    params: Any,
    dataset: Any,
    batch_size: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    count = 0
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    progress = tqdm(
        epoch_batches(dataset, batch_size, shuffle=False, drop_last=False, rng=rng),
        total=total_batches,
        miniters=max(1, total_batches // 100),
        mininterval=5,
    )
    for batch in progress:
        _, metrics = _loss(graphdef, params, batch)
        size = int(batch["digit"].shape[0])
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * size
        count += size
        progress.set_description(
            _progress_description("eval", _mean_metrics(totals, count))
        )
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


def _sync_pdf_artifacts(args: argparse.Namespace) -> None:
    if not args.remote_save_dir:
        return
    for pdf_path in sorted(glob(os.path.join(args.save_dir, "*.pdf"))):
        sync_file(
            pdf_path, os.path.join(args.remote_save_dir, os.path.basename(pdf_path))
        )


class _IndexedDataset:
    def __init__(self, dataset: Any, indices: np.ndarray):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.min_max = getattr(dataset, "min_max", {})
        self.samples = getattr(dataset, "samples", {})

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int):
        return self.dataset[int(self.indices[int(idx)])]

    def make_batch(self, batch_idx, rng=None, shuffle: bool = False):
        mapped_idx = self.indices[np.asarray(batch_idx, dtype=np.int64)]
        return self.dataset.make_batch(mapped_idx, rng=rng, shuffle=shuffle)


@dataclass
class _WarmupEMA:
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
    def init_from(cls, params: Any, batch_stats: Any) -> "_WarmupEMA":
        return cls(params=tree_copy(params), batch_stats=tree_copy(batch_stats))

    def _decay(self) -> float:
        epoch = max(self.step - self.update_after_step - 1, 0)
        if epoch <= 0:
            return 0.0
        value = 1.0 - (1.0 + epoch / self.inv_gamma) ** (-self.power)
        return min(max(value, self.min_value), self.beta)

    def update(self, params: Any, batch_stats: Any) -> None:
        current_step = self.step
        self.step += 1
        if current_step <= self.update_after_step:
            self.params = tree_copy(params)
            self.batch_stats = tree_copy(batch_stats)
            return
        if not self.initted:
            self.params = tree_copy(params)
            self.batch_stats = tree_copy(batch_stats)
            self.initted = True
        decay = self._decay()
        self.params = jax.tree_util.tree_map(
            lambda ema_value, value: ema_value * decay + value * (1.0 - decay),
            self.params,
            params,
        )
        self.batch_stats = jax.tree_util.tree_map(
            lambda ema_value, value: ema_value * decay + value * (1.0 - decay),
            self.batch_stats,
            batch_stats,
        )


def _setup_sup_aux_scope(args: argparse.Namespace) -> None:
    if args.dataset != "morphomnist":
        raise ValueError("The pure-JAX parity port supports only --dataset morphomnist")
    if args.setup != "sup_aux":
        raise ValueError("The predictor parity port supports only --setup sup_aux")
    if args.accelerator == "cpu" and args.precision != "fp32":
        raise ValueError("CPU predictor training requires --precision fp32")


def _validate_sup_aux_runtime_device(args: argparse.Namespace) -> jax.Device:
    devices = jax.devices()
    if args.accelerator == "gpu":
        matching = [device for device in devices if device.platform in {"gpu", "cuda"}]
        if not matching:
            raise RuntimeError(
                "--accelerator gpu requested, but JAX found no CUDA GPU. Install a "
                "CUDA-enabled JAX build compatible with the driver and CUDA runtime."
            )
        if len(matching) != 1:
            raise RuntimeError(
                f"Predictor training requires one visible GPU, found {len(matching)}. "
                "Set --gpu_id or CUDA_VISIBLE_DEVICES to a single device."
            )
    else:
        matching = [device for device in devices if device.platform == args.accelerator]
        if not matching:
            raise RuntimeError(
                f"--accelerator {args.accelerator} requested, but JAX devices are {devices}"
            )
    device = matching[0]
    print(f"JAX device preflight passed: platform={device.platform} device={device}")
    return device


def _configure_sup_aux_compute_policy(args: argparse.Namespace) -> jnp.dtype:
    compute_dtype = (
        jnp.bfloat16
        if args.precision == "bf16" and args.accelerator in {"gpu", "tpu"}
        else jnp.float32
    )
    matmul_precision = "default" if compute_dtype == jnp.bfloat16 else "highest"
    jax.config.update("jax_default_matmul_precision", matmul_precision)
    print(
        "JAX predictor compute policy: "
        f"precision={args.precision} compute_dtype={compute_dtype} "
        f"master_params=fp32 optimizer_state=fp32 matmul_precision={matmul_precision}"
    )
    return compute_dtype


def _configure_sup_aux_dataset_args(args: argparse.Namespace) -> None:
    if args.input_channels != 1 or args.input_res != 32 or args.pad != 4:
        raise ValueError(
            "Morpho-MNIST predictor parity requires input_channels=1, input_res=32, and pad=4"
        )
    args.parents_x = ["thickness", "intensity", "digit"]
    args.context_norm = "[-1,1]"
    args.context_dim = 12
    args.concat_pa = False


def _build_sup_aux_datasets(args: argparse.Namespace):
    from datasets import morphomnist

    datasets = morphomnist(args)
    total = len(datasets["train"])
    labelled = int(args.sup_frac * total)
    indices = np.arange(total)
    rng = np.random.RandomState(1)
    rng.shuffle(indices)
    train = _IndexedDataset(datasets["train"], indices[:labelled])
    return datasets, train


def _sup_aux_merge(graphdef: Any, params: Any, batch_stats: Any):
    return nnx.merge(graphdef, params, batch_stats)


def _sup_aux_loss_and_state(
    graphdef: Any,
    params: Any,
    batch_stats: Any,
    batch: Dict[str, jax.Array],
    *,
    training: bool,
):
    model = _sup_aux_merge(graphdef, params, batch_stats)
    if training:
        model.train()
    else:
        model.eval()
    log_probs = model.model_anticausal(**batch)
    new_params = nnx.state(model, nnx.Param).to_pure_dict()
    new_batch_stats = nnx.state(model, nnx.BatchStat).to_pure_dict()
    loss = -jnp.mean(log_probs["joint"])
    metrics = {"loss": loss}
    for key in ("thickness_aux", "intensity_aux", "digit_aux"):
        metrics[f"logp({key})"] = jnp.mean(log_probs[key])
    return loss, metrics, new_params, new_batch_stats


def _sup_aux_make_train_step(graphdef: Any, optimizer: optax.GradientTransformation):
    @jax.jit
    def train_step(params, batch_stats, opt_state, batch):
        def _loss_fn(p):
            loss, metrics, new_params, new_batch_stats = _sup_aux_loss_and_state(
                graphdef, p, batch_stats, batch, training=True
            )
            return loss, (metrics, new_params, new_batch_stats)

        (loss, (metrics, new_params, new_batch_stats)), grads = jax.value_and_grad(
            _loss_fn, has_aux=True
        )(params)
        grad_norm = optax.tree.norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, new_batch_stats, opt_state, metrics, grad_norm

    return train_step


def _sup_aux_eval_epoch(
    graphdef: Any,
    params: Any,
    batch_stats: Any,
    dataset: Any,
    batch_size: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    count = 0
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    progress = tqdm(
        epoch_batches(dataset, batch_size, shuffle=False, drop_last=False, rng=rng),
        total=total_batches,
        miniters=max(1, total_batches // 100),
        mininterval=5,
    )
    for batch in progress:
        _, metrics, _, _ = _sup_aux_loss_and_state(
            graphdef, params, batch_stats, batch, training=False
        )
        size = int(batch["digit"].shape[0])
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * size
        count += size
        progress.set_description(
            _sup_aux_progress_description(
                "eval", {k: v / max(1, count) for k, v in totals.items()}
            )
        )
    return {key: value / max(1, count) for key, value in totals.items()}


def _sup_aux_predict_metrics(
    args: argparse.Namespace,
    model: Any,
    dataset: Any,
    batch_size: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    model.eval()
    preds = {k: [] for k in model.variables.keys()}
    targets = {k: [] for k in model.variables.keys()}
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    for batch in tqdm(
        epoch_batches(dataset, batch_size, shuffle=False, drop_last=False, rng=rng),
        total=total_batches,
        miniters=max(1, total_batches // 100),
        mininterval=5,
    ):
        for k in targets:
            targets[k].extend(np.asarray(batch[k]))
        out = model.predict(**batch)
        for k, value in out.items():
            preds[k].extend(np.asarray(value))
    stats: Dict[str, float] = {}
    if args.dataset == "morphomnist":
        for k in model.variables.keys():
            if k == "digit":
                pred = np.asarray(preds[k])
                target = np.asarray(targets[k])
                stats["digit_acc"] = float(
                    (target.argmax(-1) == pred.argmax(-1)).mean()
                )
            else:
                min_val, max_val = dataset.min_max[k]
                pred = ((np.asarray(preds[k]).squeeze(-1) + 1.0) / 2.0) * (
                    max_val - min_val
                ) + min_val
                target = ((np.asarray(targets[k]).squeeze(-1) + 1.0) / 2.0) * (
                    max_val - min_val
                ) + min_val
                stats[f"{k}_mae"] = float(np.mean(np.abs(target - pred)))
    else:
        raise NotImplementedError
    return stats


def _sup_aux_checkpoint_payload(
    args: argparse.Namespace,
    model_params: Any,
    batch_stats: Any,
    ema: _WarmupEMA,
    opt_state: Any,
    epoch: int,
    step: int,
    best_loss: float,
) -> Dict[str, Any]:
    return {
        "params": ema.params,
        "ema_params": ema.params,
        "model_params": model_params,
        "batch_stats": batch_stats,
        "ema_batch_stats": ema.batch_stats,
        "opt_state": opt_state,
        "epoch": epoch,
        "step": step,
        "best_loss": best_loss,
        "ema_step": ema.step,
        "ema_initted": ema.initted,
        "hparams": vars(args),
        "format_version": 3,
    }


def _sup_aux_restore_args(args: argparse.Namespace, checkpoint: Dict[str, Any]) -> None:
    saved = checkpoint.get("hparams", {})
    preserved = {
        "accelerator": args.accelerator,
        "precision": args.precision,
        "gpu_id": args.gpu_id,
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


def _sup_aux_assert_compatible_checkpoint(
    checkpoint: Dict[str, Any], params: Any, batch_stats: Any
) -> None:
    hparams = checkpoint.get("hparams", {})
    if checkpoint.get("format_version") != 3 or hparams.get("setup") != "sup_aux":
        raise ValueError(
            "This checkpoint belongs to the old simplified PGM or a different model kind. "
            "Retrain it with the JAX sup_aux predictor implementation."
        )
    if jax.tree_util.tree_structure(
        checkpoint["model_params"]
    ) != jax.tree_util.tree_structure(params):
        raise ValueError(
            "Checkpoint parameter structure does not match the JAX predictor"
        )
    if jax.tree_util.tree_structure(
        checkpoint["batch_stats"]
    ) != jax.tree_util.tree_structure(batch_stats):
        raise ValueError(
            "Checkpoint batch-stat structure does not match the JAX predictor"
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


def _main_sup_aux(args: argparse.Namespace) -> Dict[str, float]:
    checkpoint: Optional[Dict[str, Any]] = None
    if args.load_path:
        checkpoint = load_checkpoint(args.load_path)
        _sup_aux_restore_args(args, checkpoint)
    _setup_sup_aux_scope(args)
    _validate_sup_aux_runtime_device(args)
    compute_dtype = _configure_sup_aux_compute_policy(args)
    _configure_sup_aux_dataset_args(args)
    seed_all(args.seed, args.deterministic)

    args.save_dir = experiment_run_dir(
        args.ckpt_dir, "morphomnist", args.exp_name, "pgm"
    )
    args.checkpoint_dir = checkpoint_root_dir(args.save_dir)
    args.remote_save_dir = experiment_run_dir(
        args.remote_ckpt_dir, "morphomnist", args.exp_name, "pgm"
    )
    ensure_dir(args.save_dir)
    ensure_dir(args.checkpoint_dir)
    logger = setup_logging(args)
    writer = SummaryWriter(args.save_dir)
    datasets, train_dataset = _build_sup_aux_datasets(args)
    valid_dataset = datasets["valid"]
    test_dataset = datasets["test"]

    rngs = nnx.Rngs(args.seed)
    model = MorphoMNISTSupAuxPredictor(
        input_channels=args.input_channels,
        input_res=args.input_res,
        width=8,
        std_fixed=args.std_fixed,
        compute_dtype=compute_dtype,
        rngs=rngs,
    )
    graphdef, params_state, batch_stats_state = nnx.split(
        model, nnx.Param, nnx.BatchStat
    )
    model_params = params_state.to_pure_dict()
    model_batch_stats = batch_stats_state.to_pure_dict()
    optimizer = optax.chain(
        optax.clip_by_global_norm(200.0),
        optax.adamw(args.lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=args.wd),
    )
    opt_state = optimizer.init(model_params)
    ema = _WarmupEMA.init_from(model_params, model_batch_stats)
    start_epoch = 0
    step = 0
    best_loss = float("inf")

    if checkpoint is not None:
        _sup_aux_assert_compatible_checkpoint(
            checkpoint, model_params, model_batch_stats
        )
        model_params = checkpoint["model_params"]
        model_batch_stats = checkpoint["batch_stats"]
        opt_state = checkpoint["opt_state"]
        ema = _WarmupEMA(
            params=checkpoint.get("ema_params", checkpoint["params"]),
            batch_stats=checkpoint["ema_batch_stats"],
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
        eval_model = _sup_aux_merge(graphdef, ema.params, ema.batch_stats)
        stats = _sup_aux_predict_metrics(args, eval_model, test_dataset, args.bs, rng)
        logger.info("test | %s", _sup_aux_prediction_description(stats))
        writer.close()
        return stats

    for key in sorted(vars(args)):
        logger.info("--%s=%s", key, getattr(args, key))
    logger.info(
        "Data splits: #labelled: %d - #unlabelled: %d",
        len(train_dataset),
        len(datasets["train"]) - len(train_dataset),
    )
    train_step = _sup_aux_make_train_step(graphdef, optimizer)
    final_stats: Dict[str, float] = {}
    for epoch in range(start_epoch, args.epochs):
        logger.info("Epoch %d:", epoch + 1)
        totals: Dict[str, float] = {}
        seen = 0
        total_batches = len(train_dataset) // args.bs
        progress = tqdm(
            epoch_batches(
                train_dataset,
                args.bs,
                shuffle=True,
                drop_last=True,
                rng=rng,
            ),
            total=total_batches,
            miniters=max(1, total_batches // 100),
            mininterval=5,
        )
        for batch in progress:
            model_params, model_batch_stats, opt_state, metrics, grad_norm = train_step(
                model_params, model_batch_stats, opt_state, batch
            )
            ema.update(model_params, model_batch_stats)
            size = int(batch["digit"].shape[0])
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value) * size
            seen += size
            step += 1
            progress.set_description(
                _sup_aux_progress_description(
                    "train",
                    {key: value / max(1, seen) for key, value in totals.items()},
                    float(grad_norm),
                )
            )
        train_stats = {key: value / max(1, seen) for key, value in totals.items()}

        valid_stats = _sup_aux_eval_epoch(
            graphdef, ema.params, ema.batch_stats, valid_dataset, args.bs, rng
        )
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
        _writer_add_custom_scalars(writer)
        writer.add_scalar("elbo/train", train_stats["loss"], step)
        writer.add_scalar("elbo/valid", valid_stats["loss"], step)

        metrics = _sup_aux_predict_metrics(
            args,
            _sup_aux_merge(graphdef, ema.params, ema.batch_stats),
            valid_dataset,
            args.bs,
            rng,
        )
        logger.info("valid | %s", _sup_aux_prediction_description(metrics))

        if valid_stats["loss"] < best_loss:
            best_loss = valid_stats["loss"]
            payload = _sup_aux_checkpoint_payload(
                args,
                model_params,
                model_batch_stats,
                ema,
                opt_state,
                epoch + 1,
                step,
                best_loss,
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

    writer.close()
    return final_stats


def main(args: argparse.Namespace) -> Dict[str, float]:
    if args.setup == "sup_aux":
        return _main_sup_aux(args)
    # SCM execution moved to training.scm. Keep this parser-only entrypoint so
    # established launch scripts and argument names remain usable.
    from training.scm import run_legacy_args

    return run_legacy_args(args)

    # Kept below temporarily as a migration reference for numerical parity.
    # It is intentionally unreachable; the public SCM path above is native.
    from datasets import morphomnist

    _validate_scope(args)
    checkpoint: Optional[Dict[str, Any]] = None
    if args.load_path:
        checkpoint = load_checkpoint(args.load_path)
        _restore_args(args, checkpoint)
        _validate_scope(args)
    _configure_dataset_args(args)
    seed_all(args.seed, args.deterministic)

    args.save_dir = experiment_run_dir(
        args.ckpt_dir, "morphomnist", args.exp_name, "pgm"
    )
    args.checkpoint_dir = checkpoint_root_dir(args.save_dir)
    args.remote_save_dir = experiment_run_dir(
        args.remote_ckpt_dir, "morphomnist", args.exp_name, "pgm"
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
        logger.info("test | %s", _sup_aux_prediction_description(stats))
        _plot_joint(args, graphdef, ema.params, datasets["test"], 0)
        _sync_pdf_artifacts(args)
        writer.close()
        return stats

    for key in sorted(vars(args)):
        logger.info("--%s=%s", key, getattr(args, key))
    train_step = make_train_step(graphdef, optimizer)
    final_stats: Dict[str, float] = {}
    for epoch in range(start_epoch, args.epochs):
        logger.info("Epoch %d:", epoch + 1)
        totals: Dict[str, float] = {}
        seen = 0
        last_grad_norm = 0.0
        total_batches = len(datasets["train"]) // args.bs
        progress = tqdm(
            epoch_batches(
                datasets["train"],
                args.bs,
                shuffle=True,
                drop_last=True,
                rng=rng,
            ),
            total=total_batches,
            miniters=max(1, total_batches // 100),
            mininterval=5,
        )
        for batch in progress:
            params, opt_state, metrics, grad_norm = train_step(params, opt_state, batch)
            ema.update(params)
            size = int(batch["digit"].shape[0])
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value) * size
            seen += size
            step += 1
            last_grad_norm = float(grad_norm)
            progress.set_description(
                _progress_description(
                    "train", _mean_metrics(totals, seen), last_grad_norm
                )
            )
            if args.benchmark_steps and step >= args.benchmark_steps:
                break
        train_stats = _mean_metrics(totals, seen)

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
        _sync_pdf_artifacts(args)

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
            _sync_pdf_artifacts(args)

        if args.benchmark_steps and step >= args.benchmark_steps:
            break
    writer.close()
    return final_stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", default="cpu", choices=["cpu", "gpu", "tpu"])
    parser.add_argument("--gpu_id", default="0")
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
    parser.add_argument("--setup", default="sup_pgm", choices=["sup_pgm", "sup_aux"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--sup_frac", type=float, default=1.0)
    parser.add_argument("--input_res", type=int, default=32)
    parser.add_argument("--input_channels", type=int, default=1)
    parser.add_argument("--pad", type=int, default=4)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--widths", nargs="+", type=int, default=[32, 32])
    parser.add_argument("--std_fixed", type=float, default=0.0)
    parser.add_argument("--plot_samples", type=int, default=10000)
    parser.add_argument("--benchmark_steps", type=int, default=0)
    return parser


if __name__ == "__main__":
    warnings.warn("src/pgm/train_pgm.py is a compatibility entrypoint; use scripts/run.py train-scm or train-predictor --config ...", DeprecationWarning, stacklevel=1)
    main(build_parser().parse_args())
