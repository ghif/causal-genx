from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import copy
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from utils import EMA, batch_iterator, ensure_dir, linear_warmup, load_checkpoint, materialize_nnx, postprocess, save_checkpoint, sync_tree, write_images


def preprocess_batch(args, batch, expand_pa: bool = False):
    x = np.asarray(batch["x"], dtype=np.float32)
    if x.max() > 1.5:
        x = (x - 127.5) / 127.5
    pa = np.asarray(batch["pa"], dtype=np.float32)
    if expand_pa or pa.ndim == 2:
        pa = pa[:, :, None, None]
        pa = np.repeat(pa, args.input_res, axis=2)
        pa = np.repeat(pa, args.input_res, axis=3)
    return {"x": jnp.asarray(x).transpose(0, 2, 3, 1), "pa": jnp.asarray(pa).transpose(0, 2, 3, 1) if pa.ndim == 4 else jnp.asarray(pa)}


@dataclass
class TrainState:
    params: Any
    opt_state: Any
    ema: EMA
    step: int = 0
    epoch: int = 0
    best_loss: float = float("inf")


def make_optimizer(args):
    beta1, beta2 = args.betas if hasattr(args, "betas") else (0.9, 0.9)
    return optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(learning_rate=args.lr, b1=beta1, b2=beta2, weight_decay=args.wd),
    )


def init_state(model, args, sample_batch, rng):
    params = nnx.state(model, nnx.Param).to_pure_dict()
    tx = make_optimizer(args)
    return TrainState(params=params, opt_state=tx.init(params), ema=EMA.init_from(params, args.ema_rate)), tx


def loss_fn(graphdef, params, batch, beta, rng):
    model = materialize_nnx(graphdef, params)
    out = model(batch["x"], batch["pa"], beta=beta, rng=rng)
    return out["elbo"], out


def train_step(graphdef, state: TrainState, batch, args, tx, rng):
    def _loss(params):
        return loss_fn(graphdef, params, batch, args.beta, rng)

    (loss, out), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)
    updates, new_opt_state = tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    new_ema = EMA(params=state.ema.params, decay=state.ema.decay)
    new_ema.update(new_params)
    return TrainState(new_params, new_opt_state, new_ema, step=state.step + 1, epoch=state.epoch, best_loss=state.best_loss), out


def eval_step(graphdef, params, batch, args, rng):
    model = materialize_nnx(graphdef, params)
    return model(batch["x"], batch["pa"], beta=args.beta, rng=rng)


def _tree_allclose(a, b, atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    a_leaves = jax.tree_util.tree_leaves(a)
    b_leaves = jax.tree_util.tree_leaves(b)
    if len(a_leaves) != len(b_leaves):
        return False
    return all(np.allclose(np.asarray(x), np.asarray(y), atol=atol, rtol=rtol) for x, y in zip(a_leaves, b_leaves))


def checkpoint_smoke_test(args, state: TrainState, tx, logger) -> None:
    restore_template = {
        "epoch": state.epoch,
        "step": state.step,
        "best_loss": state.best_loss,
        "params": state.params,
        "ema_params": state.ema.params,
        "opt_state": state.opt_state,
    }
    save_state(args, state, tx, state.epoch)
    restored = load_checkpoint(args.checkpoint_dir, template=restore_template)
    checks = [
        restored["epoch"] == state.epoch,
        restored["step"] == state.step,
        np.isclose(float(restored["best_loss"]), float(state.best_loss)),
        _tree_allclose(restored["params"], state.params),
        _tree_allclose(restored["ema_params"], state.ema.params),
        _tree_allclose(restored["opt_state"], state.opt_state),
    ]
    if not all(checks):
        raise RuntimeError("Checkpoint smoke test failed: restored training state does not match the saved state.")
    logger.info("checkpoint_smoke_test=passed checkpoint_dir=%s step=%d", args.checkpoint_dir, state.step)


def save_state(args, state: TrainState, tx, epoch):
    ckpt = {
        "epoch": epoch,
        "step": state.step,
        "best_loss": state.best_loss,
        "params": state.params,
        "ema_params": state.ema.params,
        "opt_state": state.opt_state,
        "hparams": vars(args),
    }
    path = args.checkpoint_dir
    save_checkpoint(ckpt, path, step=state.step, custom_metadata={"epoch": epoch, "best_loss": float(state.best_loss)})
    if hasattr(args, "remote_save_dir"):
        sync_tree(args.save_dir, args.remote_save_dir)
    return path


def trainer(args, graphdef, state: TrainState, tx, datasets, writer, logger):
    ensure_dir(args.save_dir)
    train_iter = batch_iterator(datasets["train"], args.bs, True, args.seed)
    valid_iter = batch_iterator(datasets["valid"], args.bs, False, args.seed + 1)
    steps_per_epoch = max(1, len(datasets["train"]) // args.bs)
    total_train_steps = steps_per_epoch * max(1, args.epochs - state.epoch)
    for epoch in range(state.epoch, args.epochs):
        t0 = time.perf_counter()
        epoch_step_t0 = time.perf_counter()
        train_stats = []
        for step in range(max(1, len(datasets["train"]) // args.bs)):
            fetch_t0 = time.perf_counter()
            batch = preprocess_batch(args, next(train_iter), expand_pa=True)
            batch_ready_t0 = time.perf_counter()
            rng = jax.random.PRNGKey(args.seed + state.step + step + epoch * 1000)
            state, out = train_step(graphdef, state, batch, args, tx, rng)
            step_end_t0 = time.perf_counter()
            train_stats.append(out)
            if (step + 1) % max(1, args.speed_log_freq) == 0:
                data_dt = batch_ready_t0 - fetch_t0
                compute_dt = step_end_t0 - batch_ready_t0
                step_dt = step_end_t0 - fetch_t0
                iter_per_sec = 1.0 / max(step_dt, 1e-12)
                sample_per_sec = args.bs / max(step_dt, 1e-12)
                epoch_elapsed = step_end_t0 - epoch_step_t0
                epoch_iters_per_sec = (step + 1) / max(epoch_elapsed, 1e-12)
                epoch_samples_per_sec = (step + 1) * args.bs / max(epoch_elapsed, 1e-12)
                train_steps_done = (epoch - state.epoch) * steps_per_epoch + (step + 1)
                train_steps_left = max(0, total_train_steps - train_steps_done)
                eta_sec = train_steps_left / max(epoch_iters_per_sec, 1e-12)
                logger.info(
                    "epoch=%d step=%d/%d global_step=%d elbo=%.4f data_time=%.2fs compute_time=%.2fs step_time=%.2fs iter/s=%.3f sample/s=%.3f epoch_iter/s=%.3f epoch_sample/s=%.3f eta=%.1fs",
                    epoch + 1,
                    step + 1,
                    steps_per_epoch,
                    state.step,
                    float(out["elbo"]),
                    data_dt,
                    compute_dt,
                    step_dt,
                    iter_per_sec,
                    sample_per_sec,
                    epoch_iters_per_sec,
                    epoch_samples_per_sec,
                    eta_sec,
                )
                if hasattr(writer, "add_scalar"):
                    writer.add_scalar("speed/data_time_sec", data_dt, state.step)
                    writer.add_scalar("speed/compute_time_sec", compute_dt, state.step)
                    writer.add_scalar("speed/iter_per_sec", iter_per_sec, state.step)
                    writer.add_scalar("speed/sample_per_sec", sample_per_sec, state.step)
                    writer.add_scalar("speed/step_time_sec", step_dt, state.step)
                    writer.add_scalar("speed/epoch_iter_per_sec", epoch_iters_per_sec, state.step)
                    writer.add_scalar("speed/epoch_sample_per_sec", epoch_samples_per_sec, state.step)
                    writer.add_scalar("speed/eta_sec", eta_sec, state.step)
            if getattr(args, "checkpoint_smoke_test", False) and state.step >= max(1, args.checkpoint_smoke_steps):
                checkpoint_smoke_test(args, state, tx, logger)
                return
        valid_batch = preprocess_batch(args, next(valid_iter), expand_pa=True)
        valid_out = eval_step(graphdef, state.ema.params, valid_batch, args, jax.random.PRNGKey(args.seed + epoch))
        if float(valid_out["elbo"]) < state.best_loss:
            state.best_loss = float(valid_out["elbo"])
            save_state(args, state, tx, epoch + 1)
        if hasattr(writer, "add_scalar"):
            writer.add_scalar("train/elbo", float(jnp.mean(jnp.array([o["elbo"] for o in train_stats]))), epoch + 1)
            writer.add_scalar("valid/elbo", float(valid_out["elbo"]), epoch + 1)
        if args.viz_freq and (epoch + 1) % args.viz_freq == 0:
            write_images(args, graphdef, state.ema.params, valid_batch, jax.random.PRNGKey(args.seed + epoch))
        epoch_time = time.perf_counter() - t0
        epoch_iter_per_sec = steps_per_epoch / max(epoch_time, 1e-12)
        epoch_sample_per_sec = steps_per_epoch * args.bs / max(epoch_time, 1e-12)
        logger.info(
            "epoch=%d valid_elbo=%.4f epoch_time=%.1fs epoch_iter/s=%.3f epoch_sample/s=%.3f",
            epoch + 1,
            float(valid_out["elbo"]),
            epoch_time,
            epoch_iter_per_sec,
            epoch_sample_per_sec,
        )
