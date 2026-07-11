from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict

from runtime import configure_backend_from_argv

configure_backend_from_argv()

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from utils import EMA, batch_iterator, ensure_dir, linear_warmup, load_checkpoint, materialize_nnx, save_checkpoint, sync_tree, write_images


def preprocess_batch(args, batch, expand_pa: bool = False):
    x = np.asarray(batch["x"], dtype=np.float32)
    if x.max() > 1.5:
        x = (x - 127.5) / 127.5
    pa = np.asarray(batch["pa"], dtype=np.float32)
    if expand_pa or pa.ndim == 2:
        pa = pa[:, :, None, None]
        pa = np.repeat(pa, args.input_res, axis=2)
        pa = np.repeat(pa, args.input_res, axis=3)
    x = jnp.asarray(x).transpose(0, 2, 3, 1)
    pa = jnp.asarray(pa).transpose(0, 2, 3, 1) if pa.ndim == 4 else jnp.asarray(pa)
    if getattr(args, "precision", "fp32") == "bf16":
        x = x.astype(jnp.bfloat16)
        pa = pa.astype(jnp.bfloat16)
    return {"x": x, "pa": pa}


def _block_until_ready(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return tree
    leaves[0].block_until_ready()
    return tree


def _unreplicate(tree):
    return jax.tree_util.tree_map(lambda x: x[0], tree)


def _shard_batch(batch, n_devices: int):
    def _reshape(x):
        x = jnp.asarray(x)
        if x.shape[0] % n_devices != 0:
            raise ValueError(
                f"Batch size {x.shape[0]} must be divisible by local device count {n_devices} for TPU replication."
            )
        per_device = x.shape[0] // n_devices
        return x.reshape((n_devices, per_device) + x.shape[1:])

    return jax.tree_util.tree_map(_reshape, batch)


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
    lr = args.lr
    if getattr(args, "lr_warmup_steps", 0) > 0:
        warmup = linear_warmup(args.lr_warmup_steps)
        lr = lambda step: args.lr * warmup(step)
    tx = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(learning_rate=lr, b1=beta1, b2=beta2, weight_decay=args.wd),
    )
    if getattr(args, "accu_steps", 1) > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=args.accu_steps)
    return tx


def init_state(model, args, sample_batch, rng):
    params = nnx.state(model, nnx.Param).to_pure_dict()
    tx = make_optimizer(args)
    return TrainState(params=params, opt_state=tx.init(params), ema=EMA.init_from(params, args.ema_rate)), tx


def loss_fn(graphdef, params, batch, beta, rng):
    model = materialize_nnx(graphdef, params)
    out = model(batch["x"], batch["pa"], beta=beta, rng=rng)
    return out["elbo"], out


def make_train_step(graphdef, tx, ema_decay: float):
    def _step(params, opt_state, ema_params, batch, beta, rng):
        def _loss(p):
            return loss_fn(graphdef, p, batch, beta, rng)

        (_, out), grads = jax.value_and_grad(_loss, has_aux=True)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_ema = jax.tree_util.tree_map(lambda e, p: ema_decay * e + (1.0 - ema_decay) * p, ema_params, new_params)
        return new_params, new_opt_state, new_ema, out

    return jax.jit(_step)


def make_eval_step(graphdef):
    def _step(params, batch, beta, rng):
        model = materialize_nnx(graphdef, params)
        return model(batch["x"], batch["pa"], beta=beta, rng=rng)

    return jax.jit(_step)


def make_pmap_train_step(graphdef, tx, ema_decay: float):
    def _step(params, opt_state, ema_params, batch, beta, rng):
        def _loss(p):
            return loss_fn(graphdef, p, batch, beta, rng)

        (_, out), grads = jax.value_and_grad(_loss, has_aux=True)(params)
        grads = jax.lax.pmean(grads, axis_name="devices")
        out = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name="devices"), out)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_ema = jax.tree_util.tree_map(
            lambda e, p: ema_decay * e + (1.0 - ema_decay) * p,
            ema_params,
            new_params,
        )
        return new_params, new_opt_state, new_ema, out

    return jax.pmap(_step, axis_name="devices", in_axes=(None, None, None, 0, None, 0))


def _tree_allclose(a, b, atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    if jax.tree_util.tree_structure(a) != jax.tree_util.tree_structure(b):
        return False
    a_leaves = jax.tree_util.tree_leaves(a)
    b_leaves = jax.tree_util.tree_leaves(b)
    if len(a_leaves) != len(b_leaves):
        return False
    for a_leaf, b_leaf in zip(a_leaves, b_leaves):
        if isinstance(a_leaf, (np.ndarray, jax.Array)) or isinstance(b_leaf, (np.ndarray, jax.Array)):
            if not np.allclose(np.asarray(a_leaf), np.asarray(b_leaf), atol=atol, rtol=rtol):
                return False
            continue
        if type(a_leaf) is not type(b_leaf):
            return False
        if a_leaf == b_leaf:
            continue
        if repr(a_leaf) != repr(b_leaf):
            return False
    return True


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
    steps_per_epoch = max(1, len(datasets["train"]) // args.bs)
    total_train_steps = steps_per_epoch * max(1, args.epochs - state.epoch)
    use_tpu_pmap = args.accelerator == "tpu" and jax.local_device_count() > 1
    device_count = jax.local_device_count() if use_tpu_pmap else 1
    if use_tpu_pmap and args.bs % device_count != 0:
        raise ValueError(
            f"Global batch size {args.bs} must be divisible by TPU local device count {device_count}."
        )
    if use_tpu_pmap:
        logger.info(
            "tpu_multi_core_training=enabled local_device_count=%d global_batch_size=%d per_device_batch_size=%d",
            device_count,
            args.bs,
            args.bs // device_count,
        )
        train_step_fn = make_pmap_train_step(graphdef, tx, state.ema.decay)
    else:
        train_step_fn = make_train_step(graphdef, tx, state.ema.decay)
    eval_step_fn = make_eval_step(graphdef)
    beta_warmup = linear_warmup(args.beta_warmup_steps) if getattr(args, "beta_warmup_steps", 0) > 0 else None

    def _iter_eval_batches(dataset):
        for start in range(0, len(dataset), args.bs):
            batch_idx = np.arange(start, min(start + args.bs, len(dataset)))
            if hasattr(dataset, "make_batch"):
                yield dataset.make_batch(batch_idx, shuffle=False)
            else:
                batch = [dataset[int(i)] for i in batch_idx]
                keys = batch[0].keys()
                out = {}
                for k in keys:
                    values = [np.asarray(item[k]) for item in batch]
                    out[k] = np.stack(values, axis=0)
                yield out

    def _eval_dataset(dataset):
        stats_sum = {"elbo": 0.0, "nll": 0.0, "kl": 0.0}
        sample_count = 0
        rng_key = jax.random.PRNGKey(args.seed)
        beta_scale = float(beta_warmup(state.step)) if beta_warmup is not None else 1.0
        beta = args.beta * beta_scale
        for batch_i, batch in enumerate(_iter_eval_batches(dataset)):
            batch = preprocess_batch(args, batch, expand_pa=True)
            batch_key = jax.random.fold_in(rng_key, batch_i)
            out = eval_step_fn(state.ema.params, batch, beta, batch_key)
            _block_until_ready(out)
            bs = int(batch["x"].shape[0])
            sample_count += bs
            for key in stats_sum:
                stats_sum[key] += float(out[key]) * bs
        if sample_count == 0:
            return {k: 0.0 for k in stats_sum}
        return {k: v / sample_count for k, v in stats_sum.items()}

    for epoch in range(state.epoch, args.epochs):
        t0 = time.perf_counter()
        epoch_step_t0 = time.perf_counter()
        train_stats_sum = {"elbo": 0.0, "nll": 0.0, "kl": 0.0}

        for step in range(steps_per_epoch):
            fetch_t0 = time.perf_counter()
            batch = preprocess_batch(args, next(train_iter), expand_pa=True)
            batch_ready_t0 = time.perf_counter()
            rng = jax.random.PRNGKey(args.seed + state.step + step + epoch * 1000)
            beta_scale = float(beta_warmup(state.step)) if beta_warmup is not None else 1.0
            beta = args.beta * beta_scale
            if use_tpu_pmap:
                batch = _shard_batch(batch, device_count)
                rng = jax.random.split(rng, device_count)
            new_params, new_opt_state, new_ema_params, out = train_step_fn(
                state.params,
                state.opt_state,
                state.ema.params,
                batch,
                beta,
                rng,
            )
            if use_tpu_pmap:
                new_params = _unreplicate(_block_until_ready(new_params))
                new_opt_state = _unreplicate(_block_until_ready(new_opt_state))
                new_ema_params = _unreplicate(_block_until_ready(new_ema_params))
                out = _unreplicate(_block_until_ready(out))
            else:
                _block_until_ready(out)
            state = TrainState(
                params=new_params,
                opt_state=new_opt_state,
                ema=EMA(params=new_ema_params, decay=state.ema.decay),
                step=state.step + 1,
                epoch=state.epoch,
                best_loss=state.best_loss,
            )
            step_end_t0 = time.perf_counter()
            for key in train_stats_sum:
                train_stats_sum[key] += float(out[key])

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
                    "epoch=%d step=%d/%d global_step=%d nelbo=%.4f nll=%.4f kl=%.4f data_time=%.2fs compute_time=%.2fs step_time=%.2fs iter/s=%.3f sample/s=%.3f epoch_iter/s=%.3f epoch_sample/s=%.3f eta=%.1fs",
                    epoch + 1,
                    step + 1,
                    steps_per_epoch,
                    state.step,
                    float(out["elbo"]),
                    float(out["nll"]),
                    float(out["kl"]),
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

            if getattr(args, "checkpoint_smoke_test", False):
                if args.viz_freq:
                    viz_path = write_images(
                        args,
                        graphdef,
                        state.ema.params,
                        batch,
                        jax.random.PRNGKey(args.seed + state.step),
                        step=state.step,
                    )
                    logger.info("viz_image=%s", viz_path)
                if state.step >= max(1, args.checkpoint_smoke_steps):
                    checkpoint_smoke_test(args, state, tx, logger)
                    return

        epoch_time = time.perf_counter() - t0
        epoch_iter_per_sec = steps_per_epoch / max(epoch_time, 1e-12)
        epoch_sample_per_sec = steps_per_epoch * args.bs / max(epoch_time, 1e-12)
        train_stats = {k: v / max(1, steps_per_epoch) for k, v in train_stats_sum.items()}
        if hasattr(writer, "add_scalar"):
            writer.add_scalar("nelbo/train", train_stats["elbo"], epoch + 1)
            writer.add_scalar("nll/train", train_stats["nll"], epoch + 1)
            writer.add_scalar("kl/train", train_stats["kl"], epoch + 1)
            writer.add_scalar("train/elbo", train_stats["elbo"], epoch + 1)
            writer.add_scalar("train/nll", train_stats["nll"], epoch + 1)
            writer.add_scalar("train/kl", train_stats["kl"], epoch + 1)
        logger.info(
            "=> train | nelbo: %.4f - nll: %.4f - kl: %.4f - steps: %d - it/s: %.2f - samples/s: %.1f",
            train_stats["elbo"],
            train_stats["nll"],
            train_stats["kl"],
            state.step,
            epoch_iter_per_sec,
            epoch_sample_per_sec,
        )
        if epoch % max(1, args.eval_freq) == 0:
            valid_batch = preprocess_batch(args, next(_iter_eval_batches(datasets["valid"])), expand_pa=True)
            valid_out = _eval_dataset(datasets["valid"])
            if float(valid_out["elbo"]) < state.best_loss:
                state.best_loss = float(valid_out["elbo"])
                save_state(args, state, tx, epoch + 1)
            if hasattr(writer, "add_scalar"):
                writer.add_scalar("nelbo/valid", float(valid_out["elbo"]), epoch + 1)
                writer.add_scalar("nll/valid", float(valid_out["nll"]), epoch + 1)
                writer.add_scalar("kl/valid", float(valid_out["kl"]), epoch + 1)
                writer.add_scalar("valid/elbo", float(valid_out["elbo"]), epoch + 1)
                writer.add_scalar("valid/nll", float(valid_out["nll"]), epoch + 1)
                writer.add_scalar("valid/kl", float(valid_out["kl"]), epoch + 1)
            if args.viz_freq and not getattr(args, "checkpoint_smoke_test", False) and (epoch + 1) % args.viz_freq == 0:
                viz_path = write_images(
                    args,
                    graphdef,
                    state.ema.params,
                    valid_batch,
                    jax.random.PRNGKey(args.seed + epoch),
                    step=state.step,
                )
                logger.info("viz_image=%s", viz_path)
            logger.info(
                "=> valid | nelbo: %.4f - nll: %.4f - kl: %.4f - steps: %d",
                float(valid_out["elbo"]),
                float(valid_out["nll"]),
                float(valid_out["kl"]),
                state.step,
            )
            logger.info(
                "epoch=%d epoch_time=%.1fs epoch_iter/s=%.3f epoch_sample/s=%.3f",
                epoch + 1,
                epoch_time,
                epoch_iter_per_sec,
                epoch_sample_per_sec,
            )
        else:
            logger.info(
                "epoch=%d epoch_time=%.1fs epoch_iter/s=%.3f epoch_sample/s=%.3f",
                epoch + 1,
                epoch_time,
                epoch_iter_per_sec,
                epoch_sample_per_sec,
            )
        if getattr(args, "checkpoint_smoke_test", False) and state.step >= max(1, args.checkpoint_smoke_steps):
            checkpoint_smoke_test(args, state, tx, logger)
            return
