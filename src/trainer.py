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

from utils import BackgroundArtifactWriter, EMA, append_text_file, batch_iterator, ensure_dir, linear_warmup, load_checkpoint, materialize_nnx, save_checkpoint, write_images


def preprocess_batch(args, batch, expand_pa: bool = False, compact_pa: bool = False):
    x = np.asarray(batch["x"], dtype=np.float32)
    if x.max() > 1.5:
        x = (x - 127.5) / 127.5
    pa = np.asarray(batch["pa"], dtype=np.float32)
    if not compact_pa and (expand_pa or pa.ndim == 2):
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


def _replicate(tree, devices):
    mesh = jax.sharding.Mesh(np.asarray(devices), ("devices",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("devices"))

    def _put(x):
        copies = np.stack([np.asarray(x)] * len(devices), axis=0)
        return jax.device_put(copies, sharding)

    return jax.tree_util.tree_map(_put, tree)


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
    replicated: bool = False


def _portable_state(state: TrainState) -> TrainState:
    if not state.replicated:
        return state
    return TrainState(
        params=_unreplicate(state.params),
        opt_state=_unreplicate(state.opt_state),
        ema=EMA(params=_unreplicate(state.ema.params), decay=state.ema.decay),
        step=state.step,
        epoch=state.epoch,
        best_loss=state.best_loss,
    )


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


def _select_tree(predicate, true_tree, false_tree):
    return jax.tree_util.tree_map(
        lambda new, old: jnp.where(predicate, new, old), true_tree, false_tree
    )


def _grads_are_finite(grads):
    leaves = jax.tree_util.tree_leaves(grads)
    checks = [jnp.all(jnp.isfinite(x)) for x in leaves]
    return jnp.all(jnp.stack(checks)) if checks else jnp.array(True)


def make_train_step(graphdef, tx, ema_decay: float):
    def _step(params, opt_state, ema_params, batch, beta, rng):
        def _loss(p):
            return loss_fn(graphdef, p, batch, beta, rng)

        (loss, out), grads = jax.value_and_grad(_loss, has_aux=True)(params)
        grad_norm = optax.global_norm(grads)
        finite = jnp.logical_and(jnp.isfinite(loss), _grads_are_finite(grads))
        updates, candidate_opt_state = tx.update(grads, opt_state, params)
        candidate_params = optax.apply_updates(params, updates)
        finite = jnp.logical_and(finite, _grads_are_finite(candidate_params))
        new_params = _select_tree(finite, candidate_params, params)
        new_opt_state = _select_tree(finite, candidate_opt_state, opt_state)
        candidate_ema = jax.tree_util.tree_map(lambda e, p: ema_decay * e + (1.0 - ema_decay) * p, ema_params, new_params)
        new_ema = _select_tree(finite, candidate_ema, ema_params)
        out = {**out, "grad_norm": grad_norm, "update_skipped": jnp.logical_not(finite).astype(jnp.float32)}
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

        (loss, out), grads = jax.value_and_grad(_loss, has_aux=True)(params)
        grad_norm = optax.global_norm(grads)
        local_finite = jnp.logical_and(jnp.isfinite(loss), _grads_are_finite(grads))
        finite = jax.lax.pmin(local_finite.astype(jnp.int32), axis_name="devices").astype(jnp.bool_)
        grads = jax.lax.pmean(grads, axis_name="devices")
        out = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name="devices"), out)
        updates, candidate_opt_state = tx.update(grads, opt_state, params)
        candidate_params = optax.apply_updates(params, updates)
        candidate_finite = _grads_are_finite(candidate_params)
        finite = jax.lax.pmin(
            jnp.logical_and(finite, candidate_finite).astype(jnp.int32),
            axis_name="devices",
        ).astype(jnp.bool_)
        new_params = _select_tree(finite, candidate_params, params)
        new_opt_state = _select_tree(finite, candidate_opt_state, opt_state)
        candidate_ema = jax.tree_util.tree_map(
            lambda e, p: ema_decay * e + (1.0 - ema_decay) * p,
            ema_params,
            new_params,
        )
        new_ema = _select_tree(finite, candidate_ema, ema_params)
        out = {
            **out,
            "grad_norm": jax.lax.pmean(grad_norm, axis_name="devices"),
            "update_skipped": jnp.logical_not(finite).astype(jnp.float32),
        }
        return new_params, new_opt_state, new_ema, out

    # State is replicated once outside the loop and remains resident on device.
    return jax.pmap(_step, axis_name="devices", in_axes=(0, 0, 0, 0, None, 0))


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
    state = _portable_state(state)
    restore_template = {
        "epoch": state.epoch,
        "step": state.step,
        "best_loss": state.best_loss,
        "params": state.params,
        "ema_params": state.ema.params,
        "opt_state": state.opt_state,
    }
    save_state(args, state, tx, state.epoch, wait=True)
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


def save_state(
    args,
    state: TrainState,
    tx,
    epoch,
    *,
    artifact_writer: BackgroundArtifactWriter | None = None,
    wait: bool = False,
):
    state = _portable_state(state)
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
    metadata = {"epoch": epoch, "best_loss": float(state.best_loss)}
    if artifact_writer is not None and not wait:
        remote_run_dir = getattr(args, "remote_save_dir", "")
        return artifact_writer.submit_checkpoint(
            ckpt,
            path,
            step=state.step,
            custom_metadata=metadata,
            local_tree_dir=args.checkpoint_dir if remote_run_dir else None,
            remote_tree_dir=os.path.join(remote_run_dir, "checkpoints") if remote_run_dir else None,
        )
    save_checkpoint(ckpt, path, step=state.step, custom_metadata=metadata)
    remote_run_dir = getattr(args, "remote_save_dir", "")
    if remote_run_dir:
        from utils import sync_tree

        sync_tree(args.checkpoint_dir, os.path.join(remote_run_dir, "checkpoints"))
    return path


def trainer(args, graphdef, state: TrainState, tx, datasets, writer, logger):
    ensure_dir(args.save_dir)
    requested_mode = getattr(args, "execution_mode", "auto")
    multi_tpu_available = args.accelerator == "tpu" and jax.local_device_count() > 1
    if requested_mode == "replicated" and not multi_tpu_available:
        raise ValueError("execution_mode=replicated requires accelerator=tpu with multiple local devices")
    use_tpu_pmap = multi_tpu_available and requested_mode != "single_device"
    drop_remainder = bool(getattr(args, "drop_remainder", False) or use_tpu_pmap)
    train_iter = batch_iterator(
        datasets["train"], args.bs, True, args.seed, drop_remainder=drop_remainder
    )
    steps_per_epoch = max(1, len(datasets["train"]) // args.bs)
    total_train_steps = steps_per_epoch * max(1, args.epochs - state.epoch)
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
        devices = jax.local_devices()[:device_count]
        state = TrainState(
            params=_replicate(state.params, devices),
            opt_state=_replicate(state.opt_state, devices),
            ema=EMA(params=_replicate(state.ema.params, devices), decay=state.ema.decay),
            step=state.step,
            epoch=state.epoch,
            best_loss=state.best_loss,
            replicated=True,
        )
    else:
        logger.info("execution_mode=single_device")
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
        eval_state = _portable_state(state)
        beta_scale = float(beta_warmup(eval_state.step)) if beta_warmup is not None else 1.0
        beta = args.beta * beta_scale
        for batch_i, batch in enumerate(_iter_eval_batches(dataset)):
            batch = preprocess_batch(args, batch, compact_pa=True)
            batch_key = jax.random.fold_in(rng_key, batch_i)
            out = eval_step_fn(eval_state.ema.params, batch, beta, batch_key)
            _block_until_ready(out)
            bs = int(batch["x"].shape[0])
            sample_count += bs
            for key in stats_sum:
                stats_sum[key] += float(out[key]) * bs
        if sample_count == 0:
            return {k: 0.0 for k in stats_sum}
        return {k: v / sample_count for k, v in stats_sum.items()}

    artifact_writer = BackgroundArtifactWriter()
    benchmark_steps = int(getattr(args, "benchmark_steps", 0))
    benchmark_warmup = int(getattr(args, "benchmark_warmup_steps", 20))
    benchmark_t0 = None
    run_steps = 0

    try:
        for epoch in range(state.epoch, args.epochs):
            t0 = time.perf_counter()
            epoch_step_t0 = time.perf_counter()
            train_stats_sum = {"elbo": 0.0, "nll": 0.0, "kl": 0.0}
            speed_window_t0 = epoch_step_t0
            speed_window_step = 0

            for step in range(steps_per_epoch):
                fetch_t0 = time.perf_counter()
                batch = preprocess_batch(args, next(train_iter), compact_pa=True)
                viz_batch = batch
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
                    metric_out = jax.tree_util.tree_map(lambda x: x[0], out)
                else:
                    _block_until_ready(out)
                    metric_out = out
                state = TrainState(
                    params=new_params,
                    opt_state=new_opt_state,
                    ema=EMA(params=new_ema_params, decay=state.ema.decay),
                    step=state.step + 1,
                    epoch=state.epoch,
                    best_loss=state.best_loss,
                    replicated=use_tpu_pmap,
                )
                run_steps += 1
                step_end_t0 = time.perf_counter()
                for key in train_stats_sum:
                    train_stats_sum[key] = train_stats_sum[key] + metric_out[key]

                if (step + 1) % max(1, args.speed_log_freq) == 0:
                    _block_until_ready(metric_out)
                    sync_t0 = time.perf_counter()
                    data_dt = batch_ready_t0 - fetch_t0
                    window_steps = (step + 1) - speed_window_step
                    step_dt = (sync_t0 - speed_window_t0) / max(1, window_steps)
                    compute_dt = max(0.0, step_dt - data_dt)
                    iter_per_sec = 1.0 / max(step_dt, 1e-12)
                    sample_per_sec = args.bs / max(step_dt, 1e-12)
                    epoch_elapsed = sync_t0 - epoch_step_t0
                    epoch_iters_per_sec = (step + 1) / max(epoch_elapsed, 1e-12)
                    epoch_samples_per_sec = (step + 1) * args.bs / max(epoch_elapsed, 1e-12)
                    train_steps_done = (epoch - state.epoch) * steps_per_epoch + (step + 1)
                    train_steps_left = max(0, total_train_steps - train_steps_done)
                    eta_sec = train_steps_left / max(epoch_iters_per_sec, 1e-12)
                    logger.info(
                        "epoch=%d step=%d/%d global_step=%d nelbo=%.4f nll=%.4f kl=%.4f grad_norm=%.4f update_skipped=%d data_time=%.2fs compute_time=%.2fs step_time=%.2fs iter/s=%.3f sample/s=%.3f epoch_iter/s=%.3f epoch_sample/s=%.3f eta=%.1fs",
                        epoch + 1,
                        step + 1,
                        steps_per_epoch,
                        state.step,
                        float(metric_out["elbo"]),
                        float(metric_out["nll"]),
                        float(metric_out["kl"]),
                        float(metric_out["grad_norm"]),
                        int(float(metric_out["update_skipped"])),
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
                        writer.add_scalar("train/grad_norm", float(metric_out["grad_norm"]), state.step)
                        writer.add_scalar("train/update_skipped", float(metric_out["update_skipped"]), state.step)
                    speed_window_t0 = sync_t0
                    speed_window_step = step + 1

                if getattr(args, "checkpoint_smoke_test", False):
                    viz_path = write_images(
                        args,
                        graphdef,
                        _portable_state(state).ema.params,
                        viz_batch,
                        jax.random.PRNGKey(args.seed + state.step),
                        step=state.step,
                    )
                    logger.info("viz_image=%s", viz_path)
                    if state.step >= max(1, args.checkpoint_smoke_steps):
                        artifact_writer.flush()
                        checkpoint_smoke_test(args, state, tx, logger)
                        return

                if benchmark_steps and run_steps == benchmark_warmup:
                    _block_until_ready(metric_out)
                    benchmark_t0 = time.perf_counter()
                    logger.info("benchmark_warmup_complete steps=%d", benchmark_warmup)
                if benchmark_steps and run_steps >= benchmark_warmup + benchmark_steps:
                    _block_until_ready(metric_out)
                    elapsed = time.perf_counter() - (benchmark_t0 or epoch_step_t0)
                    logger.info(
                        "benchmark_complete measured_steps=%d warmup_steps=%d elapsed=%.3fs sample/s=%.3f",
                        benchmark_steps,
                        benchmark_warmup,
                        elapsed,
                        benchmark_steps * args.bs / max(elapsed, 1e-12),
                    )
                    return

            train_stats_sum = {k: float(_block_until_ready(v)) for k, v in train_stats_sum.items()}
            train_time = time.perf_counter() - epoch_step_t0

            checkpoint_due = (epoch + 1) % max(1, args.checkpoint_freq) == 0
            eval_due = (epoch + 1) % max(1, args.eval_freq) == 0
            validation_due = checkpoint_due or eval_due
            valid_nelbo = valid_nll = valid_kl = float("nan")
            valid_viz_batch = None
            if validation_due:
                valid_stats = _eval_dataset(datasets["valid"])
                valid_nelbo = valid_stats["elbo"]
                valid_nll = valid_stats["nll"]
                valid_kl = valid_stats["kl"]
                valid_viz_batch = next(_iter_eval_batches(datasets["valid"]))
            if checkpoint_due and valid_nelbo < state.best_loss:
                state.best_loss = valid_nelbo
                save_state(args, state, tx, epoch + 1, artifact_writer=artifact_writer)
            if hasattr(writer, "add_scalar"):
                writer.add_scalar("nelbo/train", train_stats_sum["elbo"] / max(1, steps_per_epoch), epoch + 1)
                writer.add_scalar("nll/train", train_stats_sum["nll"] / max(1, steps_per_epoch), epoch + 1)
                writer.add_scalar("kl/train", train_stats_sum["kl"] / max(1, steps_per_epoch), epoch + 1)
                writer.add_scalar("train/elbo", train_stats_sum["elbo"] / max(1, steps_per_epoch), epoch + 1)
                writer.add_scalar("train/nll", train_stats_sum["nll"] / max(1, steps_per_epoch), epoch + 1)
                writer.add_scalar("train/kl", train_stats_sum["kl"] / max(1, steps_per_epoch), epoch + 1)
                if validation_due:
                    writer.add_scalar("nelbo/valid", valid_nelbo, epoch + 1)
                    writer.add_scalar("nll/valid", valid_nll, epoch + 1)
                    writer.add_scalar("kl/valid", valid_kl, epoch + 1)
                    writer.add_scalar("valid/elbo", valid_nelbo, epoch + 1)
                    writer.add_scalar("valid/nll", valid_nll, epoch + 1)
                    writer.add_scalar("valid/kl", valid_kl, epoch + 1)
            if not getattr(args, "checkpoint_smoke_test", False) and eval_due:
                viz_path = write_images(
                    args,
                    graphdef,
                    _portable_state(state).ema.params,
                    valid_viz_batch,
                    jax.random.PRNGKey(args.seed + epoch),
                    step=state.step,
                )
                logger.info("viz_image=%s", viz_path, extra={"eval_log": True})
            epoch_time = time.perf_counter() - t0
            epoch_iter_per_sec = steps_per_epoch / max(train_time, 1e-12)
            epoch_sample_per_sec = steps_per_epoch * args.bs / max(train_time, 1e-12)
            logger.info(
                "=> train | nelbo: %.4f - nll: %.4f - kl: %.4f - steps: %d - it/s: %.2f - samples/s: %.1f",
                train_stats_sum["elbo"] / max(1, steps_per_epoch),
                train_stats_sum["nll"] / max(1, steps_per_epoch),
                train_stats_sum["kl"] / max(1, steps_per_epoch),
                state.step,
                epoch_iter_per_sec,
                epoch_sample_per_sec,
                extra={"eval_log": eval_due},
            )
            if eval_due:
                logger.info(
                    "=> valid | nelbo: %.4f - nll: %.4f - kl: %.4f - steps: %d",
                    valid_nelbo,
                    valid_nll,
                    valid_kl,
                    state.step,
                    extra={"eval_log": True},
                )
                logger.info(
                    "epoch=%d train_time=%.1fs total_time=%.1fs epoch_iter/s=%.3f epoch_sample/s=%.3f",
                    epoch + 1,
                    train_time,
                    epoch_time,
                    epoch_iter_per_sec,
                    epoch_sample_per_sec,
                    extra={"eval_log": True},
                )
                trainlog_lines = [
                    f"epoch={epoch + 1} train nelbo={train_stats_sum['elbo'] / max(1, steps_per_epoch):.4f} "
                    f"nll={train_stats_sum['nll'] / max(1, steps_per_epoch):.4f} "
                    f"kl={train_stats_sum['kl'] / max(1, steps_per_epoch):.4f} "
                    f"steps={state.step} it_s={epoch_iter_per_sec:.3f} sample_s={epoch_sample_per_sec:.3f}",
                    f"epoch={epoch + 1} valid nelbo={valid_nelbo:.4f} nll={valid_nll:.4f} kl={valid_kl:.4f} steps={state.step}",
                    f"epoch={epoch + 1} train_time={train_time:.1f}s total_time={epoch_time:.1f}s epoch_iter_s={epoch_iter_per_sec:.3f} epoch_sample_s={epoch_sample_per_sec:.3f}",
                ]
                trainlog_lines.insert(0, f"viz_image={viz_path}")
                remote_trainlog = os.path.join(args.remote_save_dir, "trainlog.txt") if getattr(args, "remote_save_dir", "") else None
                append_text_file(
                    os.path.join(args.save_dir, "trainlog.txt"),
                    "\n".join(trainlog_lines) + "\n",
                    remote_path=remote_trainlog,
                )
            else:
                logger.info(
                    "epoch=%d train_time=%.1fs total_time=%.1fs epoch_iter/s=%.3f epoch_sample/s=%.3f",
                    epoch + 1,
                    train_time,
                    epoch_time,
                    epoch_iter_per_sec,
                    epoch_sample_per_sec,
                )
            if getattr(args, "checkpoint_smoke_test", False) and state.step >= max(1, args.checkpoint_smoke_steps):
                checkpoint_smoke_test(args, state, tx, logger)
                return
    finally:
        artifact_writer.close()
