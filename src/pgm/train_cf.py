from __future__ import annotations

# ruff: noqa: E402 -- backend selection must happen before importing JAX.

import argparse
import copy
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from runtime import configure_backend_from_argv

configure_backend_from_argv()

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from tqdm import tqdm

from datasets import morphomnist
from hps import add_arguments, setup_hparams
from models import HVAE
from pgm.cf_parity import (
    clip_counterfactual_grads,
    damped_lagrangian_loss,
    inherit_vae_training_config,
)
from pgm.flow_pgm import MorphoMNISTPGM
from pgm.sup_aux_pgm import MorphoMNISTSupAuxPredictor
from trainer import init_state, preprocess_batch
from utils import (
    EMA,
    SummaryWriter,
    checkpoint_root_dir,
    ensure_dir,
    experiment_run_dir,
    load_checkpoint,
    open_file,
    save_checkpoint,
    seed_all,
    sync_file,
    sync_tree,
    tree_copy,
)


def setup_logging(args):
    ensure_dir(args.save_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.save_dir, "trainlog.txt"), mode="a"),
        ],
        force=True,
    )
    return logging.getLogger("causal-genx-cf")


def _validate_runtime_device(args) -> None:
    devices = jax.devices()
    if args.accelerator == "gpu":
        gpu_devices = [device for device in devices if device.platform in {"gpu", "cuda"}]
        if not gpu_devices:
            raise RuntimeError(
                "--accelerator gpu requested, but JAX found no CUDA GPU. Install a CUDA-enabled "
                "JAX build compatible with the NVIDIA driver and CUDA runtime."
            )
        if len(gpu_devices) != 1:
            raise RuntimeError(
                f"Counterfactual finetuning requires one visible GPU, found {len(gpu_devices)}. "
                "Set --gpu_id or CUDA_VISIBLE_DEVICES to a single device."
            )
        device = gpu_devices[0]
    elif args.accelerator == "cpu":
        cpu_devices = [device for device in devices if device.platform == "cpu"]
        if not cpu_devices:
            raise RuntimeError("--accelerator cpu requested, but JAX found no CPU device")
        device = cpu_devices[0]
    else:
        matching_devices = [device for device in devices if device.platform == args.accelerator]
        if not matching_devices:
            raise RuntimeError(
                f"--accelerator {args.accelerator} requested, but JAX devices are {devices}"
            )
        device = matching_devices[0]
    print(f"JAX device preflight passed: platform={device.platform} device={device}")


def _configure_compute_policy(args) -> None:
    if args.accelerator == "cpu" and args.precision != "fp32":
        raise ValueError("CPU counterfactual finetuning requires --precision fp32")
    if args.accelerator in {"gpu", "tpu"} and args.precision == "bf16":
        matmul_precision = "default"
    else:
        matmul_precision = "highest"
    jax.config.update("jax_default_matmul_precision", matmul_precision)
    print(
        "JAX compute policy: "
        f"precision={args.precision} master_params=fp32 matmul_precision={matmul_precision}"
    )


def _vae_compute_params(args, params):
    if args.precision != "bf16" or args.accelerator not in {"gpu", "tpu"}:
        return params
    return jax.tree_util.tree_map(
        lambda value: value.astype(jnp.bfloat16)
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating)
        else value,
        params,
    )


def loginfo(title: str, logger: Any, stats: Dict[str, Any]) -> None:
    logger.info(f"{title} | " + " - ".join(f"{k}: {v:.4f}" for k, v in stats.items()))


@jax.tree_util.register_pytree_node_class
@dataclass
class Bundle:
    graphdef: Any
    params: Any
    batch_stats: Any = None

    def materialize(self):
        states = [nnx.State(self.params)]
        if self.batch_stats is not None:
            states.append(nnx.State(self.batch_stats))
        return nnx.merge(self.graphdef, *states)

    def tree_flatten(self):
        children = (self.params, self.batch_stats)
        return children, self.graphdef

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        params, batch_stats = children
        return cls(aux_data, params, batch_stats)


def _restore_args(args, checkpoint):
    saved = checkpoint.get("hparams", {})
    preserved = {
        "accelerator": args.accelerator,
        "gpu_id": args.gpu_id,
        "precision": args.precision,
        "data_dir": args.data_dir,
        "load_path": args.load_path,
        "testing": args.testing,
        "remote_ckpt_dir": args.remote_ckpt_dir,
        "pgm_path": args.pgm_path,
        "predictor_path": args.predictor_path,
        "vae_path": args.vae_path,
        "trust_incomplete_checkpoint": args.trust_incomplete_checkpoint,
        "model_validation_batches": args.model_validation_batches,
    }
    for key, value in saved.items():
        if hasattr(args, key):
            setattr(args, key, value)
    for key, value in preserved.items():
        setattr(args, key, value)


def _assert_tree_compatible(name: str, checkpoint: Dict[str, Any], tree: Any, key: str) -> None:
    if jax.tree_util.tree_structure(checkpoint[key]) != jax.tree_util.tree_structure(tree):
        raise ValueError(f"{name} checkpoint parameter structure does not match the current model")


def _load_runtime_checkpoint(args, path: str, template: Optional[Dict[str, Any]] = None):
    fallback_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    return load_checkpoint(
        path,
        template=template,
        fallback_sharding=fallback_sharding,
        allow_incomplete=args.trust_incomplete_checkpoint,
    )


def _checkpoint_root(path: str) -> str:
    path = path.rstrip("/")
    if path.split("/")[-1].isdigit():
        return path.rsplit("/", 1)[0]
    return path


def _load_vae_hparams(args) -> Dict[str, Any]:
    with open_file(f"{_checkpoint_root(args.vae_path)}/hparams.json", "r") as f:
        return json.load(f)


def _expand_parents(pa: jax.Array, input_res: int) -> jax.Array:
    if pa.ndim == 4:
        return pa
    return pa[:, None, None, :].repeat(input_res, axis=1).repeat(input_res, axis=2)


def _choose_intervention(args, dag_vars: List[str]) -> str:
    return copy.deepcopy(args.do_pa) if args.do_pa else random.choice(dag_vars)


def _batch_parent(args, batch: Dict[str, jax.Array], name: str) -> jax.Array:
    if name in batch:
        value = batch[name]
        return value[:, None] if value.ndim == 1 else value

    pa = batch["pa"]
    offset = 0
    for parent_name in args.parents_x:
        width = 10 if parent_name == "digit" else 1
        if parent_name == name:
            return pa[:, offset : offset + width]
        offset += width
    raise KeyError(f"Parent {name!r} is not present in batch or --parents_x={args.parents_x}")


def _make_intervention(
    args,
    batch: Dict[str, jax.Array],
    do_k: str,
    train_samples: Dict[str, np.ndarray],
    *,
    train: bool,
) -> Dict[str, jax.Array]:
    if train:
        parent = _batch_parent(args, batch, do_k)
        permutation = np.random.permutation(parent.shape[0])
        value = parent[permutation]
        return {do_k: value}

    idx = np.random.permutation(train_samples[do_k].shape[0])
    value = np.asarray(train_samples[do_k][idx][: batch["x"].shape[0]], dtype=np.float32)
    if value.ndim == 1:
        value = value[:, None]
    return {do_k: jnp.asarray(value)}


def _load_vae_bundle(args):
    vae_hparams = _load_vae_hparams(args)
    inherit_vae_training_config(args, vae_hparams)
    model_args = {
        key: vae_hparams.get(key, getattr(args, key))
        for key in (
            "input_channels",
            "input_res",
            "enc_arch",
            "dec_arch",
            "widths",
            "z_dim",
            "context_dim",
            "z_max_res",
            "bottleneck",
            "cond_prior",
            "q_correction",
            "bias_max_res",
            "x_like",
            "kl_free_bits",
            "std_init",
            "hps",
        )
    }
    for key, value in model_args.items():
        setattr(args, key, value)

    rngs = nnx.Rngs(args.seed)
    vae = HVAE(**model_args, rngs=rngs)
    graphdef, _ = nnx.split(vae, nnx.Param)
    params = nnx.state(vae, nnx.Param).to_pure_dict()
    _, tx = init_state(vae, args, None, jax.random.PRNGKey(args.seed))
    template = {
        "epoch": 0,
        "step": 0,
        "best_loss": float("inf"),
        "params": params,
        "ema_params": params,
        "opt_state": tx.init(params),
    }
    checkpoint = _load_runtime_checkpoint(args, args.vae_path, template=template)
    params = checkpoint.get("ema_params", checkpoint.get("params"))
    if params is None:
        raise ValueError(f"VAE checkpoint at {args.vae_path} is missing params")
    _assert_tree_compatible("VAE", checkpoint, nnx.state(vae, nnx.Param).to_pure_dict(), "params")
    return checkpoint, Bundle(graphdef, params)


def _load_pgm_bundle(args):
    pgm_ckpt = _load_runtime_checkpoint(args, args.pgm_path)
    if pgm_ckpt.get("format_version") != 2 or "ema_params" not in pgm_ckpt:
        raise ValueError(
            "The PGM checkpoint uses the old simplified Gaussian/CNN format. "
            "Retrain it with pgm/train_pgm.py before running counterfactual finetuning."
        )
    rngs = nnx.Rngs(args.seed)
    pgm_hparams = pgm_ckpt.get("hparams", {})
    pgm = MorphoMNISTPGM(widths=pgm_hparams.get("widths", [32, 32]), rngs=rngs)
    _assert_tree_compatible("PGM", pgm_ckpt, nnx.state(pgm, nnx.Param).to_pure_dict(), "ema_params")
    graphdef, _ = nnx.split(pgm, nnx.Param)
    return pgm_ckpt, Bundle(graphdef, pgm_ckpt["ema_params"])


def _load_predictor_bundle(args):
    predictor_ckpt = _load_runtime_checkpoint(args, args.predictor_path)
    if predictor_ckpt.get("format_version") != 3 or "ema_params" not in predictor_ckpt:
        raise ValueError(
            "The predictor checkpoint uses the old simplified CNN format. "
            "Retrain it with pgm/train_pgm.py before running counterfactual finetuning."
        )
    rngs = nnx.Rngs(args.seed)
    predictor_hparams = predictor_ckpt.get("hparams", {})
    predictor = MorphoMNISTSupAuxPredictor(
        input_channels=predictor_hparams.get("input_channels", args.input_channels),
        input_res=predictor_hparams.get("input_res", args.input_res),
        width=predictor_hparams.get("width", 8),
        std_fixed=predictor_hparams.get("std_fixed", 0.0),
        rngs=rngs,
    )
    graphdef, params_state, batch_stats_state = nnx.split(
        predictor, nnx.Param, nnx.BatchStat
    )
    params = predictor_ckpt["ema_params"]
    batch_stats = predictor_ckpt["ema_batch_stats"]
    _assert_tree_compatible(
        "predictor", predictor_ckpt, params_state.to_pure_dict(), "ema_params"
    )
    _assert_tree_compatible(
        "predictor", predictor_ckpt, batch_stats_state.to_pure_dict(), "ema_batch_stats"
    )
    return predictor_ckpt, Bundle(graphdef, params, batch_stats)


def _predictor_metrics(args, dataset, preds, targets):
    stats: Dict[str, float] = {}
    for k in preds.keys():
        pred = np.asarray(preds[k])
        target = np.asarray(targets[k])
        if k == "digit":
            stats["digit_acc"] = float((target.argmax(-1) == pred.argmax(-1)).mean())
        else:
            min_val, max_val = dataset.min_max[k]
            pred = ((pred.squeeze(-1) + 1.0) / 2.0) * (max_val - min_val) + min_val
            target = ((target.squeeze(-1) + 1.0) / 2.0) * (max_val - min_val) + min_val
            stats[f"{k}_mae"] = float(np.mean(np.abs(target - pred)))
    return stats


def _cf_forward(
    args,
    vae_bundle: Bundle,
    pgm_bundle: Bundle,
    predictor_bundle: Bundle,
    batch: Dict[str, jax.Array],
    do: Dict[str, jax.Array],
    rng: jax.Array,
    *,
    beta: float,
    alpha: float,
    lmbda: jax.Array,
    cf_particles: int,
    t_abduct: float = 1.0,
    training: bool = True,
):
    vae = vae_bundle.materialize()
    pgm = pgm_bundle.materialize()
    predictor = predictor_bundle.materialize()

    vae.train(training)
    pgm.eval()
    predictor.eval()

    pa = batch["pa"]
    pa_maps = _expand_parents(pa, args.input_res)
    vae_rng, counterfactual_rng = jax.random.split(rng)
    vae_out = vae(batch["x"], pa_maps, beta=beta, rng=vae_rng, training=training)

    obs_pgm = {
        "thickness": pa[:, 0],
        "intensity": pa[:, 1],
        "digit": pa[:, 2:],
    }

    if cf_particles > 1:
        cfs = {"x": jnp.zeros_like(batch["x"]), "x2": jnp.zeros_like(batch["x"])}

    particle_keys = jax.random.split(counterfactual_rng, cf_particles)
    for i in range(cf_particles):
        pgm_rng, abduct_rng, cf_rng, rec_rng = jax.random.split(particle_keys[i], 4)
        cf_pa = pgm.counterfactual(obs=obs_pgm, intervention=do, rng=pgm_rng)
        cf_pa_maps = _expand_parents(cf_pa["pa"], args.input_res)
        latents = vae.abduct(batch["x"], pa_maps, t=t_abduct, rng=abduct_rng)
        cf_loc, cf_scale = vae.forward_latents(latents, cf_pa_maps, rng=cf_rng)
        rec_loc, rec_scale = vae.forward_latents(latents, pa_maps, rng=rec_rng)
        u = (batch["x"] - rec_loc) / jnp.clip(rec_scale, min=1e-12)
        cf_x = jnp.clip(cf_loc + cf_scale * u, min=-1, max=1)
        if cf_particles > 1:
            cfs["x"] = cfs["x"] + cf_x
            cfs["x2"] = cfs["x2"] + jax.lax.stop_gradient(cf_x**2)
        else:
            cfs = {"x": cf_x}

    if cf_particles > 1:
        var_cf_x = (cfs["x2"] - cfs["x"] ** 2 / cf_particles) / cf_particles
        cfs.pop("x2", None)
        cfs["x"] = cfs["x"] / cf_particles
    else:
        var_cf_x = None

    cfs.update(cf_pa)
    log_probs = predictor.model_anticausal(**cfs)
    aux_loss = -jnp.mean(log_probs["joint"])
    constraint = args.elbo_constraint - vae_out["elbo"]
    loss = damped_lagrangian_loss(aux_loss, lmbda, constraint, args.damping)
    out = dict(vae_out)
    out.update({"loss": loss, "aux_loss": aux_loss, "cfs": cfs, "var_cf_x": var_cf_x})
    return out


def _make_losses(args, vae_bundle, pgm_bundle, predictor_bundle):
    def loss_fn(vae_params, lmbda, batch, do, rng):
        local_vae = Bundle(vae_bundle.graphdef, _vae_compute_params(args, vae_params))
        out = _cf_forward(
            args,
            local_vae,
            pgm_bundle,
            predictor_bundle,
            batch,
            do,
            rng,
            beta=args.beta,
            alpha=args.alpha,
            lmbda=lmbda,
            cf_particles=args.cf_particles,
            training=True,
        )
        return out["loss"], out

    return loss_fn


def _make_train_step(args, vae_bundle, pgm_bundle, predictor_bundle, optimizer, lambda_optimizer):
    loss_fn = _make_losses(args, vae_bundle, pgm_bundle, predictor_bundle)

    def step(vae_params, opt_state, lmbda, lambda_opt_state, batch, do, rng):
        (loss, out), grads = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(
            vae_params, lmbda, batch, do, rng
        )
        vae_grads, lmbda_grads = grads
        clipped_vae_grads, clipped_lmbda_grads, grad_norm = clip_counterfactual_grads(
            vae_grads, lmbda_grads, args.grad_clip
        )
        finite = jnp.isfinite(loss) & jnp.isfinite(grad_norm)
        finite = finite & (grad_norm < args.grad_skip)

        def _apply_updates(values):
            params, opt_state, lmbda_value, lambda_opt_state = values
            updates, opt_state = optimizer.update(clipped_vae_grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            lambda_updates, lambda_opt_state = lambda_optimizer.update(
                jax.tree_util.tree_map(lambda x: -x, clipped_lmbda_grads),
                lambda_opt_state,
                lmbda_value,
            )
            lmbda_value = optax.apply_updates(lmbda_value, lambda_updates)
            lmbda_value = jnp.clip(lmbda_value, min=0)
            return params, opt_state, lmbda_value, lambda_opt_state

        def _skip_updates(values):
            return values

        vae_params, opt_state, lmbda, lambda_opt_state = jax.lax.cond(
            finite,
            _apply_updates,
            _skip_updates,
            operand=(vae_params, opt_state, lmbda, lambda_opt_state),
        )

        out = dict(out)
        out["grad_norm"] = grad_norm
        out["update_skipped"] = jnp.logical_not(finite).astype(jnp.float32)
        return vae_params, opt_state, lmbda, lambda_opt_state, out

    return jax.jit(step, donate_argnums=(0, 1, 2, 3))


def _make_eval_step(args, vae_bundle, pgm_bundle, predictor_bundle):
    def step(vae_params, lmbda, batch, do, rng):
        local_vae = Bundle(vae_bundle.graphdef, _vae_compute_params(args, vae_params))
        out = _cf_forward(
            args,
            local_vae,
            pgm_bundle,
            predictor_bundle,
            batch,
            do,
            rng,
            beta=args.beta,
            alpha=args.alpha,
            lmbda=lmbda,
            cf_particles=args.cf_particles,
            training=False,
        )
        return out

    return jax.jit(step)


def _make_optimizers(args):
    optimizer = optax.adamw(
        learning_rate=args.lr,
        b1=args.betas[0],
        b2=args.betas[1],
        weight_decay=args.wd,
    )
    lambda_optimizer = optax.adamw(
        learning_rate=args.lr_lagrange,
        b1=args.betas[0],
        b2=args.betas[1],
        weight_decay=0.0,
    )
    return optimizer, lambda_optimizer


def _eval_split(
    args,
    split: str,
    datasets,
    state,
    vae_bundle,
    pgm_bundle,
    predictor_bundle,
    eval_step,
    train_samples,
    rng,
):
    dag_vars = list(MorphoMNISTPGM.variables.keys())
    dataset = datasets[split]
    stats = {k: 0.0 for k in ["loss", "aux_loss", "elbo", "nll", "kl", "n"]}
    preds = {k: [] for k in ["thickness", "intensity", "digit"]}
    targets = {k: [] for k in ["thickness", "intensity", "digit"]}
    total_batches = len(dataset) // args.bs if split == "train" else (len(dataset) + args.bs - 1) // args.bs
    loader = tqdm(
        _epoch_batches(dataset, args.bs, shuffle=(split == "train"), drop_last=(split == "train"), rng=rng),
        total=total_batches,
        mininterval=0.1,
    )
    grad_norm = 0.0
    predictor = predictor_bundle.materialize()
    predictor.eval()
    for i, raw_batch in enumerate(loader):
        batch = preprocess_batch(args, raw_batch, compact_pa=True)
        do_k = _choose_intervention(args, dag_vars)
        do = _make_intervention(args, batch, do_k, train_samples, train=(split == "train"))
        out = eval_step(
            state["vae_params"],
            state["lmbda"],
            batch,
            do,
            jax.random.fold_in(jax.random.PRNGKey(args.seed), i),
        )
        bs = int(batch["x"].shape[0])
        stats["n"] += bs
        stats["loss"] += float(out["loss"]) * bs
        stats["aux_loss"] += float(out["aux_loss"]) * args.alpha * bs
        stats["elbo"] += float(out["elbo"]) * bs
        stats["nll"] += float(out["nll"]) * bs
        stats["kl"] += float(out["kl"]) * bs
        grad_norm = float(out.get("grad_norm", 0.0))
        if split != "train":
            preds_cf = predictor.predict(**out["cfs"])
            for k, v in preds_cf.items():
                preds[k].append(np.asarray(v))
            for k in targets.keys():
                t_k = do[k] if k in do else out["cfs"][k]
                targets[k].append(np.asarray(t_k))
        loader.set_description(
            f"[{split}] lmbda: {float(state['lmbda']):.3f}, "
            + ", ".join(
                f"{k}: {v / max(1, stats['n']):.3f}" for k, v in stats.items() if k != "n"
            )
            + (f", grad_norm: {grad_norm:.3f}" if split == "train" else "")
        )

    mean_stats = {k: v / stats["n"] for k, v in stats.items() if k != "n"}
    if split == "train":
        return mean_stats, None

    preds = {k: np.concatenate(v, axis=0) if len(v) > 1 else np.asarray(v[0]) for k, v in preds.items()}
    targets = {k: np.concatenate(v, axis=0) if len(v) > 1 else np.asarray(v[0]) for k, v in targets.items()}
    return mean_stats, _predictor_metrics(args, dataset, preds, targets)


def _epoch_batches(dataset, batch_size: int, *, shuffle: bool, drop_last: bool, rng: np.random.Generator):
    indices = np.arange(len(dataset), dtype=np.int64)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        if drop_last and batch_idx.size < batch_size:
            continue
        if hasattr(dataset, "make_batch"):
            yield dataset.make_batch(batch_idx, rng=rng, shuffle=shuffle)
        else:
            examples = [dataset[int(i)] for i in batch_idx]
            keys = examples[0].keys()
            batch = {}
            for k in keys:
                values = [np.asarray(item[k]) for item in examples]
                batch[k] = np.stack(values, axis=0)
            yield batch


def _model_validation_batches(args, dataset):
    rng = np.random.default_rng(args.seed)
    for index, raw_batch in enumerate(
        _epoch_batches(dataset, args.bs, shuffle=False, drop_last=False, rng=rng)
    ):
        if args.model_validation_batches > 0 and index >= args.model_validation_batches:
            break
        yield preprocess_batch(args, raw_batch, compact_pa=True)


def _validated_means(name: str, path: str, totals: Dict[str, float], count: int) -> None:
    if count == 0:
        raise ValueError(f"{name} checkpoint validation found no test samples: {path}")
    means = {key: value / count for key, value in totals.items()}
    if not all(np.isfinite(value) for value in means.values()):
        raise ValueError(f"{name} checkpoint validation produced non-finite metrics at {path}: {means}")
    metrics = " - ".join(f"{key}: {value:.4f}" for key, value in means.items())
    print(f"Validated {name} checkpoint: {path} | samples: {count} | {metrics}")


def _validate_vae_checkpoint(args, bundle: Bundle, dataset) -> None:
    model = Bundle(bundle.graphdef, _vae_compute_params(args, bundle.params)).materialize()
    model.eval()
    totals = {key: 0.0 for key in ("elbo", "nll", "kl")}
    count = 0
    for index, batch in enumerate(_model_validation_batches(args, dataset)):
        parents = _expand_parents(batch["pa"], args.input_res)
        outputs = model(
            batch["x"],
            parents,
            beta=args.beta,
            rng=jax.random.PRNGKey(args.seed + index),
            training=False,
        )
        size = int(batch["x"].shape[0])
        for key in totals:
            totals[key] += float(outputs[key]) * size
        count += size
    _validated_means("VAE", args.vae_path, totals, count)


def _validate_pgm_checkpoint(args, bundle: Bundle, dataset) -> None:
    model = bundle.materialize()
    model.eval()
    totals = {"nll": 0.0, "joint_log_prob": 0.0}
    count = 0
    for batch in _model_validation_batches(args, dataset):
        pa = batch["pa"]
        outputs = model.log_prob(pa[:, 0], pa[:, 1], pa[:, 2:])
        joint = jnp.mean(outputs["joint"])
        size = int(batch["x"].shape[0])
        totals["nll"] += float(-joint) * size
        totals["joint_log_prob"] += float(joint) * size
        count += size
    _validated_means("PGM", args.pgm_path, totals, count)


def _validate_predictor_checkpoint(args, bundle: Bundle, dataset) -> None:
    model = bundle.materialize()
    model.eval()
    totals = {"nll": 0.0, "joint_log_prob": 0.0}
    count = 0
    for batch in _model_validation_batches(args, dataset):
        pa = batch["pa"]
        outputs = model.model_anticausal(
            x=batch["x"],
            thickness=pa[:, 0:1],
            intensity=pa[:, 1:2],
            digit=pa[:, 2:],
        )
        joint = jnp.mean(outputs["joint"])
        size = int(batch["x"].shape[0])
        totals["nll"] += float(-joint) * size
        totals["joint_log_prob"] += float(joint) * size
        count += size
    _validated_means("predictor", args.predictor_path, totals, count)


def _save_cf_checkpoint(args, state, epoch: int) -> str:
    payload = {
        "epoch": epoch,
        "step": state["step"],
        "best_loss": state["best_loss"],
        "vae_params": state["vae_params"],
        "ema_params": state["ema_params"],
        "opt_state": state["opt_state"],
        "lmbda": state["lmbda"],
        "lambda_opt_state": state["lambda_opt_state"],
        "hparams": vars(args),
        "format_version": 1,
    }
    save_checkpoint(payload, args.checkpoint_dir, step=state["step"], custom_metadata={"epoch": epoch, "best_loss": float(state["best_loss"])})
    if getattr(args, "remote_save_dir", ""):
        sync_tree(args.checkpoint_dir, os.path.join(args.remote_save_dir, "checkpoints"))
    return args.checkpoint_dir


def main(args):
    _validate_runtime_device(args)
    _configure_compute_policy(args)
    seed_all(args.seed, args.deterministic)
    if args.do_pa in {"None", "none", "null", ""}:
        args.do_pa = None
    if args.dataset != "morphomnist":
        raise ValueError("JAX counterfactual finetuning currently supports --dataset morphomnist only")

    if not hasattr(args, "elbo_constraint") or args.elbo_constraint is None:
        args.elbo_constraint = 1.841216802597046

    vae_ckpt, vae_bundle = _load_vae_bundle(args)
    datasets = morphomnist(args)
    _validate_vae_checkpoint(args, vae_bundle, datasets["test"])

    pgm_ckpt, pgm_bundle = _load_pgm_bundle(args)
    _validate_pgm_checkpoint(args, pgm_bundle, datasets["test"])

    predictor_ckpt, predictor_bundle = _load_predictor_bundle(args)
    _validate_predictor_checkpoint(args, predictor_bundle, datasets["test"])

    if jax.tree_util.tree_structure(vae_bundle.params) != jax.tree_util.tree_structure(
        vae_ckpt.get("ema_params", vae_ckpt.get("params"))
    ):
        raise ValueError("VAE checkpoint parameter structure is incompatible with HVAE")

    # Build a fresh model state from the loaded VAE weights.
    state = {
        "vae_params": vae_ckpt.get("ema_params", vae_ckpt.get("params")),
        "opt_state": None,
        "lmbda": jnp.asarray(args.lmbda_init, dtype=jnp.float32),
        "lambda_opt_state": None,
        "ema": EMA.init_from(
            vae_ckpt.get("ema_params", vae_ckpt.get("params")),
            args.ema_rate,
            update_after_step=100,
        ),
        "ema_params": tree_copy(vae_ckpt.get("ema_params", vae_ckpt.get("params"))),
        "step": 0,
        "epoch": 0,
        "best_loss": float("inf"),
    }

    optimizer, lambda_optimizer = _make_optimizers(args)
    state["opt_state"] = optimizer.init(state["vae_params"])
    state["lambda_opt_state"] = lambda_optimizer.init(state["lmbda"])

    if args.load_path:
        if os.path.exists(args.load_path) or str(args.load_path).startswith("gs://"):
            template = {
                "epoch": state["epoch"],
                "step": state["step"],
                "best_loss": state["best_loss"],
                "vae_params": state["vae_params"],
                "ema_params": state["ema_params"],
                "opt_state": state["opt_state"],
                "lmbda": state["lmbda"],
                "lambda_opt_state": state["lambda_opt_state"],
            }
            ckpt = _load_runtime_checkpoint(args, args.load_path, template=template)
            _restore_args(args, ckpt)
            state["vae_params"] = ckpt["vae_params"]
            state["ema_params"] = ckpt["ema_params"]
            state["ema"] = EMA(
                params=tree_copy(ckpt["ema_params"]),
                decay=args.ema_rate,
                update_after_step=100,
            )
            state["opt_state"] = ckpt["opt_state"]
            state["lmbda"] = ckpt["lmbda"]
            state["lambda_opt_state"] = ckpt["lambda_opt_state"]
            state["step"] = int(ckpt.get("step", 0))
            state["epoch"] = int(ckpt.get("epoch", 0))
            state["best_loss"] = float(ckpt.get("best_loss", float("inf")))
            optimizer, lambda_optimizer = _make_optimizers(args)
        else:
            print(f"Checkpoint not found: {args.load_path}")

    args.save_dir = experiment_run_dir(args.ckpt_dir, args.hps, args.exp_name, "cf")
    args.checkpoint_dir = checkpoint_root_dir(args.save_dir)
    args.remote_save_dir = experiment_run_dir(args.remote_ckpt_dir, args.hps, args.exp_name, "cf")
    ensure_dir(args.save_dir)
    ensure_dir(args.checkpoint_dir)
    logger = setup_logging(args)
    writer = SummaryWriter(args.save_dir)
    train_samples = datasets["train"].samples

    for key in sorted(vars(args)):
        logger.info("--%s=%s", key, getattr(args, key))

    eval_step = _make_eval_step(args, vae_bundle, pgm_bundle, predictor_bundle)
    train_step = _make_train_step(
        args, vae_bundle, pgm_bundle, predictor_bundle, optimizer, lambda_optimizer
    )
    dag_vars = list(MorphoMNISTPGM.variables.keys())
    rng = np.random.default_rng(args.seed)

    if args.testing:
        if not args.load_path:
            raise ValueError("--testing requires --load_path")
        stats, metrics = _eval_split(
            args,
            "test",
            datasets,
            state,
            vae_bundle,
            pgm_bundle,
            predictor_bundle,
            eval_step,
            train_samples,
            rng,
        )
        print("\n[test] " + " - ".join(f"{k}: {v:.4f}" for k, v in stats.items()))
        print("[test] " + " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
        writer.close()
        return

    benchmark_start_step = state["step"]
    benchmark_done = False
    for epoch in range(state["epoch"], args.epochs):
        logger.info("Epoch %d:", epoch + 1)
        totals: Dict[str, float] = {}
        seen = 0
        last_valid_stats: Optional[Dict[str, float]] = None
        last_valid_metrics: Optional[Dict[str, float]] = None
        steps_in_epoch = max(1, len(datasets["train"]) // args.bs)
        progress = tqdm(
            _epoch_batches(datasets["train"], args.bs, shuffle=True, drop_last=True, rng=rng),
            total=steps_in_epoch,
            mininterval=0.1,
        )

        for i, raw_batch in enumerate(progress):
            batch = preprocess_batch(args, raw_batch, compact_pa=True)
            do_k = _choose_intervention(args, dag_vars)
            do = _make_intervention(args, batch, do_k, train_samples, train=True)
            vae_params, opt_state, lmbda, lambda_opt_state, out = train_step(
                state["vae_params"],
                state["opt_state"],
                state["lmbda"],
                state["lambda_opt_state"],
                batch,
                do,
                jax.random.PRNGKey(args.seed + state["step"] + i + epoch * 1000),
            )
            state["vae_params"] = vae_params
            state["opt_state"] = opt_state
            state["lmbda"] = lmbda
            state["lambda_opt_state"] = lambda_opt_state
            if float(out["update_skipped"]) == 0.0:
                state["ema"].update(state["vae_params"])
                state["ema_params"] = tree_copy(state["ema"].params)
            bs = int(batch["x"].shape[0])
            seen += bs
            state["step"] += 1
            totals["loss"] = totals.get("loss", 0.0) + float(out["loss"]) * bs
            totals["aux_loss"] = totals.get("aux_loss", 0.0) + float(out["aux_loss"]) * args.alpha * bs
            totals["elbo"] = totals.get("elbo", 0.0) + float(out["elbo"]) * bs
            totals["nll"] = totals.get("nll", 0.0) + float(out["nll"]) * bs
            totals["kl"] = totals.get("kl", 0.0) + float(out["kl"]) * bs
            progress.set_description(
                f"[train] lmbda: {float(state['lmbda']):.3f}, "
                + ", ".join(f"{k}: {v / max(1, seen):.3f}" for k, v in totals.items())
                + (f", grad_norm: {float(out['grad_norm']):.3f}" if "grad_norm" in out else "")
            )

            if args.benchmark_steps > 0 and state["step"] - benchmark_start_step >= args.benchmark_steps:
                benchmark_done = True
                break

            if i % max(1, args.plot_freq) == 0:
                copy_do_pa = copy.deepcopy(args.do_pa)
                for pa_k in dag_vars + [None]:
                    args.do_pa = pa_k
                    valid_stats, valid_metrics = _eval_split(
                        args,
                        "valid",
                        datasets,
                        state,
                        vae_bundle,
                        pgm_bundle,
                        predictor_bundle,
                        eval_step,
                        train_samples,
                        rng,
                    )
                    loginfo(f"valid do({pa_k})", logger, valid_stats)
                    loginfo(f"valid do({pa_k})", logger, valid_metrics)
                    last_valid_stats, last_valid_metrics = valid_stats, valid_metrics
                args.do_pa = copy_do_pa

        if benchmark_done:
            logger.info("Benchmark completed after %d training step(s).", args.benchmark_steps)
            break

        train_stats = {k: v / max(1, seen) for k, v in totals.items()}
        loginfo("train", logger, train_stats)

        if epoch % max(1, args.eval_freq) == 0:
            copy_do_pa = copy.deepcopy(args.do_pa)
            for pa_k in dag_vars + [None]:
                args.do_pa = pa_k
                valid_stats, valid_metrics = _eval_split(
                    args,
                    "valid",
                    datasets,
                    state,
                    vae_bundle,
                    pgm_bundle,
                    predictor_bundle,
                    eval_step,
                    train_samples,
                    rng,
                )
                loginfo(f"valid do({pa_k})", logger, valid_stats)
                loginfo(f"valid do({pa_k})", logger, valid_metrics)
                last_valid_stats, last_valid_metrics = valid_stats, valid_metrics
            args.do_pa = copy_do_pa

            if last_valid_stats is not None:
                for k, v in train_stats.items():
                    writer.add_scalar("train/" + k, v, state["step"])
                    writer.add_scalar("valid/" + k, last_valid_stats[k], state["step"])
                for k, v in (last_valid_metrics or {}).items():
                    writer.add_scalar("valid/" + k, v, state["step"])
                writer.add_scalar("loss/train", train_stats["loss"], state["step"])
                writer.add_scalar("loss/valid", last_valid_stats["loss"], state["step"])
                writer.add_scalar("aux_loss/train", train_stats["aux_loss"], state["step"])
                writer.add_scalar("aux_loss/valid", last_valid_stats["aux_loss"], state["step"])

                if last_valid_stats["loss"] < state["best_loss"]:
                    state["best_loss"] = last_valid_stats["loss"]
                    _save_cf_checkpoint(args, state, epoch + 1)
                    logger.info("Model saved: %s", args.checkpoint_dir)

        writer.flush()
        if getattr(args, "remote_save_dir", ""):
            sync_file(
                os.path.join(args.save_dir, "trainlog.txt"),
                os.path.join(args.remote_save_dir, "trainlog.txt"),
            )
            sync_tree(args.checkpoint_dir, os.path.join(args.remote_save_dir, "checkpoints"))

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    parser.set_defaults(lr=1e-4, eval_freq=1)
    parser.add_argument("--load_path", type=str, default="")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--pgm_path", type=str, default="checkpoints/morphomnist/pgm/checkpoints")
    parser.add_argument("--predictor_path", type=str, default="checkpoints/morphomnist/run/checkpoints")
    parser.add_argument("--vae_path", type=str, default="checkpoints/morphomnist/run/checkpoints")
    parser.add_argument("--testing", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lmbda_init", type=float, default=0.0)
    parser.add_argument("--lr_lagrange", type=float, default=1e-2)
    parser.add_argument("--damping", type=float, default=100.0)
    parser.add_argument("--do_pa", type=str, default=None)
    parser.add_argument("--plot_freq", type=int, default=500)
    parser.add_argument("--imgs_plot", type=int, default=10)
    parser.add_argument("--cf_particles", type=int, default=1)
    parser.add_argument("--elbo_constraint", type=float, default=1.841216802597046)
    parser.add_argument(
        "--model_validation_batches",
        type=int,
        default=1,
        help="Test batches used to validate each loaded model; 0 validates the full test split.",
    )
    parser.add_argument(
        "--trust_incomplete_checkpoint",
        action="store_true",
        default=False,
        help="Restore the newest numeric step even if commit_success.txt is missing.",
    )
    args = setup_hparams(parser)
    main(args)
