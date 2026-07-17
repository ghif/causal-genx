from __future__ import annotations

import json
import io
import logging
import os
import random
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, Tuple

import time

from runtime import configure_backend_from_argv

configure_backend_from_argv()

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np


class EvalOnlyFileFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, "eval_log", False) or record.levelno >= logging.WARNING)


class SyncFileHandler(logging.FileHandler):
    def __init__(self, filename: str, mode: str = "a", encoding: str | None = None, delay: bool = False):
        super().__init__(filename, mode=mode, encoding=encoding, delay=delay)

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        if self.stream is not None and not self.stream.closed:
            self.flush()
            try:
                os.fsync(self.stream.fileno())
            except OSError:
                pass


def append_text_file(local_path: str, text: str, remote_path: str | None = None) -> None:
    ensure_parent_dir(local_path)
    with open(local_path, "a", encoding="utf-8") as f:
        f.write(text)
        if text and not text.endswith("\n"):
            f.write("\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    if remote_path:
        sync_file(local_path, remote_path)

class _NoOpMonitoring:
    def record_scalar(self, *args, **kwargs):
        return None

    def record_event(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


if not hasattr(jax, "monitoring") or not hasattr(jax.monitoring, "record_scalar") or not hasattr(jax.monitoring, "record_event"):
    jax.monitoring = _NoOpMonitoring()

import orbax.checkpoint as ocp
from etils import epath
from flax import nnx

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def seed_all(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("XLA_FLAGS", "--xla_cpu_enable_fast_math=false")


def normalize(x, x_min=None, x_max=None, zero_one=False):
    x = jnp.asarray(x, dtype=jnp.float32)
    if x_min is None:
        x_min = jnp.min(x)
    if x_max is None:
        x_max = jnp.max(x)
    x = (x - x_min) / (x_max - x_min + 1e-12)
    return x if zero_one else 2.0 * x - 1.0


def log_standardize(x):
    x = jnp.asarray(x, dtype=jnp.float32)
    lx = jnp.log(jnp.maximum(x, 1e-12))
    return (lx - jnp.mean(lx)) / jnp.maximum(jnp.std(lx), 1e-12)


def linear_warmup(warmup_iters):
    def f(step):
        return jnp.where(step > warmup_iters, 1.0, step / jnp.maximum(1, warmup_iters))

    return f


def exists(val) -> bool:
    return val is not None


def is_remote_path(path: str) -> bool:
    return path.startswith("gs://")


def _remote_fs(path: str):
    if not is_remote_path(path):
        return None
    try:
        import fsspec
    except ImportError as exc:
        raise ImportError("GCS paths require fsspec/gcsfs.") from exc
    return fsspec.filesystem("gcs")


def local_staging_path(remote_path: str) -> str:
    clean = remote_path.replace("gs://", "gs__/").strip("/")
    return os.path.join(tempfile.gettempdir(), "causal-gen-artifacts", clean)


def open_file(path: str, mode: str = "rb"):
    if is_remote_path(path):
        import fsspec

        return fsspec.open(path, mode=mode).open()
    return open(path, mode)


def path_exists(path: str) -> bool:
    fs = _remote_fs(path)
    if fs is None:
        return os.path.exists(path)
    return fs.exists(path)


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if not parent:
        return
    fs = _remote_fs(parent)
    if fs is None:
        os.makedirs(parent, exist_ok=True)
    else:
        fs.makedirs(parent, exist_ok=True)


def ensure_dir(path: str):
    fs = _remote_fs(path)
    if fs is None:
        os.makedirs(path, exist_ok=True)
    else:
        fs.makedirs(path, exist_ok=True)


def checkpoint_root_dir(save_dir: str) -> str:
    return os.path.abspath(os.path.join(save_dir, "checkpoints"))


def experiment_run_dir(root_dir: str, hps: str, exp_name: str, default_name: str) -> str:
    if not root_dir:
        return ""
    return os.path.join(root_dir, hps, exp_name or default_name)


def materialize_nnx(graphdef, params):
    return nnx.merge(graphdef, nnx.State(params))


def viz_path_for_step(save_dir: str, step: int) -> str:
    return os.path.join(save_dir, f"viz-step-{int(step)}.png")


def sync_file(local_path: str, remote_path: str) -> None:
    if not is_remote_path(remote_path) or local_path == remote_path:
        return
    ensure_parent_dir(remote_path)
    with open(local_path, "rb") as src, open_file(remote_path, "wb") as dst:
        shutil.copyfileobj(src, dst)


def sync_tree(local_dir: str, remote_dir: str) -> None:
    if not is_remote_path(remote_dir) or local_dir == remote_dir:
        return
    ensure_dir(remote_dir)
    for root, _, files in os.walk(local_dir):
        rel_root = os.path.relpath(root, local_dir)
        for name in files:
            local_path = os.path.join(root, name)
            remote_path = remote_dir if rel_root == "." else os.path.join(remote_dir, rel_root)
            sync_file(local_path, os.path.join(remote_path, name))


def _is_legacy_checkpoint_file(path: str) -> bool:
    return path.endswith(".pt") or path.endswith(".pkl")


def _checkpoint_manager(root_dir: str, *, create: bool) -> ocp.CheckpointManager:
    options = ocp.CheckpointManagerOptions(
        max_to_keep=3,
        create=create,
        save_interval_steps=1,
        enable_async_checkpointing=False,
    )
    return ocp.CheckpointManager(root_dir, options=options)


def _dir_entries(path: str):
    if is_remote_path(path):
        fs = _remote_fs(path)
        if fs is None:
            return []
        try:
            entries = fs.ls(path, detail=True)
        except Exception:
            return []
        out = []
        for entry in entries:
            if isinstance(entry, dict):
                name = entry.get("name", "")
                entry_type = entry.get("type", "")
            else:
                name = str(entry)
                entry_type = ""
            if not name.startswith("gs://"):
                name = "gs://" + name.lstrip("/")
            out.append((name.rstrip("/"), entry_type))
        return out

    if not os.path.isdir(path):
        return []
    out = []
    for name in os.listdir(path):
        child = os.path.join(path, name)
        out.append((child, "directory" if os.path.isdir(child) else "file"))
    return out


def _is_orbax_step_dir(path: str) -> bool:
    return path_exists(os.path.join(path, "_CHECKPOINT_METADATA"))


def _find_orbax_step_dir(path: str, max_depth: int = 4) -> Optional[str]:
    from collections import deque

    root = path.rstrip("/")
    queue = deque([(root, 0)])
    seen = set()
    while queue:
        current, depth = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        if _is_orbax_step_dir(current):
            return current
        if depth >= max_depth:
            continue
        for child, entry_type in _dir_entries(current):
            if entry_type in {"directory", "dir"} or not entry_type:
                queue.append((child, depth + 1))

    return None


def _is_complete_orbax_step_dir(path: str) -> bool:
    if not _is_orbax_step_dir(path):
        return False
    if path_exists(os.path.join(path, "commit_success.txt")):
        return True
    if not is_remote_path(path) and ".orbax-checkpoint-tmp" not in path:
        return True
    return False


def _find_latest_complete_orbax_step_dir(path: str) -> Optional[str]:
    root = path.rstrip("/")

    if _is_complete_orbax_step_dir(root):
        return root

    numeric_children = []
    for child, entry_type in _dir_entries(root):
        name = os.path.basename(child.rstrip("/"))
        if not name.isdigit():
            continue
        if entry_type in {"directory", "dir"} or not entry_type:
            numeric_children.append((int(name), child))

    for _, child in sorted(numeric_children, reverse=True):
        if _is_complete_orbax_step_dir(child):
            return child

    return None


def _find_latest_orbax_step_dir(path: str) -> Optional[str]:
    root = path.rstrip("/")

    if _is_orbax_step_dir(root):
        return root

    numeric_children = []
    for child, entry_type in _dir_entries(root):
        name = os.path.basename(child.rstrip("/"))
        if not name.isdigit():
            continue
        if entry_type in {"directory", "dir"} or not entry_type:
            numeric_children.append((int(name), child))

    if not numeric_children:
        return None

    return sorted(numeric_children, reverse=True)[0][1]


def _restore_orbax_step_direct(
    step_dir: str,
    template: Optional[Dict[str, Any]],
    fallback_sharding: Optional[Any],
    suppress_warnings: bool = False,
) -> Dict[str, Any]:
    parent_dir = os.path.dirname(step_dir.rstrip("/"))
    step_name = os.path.basename(step_dir.rstrip("/"))
    try:
        int(step_name)
    except ValueError as exc:
        raise ValueError(f"Unsupported Orbax step directory: {step_dir}") from exc

    with _orbax_warning_filter(suppress_warnings):
        manager = _checkpoint_manager(parent_dir, create=False)
        try:
            restore_args = ocp.args.StandardRestore(item=template, fallback_sharding=fallback_sharding)
            ckpt_args = ocp.args.Composite(default=restore_args)
            # Checkpointer.restore rejects directories without commit_success.txt.
            # Calling the handler is the explicit escape hatch for checkpoints whose
            # payload finished uploading but whose final marker was never written.
            composite = manager._checkpointer._handler.restore(  # type: ignore[attr-defined]
                epath.Path(step_dir),
                args=ckpt_args,
            )
            restored = composite["default"]
            _load_hparams_if_present(parent_dir, restored)
            return restored
        finally:
            manager.close()


@contextmanager
def _orbax_warning_filter(enabled: bool):
    if not enabled:
        yield
        return

    from absl import logging as absl_logging

    previous_verbosity = absl_logging.get_verbosity()
    absl_logging.set_verbosity(absl_logging.ERROR)
    try:
        yield
    finally:
        absl_logging.set_verbosity(previous_verbosity)


def _load_hparams_if_present(root_dir: str, restored: Dict[str, Any]) -> None:
    hparams_path = os.path.join(root_dir, "hparams.json")
    if path_exists(hparams_path):
        with open_file(hparams_path, "r") as f:
            restored["hparams"] = json.load(f)


def save_checkpoint(data: Dict[str, Any], path: str, step: Optional[int] = None, custom_metadata: Optional[Dict[str, Any]] = None) -> None:
    if _is_legacy_checkpoint_file(path):
        ensure_parent_dir(path)
        import pickle

        with open(path, "wb") as f:
            pickle.dump(data, f)
        return

    path = os.path.abspath(path)
    ensure_dir(path)
    item = dict(data)
    metadata = dict(custom_metadata or {})
    hparams = item.pop("hparams", None)
    if hparams is not None:
        metadata.setdefault("hparams", hparams)
        with open(os.path.join(path, "hparams.json"), "w", encoding="utf-8") as f:
            json.dump(hparams, f, indent=2, sort_keys=True)
    manager = _checkpoint_manager(path, create=True)
    try:
        save_step = int(step if step is not None else data.get("step", 0))
        manager.save(
            save_step,
            args=ocp.args.StandardSave(item=item, custom_metadata=metadata),
        )
        manager.wait_until_finished()
    finally:
        manager.close()


def _save_checkpoint_and_sync(
    data: Dict[str, Any],
    path: str,
    step: Optional[int],
    custom_metadata: Optional[Dict[str, Any]],
    local_tree_dir: Optional[str],
    remote_tree_dir: Optional[str],
) -> None:
    save_checkpoint(data, path, step=step, custom_metadata=custom_metadata)
    if local_tree_dir and remote_tree_dir:
        sync_tree(local_tree_dir, remote_tree_dir)


class BackgroundArtifactWriter:
    """Serialize artifact saves on the caller thread."""

    def __init__(self):
        self._closed = False

    def _submit_job(
        self,
        fn,
        *args,
        **kwargs,
    ):
        if self._closed:
            raise RuntimeError("BackgroundArtifactWriter is closed.")
        return fn(*args, **kwargs)

    def submit_checkpoint(
        self,
        data: Dict[str, Any],
        path: str,
        step: Optional[int] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
        local_tree_dir: Optional[str] = None,
        remote_tree_dir: Optional[str] = None,
    ):
        self._submit_job(
            _save_checkpoint_and_sync,
            data,
            path,
            step,
            custom_metadata,
            local_tree_dir,
            remote_tree_dir,
        )
        return path

    def submit_viz(
        self,
        args,
        model,
        params,
        batch,
        rng_key=None,
        step: Optional[int] = None,
    ):
        viz_step = int(step if step is not None else getattr(args, "iter", 0))
        viz_path = viz_path_for_step(args.save_dir, viz_step)
        self._submit_job(write_images, args, model, params, batch, rng_key, step)
        return viz_path

    def flush(self):
        return None

    def close(self):
        if self._closed:
            return
        self._closed = True


AsyncCheckpointWriter = BackgroundArtifactWriter


def resolve_checkpoint_path(path: str, allow_incomplete: bool = False) -> str:
    """Resolve a checkpoint root to the exact file or numeric Orbax step restored."""
    path = path.rstrip("/")
    if _is_legacy_checkpoint_file(path):
        return path

    if _is_orbax_step_dir(path):
        if allow_incomplete or _is_complete_orbax_step_dir(path):
            return path
        parent_dir = os.path.dirname(path)
        latest = _find_latest_complete_orbax_step_dir(parent_dir)
        if latest is not None:
            return latest
        raise ValueError(
            f"Checkpoint {path} is missing commit_success.txt. "
            "Pass allow_incomplete=True (infer.py: --trust_incomplete_checkpoint) "
            "to restore it anyway."
        )

    if allow_incomplete:
        latest = _find_latest_orbax_step_dir(path)
        if latest is not None:
            return latest

    latest_complete = _find_latest_complete_orbax_step_dir(path)
    if latest_complete is not None:
        return latest_complete

    step_dir = _find_orbax_step_dir(path)
    if step_dir is not None and step_dir != path:
        return resolve_checkpoint_path(step_dir, allow_incomplete=allow_incomplete)

    manager = _checkpoint_manager(path, create=False)
    try:
        step = manager.latest_step()
        if step is None:
            raise FileNotFoundError(f"No Orbax checkpoints found in {path}")
        return os.path.join(path, str(step))
    finally:
        manager.close()


def checkpoint_is_complete(path: str) -> bool:
    if _is_legacy_checkpoint_file(path):
        return True
    return _is_complete_orbax_step_dir(path)


def load_checkpoint_with_path(
    path: str,
    template: Optional[Dict[str, Any]] = None,
    fallback_sharding: Optional[Any] = None,
    allow_incomplete: bool = False,
) -> Tuple[Dict[str, Any], str]:
    resolved_path = resolve_checkpoint_path(path, allow_incomplete=allow_incomplete)
    if _is_legacy_checkpoint_file(resolved_path):
        import pickle

        with open(resolved_path, "rb") as f:
            return pickle.load(f), resolved_path

    restored = _restore_orbax_step_direct(
        resolved_path,
        template,
        fallback_sharding,
        suppress_warnings=allow_incomplete and not _is_complete_orbax_step_dir(resolved_path),
    )
    return restored, resolved_path


def load_checkpoint(
    path: str,
    template: Optional[Dict[str, Any]] = None,
    fallback_sharding: Optional[Any] = None,
    allow_incomplete: bool = False,
) -> Dict[str, Any]:
    restored, _ = load_checkpoint_with_path(
        path,
        template=template,
        fallback_sharding=fallback_sharding,
        allow_incomplete=allow_incomplete,
    )
    return restored


def tree_copy(tree):
    return jax.tree_util.tree_map(lambda x: x.copy() if hasattr(x, "copy") else x, tree)


@dataclass
class EMA:
    params: Any
    decay: float = 0.999
    update_after_step: Optional[int] = None
    step: int = 0
    initted: bool = False

    @classmethod
    def init_from(cls, params, decay: float = 0.999, update_after_step: Optional[int] = None):
        return cls(
            params=tree_copy(params),
            decay=decay,
            update_after_step=update_after_step,
        )

    def update(self, params):
        if self.update_after_step is not None:
            current_step = self.step
            self.step += 1
            if current_step <= self.update_after_step:
                self.params = tree_copy(params)
                return
            if not self.initted:
                self.params = tree_copy(params)
                self.initted = True

            epoch = max(self.step - self.update_after_step - 1, 0)
            current_decay = 0.0 if epoch <= 0 else min(
                self.decay, 1.0 - (1.0 + epoch) ** -1.0
            )
        else:
            current_decay = self.decay

        self.params = jax.tree_util.tree_map(
            lambda e, p: current_decay * e + (1.0 - current_decay) * p,
            self.params,
            params,
        )


def clamp(value, min_value=None, max_value=None):
    if min_value is not None:
        value = jnp.maximum(value, min_value)
    if max_value is not None:
        value = jnp.minimum(value, max_value)
    return value


def _to_uint8_image(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def postprocess(x):
    x = np.asarray(x)
    x = (x + 1.0) * 127.5
    return np.clip(x, 0, 255).astype(np.uint8)


def _ensure_nhwc(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim == 4 and images.shape[1] in (1, 3):
        return np.transpose(images, (0, 2, 3, 1))
    return images


def _repeat_batch(value: np.ndarray, count: int) -> np.ndarray:
    value = jnp.asarray(value)
    return jnp.repeat(value[None, ...], count, axis=0)


def _morphomnist_counterfactual_parents(base_pa: np.ndarray, source_idx: int, target_idx: int, context_dim: int, input_res: int) -> tuple[np.ndarray, np.ndarray]:
    pa = _repeat_batch(base_pa[source_idx], context_dim)
    cf_pa = pa.copy()
    cf_pa = cf_pa.at[0, 0].set(base_pa[target_idx, 0])
    cf_pa = cf_pa.at[1, 1].set(base_pa[target_idx, 1])
    cf_pa = cf_pa.at[2:, 2:].set(jnp.eye(10, dtype=cf_pa.dtype))
    pa = jnp.repeat(jnp.repeat(pa[:, None, None, :], input_res, axis=1), input_res, axis=2)
    cf_pa = jnp.repeat(jnp.repeat(cf_pa[:, None, None, :], input_res, axis=1), input_res, axis=2)
    return pa, cf_pa


def make_image_grid(images: Sequence[np.ndarray], n_rows: int, n_cols: int) -> np.ndarray:
    rows = [np.asarray(img) for img in images]
    if rows[0].ndim == 3:
        rows = [row[None, ...] for row in rows]
    if rows[0].ndim != 4:
        raise ValueError(f"Expected 4D row tensors, got shape {rows[0].shape}.")
    n_cols = min(n_cols, rows[0].shape[0])
    h, w = rows[0].shape[1:3]
    c = rows[0].shape[-1]
    padded_rows = []
    for row in rows:
        if row.shape[0] < n_cols:
            pad = np.zeros((n_cols - row.shape[0], h, w, c), dtype=row.dtype)
            row = np.concatenate([row, pad], axis=0)
        padded_rows.append(row[:n_cols])
    im = (
        np.concatenate(padded_rows, axis=0)
        .reshape((n_rows, n_cols, h, w, c))
        .transpose([0, 2, 1, 3, 4])
        .reshape([n_rows * h, n_cols * w, c])
    )
    return im.squeeze(-1) if im.ndim == 3 and im.shape[-1] == 1 else im


def batch_iterator(
    dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    *,
    drop_remainder: bool = False,
) -> Iterator[Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    while True:
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            if drop_remainder and len(batch_idx) < batch_size:
                continue
            if hasattr(dataset, "make_batch"):
                yield dataset.make_batch(batch_idx, rng=rng, shuffle=shuffle)
                continue
            batch = [dataset[int(i)] for i in batch_idx]
            keys = batch[0].keys()
            out = {}
            for k in keys:
                values = [np.asarray(item[k]) for item in batch]
                out[k] = np.stack(values, axis=0)
            yield out


def write_images(args, model, params, batch, rng_key=None, step: Optional[int] = None):
    viz_batch_size = int(getattr(args, "viz_batch_size", getattr(args, "viz_bs", 32)))
    x = np.asarray(batch["x"], dtype=np.float32)
    if x.max() > 1.5:
        x = (x - 127.5) / 127.5
    pa = np.asarray(batch["pa"], dtype=np.float32)
    if x.ndim == 4 and x.shape[1] in (1, 3):
        x = np.transpose(x, (0, 2, 3, 1))
    if pa.ndim == 2:
        pa = pa[:, :, None, None]
        pa = np.repeat(pa, args.input_res, axis=2)
        pa = np.repeat(pa, args.input_res, axis=3)
        pa = np.transpose(pa, (0, 2, 3, 1))
    elif pa.ndim == 4 and pa.shape[1] == args.context_dim and pa.shape[-1] == args.input_res:
        pa = np.transpose(pa, (0, 2, 3, 1))
    viz_batch_size = max(1, min(viz_batch_size, x.shape[0]))
    x = x[:viz_batch_size]
    pa = pa[:viz_batch_size]
    model = materialize_nnx(model, params)
    bs = int(min(viz_batch_size, x.shape[0]))
    x = x[:bs]
    x_jax = jnp.asarray(x)
    pa_jax = jnp.asarray(pa)
    rows = [postprocess(x)]

    def _append_counterfactual_rows(zs, pa_ctx, cf_pa_ctx, x_ctx, alpha, t):
        x_rec, _ = model.forward_latents(latents=zs, parents=pa_ctx, t=t)
        x_rec = postprocess(x_rec)

        cf_x, _ = model.forward_latents(latents=zs, parents=cf_pa_ctx, t=t)
        cf_x = postprocess(cf_x)
        rows.append(cf_x.astype(np.uint8))
        rows.append((cf_x - x_rec).astype(np.uint8))

        if getattr(model, "cond_prior", False):
            # Match the Torch visualization path: re-abduct on the counterfactual parents
            # and show the indirect and total effect rows as well.
            cf_z = model.abduct(x=x_ctx, parents=pa_ctx, cf_parents=cf_pa_ctx, alpha=alpha, t=t)

            x_indirect, _ = model.forward_latents(latents=cf_z, parents=pa_ctx, t=t)
            x_indirect = postprocess(x_indirect)
            rows.append(x_indirect.astype(np.uint8))
            rows.append((x_indirect - x_rec).astype(np.uint8))

            x_total, _ = model.forward_latents(latents=cf_z, parents=cf_pa_ctx, t=t)
            x_total = postprocess(x_total)
            rows.append(x_total.astype(np.uint8))
            rows.append((x_total - x_rec).astype(np.uint8))

    try:
        zs = model.abduct(x=x_jax, parents=pa_jax)
        n_latents_viz = 0
        l_points = np.floor(np.linspace(0, 1, n_latents_viz + 2) * len(zs)).astype(int)[1:]
        for l in l_points:
            latents = zs[:l]
            x_rec, _ = model.forward_latents(latents=latents, parents=pa_jax, t=0.1)
            rows.append(postprocess(x_rec))
    except AttributeError:
        pass

    rows.append(postprocess(x * 0))
    for temp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        sample, _ = model.sample(parents=pa_jax, return_loc=True, t=temp, rng=rng_key)
        rows.append(postprocess(sample))

    if "morphomnist" in getattr(args, "hps", ""):
        base_pa = np.asarray(pa)[:bs]
        if base_pa.ndim == 4:
            base_pa = base_pa[:, 0, 0, :]
        idx = np.arange(bs)
        np.random.RandomState(1).shuffle(idx)
        alpha, t = 0.6, 0.5
        for l in l_points:
            rows.append(postprocess(x * 0))
            for ii in range(bs):
                if getattr(model, "cond_prior", False):
                    x_ctx = _repeat_batch(x[ii], args.context_dim)
                else:
                    x_ctx = None
                pa_ctx, cf_pa_ctx = _morphomnist_counterfactual_parents(
                    base_pa=base_pa,
                    source_idx=ii,
                    target_idx=idx[ii],
                    context_dim=args.context_dim,
                    input_res=args.input_res,
                )
                z_i = []
                for z in zs:
                    if getattr(model, "cond_prior", False):
                        z_dict = {}
                        for k, v in z.items():
                            z_dict[k] = _repeat_batch(v[ii], args.context_dim)
                        z_i.append(z_dict)
                    else:
                        z_i.append(_repeat_batch(z[ii], args.context_dim))
                if getattr(model, "cond_prior", False):
                    latents = [z_i[j]["z"] for j in range(l)]
                else:
                    latents = z_i[:l]
                _append_counterfactual_rows(latents, pa_ctx, cf_pa_ctx, x_ctx, alpha, t)
                rows.append(postprocess(x * 0))

    for j, img in enumerate(rows):
        if img.shape[0] < bs:
            pad = np.zeros((bs - img.shape[0], *img.shape[1:]), dtype=np.uint8)
            rows[j] = np.concatenate([img, pad], axis=0)

    grid = make_image_grid(rows, n_rows=len(rows), n_cols=bs)
    viz_step = int(step if step is not None else getattr(args, "iter", 0))
    viz_path = viz_path_for_step(args.save_dir, viz_step)
    imageio.imwrite(viz_path, grid)
    remote_run_dir = getattr(args, "remote_save_dir", "")
    if remote_run_dir:
        sync_file(viz_path, viz_path_for_step(remote_run_dir, viz_step))
    return viz_path


class SummaryWriter:
    def __init__(self, logdir: str):
        ensure_dir(logdir)
        self._writer = EventFileWriter(logdir)

    def add_scalar(self, tag: str, value: float, step: int):
        event = Event(
            wall_time=time.time(),
            step=int(step),
            summary=Summary(value=[Summary.Value(tag=tag, simple_value=float(value))]),
        )
        self._writer.add_event(event)
        self._writer.flush()

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()
