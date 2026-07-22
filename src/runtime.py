from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RuntimeSummary:
    """Resolved JAX runtime information after backend initialization."""

    requested_accelerator: str
    backend: str
    local_device_count: int
    global_device_count: int
    process_count: int
    process_index: int
    device_kind: str
    jax_version: str


def _argv_value(flag: str) -> Optional[str]:
    argv = sys.argv[1:]
    for i, token in enumerate(argv):
        if token == flag and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def configure_backend(accelerator: str = "cpu", gpu_id: Optional[str] = None) -> str:
    # Configure before importing JAX. The cache makes expensive XLA programs,
    # such as the full HVAE train step, reusable across separate CLI runs.
    cache_root = os.environ.setdefault(
        "JAX_COMPILATION_CACHE_DIR",
        str(Path(tempfile.gettempdir()) / "causal-genx-jax-cache"),
    )
    os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "true")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    matplotlib_cache = os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "causal-genx-matplotlib-cache"),
    )
    try:
        Path(cache_root).mkdir(parents=True, exist_ok=True)
        Path(matplotlib_cache).mkdir(parents=True, exist_ok=True)
    except OSError:
        # JAX can still run without a persistent cache if the selected temp
        # location is unavailable or read-only.
        pass
    accelerator = accelerator.lower()
    if accelerator == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["ROCM_VISIBLE_DEVICES"] = ""
        os.environ["TPU_VISIBLE_DEVICES"] = ""
    elif accelerator in {"gpu", "cuda"}:
        # Selecting the platform explicitly prevents a missing CUDA plugin from
        # silently turning a requested GPU experiment into a CPU experiment.
        os.environ["JAX_PLATFORMS"] = "cuda"
        os.environ.pop("TPU_VISIBLE_DEVICES", None)
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    elif accelerator == "tpu":
        # libtpu registers the `tpu` PJRT backend. An explicit platform makes a
        # bad TPU VM/runtime fail during preflight instead of falling back to CPU.
        os.environ["JAX_PLATFORMS"] = "tpu"
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("ROCM_VISIBLE_DEVICES", None)
        os.environ.pop("TPU_VISIBLE_DEVICES", None)
    else:
        raise ValueError(f"Unsupported accelerator {accelerator!r}")
    os.environ.pop("JAX_PLATFORM_NAME", None)
    return accelerator


def validate_backend(
    accelerator: str,
    *,
    expected_local_device_count: int | None = None,
    expected_global_device_count: int | None = None,
    expected_process_count: int | None = None,
) -> RuntimeSummary:
    """Initialize JAX and fail fast when hardware differs from the config."""
    import jax

    requested = accelerator.lower()
    expected_backend = "gpu" if requested in {"gpu", "cuda"} else requested
    backend = jax.default_backend()
    if backend != expected_backend:
        install_hint = (
            'Install `jax[tpu]` on the TPU VM.'
            if requested == "tpu"
            else "Install the matching JAX accelerator package."
        )
        raise RuntimeError(
            f"Requested accelerator={requested!r}, but JAX initialized backend={backend!r}. {install_hint}"
        )

    local_count = jax.local_device_count()
    global_count = jax.device_count()
    process_count = jax.process_count()
    checks = (
        ("local_device_count", local_count, expected_local_device_count),
        ("global_device_count", global_count, expected_global_device_count),
        ("process_count", process_count, expected_process_count),
    )
    for label, actual, expected in checks:
        if expected is not None and actual != expected:
            raise RuntimeError(
                f"Runtime topology mismatch: expected {label}={expected}, got {actual}. "
                "Check the provisioned accelerator topology and experiment config."
            )

    devices = jax.local_devices()
    device_kind = str(getattr(devices[0], "device_kind", "unknown")) if devices else "none"
    return RuntimeSummary(
        requested_accelerator=requested,
        backend=backend,
        local_device_count=local_count,
        global_device_count=global_count,
        process_count=process_count,
        process_index=jax.process_index(),
        device_kind=device_kind,
        jax_version=jax.__version__,
    )


def configure_backend_from_argv(default: str = "cpu") -> str:
    return configure_backend(
        _argv_value("--accelerator") or default,
        gpu_id=_argv_value("--gpu_id"),
    )
