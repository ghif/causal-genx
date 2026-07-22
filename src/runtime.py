from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional


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
    if accelerator == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("ROCM_VISIBLE_DEVICES", "")
        os.environ.setdefault("TPU_VISIBLE_DEVICES", "")
    else:
        os.environ.pop("JAX_PLATFORMS", None)
        os.environ.pop("JAX_PLATFORM_NAME", None)
        if accelerator in {"gpu", "cuda"} and gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    return accelerator


def configure_backend_from_argv(default: str = "cpu") -> str:
    return configure_backend(
        _argv_value("--accelerator") or default,
        gpu_id=_argv_value("--gpu_id"),
    )
