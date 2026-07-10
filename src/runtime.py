from __future__ import annotations

import os
import sys
from typing import Optional


def _argv_value(flag: str) -> Optional[str]:
    argv = sys.argv[1:]
    for i, token in enumerate(argv):
        if token == flag and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def configure_backend(accelerator: str = "cpu") -> str:
    if accelerator == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("ROCM_VISIBLE_DEVICES", "")
        os.environ.setdefault("TPU_VISIBLE_DEVICES", "")
    else:
        os.environ.pop("JAX_PLATFORMS", None)
        os.environ.pop("JAX_PLATFORM_NAME", None)
    return accelerator


def configure_backend_from_argv(default: str = "cpu") -> str:
    return configure_backend(_argv_value("--accelerator") or default)
