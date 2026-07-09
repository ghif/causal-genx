import contextlib
import importlib.metadata
import os
from typing import Any, Dict, Iterable

import torch

_TOPOLOGY_ENV_VARS = (
    "TPU_PROCESS_BOUNDS",
    "TPU_PROCESS_ADDRESSES",
    "TPU_WORKER_HOSTNAMES",
    "TPU_CHIPS_PER_PROCESS_BOUNDS",
)


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def add_custom_scalars(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def _ensure_tpu_accelerator_type() -> None:
    if os.environ.get("PJRT_DEVICE", "").upper() != "TPU":
        return
    if os.environ.get("ACCELERATOR_TYPE"):
        return

    accelerator_type = os.environ.get("TPU_ACCELERATOR_TYPE", "v6e-4")
    os.environ["ACCELERATOR_TYPE"] = accelerator_type
    os.environ.setdefault("TPU_ACCELERATOR_TYPE", accelerator_type)


def _load_xla():
    _ensure_tpu_accelerator_type()
    try:
        import torch_xla.core.xla_model as xm
    except ImportError as exc:
        raise RuntimeError(
            "TPU execution requires torch-xla. Install the matched torch/torch-xla "
            "versions from requirements-tpu.txt."
        ) from exc
    return xm


def tpu_environment_available() -> bool:
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        try:
            _load_xla()
            return True
        except RuntimeError:
            return False
    return False


def is_xla_device(device: torch.device) -> bool:
    return device.type == "xla"


def xla_device() -> torch.device:
    return _load_xla().xla_device()


def world_size() -> int:
    try:
        import torch_xla.runtime as xr

        return xr.world_size()
    except (ImportError, AttributeError):
        return _load_xla().xrt_world_size()


def rank() -> int:
    try:
        import torch_xla.runtime as xr

        return xr.global_ordinal()
    except (ImportError, AttributeError):
        return _load_xla().get_ordinal()


def is_master() -> bool:
    return _load_xla().is_master_ordinal(local=False)


def master_print(*args, **kwargs) -> None:
    if os.environ.get("PJRT_DEVICE", "").upper() != "TPU" or is_master():
        print(*args, **kwargs)


def runtime_diagnostics(device: torch.device) -> Dict[str, Any]:
    diagnostics = {
        "torch": torch.__version__,
        "device": str(device),
        "world_size": 1,
        "rank": 0,
        "PJRT_DEVICE": os.environ.get("PJRT_DEVICE", ""),
    }
    if is_xla_device(device):
        import torch_xla

        diagnostics.update(
            {
                "torch_xla": torch_xla.__version__,
                "libtpu": _package_version("libtpu"),
                "accelerator_type": os.environ.get("ACCELERATOR_TYPE", "unset"),
                "world_size": world_size(),
                "rank": rank(),
                "topology_overrides": ",".join(
                    key for key in _TOPOLOGY_ENV_VARS if os.environ.get(key)
                )
                or "none",
            }
        )
    return diagnostics


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed-as-package"


def rendezvous(tag: str) -> None:
    _load_xla().rendezvous(tag)


def optimizer_step(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    if is_xla_device(device):
        _load_xla().optimizer_step(optimizer, barrier=False)
    else:
        optimizer.step()


def mark_step(device: torch.device) -> None:
    if is_xla_device(device):
        import torch_xla

        torch_xla.sync()


def synchronize(device: torch.device) -> None:
    if is_xla_device(device):
        import torch_xla

        torch_xla.sync(wait=True)
    elif device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        synchronize_fn = getattr(torch.mps, "synchronize", None)
        if synchronize_fn is not None:
            synchronize_fn()


def reduce_sum(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    if is_xla_device(device):
        return _load_xla().all_reduce(
            _load_xla().REDUCE_SUM, value, scale=1.0
        )
    return value


def reduce_stats(stats: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    if not is_xla_device(device):
        return stats

    keys = list(stats)
    values = []
    for key in keys:
        value = stats[key]
        if torch.is_tensor(value):
            value = value.detach().to(device=device, dtype=torch.float32)
        else:
            value = torch.tensor(float(value), device=device, dtype=torch.float32)
        values.append(value)
    reduced = reduce_sum(torch.stack(values), device)
    return {key: reduced[index] for index, key in enumerate(keys)}


def autocast(device: torch.device, precision: str):
    use_bf16 = is_xla_device(device) and precision in {"auto", "bf16"}
    if use_bf16:
        return torch.autocast(device_type="xla", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def wrap_loader(loader: Iterable, device: torch.device) -> Iterable:
    if not is_xla_device(device):
        return loader
    from torch_xla.distributed.parallel_loader import MpDeviceLoader

    return MpDeviceLoader(loader, device)


def launch(function, args=(), debug_single_process: bool = False):
    _ensure_tpu_accelerator_type()
    try:
        import torch_xla
    except ImportError:
        _load_xla()
        import torch_xla
    _validate_xla_version(torch_xla.__version__)
    return torch_xla.launch(
        function, args=args, debug_single_process=debug_single_process
    )


def _validate_xla_version(torch_xla_version: str) -> None:
    torch_release = torch.__version__.split("+", 1)[0]
    xla_release = torch_xla_version.split("+", 1)[0]
    if _major_minor(torch_release) != _major_minor(xla_release):
        raise RuntimeError(
            "torch and torch-xla must use the same major/minor release; "
            f"found torch=={torch.__version__} and torch-xla=={torch_xla_version}."
        )


def _major_minor(version: str):
    return tuple(version.split(".")[:2])


def save(data: Any, path: str, device: torch.device) -> None:
    if is_xla_device(device):
        _load_xla().save(data, path, master_only=False)
    else:
        torch.save(data, path)
