import contextlib
import os
from typing import Any, Dict, Iterable

import torch


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def add_custom_scalars(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def _load_xla():
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


def rendezvous(tag: str) -> None:
    _load_xla().rendezvous(tag)


def optimizer_step(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    if is_xla_device(device):
        _load_xla().optimizer_step(optimizer, barrier=False)
    else:
        optimizer.step()


def mark_step(device: torch.device) -> None:
    if is_xla_device(device):
        _load_xla().mark_step()


def synchronize(device: torch.device) -> None:
    if is_xla_device(device):
        _load_xla().mark_step()
        _load_xla().wait_device_ops()
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


def launch(function, args=()):
    try:
        import torch_xla
    except ImportError:
        _load_xla()
    return torch_xla.launch(function, args=args)


def save(data: Any, path: str, device: torch.device) -> None:
    if is_xla_device(device):
        _load_xla().save(data, path, master_only=False)
    else:
        torch.save(data, path)
