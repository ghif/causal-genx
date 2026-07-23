import os
import sys
from types import SimpleNamespace

import pytest

from runtime import configure_backend, validate_backend


def test_tpu_backend_selection_is_explicit(monkeypatch, tmp_path):
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path / "jax-cache"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("JAX_PLATFORM_NAME", "cpu")

    configure_backend("tpu")

    assert os.environ["JAX_PLATFORMS"] == "tpu"
    assert "JAX_PLATFORM_NAME" not in os.environ
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_cpu_backend_selection_overrides_inherited_platform(monkeypatch, tmp_path):
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path / "jax-cache"))
    monkeypatch.setenv("JAX_PLATFORMS", "tpu")

    configure_backend("cpu")

    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert os.environ["TPU_VISIBLE_DEVICES"] == ""


def test_v6e4_topology_validation_is_fail_fast(monkeypatch):
    fake_jax = SimpleNamespace(
        __version__="0.test",
        default_backend=lambda: "tpu",
        local_device_count=lambda: 4,
        device_count=lambda: 4,
        process_count=lambda: 1,
        process_index=lambda: 0,
        local_devices=lambda: [SimpleNamespace(device_kind="TPU v6e")],
    )
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    summary = validate_backend(
        "tpu",
        expected_local_device_count=4,
        expected_global_device_count=4,
        expected_process_count=1,
    )
    assert summary.device_kind == "TPU v6e"

    with pytest.raises(RuntimeError, match="global_device_count=8"):
        validate_backend("tpu", expected_global_device_count=8)
