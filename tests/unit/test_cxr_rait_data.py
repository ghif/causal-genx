import numpy as np
import pytest

from data.cxr_rait import CXR_RAIT_SCHEMA, CxrRaitDataset, CxrRaitProvider


def test_cxr_rait_schema():
    assert CXR_RAIT_SCHEMA.dataset_id == "cxr_rait"
    assert CXR_RAIT_SCHEMA.variable_names == ("age", "gender", "tb_status")
    assert CXR_RAIT_SCHEMA.encoded_dim == 5
    assert CXR_RAIT_SCHEMA.edges == (("age", "tb_status"),)


def test_cxr_rait_provider_mock(tmp_path):
    # Test provider instantiates and Spec returns expected details
    provider = CxrRaitProvider(root=str(tmp_path), input_res=128)
    assert provider.spec.dataset_id == "cxr_rait"
    assert provider.spec.image.height == 128
    assert provider.fingerprint() is not None


def test_cxr_rait_dataset_cache_dir_isolation(tmp_path, monkeypatch):
    import sys
    import data.cxr_rait
    monkeypatch.setattr(sys.modules["data.cxr_rait"], "_load_cxr_rait_metadata", lambda path: {"PID1": {"age": 50.0, "gender": 1.0, "tb_status": 0.0}})
    ds_standard = CxrRaitDataset(root_dir=str(tmp_path), input_res=224, torchxray_preprocessing=False)
    ds_torchxray = CxrRaitDataset(root_dir=str(tmp_path), input_res=224, torchxray_preprocessing=True)
    assert ds_standard.cache_dir.endswith("_224")
    assert ds_torchxray.cache_dir.endswith("_224_torchxray")
    assert ds_standard.cache_dir != ds_torchxray.cache_dir

