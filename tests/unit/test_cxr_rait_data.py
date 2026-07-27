import numpy as np
import pytest

from data.cxr_rait import CXR_RAIT_SCHEMA, CxrRaitDataset, CxrRaitProvider


def test_cxr_rait_schema():
    assert CXR_RAIT_SCHEMA.dataset_id == "cxr_rait"
    assert CXR_RAIT_SCHEMA.variable_names == ("age", "sex", "tb_status")
    assert CXR_RAIT_SCHEMA.encoded_dim == 5
    assert CXR_RAIT_SCHEMA.edges == (("age", "tb_status"),)


def test_cxr_rait_provider_mock(tmp_path):
    # Test provider instantiates and Spec returns expected details
    provider = CxrRaitProvider(root=str(tmp_path), input_res=128)
    assert provider.spec.dataset_id == "cxr_rait"
    assert provider.spec.image.height == 128
    assert provider.fingerprint() is not None
