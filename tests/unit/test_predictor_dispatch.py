from types import SimpleNamespace

import numpy as np
import pytest

from config import load_experiment
from data.padchest import PAD_CHEST_SCHEMA
from training import predictor


class DummyPredictor:
    parent_variables = ("age_at_study", "sex", "tb_status")
    variables = {"age_at_study": "continuous", "sex": "categorical", "tb_status": "binary"}


def test_batch_for_model_filters_to_declared_parent_variables():
    batch = {
        "x": np.zeros((2, 1, 8, 8), dtype=np.float32),
        "age_at_study": np.zeros((2,), dtype=np.float32),
        "sex": np.eye(3, dtype=np.float32)[:2],
        "tb_status": np.ones((2,), dtype=np.float32),
        "morphomnist_only": np.ones((2,), dtype=np.float32),
    }

    filtered = predictor._batch_for_model(DummyPredictor(), batch)

    assert tuple(filtered) == ("x", "age_at_study", "sex", "tb_status")
    assert filtered["age_at_study"].shape == (2, 1)


def test_batch_for_model_reports_missing_schema_key():
    with pytest.raises(KeyError, match="tb_status"):
        predictor._batch_for_model(DummyPredictor(), {"x": np.zeros((1, 1, 8, 8)), "age_at_study": np.zeros((1,)), "sex": np.zeros((1, 3))})


def test_padchest_config_resolves_cxr_predictor_and_seven_parent_schema():
    config = load_experiment("configs/padchest_predictor_tpu_v6e1.yaml")
    args = predictor._run_arguments(config)

    predictor._configure_dataset_args(args)

    assert predictor._predictor_model_family(args) == "cxr"
    assert args.parents_x == list(PAD_CHEST_SCHEMA.variable_names)
    assert len(args.parents_x) == 7
    assert args.context_dim == PAD_CHEST_SCHEMA.encoded_dim


def test_legacy_cxr_dataset_default_predictor_infers_cxr_family():
    args = SimpleNamespace(dataset="cxr_rait", predictor_model="morphomnist_image_parent_predictor")

    assert predictor._predictor_model_family(args) == "cxr"


def test_morphomnist_config_preserves_morphomnist_family():
    args = predictor._run_arguments(load_experiment("configs/morphomnist_predictor.yaml"))

    assert predictor._predictor_model_family(args) == "morphomnist"
