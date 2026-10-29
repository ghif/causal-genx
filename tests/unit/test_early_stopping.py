"""Tests for EarlyStopping mechanism and configuration propagation across all stages."""

import pytest
from config import load_experiment
from training.early_stopping import EarlyStopping
from training.scm import _run_arguments as scm_run_arguments
from training.predictor import _run_arguments as predictor_run_arguments
from training.image_model import _run_arguments as image_model_run_arguments
from training.counterfactual import _run_arguments as cf_run_arguments


def test_early_stopping_disabled_by_default():
    es = EarlyStopping(patience=0)
    assert not es.enabled
    assert not es.step(10.0, 1)
    assert not es.step(20.0, 2)
    assert not es.stop


def test_early_stopping_min_mode_triggers_on_plateau():
    es = EarlyStopping(patience=2, min_delta=0.05, mode="min")
    assert es.enabled

    # Epoch 1: Initial score
    assert not es.step(1.0, epoch=1)
    assert es.best_score == 1.0
    assert es.wait == 0

    # Epoch 2: Significant improvement (0.1 > 0.05)
    assert not es.step(0.9, epoch=2)
    assert es.best_score == 0.9
    assert es.wait == 0

    # Epoch 3: Minor improvement (0.02 < 0.05), wait counter increments
    assert not es.step(0.88, epoch=3)
    assert es.best_score == 0.9
    assert es.wait == 1

    # Epoch 4: Stagnant/worse score, wait counter reaches patience
    assert es.step(0.95, epoch=4)
    assert es.wait == 2
    assert es.stop
    assert es.stopped_epoch == 4


def test_early_stopping_max_mode():
    es = EarlyStopping(patience=2, min_delta=0.1, mode="max")
    assert es.enabled

    # Epoch 1: Initial
    assert not es.step(0.5, epoch=1)
    assert es.best_score == 0.5

    # Epoch 2: Significant improvement (0.2 > 0.1)
    assert not es.step(0.7, epoch=2)
    assert es.best_score == 0.7
    assert es.wait == 0

    # Epoch 3: Insufficient improvement (0.05 < 0.1)
    assert not es.step(0.75, epoch=3)
    assert es.wait == 1

    # Epoch 4: Stagnant/decrease -> triggers stop
    assert es.step(0.70, epoch=4)
    assert es.wait == 2
    assert es.stop


def test_early_stopping_nan_and_inf_handling():
    es = EarlyStopping(patience=2, mode="min")

    assert not es.step(float("nan"), epoch=1)
    assert es.best_score is None

    assert not es.step(float("inf"), epoch=2)
    assert es.best_score is None

    assert not es.step(2.5, epoch=3)
    assert es.best_score == 2.5
    assert es.wait == 0

    # NaN in later epoch is ignored and does not crash
    assert not es.step(float("nan"), epoch=4)
    assert es.wait == 0


def test_early_stopping_invalid_mode():
    with pytest.raises(ValueError, match="mode must be 'min' or 'max'"):
        EarlyStopping(patience=3, mode="invalid")


def test_config_propagation_scm():
    config = load_experiment(
        "configs/padchest_scm_v3_gpu.yaml",
        overrides=["workflow.early_stopping_patience=7", "workflow.early_stopping_min_delta=0.005"],
    )
    assert config.workflow.early_stopping_patience == 7
    assert config.workflow.early_stopping_min_delta == 0.005
    args = scm_run_arguments(config)
    assert args.early_stopping_patience == 7
    assert args.early_stopping_min_delta == 0.005


def test_config_propagation_predictor():
    config = load_experiment(
        "configs/padchest_predictor_v3_gpu.yaml",
        overrides=["workflow.early_stopping_patience=4", "workflow.early_stopping_min_delta=0.01"],
    )
    assert config.workflow.early_stopping_patience == 4
    assert config.workflow.early_stopping_min_delta == 0.01
    args = predictor_run_arguments(config)
    assert args.early_stopping_patience == 4
    assert args.early_stopping_min_delta == 0.01


def test_config_propagation_image_model():
    config = load_experiment(
        "configs/padchest_image_model_v3_gpu.yaml",
        overrides=["workflow.early_stopping_patience=10", "workflow.early_stopping_min_delta=0.02"],
    )
    assert config.workflow.early_stopping_patience == 10
    assert config.workflow.early_stopping_min_delta == 0.02
    args = image_model_run_arguments(config)
    assert args.early_stopping_patience == 10
    assert args.early_stopping_min_delta == 0.02


def test_config_propagation_counterfactual():
    config = load_experiment(
        "configs/padchest_counterfactual_v3_gpu.yaml",
        overrides=["workflow.early_stopping_patience=6", "workflow.early_stopping_min_delta=0.003"],
    )
    assert config.workflow.early_stopping_patience == 6
    assert config.workflow.early_stopping_min_delta == 0.003
    args = cf_run_arguments(config)
    assert args.early_stopping_patience == 6
    assert args.early_stopping_min_delta == 0.003
