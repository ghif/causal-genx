"""Inference for saved image-model artifacts."""

from __future__ import annotations

from config import ExperimentConfig, InferenceConfig

from .common import SOURCE_ROOT, run_legacy_module


def run(config: ExperimentConfig) -> str:
    workflow = config.workflow
    assert isinstance(workflow, InferenceConfig)
    run_legacy_module("infer.py", ["--checkpoint", workflow.checkpoint], cwd=SOURCE_ROOT)
    return workflow.checkpoint
