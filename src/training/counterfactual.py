"""Stage 4: fine-tune the image mechanism for counterfactual generation."""

from __future__ import annotations

from config import CounterfactualTrainingConfig, ExperimentConfig

from .common import SOURCE_ROOT, legacy_run_dir, run_legacy_module, validate_stage_artifacts


def run(config: ExperimentConfig) -> str:
    workflow = config.workflow
    assert isinstance(workflow, CounterfactualTrainingConfig)
    validate_stage_artifacts(workflow.scm_checkpoint, workflow.predictor_checkpoint, workflow.image_model_checkpoint)
    arguments = ["--dataset", config.dataset.name, "--data_dir", config.dataset.root, "--ckpt_dir", "../../checkpoints", "--exp_name", config.artifacts.run_name, "--accelerator", config.runtime.accelerator, "--precision", config.runtime.precision, "--seed", str(config.seed), "--pgm_path", workflow.scm_checkpoint, "--predictor_path", workflow.predictor_checkpoint, "--vae_path", workflow.image_model_checkpoint]
    run_legacy_module("train_cf.py", arguments, cwd=SOURCE_ROOT / "pgm")
    return str(legacy_run_dir(config))
