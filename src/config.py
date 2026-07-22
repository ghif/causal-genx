"""One fully-resolved YAML config per experiment plus dot-path overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class DatasetConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str = "morphomnist"
    root: str = "gs://medical-airnd/causal-gen/datasets/morphomnist"
    input_res: PositiveInt = 32
    pad: int = 4
    context_norm: str = "[-1,1]"


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    accelerator: Literal["cpu", "gpu", "tpu"] = "cpu"
    precision: Literal["fp32", "bf16"] = "fp32"
    gpu_id: str | None = None


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    name: Literal["hierarchical_vae", "simple_vae"] = "hierarchical_vae"
    context_dim: PositiveInt = 12


class ArtifactConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    root: str = "checkpoints"
    run_name: str = "run"
    remote_root: str = ""


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    lr: float
    weight_decay: float
    batch_size: PositiveInt


class ScmTrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["train-scm"]
    scm_model: str = "morphomnist_scm"
    strict_legacy_parity: bool = True
    epochs: PositiveInt = 1000
    eval_freq: PositiveInt = 1
    plot_samples: PositiveInt = 10000
    widths: list[PositiveInt] = [32, 32]
    benchmark_steps: int = 0


class PredictorTrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["train-predictor"]
    predictor_model: str = "morphomnist_image_parent_predictor"
    epochs: PositiveInt = 1000


class ImageModelTrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["train-image-model"]
    epochs: PositiveInt = 5000
    speed_log_freq: PositiveInt = 50
    eval_freq: PositiveInt = 5
    checkpoint_freq: PositiveInt = 1
    benchmark_steps: int = 0
    benchmark_warmup_steps: int = 20


class CounterfactualTrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["finetune-counterfactual"]
    scm_checkpoint: str
    predictor_checkpoint: str
    image_model_checkpoint: str


class InferenceConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["infer"]
    checkpoint: str


WorkflowConfig = Annotated[
    Union[ScmTrainingConfig, PredictorTrainingConfig, ImageModelTrainingConfig, CounterfactualTrainingConfig, InferenceConfig],
    Field(discriminator="type"),
]


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    version: str = "1"
    seed: int = 7
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    causal_schema: dict[str, Any] = Field(default_factory=dict)
    optimizer: OptimizerConfig
    workflow: WorkflowConfig


def load_experiment(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if "defaults" in raw:
        raise ValueError("Experiment configs must be fully resolved; `defaults` composition is not supported.")
    raw = copy.deepcopy(raw)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Overrides must use key=value syntax, got {override!r}")
        path, value = override.split("=", 1)
        target = raw
        parts = path.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ValueError(f"Cannot override non-object config path {path!r}")
        target[parts[-1]] = yaml.safe_load(value)
    return ExperimentConfig.model_validate(raw)
