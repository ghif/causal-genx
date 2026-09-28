"""One fully-resolved YAML config per experiment plus dot-path overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class CausalVariableConfig(BaseModel):
    """Typed YAML declaration for one arbitrary causal variable."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    kind: Literal["continuous", "binary", "categorical", "ordinal"]
    encoded_dim: PositiveInt = 1
    categories: tuple[str, ...] = ()
    normalization: str | None = None
    source: str | None = None
    observed: bool = True
    intervenable: bool = True

    @model_validator(mode="after")
    def validate_encoding(self):
        if self.kind in {"continuous", "binary"} and self.encoded_dim != 1:
            raise ValueError(f"{self.name}: {self.kind} variables require encoded_dim=1")
        if self.kind == "categorical" and self.encoded_dim < 2:
            raise ValueError(f"{self.name}: categorical variables require encoded_dim>=2")
        if self.categories and len(self.categories) != self.encoded_dim:
            raise ValueError(f"{self.name}: categories must match encoded_dim")
        if len(self.categories) != len(set(self.categories)):
            raise ValueError(f"{self.name}: categories must be unique")
        return self


class CausalSchemaConfig(BaseModel):
    """Typed, schema-driven causal graph section of an experiment YAML.

    String variable entries are retained for the legacy MorphoMNIST,
    CXR-RAIT, and PadChest configurations. New schemas should use the typed
    object form so kind and encoding are explicit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    version: str = "1"
    variables: list[str | CausalVariableConfig] = Field(default_factory=list)
    edges: list[tuple[str, str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph(self):
        names = [item if isinstance(item, str) else item.name for item in self.variables]
        if len(names) != len(set(names)):
            raise ValueError("causal_schema variables must have unique names")
        known = set(names)
        if any(parent not in known or child not in known for parent, child in self.edges):
            raise ValueError("causal_schema edges must reference declared variables")
        if any(parent == child for parent, child in self.edges):
            raise ValueError("causal_schema edges cannot contain self-loops")
        pending = {name: 0 for name in names}
        children = {name: [] for name in names}
        for parent, child in self.edges:
            pending[child] += 1
            children[parent].append(child)
        ready = [name for name in names if pending[name] == 0]
        visited = 0
        while ready:
            name = ready.pop()
            visited += 1
            for child in children[name]:
                pending[child] -= 1
                if pending[child] == 0:
                    ready.append(child)
        if visited != len(names):
            raise ValueError("causal_schema edges must form a directed acyclic graph")
        return self


class DatasetConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str = "morphomnist"
    root: str = "gs://medical-airnd/causal-gen/datasets/morphomnist"
    metadata: str = ""
    image_prefix: str = ""
    split_manifest: str = ""
    tb_label_mode: Literal["any_tb", "tb_only", "tb_or_sequelae"] = "tb_or_sequelae"
    min_category_count: PositiveInt = 1
    excluded_sources: list[dict[str, str] | str] = Field(default_factory=list)
    input_res: PositiveInt = 32
    pad: int = 4
    hflip: float = 0.5
    context_norm: str = "[-1,1]"
    augment: bool = True


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    accelerator: Literal["cpu", "gpu", "tpu"] = "cpu"
    precision: Literal["fp32", "bf16"] = "fp32"
    gpu_id: str | None = None
    expected_local_device_count: PositiveInt | None = None
    expected_global_device_count: PositiveInt | None = None
    expected_process_count: PositiveInt | None = None


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    name: Literal["hierarchical_vae", "simple_vae", "padchest_scm"] = "hierarchical_vae"
    context_dim: PositiveInt = 12
    cond_prior: bool = False
    enc_arch: str = "32b3d2,16b3d2,8b3d2,4b3d4,1b4"
    dec_arch: str = "1b4,4b4,8b4,16b4,32b4"
    widths: list[PositiveInt] = [16, 32, 64, 128, 256]
    bottleneck: PositiveInt = 4
    z_dim: PositiveInt = 16
    z_max_res: PositiveInt = 192
    bias_max_res: PositiveInt = 64
    x_like: str = "diag_dgauss"
    std_init: float = 0.0
    q_correction: bool = False
    kl_free_bits: float = 0.0


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
    lr_warmup_steps: int = 100
    betas: tuple[float, float] = (0.9, 0.9)


class ScmTrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["train-scm"]
    scm_model: str = "morphomnist_scm"
    strict_artifact_contract: bool = True
    epochs: PositiveInt = 1000
    speed_log_freq: PositiveInt = 50
    checkpoint_freq: PositiveInt = 1
    plot_samples: PositiveInt = 10000
    widths: list[PositiveInt] = [32, 32]
    benchmark_steps: int = 0


class PredictorTrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["train-predictor", "finetune-predictor"]
    predictor_model: str = "morphomnist_image_parent_predictor"
    epochs: PositiveInt = 1000
    speed_log_freq: PositiveInt = 50
    checkpoint_freq: PositiveInt = 1
    benchmark_steps: int = 0
    execution_mode: Literal["auto", "single_device", "replicated"] = "auto"
    drop_remainder: bool = True
    freeze_backbone: bool = True
    backbone_lr_scale: float = 1.0
    pretrained_weights_path: str = "gs://cxr-rait/checkpoints/pretrained/torchxrayvision_densenet121_flax.npz"
    warmup_epochs: int = 0
    label_smoothing: float = 0.0
    torchxray_preprocessing: bool = False
    dropout_rate: float = 0.0
    # Predictor input pipeline controls for remote image datasets. Defaults keep
    # startup diagnostics image-free and enable only bounded, opt-in/auto caches.
    normalization_sample_size: int = 0
    input_cache: Literal["auto", "off", "on"] = "auto"
    input_cache_dir: str = ""
    input_cache_max_items: int = 2048
    input_prefetch_workers: int = 8
    input_prefetch_batches: int = 2
    input_stage_mode: Literal["auto", "off", "require"] = "auto"
    input_stage_dir: str = ""
    input_stage_manifest: str = ""
    input_stage_max_items: int = 0
    input_stage_max_bytes: int = 0
    input_stage_size_sample_items: int = 256
    input_stage_workers: int = 16


class ImageModelTrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["train-image-model"]
    epochs: PositiveInt = 5000
    speed_log_freq: PositiveInt = 50
    viz_batch_size: PositiveInt = 32
    eval_freq: PositiveInt = 5
    checkpoint_freq: PositiveInt = 1
    resume: str = ""
    ema_rate: float = 0.999
    beta: float = 1.0
    beta_warmup_steps: int = 0
    grad_clip: float = 350.0
    grad_skip: float = 500.0
    accu_steps: PositiveInt = 1
    checkpoint_smoke_test: bool = False
    checkpoint_smoke_steps: PositiveInt = 1
    benchmark_steps: int = 0
    benchmark_warmup_steps: int = 20
    execution_mode: Literal["auto", "single_device", "replicated"] = "auto"
    drop_remainder: bool = False
    input_stage_mode: Literal["auto", "off", "require"] = "auto"
    input_stage_dir: str = ""
    input_stage_manifest: str = ""
    input_stage_max_items: int = 0
    input_stage_max_bytes: int = 0
    input_stage_size_sample_items: int = 256
    input_stage_workers: int = 16


class CounterfactualTrainingConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        protected_namespaces=(),
    )
    type: Literal["finetune-counterfactual"]
    scm_checkpoint: str
    predictor_checkpoint: str
    image_model_checkpoint: str
    epochs: PositiveInt = 5000
    speed_log_freq: PositiveInt = 50
    checkpoint_freq: PositiveInt = 1
    eval_freq: PositiveInt = 1
    execution_mode: Literal["auto", "single_device", "replicated"] = "auto"
    drop_remainder: bool = False
    alpha: float = 0.1
    lmbda_init: float = 0.0
    lr_lagrange: float = 1e-2
    damping: float = 100.0
    do_pa: str | None = None
    cf_particles: PositiveInt = 1
    elbo_constraint: float = 1.841216802597046
    ema_rate: float = 0.999
    model_validation_batches: int = 1
    final_eval_full: bool = False
    trust_incomplete_checkpoint: bool = False
    resume_checkpoint: str = ""
    testing: bool = False
    benchmark_steps: int = 0
    input_prefetch_workers: int = 1
    input_prefetch_batches: int = 0
    input_stage_mode: Literal["auto", "off", "require"] = "auto"
    input_stage_dir: str = ""
    input_stage_manifest: str = ""
    input_stage_max_items: int = 0
    input_stage_max_bytes: int = 0
    input_stage_size_sample_items: int = 256
    input_stage_workers: int = 16


class InferenceConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["infer"]
    checkpoint: str
    image_path: str = ""
    parents: dict[str, Any] = Field(default_factory=lambda: {"thickness": 0.0, "intensity": 0.0, "digit": 0})
    beta: float = 1.0
    num_samples: PositiveInt = 1
    latent_temperature: float = 1.0
    output_dir: str = ""
    trust_incomplete_checkpoint: bool = False


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
    causal_schema: CausalSchemaConfig | None = None
    optimizer: OptimizerConfig
    workflow: WorkflowConfig

    @model_validator(mode="after")
    def validate_typed_context(self):
        if self.causal_schema and self.causal_schema.variables and all(
            isinstance(item, CausalVariableConfig) for item in self.causal_schema.variables
        ):
            encoded_dim = sum(int(item.encoded_dim) for item in self.causal_schema.variables)
            if self.model.context_dim != encoded_dim:
                raise ValueError(
                    f"model.context_dim={self.model.context_dim} does not match causal_schema encoded_dim={encoded_dim}"
                )
        return self


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


def resolve_causal_schema(config: ExperimentConfig, fallback):
    """Resolve a typed YAML schema, retaining legacy dataset defaults.

    Existing configs list variable names only; their dataset providers remain
    authoritative for kinds and encodings. A fully typed declaration may
    describe any supported mixed schema and is converted to the shared
    ``CausalGraphSpec`` contract.
    """
    from contracts import CausalGraphSpec, VariableKind, VariableSpec

    declared = config.causal_schema
    if declared is None or not declared.variables:
        return fallback
    if all(isinstance(item, str) for item in declared.variables):
        names = tuple(declared.variables)
        if names == fallback.variable_names:
            return CausalGraphSpec(
                dataset_id=fallback.dataset_id,
                variables=fallback.variables,
                edges=tuple(declared.edges) if declared.edges else fallback.edges,
                version=declared.version,
            )
        # PadChest v2 has a stable shorthand because its six encodings are a
        # published dataset contract. Other arbitrary schemas must be typed.
        if config.dataset.name == "padchest":
            from data.padchest import PAD_CHEST_V2_SCHEMA
            if names == PAD_CHEST_V2_SCHEMA.variable_names:
                return CausalGraphSpec(
                    dataset_id=PAD_CHEST_V2_SCHEMA.dataset_id,
                    variables=PAD_CHEST_V2_SCHEMA.variables,
                    edges=tuple(declared.edges) if declared.edges else PAD_CHEST_V2_SCHEMA.edges,
                    version=declared.version,
                )
        raise ValueError(
            "String causal_schema variables are only supported for a legacy dataset schema or PadChest v2; "
            "declare kind and encoded_dim for arbitrary variables"
        )
    if any(isinstance(item, str) for item in declared.variables):
        raise ValueError("causal_schema.variables must use either all names or all typed objects")
    variables = tuple(
        VariableSpec(
            item.name,
            VariableKind(item.kind),
            encoded_dim=int(item.encoded_dim),
            categories=tuple(item.categories),
            normalization=item.normalization,
            source=item.source,
            observed=item.observed,
            intervenable=item.intervenable,
        )
        for item in declared.variables
    )
    return CausalGraphSpec(
        dataset_id=config.dataset.name,
        variables=variables,
        edges=tuple(declared.edges),
        version=declared.version,
    )
