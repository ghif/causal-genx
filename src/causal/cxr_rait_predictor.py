from __future__ import annotations

from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .image_parent_predictor import (
    CNNEncoder,
    _as_column,
    _normal_log_prob,
    _positive_scale,
    _set_variable_value,
)


def _conv2d(
    in_features: int,
    out_features: int,
    kernel_size: int,
    *,
    stride: int = 1,
    padding: int = 0,
    compute_dtype: jnp.dtype,
    rngs: nnx.Rngs,
) -> nnx.Conv:
    """Create the bias-free convolutions used by TorchXRayVision DenseNet."""
    return nnx.Conv(
        in_features=in_features,
        out_features=out_features,
        kernel_size=(kernel_size, kernel_size),
        strides=(stride, stride),
        padding=((padding, padding), (padding, padding)),
        use_bias=False,
        dtype=compute_dtype,
        param_dtype=jnp.float32,
        rngs=rngs,
    )


class _DenseLayer(nnx.Module):
    """TorchXRayVision's torchvision-compatible DenseNet bottleneck layer."""

    def __init__(self, in_features: int, *, growth_rate: int, compute_dtype: jnp.dtype, rngs: nnx.Rngs):
        bottleneck_features = 4 * growth_rate
        self.norm1 = nnx.BatchNorm(in_features, momentum=0.9, epsilon=1e-5, dtype=compute_dtype, param_dtype=jnp.float32, rngs=rngs)
        self.conv1 = _conv2d(in_features, bottleneck_features, 1, compute_dtype=compute_dtype, rngs=rngs)
        self.norm2 = nnx.BatchNorm(bottleneck_features, momentum=0.9, epsilon=1e-5, dtype=compute_dtype, param_dtype=jnp.float32, rngs=rngs)
        self.conv2 = _conv2d(bottleneck_features, growth_rate, 3, padding=1, compute_dtype=compute_dtype, rngs=rngs)

    def __call__(self, x):
        h = self.conv1(jax.nn.relu(self.norm1(x)))
        h = self.conv2(jax.nn.relu(self.norm2(h)))
        return jnp.concatenate((x, h), axis=-1)


class _DenseBlock(nnx.Module):
    def __init__(self, num_layers: int, in_features: int, *, growth_rate: int, compute_dtype: jnp.dtype, rngs: nnx.Rngs):
        self.layers = nnx.List()
        features = in_features
        for _ in range(num_layers):
            self.layers.append(_DenseLayer(features, growth_rate=growth_rate, compute_dtype=compute_dtype, rngs=rngs))
            features += growth_rate

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition(nnx.Module):
    def __init__(self, in_features: int, out_features: int, *, compute_dtype: jnp.dtype, rngs: nnx.Rngs):
        self.norm = nnx.BatchNorm(in_features, momentum=0.9, epsilon=1e-5, dtype=compute_dtype, param_dtype=jnp.float32, rngs=rngs)
        self.conv = _conv2d(in_features, out_features, 1, compute_dtype=compute_dtype, rngs=rngs)

    def __call__(self, x):
        x = self.conv(jax.nn.relu(self.norm(x)))
        return jax.lax.reduce_window(x, 0.0, jax.lax.add, (1, 2, 2, 1), (1, 2, 2, 1), "VALID") / 4.0


class TorchXRayVisionDenseNet121(nnx.Module):
    """DenseNet-121 feature extractor with the exact TorchXRayVision layout."""

    output_features = 1024

    def __init__(self, *, compute_dtype: jnp.dtype, rngs: nnx.Rngs):
        self.conv0 = _conv2d(1, 64, 7, stride=2, padding=3, compute_dtype=compute_dtype, rngs=rngs)
        self.norm0 = nnx.BatchNorm(64, momentum=0.9, epsilon=1e-5, dtype=compute_dtype, param_dtype=jnp.float32, rngs=rngs)
        self.denseblock1 = _DenseBlock(6, 64, growth_rate=32, compute_dtype=compute_dtype, rngs=rngs)
        self.transition1 = _Transition(256, 128, compute_dtype=compute_dtype, rngs=rngs)
        self.denseblock2 = _DenseBlock(12, 128, growth_rate=32, compute_dtype=compute_dtype, rngs=rngs)
        self.transition2 = _Transition(512, 256, compute_dtype=compute_dtype, rngs=rngs)
        self.denseblock3 = _DenseBlock(24, 256, growth_rate=32, compute_dtype=compute_dtype, rngs=rngs)
        self.transition3 = _Transition(1024, 512, compute_dtype=compute_dtype, rngs=rngs)
        self.denseblock4 = _DenseBlock(16, 512, growth_rate=32, compute_dtype=compute_dtype, rngs=rngs)
        self.norm5 = nnx.BatchNorm(1024, momentum=0.9, epsilon=1e-5, dtype=compute_dtype, param_dtype=jnp.float32, rngs=rngs)

    def __call__(self, x):
        x = jnp.asarray(x, dtype=self.conv0.dtype)
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D image batch, got shape {x.shape}")
        if x.shape[1] == 1:
            x = jnp.transpose(x, (0, 2, 3, 1))
        elif x.shape[-1] != 1:
            raise ValueError(f"TorchXRayVision DenseNet-121 requires one-channel images, got shape {x.shape}")
        x = self.conv0(x)
        x = jax.nn.relu(self.norm0(x))
        x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 2, 2, 1), ((0, 0), (1, 1), (1, 1), (0, 0)))
        x = self.transition1(self.denseblock1(x))
        x = self.transition2(self.denseblock2(x))
        x = self.transition3(self.denseblock3(x))
        x = self.denseblock4(x)
        return jax.nn.relu(self.norm5(x)).mean(axis=(1, 2))


def _load_torchxrayvision_weights(backbone: TorchXRayVisionDenseNet121, weights_path: str) -> None:
    """Load a converted archive by its PyTorch state-dict names, strictly."""
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"TorchXRayVision DenseNet-121 weights were requested but not found: {path}. "
            "Create them with scripts/convert_cxr_weights.py or set workflow.pretrained_weights_path."
        )
    with np.load(path) as archive:
        weights = {key: archive[key] for key in archive.files}

    def assign(variable, key: str) -> None:
        if key not in weights:
            raise ValueError(f"Converted TorchXRayVision archive is missing required tensor {key!r}")
        value = weights[key]
        if value.shape != variable.value.shape:
            raise ValueError(f"Tensor shape mismatch for {key}: archive={value.shape}, model={variable.value.shape}")
        _set_variable_value(variable, jnp.asarray(value, dtype=variable.value.dtype))

    def assign_bn(module: nnx.BatchNorm, prefix: str) -> None:
        assign(module.scale, f"{prefix}.weight")
        assign(module.bias, f"{prefix}.bias")
        assign(module.mean, f"{prefix}.running_mean")
        assign(module.var, f"{prefix}.running_var")

    assign(backbone.conv0.kernel, "features.conv0.weight")
    assign_bn(backbone.norm0, "features.norm0")
    for block_index, block in enumerate((backbone.denseblock1, backbone.denseblock2, backbone.denseblock3, backbone.denseblock4), start=1):
        for layer_index, layer in enumerate(block.layers, start=1):
            prefix = f"features.denseblock{block_index}.denselayer{layer_index}"
            assign_bn(layer.norm1, f"{prefix}.norm1")
            assign(layer.conv1.kernel, f"{prefix}.conv1.weight")
            assign_bn(layer.norm2, f"{prefix}.norm2")
            assign(layer.conv2.kernel, f"{prefix}.conv2.weight")
        if block_index < 4:
            transition = (backbone.transition1, backbone.transition2, backbone.transition3)[block_index - 1]
            assign_bn(transition.norm, f"features.transition{block_index}.norm")
            assign(transition.conv.kernel, f"features.transition{block_index}.conv.weight")
    assign_bn(backbone.norm5, "features.norm5")


class CxrRaitSupAuxPredictor(nnx.Module):
    """Anticausal image parent predictor for CXR-RAIT chest X-rays.
    
    Predicts parents: age (continuous), gender (categorical 2D), and tb_status (categorical 2D).
    """

    variables = {
        "age": "continuous",
        "gender": "categorical",
        "tb_status": "categorical",
    }

    def __init__(
        self,
        input_channels: int = 1,
        input_res: int = 128,
        width: int = 16,
        std_fixed: float = 0.0,
        compute_dtype: jnp.dtype = jnp.float32,
        rngs: Optional[nnx.Rngs] = None,
    ):
        rngs = rngs or nnx.Rngs(0)
        self.input_channels = int(input_channels)
        self.input_res = int(input_res)
        self.width = int(width)
        self.std_fixed = float(std_fixed)
        self.compute_dtype = compute_dtype
        input_shape = (self.input_channels, self.input_res, self.input_res)

        # Age encoder (predicts mean and log-scale)
        self.encoder_a = CNNEncoder(
            input_shape,
            width=self.width,
            num_outputs=2,
            context_dim=0,
            compute_dtype=self.compute_dtype,
            rngs=rngs,
        )

        # Gender encoder (predicts 2-class logits)
        self.encoder_g = CNNEncoder(
            input_shape,
            width=self.width,
            num_outputs=2,
            context_dim=0,
            compute_dtype=self.compute_dtype,
            rngs=rngs,
        )

        # TB status encoder (predicts 2-class logits)
        self.encoder_tb = CNNEncoder(
            input_shape,
            width=self.width,
            num_outputs=2,
            context_dim=0,
            compute_dtype=self.compute_dtype,
            rngs=rngs,
        )

    def _age_params(self, x):
        loc, logscale = jnp.split(self.encoder_a(x), 2, axis=-1)
        return jnp.tanh(loc.astype(jnp.float32)), logscale.astype(jnp.float32)

    def _gender_logits(self, x):
        return self.encoder_g(x).astype(jnp.float32)

    def _tb_logits(self, x):
        return self.encoder_tb(x).astype(jnp.float32)

    def predict(self, *, x, **_):
        a_loc, _ = self._age_params(x)
        gender_logits = self._gender_logits(x)
        tb_logits = self._tb_logits(x)
        return {
            "age": a_loc,
            "gender": jax.nn.softmax(gender_logits, axis=-1),
            "tb_status": jax.nn.softmax(tb_logits, axis=-1),
        }

    def anticausal_log_probs(self, *, x, age, gender, tb_status, **_):
        a_loc, a_logscale = self._age_params(x)
        gender_logits = self._gender_logits(x)
        tb_logits = self._tb_logits(x)

        a_scale = _positive_scale(a_logscale, self.std_fixed)
        age = _as_column(age)

        age_log_prob = jnp.sum(
            _normal_log_prob((age - a_loc) / a_scale) - jnp.log(a_scale),
            axis=-1,
        )
        gender_log_prob = jnp.sum(
            jnp.asarray(gender, dtype=jnp.float32)
            * jax.nn.log_softmax(gender_logits, axis=-1),
            axis=-1,
        )
        tb_log_prob = jnp.sum(
            jnp.asarray(tb_status, dtype=jnp.float32)
            * jax.nn.log_softmax(tb_logits, axis=-1),
            axis=-1,
        )
        joint = age_log_prob + gender_log_prob + tb_log_prob
        return {
            "age_aux": age_log_prob,
            "gender_aux": gender_log_prob,
            "tb_status_aux": tb_log_prob,
            "joint": joint,
        }

    def model_anticausal(self, **obs):
        return self.anticausal_log_probs(**obs)

    def svi_model(self, **obs):
        return self.model_anticausal(**obs)

    def guide_pass(self, **obs):
        del obs
        return None


class CxrRaitPretrainedPredictor(nnx.Module):
    """Anticausal predictor for CXR-RAIT chest X-rays using pre-trained CXR weights converted to Flax."""

    variables = {
        "age": "continuous",
        "gender": "categorical",
        "tb_status": "categorical",
    }

    def __init__(
        self,
        input_channels: int = 1,
        input_res: int = 128,
        width: int = 32,
        std_fixed: float = 0.0,
        freeze_backbone: bool = True,
        weights_path: str = "checkpoints/pretrained/torchxrayvision_densenet121_flax.npz",

        dropout_rate: float = 0.0,
        label_smoothing: float = 0.0,
        compute_dtype: jnp.dtype = jnp.float32,
        rngs: Optional[nnx.Rngs] = None,
    ):
        rngs = rngs or nnx.Rngs(0)
        if input_channels != 1:
            raise ValueError("CxrRaitPretrainedPredictor requires input_channels=1 for TorchXRayVision DenseNet-121")
        self.input_channels = int(input_channels)
        self.input_res = int(input_res)
        self.width = int(width)  # Retained for checkpoint/config compatibility; DenseNet-121 has fixed widths.
        self.std_fixed = float(std_fixed)
        self.freeze_backbone = bool(freeze_backbone)
        self.dropout_rate = float(dropout_rate)
        self.label_smoothing = float(label_smoothing)
        self.compute_dtype = compute_dtype
        # Base feature extractor mirrors torchxrayvision.models.DenseNet
        # (DenseNet-121: growth_rate=32, block_config=(6, 12, 24, 16)).
        # Its converted state-dict archive is loaded immediately and strictly.
        self.encoder_shared = TorchXRayVisionDenseNet121(compute_dtype=self.compute_dtype, rngs=rngs)
        _load_torchxrayvision_weights(self.encoder_shared, weights_path)

        if self.dropout_rate > 0.0:
            self.dropout = nnx.Dropout(self.dropout_rate, rngs=rngs)

        # Prediction Heads conditioned on shared representation
        self.head_age = nnx.Linear(self.encoder_shared.output_features, 2, rngs=rngs)
        self.head_gender = nnx.Linear(self.encoder_shared.output_features, 2, rngs=rngs)
        self.head_tb = nnx.Linear(self.encoder_shared.output_features, 2, rngs=rngs)

    def _extract_features(self, x):
        h = self.encoder_shared(x)
        if getattr(self, "freeze_backbone", True):
            h = jax.lax.stop_gradient(h)
        if hasattr(self, "dropout") and self.dropout_rate > 0.0:
            h = self.dropout(h)
        return h


    def _age_params(self, x):
        h = self._extract_features(x)
        loc, logscale = jnp.split(self.head_age(h), 2, axis=-1)
        return jnp.tanh(loc.astype(jnp.float32)), logscale.astype(jnp.float32)

    def _gender_logits(self, x):
        h = self._extract_features(x)
        return self.head_gender(h).astype(jnp.float32)

    def _tb_logits(self, x):
        h = self._extract_features(x)
        return self.head_tb(h).astype(jnp.float32)

    def predict(self, *, x, **_):
        h = self._extract_features(x)
        loc, _ = jnp.split(self.head_age(h), 2, axis=-1)
        a_loc = jnp.tanh(loc.astype(jnp.float32))
        gender_logits = self.head_gender(h).astype(jnp.float32)
        tb_logits = self.head_tb(h).astype(jnp.float32)
        return {
            "age": a_loc,
            "gender": jax.nn.softmax(gender_logits, axis=-1),
            "tb_status": jax.nn.softmax(tb_logits, axis=-1),
        }

    def anticausal_log_probs(self, *, x, age, gender, tb_status, **_):
        h = self._extract_features(x)
        loc, a_logscale = jnp.split(self.head_age(h), 2, axis=-1)
        a_loc = jnp.tanh(loc.astype(jnp.float32))
        a_logscale = a_logscale.astype(jnp.float32)
        gender_logits = self.head_gender(h).astype(jnp.float32)
        tb_logits = self.head_tb(h).astype(jnp.float32)

        a_scale = _positive_scale(a_logscale, self.std_fixed)
        age = _as_column(age)

        gender_target = jnp.asarray(gender, dtype=jnp.float32)
        tb_target = jnp.asarray(tb_status, dtype=jnp.float32)
        if getattr(self, "label_smoothing", 0.0) > 0.0:
            smooth = self.label_smoothing / 2.0
            gender_target = gender_target * (1.0 - self.label_smoothing) + smooth
            tb_target = tb_target * (1.0 - self.label_smoothing) + smooth

        age_log_prob = jnp.sum(
            _normal_log_prob((age - a_loc) / a_scale) - jnp.log(a_scale),
            axis=-1,
        )
        gender_log_prob = jnp.sum(
            gender_target * jax.nn.log_softmax(gender_logits, axis=-1),
            axis=-1,
        )
        tb_log_prob = jnp.sum(
            tb_target * jax.nn.log_softmax(tb_logits, axis=-1),
            axis=-1,
        )

        joint = age_log_prob + gender_log_prob + tb_log_prob
        return {
            "age_aux": age_log_prob,
            "gender_aux": gender_log_prob,
            "tb_status_aux": tb_log_prob,
            "joint": joint,
        }

    def model_anticausal(self, **obs):
        return self.anticausal_log_probs(**obs)

    def svi_model(self, **obs):
        return self.model_anticausal(**obs)

    def guide_pass(self, **obs):
        del obs
        return None
