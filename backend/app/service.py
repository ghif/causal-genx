"""Model restoration and synchronous MorphoMNIST counterfactual inference."""

from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from PIL import Image

from causal.flow_scm import MorphoMNISTPGM
from causal.image_parent_predictor import MorphoMNISTSupAuxPredictor
from models.image_vae import HVAE, SimpleVAE
from utils import load_checkpoint_with_path, open_file, path_exists, postprocess


THICKNESS_RANGE = (0.87598526, 6.255515)
INTENSITY_RANGE = (66.601204, 254.90317)
DEFAULT_SCM_CHECKPOINT = "gs://medical-airnd/causal-gen/checkpoints/morphomnist/scm_jax-cpu-v2_23-07-2026/checkpoints"
DEFAULT_PREDICTOR_CHECKPOINT = "gs://medical-airnd/causal-gen/checkpoints/morphomnist/predictor_jax-tpu-v6e-4_23-07-2026/checkpoints"
DEFAULT_IMAGE_CHECKPOINT = "gs://medical-airnd/causal-gen/checkpoints/morphomnist/hvae_jax-tpu-v6e4-v2_23-07-2026/checkpoints"
DEFAULT_CF_CHECKPOINT = "gs://medical-airnd/causal-gen/checkpoints/morphomnist/cf_jax-tpu-v6e4_23-07-2026/checkpoints/15795"


class ServiceUnavailableError(RuntimeError):
    """Raised when inference is requested before model startup succeeds."""


class RequestValidationError(ValueError):
    """Raised for a valid HTTP request whose model inputs are invalid."""


@dataclass(frozen=True)
class ModelSettings:
    cf_checkpoint: str
    scm_checkpoint: str
    predictor_checkpoint: str
    image_checkpoint: str
    trust_incomplete_checkpoint: bool
    default_seed: int

    @classmethod
    def from_environment(cls) -> "ModelSettings":
        return cls(
            cf_checkpoint=os.getenv("CF_CHECKPOINT", DEFAULT_CF_CHECKPOINT),
            scm_checkpoint=os.getenv("SCM_CHECKPOINT", DEFAULT_SCM_CHECKPOINT),
            predictor_checkpoint=os.getenv("PREDICTOR_CHECKPOINT", DEFAULT_PREDICTOR_CHECKPOINT),
            image_checkpoint=os.getenv("IMAGE_MODEL_CHECKPOINT", DEFAULT_IMAGE_CHECKPOINT),
            trust_incomplete_checkpoint=os.getenv("TRUST_INCOMPLETE_CHECKPOINT", "true").lower() == "true",
            default_seed=int(os.getenv("DEFAULT_SEED", "7")),
        )


def _checkpoint_root(path: str) -> str:
    path = path.rstrip("/")
    return path.rsplit("/", 1)[0] if path.rsplit("/", 1)[-1].isdigit() else path


def _hparams(path: str) -> dict[str, Any]:
    location = f"{_checkpoint_root(path)}/hparams.json"
    if not path_exists(location):
        raise FileNotFoundError(f"Checkpoint metadata is missing: {location}")
    with open_file(location, "r") as handle:
        return json.load(handle)


def _single_device_sharding() -> jax.sharding.SingleDeviceSharding:
    return jax.sharding.SingleDeviceSharding(jax.devices()[0])


def _model_from_hparams(hparams: dict[str, Any], seed: int):
    model_class = HVAE if hparams.get("vae", "hierarchical") == "hierarchical" else SimpleVAE
    required = ("input_channels", "input_res", "enc_arch", "dec_arch", "widths", "z_dim", "context_dim")
    missing = [key for key in required if key not in hparams]
    if missing:
        raise ValueError(f"Image-model metadata is incomplete: {missing}")
    return model_class(
        input_channels=hparams["input_channels"], input_res=hparams["input_res"],
        enc_arch=hparams["enc_arch"], dec_arch=hparams["dec_arch"], widths=hparams["widths"],
        z_dim=hparams["z_dim"], context_dim=hparams["context_dim"],
        z_max_res=hparams.get("z_max_res", 192), bottleneck=hparams.get("bottleneck", 4),
        cond_prior=hparams.get("cond_prior", False), q_correction=hparams.get("q_correction", False),
        bias_max_res=hparams.get("bias_max_res", 64), x_like=hparams.get("x_like", "diag_dgauss"),
        kl_free_bits=hparams.get("kl_free_bits", 0.0), std_init=hparams.get("std_init", 0.0),
        dataset_id=hparams.get("dataset", "morphomnist"), rngs=nnx.Rngs(seed),
    )


def _restore(path: str, template: dict[str, Any], *, allow_incomplete: bool) -> tuple[dict[str, Any], str]:
    return load_checkpoint_with_path(
        path, template=template, fallback_sharding=_single_device_sharding(),
        allow_incomplete=allow_incomplete, partial_restore=True,
    )


def _normalise(value: float, limits: tuple[float, float]) -> float:
    low, high = limits
    return (2.0 * (value - low) / (high - low)) - 1.0


def _denormalise(value: float, limits: tuple[float, float]) -> float:
    low, high = limits
    return ((value + 1.0) * 0.5 * (high - low)) + low


def preprocess_image(data: bytes) -> jax.Array:
    """Apply the deterministic MorphoMNIST evaluation transform to an upload."""
    try:
        image = Image.open(io.BytesIO(data)).convert("L")
        image.load()
    except Exception as exc:
        raise RequestValidationError("image must be a readable PNG or JPEG") from exc
    if image.size == (28, 28):
        padded = Image.new("L", (32, 32), color=0)
        padded.paste(image, (2, 2))
        image = padded
    elif image.size != (32, 32):
        raise RequestValidationError("image must be 28x28 or 32x32 pixels")
    values = np.asarray(image, dtype=np.float32)[None, ..., None]
    return jnp.asarray((values - 127.5) / 127.5)


def _encode_png(image: jax.Array) -> str:
    values = postprocess(np.asarray(image[0]))
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    buffer = io.BytesIO()
    Image.fromarray(values, mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class CounterfactualModels:
    """Immutable models and the request-serialised counterfactual operation."""

    def __init__(self, settings: ModelSettings):
        self.settings = settings
        self._lock = threading.Lock()
        self.ready = False
        self.error: str | None = None
        self.vae = None
        self.pgm = None
        self.predictor = None
        self.input_res = 32
        self.resolved_cf_checkpoint = settings.cf_checkpoint

    def load(self) -> None:
        try:
            image_hparams = _hparams(self.settings.image_checkpoint)
            vae = _model_from_hparams(image_hparams, self.settings.default_seed)
            vae_graphdef, vae_params = nnx.split(vae, nnx.Param)
            cf_checkpoint, self.resolved_cf_checkpoint = _restore(
                self.settings.cf_checkpoint, {"ema_params": vae_params.to_pure_dict()},
                allow_incomplete=self.settings.trust_incomplete_checkpoint,
            )
            if "ema_params" not in cf_checkpoint:
                raise ValueError("Counterfactual checkpoint has no ema_params")
            self.vae = nnx.merge(vae_graphdef, cf_checkpoint["ema_params"])
            self.vae.eval()
            self.input_res = int(image_hparams["input_res"])

            scm_hparams = _hparams(self.settings.scm_checkpoint)
            if scm_hparams.get("setup") != "sup_pgm":
                raise ValueError("Configured SCM checkpoint is not a sup_pgm artifact")
            pgm = MorphoMNISTPGM(widths=scm_hparams.get("widths", [32, 32]), rngs=nnx.Rngs(self.settings.default_seed))
            pgm_graphdef, pgm_params = nnx.split(pgm, nnx.Param)
            pgm_checkpoint, _ = _restore(
                self.settings.scm_checkpoint, {"ema_params": pgm_params.to_pure_dict(), "format_version": 0},
                allow_incomplete=self.settings.trust_incomplete_checkpoint,
            )
            if pgm_checkpoint.get("format_version") != 2 or "ema_params" not in pgm_checkpoint:
                raise ValueError("Configured SCM checkpoint is not the supported format_version=2 artifact")
            self.pgm = nnx.merge(pgm_graphdef, pgm_checkpoint["ema_params"])
            self.pgm.eval()

            predictor_hparams = _hparams(self.settings.predictor_checkpoint)
            if predictor_hparams.get("setup") != "sup_aux":
                raise ValueError("Configured predictor checkpoint is not a sup_aux artifact")
            predictor = MorphoMNISTSupAuxPredictor(
                input_channels=predictor_hparams.get("input_channels", image_hparams["input_channels"]),
                input_res=predictor_hparams.get("input_res", self.input_res),
                width=predictor_hparams.get("width", 8), std_fixed=predictor_hparams.get("std_fixed", 0.0),
                rngs=nnx.Rngs(self.settings.default_seed),
            )
            predictor_graphdef, predictor_params, predictor_stats = nnx.split(predictor, nnx.Param, nnx.BatchStat)
            predictor_checkpoint, _ = _restore(
                self.settings.predictor_checkpoint,
                {"ema_params": predictor_params.to_pure_dict(), "ema_batch_stats": predictor_stats.to_pure_dict(), "format_version": 0},
                allow_incomplete=self.settings.trust_incomplete_checkpoint,
            )
            if predictor_checkpoint.get("format_version") != 3 or "ema_params" not in predictor_checkpoint:
                raise ValueError("Configured predictor checkpoint is not the supported format_version=3 artifact")
            self.predictor = nnx.merge(
                predictor_graphdef, predictor_checkpoint["ema_params"], predictor_checkpoint["ema_batch_stats"]
            )
            self.predictor.eval()
            self.ready = True
            self.error = None
        except Exception as exc:
            self.ready = False
            self.error = str(exc)
            raise

    def _predict_factual_parents(self, image: jax.Array) -> dict[str, jax.Array]:
        assert self.predictor is not None
        initial = self.predictor.predict(x=image, intensity=jnp.zeros((1, 1), dtype=jnp.float32))
        predicted = self.predictor.predict(x=image, intensity=initial["intensity"])
        digit = jax.nn.one_hot(jnp.argmax(predicted["digit"], axis=-1), 10, dtype=jnp.float32)
        return {"thickness": predicted["thickness"], "intensity": predicted["intensity"], "digit": digit}

    def _require_ready(self) -> None:
        if not self.ready or self.vae is None or self.pgm is None or self.predictor is None:
            raise ServiceUnavailableError(self.error or "models are still loading")

    def _validate_factors(self, digit: int, thickness: float, intensity: float) -> None:
        if not 0 <= digit <= 9:
            raise RequestValidationError("digit must be an integer from 0 through 9")
        if not THICKNESS_RANGE[0] <= thickness <= THICKNESS_RANGE[1]:
            raise RequestValidationError(f"thickness must be within {THICKNESS_RANGE[0]} through {THICKNESS_RANGE[1]}")
        if not INTENSITY_RANGE[0] <= intensity <= INTENSITY_RANGE[1]:
            raise RequestValidationError(f"intensity must be within {INTENSITY_RANGE[0]} through {INTENSITY_RANGE[1]}")

    def _parent_vector(self, digit: int, thickness: float, intensity: float) -> jax.Array:
        self._validate_factors(digit, thickness, intensity)
        continuous = jnp.asarray(
            [[_normalise(thickness, THICKNESS_RANGE), _normalise(intensity, INTENSITY_RANGE)]],
            dtype=jnp.float32,
        )
        return jnp.concatenate(
            [continuous, jax.nn.one_hot(jnp.asarray([digit]), 10, dtype=jnp.float32)], axis=-1
        )

    def _maps(self, parents: jax.Array) -> jax.Array:
        return jnp.broadcast_to(
            parents[:, None, None, :], (1, self.input_res, self.input_res, parents.shape[-1])
        )

    def predict_parents(self, image_data: bytes) -> dict[str, Any]:
        self._require_ready()
        started = time.perf_counter()
        image = preprocess_image(image_data)
        with self._lock:
            factual = self._predict_factual_parents(image)
            jax.tree.leaves(factual)[0].block_until_ready()
        return {
            "model_version": self.resolved_cf_checkpoint,
            "factual_parents": _parent_response(factual),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def generate_from_sliders(self, digit: int, thickness: float, intensity: float, style_seed: int | None = None) -> dict[str, Any]:
        self._require_ready()
        if style_seed is not None and style_seed < 0:
            raise RequestValidationError("style_seed must be non-negative")
        selected_seed = self.settings.default_seed if style_seed is None else style_seed
        started = time.perf_counter()
        parents = self._parent_vector(digit, thickness, intensity)
        with self._lock:
            image, _ = self.vae.sample(self._maps(parents), return_loc=True, rng=jax.random.PRNGKey(selected_seed))
            image.block_until_ready()
        return {
            "model_version": self.resolved_cf_checkpoint,
            "style_seed": selected_seed,
            "generated_parents": _parent_response({
                "thickness": parents[:, 0:1], "intensity": parents[:, 1:2], "digit": parents[:, 2:]
            }),
            "image_png_base64": _encode_png(image),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def linked_intensity(self, thickness: float, image_data: bytes | None = None) -> dict[str, Any]:
        self._require_ready()
        if not THICKNESS_RANGE[0] <= thickness <= THICKNESS_RANGE[1]:
            raise RequestValidationError(f"thickness must be within {THICKNESS_RANGE[0]} through {THICKNESS_RANGE[1]}")
        started = time.perf_counter()
        normalized_thickness = jnp.asarray([[_normalise(thickness, THICKNESS_RANGE)]], dtype=jnp.float32)
        with self._lock:
            if image_data is None:
                intensity, _ = self.pgm.intensity_forward(jnp.zeros_like(normalized_thickness), normalized_thickness)
            else:
                factual = self._predict_factual_parents(preprocess_image(image_data))
                intensity = self.pgm.counterfactual(factual, {"thickness": normalized_thickness})["intensity"]
            intensity.block_until_ready()
        normalized = float(np.asarray(intensity)[0, 0])
        return {
            "model_version": self.resolved_cf_checkpoint,
            "thickness_physical": thickness,
            "intensity_normalized": normalized,
            "intensity_physical": _denormalise(normalized, INTENSITY_RANGE),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def render_counterfactual(self, image_data: bytes, digit: int, thickness: float, intensity: float, seed: int | None = None) -> dict[str, Any]:
        self._require_ready()
        if seed is not None and seed < 0:
            raise RequestValidationError("seed must be non-negative")
        selected_seed = self.settings.default_seed if seed is None else seed
        started = time.perf_counter()
        image = preprocess_image(image_data)
        target = self._parent_vector(digit, thickness, intensity)
        with self._lock:
            factual = self._predict_factual_parents(image)
            counterfactual = self.pgm.counterfactual(
                factual,
                {"digit": target[:, 2:], "thickness": target[:, 0:1], "intensity": target[:, 1:2]},
                rng=jax.random.PRNGKey(selected_seed),
            )
            factual_pa = jnp.concatenate([factual["thickness"], factual["intensity"], factual["digit"]], axis=-1)
            abduct_key, cf_key, reconstruction_key = jax.random.split(jax.random.PRNGKey(selected_seed), 3)
            latents = self.vae.abduct(image, self._maps(factual_pa), t=1.0, rng=abduct_key)
            cf_loc, cf_scale = self.vae.forward_latents(latents, self._maps(counterfactual["pa"]), rng=cf_key)
            rec_loc, rec_scale = self.vae.forward_latents(latents, self._maps(factual_pa), rng=reconstruction_key)
            generated = jnp.clip(cf_loc + cf_scale * ((image - rec_loc) / jnp.clip(rec_scale, min=1e-12)), min=-1.0, max=1.0)
            generated.block_until_ready()
        return {
            "model_version": self.resolved_cf_checkpoint,
            "seed": selected_seed,
            "target_parents": _parent_response({"thickness": target[:, 0:1], "intensity": target[:, 1:2], "digit": target[:, 2:]}),
            "factual_parents": _parent_response(factual),
            "counterfactual_parents": _parent_response(counterfactual),
            "seed_image_png_base64": _encode_png(image),
            "image_png_base64": _encode_png(generated),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def generate(self, image_data: bytes, intervention_name: str, intervention_value: float, seed: int | None = None) -> dict[str, Any]:
        self._require_ready()
        if intervention_name not in {"digit", "thickness", "intensity"}:
            raise RequestValidationError("intervention_name must be digit, thickness, or intensity")
        if intervention_name == "digit":
            if int(intervention_value) != intervention_value or not 0 <= intervention_value <= 9:
                raise RequestValidationError("digit intervention_value must be an integer from 0 through 9")
        else:
            limits = THICKNESS_RANGE if intervention_name == "thickness" else INTENSITY_RANGE
            if not limits[0] <= intervention_value <= limits[1]:
                raise RequestValidationError(f"{intervention_name} must be within {limits[0]} through {limits[1]}")
        selected_seed = self.settings.default_seed if seed is None else seed
        if selected_seed < 0:
            raise RequestValidationError("seed must be non-negative")
        started = time.perf_counter()
        image = preprocess_image(image_data)
        with self._lock:
            factual = self._predict_factual_parents(image)
            if intervention_name == "digit":
                intervention = {"digit": jax.nn.one_hot(jnp.asarray([int(intervention_value)]), 10, dtype=jnp.float32)}
            else:
                limits = THICKNESS_RANGE if intervention_name == "thickness" else INTENSITY_RANGE
                intervention = {intervention_name: jnp.asarray([[_normalise(intervention_value, limits)]], dtype=jnp.float32)}
            counterfactual = self.pgm.counterfactual(factual, intervention, rng=jax.random.PRNGKey(selected_seed))
            factual_pa = jnp.concatenate([factual["thickness"], factual["intensity"], factual["digit"]], axis=-1)
            cf_pa = counterfactual["pa"]
            factual_maps = jnp.broadcast_to(factual_pa[:, None, None, :], (1, self.input_res, self.input_res, factual_pa.shape[-1]))
            cf_maps = jnp.broadcast_to(cf_pa[:, None, None, :], (1, self.input_res, self.input_res, cf_pa.shape[-1]))
            abduct_key, cf_key, reconstruction_key = jax.random.split(jax.random.PRNGKey(selected_seed), 3)
            latents = self.vae.abduct(image, factual_maps, t=1.0, rng=abduct_key)
            cf_loc, cf_scale = self.vae.forward_latents(latents, cf_maps, rng=cf_key)
            rec_loc, rec_scale = self.vae.forward_latents(latents, factual_maps, rng=reconstruction_key)
            residual = (image - rec_loc) / jnp.clip(rec_scale, min=1e-12)
            generated = jnp.clip(cf_loc + cf_scale * residual, min=-1.0, max=1.0)
            generated.block_until_ready()
        return {
            "model_version": self.resolved_cf_checkpoint,
            "seed": selected_seed,
            "intervention": {"name": intervention_name, "value_physical": intervention_value},
            "factual_parents": _parent_response(factual),
            "counterfactual_parents": _parent_response(counterfactual),
            "image_png_base64": _encode_png(generated),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }


def _parent_response(parents: dict[str, jax.Array]) -> dict[str, dict[str, float | int]]:
    thickness = float(np.asarray(parents["thickness"])[0, 0])
    intensity = float(np.asarray(parents["intensity"])[0, 0])
    digit = int(np.asarray(parents["digit"])[0].argmax())
    return {
        "normalized": {"thickness": thickness, "intensity": intensity, "digit": digit},
        "physical": {
            "thickness": _denormalise(thickness, THICKNESS_RANGE),
            "intensity": _denormalise(intensity, INTENSITY_RANGE),
            "digit": digit,
        },
    }
