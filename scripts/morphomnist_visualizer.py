#!/usr/bin/env python3
"""Interactive MorphoMNIST causal visualizer backed entirely by JAX.

Launch from the ``causal-genx`` repository root:

    python scripts/morphomnist_visualizer.py

The default checkpoint is the newest Orbax step found in the requested GCS
checkpoint root on 18 July 2026.  A different counterfactual checkpoint root or
numeric step may be supplied with ``--checkpoint``.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import inspect
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple


CACHE_ROOT = Path(tempfile.gettempdir()) / "causal-genx-cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(CACHE_ROOT / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(CACHE_ROOT / "xdg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for path in (str(SRC_DIR),):
    if path not in sys.path:
        sys.path.insert(0, path)

# Select the JAX backend before importing JAX itself.
from runtime import configure_backend_from_argv  # noqa: E402

configure_backend_from_argv()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import nnx  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

try:  # Gradio 4 imports HfFolder, which was removed in newer hub releases.
    import huggingface_hub as _huggingface_hub  # noqa: E402

    if not hasattr(_huggingface_hub, "HfFolder"):
        class _HfFolder:
            @staticmethod
            def get_token() -> str | None:
                return None

            @staticmethod
            def save_token(token: str) -> None:
                del token

            @staticmethod
            def delete_token() -> None:
                return None

        _huggingface_hub.HfFolder = _HfFolder  # type: ignore[attr-defined]
except Exception:
    pass

import gradio as gr  # noqa: E402


def patch_gradio_asyncio_compatibility() -> None:
    """Keep Gradio 4 queues usable after Python 3.13's event-loop change."""
    if sys.version_info < (3, 13):
        return

    from gradio import queueing, utils as gradio_utils

    def compatible_lock() -> asyncio.Lock:
        return asyncio.Lock()

    def compatible_stop_event() -> asyncio.Event:
        return asyncio.Event()

    # Gradio 4.44 calls get_event_loop() first; Python 3.13 raises when no loop
    # is set and Gradio consequently stores None for both synchronization
    # primitives. Modern asyncio primitives bind lazily when first awaited.
    gradio_utils.safe_get_lock = compatible_lock
    gradio_utils.safe_get_stop_event = compatible_stop_event
    queueing.safe_get_lock = compatible_lock


patch_gradio_asyncio_compatibility()

from models.image_vae import HVAE, SimpleVAE  # noqa: E402
from causal.flow_scm import MorphoMNISTPGM  # noqa: E402
from causal.image_parent_predictor import MorphoMNISTSupAuxPredictor  # noqa: E402
from utils import load_checkpoint_with_path, materialize_nnx, open_file, seed_all  # noqa: E402


DEFAULT_CHECKPOINT = (
    "gs://medical-airnd/causal-gen/checkpoints/morphomnist/"
    "cf_jax-gpu-g4_17-07-2026/checkpoints/15444"
)

MORPHO_MIN_MAX = {
    "thickness": (0.87598526, 6.255515),
    "intensity": (66.601204, 254.90317),
}

APP_CSS = """
#generated-preview img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
    image-rendering: pixelated;
}

#original-preview img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
    image-rendering: pixelated;
}
#counterfactual-preview img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
    image-rendering: pixelated;
}
"""


def patch_gradio_runtime_compatibility() -> None:
    """Bridge Gradio 4 APIs to newer Pydantic and Starlette releases."""
    try:
        from gradio_client import utils as client_utils
    except ImportError:
        client_utils = None

    if client_utils is not None:
        converter = getattr(client_utils, "_json_schema_to_python_type", None)
        if converter is not None and not getattr(converter, "_causal_genx_compatible", False):
            def compatible_converter(schema, defs):
                if isinstance(schema, bool):
                    return "Any"
                return converter(schema, defs)

            compatible_converter._causal_genx_compatible = True
            client_utils._json_schema_to_python_type = compatible_converter

    try:
        from starlette.templating import Jinja2Templates
    except ImportError:
        return

    template_response = Jinja2Templates.TemplateResponse
    if getattr(template_response, "_causal_genx_compatible", False):
        return

    parameters = list(inspect.signature(template_response).parameters)
    if len(parameters) < 2 or parameters[1] != "request":
        # Starlette <1.0 already accepts Gradio 4's (name, context) call.
        return

    def compatible_template_response(self, *args, **kwargs):
        if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], dict):
            name, context = args[:2]
            return template_response(
                self, context.get("request"), name, context, *args[2:], **kwargs
            )
        return template_response(self, *args, **kwargs)

    compatible_template_response._causal_genx_compatible = True
    Jinja2Templates.TemplateResponse = compatible_template_response


patch_gradio_runtime_compatibility()


@dataclass
class VisualizerBundle:
    args: SimpleNamespace
    vae: Any
    pgm: Any
    predictor: Any
    checkpoint_path: str


def normalize_value(value: float, key: str) -> float:
    min_v, max_v = MORPHO_MIN_MAX[key]
    if max_v <= min_v:
        return 0.0
    normalized = ((float(value) - min_v) / (max_v - min_v)) * 2.0 - 1.0
    return float(np.clip(normalized, -1.0, 1.0))


def denormalize_value(value: float, key: str) -> float:
    min_v, max_v = MORPHO_MIN_MAX[key]
    scaled = ((float(value) + 1.0) / 2.0) * (max_v - min_v) + min_v
    return float(np.clip(scaled, min_v, max_v))


def array_to_pil(x: Any) -> Image.Image:
    array = np.asarray(jax.device_get(x))
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    elif array.ndim == 3:
        array = array[0]
    array = np.rint((np.clip(array, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)
    return Image.fromarray(array, mode="L")


def preprocess_seed_image(image: Image.Image, input_res: int) -> jax.Array:
    image = image.convert("L")
    if image.size != (input_res, input_res):
        if image.size != (28, 28):
            image = image.resize((28, 28), Image.Resampling.BILINEAR)
        image = ImageOps.expand(image, border=2, fill=0)
    array = np.asarray(image, dtype=np.float32)[None, ..., None]
    return jnp.asarray((array - 127.5) / 127.5, dtype=jnp.float32)


def build_parent_vector(digit: int, thickness: float, intensity: float) -> jax.Array:
    digit_oh = jax.nn.one_hot(jnp.asarray([int(digit)]), 10, dtype=jnp.float32)
    continuous = jnp.asarray(
        [[
            normalize_value(thickness, "thickness"),
            normalize_value(intensity, "intensity"),
        ]],
        dtype=jnp.float32,
    )
    return jnp.concatenate([continuous, digit_oh], axis=-1)


def parent_dict(pa: jax.Array) -> Dict[str, jax.Array]:
    vector = pa[:, 0, 0, :] if pa.ndim == 4 else pa
    return {
        "thickness": vector[:, 0:1],
        "intensity": vector[:, 1:2],
        "digit": vector[:, 2:12],
    }


def summarize_parents(pa: jax.Array | Dict[str, jax.Array]) -> Dict[str, float]:
    values = parent_dict(pa) if not isinstance(pa, dict) else pa
    digit = int(np.asarray(jax.device_get(values["digit"])).argmax(axis=-1)[0])
    thickness = float(np.asarray(jax.device_get(values["thickness"])).reshape(-1)[0])
    intensity = float(np.asarray(jax.device_get(values["intensity"])).reshape(-1)[0])
    return {
        "digit": digit,
        "thickness": round(denormalize_value(thickness, "thickness"), 3),
        "intensity": round(denormalize_value(intensity, "intensity"), 3),
    }


def spatial_parents(pa: jax.Array, input_res: int) -> jax.Array:
    if pa.ndim == 4:
        return pa
    return jnp.broadcast_to(pa[:, None, None, :], (pa.shape[0], input_res, input_res, pa.shape[-1]))


def _checkpoint_root(path: str) -> str:
    path = path.rstrip("/")
    return path.rsplit("/", 1)[0] if path.split("/")[-1].isdigit() else path


def _read_hparams(path: str) -> Dict[str, Any]:
    with open_file(f"{_checkpoint_root(path)}/hparams.json", "r") as handle:
        return json.load(handle)


def _load_checkpoint(path: str, trust_incomplete: bool) -> Tuple[Dict[str, Any], str]:
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    return load_checkpoint_with_path(
        path,
        fallback_sharding=sharding,
        allow_incomplete=trust_incomplete,
    )


def _build_vae(hparams: Dict[str, Any], seed: int):
    model_cls = HVAE if hparams.get("vae", "hierarchical") == "hierarchical" else SimpleVAE
    return model_cls(
        input_channels=hparams["input_channels"],
        input_res=hparams["input_res"],
        enc_arch=hparams["enc_arch"],
        dec_arch=hparams["dec_arch"],
        widths=hparams["widths"],
        z_dim=hparams["z_dim"],
        context_dim=hparams["context_dim"],
        z_max_res=hparams["z_max_res"],
        bottleneck=hparams["bottleneck"],
        cond_prior=hparams["cond_prior"],
        q_correction=hparams["q_correction"],
        bias_max_res=hparams["bias_max_res"],
        x_like=hparams["x_like"],
        kl_free_bits=hparams["kl_free_bits"],
        std_init=hparams["std_init"],
        dataset_id=hparams.get("dataset", hparams.get("hps", "morphomnist")),
        rngs=nnx.Rngs(seed),
    )


def _align_restored_state(template: Any, restored: Any) -> Any:
    """Match Orbax stringified list keys to the NNX graph's numeric key order."""
    if isinstance(template, Mapping) and isinstance(restored, Mapping):
        aligned = {}
        for key, template_value in template.items():
            restored_key = key if key in restored else str(key)
            if restored_key not in restored:
                raise KeyError(f"Restored NNX state is missing key {key!r}")
            aligned[key] = _align_restored_state(
                template_value, restored[restored_key]
            )
        return aligned
    return restored


def _materialize_vae(hparams: Dict[str, Any], weights: Any, seed: int):
    model = _build_vae(hparams, seed)
    graphdef, _ = nnx.split(model, nnx.Param)
    template = nnx.state(model, nnx.Param).to_pure_dict()
    model = materialize_nnx(graphdef, _align_restored_state(template, weights))
    model.eval()
    return model


def _load_pgm(path: str, trust_incomplete: bool, seed: int):
    checkpoint, resolved = _load_checkpoint(path, trust_incomplete)
    if checkpoint.get("format_version") != 2 or "ema_params" not in checkpoint:
        raise ValueError(f"Unsupported JAX PGM checkpoint format at {resolved}")
    hparams = checkpoint.get("hparams", {})
    model = MorphoMNISTPGM(widths=hparams.get("widths", [32, 32]), rngs=nnx.Rngs(seed))
    graphdef, _ = nnx.split(model, nnx.Param)
    template = nnx.state(model, nnx.Param).to_pure_dict()
    weights = _align_restored_state(template, checkpoint["ema_params"])
    model = materialize_nnx(graphdef, weights)
    model.eval()
    return model, resolved


def _load_predictor(path: str, trust_incomplete: bool, seed: int, args: SimpleNamespace):
    checkpoint, resolved = _load_checkpoint(path, trust_incomplete)
    if checkpoint.get("format_version") != 3:
        raise ValueError(f"Unsupported JAX predictor checkpoint format at {resolved}")
    hparams = checkpoint.get("hparams", {})
    model = MorphoMNISTSupAuxPredictor(
        input_channels=hparams.get("input_channels", args.input_channels),
        input_res=hparams.get("input_res", args.input_res),
        width=hparams.get("width", 8),
        std_fixed=hparams.get("std_fixed", 0.0),
        rngs=nnx.Rngs(seed),
    )
    graphdef, params_state, batch_stats_state = nnx.split(
        model, nnx.Param, nnx.BatchStat
    )
    params = _align_restored_state(
        params_state.to_pure_dict(), checkpoint["ema_params"]
    )
    batch_stats = _align_restored_state(
        batch_stats_state.to_pure_dict(), checkpoint["ema_batch_stats"]
    )
    model = nnx.merge(
        graphdef,
        nnx.State(params),
        nnx.State(batch_stats),
    )
    model.eval()
    return model, resolved


@functools.lru_cache(maxsize=1)
def load_visualizer_bundle(
    checkpoint_path: str, trust_incomplete: bool, seed: int
) -> VisualizerBundle:
    hparams = _read_hparams(checkpoint_path)
    args = SimpleNamespace(**hparams)

    checkpoint, resolved = _load_checkpoint(checkpoint_path, trust_incomplete)
    weights = checkpoint.get("ema_params", checkpoint.get("vae_params"))
    if weights is None:
        raise ValueError(f"Counterfactual checkpoint at {resolved} has no VAE weights")
    vae = _materialize_vae(hparams, weights, seed)

    pgm_path = hparams.get("resolved_pgm_path") or hparams.get("pgm_path")
    predictor_path = hparams.get("resolved_predictor_path") or hparams.get("predictor_path")
    if not pgm_path or not predictor_path:
        raise ValueError("Checkpoint hparams do not contain JAX PGM and predictor paths")

    pgm, resolved_pgm = _load_pgm(pgm_path, trust_incomplete, seed)
    predictor, resolved_predictor = _load_predictor(
        predictor_path, trust_incomplete, seed, args
    )
    print(f"Loaded counterfactual checkpoint: {resolved}")
    print(f"Loaded PGM checkpoint: {resolved_pgm}")
    print(f"Loaded predictor checkpoint: {resolved_predictor}")
    return VisualizerBundle(args, vae, pgm, predictor, resolved)


def predict_image_parents(bundle: VisualizerBundle, image: jax.Array) -> Dict[str, jax.Array]:
    intensity_loc, _ = jnp.split(bundle.predictor.encoder_i(image), 2, axis=-1)
    intensity = jnp.tanh(intensity_loc.astype(jnp.float32))
    predictions = bundle.predictor.predict(x=image, intensity=intensity)
    digit_idx = jnp.argmax(predictions["digit"], axis=-1)
    return {
        "thickness": jnp.clip(predictions["thickness"], -1.0, 1.0),
        "intensity": jnp.clip(predictions["intensity"], -1.0, 1.0),
        "digit": jax.nn.one_hot(digit_idx, 10, dtype=jnp.float32),
    }


def dict_to_vector(values: Dict[str, jax.Array]) -> jax.Array:
    return jnp.concatenate(
        [values["thickness"], values["intensity"], values["digit"]], axis=-1
    )


def generate_from_sliders(
    bundle: VisualizerBundle,
    digit: int,
    thickness: float,
    intensity: float,
    style_seed: int = 0,
) -> Tuple[Image.Image, Dict[str, float]]:
    parents = build_parent_vector(digit, thickness, intensity)
    context = spatial_parents(parents, bundle.args.input_res)
    image, _ = bundle.vae.sample(
        parents=context,
        return_loc=True,
        rng=jax.random.PRNGKey(int(style_seed)),
    )
    summary = summarize_parents(parents)
    summary["style_seed"] = int(style_seed)
    return array_to_pil(image), summary


def linked_intensity_from_thickness(
    bundle: VisualizerBundle,
    thickness: float,
    seed_image: Image.Image | None = None,
) -> float:
    normalized_thickness = jnp.asarray(
        [[normalize_value(thickness, "thickness")]], dtype=jnp.float32
    )
    if seed_image is not None:
        source = preprocess_seed_image(seed_image, bundle.args.input_res)
        factual = predict_image_parents(bundle, source)
        linked = bundle.pgm.counterfactual(
            obs=factual,
            intervention={"thickness": normalized_thickness},
        )["intensity"]
    else:
        linked, _ = bundle.pgm.intensity_forward(
            jnp.zeros_like(normalized_thickness), normalized_thickness
        )
    normalized = float(np.asarray(jax.device_get(linked)).reshape(-1)[0])
    return round(denormalize_value(normalized, "intensity"), 3)


def update_linked_preview(
    bundle: VisualizerBundle,
    seed_image: Image.Image | None,
    digit: int,
    thickness: float,
    style_seed: int = 0,
) -> Tuple[float, Image.Image, Dict[str, float]]:
    intensity = linked_intensity_from_thickness(bundle, thickness, seed_image)
    image, factors = generate_from_sliders(
        bundle, digit, thickness, intensity, style_seed
    )
    return intensity, image, factors


def predict_seed_factors(
    bundle: VisualizerBundle, image: Image.Image
) -> Tuple[Any, Any, Any, Dict[str, float]]:
    if image is None:
        raise gr.Error("Upload a seed image first.")
    source = preprocess_seed_image(image, bundle.args.input_res)
    summary = summarize_parents(predict_image_parents(bundle, source))
    return (
        gr.update(value=str(summary["digit"])),
        gr.update(value=summary["thickness"]),
        gr.update(value=summary["intensity"]),
        summary,
    )


def render_counterfactual(
    bundle: VisualizerBundle,
    image: Image.Image,
    digit: int,
    thickness: float,
    intensity: float,
) -> Tuple[Image.Image, Image.Image, Dict[str, float], Dict[str, float]]:
    target_vector = build_parent_vector(digit, thickness, intensity)
    target = parent_dict(target_vector)

    if image is None:
        source_image, factual_summary = generate_from_sliders(
            bundle, digit, thickness, intensity
        )
        source = preprocess_seed_image(source_image, bundle.args.input_res)
        factual = target
    else:
        source = preprocess_seed_image(image, bundle.args.input_res)
        factual = predict_image_parents(bundle, source)
        factual_summary = summarize_parents(factual)

    cf = bundle.pgm.counterfactual(obs=factual, intervention=target)
    factual_context = spatial_parents(dict_to_vector(factual), bundle.args.input_res)
    cf_context = spatial_parents(cf["pa"], bundle.args.input_res)
    rng = jax.random.PRNGKey(int(bundle.args.seed))
    abduct_key, rec_key, cf_key = jax.random.split(rng, 3)
    latents = bundle.vae.abduct(
        source, parents=factual_context, t=1.0, rng=abduct_key
    )
    rec_x, rec_scale = bundle.vae.forward_latents(
        latents, parents=factual_context, rng=rec_key
    )
    cf_x, cf_scale = bundle.vae.forward_latents(
        latents, parents=cf_context, rng=cf_key
    )
    residual = (source - rec_x) / jnp.clip(rec_scale, min=1e-12)
    cf_x = jnp.clip(cf_x + cf_scale * residual, -1.0, 1.0)
    return (
        array_to_pil(source),
        array_to_pil(cf_x),
        factual_summary,
        summarize_parents(cf["pa"]),
    )


def build_app(bundle: VisualizerBundle) -> gr.Blocks:
    with gr.Blocks(title="MorphoMNIST Causal Visualizer", css=APP_CSS) as demo:
        gr.Markdown(
            """
            # MorphoMNIST Causal Visualizer

            Control the MorphoMNIST causal factors directly or upload a seed digit and
            render a counterfactual edit from the trained final checkpoint.

            This GUI uses the Causal-Gen model trained entirely in JAX, including its
            generative model, causal graphical model, and image-to-factor predictor.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                digit = gr.Radio(
                    choices=[str(i) for i in range(10)], value="3", label="Digit"
                )
                style_seed = gr.Slider(
                    minimum=0, maximum=999999, value=0, step=1, label="Style seed"
                )
                thickness = gr.Slider(
                    minimum=MORPHO_MIN_MAX["thickness"][0],
                    maximum=MORPHO_MIN_MAX["thickness"][1],
                    value=3.5,
                    step=0.01,
                    label="Thickness",
                )
                intensity = gr.Slider(
                    minimum=MORPHO_MIN_MAX["intensity"][0],
                    maximum=MORPHO_MIN_MAX["intensity"][1],
                    value=160.0,
                    step=0.5,
                    label="Intensity",
                )
                seed_image = gr.Image(
                    type="pil",
                    label="Seed image for counterfactual editing",
                    image_mode="L",
                )
                with gr.Row():
                    generate_btn = gr.Button("Generate from sliders", variant="primary")
                    cf_btn = gr.Button("Render counterfactual", variant="secondary")
                load_seed_btn = gr.Button("Load seed factors from image")

            with gr.Column(scale=1):
                generated = gr.Image(
                    type="pil",
                    label="Generated image",
                    width="100%",
                    height=480,
                    elem_id="generated-preview",
                )
                original = gr.Image(
                    type="pil", 
                    label="Seed / factual image",
                    width="100%",
                    height=480,
                    elem_id="original-preview",
                )
                counterfactual = gr.Image(
                    type="pil", 
                    label="Counterfactual image",
                    width="100%",
                    height=480,
                    elem_id="counterfactual-preview"
                )
                factual_json = gr.JSON(label="Factual factors")
                target_json = gr.JSON(label="Target factors")
                slider_json = gr.JSON(label="Generated slider factors")

        generate_fn = lambda d, t, i, s: generate_from_sliders(
            bundle, int(d), t, i, int(s)
        )
        generate_btn.click(
            fn=generate_fn,
            inputs=[digit, thickness, intensity, style_seed],
            outputs=[generated, slider_json],
            api_name=False,
            show_progress="hidden",
            trigger_mode="always_last",
            concurrency_limit=1,
            concurrency_id="slider-preview",
        )

        thickness.release(
            fn=lambda img, d, t, s: update_linked_preview(
                bundle, img, int(d), t, int(s)
            ),
            inputs=[seed_image, digit, thickness, style_seed],
            outputs=[intensity, generated, slider_json],
            api_name=False,
            show_progress="hidden",
            trigger_mode="always_last",
            concurrency_limit=1,
            concurrency_id="slider-preview",
        )

        for component, event in (
            (intensity, intensity.release),
            (digit, digit.change),
            (style_seed, style_seed.release),
        ):
            del component
            event(
                fn=generate_fn,
                inputs=[digit, thickness, intensity, style_seed],
                outputs=[generated, slider_json],
                api_name=False,
                show_progress="hidden",
                trigger_mode="always_last",
                concurrency_limit=1,
                concurrency_id="slider-preview",
            )

        load_seed_btn.click(
            fn=lambda img: predict_seed_factors(bundle, img),
            inputs=[seed_image],
            outputs=[digit, thickness, intensity, factual_json],
            api_name=False,
        )
        cf_btn.click(
            fn=lambda img, d, t, i: render_counterfactual(
                bundle, img, int(d), t, i
            ),
            inputs=[seed_image, digit, thickness, intensity],
            outputs=[original, counterfactual, factual_json, target_json],
            api_name=False,
        )
        demo.load(
            fn=lambda img, d, t, s: update_linked_preview(
                bundle, img, int(d), t, int(s)
            ),
            inputs=[seed_image, digit, thickness, style_seed],
            outputs=[intensity, generated, slider_json],
            api_name=False,
            show_progress="hidden",
            trigger_mode="always_last",
            concurrency_limit=1,
            concurrency_id="slider-preview",
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure-JAX MorphoMNIST causal visualizer")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="JAX counterfactual Orbax checkpoint root or numeric step directory.",
    )
    parser.add_argument(
        "--accelerator",
        default="cpu",
        choices=["cpu", "gpu", "tpu"],
        help="JAX inference backend (selected before JAX is imported).",
    )
    parser.add_argument("--gpu_id", default=None, help="Optional visible GPU index.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--trust-incomplete-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore uploaded Orbax steps that lack commit_success.txt (default: true).",
    )
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    seed_all(cli.seed, deterministic=True)
    bundle = load_visualizer_bundle(
        cli.checkpoint, cli.trust_incomplete_checkpoint, cli.seed
    )
    app = build_app(bundle)
    app.launch(
        server_name=cli.server_name,
        server_port=cli.server_port,
        share=cli.share,
        show_api=False,
    )


if __name__ == "__main__":
    main()
