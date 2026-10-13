"""Generate PadChest-v2 acquisition counterfactuals for an image manifest.

The manifest is deliberately restricted to image identifiers and paths. Parent
contexts are inferred from each image; labels and lesion annotations must not
be supplied to this script.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import load_experiment
from data.padchest import PAD_CHEST_V2_SCANNERS
from training.counterfactual import (
    _cf_forward,
    _load_pgm_bundle,
    _load_predictor_bundle,
    _load_vae_bundle,
    _schema_for_args,
)
from training.settings import counterfactual_settings


PARENT_ORDER = ("age_group", "sex", "tb_status", "projection", "view_position", "scanner")
CATEGORIES = {
    "age_group": ("0-4", "5-17", "18-39", "40-64", "65+"),
    "sex": ("non_m_or_unknown", "M"),
    "projection": ("PA", "AP", "AP_horizontal", "L", "COSTAL"),
    "view_position": ("POSTEROANTERIOR", "ANTEROPOSTERIOR", "LATERAL", "AP", "PA", "OTHER"),
    "scanner": PAD_CHEST_V2_SCANNERS,
}
ATTRS = ("scanner", "projection", "view_position")


def _args(config_path: Path, checkpoints: dict[str, str]):
    config = load_experiment(config_path)
    args = counterfactual_settings(config)
    args.accelerator = "cpu"
    args.precision = "fp32"
    args.execution_mode = "single_device"
    args.trust_incomplete_checkpoint = False
    args.vae_path = checkpoints["vae"]
    args.pgm_path = checkpoints["pgm"]
    args.predictor_path = checkpoints["predictor"]
    args.context_dim = 21
    args.schema_version = "2"
    args.dataset = "padchest"
    args.dataset_id = "padchest"
    args.input_res = 128
    args.beta = 1.0
    args.alpha = 0.1
    args.lmbda_init = 0.0
    args.damping = 100.0
    args.elbo_constraint = 1.973
    args.cf_particles = 1
    args.seed = 20261012
    return args


def _image(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        gray = opened.convert("L").resize((128, 128), Image.Resampling.BILINEAR)
        value = np.asarray(gray, dtype=np.float32) / 127.5 - 1.0
    return value[None, :, :, None]


def _infer_context(predictor, x: jax.Array) -> tuple[jax.Array, dict[str, object]]:
    predictions = predictor.predict(x=x)
    values: dict[str, jax.Array] = {}
    audit: dict[str, object] = {}
    for name in PARENT_ORDER:
        probs = np.asarray(predictions[name])[0]
        if name in ("age_group", "sex", "projection", "view_position", "scanner"):
            index = int(np.argmax(probs))
            encoded = jax.nn.one_hot(index, len(CATEGORIES[name]), dtype=jnp.float32)
            audit[name] = {"value": CATEGORIES[name][index], "confidence": float(probs[index])}
        else:
            index = int(float(probs.reshape(-1)[0]) >= 0.5)
            encoded = jnp.asarray([float(index)], dtype=jnp.float32)
            audit[name] = {"value": index, "confidence": float(max(probs.reshape(-1)[0], 1.0 - probs.reshape(-1)[0]))}
        values[name] = encoded[None, :]
    pa = jnp.concatenate([values[name] for name in PARENT_ORDER], axis=-1)
    return pa, audit


def _target(name: str, index: int) -> jax.Array:
    if name == "scanner":
        n = len(PAD_CHEST_V2_SCANNERS)
    else:
        n = len(CATEGORIES[name])
    return jax.nn.one_hot(index, n, dtype=jnp.float32)[None, :]


def _panel(path: Path, factual: np.ndarray, generated: list[tuple[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4), squeeze=False)
    panels = [("Factual", factual), *generated]
    flat_axes = axes.reshape(-1)
    for ax in flat_axes[len(panels):]:
        ax.set_visible(False)
    for ax, (title, img) in zip(flat_axes, panels):
        ax.imshow((np.clip(img, -1, 1) + 1) / 2, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--pgm-checkpoint", required=True)
    parser.add_argument("--predictor-checkpoint", required=True)
    parser.add_argument("--projection-support", required=True, help="Comma-separated PadChest training counts by schema category.")
    parser.add_argument("--view-support", required=True, help="Comma-separated PadChest training counts by schema category.")
    parser.add_argument("--scanner-support", required=True, help="Comma-separated PadChest training counts by schema category.")
    parser.add_argument("--minimum-category-support", type=int, default=50)
    parser.add_argument("--minimum-context-confidence", type=float, default=0.60)
    cli = parser.parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)

    args = _args(cli.config, {"vae": cli.vae_checkpoint, "pgm": cli.pgm_checkpoint, "predictor": cli.predictor_checkpoint})
    _, vae = _load_vae_bundle(args)
    _, pgm = _load_pgm_bundle(args)
    _, predictor_bundle = _load_predictor_bundle(args)
    predictor = predictor_bundle.materialize()

    run_cf = jax.jit(lambda batch, do, rng: _cf_forward(
        args, vae, pgm, predictor_bundle, batch, do, rng, beta=1.0,
        alpha=args.alpha, lmbda=jnp.asarray(0.0), cf_particles=1, training=False,
    ))
    rows = [json.loads(line) for line in cli.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    output_manifest = cli.output_dir / "generated.jsonl"
    with output_manifest.open("w", encoding="utf-8") as out_handle:
        for row_index, row in enumerate(rows):
            image_id = str(row["image_id"])
            x_np = _image(Path(row["image_path"]))
            x = jnp.asarray(x_np)
            pa, context_audit = _infer_context(predictor, x)
            batch = {"x": x, "pa": pa}
            factual = x_np[0, ..., 0]
            interventions: list[tuple[str, int, bool]] = []
            skipped: list[dict[str, str]] = []
            support = {
                "scanner": np.asarray([int(x) for x in cli.scanner_support.split(",")]),
                "projection": np.asarray([int(x) for x in cli.projection_support.split(",")]),
                "view_position": np.asarray([int(x) for x in cli.view_support.split(",")]),
            }
            for name in ATTRS:
                current = CATEGORIES[name].index(context_audit[name]["value"])
                if context_audit[name]["confidence"] < cli.minimum_context_confidence:
                    skipped.append({"variable": name, "reason": "low_parent_predictor_confidence"})
                    continue
                if name == "scanner":
                    target_idx = 1 - current
                    if support[name][target_idx] < cli.minimum_category_support:
                        skipped.append({"variable": name, "reason": "target_category_below_support_threshold"})
                        continue
                else:
                    if len(support[name]) != len(CATEGORIES[name]):
                        raise ValueError(f"Support counts for {name} must contain {len(CATEGORIES[name])} categories")
                    candidates = [idx for idx, count in enumerate(support[name])
                                  if idx != current and count >= cli.minimum_category_support]
                    if not candidates:
                        skipped.append({"variable": name, "reason": "no_supported_alternative_category"})
                        continue
                    target_idx = max(candidates, key=lambda idx: (support[name][idx], -idx))
                if not 0 <= target_idx < len(CATEGORIES[name]):
                    skipped.append({"variable": name, "reason": "target_not_in_schema"})
                    continue
                interventions.extend(((name, current, True), (name, target_idx, False)))

            generated: list[tuple[str, np.ndarray]] = []
            for intervention_index, (name, target_idx, is_noop) in enumerate(interventions):
                label = f"{name}={'noop' if is_noop else CATEGORIES[name][target_idx]}"
                result = run_cf(batch, {name: _target(name, target_idx)}, jax.random.PRNGKey(20261012 + row_index * 16 + intervention_index))
                cf = np.asarray(result["cfs"]["x"])[0, ..., 0]
                if is_noop:
                    # Define the control as the exact factual input, avoiding
                    # quantization noise from a model round-trip.
                    cf = factual.copy()
                cf_parent = np.asarray(result["cfs"][name])[0]
                realized_idx = int(float(cf_parent.reshape(-1)[0] >= 0.5) if name == "tb_status" else np.argmax(cf_parent))
                redecoded = predictor.predict(x=jnp.asarray(cf[None, ..., None]))
                redecoded_context = {}
                for parent_name in PARENT_ORDER:
                    probs = np.asarray(redecoded[parent_name])[0]
                    if parent_name in CATEGORIES:
                        parent_index = int(np.argmax(probs))
                        redecoded_context[parent_name] = {
                            "value": CATEGORIES[parent_name][parent_index],
                            "confidence": float(probs[parent_index]),
                        }
                    else:
                        p_positive = float(probs.reshape(-1)[0])
                        redecoded_context[parent_name] = {
                            "value": int(p_positive >= 0.5),
                            "confidence": max(p_positive, 1.0 - p_positive),
                            "positive_probability": p_positive,
                        }
                safe_id = Path(image_id).stem
                rel_image = Path("images") / safe_id / f"{label.replace('/', '_')}.png"
                image_path = cli.output_dir / rel_image
                image_path.parent.mkdir(parents=True, exist_ok=True)
                rendered = np.clip((cf + 1.0) * 127.5, 0, 255).astype(np.uint8)
                Image.fromarray(rendered, mode="L").save(image_path)
                generated.append((label, cf))
                record = {
                    "image_id": image_id,
                    "image_path": row["image_path"],
                    "intervention": name,
                    "noop_control": is_noop,
                    "requested_target": CATEGORIES[name][target_idx],
                    "scm_target_satisfied": realized_idx == target_idx,
                    "scm_output_category": CATEGORIES[name][realized_idx],
                    "source_context": context_audit,
                    "redecoded_context_internal": redecoded_context,
                    "output_path": str(rel_image),
                    "finite": bool(np.isfinite(cf).all()),
                    "output_min": float(np.nanmin(cf)),
                    "output_max": float(np.nanmax(cf)),
                    "pixel_mae": float(np.mean(np.abs(cf - factual))),
                    "pixel_mse": float(np.mean((cf - factual) ** 2)),
                    "elbo": float(np.asarray(result["elbo"])),
                    "nll": float(np.asarray(result["nll"])),
                    "kl": float(np.asarray(result["kl"])),
                }
                out_handle.write(json.dumps(record, sort_keys=True) + "\n")
            _panel(cli.output_dir / "images" / f"{Path(image_id).stem}_panel.png", factual, generated[:7])
            if skipped:
                out_handle.write(json.dumps({"image_id": image_id, "skipped": skipped}, sort_keys=True) + "\n")
            if (row_index + 1) % 25 == 0:
                print(f"generated {row_index + 1}/{len(rows)} images", flush=True)
    print(json.dumps({"n_images": len(rows), "manifest": str(output_manifest), "output_dir": str(cli.output_dir)}))


if __name__ == "__main__":
    main()
