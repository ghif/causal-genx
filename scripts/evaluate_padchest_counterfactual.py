"""Evaluate the PadChest v2 counterfactual image model on selected factual studies.

This is intentionally a small, artifact-level smoke evaluation: it restores the
latest complete local copies of the finetuned HVAE, SCM, and image-parent
predictor, applies named interventions, and writes quantitative JSON plus
captioned image panels for the journal report.
"""

from __future__ import annotations

import ast
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import load_experiment
from data.padchest import PAD_CHEST_V2_AGE_GROUPS, PAD_CHEST_V2_SCANNERS, derive_padchest_v2_row
from training.counterfactual import (
    _cf_forward,
    _load_pgm_bundle,
    _load_predictor_bundle,
    _load_vae_bundle,
    _schema_for_args,
)
from training.settings import counterfactual_settings


ROOT = Path(__file__).resolve().parents[1]
TMP = Path("/private/tmp/padchest_cf_eval")
CSV_PATH = Path("/private/tmp/padchest_eval_v2/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv")
OUT = ROOT / "journal/artifacts/padchest_cf_v2_eval"

SAMPLES = {
    "tb_sequela_pa": "144161045940829338272357413255772911245_6ca9jm.png",
    "normal_pa": "20536686640136348236148679891455886468_k6ga29.png",
    "male_pa": "135803415504923515076821959678074435083_fzis7b.png",
    "male_lateral": "135803415504923515076821959678074435083_fzis7d.png",
    "male_ap": "313903302629300007485735352869488750471_75sg0k.png",
}


def _load_rows() -> dict[str, dict[str, str]]:
    wanted = set(SAMPLES.values())
    out = {}
    with CSV_PATH.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["ImageID"] in wanted:
                out[row["ImageID"]] = row
    return out


def _one_hot(size: int, index: int) -> np.ndarray:
    value = np.zeros(size, dtype=np.float32)
    value[index] = 1.0
    return value


def _batch(row: dict[str, str], image_path: Path) -> tuple[dict[str, jax.Array], dict[str, object]]:
    derived = derive_padchest_v2_row(row, "tb_or_sequelae")
    # PadChest's PNG export is commonly 16-bit ``I;16``.  Converting directly
    # with PIL's ``convert("L")`` clips those values and produces an almost
    # white image.  Match PadChestDataset's uint16 path exactly: scale to 8-bit
    # before resizing and normalizing to the HVAE's [-1, 1] contract.
    raw = Image.open(image_path)
    arr = np.asarray(raw)
    if arr.dtype == np.uint16 or raw.mode.startswith("I"):
        arr_u8 = (arr.astype(np.float32) / 256.0).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(arr_u8, mode="L")
    else:
        image = raw.convert("L")
    image = image.resize((128, 128), Image.Resampling.BILINEAR)
    x = np.asarray(image, dtype=np.float32)[None, :, :, None] / 127.5 - 1.0
    pa = np.concatenate([derived[name] for name in ("age_group", "sex", "tb_status", "projection", "view_position", "scanner")])[None]
    batch = {"x": jnp.asarray(x), "pa": jnp.asarray(pa)}
    return batch, derived["_audit"]


def _target(args, name: str, index: int | None = None, binary: float | None = None) -> jax.Array:
    spec = {v.name: v for v in _schema_for_args(args).variables}[name]
    if binary is not None:
        return jnp.asarray([[binary]], dtype=jnp.float32)
    assert index is not None
    return jnp.asarray([_one_hot(spec.encoded_dim, index)], dtype=jnp.float32)


def _interventions(args) -> dict[str, dict[str, jax.Array]]:
    return {
        "tb_status=1": {"tb_status": _target(args, "tb_status", binary=1.0)},
        "tb_status=0": {"tb_status": _target(args, "tb_status", binary=0.0)},
        "age_group=65+": {"age_group": _target(args, "age_group", index=4)},
        "sex=M": {"sex": _target(args, "sex", index=1)},
        "projection=AP": {"projection": _target(args, "projection", index=1)},
        "view_position=LATERAL": {"view_position": _target(args, "view_position", index=2)},
        "scanner=Philips": {"scanner": _target(args, "scanner", index=1)},
    }


def _label_from_vector(name: str, value: np.ndarray) -> str:
    if name == "tb_status":
        return "positive" if float(value.reshape(-1)[0]) >= 0.5 else "negative"
    categories = {
        "age_group": PAD_CHEST_V2_AGE_GROUPS,
        "sex": ("non-M/unknown", "M"),
        "projection": ("PA", "AP", "AP_horizontal", "L", "COSTAL"),
        "view_position": ("POSTEROANTERIOR", "ANTEROPOSTERIOR", "LATERAL", "AP", "PA", "OTHER"),
        "scanner": PAD_CHEST_V2_SCANNERS,
    }
    return categories[name][int(np.argmax(value))]


def _render_panel(sample_name: str, factual: np.ndarray, outputs: list[tuple[str, np.ndarray]], caption: str) -> None:
    # Use a 2x4 layout rather than an eight-column strip.  The latter is
    # technically correct but gets downsampled so aggressively in Markdown
    # previews that the radiographs appear to have broken contrast.
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.8), squeeze=False)
    axes = axes.reshape(-1)
    panels = [("Factual", factual), *outputs]
    for ax, (label, image) in zip(axes, panels):
        ax.imshow((image + 1) / 2, cmap="gray", vmin=0, vmax=1)
        ax.set_title(label, fontsize=8)
        ax.axis("off")
    fig.suptitle(caption, fontsize=10, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / f"{sample_name}.png", dpi=180)
    plt.close(fig)

    # Render spatial change maps directly from the model tensors, before any
    # panel rasterization. A shared robust scale makes the seven interventions
    # visually comparable within each study without allowing isolated extremes
    # to dominate the color range.
    differences = [np.abs(image - factual) for _, image in outputs]
    vmax = max(float(np.percentile(np.concatenate([d.ravel() for d in differences]), 99)), 1e-6)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.8), squeeze=False)
    axes_flat = axes.reshape(-1)
    axes_flat[0].imshow((factual + 1) / 2, cmap="gray", vmin=0, vmax=1)
    axes_flat[0].set_title("Factual reference", fontsize=8)
    for ax, (label, _), difference in zip(axes_flat[1:], outputs, differences):
        heat = ax.imshow(difference, cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(label, fontsize=8)
    for ax in axes_flat:
        ax.axis("off")
    fig.suptitle(
        f"Absolute factual-to-counterfactual pixel difference — {sample_name}\n"
        f"Shared display ceiling: {vmax:.4f} normalized intensity units (per-study 99th percentile)",
        fontsize=10,
        y=0.98,
    )
    fig.colorbar(heat, ax=axes, fraction=0.025, pad=0.02, label="Absolute difference (normalized units)")
    fig.savefig(OUT / f"{sample_name}_difference_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_experiment(ROOT / "configs/padchest_counterfactual_v2_tpu_v6e1.yaml")
    args = counterfactual_settings(config)
    args.accelerator = "cpu"
    args.precision = "fp32"
    args.execution_mode = "single_device"
    args.trust_incomplete_checkpoint = False
    args.vae_path = str(TMP / "counterfactual_padchest_v2_tpu_v6e1/checkpoints/80200")
    args.pgm_path = str(TMP / "scm_padchest_v2_tpu_v6e1/checkpoints/198495")
    args.predictor_path = str(TMP / "predictor_padchest_v2_tpu_v6e1/checkpoints/26065")
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
    args.seed = 7

    print("Loading artifacts on", jax.devices()[0])
    _, vae = _load_vae_bundle(args)
    _, pgm = _load_pgm_bundle(args)
    _, predictor = _load_predictor_bundle(args)
    rows = _load_rows()
    all_results: dict[str, object] = {"checkpoint_step": 80200, "evaluated_samples": {}}
    interventions = _interventions(args)

    for sample_name, image_id in SAMPLES.items():
        row = rows[image_id]
        batch, audit = _batch(row, TMP / "images" / image_id)
        factual = np.asarray(batch["x"])[0, :, :, 0]
        outputs = []
        raw_image = Image.open(TMP / "images" / image_id)
        raw_array = np.asarray(raw_image)
        sample_results = {
            "image_id": image_id,
            "audit": audit,
            "input_image": {
                "mode": raw_image.mode,
                "dtype": str(raw_array.dtype),
                "raw_min": int(raw_array.min()),
                "raw_max": int(raw_array.max()),
                "raw_mean": float(raw_array.mean()),
                "raw_std": float(raw_array.std()),
                "normalized_min": float(factual.min()),
                "normalized_max": float(factual.max()),
                "normalized_mean": float(factual.mean()),
                "normalized_std": float(factual.std()),
            },
            "interventions": {},
        }
        for label, do in interventions.items():
            out = _cf_forward(args, vae, pgm, predictor, batch, do, jax.random.PRNGKey(args.seed), beta=1.0, alpha=args.alpha, lmbda=jnp.asarray(0.0), cf_particles=1, training=False)
            cf_x = np.asarray(out["cfs"]["x"])[0, :, :, 0]
            cf_pa = {name: np.asarray(value)[0] for name, value in out["cfs"].items() if name != "x"}
            mae = float(np.mean(np.abs(cf_x - factual)))
            psnr = float(10.0 * np.log10(4.0 / max(np.mean((cf_x - factual) ** 2), 1e-12)))
            intervention_name, target = next(iter(do.items()))
            predicted_target = _label_from_vector(intervention_name, cf_pa[intervention_name])
            target_label = _label_from_vector(intervention_name, np.asarray(target)[0])
            sample_results["interventions"][label] = {
                "target": target_label,
                "scm_output": {name: _label_from_vector(name, value) for name, value in cf_pa.items() if name != "pa"},
                "target_satisfied": bool(np.allclose(cf_pa[intervention_name], np.asarray(target)[0], atol=1e-4)),
                "pixel_mae_0_2": mae,
                "psnr_db": psnr,
                "elbo": float(np.asarray(out["elbo"])),
                "nll": float(np.asarray(out["nll"])),
                "kl": float(np.asarray(out["kl"])),
            }
            outputs.append((label, cf_x))
        _render_panel(sample_name, factual, outputs, f"PadChest v2 counterfactuals — {sample_name}")
        all_results["evaluated_samples"][sample_name] = sample_results

    # Aggregate all pairs so the report can quote exact sample-level diagnostics.
    pairs = [r for sample in all_results["evaluated_samples"].values() for r in sample["interventions"].values()]
    all_results["aggregate"] = {
        "pairs": len(pairs),
        "target_satisfaction_rate": float(np.mean([p["target_satisfied"] for p in pairs])),
        "mean_pixel_mae_0_2": float(np.mean([p["pixel_mae_0_2"] for p in pairs])),
        "mean_psnr_db": float(np.mean([p["psnr_db"] for p in pairs])),
        "mean_elbo": float(np.mean([p["elbo"] for p in pairs])),
        "mean_nll": float(np.mean([p["nll"] for p in pairs])),
        "mean_kl": float(np.mean([p["kl"] for p in pairs])),
    }
    (OUT / "metrics.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(json.dumps(all_results["aggregate"], indent=2))


if __name__ == "__main__":
    main()
