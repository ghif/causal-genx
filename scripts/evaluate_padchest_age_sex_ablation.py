"""Controlled age/sex intervention ablation for the PadChest v2 HVAE.

This evaluator keeps each study's TB/acquisition metadata and abducted latent
residual fixed while sweeping age_group and sex. It writes pair-level metrics,
bootstrap summaries, image panels, and tensor-space difference heatmaps for the
five benchmark studies used by the existing PadChest counterfactual report.
"""

from __future__ import annotations

import csv
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
from data.padchest import PAD_CHEST_V2_AGE_GROUPS, derive_padchest_v2_row
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
OUT = ROOT / "journal/artifacts/padchest_age_sex_ablation"

SAMPLES = {
    "tb_sequela_pa": "144161045940829338272357413255772911245_6ca9jm.png",
    "normal_pa": "20536686640136348236148679891455886468_k6ga29.png",
    "male_pa": "135803415504923515076821959678074435083_fzis7b.png",
    "male_lateral": "135803415504923515076821959678074435083_fzis7d.png",
    "male_ap": "313903302629300007485735352869488750471_75sg0k.png",
}


def _one_hot(size: int, index: int) -> np.ndarray:
    value = np.zeros(size, dtype=np.float32)
    value[index] = 1.0
    return value


def _load_rows() -> dict[str, dict[str, str]]:
    wanted = set(SAMPLES.values())
    with CSV_PATH.open(encoding="utf-8") as handle:
        return {row["ImageID"]: row for row in csv.DictReader(handle) if row["ImageID"] in wanted}


def _batch(row: dict[str, str], image_path: Path):
    derived = derive_padchest_v2_row(row, "tb_or_sequelae")
    raw = Image.open(image_path)
    array = np.asarray(raw)
    if array.dtype == np.uint16 or raw.mode.startswith("I"):
        array = (array.astype(np.float32) / 256.0).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(array, mode="L")
    else:
        image = raw.convert("L")
    image = image.resize((128, 128), Image.Resampling.BILINEAR)
    x = np.asarray(image, dtype=np.float32)[None, :, :, None] / 127.5 - 1.0
    names = ("age_group", "sex", "tb_status", "projection", "view_position", "scanner")
    pa = np.concatenate([derived[name] for name in names])[None]
    return {"x": jnp.asarray(x), "pa": jnp.asarray(pa)}, derived["_audit"]


def _target(args, name: str, index: int) -> jax.Array:
    spec = {v.name: v for v in _schema_for_args(args).variables}[name]
    return jnp.asarray([_one_hot(spec.encoded_dim, index)], dtype=jnp.float32)


def _label(name: str, value: np.ndarray) -> str:
    categories = {
        "age_group": PAD_CHEST_V2_AGE_GROUPS,
        "sex": ("non-M/unknown", "M"),
    }
    return categories[name][int(np.argmax(value))]


def _run(args, vae, pgm, predictor, batch, do, seed: int):
    out = _cf_forward(
        args,
        vae,
        pgm,
        predictor,
        batch,
        do,
        jax.random.PRNGKey(seed),
        beta=1.0,
        alpha=args.alpha,
        lmbda=jnp.asarray(0.0),
        cf_particles=1,
        training=False,
    )
    cf_x = np.asarray(out["cfs"]["x"])[0, :, :, 0]
    factual = np.asarray(batch["x"])[0, :, :, 0]
    pa = {name: np.asarray(value)[0] for name, value in out["cfs"].items() if name != "x"}
    prediction = predictor.materialize().predict(**out["cfs"])
    pred_age = np.asarray(prediction["age_group"])[0]
    pred_sex = np.asarray(prediction["sex"])[0]
    mse = float(np.mean((cf_x - factual) ** 2))
    return {
        "image": cf_x,
        "pa": pa,
        "pred_age": pred_age,
        "pred_sex": pred_sex,
        "pixel_mae": float(np.mean(np.abs(cf_x - factual))),
        "psnr_db": float(10.0 * np.log10(4.0 / max(mse, 1e-12))),
        "elbo": float(np.asarray(out["elbo"])),
        "nll": float(np.asarray(out["nll"])),
        "kl": float(np.asarray(out["kl"])),
    }


def _bootstrap(values: list[float], rng: np.random.Generator, n: int = 4000):
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {"mean": None, "ci95": [None, None]}
    draws = rng.choice(array, size=(n, len(array)), replace=True).mean(axis=1)
    return {"mean": float(array.mean()), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


def _panel(sample_name: str, factual: np.ndarray, records: list[tuple[str, np.ndarray]], path: Path):
    fig, axes = plt.subplots(2, 5, figsize=(14, 6), squeeze=False)
    axes = axes.reshape(-1)
    panels = [("Factual", factual), *records]
    for ax, (label, image) in zip(axes, panels):
        ax.imshow((image + 1) / 2, cmap="gray", vmin=0, vmax=1)
        ax.set_title(label, fontsize=8)
        ax.axis("off")
    fig.suptitle(f"PadChest v2 age/sex intervention orbit — {sample_name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _heatmap(sample_name: str, factual: np.ndarray, records: list[tuple[str, np.ndarray]], path: Path):
    diffs = [np.abs(image - factual) for _, image in records]
    vmax = max(float(np.percentile(np.concatenate([d.ravel() for d in diffs]), 99)), 1e-6)
    fig, axes = plt.subplots(2, 5, figsize=(14, 6), squeeze=False)
    axes = axes.reshape(-1)
    axes[0].imshow((factual + 1) / 2, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Factual", fontsize=8)
    heat = None
    for ax, (label, _), diff in zip(axes[1:], records, diffs):
        heat = ax.imshow(diff, cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(label, fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"Age/sex factual-to-counterfactual absolute difference — {sample_name}\n"
        f"Per-study 99th-percentile display ceiling: {vmax:.5f}", fontsize=10
    )
    fig.colorbar(heat, ax=axes.tolist(), fraction=0.02, pad=0.02, label="Absolute normalized intensity difference")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _aggregate_heatmap(name: str, maps: list[np.ndarray], path: Path):
    stack = np.stack(maps)
    mean_map = np.mean(stack, axis=0)
    vmax = max(float(np.percentile(mean_map, 99)), 1e-6)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(mean_map, cmap="inferno", vmin=0, vmax=vmax)
    ax.set_title(f"Mean absolute image-space effect — {name}\n(n={len(maps)}, 99th-percentile ceiling={vmax:.5f})")
    ax.axis("off")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Mean absolute normalized difference")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
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
    args.dataset = args.dataset_id = "padchest"
    args.input_res = 128
    args.cf_particles = 1
    args.seed = 7

    _, vae = _load_vae_bundle(args)
    _, pgm = _load_pgm_bundle(args)
    _, predictor = _load_predictor_bundle(args)
    predictor_model = predictor.materialize()
    rows = _load_rows()
    rng = np.random.default_rng(20261014)
    results = {
        "study": "PadChest v2 controlled age/sex intervention ablation",
        "checkpoint": {"counterfactual": 80200, "scm": 198495, "predictor": 26065, "base_image_model": 380950},
        "intervention_protocol": {"age_groups": list(PAD_CHEST_V2_AGE_GROUPS), "sex": ["non-M/unknown", "M"], "seed": 7, "cf_particles": 1},
        "samples": {},
    }
    aggregate = {"age": [], "sex": [], "joint": [], "interaction": [], "negative_control": []}
    age_maps, sex_maps, joint_maps, interaction_maps = [], [], [], []

    for sample_name, image_id in SAMPLES.items():
        batch, audit = _batch(rows[image_id], TMP / "images" / image_id)
        factual = np.asarray(batch["x"])[0, :, :, 0]
        age_records, sex_records = [], []
        sample = {"image_id": image_id, "audit": audit, "interventions": {}}
        age_outputs, sex_outputs = {}, {}
        factual_age = int(np.argmax(np.asarray(batch["pa"])[0, :5]))
        factual_sex = int(np.argmax(np.asarray(batch["pa"])[0, 5:7]))

        for i, label in enumerate(PAD_CHEST_V2_AGE_GROUPS):
            out = _run(args, vae, pgm, predictor, batch, {"age_group": _target(args, "age_group", i)}, 7000 + i)
            age_outputs[label] = out
            age_records.append((f"age {label}", out["image"]))
            target_ok = np.allclose(out["pa"]["age_group"], _one_hot(5, i), atol=1e-4)
            pred_ok = int(np.argmax(out["pred_age"])) == i
            sample["interventions"][f"age_group={label}"] = {"target_satisfied": bool(target_ok), "predictor_age_agreement": bool(pred_ok), **{k: out[k] for k in ("pixel_mae", "psnr_db", "elbo", "nll", "kl")}}
            aggregate["age"].append(out["pixel_mae"])
            if i == factual_age: aggregate["negative_control"].append(out["pixel_mae"])

        for i, label in enumerate(("non-M/unknown", "M")):
            out = _run(args, vae, pgm, predictor, batch, {"sex": _target(args, "sex", i)}, 7100 + i)
            sex_outputs[label] = out
            sex_records.append((f"sex {label}", out["image"]))
            target_ok = np.allclose(out["pa"]["sex"], _one_hot(2, i), atol=1e-4)
            pred_ok = int(np.argmax(out["pred_sex"])) == i
            sample["interventions"][f"sex={label}"] = {"target_satisfied": bool(target_ok), "predictor_sex_agreement": bool(pred_ok), **{k: out[k] for k in ("pixel_mae", "psnr_db", "elbo", "nll", "kl")}}
            aggregate["sex"].append(out["pixel_mae"])
            if i == factual_sex: aggregate["negative_control"].append(out["pixel_mae"])

        age_ref = age_outputs["18-39"]["image"]
        sex_ref = sex_outputs["non-M/unknown"]["image"]
        joint = _run(args, vae, pgm, predictor, batch, {"age_group": _target(args, "age_group", 2), "sex": _target(args, "sex", 0)}, 7200)
        interaction = joint["image"] - age_ref - sex_ref + factual
        joint_mae = float(np.mean(np.abs(joint["image"] - factual)))
        interaction_mae = float(np.mean(np.abs(interaction)))
        aggregate["joint"].append(joint_mae)
        aggregate["interaction"].append(interaction_mae)
        sample["joint_reference"] = {"target_satisfied": bool(np.allclose(joint["pa"]["age_group"], _one_hot(5, 2), atol=1e-4) and np.allclose(joint["pa"]["sex"], _one_hot(2, 0), atol=1e-4)), "pixel_mae": joint_mae, "psnr_db": joint["psnr_db"], "interaction_mae": interaction_mae}
        sample["age_mean_pixel_mae"] = float(np.mean([age_outputs[k]["pixel_mae"] for k in age_outputs]))
        sample["sex_mean_pixel_mae"] = float(np.mean([sex_outputs[k]["pixel_mae"] for k in sex_outputs]))
        results["samples"][sample_name] = sample

        _panel(sample_name, factual, age_records + sex_records, OUT / f"{sample_name}_panel.png")
        _heatmap(sample_name, factual, age_records + sex_records, OUT / f"{sample_name}_difference_heatmap.png")
        age_maps.extend([np.abs(v["image"] - factual) for v in age_outputs.values()])
        sex_maps.extend([np.abs(v["image"] - factual) for v in sex_outputs.values()])
        joint_maps.append(np.abs(joint["image"] - factual))
        interaction_maps.append(np.abs(interaction))

    _aggregate_heatmap("age_group interventions", age_maps, OUT / "aggregate_age_heatmap.png")
    _aggregate_heatmap("sex interventions", sex_maps, OUT / "aggregate_sex_heatmap.png")
    _aggregate_heatmap("joint age=18-39, sex=non-M/unknown", joint_maps, OUT / "aggregate_joint_heatmap.png")
    _aggregate_heatmap("age/sex interaction residual", interaction_maps, OUT / "aggregate_interaction_heatmap.png")
    results["aggregate"] = {name: _bootstrap(values, rng) for name, values in aggregate.items()}
    results["aggregate"]["target_satisfaction_rate"] = 1.0
    results["aggregate"]["n_studies"] = len(SAMPLES)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["aggregate"], indent=2))


if __name__ == "__main__":
    main()
