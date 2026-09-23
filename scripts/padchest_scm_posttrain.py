#!/usr/bin/env python3
"""Generate PadChest SCM post-training validation artifacts.

This is intentionally local-first: it reads an existing Orbax checkpoint, a
PadChest metadata CSV (local cache or accessible URI), and writes JSON artifacts
under a caller-selected directory.  It never uploads or mutates the completed
checkpoint tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from causal.cxr_rait_scm import PadChestPGM
from data.padchest import (
    AGE_AT_STUDY_MAX,
    AGE_AT_STUDY_MIN,
    PAD_CHEST_SCHEMA,
    PROJECTIONS,
    STUDY_YEAR_MAX,
    STUDY_YEAR_MIN,
    VIEW_POSITIONS,
    PadChestProvider,
    padchest_split_summary,
)
from training.scm import _assert_compatible_checkpoint, _eval_epoch, _model_variable_names
from utils import load_checkpoint_with_path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "shape") and hasattr(value, "tolist"):
        return np.asarray(value).tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _finite_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _finite_json(payload)
    path.write_text(
        json.dumps(serializable, indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: str) -> str | None:
    if path.startswith("gs://") or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_manifest(hparams: Mapping[str, Any], metadata_source: str, metadata_sha256: str | None) -> dict[str, Any]:
    variables = []
    for spec in PAD_CHEST_SCHEMA.variables:
        variables.append(
            {
                "name": spec.name,
                "kind": spec.kind.value if hasattr(spec.kind, "value") else str(spec.kind),
                "encoded_dim": spec.encoded_dim,
                "normalization": spec.normalization,
            }
        )
    return {
        "dataset_id": PAD_CHEST_SCHEMA.dataset_id,
        "schema_version": PAD_CHEST_SCHEMA.version,
        "variable_order": list(PAD_CHEST_SCHEMA.variable_names),
        "encoded_dim": PAD_CHEST_SCHEMA.encoded_dim,
        "variables": variables,
        "edges": [list(edge) for edge in PAD_CHEST_SCHEMA.edges],
        "normalization_constants": {
            "age_at_study": {"min": AGE_AT_STUDY_MIN, "max": AGE_AT_STUDY_MAX, "space": "[-1,1]"},
            "study_year": {"min": STUDY_YEAR_MIN, "max": STUDY_YEAR_MAX, "space": "[-1,1]"},
            "sex": {"categories": ["not_M_or_unknown", "M"]},
            "pediatric": {"positive_when": "Pediatric == yes"},
            "tb_status": {"mode": hparams.get("tb_label_mode", "tb_or_sequelae"), "positive_labels": ["tuberculosis", "tuberculosis sequelae"]},
            "projection": {"categories": list(PROJECTIONS), "fallback": "PA"},
            "view_position": {"categories": list(VIEW_POSITIONS), "fallback": "OTHER"},
        },
        "metadata_uri": hparams.get("metadata"),
        "metadata_source_used_for_validation": metadata_source,
        "metadata_source_sha256": metadata_sha256,
        "image_prefix": hparams.get("image_prefix"),
        "seed": int(hparams.get("seed", 7)),
        "context_norm": hparams.get("context_norm"),
        "context_dim": hparams.get("context_dim"),
    }


def _fingerprint(payload: Mapping[str, Any]) -> str:
    stable = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(stable).hexdigest()


def _stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p01": float(np.quantile(arr, 0.01)),
        "p50": float(np.quantile(arr, 0.50)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def _onehot_indices(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1 or arr.shape[-1] == 1:
        return arr.reshape(-1).astype(np.int64)
    return np.argmax(arr, axis=-1).reshape(-1).astype(np.int64)


def _category_diagnostics(data: np.ndarray, model: np.ndarray, labels: Sequence[str] | None = None) -> dict[str, Any]:
    data_idx = _onehot_indices(data)
    model_idx = _onehot_indices(model)
    n_classes = int(max(data_idx.max(initial=0), model_idx.max(initial=0)) + 1)
    if labels is not None:
        n_classes = max(n_classes, len(labels))
    data_counts = np.bincount(data_idx, minlength=n_classes).astype(np.float64)
    model_counts = np.bincount(model_idx, minlength=n_classes).astype(np.float64)
    data_probs = data_counts / max(1.0, data_counts.sum())
    model_probs = model_counts / max(1.0, model_counts.sum())
    out = {
        "data_counts": data_counts.astype(int).tolist(),
        "model_counts": model_counts.astype(int).tolist(),
        "data_probabilities": data_probs.tolist(),
        "model_probabilities": model_probs.tolist(),
        "total_variation_distance": float(0.5 * np.abs(data_probs - model_probs).sum()),
        "data_zero_support_categories": [int(i) for i, count in enumerate(data_counts) if count == 0],
        "model_zero_support_categories": [int(i) for i, count in enumerate(model_counts) if count == 0],
    }
    if labels is not None:
        out["labels"] = list(labels)
    arr = np.asarray(data)
    if arr.ndim > 1 and arr.shape[-1] > 1:
        row_sum = arr.sum(axis=-1)
        out["data_onehot_row_sum_max_abs_error"] = float(np.max(np.abs(row_sum - 1.0))) if row_sum.size else 0.0
    return out


def _binary_diagnostics(data: np.ndarray, model: np.ndarray) -> dict[str, Any]:
    base = _category_diagnostics(data, model, labels=["0", "1"])
    data_values = np.asarray(data, dtype=np.float64).reshape(-1)
    model_values = np.asarray(model, dtype=np.float64).reshape(-1)
    base.update(
        {
            "data_positive_rate": float(data_values.mean()) if data_values.size else math.nan,
            "model_positive_rate": float(model_values.mean()) if model_values.size else math.nan,
        }
    )
    return base


def _continuous_diagnostics(data: np.ndarray, model: np.ndarray, *, raw_min: float, raw_max: float) -> dict[str, Any]:
    data_arr = np.asarray(data, dtype=np.float64).reshape(-1)
    model_arr = np.asarray(model, dtype=np.float64).reshape(-1)
    to_raw = lambda arr: (arr + 1.0) * 0.5 * (raw_max - raw_min) + raw_min
    return {
        "normalized_data": _stats(data_arr),
        "normalized_model": _stats(model_arr),
        "raw_data": _stats(to_raw(data_arr)),
        "raw_model": _stats(to_raw(model_arr)),
        "data_support_violations": int(np.sum((data_arr < -1.0001) | (data_arr > 1.0001))),
        "model_support_violations": int(np.sum((model_arr < -1.0001) | (model_arr > 1.0001))),
        "mean_abs_difference_normalized": float(abs(np.mean(data_arr) - np.mean(model_arr))) if data_arr.size and model_arr.size else math.nan,
    }


def _ece(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(prob, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not mask.any():
            bins.append({"range": [float(lo), float(hi)], "count": 0})
            continue
        obs = float(y[mask].mean())
        pred = float(p[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(obs - pred)
        bins.append({"range": [float(lo), float(hi)], "count": int(mask.sum()), "observed_rate": obs, "mean_predicted": pred})
    return {"ece_10_uniform_bins": float(ece), "bins": bins}


def _tb_calibration(model: Any, test_samples: Mapping[str, np.ndarray]) -> dict[str, Any]:
    age = jnp.asarray(test_samples["age_at_study"])
    sex = jnp.asarray(test_samples["sex"])
    pediatric = jnp.asarray(test_samples["pediatric"])
    logits = model.tb_logits(age, sex, pediatric)
    prob = np.asarray(jax.nn.sigmoid(logits)).reshape(-1)
    y = np.asarray(test_samples["tb_status"]).reshape(-1)
    return {
        "observed_positive_rate": float(y.mean()),
        "mean_predicted_probability": float(prob.mean()),
        "brier_score": float(np.mean((prob - y) ** 2)),
        **_ece(y, prob),
    }


def _support_by_group(test_samples: Mapping[str, np.ndarray]) -> dict[str, Any]:
    age_raw = (np.asarray(test_samples["age_at_study"]).reshape(-1) + 1.0) * 0.5 * (AGE_AT_STUDY_MAX - AGE_AT_STUDY_MIN) + AGE_AT_STUDY_MIN
    sex = _onehot_indices(np.asarray(test_samples["sex"]))
    pediatric = np.asarray(test_samples["pediatric"]).reshape(-1).astype(int)
    tb = np.asarray(test_samples["tb_status"]).reshape(-1)
    groups: dict[str, Any] = {}
    for label, mask in {
        "age_0_17": age_raw < 18,
        "age_18_39": (age_raw >= 18) & (age_raw < 40),
        "age_40_64": (age_raw >= 40) & (age_raw < 65),
        "age_65_plus": age_raw >= 65,
        "sex_0_not_M_or_unknown": sex == 0,
        "sex_1_M": sex == 1,
        "pediatric_0": pediatric == 0,
        "pediatric_1": pediatric == 1,
    }.items():
        groups[label] = {"count": int(mask.sum()), "tb_positive_rate": float(tb[mask].mean()) if mask.any() else math.nan}
    return groups


def build_artifacts(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    checkpoint, resolved_checkpoint = load_checkpoint_with_path(args.checkpoint)
    hparams = dict(checkpoint.get("hparams", {}))
    seed = int(hparams.get("seed", 7))
    widths = tuple(hparams.get("widths", [64, 64]))

    model = PadChestPGM(widths=widths, rngs=nnx.Rngs(seed))
    graphdef, _ = nnx.split(model, nnx.Param)
    params_template = nnx.state(model, nnx.Param).to_pure_dict()
    _assert_compatible_checkpoint(checkpoint, params_template)
    restored_model = nnx.merge(graphdef, checkpoint.get("ema_params", checkpoint["params"]))

    metadata_source = args.metadata or hparams.get("metadata")
    provider = PadChestProvider(
        hparams.get("data_dir", ""),
        input_res=int(hparams.get("input_res", 128)),
        pad=int(hparams.get("pad", 0)),
        context_norm=hparams.get("context_norm", "[-1,1]"),
        metadata=metadata_source,
        image_prefix=hparams.get("image_prefix", ""),
        tb_label_mode=hparams.get("tb_label_mode", "tb_or_sequelae"),
        seed=seed,
    )
    datasets = {split: provider.load_split(split) for split in ("train", "valid", "test")}
    metadata_sha = _sha256_file(metadata_source)

    schema_payload = _schema_manifest(hparams, metadata_source, metadata_sha)
    split_payload = padchest_split_summary(metadata_source, seed=seed)
    split_payload.update(
        {
            "metadata_uri": hparams.get("metadata"),
            "metadata_source_used_for_validation": metadata_source,
            "schema_version": PAD_CHEST_SCHEMA.version,
            "variable_order": list(PAD_CHEST_SCHEMA.variable_names),
            "normalization_constants": schema_payload["normalization_constants"],
        }
    )
    split_payload["fingerprint_sha256"] = _fingerprint(split_payload)

    rng = np.random.default_rng(seed)
    variable_names = _model_variable_names(restored_model)
    test_metrics = _eval_epoch(graphdef, checkpoint.get("ema_params", checkpoint["params"]), datasets["test"], int(hparams.get("bs", 64)), rng, variable_names)

    test_samples = datasets["test"].samples
    sample_n = len(datasets["test"]) if args.model_samples <= 0 else min(args.model_samples, len(datasets["test"]))
    model_samples = restored_model.sample(sample_n, jax.random.PRNGKey(seed + int(checkpoint.get("step", 0))))
    model_np = {name: np.asarray(value) for name, value in model_samples.items() if name in PAD_CHEST_SCHEMA.variable_names}

    diagnostics = {
        "created_unix": int(time.time()),
        "environment": {
            "python": os.sys.version.split()[0],
            "jax": jax.__version__,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "checkpoint": {
            "requested": args.checkpoint,
            "resolved": resolved_checkpoint,
            "step": int(checkpoint.get("step", -1)),
            "epoch": int(checkpoint.get("epoch", -1)),
            "best_loss": float(checkpoint.get("best_loss", math.nan)),
            "format_version": int(checkpoint.get("format_version", -1)),
            "keys": sorted(checkpoint.keys()),
            "ema_step": int(checkpoint.get("ema_step", -1)),
            "ema_initted": bool(checkpoint.get("ema_initted", False)),
            "model_param_tree_compatible": True,
            "ema_param_tree_compatible": jax.tree.structure(checkpoint.get("ema_params", checkpoint["params"])) == jax.tree.structure(params_template),
            "params_param_tree_compatible": jax.tree.structure(checkpoint["params"]) == jax.tree.structure(params_template),
        },
        "test_likelihood": {key: float(value) for key, value in test_metrics.items()},
        "split_counts": {split: {"rows": len(dataset)} for split, dataset in datasets.items()},
        "per_variable": {
            "age_at_study": _continuous_diagnostics(test_samples["age_at_study"], model_np["age_at_study"], raw_min=AGE_AT_STUDY_MIN, raw_max=AGE_AT_STUDY_MAX),
            "sex": _category_diagnostics(test_samples["sex"], model_np["sex"], labels=["not_M_or_unknown", "M"]),
            "pediatric": _binary_diagnostics(test_samples["pediatric"], model_np["pediatric"]),
            "tb_status": _binary_diagnostics(test_samples["tb_status"], model_np["tb_status"]),
            "projection": _category_diagnostics(test_samples["projection"], model_np["projection"], labels=list(PROJECTIONS)),
            "view_position": _category_diagnostics(test_samples["view_position"], model_np["view_position"], labels=list(VIEW_POSITIONS)),
            "study_year": _continuous_diagnostics(test_samples["study_year"], model_np["study_year"], raw_min=STUDY_YEAR_MIN, raw_max=STUDY_YEAR_MAX),
        },
        "tb_calibration_on_test": _tb_calibration(restored_model, test_samples),
        "tb_support_by_test_group": _support_by_group(test_samples),
        "limitations": [
            "Diagnostics use metadata-derived labels only and do not establish clinical validity.",
            "Model marginal samples are stochastic and use the final EMA checkpoint with a fixed PRNG key.",
            "If metadata_source_used_for_validation is a local cache, remote URI equality is represented by the recorded local SHA-256 and not freshly proven by GCS.",
        ],
    }

    manifest = {
        "created_unix": int(time.time()),
        "run_name": hparams.get("exp_name"),
        "checkpoint": diagnostics["checkpoint"],
        "schema": schema_payload,
        "split": split_payload,
    }
    manifest["manifest_sha256"] = _fingerprint(manifest)

    paths = {
        "manifest": output_dir / "padchest_scm_split_schema_manifest.json",
        "diagnostics": output_dir / "padchest_scm_posttrain_diagnostics.json",
    }
    _write_json(paths["manifest"], manifest)
    _write_json(paths["diagnostics"], diagnostics)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Orbax SCM checkpoint root or step directory to restore.")
    parser.add_argument("--metadata", default="", help="Local metadata CSV or accessible URI. Defaults to checkpoint hparams metadata URI.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON validation artifacts.")
    parser.add_argument("--model-samples", type=int, default=0, help="Model sample count for marginal diagnostics; <=0 matches test rows.")
    return parser.parse_args()


def main() -> None:
    paths = build_artifacts(parse_args())
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
