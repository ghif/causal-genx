#!/usr/bin/env python3
"""Single researcher/developer entrypoint for Causal-GenX workflows."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Causal-GenX experiment runner")
    parser.add_argument("command", choices=("train-scm", "train-predictor", "train-image-model", "finetune-counterfactual", "infer"))
    parser.add_argument("--config", required=True, help="Fully resolved experiment YAML")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and report selected workflow without training")
    parser.add_argument("--dry-run-image", action="store_true", help="Build the image model and write one visualization without training")
    # parse_known_args deliberately permits overrides after normal options,
    # matching the documented `workflow.checkpoint=...` invocation style.
    args, overrides = parser.parse_known_args(argv)
    if any(item.startswith("-") for item in overrides):
        parser.error(f"unrecognized arguments: {' '.join(overrides)}")
    from config import load_experiment
    from runtime import configure_backend
    config = load_experiment(args.config, overrides)
    if args.command != config.workflow.type:
        parser.error(f"command {args.command!r} does not match config workflow.type={config.workflow.type!r}")
    if args.dry_run_image and args.command != "train-image-model":
        parser.error("--dry-run-image is only supported for train-image-model")
    configure_backend(config.runtime.accelerator, config.runtime.gpu_id)
    if args.dry_run:
        output = (
            ROOT / config.artifacts.root / config.dataset.name / config.artifacts.run_name
            if args.command in {"train-scm", "train-predictor", "train-image-model"}
            else config.artifacts.root
        )
        print(f"validated stage={config.workflow.type} output={output}")
        return 0
    if args.dry_run_image:
        from training.image_model import dry_run_image
        print(dry_run_image(config), flush=True)
        return 0
    from runtime import validate_backend
    summary = validate_backend(
        config.runtime.accelerator,
        expected_local_device_count=config.runtime.expected_local_device_count,
        expected_global_device_count=config.runtime.expected_global_device_count,
        expected_process_count=config.runtime.expected_process_count,
    )
    print(
        "runtime_preflight "
        f"backend={summary.backend} device_kind={summary.device_kind} "
        f"local_devices={summary.local_device_count} global_devices={summary.global_device_count} "
        f"processes={summary.process_count} process_index={summary.process_index} jax={summary.jax_version}",
        flush=True,
    )
    stage_modules = {
        "train-scm": "training.scm",
        "train-predictor": "training.predictor",
        "train-image-model": "training.image_model",
        "finetune-counterfactual": "training.counterfactual",
        "infer": "training.inference",
    }
    # Import only the selected stage, after runtime initialization. This avoids
    # unrelated model modules initializing JAX or accelerator libraries.
    stage = importlib.import_module(stage_modules[args.command]).run
    print(stage(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
