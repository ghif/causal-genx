#!/usr/bin/env python3
"""Single researcher/developer entrypoint for Causal-GenX workflows."""

from __future__ import annotations

import argparse
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
    # parse_known_args deliberately permits overrides after normal options,
    # matching the documented `workflow.checkpoint=...` invocation style.
    args, overrides = parser.parse_known_args(argv)
    if any(item.startswith("-") for item in overrides):
        parser.error(f"unrecognized arguments: {' '.join(overrides)}")
    from config import load_experiment
    from runtime import configure_backend
    config = load_experiment(args.config, overrides)
    configure_backend(config.runtime.accelerator, config.runtime.gpu_id)
    if args.dry_run:
        from training.common import legacy_run_dir
        output = legacy_run_dir(config) if args.command in {"train-scm", "train-predictor", "train-image-model"} else config.artifacts.root
        print(f"validated stage={config.workflow.type} output={output}")
        return 0
    from training.counterfactual import run as finetune_counterfactual
    from training.image_model import run as train_image_model
    from training.inference import run as infer
    from training.predictor import run as train_predictor
    from training.scm import run as train_scm
    stages = {"train-scm": train_scm, "train-predictor": train_predictor, "train-image-model": train_image_model, "finetune-counterfactual": finetune_counterfactual, "infer": infer}
    if args.command != config.workflow.type:
        parser.error(f"command {args.command!r} does not match config workflow.type={config.workflow.type!r}")
    print(stages[args.command](config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
