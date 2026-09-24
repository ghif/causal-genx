#!/usr/bin/env python3
"""Plan or populate a bounded local staging tree for PadChest predictor PNGs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="PadChest predictor YAML")
    parser.add_argument("--execute", action="store_true", help="copy images; default only reports the plan")
    parser.add_argument("--splits", default="train,valid,test", help="comma-separated splits to stage")
    parser.add_argument("--limit", type=int, default=0, help="optional small item limit for smoke tests")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON only")
    args, overrides = parser.parse_known_args(argv)
    if any(item.startswith("-") for item in overrides):
        parser.error(f"unrecognized arguments: {' '.join(overrides)}")

    from config import load_experiment
    from data.padchest import stage_padchest_images
    from training.predictor import _run_arguments

    config = load_experiment(args.config, overrides)
    if config.dataset.name != "padchest":
        parser.error("PadChest staging requires dataset.name=padchest")
    if config.workflow.type not in {"train-predictor", "finetune-predictor"}:
        parser.error("PadChest staging currently targets predictor workflows")
    splits = tuple(split.strip() for split in args.splits.split(",") if split.strip())
    summary = stage_padchest_images(
        _run_arguments(config),
        splits=splits,
        execute=args.execute,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps(summary, sort_keys=True))
        return 2 if summary.get("refusal") else 0

    action = "execute" if args.execute else "plan"
    print(f"padchest_input_staging {action}")
    print(f"  stage_dir:       {summary['stage_dir']}")
    print(f"  manifest:        {summary['stage_manifest']}")
    print(f"  splits:          {','.join(summary['splits'])} rows={summary['split_rows']}")
    print(f"  required_items:  {summary['required_items']} (max {summary['max_items'] or 'unbounded'})")
    print(f"  estimated_bytes: {summary['estimated_bytes']} (max {summary['max_bytes'] or 'unbounded'}, unknown {summary['unknown_sizes']}, basis {summary.get('size_basis', 'n/a')})")
    print(f"  transfer:        {summary.get('transfer_method', 'python-copy')}")
    if summary.get("transfer_prerequisite"):
        print(f"  prerequisite:    {summary['transfer_prerequisite']}")
    if summary.get("refusal"):
        print(f"  REFUSED:         {summary['refusal']}")
        print("Adjust workflow.input_stage_max_items/input_stage_max_bytes or choose fewer splits before using --execute.")
        return 2
    if args.execute:
        print(f"  copied:          {summary['copied']}")
        print(f"  reused:          {summary['reused']}")
        print(f"  throughput:      {summary.get('items_per_second', 0.0):.2f} items/s, {summary.get('bytes_per_second', 0.0):.0f} bytes/s")
    else:
        print("  next:            rerun with --execute to populate the local stage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
