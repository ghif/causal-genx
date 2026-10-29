#!/usr/bin/env python3
"""Probe Google Cloud Storage write and delete permissions for Causal-GenX artifact destinations."""

from __future__ import annotations

import argparse
import sys
import time

DEFAULT_TARGET = "gs://external-cxr-dataset/padchest/checkpoints_v3/probe_write_test.txt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=f"Target GCS URI to test write/delete against (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Do not delete the probe file after writing",
    )
    args = parser.parse_args(argv)

    target = args.target.strip()
    if not target.startswith("gs://"):
        parser.error(f"--target must be a gs:// URI; got: {target!r}")

    try:
        import fsspec
    except ImportError as exc:
        print(f"Error: fsspec/gcsfs not installed in active environment: {exc}", file=sys.stderr)
        return 1

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    payload = f"Causal-GenX GCS write probe at {timestamp}\n".encode("utf-8")

    print(f"Probing GCS write: {target}")
    try:
        with fsspec.open(target, "wb") as f:
            f.write(payload)
        print("  -> Write succeeded.")
    except Exception as exc:
        print(f"  -> Write FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.no_cleanup:
        print("  -> Cleanup skipped (--no-cleanup set).")
        return 0

    print("Cleaning up probe object...")
    try:
        fs = fsspec.filesystem("gcs")
        fs.rm(target)
        print("  -> Probe object removed successfully.")
    except Exception as exc:
        print(f"  -> Cleanup warning: failed to remove {target}: {exc}", file=sys.stderr)
        return 0

    print("All checks passed: GCS bucket is writable and permissions are shipshape.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
