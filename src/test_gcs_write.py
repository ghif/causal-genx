#!/usr/bin/env python3
"""Verify that the current credentials can write to a GCS prefix."""

from __future__ import annotations

import argparse
import os
import uuid

import fsspec


DEFAULT_GCS_PREFIX = "gs://medical-airnd/causal-gen/checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gcs_prefix",
        nargs="?",
        default=DEFAULT_GCS_PREFIX,
        help=f"GCS directory to test (default: {DEFAULT_GCS_PREFIX})",
    )
    args = parser.parse_args()

    prefix = args.gcs_prefix.rstrip("/")
    if not prefix.startswith("gs://"):
        parser.error("gcs_prefix must start with gs://")

    test_path = f"{prefix}/auth-test-{uuid.uuid4().hex}.txt"
    expected = f"GCS write authorization verified by PID {os.getpid()}\n"
    fs = fsspec.filesystem("gcs")

    print(f"Writing test object: {test_path}")
    try:
        with fsspec.open(test_path, "w") as file:
            file.write(expected)

        with fsspec.open(test_path, "r") as file:
            actual = file.read()

        if actual != expected:
            raise RuntimeError(
                f"GCS read-back mismatch: expected {expected!r}, got {actual!r}"
            )

        print("GCS write/read authorization test passed.")
    finally:
        if fs.exists(test_path):
            fs.rm(test_path)
            print(f"Removed test object: {test_path}")


if __name__ == "__main__":
    main()
