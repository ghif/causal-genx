import argparse
import os
import runpy
import sys

from xla_runtime import launch


def _run(entrypoint: str, arguments):
    os.environ["CAUSAL_GEN_XLA_WORKER"] = "1"
    sys.path.insert(0, os.path.abspath(os.path.dirname(entrypoint)))
    sys.argv = [entrypoint, *arguments]
    runpy.run_path(entrypoint, run_name="__main__")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch a causal-gen entrypoint on all local PJRT TPU devices."
    )
    parser.add_argument(
        "entrypoint",
        choices=["main.py", "benchmark.py", "pgm/train_pgm.py", "pgm/train_cf.py"],
    )
    known, arguments = parser.parse_known_args()
    launch(_run, args=(known.entrypoint, arguments))
