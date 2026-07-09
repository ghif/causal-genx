import argparse
import os
import runpy
import sys

from xla_runtime import launch


def _run(_ordinal: int, entrypoint: str, arguments):
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
    parser.add_argument(
        "--debug_single_process",
        action="store_true",
        help="Run one TPU process to isolate model/compile failures from collectives.",
    )
    known, arguments = parser.parse_known_args()
    launch(
        _run,
        args=(known.entrypoint, arguments),
        debug_single_process=known.debug_single_process,
    )
