"""Deprecated compatibility entrypoint for image-model training.

Use ``scripts/run.py train-image-model --config ...`` for new experiments.
This module remains available for the historical shell launchers and delegates
all execution to the native image-model stage.
"""

from __future__ import annotations

import argparse
import warnings

from hps import add_arguments, setup_hparams


def main(args):
    """Delegate legacy CLI arguments to the native image-model stage."""
    from training.image_model import run_legacy_args

    return run_legacy_args(args)


if __name__ == "__main__":
    warnings.warn(
        "src/main.py is a compatibility entrypoint; use scripts/run.py "
        "train-image-model --config ...",
        DeprecationWarning,
        stacklevel=1,
    )
    parser = add_arguments(argparse.ArgumentParser())
    main(setup_hparams(parser))
