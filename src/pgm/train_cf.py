"""Deprecated compatibility wrapper for counterfactual fine-tuning."""

from __future__ import annotations

import sys
import warnings


if __name__ == "__main__":
    from runtime import configure_backend_from_argv

    configure_backend_from_argv()

    from hps import setup_hparams
    from training import counterfactual as _counterfactual

    warnings.warn(
        "src/pgm/train_cf.py is deprecated; use scripts/run.py "
        "finetune-counterfactual --config ...",
        DeprecationWarning,
        stacklevel=1,
    )
    _counterfactual.run_legacy_args(
        setup_hparams(_counterfactual.legacy_argument_parser())
    )
else:
    # Preserve imports and parity tests that historically targeted this module,
    # while keeping the implementation in the named Stage 4 module.
    from training import counterfactual as _counterfactual

    sys.modules[__name__] = _counterfactual
