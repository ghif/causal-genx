"""Reusable early stopping tracker for Causal-GenX training loops."""

from __future__ import annotations

import math
from typing import Literal


class EarlyStopping:
    """Monitors a validation metric and signals when improvement has plateaued.

    Patience counts the number of consecutive evaluated validation checks without
    an improvement of at least ``min_delta``.
    """

    def __init__(
        self,
        patience: int = 0,
        min_delta: float = 0.0,
        mode: Literal["min", "max"] = "min",
    ) -> None:
        self.patience = max(0, int(patience or 0))
        self.min_delta = float(min_delta or 0.0)
        self.mode = str(mode or "min").lower()
        if self.mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got: {mode!r}")

        self.best_score: float | None = None
        self.wait: int = 0
        self.stopped_epoch: int = 0
        self.stop: bool = False

    @property
    def enabled(self) -> bool:
        return self.patience > 0

    def step(self, current_score: float, epoch: int) -> bool:
        """Evaluate the current validation score and return True if training should halt.

        Parameters
        ----------
        current_score:
            The newly computed validation metric (e.g. loss or negative ELBO).
        epoch:
            The current epoch number (1-based or 0-based, recorded if stopped).

        Returns
        -------
        bool:
            True if patience has been exhausted and training should stop; False otherwise.
        """
        if not self.enabled:
            return False

        current = float(current_score)
        if math.isnan(current) or math.isinf(current):
            return False

        if self.best_score is None:
            self.best_score = current
            self.wait = 0
            return False

        if self.mode == "min":
            improvement = (self.best_score - current) > self.min_delta
        else:
            improvement = (current - self.best_score) > self.min_delta

        if improvement:
            self.best_score = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop = True
                self.stopped_epoch = epoch
                return True

        return False

    def state_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "best_score": self.best_score,
            "wait": self.wait,
            "stopped_epoch": self.stopped_epoch,
            "stop": self.stop,
        }
