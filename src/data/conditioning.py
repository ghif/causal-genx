"""Schema-driven parent construction at the image-model boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from contracts import CausalGraphSpec


@dataclass(frozen=True)
class ParentEncoder:
    schema: CausalGraphSpec

    def vector(self, variables: Mapping[str, np.ndarray]) -> np.ndarray:
        missing = set(self.schema.variable_names).difference(variables)
        if missing:
            raise ValueError(f"Variables missing from batch: {sorted(missing)}")
        parts = []
        for spec in self.schema.variables:
            value = np.asarray(variables[spec.name], dtype=np.float32)
            value = value[:, None] if value.ndim == 1 else value.reshape(value.shape[0], -1)
            if value.shape[1] != spec.encoded_dim:
                raise ValueError(f"{spec.name}: expected encoded_dim={spec.encoded_dim}, got {value.shape[1]}")
            parts.append(value)
        return np.concatenate(parts, axis=1)

    def spatial(self, variables: Mapping[str, np.ndarray], height: int, width: int) -> np.ndarray:
        vector = self.vector(variables)
        return np.broadcast_to(vector[:, :, None, None], (vector.shape[0], vector.shape[1], height, width)).copy()

