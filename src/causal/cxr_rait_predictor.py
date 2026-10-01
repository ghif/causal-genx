"""Backward-compatibility shim for CXR predictor module.

This module re-exports all components from `causal.cxr_predictor`.
New code should import directly from `causal.cxr_predictor`.
"""

from .cxr_predictor import (
    CxrImageParentPredictor,
    CxrPretrainedImageParentPredictor,
    CxrRaitPretrainedPredictor,
    CxrRaitSupAuxPredictor,
    TorchXRayVisionDenseNet121,
    _DenseBlock,
    _DenseLayer,
    _Transition,
    _conv2d,
)

__all__ = [
    "CxrImageParentPredictor",
    "CxrPretrainedImageParentPredictor",
    "CxrRaitPretrainedPredictor",
    "CxrRaitSupAuxPredictor",
    "TorchXRayVisionDenseNet121",
]
