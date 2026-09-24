"""Dataset implementations and conditioning adapters."""

from .conditioning import ParentEncoder
from .morphomnist import MORPHOMNIST_SCHEMA, MorphoMNIST, MorphoMNISTProvider, _DATASET_FACTORIES, create_dataset, morphomnist
from .cxr_rait import CXR_RAIT_SCHEMA, CxrRaitDataset, CxrRaitProvider, cxr_rait
from .padchest import PAD_CHEST_SCHEMA, PadChestDataset, PadChestProvider, padchest

_DATASET_FACTORIES["cxr_rait"] = CxrRaitProvider
_DATASET_FACTORIES["padchest"] = PadChestProvider

__all__ = [
    "ParentEncoder",
    "MORPHOMNIST_SCHEMA",
    "MorphoMNIST",
    "MorphoMNISTProvider",
    "CXR_RAIT_SCHEMA",
    "CxrRaitDataset",
    "CxrRaitProvider",
    "PAD_CHEST_SCHEMA",
    "PadChestDataset",
    "PadChestProvider",
    "create_dataset",
    "morphomnist",
    "cxr_rait",
    "padchest",
]
