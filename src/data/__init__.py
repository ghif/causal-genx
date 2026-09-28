"""Dataset implementations and conditioning adapters."""

from .conditioning import ParentEncoder
from .metadata import MetadataEncoder
from .morphomnist import MORPHOMNIST_SCHEMA, MorphoMNIST, MorphoMNISTProvider, _DATASET_FACTORIES, create_dataset, morphomnist
from .cxr_rait import CXR_RAIT_SCHEMA, CxrRaitDataset, CxrRaitProvider, cxr_rait
from .padchest import (
    PAD_CHEST_SCHEMA,
    PAD_CHEST_V2_SCHEMA,
    PadChestDataset,
    PadChestProvider,
    audit_padchest_v2_rows,
    derive_padchest_v2_row,
    padchest,
)

_DATASET_FACTORIES["cxr_rait"] = CxrRaitProvider
_DATASET_FACTORIES["padchest"] = PadChestProvider

__all__ = [
    "ParentEncoder",
    "MORPHOMNIST_SCHEMA",
    "MetadataEncoder",
    "MorphoMNIST",
    "MorphoMNISTProvider",
    "CXR_RAIT_SCHEMA",
    "CxrRaitDataset",
    "CxrRaitProvider",
    "PAD_CHEST_SCHEMA",
    "PAD_CHEST_V2_SCHEMA",
    "audit_padchest_v2_rows",
    "derive_padchest_v2_row",
    "PadChestDataset",
    "PadChestProvider",
    "create_dataset",
    "morphomnist",
    "cxr_rait",
    "padchest",
]
