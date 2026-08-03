#!/usr/bin/env python3
"""Convert pre-trained TorchXRayVision PyTorch weights to Flax/JAX format."""

import argparse
import os
import sys
import numpy as np


def convert_torchxrayvision_to_flax(weights_name: str = "densenet121-res224-all", output_path: str = "checkpoints/pretrained/torchxrayvision_densenet121_flax.npz"):
    import torch
    import torchxrayvision as xrv

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Loading pre-trained TorchXRayVision model (weights='{weights_name}')...")
    model = xrv.models.DenseNet(weights=weights_name)
    state_dict = model.state_dict()

    flax_weights = {}
    print("Converting PyTorch tensors to Flax/JAX format...")

    for key, tensor in state_dict.items():
        arr = tensor.cpu().detach().numpy()

        # Handle Conv2D weights: (out_channels, in_channels, H, W) -> (H, W, in_channels, out_channels)
        if "conv" in key and "weight" in key and arr.ndim == 4:
            arr = np.transpose(arr, (2, 3, 1, 0))

        # Handle Linear / FC weights: (out_features, in_features) -> (in_features, out_features)
        elif ("classifier" in key or "fc" in key) and "weight" in key and arr.ndim == 2:
            arr = np.transpose(arr, (1, 0))

        flax_weights[key] = arr

    np.savez_compressed(output_path, **flax_weights)
    print(f"✅ Successfully converted {len(flax_weights)} layers!")
    print(f"Saved JAX/Flax checkpoint to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert TorchXRayVision PyTorch weights to JAX/Flax format.")
    parser.add_argument("--weights", type=str, default="densenet121-res224-all", help="TorchXRayVision weights specifier")
    parser.add_argument("--output", type=str, default="checkpoints/pretrained/torchxrayvision_densenet121_flax.npz", help="Output .npz file path")
    args = parser.parse_args()

    convert_torchxrayvision_to_flax(weights_name=args.weights, output_path=args.output)


if __name__ == "__main__":
    main()
