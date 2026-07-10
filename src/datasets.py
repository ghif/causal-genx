from __future__ import annotations

import gzip
import os
import random
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from hps import Hparams
from utils import log_standardize, normalize


def _open_binary(path: str):
    if path.startswith("gs://"):
        import fsspec

        return fsspec.open(path, mode="rb").open()
    return open(path, "rb")


def _open_image(path: str) -> Image.Image:
    with _open_binary(path) as f:
        img = Image.open(f)
        img.load()
        return img.copy()


def _load_uint8(f):
    idx_dtype, ndim = struct.unpack("BBBB", f.read(4))[2:]
    shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
    buffer_length = int(np.prod(shape))
    data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data


def load_idx(path: str) -> np.ndarray:
    with _open_binary(path) as f:
        if path.endswith(".gz"):
            with gzip.GzipFile(fileobj=f, mode="rb") as gz:
                return _load_uint8(gz)
        return _load_uint8(f)


def _get_paths(root_dir, train):
    prefix = "train" if train else "t10k"
    return (
        os.path.join(root_dir, prefix + "-images-idx3-ubyte.gz"),
        os.path.join(root_dir, prefix + "-labels-idx1-ubyte.gz"),
        os.path.join(root_dir, prefix + "-morpho.csv"),
    )


def load_morphomnist_like(root_dir, train: bool = True, columns=None):
    images_path, labels_path, metrics_path = _get_paths(root_dir, train)
    images = load_idx(images_path)
    labels = load_idx(labels_path)
    usecols = ["index"] + list(columns) if columns is not None and "index" not in columns else columns
    with _open_binary(metrics_path) as f:
        metrics = pd.read_csv(f, usecols=usecols, index_col="index")
    return images, labels, metrics


class MorphoMNIST:
    def __init__(
        self,
        root_dir: str,
        train: bool = True,
        transform=None,
        columns: Optional[List[str]] = None,
        norm: Optional[str] = None,
        concat_pa: bool = True,
    ):
        self.train = train
        self.transform = transform
        self.columns = columns
        self.concat_pa = concat_pa
        self.norm = norm
        cols_not_digit = [c for c in self.columns if c != "digit"]
        images, labels, metrics_df = load_morphomnist_like(root_dir, train, cols_not_digit)
        self.images = np.asarray(images)
        self.labels = np.eye(10, dtype=np.float32)[np.asarray(labels)]
        if self.columns is None:
            self.columns = list(metrics_df.columns)
        self.samples = {k: np.asarray(metrics_df[k], dtype=np.float32) for k in cols_not_digit}
        self.min_max = {"thickness": [0.87598526, 6.255515], "intensity": [66.601204, 254.90317]}
        for k, v in list(self.samples.items()):
            if norm == "[-1,1]":
                self.samples[k] = np.asarray(normalize(v, x_min=self.min_max[k][0], x_max=self.min_max[k][1]))
            elif norm == "[0,1]":
                self.samples[k] = np.asarray(normalize(v, x_min=self.min_max[k][0], x_max=self.min_max[k][1], zero_one=True))
        self.samples["digit"] = self.labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        sample = {"x": self.images[idx][None, ...].astype(np.float32)}
        if self.transform is not None:
            sample["x"] = self.transform(sample["x"])
        if self.concat_pa:
            sample["pa"] = np.concatenate(
                [(self.samples[k][idx][None] if k != "digit" else self.samples[k][idx]) for k, v in self.samples.items()],
                axis=0,
            ).astype(np.float32)
        else:
            sample.update({k: v[idx] for k, v in self.samples.items()})
        return sample


def _train_transform(x, pad=4, input_res=32):
    img = Image.fromarray(np.squeeze(x).astype(np.uint8))
    if pad:
        img = ImageOps.expand(img, border=pad, fill=0)
    if img.size != (input_res, input_res):
        top = random.randint(0, max(0, img.size[1] - input_res))
        left = random.randint(0, max(0, img.size[0] - input_res))
        img = img.crop((left, top, left + input_res, top + input_res))
    return np.asarray(img, dtype=np.float32)[None, ...]


def _eval_transform(x, pad=2, input_res=32):
    img = Image.fromarray(np.squeeze(x).astype(np.uint8))
    if pad:
        img = ImageOps.expand(img, border=pad, fill=0)
    if img.size != (input_res, input_res):
        img = img.resize((input_res, input_res), resample=Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32)[None, ...]


def morphomnist(args: Hparams) -> Dict[str, MorphoMNIST]:
    if not args.data_dir:
        args.data_dir = "gs://medical-airnd/causal-gen/datasets/morphomnist"
    datasets = {}
    for split in ["train", "valid", "test"]:
        datasets[split] = MorphoMNIST(
            root_dir=args.data_dir,
            train=(split == "train"),
            transform=(lambda x, split=split: _train_transform(x, pad=args.pad, input_res=args.input_res))
            if split == "train"
            else (lambda x, split=split: _eval_transform(x, pad=2, input_res=args.input_res)),
            columns=args.parents_x,
            norm=args.context_norm,
            concat_pa=args.concat_pa,
        )
    return datasets
