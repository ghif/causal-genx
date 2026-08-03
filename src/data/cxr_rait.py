from __future__ import annotations

import hashlib
import io
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from utils import normalize
from contracts import Batch, CausalGraphSpec, DatasetSpec, ImageSpec, VariableKind, VariableSpec


_GCS_CLIENT = None


def _get_gcs_client():
    global _GCS_CLIENT
    if _GCS_CLIENT is None:
        import warnings
        from google.cloud import storage

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            _GCS_CLIENT = storage.Client()
    return _GCS_CLIENT


def _open_binary(path: str):
    if path.startswith("gs://"):
        try:
            clean_path = path.removeprefix("gs://")
            bucket_name, blob_name = clean_path.split("/", 1)
            client = _get_gcs_client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            return io.BytesIO(blob.download_as_bytes())
        except Exception:
            import fsspec

            return fsspec.open(path, mode="rb").open()
    return open(path, "rb")


def _load_dicom_image(path: str, target_res: int = 128) -> np.ndarray:
    """Load DICOM file, handle VOI LUT windowing, MONOCHROME inversion, and resize."""
    import pydicom
    from PIL import Image

    with _open_binary(path) as f:
        bytes_data = f.read()
    ds = pydicom.dcmread(io.BytesIO(bytes_data))
    pixel_array = ds.pixel_array.astype(np.float32)

    # Rescale Slope & Intercept
    slope = getattr(ds, "RescaleSlope", 1.0)
    intercept = getattr(ds, "RescaleIntercept", 0.0)
    pixel_array = pixel_array * float(slope) + float(intercept)

    # Handle Photometric Interpretation
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        pixel_array = np.max(pixel_array) - pixel_array

    # Normalize to [0, 255] for PIL resizing
    p_min, p_max = pixel_array.min(), pixel_array.max()
    if p_max > p_min:
        norm_img = ((pixel_array - p_min) / (p_max - p_min) * 255.0).astype(np.uint8)
    else:
        norm_img = np.zeros_like(pixel_array, dtype=np.uint8)

    pil_img = Image.fromarray(norm_img)
    if pil_img.size != (target_res, target_res):
        pil_img = pil_img.resize((target_res, target_res), resample=Image.Resampling.BILINEAR)

    # Scale to [0.0, 1.0] float32 array
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    if torchxray_norm:
        arr = (arr * 2048.0) - 1024.0
    return arr


def preprocess_cxr_image(img: np.ndarray, target_res: int = 224, torchxray_norm: bool = False) -> np.ndarray:
    """Resize and normalize chest X-ray image to match TorchXRayVision expectations."""
    from PIL import Image
    if img.ndim == 2:
        img = img[:, :, None]
    if img.shape[0] == 1 and img.ndim == 3:
        img = img[0]
    if img.shape[:2] != (target_res, target_res):
        pil_img = Image.fromarray((img * 255.0).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8))
        pil_img = pil_img.resize((target_res, target_res), resample=Image.Resampling.BILINEAR)
        img = np.asarray(pil_img, dtype=np.float32) / 255.0
    if torchxray_norm:
        img = (img * 2048.0) - 1024.0
    if img.ndim == 2:
        img = img[None, :, :]
    return img.astype(np.float32)



def _load_cxr_rait_metadata(metadata_path: str) -> Dict[str, Dict[str, float]]:
    """Parse demography Excel spreadsheet and return patient metadata dict."""
    import pandas as pd

    with _open_binary(metadata_path) as f:
        df = pd.read_excel(f, sheet_name=0)

    metadata = {}
    for _, row in df.iterrows():
        pid = str(row.get("Patient ID", "")).strip()
        if not pid or pid == "nan":
            continue

        # Parse Age across possible shifted columns ('Usia', 'Birth', 'Unnamed: 15')
        age = None
        for candidate_col in ["Usia", "Birth", "Unnamed: 15"]:
            val_raw = row.get(candidate_col, None)
            try:
                if val_raw is not None and not pd.isna(val_raw):
                    val = float(val_raw)
                    if 0.0 < val < 110.0:
                        age = val
                        break
            except (ValueError, TypeError):
                pass
        if age is None:
            age = 45.0  # Median fallback if missing

        # Parse Gender
        gender_raw = row.get("Gender", "")
        g_str = str(gender_raw).strip().upper()
        gender = 1.0 if g_str in ("1", "1.0", "M", "L", "MALE", "LAKI-LAKI") else 0.0

        # Parse TB status ('BTA.1' or 'BTA')
        tb_status = 0.0
        bta1_raw = row.get("BTA.1", None)
        try:
            if bta1_raw is not None and not pd.isna(bta1_raw):
                tb_status = 1.0 if float(bta1_raw) == 1.0 else 0.0
        except (ValueError, TypeError):
            bta0_str = str(row.get("BTA", "0")).strip()
            tb_status = 1.0 if bta0_str in ("1", "+") else 0.0

        metadata[pid] = {
            "age": age,
            "gender": gender,
            "tb_status": tb_status,
        }
    return metadata



def apply_cxr_augmentations(
    imgs: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    hflip_prob: float = 0.5,
    max_rotation_deg: float = 7.5,
    max_translation_ratio: float = 0.05,
    max_scale_ratio: float = 0.08,
    contrast_range: Tuple[float, float] = (0.88, 1.12),
    brightness_range: Tuple[float, float] = (-0.05, 0.05),
) -> np.ndarray:
    """Apply clinically suitable chest X-ray image augmentations.

    Augmentations:
    1. Random Horizontal Flip (p=0.5)
    2. Small Random Rotation (±7.5°)
    3. Small Random Scale (0.92x - 1.08x) & Translation (±5%)
    4. Random Contrast & Brightness Adjustment
    """
    if rng is None:
        rng = np.random.default_rng()

    is_batch = imgs.ndim == 4
    if not is_batch:
        imgs = imgs[None, ...]

    B, C, H, W = imgs.shape
    augmented = np.empty_like(imgs)

    try:
        from scipy.ndimage import affine_transform, rotate
        has_scipy = True
    except Exception:
        has_scipy = False

    for b in range(B):
        img = imgs[b].copy()  # (C, H, W)

        # 1. Random Horizontal Flip
        if rng.random() < hflip_prob:
            img = np.flip(img, axis=-1)

        if has_scipy:
            # 2. Random Rotation
            angle = rng.uniform(-max_rotation_deg, max_rotation_deg)
            if abs(angle) > 0.1:
                for c in range(C):
                    img[c] = rotate(img[c], angle, reshape=False, mode="nearest", order=1)

            # 3. Random Scale & Translation
            scale = rng.uniform(1.0 - max_scale_ratio, 1.0 + max_scale_ratio)
            trans_x = rng.uniform(-max_translation_ratio, max_translation_ratio) * W
            trans_y = rng.uniform(-max_translation_ratio, max_translation_ratio) * H

            if abs(scale - 1.0) > 0.01 or abs(trans_x) > 0.5 or abs(trans_y) > 0.5:
                center_x, center_y = W / 2.0, H / 2.0
                matrix = np.array([[1.0 / scale, 0.0], [0.0, 1.0 / scale]])
                offset = np.array([center_y - (center_y - trans_y) / scale, center_x - (center_x - trans_x) / scale])
                for c in range(C):
                    img[c] = affine_transform(img[c], matrix=matrix, offset=offset, mode="nearest", order=1)

        # 4. Random Contrast & Brightness Adjustment
        contrast = rng.uniform(contrast_range[0], contrast_range[1])
        brightness = rng.uniform(brightness_range[0], brightness_range[1])
        img = img * contrast + brightness
        img = np.clip(img, 0.0, 1.0)

        augmented[b] = img

    return augmented if is_batch else augmented[0]


class CxrRaitDataset:
    """CXR-RAIT Chest X-ray dataset containing DICOM images and tabular demographic metadata."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        columns: Optional[List[str]] = None,
        norm: Optional[str] = "[-1,1]",
        concat_pa: bool = True,
        input_res: int = 128,
        augment: bool = False,
        torchxray_preprocessing: bool = False,
        seed: int = 7,
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.columns = columns or ["gender", "age", "tb_status"]
        self.norm = norm
        self.concat_pa = concat_pa
        self.input_res = input_res
        self.augment = augment
        self.torchxray_preprocessing = torchxray_preprocessing


        metadata_path = os.path.join(root_dir, "data_demography.xlsx")
        meta_dict = _load_cxr_rait_metadata(metadata_path)

        # Index DICOM image paths
        all_patient_ids = sorted(list(meta_dict.keys()))
        rng = random.Random(seed)
        rng.shuffle(all_patient_ids)

        n = len(all_patient_ids)
        n_train = int(0.8 * n)
        n_valid = int(0.1 * n)

        if split == "train":
            patient_set = set(all_patient_ids[:n_train])
        elif split == "valid":
            patient_set = set(all_patient_ids[n_train : n_train + n_valid])
        else:
            patient_set = set(all_patient_ids[n_train + n_valid :])

        # Discover actual DICOM file paths under root_dir
        dicom_map = {}
        if root_dir.startswith("gs://"):
            try:
                client = _get_gcs_client()
                clean_path = root_dir.removeprefix("gs://")
                bucket_name, prefix = clean_path.split("/", 1) if "/" in clean_path else (clean_path, "")
                bucket = client.bucket(bucket_name)
                for blob in bucket.list_blobs(prefix=prefix):
                    if blob.name.endswith(".dcm"):
                        fname = os.path.basename(blob.name)
                        pid = fname.removesuffix(".dcm").strip()
                        dicom_map[pid] = f"gs://{bucket_name}/{blob.name}"
            except Exception:
                pass
        elif os.path.exists(root_dir):
            from pathlib import Path
            for p in Path(root_dir).rglob("*.dcm"):
                pid = p.name.removesuffix(".dcm").strip()
                dicom_map[pid] = str(p)

        self.samples_meta = []
        self.image_paths = []

        for pid in patient_set:
            meta = meta_dict[pid]
            if pid in dicom_map:
                full_path = dicom_map[pid]
            else:
                full_path = os.path.join(root_dir, f"{pid}.dcm")
            self.image_paths.append(full_path)
            self.samples_meta.append((pid, meta))

        self.num_samples = len(self.samples_meta)
        if self.num_samples == 0:
            # Fallback synthetic entries if files not downloaded in local test mode
            for pid in list(patient_set)[:10]:
                self.samples_meta.append((pid, meta_dict[pid]))
                self.image_paths.append(os.path.join(root_dir, f"{pid}.dcm"))
            self.num_samples = len(self.samples_meta)

        # Extract variable arrays
        ages = np.array([m["age"] for _, m in self.samples_meta], dtype=np.float32)
        genders = np.array([m["gender"] for _, m in self.samples_meta], dtype=np.float32)
        tbs = np.array([m["tb_status"] for _, m in self.samples_meta], dtype=np.float32)

        # Age normalization
        self.age_min, self.age_max = 0.0, 100.0
        if norm == "[-1,1]":
            ages_norm = normalize(ages, x_min=self.age_min, x_max=self.age_max)
        elif norm == "[0,1]":
            ages_norm = normalize(ages, x_min=self.age_min, x_max=self.age_max, zero_one=True)
        else:
            ages_norm = ages

        # One-hot encodings
        gender_onehot = np.eye(2, dtype=np.float32)[genders.astype(int)]
        tb_onehot = np.eye(2, dtype=np.float32)[tbs.astype(int)]

        self.samples = {
            "age": ages_norm,
            "gender": gender_onehot,
            "tb_status": tb_onehot,
        }

        self._image_cache: Dict[int, np.ndarray] = {}
        self.cache_dir = os.path.expanduser(f"~/.cache/cxr_rait_{self.input_res}")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._preload_images()

    def _preload_images(self, max_workers: int = 16):
        from concurrent.futures import ThreadPoolExecutor

        def _fetch_one(idx: int):
            self._get_image(idx)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(_fetch_one, range(self.num_samples)))

    def __len__(self):
        return self.num_samples

    def _get_image(self, idx: int) -> np.ndarray:
        if idx in self._image_cache:
            return self._image_cache[idx]

        pid, _ = self.samples_meta[idx]
        disk_cache_file = os.path.join(self.cache_dir, f"{pid}.npy")
        if os.path.exists(disk_cache_file):
            try:
                res = np.load(disk_cache_file)
                self._image_cache[idx] = res
                return res
            except Exception:
                pass

        path = self.image_paths[idx]
        try:
            img = _load_dicom_image(path, target_res=self.input_res, torchxray_norm=self.torchxray_preprocessing)
        except Exception:

            # Synthetic fallback blank image for dry-run/mock tests
            img = np.zeros((self.input_res, self.input_res), dtype=np.float32)
        res = img[None, ...]  # 1 x H x W
        self._image_cache[idx] = res
        try:
            np.save(disk_cache_file, res)
        except Exception:
            pass
        return res

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        img = self._get_image(idx)
        sample = {"x": img}
        if self.transform is not None:
            sample["x"] = self.transform(sample["x"])
        if self.concat_pa:
            parts = []
            for k in ["age", "gender", "tb_status"]:
                v = self.samples[k][idx]
                if v.ndim == 0:
                    v = v[None]
                elif k == "age" and v.ndim == 1 and len(v) == 1:
                    pass
                elif k == "age" and v.ndim == 0:
                    v = v[None]
                parts.append(v if v.ndim >= 1 else v[None])
            sample["pa"] = np.concatenate(parts, axis=-1).astype(np.float32)
        else:
            sample.update({k: self.samples[k][idx] for k in ["age", "gender", "tb_status"]})
        return sample

    def make_batch(self, batch_idx, rng=None, shuffle: bool = False):
        batch_idx = np.asarray(batch_idx, dtype=np.int64)
        imgs = [self._get_image(i) for i in batch_idx]
        imgs_arr = np.stack(imgs, axis=0)

        if self.augment and (self.split == "train" or shuffle):
            imgs_arr = apply_cxr_augmentations(imgs_arr, rng=rng)

        sample = {"x": imgs_arr}
        if self.concat_pa:
            parts = []
            for k in ["age", "gender", "tb_status"]:
                v = np.asarray(self.samples[k][batch_idx], dtype=np.float32)
                if k == "age" and v.ndim == 1:
                    v = v[:, None]
                parts.append(v)
            sample["pa"] = np.concatenate(parts, axis=1).astype(np.float32)
        else:
            sample.update({k: np.asarray(self.samples[k][batch_idx], dtype=np.float32) for k in ["age", "gender", "tb_status"]})
        return sample


CXR_RAIT_SCHEMA = CausalGraphSpec(
    dataset_id="cxr_rait",
    version="1",
    variables=(
        VariableSpec("age", VariableKind.CONTINUOUS, normalization="[-1,1]"),
        VariableSpec("gender", VariableKind.CATEGORICAL, encoded_dim=2),
        VariableSpec("tb_status", VariableKind.CATEGORICAL, encoded_dim=2),
    ),
    edges=(("age", "tb_status"),),
)


class CxrRaitProvider:
    schema = CXR_RAIT_SCHEMA

    def __init__(self, root: str, input_res: int = 128, pad: int = 0, context_norm: str = "[-1,1]"):
        self.root = root
        self.input_res = input_res
        self.pad = pad
        self.context_norm = context_norm

    @property
    def spec(self) -> DatasetSpec:
        return DatasetSpec(
            "cxr_rait",
            self.root,
            ImageSpec(1, self.input_res, self.input_res),
            {"train": "train", "valid": "valid", "test": "test"},
            "data_demography.xlsx",
        )

    def load_split(self, split: str) -> CxrRaitDataset:
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unknown split {split!r}")
        return CxrRaitDataset(
            self.root,
            split=split,
            columns=list(self.schema.variable_names),
            norm=self.context_norm,
            concat_pa=False,
            input_res=self.input_res,
        )

    def make_batch(self, split: str, indices: Sequence[int], *, rng=None, training: bool = False) -> Batch:
        raw = self.load_split(split).make_batch(indices, rng=rng, shuffle=training)
        return Batch(
            np.asarray(raw["x"], dtype=np.float32),
            {name: np.asarray(raw[name], dtype=np.float32) for name in self.schema.variable_names},
        )

    def fingerprint(self) -> str:
        value = f"{self.root}|{self.input_res}|{self.context_norm}|{self.schema.version}"
        return hashlib.sha256(value.encode()).hexdigest()[:16]


def cxr_rait(settings) -> Dict[str, CxrRaitDataset]:
    """Build the three CXR-RAIT splits from settings."""
    if not settings.data_dir:
        raise ValueError("CXR-RAIT requires an explicit dataset.root")
    datasets = {}
    should_augment = getattr(settings, "augment", True)
    for split in ["train", "valid", "test"]:
        datasets[split] = CxrRaitDataset(
            root_dir=settings.data_dir,
            split=split,
            columns=settings.parents_x,
            norm=settings.context_norm,
            concat_pa=settings.concat_pa,
            input_res=settings.input_res,
            augment=(should_augment if split == "train" else False),
            torchxray_preprocessing=getattr(settings, "torchxray_preprocessing", False),
        )
    return datasets

