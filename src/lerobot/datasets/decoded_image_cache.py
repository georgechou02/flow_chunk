#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Lossless, memory-mapped cache for image-backed LeRobot observations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from .dataset_metadata import LeRobotDatasetMetadata


DECODED_IMAGE_CACHE_FORMAT_VERSION = 1
DECODED_IMAGE_CACHE_MANIFEST = "manifest.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_cache_identity(meta: LeRobotDatasetMetadata) -> dict[str, Any]:
    """Return the immutable source identity recorded in a decoded-image cache."""
    info_path = Path(meta.root) / "meta" / "info.json"
    return {
        "repo_id": meta.repo_id,
        "source_root": str(Path(meta.root).resolve()),
        "info_sha256": _sha256_file(info_path),
        "total_frames": int(meta.total_frames),
    }


def chw_shape_from_feature(feature: dict[str, Any]) -> tuple[int, int, int]:
    """Convert an RGB image feature shape to the CHW layout used by PyTorch."""
    shape = tuple(int(value) for value in feature["shape"])
    names = tuple(feature.get("names", ()))
    if len(shape) != 3:
        raise ValueError(f"Decoded image cache only supports rank-3 RGB images, got shape={shape}.")

    if names == ("height", "width", "channel") or (shape[-1] in (1, 3, 4) and shape[0] not in (1, 3, 4)):
        height, width, channels = shape
        return channels, height, width
    if names == ("channel", "height", "width") or shape[0] in (1, 3, 4):
        channels, height, width = shape
        return channels, height, width
    raise ValueError(f"Could not infer image layout from shape={shape}, names={names}.")


class DecodedImageCache:
    """Read losslessly decoded uint8 camera frames from sharded NPY memmaps.

    Cache slots are addressed by the dataset's absolute ``index`` column, so a
    single full-dataset cache can be shared by arbitrary episode subsets.
    Memmaps are opened lazily in each DataLoader worker and are never pickled.
    """

    def __init__(self, root: str | Path, meta: LeRobotDatasetMetadata):
        self.root = Path(root)
        manifest_path = self.root / DECODED_IMAGE_CACHE_MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Decoded image cache is incomplete or missing {manifest_path}. "
                "Build it before starting training."
            )

        with manifest_path.open() as file:
            self.manifest = json.load(file)

        version = self.manifest.get("format_version")
        if version != DECODED_IMAGE_CACHE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported decoded image cache format {version!r}; "
                f"expected {DECODED_IMAGE_CACHE_FORMAT_VERSION}."
            )

        expected_identity = dataset_cache_identity(meta)
        actual_identity = self.manifest.get("dataset")
        if actual_identity != expected_identity:
            raise ValueError(
                "Decoded image cache does not match the requested dataset. "
                f"expected={expected_identity}, cache={actual_identity}"
            )

        self.frames_per_shard = int(self.manifest["frames_per_shard"])
        if self.frames_per_shard <= 0:
            raise ValueError("Decoded image cache frames_per_shard must be positive.")

        image_keys = [key for key in meta.image_keys if key not in meta.depth_keys]
        camera_specs = self.manifest.get("cameras", {})
        missing = sorted(set(image_keys) - set(camera_specs))
        if missing:
            raise ValueError(f"Decoded image cache is missing image keys: {missing}")

        self.camera_specs: dict[str, dict[str, Any]] = {}
        for key in image_keys:
            spec = camera_specs[key]
            expected_shape = chw_shape_from_feature(meta.features[key])
            cached_shape = tuple(int(value) for value in spec["shape"])
            if cached_shape != expected_shape:
                raise ValueError(
                    f"Decoded image cache shape mismatch for {key}: "
                    f"expected {expected_shape}, got {cached_shape}."
                )
            if spec.get("dtype") != "uint8":
                raise ValueError(f"Decoded image cache for {key} must use uint8.")
            self.camera_specs[key] = spec

        self.total_frames = expected_identity["total_frames"]
        self._memmaps: dict[tuple[str, int], np.ndarray] = {}

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self.camera_specs)

    def __contains__(self, key: str) -> bool:
        return key in self.camera_specs

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_memmaps"] = {}
        return state

    def _open_shard(self, key: str, shard_index: int) -> np.ndarray:
        cache_key = (key, shard_index)
        array = self._memmaps.get(cache_key)
        if array is None:
            spec = self.camera_specs[key]
            shard_path = self.root / spec["directory"] / f"shard-{shard_index:05d}.npy"
            if not shard_path.is_file():
                raise FileNotFoundError(f"Decoded image cache shard is missing: {shard_path}")
            array = np.load(shard_path, mmap_mode="r", allow_pickle=False)
            expected_frames = min(
                self.frames_per_shard,
                self.total_frames - shard_index * self.frames_per_shard,
            )
            expected_shape = (expected_frames, *tuple(spec["shape"]))
            if array.dtype != np.uint8 or array.shape != expected_shape:
                raise ValueError(
                    f"Invalid decoded image cache shard {shard_path}: "
                    f"expected uint8 {expected_shape}, got {array.dtype} {array.shape}."
                )
            self._memmaps[cache_key] = array
        return array

    def get_uint8(self, key: str, absolute_indices: list[int]) -> torch.Tensor:
        """Read frames in CHW uint8 layout, preserving order and duplicates."""
        if key not in self.camera_specs:
            raise KeyError(key)
        indices = np.asarray(absolute_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("Decoded image cache indices must be one-dimensional.")
        if len(indices) == 0:
            shape = tuple(self.camera_specs[key]["shape"])
            return torch.empty((0, *shape), dtype=torch.uint8)
        if int(indices.min()) < 0 or int(indices.max()) >= self.total_frames:
            raise IndexError(
                f"Decoded image cache index range [{indices.min()}, {indices.max()}] "
                f"is outside [0, {self.total_frames})."
            )

        shape = tuple(self.camera_specs[key]["shape"])
        output = np.empty((len(indices), *shape), dtype=np.uint8)
        shard_indices = indices // self.frames_per_shard
        for shard_index in np.unique(shard_indices):
            positions = np.flatnonzero(shard_indices == shard_index)
            offsets = indices[positions] - int(shard_index) * self.frames_per_shard
            output[positions] = self._open_shard(key, int(shard_index))[offsets]
        return torch.from_numpy(output)

    def get_float32(self, key: str, absolute_indices: list[int]) -> torch.Tensor:
        """Match torchvision ``ToTensor`` output: CHW float32 in [0, 1]."""
        return self.get_uint8(key, absolute_indices).to(dtype=torch.float32).div_(255.0)
