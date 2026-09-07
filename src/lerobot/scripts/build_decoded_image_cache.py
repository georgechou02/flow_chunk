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
"""Build a lossless sharded uint8 mmap cache for image-backed datasets.

Example:

    python -m lerobot.scripts.build_decoded_image_cache \
        --repo-id HuggingFaceVLA/libero \
        --root /path/to/libero/snapshot \
        --output-dir /home/user/.cache/lerobot_decoded/libero_commit \
        --num-workers 24

The builder is resumable at shard boundaries. It never rewrites the source
dataset and writes ``manifest.json`` only after every frame is complete.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from lerobot.datasets.decoded_image_cache import (
    DECODED_IMAGE_CACHE_FORMAT_VERSION,
    DECODED_IMAGE_CACHE_MANIFEST,
    DecodedImageCache,
    chw_shape_from_feature,
    dataset_cache_identity,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset

BUILD_STATE_NAME = "build_state.json"


def _image_rows_to_uint8(items: dict[str, list[Any]]) -> dict[str, list[torch.Tensor]]:
    """HF transform that avoids float32 image IPC while building the cache."""
    transformed: dict[str, list[torch.Tensor]] = {}
    for key, values in items.items():
        first = values[0]
        if isinstance(first, PILImage.Image):
            tensors = []
            for image in values:
                array = np.array(image, dtype=np.uint8, copy=True)
                if array.ndim == 2:
                    array = array[..., None]
                tensors.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
            transformed[key] = tensors
        else:
            transformed[key] = [torch.as_tensor(value) for value in values]
    return transformed


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary_path.replace(path)


class ShardedCacheWriter:
    def __init__(
        self,
        output_dir: Path,
        camera_specs: dict[str, dict[str, Any]],
        total_frames: int,
        frames_per_shard: int,
        state_path: Path,
        state: dict[str, Any],
    ):
        self.output_dir = output_dir
        self.camera_specs = camera_specs
        self.total_frames = total_frames
        self.frames_per_shard = frames_per_shard
        self.state_path = state_path
        self.state = state
        self.current_shard: int | None = None
        self.arrays: dict[str, np.memmap] = {}
        self.next_expected_index = int(state["completed_shards"]) * frames_per_shard

    def _open_shard(self, shard_index: int) -> None:
        shard_start = shard_index * self.frames_per_shard
        shard_frames = min(self.frames_per_shard, self.total_frames - shard_start)
        self.current_shard = shard_index
        self.arrays = {}
        for key, spec in self.camera_specs.items():
            directory = self.output_dir / spec["directory"]
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"shard-{shard_index:05d}.npy"
            self.arrays[key] = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=np.uint8,
                shape=(shard_frames, *tuple(spec["shape"])),
            )

    def _finish_current_shard(self) -> None:
        if self.current_shard is None:
            return
        for array in self.arrays.values():
            array.flush()
        self.arrays.clear()
        completed_shards = self.current_shard + 1
        self.state["completed_shards"] = completed_shards
        _write_json_atomic(self.state_path, self.state)
        self.current_shard = None

    def write_batch(self, indices: torch.Tensor, batch: dict[str, torch.Tensor]) -> None:
        absolute_indices = indices.to(dtype=torch.int64).cpu().numpy()
        if len(absolute_indices) == 0:
            return
        expected = np.arange(
            self.next_expected_index,
            self.next_expected_index + len(absolute_indices),
            dtype=np.int64,
        )
        if not np.array_equal(absolute_indices, expected):
            raise ValueError(
                "Dataset index column must be contiguous and ordered to build the decoded cache: "
                f"expected {expected[[0, -1]].tolist()}, got {absolute_indices[[0, -1]].tolist()}."
            )

        shard_indices = absolute_indices // self.frames_per_shard
        for shard_index in np.unique(shard_indices):
            shard_index = int(shard_index)
            if self.current_shard != shard_index:
                self._finish_current_shard()
                self._open_shard(shard_index)
            positions = np.flatnonzero(shard_indices == shard_index)
            offsets = absolute_indices[positions] - shard_index * self.frames_per_shard
            for key, array in self.arrays.items():
                values = batch[key][torch.from_numpy(positions)].cpu().numpy()
                if values.dtype != np.uint8:
                    raise TypeError(f"Builder expected uint8 for {key}, got {values.dtype}.")
                array[offsets] = values
        self.next_expected_index += len(absolute_indices)

    def finish(self) -> None:
        self._finish_current_shard()
        if self.next_expected_index != self.total_frames:
            raise RuntimeError(
                f"Decoded image cache ended at frame {self.next_expected_index}, "
                f"expected {self.total_frames}."
            )


def build_cache(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / DECODED_IMAGE_CACHE_MANIFEST
    dataset = LeRobotDataset(args.repo_id, root=args.root, revision=args.revision)
    if manifest_path.is_file():
        DecodedImageCache(output_dir, dataset.meta)
        print(f"Decoded image cache is already complete and valid: {manifest_path}")
        return
    image_keys = [key for key in dataset.meta.image_keys if key not in dataset.meta.depth_keys]
    if not image_keys:
        raise ValueError("Dataset has no image-backed RGB observations to cache.")

    identity = dataset_cache_identity(dataset.meta)
    camera_specs = {
        key: {
            "directory": f"camera-{camera_index:03d}",
            "dtype": "uint8",
            "shape": list(chw_shape_from_feature(dataset.meta.features[key])),
        }
        for camera_index, key in enumerate(image_keys)
    }
    state_template = {
        "format_version": DECODED_IMAGE_CACHE_FORMAT_VERSION,
        "dataset": identity,
        "frames_per_shard": args.frames_per_shard,
        "cameras": camera_specs,
        "completed_shards": 0,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / BUILD_STATE_NAME
    if state_path.is_file():
        with state_path.open() as file:
            state = json.load(file)
        comparable_state = dict(state)
        comparable_state["completed_shards"] = 0
        if comparable_state != state_template:
            raise ValueError(
                f"Existing cache build state does not match this request: {state_path}. "
                "Use a different output directory."
            )
    else:
        unexpected = [path for path in output_dir.iterdir() if path.name != BUILD_STATE_NAME]
        if unexpected:
            raise FileExistsError(
                f"Output directory is non-empty but has no resumable state: {output_dir}. "
                "Use a new output directory."
            )
        state = state_template
        _write_json_atomic(state_path, state)

    total_frames = int(dataset.meta.total_frames)
    start_index = int(state["completed_shards"]) * args.frames_per_shard
    bytes_per_frame = sum(int(np.prod(spec["shape"])) for spec in camera_specs.values())
    required_bytes = total_frames * bytes_per_frame
    remaining_bytes = max(0, total_frames - start_index) * bytes_per_frame
    free_bytes = shutil.disk_usage(output_dir).free
    print(
        f"Caching {total_frames:,} frames and {len(image_keys)} cameras as lossless uint8 "
        f"({required_bytes / 1e9:.1f} GB); filesystem free={free_bytes / 1e9:.1f} GB."
    )
    if free_bytes < remaining_bytes + args.min_free_gb * 1_000_000_000:
        raise OSError(
            f"Insufficient free space in {output_dir}: need cache plus {args.min_free_gb} GB headroom."
        )

    columns = ["index", *image_keys]
    image_dataset = dataset.hf_dataset.select_columns(columns)
    image_dataset.set_transform(_image_rows_to_uint8)

    if start_index >= total_frames:
        print("All shards are present; finalizing manifest.")
    else:
        selected = Subset(image_dataset, range(start_index, total_frames))
        loader = DataLoader(
            selected,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0,
        )
        writer = ShardedCacheWriter(
            output_dir,
            camera_specs,
            total_frames,
            args.frames_per_shard,
            state_path,
            state,
        )
        start_time = time.perf_counter()
        with tqdm(total=total_frames - start_index, unit="frame", desc="Decoding images") as progress:
            for batch in loader:
                writer.write_batch(batch["index"], batch)
                progress.update(len(batch["index"]))
        writer.finish()
        elapsed = time.perf_counter() - start_time
        print(f"Decoded {total_frames - start_index:,} frames in {elapsed / 60:.1f} minutes.")

    manifest = {
        "format_version": DECODED_IMAGE_CACHE_FORMAT_VERSION,
        "dataset": identity,
        "frames_per_shard": args.frames_per_shard,
        "cameras": camera_specs,
    }
    _write_json_atomic(manifest_path, manifest)
    state_path.unlink(missing_ok=True)
    print(f"Decoded image cache is complete: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--frames-per-shard", type=int, default=8192)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    args = parser.parse_args()
    for name in ("num_workers", "batch_size", "prefetch_factor", "frames_per_shard"):
        if getattr(args, name) <= 0 and name != "num_workers":
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.min_free_gb < 0:
        parser.error("--min-free-gb must be non-negative")
    return args


if __name__ == "__main__":
    build_cache(parse_args())
