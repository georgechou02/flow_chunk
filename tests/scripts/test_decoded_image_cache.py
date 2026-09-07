from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="Cache builder requires lerobot[dataset]")

from lerobot.datasets.decoded_image_cache import DecodedImageCache, dataset_cache_identity  # noqa: E402
from lerobot.scripts.build_decoded_image_cache import ShardedCacheWriter  # noqa: E402


def _make_meta(source_root: Path, total_frames: int = 5):
    (source_root / "meta").mkdir(parents=True)
    (source_root / "meta" / "info.json").write_text('{"test": true}\n')
    return SimpleNamespace(
        repo_id="test/cache",
        root=source_root,
        total_frames=total_frames,
        image_keys=["observation.images.one"],
        depth_keys=[],
        features={
            "observation.images.one": {
                "dtype": "image",
                "shape": [2, 3, 3],
                "names": ["height", "width", "channel"],
            }
        },
    )


def _make_cache(cache_root: Path, meta, values: np.ndarray, frames_per_shard: int = 3):
    camera_dir = cache_root / "camera-000"
    camera_dir.mkdir(parents=True)
    for shard_index, start in enumerate(range(0, len(values), frames_per_shard)):
        np.save(camera_dir / f"shard-{shard_index:05d}.npy", values[start : start + frames_per_shard])
    manifest = {
        "format_version": 1,
        "dataset": dataset_cache_identity(meta),
        "frames_per_shard": frames_per_shard,
        "cameras": {
            "observation.images.one": {
                "directory": "camera-000",
                "dtype": "uint8",
                "shape": [3, 2, 3],
            }
        },
    }
    (cache_root / "manifest.json").write_text(json.dumps(manifest))


def test_decoded_image_cache_preserves_order_duplicates_and_torch_conversion(tmp_path):
    meta = _make_meta(tmp_path / "source")
    values = np.arange(5 * 3 * 2 * 3, dtype=np.uint8).reshape(5, 3, 2, 3)
    cache_root = tmp_path / "cache"
    _make_cache(cache_root, meta, values)

    cache = DecodedImageCache(cache_root, meta)
    indices = [4, 0, 3, 3, 1]
    actual_uint8 = cache.get_uint8("observation.images.one", indices)
    expected_uint8 = torch.from_numpy(values[indices])
    assert torch.equal(actual_uint8, expected_uint8)
    assert torch.equal(cache.get_float32("observation.images.one", indices), expected_uint8.float() / 255)

    # DataLoader workers pickle datasets under spawn; mmap handles must reopen lazily.
    restored = pickle.loads(pickle.dumps(cache))
    assert torch.equal(restored.get_uint8("observation.images.one", indices), expected_uint8)


def test_decoded_image_cache_rejects_different_dataset(tmp_path):
    meta = _make_meta(tmp_path / "source")
    values = np.zeros((5, 3, 2, 3), dtype=np.uint8)
    cache_root = tmp_path / "cache"
    _make_cache(cache_root, meta, values)

    (meta.root / "meta" / "info.json").write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match="does not match"):
        DecodedImageCache(cache_root, meta)


def test_sharded_cache_writer_handles_batch_across_boundary(tmp_path):
    output_dir = tmp_path / "cache"
    output_dir.mkdir()
    state_path = output_dir / "build_state.json"
    state = {"completed_shards": 0}
    specs = {
        "camera": {
            "directory": "camera-000",
            "dtype": "uint8",
            "shape": [1, 1, 2],
        }
    }
    writer = ShardedCacheWriter(output_dir, specs, 5, 3, state_path, state)
    values = torch.arange(10, dtype=torch.uint8).reshape(5, 1, 1, 2)
    writer.write_batch(torch.tensor([0, 1, 2, 3]), {"camera": values[:4]})
    writer.write_batch(torch.tensor([4]), {"camera": values[4:]})
    writer.finish()

    shards = [
        np.load(output_dir / "camera-000" / "shard-00000.npy"),
        np.load(output_dir / "camera-000" / "shard-00001.npy"),
    ]
    assert np.array_equal(np.concatenate(shards), values.numpy())
    assert json.loads(state_path.read_text())["completed_shards"] == 2
