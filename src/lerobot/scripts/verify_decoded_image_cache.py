#!/usr/bin/env python

"""Verify that a decoded-image cache is tensor-identical to the source dataset."""

from __future__ import annotations

import argparse
import random
import statistics
import time

import numpy as np
import torch

import lerobot.policies  # noqa: F401 - register policy config subclasses
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset


def _assert_items_equal(source: dict, cached: dict, index: int) -> None:
    if source.keys() != cached.keys():
        raise AssertionError(
            f"Item {index} keys differ: source-only={source.keys() - cached.keys()}, "
            f"cache-only={cached.keys() - source.keys()}"
        )
    for key, source_value in source.items():
        cached_value = cached[key]
        if isinstance(source_value, torch.Tensor):
            if not torch.equal(source_value, cached_value):
                max_abs_diff = (
                    (source_value - cached_value).abs().max().item()
                    if source_value.is_floating_point()
                    else None
                )
                raise AssertionError(f"Item {index} tensor {key!r} differs; max_abs_diff={max_abs_diff}.")
        elif source_value != cached_value:
            raise AssertionError(f"Item {index} value {key!r} differs: {source_value!r} != {cached_value!r}.")


def _seed_augmentation(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def verify(args: argparse.Namespace) -> None:
    raw_cfg = TrainPipelineConfig.from_pretrained(args.train_config, local_files_only=True)
    raw_cfg.dataset.image_transforms.enable = False
    raw_cfg.dataset.decoded_image_cache_root = None
    source_dataset = make_dataset(raw_cfg)

    cached_cfg = TrainPipelineConfig.from_pretrained(args.train_config, local_files_only=True)
    cached_cfg.dataset.image_transforms.enable = False
    cached_cfg.dataset.decoded_image_cache_root = args.cache_root
    cached_dataset = make_dataset(cached_cfg)

    rng = random.Random(args.seed)
    indices = [rng.randrange(len(source_dataset)) for _ in range(args.num_samples)]
    source_times = []
    cached_times = []
    for sample_number, index in enumerate(indices, start=1):
        start = time.perf_counter()
        source = source_dataset[index]
        source_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        cached = cached_dataset[index]
        cached_times.append(time.perf_counter() - start)
        _assert_items_equal(source, cached, index)
        if sample_number % 100 == 0:
            print(f"verified_without_augmentation={sample_number}/{len(indices)}", flush=True)

    raw_aug_cfg = TrainPipelineConfig.from_pretrained(args.train_config, local_files_only=True)
    raw_aug_cfg.dataset.decoded_image_cache_root = None
    source_augmented = make_dataset(raw_aug_cfg)
    cached_aug_cfg = TrainPipelineConfig.from_pretrained(args.train_config, local_files_only=True)
    cached_aug_cfg.dataset.decoded_image_cache_root = args.cache_root
    cached_augmented = make_dataset(cached_aug_cfg)
    for sample_number, index in enumerate(indices[: args.augmentation_samples], start=1):
        augmentation_seed = args.seed + sample_number
        _seed_augmentation(augmentation_seed)
        source = source_augmented[index]
        _seed_augmentation(augmentation_seed)
        cached = cached_augmented[index]
        _assert_items_equal(source, cached, index)

    source_mean = statistics.mean(source_times)
    cached_mean = statistics.mean(cached_times)
    print(f"torch_equal_without_augmentation={len(indices)}", flush=True)
    print(f"torch_equal_with_augmentation={min(len(indices), args.augmentation_samples)}", flush=True)
    print(f"source_mean_item_s={source_mean:.6f}", flush=True)
    print(f"cached_mean_item_s={cached_mean:.6f}", flush=True)
    print(f"item_speedup={source_mean / cached_mean:.2f}x", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--augmentation-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.augmentation_samples < 0:
        parser.error("--augmentation-samples must be non-negative")
    return args


if __name__ == "__main__":
    verify(parse_args())
