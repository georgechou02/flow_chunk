#!/usr/bin/env python3

"""Measure how well low-frequency DCT coefficients reconstruct LIBERO action chunks.

The analysis applies an orthonormal DCT-II along the temporal axis of fixed-size
action chunks. By default, only the first six continuous robot action dimensions
are transformed; the discrete gripper dimension is retained unchanged.

Example:
    python scripts/dct_action_expressivity.py \
        --suite libero_10 \
        --env-task-id 2 \
        --horizon 32 \
        --stride 1 \
        --output-dir outputs/dct_analysis/libero_10_task2_h32
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lerobot-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as pds
import pyarrow.parquet as pq
from libero.libero import benchmark
from scipy.fft import dct, idct

DEFAULT_SNAPSHOT_DIR = (
    Path.home() / ".cache/huggingface/lerobot/hub/datasets--HuggingFaceVLA--libero/snapshots"
)
# Match FAFM's convention: M is the highest retained frequency index, so M+1
# coefficients (modes 0 through M, inclusive) are kept.
DEFAULT_M_VALUES = "1,2,4,8,12,16,24,31"
ARM_DIM_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="LeRobot dataset snapshot containing meta/ and data/. Auto-detected by default.",
    )
    parser.add_argument("--suite", default="libero_10", help="LIBERO benchmark suite name.")
    parser.add_argument(
        "--env-task-id",
        type=int,
        default=2,
        help="Zero-based task id in the LIBERO environment suite.",
    )
    parser.add_argument("--horizon", type=int, default=32, help="Action chunk length.")
    parser.add_argument("--stride", type=int, default=1, help="Sliding-window stride within each episode.")
    parser.add_argument(
        "--arm-dims",
        type=int,
        default=6,
        help="Number of leading continuous action dimensions transformed by DCT.",
    )
    parser.add_argument(
        "--m-values",
        default=DEFAULT_M_VALUES,
        help=(
            "Comma-separated highest retained DCT mode indices highlighted in the report. "
            "Matches FAFM: M keeps modes 0 through M, i.e. M+1 coefficients."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/dct_analysis/<suite>_task<ID>_h<H>.",
    )
    return parser.parse_args()


def resolve_dataset_root(dataset_root: Path | None) -> Path:
    if dataset_root is not None:
        root = dataset_root.expanduser().resolve()
        if not (root / "meta/info.json").exists():
            raise FileNotFoundError(f"Not a LeRobot dataset snapshot: {root}")
        return root

    if not DEFAULT_SNAPSHOT_DIR.exists():
        raise FileNotFoundError(
            "Could not auto-detect HuggingFaceVLA/libero. Pass --dataset-root explicitly."
        )
    snapshots = sorted(
        (path for path in DEFAULT_SNAPSHOT_DIR.iterdir() if (path / "meta/info.json").exists()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not snapshots:
        raise FileNotFoundError(f"No usable dataset snapshots found under {DEFAULT_SNAPSHOT_DIR}.")
    return snapshots[0].resolve()


def normalize_task_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("_", " ").strip().lower())


def resolve_task(dataset_root: Path, suite_name: str, env_task_id: int) -> tuple[str, int]:
    suites = benchmark.get_benchmark_dict()
    if suite_name not in suites:
        raise ValueError(f"Unknown suite {suite_name!r}; available={sorted(suites)}")
    suite = suites[suite_name]()
    if env_task_id < 0 or env_task_id >= suite.n_tasks:
        raise ValueError(f"env task id {env_task_id} is outside [0, {suite.n_tasks - 1}]")

    language = str(suite.get_task(env_task_id).language)
    tasks = pq.read_table(dataset_root / "meta/tasks.parquet").to_pandas()
    target = normalize_task_text(language)
    matches = [index for index in tasks.index if normalize_task_text(str(index)) == target]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one dataset task matching {language!r}, found {len(matches)}: {matches}"
        )
    dataset_task_index = int(tasks.loc[matches[0], "task_index"])
    return language, dataset_task_index


def parse_m_values(raw: str, horizon: int) -> list[int]:
    values = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    invalid = [value for value in values if value < 0 or value >= horizon]
    if invalid:
        raise ValueError(f"DCT mode indices M must be within [0, {horizon - 1}]: {invalid}")
    if horizon - 1 not in values:
        values.append(horizon - 1)
    return sorted(values)


def load_task_actions(dataset_root: Path, dataset_task_index: int) -> pd.DataFrame:
    dataset = pds.dataset(dataset_root / "data", format="parquet", exclude_invalid_files=True)
    table = dataset.to_table(
        columns=["episode_index", "frame_index", "task_index", "action"],
        filter=pds.field("task_index") == dataset_task_index,
    )
    frame = table.to_pandas()
    if frame.empty:
        raise RuntimeError(f"No frames found for dataset task_index={dataset_task_index}")
    frame = frame.sort_values(["episode_index", "frame_index"], kind="stable").reset_index(drop=True)
    return frame


def build_action_chunks(
    frame: pd.DataFrame, horizon: int, stride: int
) -> tuple[np.ndarray, pd.DataFrame, list[int]]:
    if horizon < 2:
        raise ValueError("horizon must be at least 2")
    if stride < 1:
        raise ValueError("stride must be positive")

    chunks: list[np.ndarray] = []
    chunk_records: list[dict[str, int]] = []
    episode_lengths: list[int] = []
    for episode_index, episode in frame.groupby("episode_index", sort=True):
        episode = episode.sort_values("frame_index", kind="stable")
        frame_indices = episode["frame_index"].to_numpy(dtype=np.int64)
        if len(frame_indices) > 1 and not np.all(np.diff(frame_indices) == 1):
            raise RuntimeError(f"episode {episode_index} has non-contiguous frame indices")
        actions = np.stack(episode["action"].to_numpy()).astype(np.float64, copy=False)
        episode_lengths.append(len(actions))
        for start in range(0, len(actions) - horizon + 1, stride):
            chunks.append(actions[start : start + horizon])
            chunk_records.append(
                {
                    "episode_index": int(episode_index),
                    "start_frame": int(frame_indices[start]),
                }
            )

    if not chunks:
        raise RuntimeError(
            f"No complete {horizon}-frame chunks could be built from {len(episode_lengths)} episodes"
        )
    return np.stack(chunks), pd.DataFrame(chunk_records), episode_lengths


def compute_metrics(
    arm_chunks: np.ndarray,
    coefficients: np.ndarray,
    action_std: np.ndarray,
    selected_m: set[int],
    non_dct_dims: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, horizon, arm_dims = arm_chunks.shape
    rows: list[dict[str, float | int | bool]] = []
    per_dim_rows: list[dict[str, float | int | str]] = []
    total_energy = float(np.sum(coefficients**2))
    chunk_total_energy = np.sum(coefficients**2, axis=(1, 2))
    original_representation_values = horizon * (arm_dims + non_dct_dims)
    eps = np.finfo(np.float64).eps

    for m in range(horizon):
        coefficient_count = m + 1
        truncated = np.zeros_like(coefficients)
        truncated[:, :coefficient_count, :] = coefficients[:, :coefficient_count, :]
        reconstructed = idct(truncated, type=2, axis=1, norm="ortho")
        error = reconstructed - arm_chunks
        abs_error = np.abs(error)
        retained_energy = float(np.sum(coefficients[:, :coefficient_count, :] ** 2) / total_energy)
        per_chunk_energy_retained = np.sum(
            coefficients[:, :coefficient_count, :] ** 2, axis=(1, 2)
        ) / np.maximum(chunk_total_energy, eps)
        dim_balanced_nrmse = float(np.sqrt(np.mean((error / action_std[None, None, :]) ** 2)))
        relative_l2 = float(np.sqrt(np.sum(error**2) / max(float(np.sum(arm_chunks**2)), eps)))
        representation_values = coefficient_count * arm_dims + horizon * non_dct_dims
        rows.append(
            {
                "m": m,
                "coefficient_count": coefficient_count,
                "selected": m in selected_m,
                "arm_coefficient_fraction": coefficient_count / horizon,
                "total_representation_values": representation_values,
                "total_representation_fraction": representation_values / original_representation_values,
                "compression_factor": original_representation_values / representation_values,
                "energy_retained": retained_energy,
                "per_chunk_energy_retained_p10": float(np.quantile(per_chunk_energy_retained, 0.10)),
                "per_chunk_energy_retained_median": float(np.quantile(per_chunk_energy_retained, 0.50)),
                "per_chunk_energy_retained_p90": float(np.quantile(per_chunk_energy_retained, 0.90)),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(abs_error)),
                "p95_absolute_error": float(np.quantile(abs_error, 0.95)),
                "max_absolute_error": float(np.max(abs_error)),
                "dimension_balanced_nrmse": dim_balanced_nrmse,
                "relative_l2_error": relative_l2,
            }
        )

        rmse_per_dim = np.sqrt(np.mean(error**2, axis=(0, 1)))
        mae_per_dim = np.mean(abs_error, axis=(0, 1))
        for dim in range(arm_dims):
            name = ARM_DIM_NAMES[dim] if dim < len(ARM_DIM_NAMES) else f"action_{dim}"
            per_dim_rows.append(
                {
                    "m": m,
                    "coefficient_count": coefficient_count,
                    "dimension": dim,
                    "name": name,
                    "rmse": float(rmse_per_dim[dim]),
                    "normalized_rmse": float(rmse_per_dim[dim] / action_std[dim]),
                    "mae": float(mae_per_dim[dim]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(per_dim_rows)


def compute_energy_table(coefficients: np.ndarray) -> pd.DataFrame:
    _, horizon, arm_dims = coefficients.shape
    energy = np.sum(coefficients**2, axis=0)
    total_by_dim = np.maximum(np.sum(energy, axis=0), np.finfo(np.float64).eps)
    global_total = float(np.sum(energy))
    rows: list[dict[str, float | int | str]] = []
    for frequency in range(horizon):
        for dim in range(arm_dims):
            name = ARM_DIM_NAMES[dim] if dim < len(ARM_DIM_NAMES) else f"action_{dim}"
            rows.append(
                {
                    "frequency": frequency,
                    "dimension": dim,
                    "name": name,
                    "energy": float(energy[frequency, dim]),
                    "dimension_energy_fraction": float(energy[frequency, dim] / total_by_dim[dim]),
                    "global_energy_fraction": float(energy[frequency, dim] / global_total),
                }
            )
    return pd.DataFrame(rows)


def dct_analytic_derivative_basis(horizon: int, fps: float) -> np.ndarray:
    """Return DCT-II basis derivatives at the current action sample times.

    DCT-II represents sample n with cos(pi * k * (n + 1/2) / N). A forward
    difference from samples n to n+1 is assigned to the current sample n. The
    analytic derivative is therefore evaluated at DCT coordinate n+1/2.
    Multiplication by fps converts from action-units/sample to action-units/second.
    """

    frequency = np.arange(horizon, dtype=np.float64)
    current_sample_coordinate = np.arange(horizon - 1, dtype=np.float64) + 0.5
    scale = np.full(horizon, np.sqrt(2.0 / horizon), dtype=np.float64)
    scale[0] = np.sqrt(1.0 / horizon)
    angular_frequency = np.pi * frequency / horizon
    return (
        -scale[:, None]
        * angular_frequency[:, None]
        * fps
        * np.sin(angular_frequency[:, None] * current_sample_coordinate[None, :])
    )


def safe_correlation(target: np.ndarray, prediction: np.ndarray) -> float:
    target_centered = target - np.mean(target)
    prediction_centered = prediction - np.mean(prediction)
    denominator = float(np.sqrt(np.sum(target_centered**2) * np.sum(prediction_centered**2)))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.sum(target_centered * prediction_centered) / denominator)


def reconstruct_analytic_derivative(
    coefficients: np.ndarray, derivative_basis: np.ndarray, m: int
) -> np.ndarray:
    coefficient_count = m + 1
    return np.einsum(
        "bkd,kt->btd",
        coefficients[:, :coefficient_count, :],
        derivative_basis[:coefficient_count, :],
        optimize=True,
    )


def compute_derivative_metrics(
    arm_chunks: np.ndarray,
    coefficients: np.ndarray,
    derivative_basis: np.ndarray,
    fps: float,
    selected_m: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    dataset_derivative = np.diff(arm_chunks, axis=1) * fps
    derivative_std = np.std(dataset_derivative, axis=(0, 1))
    if np.any(derivative_std <= np.finfo(np.float64).eps):
        raise RuntimeError(
            f"At least one dataset action-derivative dimension has zero variance: {derivative_std}"
        )

    rows: list[dict[str, float | int | bool]] = []
    per_dim_rows: list[dict[str, float | int | str]] = []
    horizon = arm_chunks.shape[1]
    target_rms = float(np.sqrt(np.mean(dataset_derivative**2)))
    for m in range(horizon):
        coefficient_count = m + 1
        analytic_derivative = reconstruct_analytic_derivative(coefficients, derivative_basis, m)
        error = analytic_derivative - dataset_derivative
        abs_error = np.abs(error)
        correlations = [
            safe_correlation(
                dataset_derivative[:, :, dim].ravel(),
                analytic_derivative[:, :, dim].ravel(),
            )
            for dim in range(arm_chunks.shape[2])
        ]
        rows.append(
            {
                "m": m,
                "coefficient_count": coefficient_count,
                "selected": m in selected_m,
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(abs_error)),
                "p95_absolute_error": float(np.quantile(abs_error, 0.95)),
                "max_absolute_error": float(np.max(abs_error)),
                "dimension_balanced_nrmse": float(
                    np.sqrt(np.mean((error / derivative_std[None, None, :]) ** 2))
                ),
                "mean_per_dimension_correlation": float(np.mean(correlations)),
                "minimum_per_dimension_correlation": float(np.min(correlations)),
                "analytic_to_dataset_rms_ratio": float(np.sqrt(np.mean(analytic_derivative**2)) / target_rms),
            }
        )

        rmse_per_dim = np.sqrt(np.mean(error**2, axis=(0, 1)))
        mae_per_dim = np.mean(abs_error, axis=(0, 1))
        prediction_std = np.std(analytic_derivative, axis=(0, 1))
        for dim in range(arm_chunks.shape[2]):
            name = ARM_DIM_NAMES[dim] if dim < len(ARM_DIM_NAMES) else f"action_{dim}"
            per_dim_rows.append(
                {
                    "m": m,
                    "coefficient_count": coefficient_count,
                    "dimension": dim,
                    "name": name,
                    "rmse": float(rmse_per_dim[dim]),
                    "normalized_rmse": float(rmse_per_dim[dim] / derivative_std[dim]),
                    "mae": float(mae_per_dim[dim]),
                    "correlation": correlations[dim],
                    "dataset_derivative_std": float(derivative_std[dim]),
                    "analytic_derivative_std": float(prediction_std[dim]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(per_dim_rows), derivative_std


def first_m_at_threshold(metrics: pd.DataFrame, column: str, threshold: float) -> int:
    matches = metrics.loc[metrics[column] >= threshold, "m"]
    return int(matches.iloc[0]) if not matches.empty else int(metrics["m"].max())


def plot_overview(
    metrics: pd.DataFrame,
    per_dim_metrics: pd.DataFrame,
    energy_table: pd.DataFrame,
    selected_m: list[int],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    axes[0, 0].fill_between(
        metrics["m"],
        metrics["per_chunk_energy_retained_p10"] * 100,
        metrics["per_chunk_energy_retained_p90"] * 100,
        color="tab:blue",
        alpha=0.15,
        label="chunk p10-p90",
    )
    axes[0, 0].plot(metrics["m"], metrics["energy_retained"] * 100, marker="o", markersize=3, label="global")
    axes[0, 0].plot(
        metrics["m"],
        metrics["per_chunk_energy_retained_median"] * 100,
        linestyle="--",
        linewidth=1.2,
        label="chunk median",
    )
    for threshold in (90, 95, 99):
        axes[0, 0].axhline(threshold, color="gray", linestyle="--", linewidth=0.8)
    axes[0, 0].set(
        title="Cumulative arm-action DCT energy",
        xlabel="Highest retained mode M (keeps M+1 coefficients)",
        ylabel="Energy retained (%)",
    )
    axes[0, 0].set_xticks(selected_m)
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(metrics["m"], metrics["dimension_balanced_nrmse"], marker="o", markersize=3)
    axes[0, 1].set(
        title="Dimension-balanced reconstruction error",
        xlabel="Highest retained mode M (keeps M+1 coefficients)",
        ylabel="NRMSE (relative to per-dim std)",
    )
    axes[0, 1].set_xticks(selected_m)
    axes[0, 1].grid(alpha=0.25)

    selected = per_dim_metrics[per_dim_metrics["m"].isin(selected_m)]
    for name, group in selected.groupby("name", sort=False):
        axes[1, 0].plot(group["m"], group["normalized_rmse"], marker="o", label=name)
    axes[1, 0].set(
        title="Per-dimension normalized RMSE",
        xlabel="Highest retained mode M (keeps M+1 coefficients)",
        ylabel="NRMSE",
    )
    axes[1, 0].set_xticks(selected_m)
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(ncols=2, fontsize=8)

    pivot = energy_table.pivot(index="dimension", columns="frequency", values="dimension_energy_fraction")
    image = axes[1, 1].imshow(pivot.to_numpy(), aspect="auto", cmap="magma")
    axes[1, 1].set(
        title="DCT energy distribution per action dimension",
        xlabel="Frequency index",
        ylabel="Action dimension",
        yticks=np.arange(len(pivot.index)),
        yticklabels=[ARM_DIM_NAMES[i] if i < len(ARM_DIM_NAMES) else str(i) for i in pivot.index],
    )
    fig.colorbar(image, ax=axes[1, 1], label="Within-dimension energy fraction")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_derivative_overview(
    derivative_metrics: pd.DataFrame,
    per_dim_derivative_metrics: pd.DataFrame,
    selected_m: list[int],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    axes[0, 0].plot(
        derivative_metrics["m"],
        derivative_metrics["dimension_balanced_nrmse"],
        marker="o",
        markersize=3,
    )
    axes[0, 0].set(
        title="Analytic DCT derivative vs. dataset finite difference",
        xlabel="Highest retained mode M (keeps M+1 coefficients)",
        ylabel="Dimension-balanced derivative NRMSE",
    )
    axes[0, 0].set_xticks(selected_m)
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(
        derivative_metrics["m"],
        derivative_metrics["mean_per_dimension_correlation"],
        marker="o",
        markersize=3,
        label="mean",
    )
    axes[0, 1].plot(
        derivative_metrics["m"],
        derivative_metrics["minimum_per_dimension_correlation"],
        marker="o",
        markersize=3,
        label="minimum",
    )
    axes[0, 1].set(
        title="Derivative correlation",
        xlabel="Highest retained mode M (keeps M+1 coefficients)",
        ylabel="Pearson correlation",
        ylim=(-0.05, 1.05),
    )
    axes[0, 1].set_xticks(selected_m)
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()

    axes[1, 0].plot(
        derivative_metrics["m"],
        derivative_metrics["analytic_to_dataset_rms_ratio"],
        marker="o",
        markersize=3,
    )
    axes[1, 0].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[1, 0].set(
        title="Derivative RMS magnitude ratio",
        xlabel="Highest retained mode M (keeps M+1 coefficients)",
        ylabel="Analytic DCT / dataset finite difference",
    )
    axes[1, 0].set_xticks(selected_m)
    axes[1, 0].grid(alpha=0.25)

    selected = per_dim_derivative_metrics[per_dim_derivative_metrics["m"].isin(selected_m)]
    for name, group in selected.groupby("name", sort=False):
        axes[1, 1].plot(group["m"], group["normalized_rmse"], marker="o", label=name)
    axes[1, 1].set(
        title="Per-dimension derivative NRMSE",
        xlabel="Highest retained mode M (keeps M+1 coefficients)",
        ylabel="NRMSE",
    )
    axes[1, 1].set_xticks(selected_m)
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(ncols=2, fontsize=8)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_sample_reconstruction(
    arm_chunks: np.ndarray,
    coefficients: np.ndarray,
    selected_m: list[int],
    chunk_metadata: pd.DataFrame,
    output_path: Path,
) -> dict[str, int]:
    reference_m = min(selected_m, key=lambda value: abs(value - 8))
    reference_coefficient_count = reference_m + 1
    chunk_energy = np.sum(coefficients**2, axis=(1, 2))
    retained = np.sum(coefficients[:, :reference_coefficient_count, :] ** 2, axis=(1, 2)) / np.maximum(
        chunk_energy, np.finfo(np.float64).eps
    )
    sample_index = int(np.argmin(np.abs(retained - np.median(retained))))
    plot_m = sorted({value for value in selected_m if value in (4, 8, 16, 24, arm_chunks.shape[1] - 1)})
    if arm_chunks.shape[1] - 1 not in plot_m:
        plot_m.append(arm_chunks.shape[1] - 1)

    arm_dims = arm_chunks.shape[2]
    fig, axes = plt.subplots(arm_dims, 1, figsize=(12, 2.1 * arm_dims), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    time = np.arange(arm_chunks.shape[1])
    for dim, axis in enumerate(axes):
        name = ARM_DIM_NAMES[dim] if dim < len(ARM_DIM_NAMES) else f"action_{dim}"
        axis.plot(time, arm_chunks[sample_index, :, dim], color="black", linewidth=2, label="original")
        for m in plot_m:
            coefficient_count = m + 1
            truncated = np.zeros_like(coefficients[sample_index : sample_index + 1])
            truncated[:, :coefficient_count, :] = coefficients[
                sample_index : sample_index + 1, :coefficient_count, :
            ]
            reconstructed = idct(truncated, type=2, axis=1, norm="ortho")[0, :, dim]
            axis.plot(time, reconstructed, linewidth=1.2, label=f"M={m}")
        axis.set_ylabel(name)
        axis.grid(alpha=0.2)
    axes[0].legend(ncols=min(5, len(plot_m) + 1), fontsize=8)
    axes[-1].set_xlabel("Chunk timestep")
    metadata = chunk_metadata.iloc[sample_index]
    fig.suptitle(
        f"Representative chunk: episode {int(metadata.episode_index)}, start frame {int(metadata.start_frame)}"
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {
        "chunk_index": sample_index,
        "episode_index": int(metadata.episode_index),
        "start_frame": int(metadata.start_frame),
        "reference_m": reference_m,
    }


def plot_sample_derivative_comparison(
    arm_chunks: np.ndarray,
    coefficients: np.ndarray,
    derivative_basis: np.ndarray,
    fps: float,
    selected_m: list[int],
    sample: dict[str, int],
    output_path: Path,
) -> None:
    sample_index = sample["chunk_index"]
    dataset_derivative = np.diff(arm_chunks[sample_index], axis=0) * fps
    plot_m = sorted({value for value in selected_m if value in (8, 16, 24, arm_chunks.shape[1] - 1)})
    if arm_chunks.shape[1] - 1 not in plot_m:
        plot_m.append(arm_chunks.shape[1] - 1)
    analytic_derivatives = {
        m: reconstruct_analytic_derivative(
            coefficients[sample_index : sample_index + 1], derivative_basis, m
        )[0]
        for m in plot_m
    }

    current_sample_time = np.arange(arm_chunks.shape[1] - 1) / fps
    fig, axes = plt.subplots(
        arm_chunks.shape[2],
        1,
        figsize=(12, 2.1 * arm_chunks.shape[2]),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for dim, axis in enumerate(axes):
        name = ARM_DIM_NAMES[dim] if dim < len(ARM_DIM_NAMES) else f"action_{dim}"
        axis.plot(
            current_sample_time,
            dataset_derivative[:, dim],
            color="black",
            linewidth=2,
            label="dataset finite difference",
        )
        for m, analytic_derivative in analytic_derivatives.items():
            axis.plot(
                current_sample_time,
                analytic_derivative[:, dim],
                linewidth=1.2,
                label=f"analytic M={m}",
            )
        axis.set_ylabel(f"d{name}/dt")
        axis.grid(alpha=0.2)
    axes[0].legend(ncols=min(5, len(plot_m) + 1), fontsize=8)
    axes[-1].set_xlabel("Time within action chunk (seconds)")
    fig.suptitle(
        "Dataset finite-difference derivative vs. analytic DCT derivative: "
        f"episode {sample['episode_index']}, start frame {sample['start_frame']}"
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    output_path: Path,
    summary: dict,
    metrics: pd.DataFrame,
    derivative_metrics: pd.DataFrame,
    selected_m: list[int],
) -> None:
    selected = metrics[metrics["m"].isin(selected_m)].copy()
    gripper_transition_percent = summary["gripper"]["chunks_with_transition_percent"]
    global_m = summary["global_energy_threshold_m"]
    p10_m = summary["chunk_p10_energy_threshold_m"]
    recommendation = summary["offline_recommendation"]
    lines = [
        "# LIBERO action DCT expressivity",
        "",
        f"- Suite / env task: `{summary['suite']}` / `{summary['env_task_id']}`",
        f"- Task: {summary['task_language']}",
        f"- Dataset task index: `{summary['dataset_task_index']}`",
        f"- Episodes / frames / chunks: {summary['num_episodes']} / {summary['num_frames']} / {summary['num_chunks']}",
        f"- Chunk horizon / stride: {summary['horizon']} / {summary['stride']}",
        f"- Gripper is not transformed; chunks with a gripper transition: {gripper_transition_percent:.2f}%",
        "",
        "## Energy thresholds",
        "",
        "- Global 90% / 95% / 99% energy: "
        f"M={global_m['90_percent']} / M={global_m['95_percent']} / M={global_m['99_percent']}",
        "- Hardest-10% chunks 90% / 95% / 99% energy: "
        f"M={p10_m['90_percent']} / M={p10_m['95_percent']} / M={p10_m['99_percent']}",
        "",
        "## Offline recommendation",
        "",
        f"Use **M={recommendation['m']}** ({recommendation['coefficient_count']} coefficients) "
        "as the balanced starting point: "
        f"{recommendation['chunk_p10_energy_retained']:.2%} energy in the hardest 10% of chunks, "
        f"dimension-balanced NRMSE {recommendation['dimension_balanced_nrmse']:.4f}, "
        f"and {recommendation['compression_factor']:.2f}x total compression when the gripper "
        "sequence is retained unchanged.",
        "",
        "## Selected reconstruction points",
        "",
        "| M | Coefficients | Total representation | Compression | Global energy | Chunk p10 energy | "
        "RMSE | Dimension-balanced NRMSE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.m} | {row.coefficient_count} | {row.total_representation_fraction:.2%} | "
            f"{row.compression_factor:.2f}x | {row.energy_retained:.2%} | "
            f"{row.per_chunk_energy_retained_p10:.2%} | {row.rmse:.6f} | "
            f"{row.dimension_balanced_nrmse:.4f} |"
        )
    selected_derivative = derivative_metrics[derivative_metrics["m"].isin(selected_m)]
    lines.extend(
        [
            "",
            "## Dataset derivative vs. analytic DCT derivative",
            "",
            "The dataset derivative is `(a[t+1] - a[t]) * fps`. The analytic derivative is "
            "the exact time derivative of the truncated continuous DCT-II cosine expansion, "
            "evaluated at the current sample time `t`, matching FAFM's query-time convention.",
            "",
            "| M | Coefficients | Derivative RMSE | Derivative NRMSE | Mean correlation | Minimum correlation | RMS ratio |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in selected_derivative.itertuples(index=False):
        lines.append(
            f"| {row.m} | {row.coefficient_count} | {row.rmse:.6f} | "
            f"{row.dimension_balanced_nrmse:.4f} | "
            f"{row.mean_per_dimension_correlation:.4f} | "
            f"{row.minimum_per_dimension_correlation:.4f} | "
            f"{row.analytic_to_dataset_rms_ratio:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Plots",
            "",
            "![DCT overview](dct_overview.png)",
            "",
            "![Representative reconstruction](sample_reconstruction.png)",
            "",
            "![Derivative overview](derivative_overview.png)",
            "",
            "![Representative derivative comparison](sample_derivative_comparison.png)",
            "",
            "Offline reconstruction is a representation-capacity diagnostic. Rollout success is still "
            "required to measure control quality.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root)
    task_language, dataset_task_index = resolve_task(dataset_root, args.suite, args.env_task_id)
    selected_m = parse_m_values(args.m_values, args.horizon)
    output_dir = args.output_dir or Path(
        f"outputs/dct_analysis/{args.suite}_task{args.env_task_id}_h{args.horizon}"
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"dataset_root={dataset_root}")
    print(f"suite={args.suite} env_task_id={args.env_task_id}")
    print(f"task={task_language!r} dataset_task_index={dataset_task_index}")
    frame = load_task_actions(dataset_root, dataset_task_index)
    chunks, chunk_metadata, episode_lengths = build_action_chunks(frame, args.horizon, args.stride)
    if args.arm_dims < 1 or args.arm_dims >= chunks.shape[2]:
        raise ValueError(
            f"arm-dims must be within [1, action_dim-1]; got {args.arm_dims} for action_dim={chunks.shape[2]}"
        )

    arm_chunks = chunks[:, :, : args.arm_dims]
    gripper_chunks = chunks[:, :, args.arm_dims :]
    dataset_info = json.loads((dataset_root / "meta/info.json").read_text())
    fps = float(dataset_info["fps"])
    action_std = np.std(arm_chunks, axis=(0, 1))
    if np.any(action_std <= np.finfo(np.float64).eps):
        raise RuntimeError(f"At least one continuous action dimension has zero variance: {action_std}")
    coefficients = dct(arm_chunks, type=2, axis=1, norm="ortho")
    derivative_basis = dct_analytic_derivative_basis(args.horizon, fps)

    metrics, per_dim_metrics = compute_metrics(
        arm_chunks,
        coefficients,
        action_std,
        set(selected_m),
        non_dct_dims=gripper_chunks.shape[2],
    )
    energy_table = compute_energy_table(coefficients)
    derivative_metrics, per_dim_derivative_metrics, derivative_std = compute_derivative_metrics(
        arm_chunks,
        coefficients,
        derivative_basis,
        fps,
        set(selected_m),
    )
    full_rmse = float(metrics.loc[metrics["m"] == args.horizon - 1, "rmse"].iloc[0])
    if full_rmse > 1e-10:
        raise RuntimeError(f"Full DCT reconstruction failed numerical check: rmse={full_rmse}")

    gripper_transitions = np.count_nonzero(np.diff(gripper_chunks, axis=1), axis=(1, 2))
    chunks_with_transition = float(np.mean(gripper_transitions > 0) * 100)
    sample = plot_sample_reconstruction(
        arm_chunks,
        coefficients,
        selected_m,
        chunk_metadata,
        output_dir / "sample_reconstruction.png",
    )
    plot_overview(
        metrics,
        per_dim_metrics,
        energy_table,
        selected_m,
        output_dir / "dct_overview.png",
    )
    plot_derivative_overview(
        derivative_metrics,
        per_dim_derivative_metrics,
        selected_m,
        output_dir / "derivative_overview.png",
    )
    plot_sample_derivative_comparison(
        arm_chunks,
        coefficients,
        derivative_basis,
        fps,
        selected_m,
        sample,
        output_dir / "sample_derivative_comparison.png",
    )

    recommendation_candidates = metrics[
        metrics["selected"]
        & (metrics["per_chunk_energy_retained_p10"] >= 0.99)
        & (metrics["dimension_balanced_nrmse"] <= 0.10)
    ]
    recommendation = (
        recommendation_candidates.iloc[0] if not recommendation_candidates.empty else metrics.iloc[-1]
    )

    summary = {
        "dataset_root": str(dataset_root),
        "suite": args.suite,
        "env_task_id": args.env_task_id,
        "dataset_task_index": dataset_task_index,
        "task_language": task_language,
        "fps": fps,
        "num_episodes": int(frame["episode_index"].nunique()),
        "num_frames": int(len(frame)),
        "episode_length": {
            "min": int(np.min(episode_lengths)),
            "mean": float(np.mean(episode_lengths)),
            "median": float(np.median(episode_lengths)),
            "max": int(np.max(episode_lengths)),
        },
        "num_chunks": int(len(chunks)),
        "horizon": args.horizon,
        "stride": args.stride,
        "action_dim": int(chunks.shape[2]),
        "dct_action_dims": args.arm_dims,
        "selected_m": selected_m,
        "mode_convention": "FAFM: M is the highest retained mode; modes 0..M give M+1 coefficients",
        "action_std": action_std.tolist(),
        "global_energy_threshold_m": {
            "90_percent": first_m_at_threshold(metrics, "energy_retained", 0.90),
            "95_percent": first_m_at_threshold(metrics, "energy_retained", 0.95),
            "99_percent": first_m_at_threshold(metrics, "energy_retained", 0.99),
        },
        "chunk_p10_energy_threshold_m": {
            "90_percent": first_m_at_threshold(metrics, "per_chunk_energy_retained_p10", 0.90),
            "95_percent": first_m_at_threshold(metrics, "per_chunk_energy_retained_p10", 0.95),
            "99_percent": first_m_at_threshold(metrics, "per_chunk_energy_retained_p10", 0.99),
        },
        "offline_recommendation": {
            "selection_rule": "selected M with chunk p10 energy >= 99% and dimension-balanced NRMSE <= 0.10",
            "m": int(recommendation["m"]),
            "coefficient_count": int(recommendation["coefficient_count"]),
            "chunk_p10_energy_retained": float(recommendation["per_chunk_energy_retained_p10"]),
            "dimension_balanced_nrmse": float(recommendation["dimension_balanced_nrmse"]),
            "total_representation_fraction": float(recommendation["total_representation_fraction"]),
            "compression_factor": float(recommendation["compression_factor"]),
        },
        "gripper": {
            "dimensions": int(gripper_chunks.shape[2]),
            "strategy": "retained_without_dct",
            "chunks_with_transition_percent": chunks_with_transition,
            "mean_transitions_per_chunk": float(np.mean(gripper_transitions)),
            "max_transitions_per_chunk": int(np.max(gripper_transitions)),
        },
        "derivative_comparison": {
            "dataset_derivative": "(action[t+1] - action[t]) * fps",
            "analytic_derivative": "exact derivative of truncated orthonormal DCT-II cosine expansion",
            "evaluation_points": "current action sample times t_n for n=0..N-2",
            "dataset_derivative_std": derivative_std.tolist(),
            "selected_metrics": {
                str(int(row.m)): {
                    "coefficient_count": int(row.coefficient_count),
                    "rmse": float(row.rmse),
                    "dimension_balanced_nrmse": float(row.dimension_balanced_nrmse),
                    "mean_per_dimension_correlation": float(row.mean_per_dimension_correlation),
                    "minimum_per_dimension_correlation": float(row.minimum_per_dimension_correlation),
                    "analytic_to_dataset_rms_ratio": float(row.analytic_to_dataset_rms_ratio),
                }
                for row in derivative_metrics[derivative_metrics["m"].isin(selected_m)].itertuples(
                    index=False
                )
            },
        },
        "representative_sample": sample,
    }

    metrics.to_csv(output_dir / "reconstruction_metrics.csv", index=False)
    per_dim_metrics.to_csv(output_dir / "per_dimension_metrics.csv", index=False)
    energy_table.to_csv(output_dir / "coefficient_energy.csv", index=False)
    derivative_metrics.to_csv(output_dir / "derivative_metrics.csv", index=False)
    per_dim_derivative_metrics.to_csv(output_dir / "per_dimension_derivative_metrics.csv", index=False)
    chunk_metadata.to_csv(output_dir / "chunk_index.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(output_dir / "report.md", summary, metrics, derivative_metrics, selected_m)

    print(f"episodes={summary['num_episodes']} frames={summary['num_frames']} chunks={summary['num_chunks']}")
    print(f"global_energy_threshold_m={summary['global_energy_threshold_m']}")
    print(f"chunk_p10_energy_threshold_m={summary['chunk_p10_energy_threshold_m']}")
    print(f"offline_recommendation={summary['offline_recommendation']}")
    print(metrics.loc[metrics["m"].isin(selected_m)].to_string(index=False))
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
