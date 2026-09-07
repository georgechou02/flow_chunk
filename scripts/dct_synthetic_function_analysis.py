#!/usr/bin/env python3

"""Compare discrete samples and finite differences with a FAFM-style DCT model.

The source is a deterministic, analytic six-dimensional function built from
multiple frequencies, a chirp, amplitude modulation, local smooth pulses, and a
sharp tanh transition. Both its value and exact derivative are available.

FAFM convention is used throughout: M is the highest retained DCT frequency,
so modes 0 through M (M+1 coefficients) are retained.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lerobot-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.special import expit

DIMENSION_NAMES = ["f0", "f1", "f2", "f3", "f4", "f5"]
DEFAULT_M_VALUES = "1,2,4,8,12,16,24,31"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--function-variant",
        choices=("original", "coupled_delayed_exponential"),
        default="original",
        help="Synthetic analytic function family.",
    )
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--m-values",
        default=DEFAULT_M_VALUES,
        help="Highlighted FAFM mode indices; M retains modes 0 through M.",
    )
    parser.add_argument("--dense-points", type=int, default=1200)
    parser.add_argument(
        "--noise-std-fraction",
        type=float,
        default=0.0,
        help=(
            "Gaussian noise std as a fraction of each clean sample dimension's std. "
            "Noise is added only after sampling."
        ),
    )
    parser.add_argument("--noise-seed", type=int, default=20260721)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dct_analysis/synthetic_complex_6d_n32_fps10"),
    )
    return parser.parse_args()


def parse_m_values(raw: str, num_samples: int) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    invalid = [m for m in values if m < 0 or m >= num_samples]
    if invalid:
        raise ValueError(f"M must be in [0, {num_samples - 1}], got {invalid}")
    if num_samples - 1 not in values:
        values.append(num_samples - 1)
    return sorted(values)


def complex_function(t: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Return an analytic six-dimensional function and its exact derivative."""

    t = np.asarray(t, dtype=np.float64)
    two_pi = 2.0 * np.pi
    u = t / period

    phase0 = two_pi * 0.55 * t + 0.20
    latent0 = np.sin(phase0)
    dlatent0 = two_pi * 0.55 * np.cos(phase0)

    phase1 = two_pi * 1.35 * t - 0.40
    latent1 = np.cos(phase1)
    dlatent1 = -two_pi * 1.35 * np.sin(phase1)

    chirp_phase = two_pi * (0.35 * t + 0.5 * 0.55 * t**2) + 0.10
    chirp_rate = two_pi * (0.35 + 0.55 * t)
    latent2 = np.sin(chirp_phase)
    dlatent2 = chirp_rate * np.cos(chirp_phase)

    pulse_center = 0.29 * period
    pulse_width = 0.045 * period
    latent3 = np.exp(-0.5 * ((t - pulse_center) / pulse_width) ** 2)
    dlatent3 = -(t - pulse_center) / pulse_width**2 * latent3

    step_center = 0.54 * period
    step_width = 0.04 * period
    step_argument = (t - step_center) / step_width
    latent4 = np.tanh(step_argument)
    dlatent4 = (1.0 - latent4**2) / step_width

    modulation_phase = two_pi * 0.30 * t
    carrier_phase = two_pi * 2.55 * t + 0.50
    amplitude = 0.62 + 0.23 * np.cos(modulation_phase)
    damplitude = -0.23 * two_pi * 0.30 * np.sin(modulation_phase)
    latent5 = amplitude * np.sin(carrier_phase)
    dlatent5 = damplitude * np.sin(carrier_phase) + amplitude * two_pi * 2.55 * np.cos(carrier_phase)

    local_center = 0.77 * period
    local_width = 0.065 * period
    envelope = np.exp(-0.5 * ((t - local_center) / local_width) ** 2)
    denvelope = -(t - local_center) / local_width**2 * envelope
    local_phase = two_pi * 4.10 * t - 0.30
    latent6 = envelope * np.cos(local_phase)
    dlatent6 = denvelope * np.cos(local_phase) - envelope * two_pi * 4.10 * np.sin(local_phase)

    centered_u = u - 0.5
    latent7 = 2.0 * centered_u**3 - 0.35 * centered_u
    dlatent7 = (6.0 * centered_u**2 - 0.35) / period

    latent = np.stack(
        [latent0, latent1, latent2, latent3, latent4, latent5, latent6, latent7],
        axis=-1,
    )
    dlatent = np.stack(
        [
            dlatent0,
            dlatent1,
            dlatent2,
            dlatent3,
            dlatent4,
            dlatent5,
            dlatent6,
            dlatent7,
        ],
        axis=-1,
    )

    mixing = np.array(
        [
            [0.55, 0.18, 0.24, 0.28, 0.06, 0.18, 0.08, 0.10],
            [-0.12, 0.50, 0.22, -0.14, 0.09, 0.25, -0.07, 0.18],
            [0.20, -0.15, 0.45, 0.18, -0.12, 0.12, 0.15, -0.10],
            [0.08, 0.20, -0.18, 0.10, 0.35, 0.25, 0.12, 0.08],
            [0.15, 0.10, 0.12, -0.20, 0.20, -0.18, 0.38, 0.10],
            [-0.20, 0.16, 0.25, 0.15, -0.18, 0.20, 0.22, 0.25],
        ],
        dtype=np.float64,
    )
    offsets = np.array([0.12, -0.08, 0.05, -0.15, 0.10, -0.04], dtype=np.float64)
    dimension_scales = np.array([1.00, 0.85, 1.10, 0.55, 0.70, 0.90], dtype=np.float64)

    values = (latent @ mixing.T + offsets) * dimension_scales
    derivatives = (dlatent @ mixing.T) * dimension_scales
    return values, derivatives


def delayed_ring(
    t: np.ndarray,
    delay: float,
    gate_width: float,
    decay: float,
    frequency: float,
    phase: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothly activated, exponentially damped oscillation and its derivative."""

    relative_time = t - delay
    gate = expit(relative_time / gate_width)
    dgate = gate * (1.0 - gate) / gate_width
    positive_time = gate_width * np.logaddexp(0.0, relative_time / gate_width)
    dpositive_time = gate
    exponential = np.exp(-decay * positive_time)
    envelope = gate * exponential
    denvelope = exponential * (dgate - decay * gate * dpositive_time)
    oscillation_phase = 2.0 * np.pi * frequency * relative_time + phase
    value = envelope * np.sin(oscillation_phase)
    derivative = denvelope * np.sin(oscillation_phase) + envelope * (
        2.0 * np.pi * frequency * np.cos(oscillation_phase)
    )
    return value, derivative


def coupled_base_state(t: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Uncoupled analytic drivers used inside the coupled delayed system."""

    t = np.asarray(t, dtype=np.float64)
    two_pi = 2.0 * np.pi

    modulation_phase = two_pi * 0.37 * t
    phase0 = two_pi * (0.55 * t + 0.13 * np.sin(modulation_phase)) + 0.15
    dphase0 = two_pi * (0.55 + 0.13 * two_pi * 0.37 * np.cos(modulation_phase))
    exponential0 = np.exp(0.75 * (t / period - 0.5))
    state0 = 0.55 * np.sin(phase0) + 0.11 * (exponential0 - 1.0)
    dstate0 = 0.55 * dphase0 * np.cos(phase0) + 0.11 * 0.75 / period * exponential0

    chirp_phase = two_pi * (0.24 * t + 0.5 * 0.82 * t**2 + 0.012 * t**3) - 0.35
    dchirp_phase = two_pi * (0.24 + 0.82 * t + 0.036 * t**2)
    pulse1_width = 0.045 * period
    pulse1_center = 0.34 * period
    pulse1 = np.exp(-0.5 * ((t - pulse1_center) / pulse1_width) ** 2)
    dpulse1 = -(t - pulse1_center) / pulse1_width**2 * pulse1
    state1 = 0.48 * np.sin(chirp_phase) + 0.24 * pulse1
    dstate1 = 0.48 * dchirp_phase * np.cos(chirp_phase) + 0.24 * dpulse1

    ring2, dring2 = delayed_ring(
        t,
        delay=0.18 * period,
        gate_width=0.025 * period,
        decay=1.25,
        frequency=2.65,
        phase=0.20,
    )
    phase2 = two_pi * 1.05 * t + 0.60
    state2 = 0.52 * ring2 + 0.18 * np.cos(phase2)
    dstate2 = 0.52 * dring2 - 0.18 * two_pi * 1.05 * np.sin(phase2)

    relative3 = t - 0.31 * period
    width3 = 0.03 * period
    positive3 = width3 * np.logaddexp(0.0, relative3 / width3)
    dpositive3 = expit(relative3 / width3)
    rise3 = 1.0 - np.exp(-2.2 * positive3)
    drise3 = 2.2 * np.exp(-2.2 * positive3) * dpositive3
    ring3, dring3 = delayed_ring(
        t,
        delay=0.57 * period,
        gate_width=0.018 * period,
        decay=2.1,
        frequency=4.25,
        phase=-0.50,
    )
    state3 = 0.34 * rise3 + 0.27 * ring3 - 0.12
    dstate3 = 0.34 * drise3 + 0.27 * dring3

    burst4_center = 0.73 * period
    burst4_width = 0.055 * period
    burst4_envelope = np.exp(-0.5 * ((t - burst4_center) / burst4_width) ** 2)
    dburst4_envelope = -(t - burst4_center) / burst4_width**2 * burst4_envelope
    burst4_phase = two_pi * 4.55 * t + 0.30
    burst4 = burst4_envelope * np.sin(burst4_phase)
    dburst4 = dburst4_envelope * np.sin(burst4_phase) + burst4_envelope * (
        two_pi * 4.55 * np.cos(burst4_phase)
    )
    step4a = np.tanh((t - 0.46 * period) / (0.035 * period))
    step4b = np.tanh((t - 0.84 * period) / (0.028 * period))
    dstep4a = (1.0 - step4a**2) / (0.035 * period)
    dstep4b = (1.0 - step4b**2) / (0.028 * period)
    state4 = 0.42 * burst4 + 0.22 * (step4a - step4b)
    dstate4 = 0.42 * dburst4 + 0.22 * (dstep4a - dstep4b)

    ring5a, dring5a = delayed_ring(
        t,
        delay=0.10 * period,
        gate_width=0.02 * period,
        decay=0.65,
        frequency=1.75,
        phase=0.90,
    )
    ring5b, dring5b = delayed_ring(
        t,
        delay=0.68 * period,
        gate_width=0.018 * period,
        decay=3.0,
        frequency=3.70,
        phase=-0.20,
    )
    window_start = 0.38 * period
    window_end = 0.62 * period
    window_width = 0.025 * period
    window = (
        window_width * np.logaddexp(0.0, (t - window_start) / window_width)
        - window_width * np.logaddexp(0.0, (t - window_end) / window_width)
    ) / (window_end - window_start)
    dwindow = (expit((t - window_start) / window_width) - expit((t - window_end) / window_width)) / (
        window_end - window_start
    )
    state5 = 0.34 * ring5a - 0.30 * ring5b + 0.16 * window
    dstate5 = 0.34 * dring5a - 0.30 * dring5b + 0.16 * dwindow

    state = np.stack([state0, state1, state2, state3, state4, state5], axis=-1)
    derivative = np.stack([dstate0, dstate1, dstate2, dstate3, dstate4, dstate5], axis=-1)
    return state, derivative


def coupled_delayed_exponential_function(t: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Six-dimensional analytic system with explicit instantaneous and delayed coupling."""

    t = np.asarray(t, dtype=np.float64)
    base, dbase = coupled_base_state(t, period)

    instantaneous_coupling = np.array(
        [
            [0.00, 0.22, -0.12, 0.00, 0.10, 0.15],
            [0.16, 0.00, 0.14, -0.10, 0.00, 0.08],
            [-0.10, 0.18, 0.00, 0.17, -0.11, 0.00],
            [0.07, -0.15, 0.19, 0.00, 0.14, 0.06],
            [0.13, 0.00, -0.10, 0.18, 0.00, 0.16],
            [-0.14, 0.12, 0.08, -0.09, 0.20, 0.00],
        ],
        dtype=np.float64,
    )
    nonlinear_base = np.tanh(1.4 * base)
    dnonlinear_base = 1.4 * (1.0 - nonlinear_base**2) * dbase
    instantaneous = nonlinear_base @ instantaneous_coupling.T
    dinstantaneous = dnonlinear_base @ instantaneous_coupling.T

    delays = period * np.array([0.06, 0.10, 0.14, 0.18, 0.23, 0.28])
    delay_widths = period * np.array([0.018, 0.020, 0.022, 0.024, 0.026, 0.028])
    delayed_sources = []
    ddelayed_sources = []
    for source_dim, (delay, width) in enumerate(zip(delays, delay_widths, strict=True)):
        shifted_base, shifted_derivative = coupled_base_state(t - delay, period)
        gate = expit((t - delay) / width)
        dgate = gate * (1.0 - gate) / width
        shifted_nonlinear = np.tanh(1.6 * shifted_base[..., source_dim])
        dshifted_nonlinear = 1.6 * (1.0 - shifted_nonlinear**2) * shifted_derivative[..., source_dim]
        delayed_sources.append(gate * shifted_nonlinear)
        ddelayed_sources.append(dgate * shifted_nonlinear + gate * dshifted_nonlinear)
    delayed_sources_array = np.stack(delayed_sources, axis=-1)
    ddelayed_sources_array = np.stack(ddelayed_sources, axis=-1)
    delayed_coupling = np.array(
        [
            [0.00, 0.18, 0.00, -0.12, 0.00, 0.14],
            [-0.10, 0.00, 0.20, 0.00, 0.13, 0.00],
            [0.16, -0.11, 0.00, 0.19, 0.00, 0.00],
            [0.00, 0.15, -0.12, 0.00, 0.18, 0.00],
            [0.14, 0.00, 0.11, -0.16, 0.00, 0.17],
            [-0.13, 0.16, 0.00, 0.12, -0.10, 0.00],
        ],
        dtype=np.float64,
    )
    delayed = delayed_sources_array @ delayed_coupling.T
    ddelayed = ddelayed_sources_array @ delayed_coupling.T

    bilinear = np.stack(
        [
            base[..., 1] * base[..., 4],
            base[..., 2] * base[..., 0],
            base[..., 3] * base[..., 1],
            base[..., 4] * base[..., 2],
            base[..., 5] * base[..., 3],
            base[..., 0] * base[..., 5],
        ],
        axis=-1,
    )
    dbilinear = np.stack(
        [
            dbase[..., 1] * base[..., 4] + base[..., 1] * dbase[..., 4],
            dbase[..., 2] * base[..., 0] + base[..., 2] * dbase[..., 0],
            dbase[..., 3] * base[..., 1] + base[..., 3] * dbase[..., 1],
            dbase[..., 4] * base[..., 2] + base[..., 4] * dbase[..., 2],
            dbase[..., 5] * base[..., 3] + base[..., 5] * dbase[..., 3],
            dbase[..., 0] * base[..., 5] + base[..., 0] * dbase[..., 5],
        ],
        axis=-1,
    )
    bilinear_scale = np.array([0.22, -0.18, 0.24, -0.20, 0.19, -0.21])

    dimension_scale = np.array([1.00, 0.92, 1.08, 0.82, 0.88, 0.96])
    values = (base + instantaneous + delayed + bilinear_scale * bilinear) * dimension_scale
    derivatives = (dbase + dinstantaneous + ddelayed + bilinear_scale * dbilinear) * dimension_scale
    return values, derivatives


def evaluate_source_function(t: np.ndarray, period: float, variant: str) -> tuple[np.ndarray, np.ndarray]:
    if variant == "original":
        return complex_function(t, period)
    if variant == "coupled_delayed_exponential":
        return coupled_delayed_exponential_function(t, period)
    raise ValueError(f"Unknown function variant: {variant}")


def dct_scale(num_samples: int) -> np.ndarray:
    scale = np.full(num_samples, np.sqrt(2.0 / num_samples), dtype=np.float64)
    scale[0] = np.sqrt(1.0 / num_samples)
    return scale


def evaluate_dct(
    coefficients: np.ndarray,
    query_time: np.ndarray,
    fps: float,
    m: int,
    *,
    derivative: bool,
) -> np.ndarray:
    """Evaluate FAFM's continuous DCT-II decoder or its analytic derivative."""

    num_samples = coefficients.shape[0]
    frequency = np.arange(m + 1, dtype=np.float64)
    phase = (
        np.pi
        * frequency[:, None]
        * (np.asarray(query_time, dtype=np.float64)[None, :] * fps + 0.5)
        / num_samples
    )
    basis = dct_scale(num_samples)[: m + 1, None]
    if derivative:
        basis = -basis * (np.pi * frequency[:, None] * fps / num_samples) * np.sin(phase)
    else:
        basis = basis * np.cos(phase)
    return np.einsum("kt,kd->td", basis, coefficients[: m + 1], optimize=True)


def balanced_nrmse(error: np.ndarray, reference: np.ndarray) -> float:
    scale = np.std(reference, axis=0)
    if np.any(scale <= np.finfo(np.float64).eps):
        raise RuntimeError(f"At least one reference dimension has zero variance: {scale}")
    return float(np.sqrt(np.mean((error / scale[None, :]) ** 2)))


def correlation(reference: np.ndarray, prediction: np.ndarray) -> float:
    correlations = []
    for dim in range(reference.shape[1]):
        reference_dim = reference[:, dim]
        prediction_dim = prediction[:, dim]
        if (
            np.std(reference_dim) <= np.finfo(np.float64).eps
            or np.std(prediction_dim) <= np.finfo(np.float64).eps
        ):
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(reference_dim, prediction_dim)[0, 1]))
    return float(np.mean(correlations))


def compute_metrics(
    samples: np.ndarray,
    true_dense: np.ndarray,
    true_derivative_current: np.ndarray,
    finite_difference: np.ndarray,
    coefficients: np.ndarray,
    sample_time: np.ndarray,
    dense_time: np.ndarray,
    fps: float,
    selected_m: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | bool]] = []
    per_dim_rows: list[dict[str, float | int | str]] = []
    current_time = sample_time[:-1]

    for m in range(len(sample_time)):
        reconstructed_samples = evaluate_dct(coefficients, sample_time, fps, m, derivative=False)
        reconstructed_dense = evaluate_dct(coefficients, dense_time, fps, m, derivative=False)
        dct_derivative = evaluate_dct(coefficients, current_time, fps, m, derivative=True)

        sample_error = reconstructed_samples - samples
        dense_error = reconstructed_dense - true_dense
        fd_error = dct_derivative - finite_difference
        true_derivative_error = dct_derivative - true_derivative_current
        rows.append(
            {
                "m": m,
                "coefficient_count": m + 1,
                "selected": m in selected_m,
                "sample_reconstruction_rmse": float(np.sqrt(np.mean(sample_error**2))),
                "sample_reconstruction_nrmse": balanced_nrmse(sample_error, samples),
                "continuous_function_rmse": float(np.sqrt(np.mean(dense_error**2))),
                "continuous_function_nrmse": balanced_nrmse(dense_error, true_dense),
                "finite_difference_derivative_rmse": float(np.sqrt(np.mean(fd_error**2))),
                "finite_difference_derivative_nrmse": balanced_nrmse(fd_error, finite_difference),
                "finite_difference_derivative_correlation": correlation(finite_difference, dct_derivative),
                "true_derivative_rmse": float(np.sqrt(np.mean(true_derivative_error**2))),
                "true_derivative_nrmse": balanced_nrmse(true_derivative_error, true_derivative_current),
                "true_derivative_correlation": correlation(true_derivative_current, dct_derivative),
            }
        )

        for dim, name in enumerate(DIMENSION_NAMES):
            per_dim_rows.append(
                {
                    "m": m,
                    "coefficient_count": m + 1,
                    "dimension": dim,
                    "name": name,
                    "sample_reconstruction_rmse": float(np.sqrt(np.mean(sample_error[:, dim] ** 2))),
                    "continuous_function_rmse": float(np.sqrt(np.mean(dense_error[:, dim] ** 2))),
                    "finite_difference_derivative_rmse": float(np.sqrt(np.mean(fd_error[:, dim] ** 2))),
                    "true_derivative_rmse": float(np.sqrt(np.mean(true_derivative_error[:, dim] ** 2))),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(per_dim_rows)


def plot_signal_comparison(
    samples: np.ndarray,
    true_dense: np.ndarray,
    coefficients: np.ndarray,
    sample_time: np.ndarray,
    dense_time: np.ndarray,
    fps: float,
    plot_m: list[int],
    samples_are_noisy: bool,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(6, 1, figsize=(13, 13), sharex=True, constrained_layout=True)
    for dim, axis in enumerate(axes):
        axis.plot(dense_time, true_dense[:, dim], color="0.65", linestyle="--", label="true function")
        sample_label = "noisy discrete samples" if samples_are_noisy else "discrete samples"
        axis.scatter(sample_time, samples[:, dim], color="black", s=16, zorder=5, label=sample_label)
        for m in plot_m:
            prediction = evaluate_dct(coefficients, dense_time, fps, m, derivative=False)
            axis.plot(dense_time, prediction[:, dim], linewidth=1.2, label=f"DCT M={m}")
        axis.set_ylabel(DIMENSION_NAMES[dim])
        axis.grid(alpha=0.2)
    axes[0].legend(ncols=min(6, len(plot_m) + 2), fontsize=8)
    axes[-1].set_xlabel("Time (seconds)")
    noise_text = "noisy " if samples_are_noisy else ""
    fig.suptitle(f"Continuous function and {noise_text}discrete samples vs. FAFM-style DCT reconstruction")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_derivative_comparison(
    finite_difference: np.ndarray,
    true_derivative_dense: np.ndarray,
    coefficients: np.ndarray,
    sample_time: np.ndarray,
    derivative_dense_time: np.ndarray,
    fps: float,
    plot_m: list[int],
    samples_are_noisy: bool,
    output_path: Path,
) -> None:
    current_time = sample_time[:-1]
    fig, axes = plt.subplots(6, 1, figsize=(13, 13), sharex=True, constrained_layout=True)
    for dim, axis in enumerate(axes):
        axis.plot(
            derivative_dense_time,
            true_derivative_dense[:, dim],
            color="0.65",
            linestyle="--",
            label="true analytic derivative",
        )
        axis.plot(
            current_time,
            finite_difference[:, dim],
            color="black",
            marker="o",
            markersize=3,
            linewidth=1.7,
            label="forward difference at current sample",
        )
        for m in plot_m:
            derivative = evaluate_dct(coefficients, derivative_dense_time, fps, m, derivative=True)
            axis.plot(derivative_dense_time, derivative[:, dim], linewidth=1.2, label=f"DCT analytic M={m}")
        axis.set_ylabel(f"d{DIMENSION_NAMES[dim]}/dt")
        axis.grid(alpha=0.2)
    axes[0].legend(ncols=min(6, len(plot_m) + 2), fontsize=8)
    axes[-1].set_xlabel("Current sample time (seconds)")
    source_text = " from noisy samples" if samples_are_noisy else ""
    fig.suptitle(f"Forward finite difference vs. FAFM-style DCT analytic derivative{source_text}")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def compute_derivative_method_metrics(
    samples: np.ndarray,
    true_derivative_samples: np.ndarray,
    coefficients: np.ndarray,
    sample_time: np.ndarray,
    fps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare all methods on the common interior grid t[1:-1]."""

    common_time = sample_time[1:-1]
    true_derivative = true_derivative_samples[1:-1]
    forward_difference = np.diff(samples, axis=0)[1:] * fps
    central_difference = (samples[2:] - samples[:-2]) * (0.5 * fps)
    true_std = np.std(true_derivative, axis=0)

    rows: list[dict[str, float | int | str | None]] = []
    per_dim_rows: list[dict[str, float | int | str | None]] = []

    def add_method(method: str, prediction: np.ndarray, m: int | None) -> None:
        error = prediction - true_derivative
        rows.append(
            {
                "method": method,
                "m": m,
                "coefficient_count": None if m is None else m + 1,
                "rmse": float(np.sqrt(np.mean(error**2))),
                "dimension_balanced_nrmse": balanced_nrmse(error, true_derivative),
                "mae": float(np.mean(np.abs(error))),
                "p95_absolute_error": float(np.quantile(np.abs(error), 0.95)),
                "mean_per_dimension_correlation": correlation(true_derivative, prediction),
            }
        )
        for dim, name in enumerate(DIMENSION_NAMES):
            rmse = float(np.sqrt(np.mean(error[:, dim] ** 2)))
            per_dim_rows.append(
                {
                    "method": method,
                    "m": m,
                    "coefficient_count": None if m is None else m + 1,
                    "dimension": dim,
                    "name": name,
                    "rmse": rmse,
                    "normalized_rmse": float(rmse / true_std[dim]),
                    "mae": float(np.mean(np.abs(error[:, dim]))),
                    "correlation": correlation(
                        true_derivative[:, dim : dim + 1],
                        prediction[:, dim : dim + 1],
                    ),
                }
            )

    add_method("forward_difference", forward_difference, None)
    add_method("central_difference", central_difference, None)
    for m in range(len(sample_time)):
        add_method(
            "dct_analytic",
            evaluate_dct(coefficients, common_time, fps, m, derivative=True),
            m,
        )
    return pd.DataFrame(rows), pd.DataFrame(per_dim_rows)


def plot_derivative_methods_vs_true(
    samples: np.ndarray,
    true_derivative_samples: np.ndarray,
    coefficients: np.ndarray,
    sample_time: np.ndarray,
    fps: float,
    comparison_m: list[int],
    samples_are_noisy: bool,
    output_path: Path,
) -> None:
    common_time = sample_time[1:-1]
    true_derivative = true_derivative_samples[1:-1]
    forward_difference = np.diff(samples, axis=0)[1:] * fps
    central_difference = (samples[2:] - samples[:-2]) * (0.5 * fps)
    dct_derivatives = {
        m: evaluate_dct(coefficients, common_time, fps, m, derivative=True) for m in comparison_m
    }
    dct_colors = ["tab:blue", "tab:purple", "tab:red", "tab:brown"]

    fig, axes = plt.subplots(6, 1, figsize=(13, 13), sharex=True, constrained_layout=True)
    for dim, axis in enumerate(axes):
        axis.plot(
            common_time,
            true_derivative[:, dim],
            color="black",
            linewidth=2.2,
            label="true derivative",
        )
        axis.plot(
            common_time,
            forward_difference[:, dim],
            color="tab:orange",
            marker="o",
            markersize=3,
            linewidth=1.2,
            label="forward difference",
        )
        axis.plot(
            common_time,
            central_difference[:, dim],
            color="tab:green",
            marker="o",
            markersize=3,
            linewidth=1.2,
            label="central difference",
        )
        for (m, derivative), color in zip(dct_derivatives.items(), dct_colors, strict=False):
            axis.plot(
                common_time,
                derivative[:, dim],
                color=color,
                linewidth=1.2,
                label=f"DCT analytic M={m}",
            )
        axis.set_ylabel(f"d{DIMENSION_NAMES[dim]}/dt")
        axis.grid(alpha=0.2)
    axes[0].legend(ncols=min(6, len(comparison_m) + 3), fontsize=8)
    axes[-1].set_xlabel("Common interior sample time (seconds)")
    source_text = " (estimators use noisy samples)" if samples_are_noisy else ""
    fig.suptitle(
        "Forward difference, central difference, and DCT analytic derivative vs. truth" + source_text
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_derivative_method_errors(
    method_metrics: pd.DataFrame,
    per_dim_method_metrics: pd.DataFrame,
    selected_m: list[int],
    comparison_m: list[int],
    output_path: Path,
) -> None:
    forward = method_metrics[method_metrics["method"] == "forward_difference"].iloc[0]
    central = method_metrics[method_metrics["method"] == "central_difference"].iloc[0]
    dct_metrics = method_metrics[method_metrics["method"] == "dct_analytic"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for column, title, ylabel, axis in (
        (
            "dimension_balanced_nrmse",
            "Derivative error vs. true derivative",
            "Dimension-balanced NRMSE",
            axes[0, 0],
        ),
        ("rmse", "Derivative RMSE vs. true derivative", "RMSE", axes[0, 1]),
        (
            "mean_per_dimension_correlation",
            "Derivative correlation with truth",
            "Mean per-dimension correlation",
            axes[1, 0],
        ),
    ):
        axis.plot(dct_metrics["m"], dct_metrics[column], marker="o", markersize=3, label="DCT analytic")
        axis.axhline(float(forward[column]), color="tab:orange", linestyle="--", label="forward difference")
        axis.axhline(float(central[column]), color="tab:green", linestyle="--", label="central difference")
        axis.set(title=title, xlabel="Highest retained mode M", ylabel=ylabel)
        axis.set_xticks(selected_m)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    method_specs: list[tuple[str, int | None, str]] = [
        ("forward_difference", None, "Forward"),
        ("central_difference", None, "Central"),
    ] + [("dct_analytic", m, f"DCT M={m}") for m in comparison_m]
    x = np.arange(len(DIMENSION_NAMES), dtype=np.float64)
    width = 0.82 / len(method_specs)
    for index, (method, m, label) in enumerate(method_specs):
        selected = per_dim_method_metrics[
            (per_dim_method_metrics["method"] == method)
            & (per_dim_method_metrics["m"].isna() if m is None else per_dim_method_metrics["m"].eq(m))
        ].sort_values("dimension")
        axes[1, 1].bar(
            x + (index - (len(method_specs) - 1) / 2) * width,
            selected["normalized_rmse"],
            width=width,
            label=label,
        )
    axes[1, 1].set(
        title="Per-dimension error on common sample grid",
        xlabel="Function dimension",
        ylabel="NRMSE",
        xticks=x,
        xticklabels=DIMENSION_NAMES,
    )
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].legend(ncols=2, fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_error_overview(metrics: pd.DataFrame, selected_m: list[int], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].plot(metrics["m"], metrics["sample_reconstruction_nrmse"], marker="o", markersize=3)
    axes[0, 0].set(title="Discrete-sample reconstruction", ylabel="Dimension-balanced NRMSE")
    axes[0, 1].plot(metrics["m"], metrics["continuous_function_nrmse"], marker="o", markersize=3)
    axes[0, 1].set(title="Continuous-function reconstruction", ylabel="Dimension-balanced NRMSE")
    axes[1, 0].plot(metrics["m"], metrics["finite_difference_derivative_nrmse"], marker="o", markersize=3)
    axes[1, 0].set(title="DCT derivative vs. finite difference", ylabel="Dimension-balanced NRMSE")
    axes[1, 1].plot(metrics["m"], metrics["true_derivative_nrmse"], marker="o", markersize=3)
    axes[1, 1].set(title="DCT derivative vs. true derivative", ylabel="Dimension-balanced NRMSE")
    for axis in axes.ravel():
        axis.set_xlabel("Highest retained mode M (keeps M+1 coefficients)")
        axis.set_xticks(selected_m)
        axis.grid(alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    output_path: Path,
    args: argparse.Namespace,
    metrics: pd.DataFrame,
    method_metrics: pd.DataFrame,
    selected_m: list[int],
    comparison_m: list[int],
    finite_difference_baseline: dict[str, float],
) -> None:
    selected = metrics[metrics["m"].isin(selected_m)]
    lines = [
        "# Synthetic multidimensional DCT experiment",
        "",
        f"- Samples / fps: `{args.num_samples}` / `{args.fps:g} Hz`",
        f"- Function variant: `{args.function_variant}`",
        "- Coupling: off-diagonal instantaneous `tanh` coupling, six directed delayed "
        "cross-dimensional feedback paths, and cyclic bilinear interactions."
        if args.function_variant == "coupled_delayed_exponential"
        else "- Coupling: none beyond shared latent mixing.",
        f"- DCT period: `{args.num_samples / args.fps:g} s`",
        "- Dimensions: `6`",
        f"- Sampling-noise std: `{args.noise_std_fraction:.2%}` of each dimension's "
        "clean-sample standard deviation",
        f"- Noise seed: `{args.noise_seed}`",
        "- FAFM convention: `M` retains modes `0..M`, i.e. `M+1` coefficients.",
        "- Derivatives are evaluated at the current sample `t_n`; the finite difference is "
        "`(f(t_{n+1}) - f(t_n)) * fps`.",
        "",
        "The analytic function is unchanged. Noise is added only after discrete sampling; "
        "DCT coefficients and both difference estimators use only those samples. The known "
        "true function and derivative are used exclusively for evaluation.",
        "",
        "## Finite-difference baseline",
        "",
        f"Finite difference vs. true derivative at the current sample: RMSE "
        f"`{finite_difference_baseline['rmse']:.6f}`, dimension-balanced NRMSE "
        f"`{finite_difference_baseline['nrmse']:.6f}`.",
        "",
        "## Selected modes",
        "",
        "| M | Coefficients | Sample NRMSE | Continuous NRMSE | FD derivative NRMSE | "
        "True derivative NRMSE | FD correlation |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.m} | {row.coefficient_count} | {row.sample_reconstruction_nrmse:.6f} | "
            f"{row.continuous_function_nrmse:.6f} | "
            f"{row.finite_difference_derivative_nrmse:.6f} | "
            f"{row.true_derivative_nrmse:.6f} | "
            f"{row.finite_difference_derivative_correlation:.6f} |"
        )
    forward = method_metrics[method_metrics["method"] == "forward_difference"].iloc[0]
    central = method_metrics[method_metrics["method"] == "central_difference"].iloc[0]
    lines.extend(
        [
            "",
            "## Forward, central, and DCT derivatives vs. truth",
            "",
            "All methods below are evaluated on the same interior sample grid `t[1:-1]`.",
            "",
            "| Method | RMSE | Dimension-balanced NRMSE | Mean correlation |",
            "|:---|---:|---:|---:|",
            f"| Forward difference | {forward.rmse:.6f} | "
            f"{forward.dimension_balanced_nrmse:.6f} | "
            f"{forward.mean_per_dimension_correlation:.6f} |",
            f"| Central difference | {central.rmse:.6f} | "
            f"{central.dimension_balanced_nrmse:.6f} | "
            f"{central.mean_per_dimension_correlation:.6f} |",
        ]
    )
    for m in comparison_m:
        row = method_metrics[(method_metrics["method"] == "dct_analytic") & method_metrics["m"].eq(m)].iloc[0]
        lines.append(
            f"| DCT analytic M={m} | {row.rmse:.6f} | "
            f"{row.dimension_balanced_nrmse:.6f} | "
            f"{row.mean_per_dimension_correlation:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Plots",
            "",
            "![Signal comparison](signal_comparison.png)",
            "",
            "![Derivative comparison](derivative_comparison.png)",
            "",
            "![Derivative methods vs truth](derivative_methods_vs_true.png)",
            "",
            "![Derivative method errors](derivative_method_errors.png)",
            "",
            "![Error overview](error_overview.png)",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.num_samples < 3:
        raise ValueError("num-samples must be at least 3")
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.dense_points < 100:
        raise ValueError("dense-points must be at least 100")
    if args.noise_std_fraction < 0:
        raise ValueError("noise-std-fraction must be non-negative")

    selected_m = parse_m_values(args.m_values, args.num_samples)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    period = args.num_samples / args.fps
    sample_time = np.arange(args.num_samples, dtype=np.float64) / args.fps
    dense_time = np.linspace(sample_time[0], sample_time[-1], args.dense_points)
    derivative_dense_time = np.linspace(sample_time[0], sample_time[-2], args.dense_points)

    clean_samples, true_derivative_samples = evaluate_source_function(
        sample_time, period, args.function_variant
    )
    true_dense, _ = evaluate_source_function(dense_time, period, args.function_variant)
    _, true_derivative_dense = evaluate_source_function(derivative_dense_time, period, args.function_variant)
    noise_std = args.noise_std_fraction * np.std(clean_samples, axis=0)
    rng = np.random.default_rng(args.noise_seed)
    sample_noise = rng.normal(size=clean_samples.shape) * noise_std[None, :]
    samples = clean_samples + sample_noise
    samples_are_noisy = args.noise_std_fraction > 0
    finite_difference = np.diff(samples, axis=0) * args.fps
    true_derivative_current = true_derivative_samples[:-1]
    coefficients = dct(samples, type=2, axis=0, norm="ortho")

    metrics, per_dim_metrics = compute_metrics(
        samples,
        true_dense,
        true_derivative_current,
        finite_difference,
        coefficients,
        sample_time,
        dense_time,
        args.fps,
        set(selected_m),
    )
    method_metrics, per_dim_method_metrics = compute_derivative_method_metrics(
        samples,
        true_derivative_samples,
        coefficients,
        sample_time,
        args.fps,
    )
    full_sample_rmse = float(
        metrics.loc[metrics["m"] == args.num_samples - 1, "sample_reconstruction_rmse"].iloc[0]
    )
    if full_sample_rmse > 1e-10:
        raise RuntimeError(f"Full DCT failed to reconstruct samples: RMSE={full_sample_rmse}")

    plot_m = [m for m in selected_m if m in (8, 16, 24, args.num_samples - 1)]
    if not plot_m:
        plot_m = selected_m[-min(4, len(selected_m)) :]
    comparison_m = [m for m in selected_m if m in (16, 24, args.num_samples - 1)]
    if not comparison_m:
        comparison_m = plot_m[-min(3, len(plot_m)) :]
    plot_signal_comparison(
        samples,
        true_dense,
        coefficients,
        sample_time,
        dense_time,
        args.fps,
        plot_m,
        samples_are_noisy,
        output_dir / "signal_comparison.png",
    )
    plot_derivative_comparison(
        finite_difference,
        true_derivative_dense,
        coefficients,
        sample_time,
        derivative_dense_time,
        args.fps,
        plot_m,
        samples_are_noisy,
        output_dir / "derivative_comparison.png",
    )
    plot_derivative_methods_vs_true(
        samples,
        true_derivative_samples,
        coefficients,
        sample_time,
        args.fps,
        comparison_m,
        samples_are_noisy,
        output_dir / "derivative_methods_vs_true.png",
    )
    plot_derivative_method_errors(
        method_metrics,
        per_dim_method_metrics,
        selected_m,
        comparison_m,
        output_dir / "derivative_method_errors.png",
    )
    plot_error_overview(metrics, selected_m, output_dir / "error_overview.png")

    baseline_error = finite_difference - true_derivative_current
    finite_difference_baseline = {
        "rmse": float(np.sqrt(np.mean(baseline_error**2))),
        "nrmse": balanced_nrmse(baseline_error, true_derivative_current),
        "mean_per_dimension_correlation": correlation(true_derivative_current, finite_difference),
    }
    summary = {
        "function_variant": args.function_variant,
        "function": (
            "six-dimensional analytic system with off-diagonal instantaneous nonlinear "
            "coupling, per-source delayed cross-dimensional feedback, bilinear interactions, "
            "exponential transients, delayed ringing, chirps, narrow pulses, and localized "
            "near-Nyquist oscillations"
            if args.function_variant == "coupled_delayed_exponential"
            else "six-dimensional analytic mixture of sinusoids, chirp, Gaussian pulse, "
            "tanh transition, amplitude modulation, localized near-Nyquist oscillation, "
            "and polynomial trend"
        ),
        "num_samples": args.num_samples,
        "fps": args.fps,
        "period": period,
        "dimension": samples.shape[1],
        "sampling_noise": {
            "std_fraction_of_clean_dimension_std": args.noise_std_fraction,
            "seed": args.noise_seed,
            "per_dimension_std": noise_std.tolist(),
        },
        "selected_m": selected_m,
        "plotted_m": plot_m,
        "derivative_method_comparison_m": comparison_m,
        "mode_convention": "FAFM: M keeps modes 0 through M (M+1 coefficients)",
        "derivative_evaluation_points": "current samples t_n, n=0..N-2",
        "finite_difference": "(f(t[n+1]) - f(t[n])) * fps, assigned to t[n]",
        "finite_difference_vs_true_derivative": finite_difference_baseline,
        "common_interior_grid_derivative_methods": {
            str(row.method if row.method != "dct_analytic" else f"dct_m{int(row.m)}"): {
                "rmse": float(row.rmse),
                "dimension_balanced_nrmse": float(row.dimension_balanced_nrmse),
                "mean_per_dimension_correlation": float(row.mean_per_dimension_correlation),
            }
            for row in method_metrics[
                (method_metrics["method"] != "dct_analytic") | method_metrics["m"].isin(comparison_m)
            ].itertuples(index=False)
        },
        "selected_metrics": {
            str(int(row.m)): {
                key: float(getattr(row, key))
                for key in (
                    "sample_reconstruction_nrmse",
                    "continuous_function_nrmse",
                    "finite_difference_derivative_nrmse",
                    "finite_difference_derivative_correlation",
                    "true_derivative_nrmse",
                    "true_derivative_correlation",
                )
            }
            for row in metrics[metrics["m"].isin(selected_m)].itertuples(index=False)
        },
    }

    np.savez_compressed(
        output_dir / "synthetic_data.npz",
        sample_time=sample_time,
        clean_samples=clean_samples,
        noisy_samples=samples,
        samples=samples,
        sample_noise=sample_noise,
        noise_std=noise_std,
        true_derivative_at_samples=true_derivative_samples,
        finite_difference=finite_difference,
        dct_coefficients=coefficients,
        dense_time=dense_time,
        true_dense=true_dense,
    )
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    per_dim_metrics.to_csv(output_dir / "per_dimension_metrics.csv", index=False)
    method_metrics.to_csv(output_dir / "derivative_method_metrics.csv", index=False)
    per_dim_method_metrics.to_csv(output_dir / "derivative_method_per_dimension_metrics.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(
        output_dir / "report.md",
        args,
        metrics,
        method_metrics,
        selected_m,
        comparison_m,
        finite_difference_baseline,
    )

    print(f"output_dir={output_dir}")
    print(f"noise_std_fraction={args.noise_std_fraction} noise_std={noise_std.tolist()}")
    print(f"finite_difference_vs_true={finite_difference_baseline}")
    print(metrics.loc[metrics["m"].isin(selected_m)].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
