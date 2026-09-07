"""Numpy derivative stencils shared by Multi-Task DiT's ``_action_dot``.

These match ``qc-main/utils/action_derivatives.py``: Savitzky-Golay, clamped
B-spline, and truncated Chebyshev-T. DCT and central differences stay inline
in ``FlowMatchingObjective`` so the existing DCT tests keep a single source.
"""

from __future__ import annotations

import functools

import numpy as np


def _freeze(matrix: np.ndarray) -> np.ndarray:
    matrix.flags.writeable = False
    return matrix


@functools.lru_cache(maxsize=64)
def savgol_derivative_matrix(num_samples: int, window: int, polyorder: int) -> np.ndarray:
    if polyorder < 1:
        raise ValueError(f"savgol polyorder must be >= 1 to differentiate, got {polyorder}")
    if window <= polyorder:
        raise ValueError(f"savgol window must exceed polyorder, got {window} <= {polyorder}")
    if window > num_samples:
        raise ValueError(f"savgol window {window} exceeds chunk length {num_samples}")

    powers = np.arange(polyorder + 1)
    half = window // 2
    matrix = np.zeros((num_samples, num_samples), dtype=np.float64)
    for center in range(num_samples):
        start = min(max(center - half, 0), num_samples - window)
        offsets = np.arange(start, start + window) - center
        vandermonde = offsets[:, None].astype(np.float64) ** powers[None, :]
        matrix[center, start : start + window] = np.linalg.pinv(vandermonde)[1]
    return _freeze(matrix)


def _parameter_averaged_bspline_knots(degree: int, coefficient_count: int) -> np.ndarray:
    """Clamped knots by averaging an M-point uniform pseudo-grid (same as DiT v2).

    With ``u_i = i / (M - 1)``, interior knots are
    ``xi[p+j] = mean(u_j, ..., u_{j+p-1})`` for ``j = 1, ..., M-p-1``.
    Uniform interior knots are more ill-conditioned on uniform samples.
    """
    if degree < 1 or coefficient_count <= degree:
        raise ValueError(
            "B-spline knots require degree >= 1 and coefficient_count > degree, "
            f"got p={degree}, M={coefficient_count}"
        )
    pseudo_grid = np.linspace(0.0, 1.0, coefficient_count, dtype=np.float64)
    internal = np.array(
        [pseudo_grid[index : index + degree].mean() for index in range(1, coefficient_count - degree)],
        dtype=np.float64,
    )
    return np.concatenate(
        [np.zeros(degree + 1, dtype=np.float64), internal, np.ones(degree + 1, dtype=np.float64)]
    )


def _cox_de_boor(knots: np.ndarray, degree: int, sites: np.ndarray) -> np.ndarray:
    """Cox-de Boor evaluation, including the closed right endpoint at t=1."""
    n_knots = len(knots)
    basis = ((sites[:, None] >= knots[:-1]) & (sites[:, None] < knots[1:])).astype(np.float64)
    endpoint = np.isclose(sites, knots[-1])
    if np.any(endpoint):
        positive_final = np.flatnonzero((knots[:-1] < knots[-1]) & (knots[1:] == knots[-1]))
        if positive_final.size == 0:
            raise ValueError("B-spline knot vector has no positive-width final span")
        endpoint_basis = np.zeros_like(basis)
        endpoint_basis[:, positive_final[-1]] = 1.0
        basis[endpoint] = endpoint_basis[endpoint]
    for order in range(1, degree + 1):
        num_basis = n_knots - order - 1
        left_span = knots[order : order + num_basis] - knots[:num_basis]
        right_span = knots[order + 1 : order + num_basis + 1] - knots[1 : num_basis + 1]
        left_weight = np.divide(
            sites[:, None] - knots[:num_basis],
            left_span,
            out=np.zeros((sites.shape[0], num_basis), dtype=np.float64),
            where=left_span != 0,
        )
        right_weight = np.divide(
            knots[order + 1 : order + num_basis + 1] - sites[:, None],
            right_span,
            out=np.zeros((sites.shape[0], num_basis), dtype=np.float64),
            where=right_span != 0,
        )
        basis = left_weight * basis[:, :num_basis] * (left_span != 0) + right_weight * basis[
            :, 1 : num_basis + 1
        ] * (right_span != 0)
    return basis


@functools.lru_cache(maxsize=64)
def bspline_derivative_matrix(num_samples: int, num_control_points: int, degree: int) -> np.ndarray:
    if degree < 2:
        raise ValueError(f"bspline degree must be >= 2, got {degree}")
    if not degree < num_control_points <= num_samples:
        raise ValueError(
            "bspline requires 2 <= degree < num_control_points <= chunk length, "
            f"got p={degree}, M={num_control_points}, H={num_samples}"
        )
    if num_samples < 2:
        raise ValueError(f"bspline derivative needs at least 2 samples, got {num_samples}")

    knots = _parameter_averaged_bspline_knots(degree, num_control_points)
    sites = np.linspace(0.0, 1.0, num_samples, dtype=np.float64)
    basis = _cox_de_boor(knots, degree, sites)[:, :num_control_points]
    lower = _cox_de_boor(knots, degree - 1, sites)
    derivative = np.zeros_like(basis)
    for i in range(num_control_points):
        left_span = knots[i + degree] - knots[i]
        right_span = knots[i + degree + 1] - knots[i + 1]
        if left_span > 0:
            derivative[:, i] += degree / left_span * lower[:, i]
        if right_span > 0:
            derivative[:, i] -= degree / right_span * lower[:, i + 1]

    pinv_rtol = 1e-13
    singular_values = np.linalg.svd(basis, compute_uv=False)
    effective_rank = int(np.count_nonzero(singular_values > singular_values[0] * pinv_rtol))
    if effective_rank != num_control_points:
        raise ValueError(
            "B-spline collocation matrix is rank deficient at the configured cutoff "
            f"for p={degree}, M={num_control_points}, H={num_samples}"
        )
    analysis = np.linalg.pinv(basis, rcond=pinv_rtol)
    recovery = analysis @ basis
    if np.max(np.abs(recovery - np.eye(num_control_points))) > 1e-8:
        raise ValueError(
            "B-spline collocation matrix is too ill-conditioned for coefficient recovery "
            f"for p={degree}, M={num_control_points}, H={num_samples}"
        )
    matrix = derivative @ analysis / (num_samples - 1)
    return _freeze(matrix)


@functools.lru_cache(maxsize=64)
def chebyshev_derivative_matrix(num_samples: int, num_modes: int) -> np.ndarray:
    if num_samples < 2:
        raise ValueError(f"chebyshev derivative needs at least 2 samples, got {num_samples}")
    if not 1 <= num_modes <= num_samples:
        raise ValueError(f"chebyshev num_modes must be in [1, {num_samples}], got {num_modes}")

    x = np.linspace(-1.0, 1.0, num_samples, dtype=np.float64)
    basis = np.empty((num_samples, num_modes), dtype=np.float64)
    basis[:, 0] = 1.0
    if num_modes > 1:
        basis[:, 1] = x
    for degree in range(2, num_modes):
        basis[:, degree] = 2.0 * x * basis[:, degree - 1] - basis[:, degree - 2]

    second_kind = np.empty((num_samples, max(num_modes - 1, 1)), dtype=np.float64)
    second_kind[:, 0] = 1.0
    if num_modes > 2:
        second_kind[:, 1] = 2.0 * x
        for degree in range(2, num_modes - 1):
            second_kind[:, degree] = (
                2.0 * x * second_kind[:, degree - 1] - second_kind[:, degree - 2]
            )

    basis_x = np.zeros_like(basis)
    for degree in range(1, num_modes):
        basis_x[:, degree] = degree * second_kind[:, degree - 1]

    dx_dstep = 2.0 / (num_samples - 1)
    return _freeze((basis_x * dx_dstep) @ np.linalg.pinv(basis))
