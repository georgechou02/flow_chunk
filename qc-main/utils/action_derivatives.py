"""Temporal derivatives of action chunks.

Estimators for the kinematic target ``ȧ``. They all return ``da/dstep * fps``
and are interchangeable drop-ins:

- ``forward``: ``(a_{t+1} - a_t) * fps``. No smoothing; needs ``next_actions``.
- ``dct``: truncated DCT-II spectral derivative on a global cosine basis.
- ``savgol``: Savitzky-Golay (local polynomial least squares). ``window`` is
  the smoothing scale in samples.
- ``bspline``: least-squares clamped B-spline, differentiated analytically.
  Local support; no Neumann (vanishing-derivative) condition at the edges.
- ``chebyshev``: truncated Chebyshev-T series on the uniform grid, then
  differentiated. Like DCT this is a global polynomial basis, but the
  endpoint derivative is not structurally forced toward zero.

Effective smoothing scale in samples, for a chunk of length ``H``:

- ``forward``: 1
- ``savgol`` with window ``w``: ``w``
- ``dct`` / ``chebyshev`` keeping ``M`` modes: ``~H / M``
- ``bspline`` with ``C`` control points: ``~H / C``

QC uses forward differences (the dataset already stores ``next_actions``)
instead of the H+2 central-difference stencil used by DiT when
``dct_coe_num == 0`` and ``derivative_kind == "auto"``.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np

DERIVATIVE_KINDS = ("auto", "forward", "dct", "savgol", "bspline", "chebyshev")


def resolve_kind(kind, dct_coe_num=0):
    """Map ``kind`` (possibly ``"auto"``) onto a concrete estimator name."""
    if kind not in DERIVATIVE_KINDS:
        raise ValueError(f"kind must be one of {DERIVATIVE_KINDS}, got {kind!r}")
    if kind == "auto":
        return "forward" if dct_coe_num == 0 else "dct"
    return kind


def mixes_frames(kind, dct_coe_num=0):
    """Whether the estimator combines several frames of the chunk.

    Callers drop a sample that crosses an episode boundary entirely, instead of
    masking per step, when this is true.
    """
    return resolve_kind(kind, dct_coe_num) != "forward"


def broadcast_fps(fps, reference):
    """Broadcast a scalar or per-sample fps onto ``reference``'s leading dims."""
    fps_array = jnp.asarray(fps, dtype=reference.dtype)
    if fps_array.ndim == 0:
        return fps_array
    if fps_array.shape[0] != reference.shape[0]:
        raise ValueError(
            f"fps batch dimension {fps_array.shape[0]} does not match "
            f"batch size {reference.shape[0]}"
        )
    return fps_array.reshape((reference.shape[0],) + (1,) * (reference.ndim - 1))


def action_dot_forward(actions, next_actions, fps=1.0):
    """Forward-difference action derivative.

    Args:
        actions: ``(B, H, A)`` action chunk.
        next_actions: ``(B, H, A)`` one-step-shifted actions (same layout as
            ``Dataset.sample_sequence``'s ``next_actions``).
        fps: Control frequency in Hz. The derivative is ``delta * fps``.

    Returns:
        ``(B, H, A)`` action time-derivative.
    """
    if actions.shape != next_actions.shape:
        raise ValueError(
            "Forward-difference actions and next_actions must share shape, "
            f"got {tuple(actions.shape)} vs {tuple(next_actions.shape)}"
        )
    return (next_actions - actions) * broadcast_fps(fps, actions)


def action_dot_dct(actions, fps=1.0, num_modes=None):
    """DCT-II spectral derivative of an action chunk.

    This is a JAX port of ``FlowMatchingObjective._action_dot`` when the
    DCT estimator is selected. Modes beyond ``num_modes`` are dropped before
    differentiating the cosine basis.

    Args:
        actions: ``(B, H, A)`` action chunk.
        fps: Control frequency in Hz.
        num_modes: Number of retained DCT coefficients. ``None`` keeps all
            ``H`` modes.

    Returns:
        ``(B, H, A)`` action time-derivative.
    """
    if actions.ndim != 3:
        raise ValueError(f"DCT action derivative expects (B, H, A), got {tuple(actions.shape)}")

    num_samples = actions.shape[1]
    if num_modes is None:
        num_modes = num_samples
    if not 1 <= num_modes <= num_samples:
        raise ValueError(f"num_modes must be in [1, {num_samples}], got {num_modes}")

    dtype = actions.dtype
    modes = jnp.arange(num_modes, dtype=dtype)
    sample_points = jnp.arange(num_samples, dtype=dtype) + jnp.asarray(0.5, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    two = jnp.asarray(2.0, dtype=dtype)
    alpha = jnp.where(
        modes == 0,
        jnp.sqrt(one / num_samples),
        jnp.sqrt(two / num_samples),
    )
    phase = jnp.pi * modes[:, None] * sample_points[None, :] / num_samples
    basis = alpha[:, None] * jnp.cos(phase)
    coefficients = jnp.einsum("bhd,kh->bkd", actions, basis)
    derivative_basis = (
        -alpha[:, None] * (jnp.pi * modes[:, None] / num_samples) * jnp.sin(phase)
    )
    action_dot = jnp.einsum("bkd,kh->bhd", coefficients, derivative_basis)
    return action_dot * broadcast_fps(fps, actions)


def _freeze(matrix):
    matrix.flags.writeable = False
    return matrix


@functools.lru_cache(maxsize=64)
def savgol_derivative_matrix(num_samples, window, polyorder):
    """``(H, H)`` matrix mapping a chunk onto its Savitzky-Golay derivative."""
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
        # Shift the window at the edges rather than pad, so every fit sees
        # ``window`` real samples and the expansion point stays at ``center``.
        start = min(max(center - half, 0), num_samples - window)
        offsets = np.arange(start, start + window) - center
        vandermonde = offsets[:, None].astype(np.float64) ** powers[None, :]
        # Row 1 of the pseudo-inverse is the linear coefficient = derivative at 0.
        matrix[center, start : start + window] = np.linalg.pinv(vandermonde)[1]
    return _freeze(matrix)


def _parameter_averaged_bspline_knots(degree, coefficient_count):
    """Clamped knots by averaging an M-point uniform pseudo-grid (same as DiT v2)."""
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


def _cox_de_boor(knots, degree, sites):
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
def bspline_derivative_matrix(num_samples, num_control_points, degree):
    """``(H, H)`` matrix mapping a chunk onto its B-spline derivative."""
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
def chebyshev_derivative_matrix(num_samples, num_modes):
    """``(H, H)`` matrix mapping a chunk onto its truncated Chebyshev derivative."""
    if num_samples < 2:
        raise ValueError(f"chebyshev derivative needs at least 2 samples, got {num_samples}")
    if not 1 <= num_modes <= num_samples:
        raise ValueError(f"chebyshev num_modes must be in [1, {num_samples}], got {num_modes}")

    # Uniform samples mapped onto [-1, 1]. T_k'(x) = k U_{k-1}(x).
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
            second_kind[:, degree] = 2.0 * x * second_kind[:, degree - 1] - second_kind[:, degree - 2]

    basis_x = np.zeros_like(basis)
    for degree in range(1, num_modes):
        basis_x[:, degree] = degree * second_kind[:, degree - 1]

    dx_dstep = 2.0 / (num_samples - 1)
    matrix = (basis_x * dx_dstep) @ np.linalg.pinv(basis)
    return _freeze(matrix)


def _apply_stencil(actions, matrix, fps):
    stencil = jnp.asarray(matrix, dtype=actions.dtype)
    action_dot = jnp.einsum("ht,btd->bhd", stencil, actions)
    return action_dot * broadcast_fps(fps, actions)


def action_dot_savgol(actions, fps=1.0, window=5, polyorder=2):
    """Savitzky-Golay derivative: local polynomial least squares, then d/dt.

    ``window == 3`` with ``polyorder == 1`` reproduces the interior central
    difference, so ``window`` is a direct handle on the smoothing scale.

    Args:
        actions: ``(B, H, A)`` action chunk.
        fps: Control frequency in Hz.
        window: Samples per local fit. Odd values keep the fit centred away
            from the chunk edges.
        polyorder: Degree of the local polynomial; must be ``< window``.

    Returns:
        ``(B, H, A)`` action time-derivative.
    """
    if actions.ndim != 3:
        raise ValueError(f"Savgol action derivative expects (B, H, A), got {tuple(actions.shape)}")
    matrix = savgol_derivative_matrix(int(actions.shape[1]), int(window), int(polyorder))
    return _apply_stencil(actions, matrix, fps)


def action_dot_bspline(actions, fps=1.0, num_control_points=None, degree=2):
    """B-spline derivative: least-squares spline fit, differentiated exactly.

    The basis has local support and a clamped knot vector, so the derivative at
    the chunk edges is unconstrained -- unlike the cosine basis, whose Neumann
    boundary condition drives the endpoint derivative toward zero.

    Args:
        actions: ``(B, H, A)`` action chunk.
        fps: Control frequency in Hz.
        num_control_points: Spline coefficients to fit. Fewer means smoother.
            ``None`` uses the chunk length ``H`` (full fit).
        degree: Spline degree; 2 matches the DiT training script.

    Returns:
        ``(B, H, A)`` action time-derivative.
    """
    if actions.ndim != 3:
        raise ValueError(
            f"B-spline action derivative expects (B, H, A), got {tuple(actions.shape)}"
        )
    num_samples = int(actions.shape[1])
    if num_control_points is None:
        num_control_points = num_samples
    matrix = bspline_derivative_matrix(num_samples, int(num_control_points), int(degree))
    return _apply_stencil(actions, matrix, fps)


def action_dot_chebyshev(actions, fps=1.0, num_modes=None):
    """Truncated Chebyshev-T derivative on the uniform action grid.

    Least-squares fit of ``T_0 ... T_{M-1}`` on ``[-1, 1]``, then analytic
    differentiation. Truncating ``M`` sets the smoothing scale, analogous to
    ``dct_coe_num``.

    Args:
        actions: ``(B, H, A)`` action chunk.
        fps: Control frequency in Hz.
        num_modes: Retained Chebyshev degrees. ``None`` keeps all ``H`` modes.

    Returns:
        ``(B, H, A)`` action time-derivative.
    """
    if actions.ndim != 3:
        raise ValueError(
            f"Chebyshev action derivative expects (B, H, A), got {tuple(actions.shape)}"
        )
    num_samples = int(actions.shape[1])
    if num_modes is None:
        num_modes = num_samples
    matrix = chebyshev_derivative_matrix(num_samples, int(num_modes))
    return _apply_stencil(actions, matrix, fps)


def action_dot(
    actions,
    next_actions=None,
    fps=1.0,
    dct_coe_num=0,
    kind="auto",
    savgol_window=5,
    savgol_polyorder=2,
    bspline_num_control_points=None,
    bspline_degree=2,
    chebyshev_num_modes=None,
):
    """Dispatch to one of the action-derivative estimators.

    Args:
        actions: ``(B, H, A)`` action chunk.
        next_actions: Required by the forward-difference estimator.
        fps: Control frequency in Hz.
        dct_coe_num: Retained DCT modes. Also selects the estimator when
            ``kind == "auto"``: ``0`` gives forward differences, ``> 0`` DCT.
        kind: One of ``DERIVATIVE_KINDS``.
        savgol_window: Window for ``kind == "savgol"``.
        savgol_polyorder: Polynomial order for ``kind == "savgol"``.
        bspline_num_control_points: Control points for ``kind == "bspline"``.
        bspline_degree: Spline degree for ``kind == "bspline"``.
        chebyshev_num_modes: Retained modes for ``kind == "chebyshev"``.
            ``None`` or ``0`` keeps all ``H`` modes.
    """
    resolved = resolve_kind(kind, dct_coe_num)
    if resolved == "forward":
        if next_actions is None:
            raise ValueError("Forward-difference action derivatives require next_actions.")
        return action_dot_forward(actions, next_actions, fps=fps)
    if resolved == "dct":
        num_modes = dct_coe_num if dct_coe_num > 0 else None
        return action_dot_dct(actions, fps=fps, num_modes=num_modes)
    if resolved == "savgol":
        return action_dot_savgol(
            actions, fps=fps, window=savgol_window, polyorder=savgol_polyorder
        )
    if resolved == "bspline":
        return action_dot_bspline(
            actions,
            fps=fps,
            num_control_points=bspline_num_control_points,
            degree=bspline_degree,
        )
    if chebyshev_num_modes == 0:
        chebyshev_num_modes = None
    return action_dot_chebyshev(actions, fps=fps, num_modes=chebyshev_num_modes)


def physical_action_dot(
    flow_error,
    fps=1.0,
    dct_coe_num=0,
    kind="auto",
    savgol_window=5,
    savgol_polyorder=2,
    bspline_num_control_points=None,
    bspline_degree=2,
    chebyshev_num_modes=None,
):
    """Differentiate an H-frame flow error for the value-level physical residual.

    DCT / Savitzky-Golay / B-spline / Chebyshev already operate on H frames, so
    they reuse ``action_dot``. Forward differences normally need ``next_actions``;
    a predicted flow error has no shifted neighbor, so use the same interior
    centered stencil as DiT with second-order one-sided closures at the two
    boundaries.
    """
    if flow_error.ndim != 3:
        raise ValueError(
            f"Physical action derivative expects (B, H, A), got {tuple(flow_error.shape)}"
        )
    resolved = resolve_kind(kind, dct_coe_num)
    if resolved != "forward":
        return action_dot(
            flow_error,
            fps=fps,
            dct_coe_num=dct_coe_num,
            kind=kind,
            savgol_window=savgol_window,
            savgol_polyorder=savgol_polyorder,
            bspline_num_control_points=bspline_num_control_points,
            bspline_degree=bspline_degree,
            chebyshev_num_modes=chebyshev_num_modes,
        )

    horizon = flow_error.shape[1]
    fps_b = broadcast_fps(fps, flow_error)
    if horizon == 1:
        return jnp.zeros_like(flow_error)
    if horizon == 2:
        slope = (flow_error[:, 1:2] - flow_error[:, :1]) * fps_b
        return jnp.repeat(slope, 2, axis=1)

    dtype = flow_error.dtype
    first = (
        jnp.asarray(-1.5, dtype=dtype) * flow_error[:, :1]
        + jnp.asarray(2.0, dtype=dtype) * flow_error[:, 1:2]
        - jnp.asarray(0.5, dtype=dtype) * flow_error[:, 2:3]
    ) * fps_b
    interior = (flow_error[:, 2:] - flow_error[:, :-2]) * (fps_b * jnp.asarray(0.5, dtype=dtype))
    last = (
        jnp.asarray(0.5, dtype=dtype) * flow_error[:, -3:-2]
        - jnp.asarray(2.0, dtype=dtype) * flow_error[:, -2:-1]
        + jnp.asarray(1.5, dtype=dtype) * flow_error[:, -1:]
    ) * fps_b
    return jnp.concatenate((first, interior, last), axis=1)
