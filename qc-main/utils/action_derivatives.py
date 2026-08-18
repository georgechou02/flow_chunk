"""Temporal derivatives of action chunks.

Two estimators, matching the kinematic targets used by the multi-task DiT flow
objective:

- ``dct_coe_num == 0``: forward difference ``(a_{t+1} - a_t) * fps``
- ``dct_coe_num > 0``: truncated DCT-II spectral derivative, then scaled by ``fps``

QC uses forward differences (the dataset already stores ``next_actions``) instead
of the H+2 central-difference stencil used by DiT when ``dct_coe_num == 0``.
"""

from __future__ import annotations

import jax.numpy as jnp


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

    This is a JAX port of ``FlowMatchingObjective._action_dot`` when
    ``dct_coe_num > 0``. Modes beyond ``num_modes`` are dropped before
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


def action_dot(actions, next_actions=None, fps=1.0, dct_coe_num=0):
    """Dispatch between forward-difference and DCT action derivatives.

    Args:
        actions: ``(B, H, A)`` action chunk.
        next_actions: Required when ``dct_coe_num == 0``.
        fps: Control frequency in Hz.
        dct_coe_num: ``0`` selects forward differences; ``> 0`` selects DCT
            with that many retained modes.
    """
    if dct_coe_num == 0:
        if next_actions is None:
            raise ValueError("Forward-difference action derivatives require next_actions.")
        return action_dot_forward(actions, next_actions, fps=fps)
    return action_dot_dct(actions, fps=fps, num_modes=dct_coe_num)
