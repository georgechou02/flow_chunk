#!/usr/bin/env python

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import (
    FlowMatchingObjective,
    _bspline_basis_matrix,
    _bspline_design_and_derivative,
    _parameter_averaged_bspline_knots,
)
from lerobot.utils.constants import ACTION


def _make_objective(
    horizon: int = 8,
    action_dim: int = 2,
    degree: int = 2,
    coefficient_count: int = 5,
    **overrides,
) -> FlowMatchingObjective:
    values = {
        "interpolation_mode": "bspline",
        "bspline_degree": degree,
        "bspline_coe_num": coefficient_count,
        "dct_coe_num": 0,
        "gripper_first": True,
        "image_only_condition_jvp": False,
        "lambda_flow_k": 1.0,
        "phy_loss_weight": 0.0,
        "pre_train_steps": 0,
        "sample_frequency": 10.0,
        "sigma_min": 0.0,
        "action_jvp_grad_scale": 1.0,
        "timestep_sampling_strategy": "uniform",
    }
    values.update(overrides)
    config = SimpleNamespace(**values)
    return FlowMatchingObjective(config, action_dim=action_dim, horizon=horizon)


class _TinyFlowModel(nn.Module):
    def __init__(self, action_dim: int, conditioning_dim: int):
        super().__init__()
        self.action_projection = nn.Linear(action_dim, action_dim, bias=False)
        self.conditioning_projection = nn.Linear(conditioning_dim, action_dim, bias=False)

    def forward(self, actions: Tensor, timesteps: Tensor, conditioning_vec: Tensor) -> Tensor:
        return (
            self.action_projection(actions)
            + self.conditioning_projection(conditioning_vec)[:, None]
            + timesteps[:, None, None]
        )


class _ScaledActionFlowModel(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, actions: Tensor, timesteps: Tensor, conditioning_vec: Tensor) -> Tensor:
        del timesteps, conditioning_vec
        return self.scale * actions


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"interpolation_mode": "unknown"}, "interpolation_mode"),
        ({"interpolation_mode": "bspline", "bspline_degree": 1, "bspline_coe_num": 3}, "bspline"),
        ({"interpolation_mode": "bspline", "bspline_degree": 3, "bspline_coe_num": 3}, "bspline"),
        ({"interpolation_mode": "bspline", "bspline_degree": 2, "bspline_coe_num": 9}, "bspline"),
        ({"interpolation_mode": "bspline", "bspline_degree": 2.5, "bspline_coe_num": 5}, "integers"),
    ],
)
def test_bspline_config_validation_rejects_invalid_values(overrides: dict, message: str):
    with pytest.raises(ValueError, match=message):
        MultiTaskDiTConfig(horizon=8, **overrides)


def test_bspline_config_accepts_minimum_coefficient_count_and_normalizes_mode():
    config = MultiTaskDiTConfig(
        horizon=8,
        interpolation_mode="BSPLINE",
        bspline_degree=2,
        bspline_coe_num=3,
    )

    assert config.interpolation_mode == "bspline"
    assert config.bspline_degree == 2
    assert config.bspline_coe_num == 3


def test_default_dct_mode_does_not_require_bspline_parameters():
    config = MultiTaskDiTConfig(horizon=4)

    assert config.interpolation_mode == "dct"
    assert config.bspline_degree == 0
    assert config.bspline_coe_num == 0


@pytest.mark.parametrize(
    ("mode", "dct_coe_num", "expected_indices"),
    [
        ("dct", 0, [-2, -1, 0, 1, 2, 3]),
        ("dct", 4, [-1, 0, 1, 2]),
        ("bspline", 0, [-1, 0, 1, 2]),
    ],
)
def test_action_delta_indices_preserve_dct_k0_and_use_h_frames_for_bspline(
    mode: str,
    dct_coe_num: int,
    expected_indices: list[int],
):
    config = MultiTaskDiTConfig(
        horizon=4,
        n_obs_steps=2,
        objective="flow_matching",
        lambda_flow_k=1.0,
        interpolation_mode=mode,
        dct_coe_num=dct_coe_num,
        bspline_degree=2 if mode == "bspline" else 0,
        bspline_coe_num=3 if mode == "bspline" else 0,
    )

    assert config.action_delta_indices == expected_indices


@pytest.mark.parametrize("degree", [2, 3, 4])
def test_parameter_averaged_basis_supports_minimum_m_and_partitions_unity(degree: int):
    coefficient_count = degree + 1
    knots = _parameter_averaged_bspline_knots(degree, coefficient_count)
    query = torch.linspace(0.0, 1.0, 31, dtype=torch.float64)
    basis = _bspline_basis_matrix(query, knots, degree)

    assert basis.shape == (len(query), coefficient_count)
    torch.testing.assert_close(basis.sum(dim=1), torch.ones_like(query), atol=2e-14, rtol=0)
    torch.testing.assert_close(basis[0], torch.eye(coefficient_count, dtype=torch.float64)[0])
    torch.testing.assert_close(basis[-1], torch.eye(coefficient_count, dtype=torch.float64)[-1])


def test_bspline_full_coefficient_fit_is_lossless_at_samples():
    horizon = 12
    design, _ = _bspline_design_and_derivative(horizon, degree=2, coefficient_count=horizon)
    actions = torch.randn(horizon, 4, dtype=torch.float64)
    coefficients = torch.linalg.pinv(design) @ actions

    torch.testing.assert_close(design @ coefficients, actions, atol=2e-12, rtol=2e-12)


def test_bspline_objective_rejects_numerically_rank_deficient_fit():
    with pytest.raises(ValueError, match="numerically rank deficient"):
        _make_objective(horizon=48, degree=47, coefficient_count=48)


def test_bspline_analytic_derivative_matches_quadratic_physical_time_derivative():
    horizon = 11
    fps = torch.tensor([10.0, 25.0], dtype=torch.float64).reshape(2, 1, 1)
    u = torch.linspace(0.0, 1.0, horizon, dtype=torch.float64)
    actions = torch.stack((u.square(), 3.0 * u - 2.0), dim=-1)
    actions = actions.unsqueeze(0).expand(2, -1, -1).clone()
    objective = _make_objective(
        horizon=horizon,
        action_dim=2,
        degree=2,
        coefficient_count=6,
    )

    expected_du = torch.stack((2.0 * u, torch.full_like(u, 3.0)), dim=-1)
    expected = expected_du.unsqueeze(0) * fps / (horizon - 1)
    actual = objective._action_dot(actions, fps)

    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_bspline_derivative_preserves_dtype_device_and_is_finite(dtype: torch.dtype):
    objective = _make_objective(horizon=8, action_dim=3, degree=2, coefficient_count=5)
    actions = torch.randn(2, 8, 3, dtype=dtype)

    actual = objective._action_dot(actions, torch.tensor(30.0, dtype=dtype))

    assert actual.dtype == dtype
    assert actual.device == actions.device
    assert torch.isfinite(actual).all()


def test_bspline_derivative_requires_exactly_h_action_frames():
    objective = _make_objective(horizon=8)

    with pytest.raises(ValueError, match="B-spline.*exactly horizon"):
        objective._action_dot(torch.randn(1, 10, 2), torch.tensor(10.0))


def test_bspline_compute_loss_rejects_extra_neighbor_frames_during_kinematic_warmup():
    horizon = 8
    objective = _make_objective(horizon=horizon)
    objective.config.pre_train_steps = 100

    with pytest.raises(ValueError, match="Flow-matching actions must have shape"):
        objective.compute_loss(
            _TinyFlowModel(action_dim=2, conditioning_dim=4),
            {ACTION: torch.randn(1, horizon + 2, 2)},
            torch.randn(1, 4),
            train_step=0,
        )


def test_bspline_physical_loss_matches_value_level_formula_and_total(monkeypatch):
    horizon = 7
    action_dim = 2
    weight = 1.75
    objective = _make_objective(
        horizon=horizon,
        action_dim=action_dim,
        degree=2,
        coefficient_count=5,
        lambda_flow_k=0.0,
        phy_loss_weight=weight,
    )
    timesteps = torch.tensor([0.2, 0.6])
    monkeypatch.setattr(
        objective,
        "_sample_timesteps",
        lambda batch_size, device: timesteps.to(device),
    )
    monkeypatch.setattr(torch, "randn_like", lambda data: torch.zeros_like(data))
    u = torch.linspace(0.0, 1.0, horizon)
    actions = torch.stack((u.square(), 3.0 * u - 1.0), dim=-1)
    actions = actions.unsqueeze(0).repeat(2, 1, 1)
    fps = torch.tensor([10.0, 25.0])
    model = _ScaledActionFlowModel(scale=0.4)

    loss, metrics = objective.compute_loss(
        model,
        {ACTION: actions, "sample_frequency": fps},
        conditioning_vec=torch.zeros(2, 1),
    )

    t_expanded = timesteps.view(-1, 1, 1)
    predicted_velocity = model.scale * t_expanded * actions
    target_velocity = actions
    flow_error = predicted_velocity - target_velocity
    g_flow_error = objective._action_dot(flow_error, fps.view(-1, 1, 1))
    physical_residual = t_expanded * (1 - t_expanded) * g_flow_error
    expected_physical_loss = physical_residual.square().mean()
    expected_total = flow_error.square().mean() + weight * expected_physical_loss

    torch.testing.assert_close(loss, expected_total)
    assert metrics["physical_loss"] == pytest.approx(expected_physical_loss.item())
    assert metrics["weighted_physical_loss"] == pytest.approx((weight * expected_physical_loss).item())
    loss.backward()
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad)


def test_bspline_physical_loss_excludes_discrete_gripper(monkeypatch):
    horizon = 5
    objective = _make_objective(
        horizon=horizon,
        action_dim=3,
        degree=2,
        coefficient_count=4,
        gripper_first=False,
        lambda_flow_k=0.0,
        phy_loss_weight=1.0,
    )
    monkeypatch.setattr(
        objective,
        "_sample_timesteps",
        lambda batch_size, device: torch.full((batch_size,), 0.5, device=device),
    )
    monkeypatch.setattr(torch, "randn_like", lambda data: torch.zeros_like(data))
    actions = torch.zeros(1, horizon, 3)
    actions[..., -1] = torch.tensor([-1.0, -1.0, 1.0, 1.0, -1.0])
    model = _ScaledActionFlowModel(scale=0.0)

    loss, metrics = objective.compute_loss(
        model,
        {ACTION: actions},
        conditioning_vec=torch.zeros(1, 1),
    )

    expected_flow_loss = actions.square().mean()
    torch.testing.assert_close(loss, expected_flow_loss)
    assert metrics["physical_loss"] == 0.0
    assert metrics["weighted_physical_loss"] == 0.0


def test_bspline_physical_loss_excludes_entire_padded_chunk(monkeypatch):
    horizon = 5
    weight = 0.75
    objective = _make_objective(
        horizon=horizon,
        action_dim=3,
        degree=2,
        coefficient_count=4,
        gripper_first=False,
        lambda_flow_k=0.0,
        phy_loss_weight=weight,
    )
    timesteps = torch.full((2,), 0.5)
    monkeypatch.setattr(
        objective,
        "_sample_timesteps",
        lambda batch_size, device: timesteps.to(device),
    )
    monkeypatch.setattr(torch, "randn_like", lambda data: torch.zeros_like(data))
    sample_index = torch.arange(horizon, dtype=torch.float32)
    valid_actions = torch.stack(
        (sample_index.square(), 2.0 * sample_index, torch.zeros_like(sample_index)),
        dim=-1,
    )
    padded_actions = 100.0 * valid_actions
    actions = torch.stack((valid_actions, padded_actions))
    action_is_pad = torch.tensor([[False, False, False, False, False], [False, False, True, False, False]])
    model = _ScaledActionFlowModel(scale=0.0)

    loss, metrics = objective.compute_loss(
        model,
        {ACTION: actions, "action_is_pad": action_is_pad},
        conditioning_vec=torch.zeros(2, 1),
    )

    flow_error = -actions
    g_flow_error = objective._action_dot(flow_error, torch.tensor(10.0))
    physical_residual = timesteps.view(-1, 1, 1) * (1 - timesteps.view(-1, 1, 1)) * g_flow_error
    expected_physical_loss = physical_residual[:1, :, :-1].square().mean()
    expected_total = flow_error.square().mean() + weight * expected_physical_loss

    torch.testing.assert_close(loss, expected_total)
    assert metrics["physical_loss"] == pytest.approx(expected_physical_loss.item())
    assert metrics["weighted_physical_loss"] == pytest.approx((weight * expected_physical_loss).item())


def test_bspline_padding_invalidates_global_fit():
    objective = _make_objective(horizon=5, coefficient_count=4)
    data = torch.zeros(2, 5, 2)
    conditioning_steps = torch.zeros(2, 2, 3)
    batch = {
        "action_is_pad": torch.tensor(
            [[False, False, False, False, False], [False, False, True, False, False]]
        )
    }

    valid = objective._kinematic_valid_mask(batch, data, conditioning_steps)

    assert torch.equal(
        valid,
        torch.tensor([[True, True, True, True, True], [False, False, False, False, False]]),
    )


def test_bspline_kinematic_branch_supports_forward_and_backward():
    torch.manual_seed(0)
    horizon = 6
    action_dim = 2
    conditioning_dim = 4
    batch_size = 2
    objective = _make_objective(
        horizon=horizon,
        action_dim=action_dim,
        degree=2,
        coefficient_count=4,
    )
    model = _TinyFlowModel(action_dim, conditioning_dim)
    conditioning_steps = torch.randn(batch_size, 2, conditioning_dim // 2)
    derivative_stencil = torch.cat(
        (conditioning_steps[:, :1] - 0.1, conditioning_steps),
        dim=1,
    )

    loss, metrics = objective.compute_loss(
        model,
        {ACTION: torch.randn(batch_size, horizon, action_dim)},
        conditioning_steps.flatten(start_dim=1),
        conditioning_steps=conditioning_steps,
        derivative_conditioning_steps=derivative_stencil,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["kinematic_valid_ratio"] == 1.0
    assert metrics["physical_loss"] > 0
    assert metrics["weighted_physical_loss"] == 0.0
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
