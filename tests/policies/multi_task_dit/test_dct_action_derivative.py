#!/usr/bin/env python

# Copyright 2025 Bryson Jones and The HuggingFace Inc. team. All rights reserved.
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

import math
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import (
    FlowMatchingObjective,
    MultiTaskDiTPolicy,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def _make_objective(
    horizon: int = 7,
    action_dim: int = 3,
    **overrides,
) -> FlowMatchingObjective:
    do_mask_loss_for_padding = overrides.pop("do_mask_loss_for_padding", False)
    values = {
        "dct_coe_num": 0,
        "conditioning_derivative_mode": "reverse",
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
    return FlowMatchingObjective(
        SimpleNamespace(**values),
        action_dim=action_dim,
        horizon=horizon,
        do_mask_loss_for_padding=do_mask_loss_for_padding,
    )


def _single_mode_actions(
    horizon: int,
    mode: int,
    coefficients: Tensor,
    batch_scales: Tensor | None = None,
) -> Tensor:
    num_samples = horizon
    alpha = math.sqrt((1.0 if mode == 0 else 2.0) / num_samples)
    sample_points = torch.arange(
        num_samples,
        device=coefficients.device,
        dtype=coefficients.dtype,
    )
    basis = alpha * torch.cos(math.pi * mode * (sample_points + 0.5) / num_samples)
    actions = basis[None, :, None] * coefficients[None, None, :]
    if batch_scales is not None:
        actions = actions * batch_scales[:, None, None]
    return actions


@pytest.mark.parametrize("clean_action_log_freq", [-1, 1.5, True])
def test_clean_action_log_frequency_rejects_invalid_values(clean_action_log_freq):
    with pytest.raises(ValueError, match="clean_action_log_freq"):
        MultiTaskDiTConfig(clean_action_log_freq=clean_action_log_freq)


def test_clean_action_metrics_follow_cadence_and_restore_training_state(monkeypatch):
    config = MultiTaskDiTConfig(
        device="cpu",
        input_features={OBS_STATE: PolicyFeature(FeatureType.STATE, (3,))},
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (2,))},
        objective="flow_matching",
        single_task=True,
        vision_encoder_type="resnet",
        image_crop_shape=None,
        n_obs_steps=2,
        horizon=4,
        n_action_steps=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=1,
        timestep_embed_dim=4,
        dropout=0.0,
        lambda_flow_k=0.0,
        dct_coe_num=4,
        num_integration_steps=1,
        clean_action_log_freq=2,
    )
    policy = MultiTaskDiTPolicy(config).train()
    policy.noise_predictor = TinyFlowModel(action_dim=2, conditioning_dim=6)
    batch = {
        OBS_STATE: torch.randn(2, 2, 3),
        ACTION: torch.randn(2, 4, 2),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
    }
    batch["action_is_pad"][0, 1] = True
    sampler_calls = 0

    def fake_conditional_sample(model, batch_size, conditioning_vec, noise=None):
        nonlocal sampler_calls
        sampler_calls += 1
        assert not policy.training
        assert batch_size == 2
        assert conditioning_vec.shape == (2, 6)
        assert noise is not None
        return batch[ACTION] + 3.0

    monkeypatch.setattr(policy.objective, "conditional_sample", fake_conditional_sample)

    policy.set_train_step(0)
    _, first_metrics = policy(batch)
    assert "clean_action_mse" not in first_metrics
    assert sampler_calls == 0

    policy.set_train_step(1)
    _, second_metrics = policy(batch)
    assert second_metrics["clean_action_mse"] == pytest.approx(9.0)
    assert second_metrics["clean_action_valid_ratio"] == pytest.approx(3 / 4)
    assert sampler_calls == 1
    assert policy.training
    assert policy.noise_predictor.training


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (15_999, 0.01),
        (16_000, 0.01),
        (17_000, 0.005 * (1 + math.sqrt(0.5))),
        (18_000, 0.005),
        (19_000, 0.005 * (1 - math.sqrt(0.5))),
        (20_000, 0.0),
    ],
)
def test_kinematic_weight_cosine_schedule(step: int, expected: float):
    objective = _make_objective(
        lambda_flow_k=0.01,
        lambda_flow_k_schedule="cosine",
        lambda_flow_k_schedule_start_step=16_000,
        lambda_flow_k_schedule_end_step=20_000,
        lambda_flow_k_schedule_final=0.0,
    )

    assert objective._effective_lambda_flow_k(step) == pytest.approx(expected)


def test_kinematic_weight_schedule_supports_hard_switch_without_changing_configured_weight():
    objective = _make_objective(
        lambda_flow_k=0.01,
        lambda_flow_k_schedule="cosine",
        lambda_flow_k_schedule_start_step=16_000,
        lambda_flow_k_schedule_end_step=16_000,
        lambda_flow_k_schedule_final=0.0,
    )

    assert objective.config.lambda_flow_k == 0.01
    assert objective._effective_lambda_flow_k(15_999) == 0.01
    assert objective._effective_lambda_flow_k(16_000) == 0.0


def _single_mode_derivative(
    horizon: int,
    mode: int,
    coefficients: Tensor,
    fps: Tensor,
    batch_scales: Tensor | None = None,
) -> Tensor:
    num_samples = horizon
    alpha = math.sqrt((1.0 if mode == 0 else 2.0) / num_samples)
    query_points = torch.arange(
        horizon,
        device=coefficients.device,
        dtype=coefficients.dtype,
    )
    basis_dot = (
        -alpha
        * (math.pi * mode / num_samples)
        * torch.sin(math.pi * mode * (query_points + 0.5) / num_samples)
    )
    derivative = basis_dot[None, :, None] * coefficients[None, None, :]
    if batch_scales is not None:
        derivative = derivative * batch_scales[:, None, None]
    return derivative * fps


def _reverse_conditioning_stencil(conditioning_steps: Tensor) -> Tensor:
    """Add the preceding observation required to differentiate every policy observation."""
    preceding_step = conditioning_steps[:, :1] - 1
    return torch.cat((preceding_step, conditioning_steps), dim=1)


class TinyFlowModel(nn.Module):
    def __init__(self, action_dim: int, conditioning_dim: int):
        super().__init__()
        self.action_projection = nn.Linear(action_dim, action_dim, bias=False)
        self.conditioning_projection = nn.Linear(conditioning_dim, action_dim, bias=False)
        self.time_scale = nn.Parameter(torch.randn(action_dim))

    def forward(self, actions: Tensor, timesteps: Tensor, conditioning_vec: Tensor) -> Tensor:
        return (
            self.action_projection(actions)
            + self.conditioning_projection(conditioning_vec)[:, None, :]
            + timesteps[:, None, None] * self.time_scale
        )


class RecordingFlowModel(TinyFlowModel):
    def __init__(self, action_dim: int, conditioning_dim: int):
        super().__init__(action_dim, conditioning_dim)
        self.last_actions: Tensor | None = None

    def forward(self, actions: Tensor, timesteps: Tensor, conditioning_vec: Tensor) -> Tensor:
        self.last_actions = actions.detach().clone()
        return super().forward(actions, timesteps, conditioning_vec)


class ScaledActionFlowModel(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, actions: Tensor, timesteps: Tensor, conditioning_vec: Tensor) -> Tensor:
        del timesteps, conditioning_vec
        return self.scale * actions


class DropoutFlowModel(TinyFlowModel):
    def __init__(self, action_dim: int, conditioning_dim: int, dropout: float):
        super().__init__(action_dim, conditioning_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, actions: Tensor, timesteps: Tensor, conditioning_vec: Tensor) -> Tensor:
        return self.dropout(super().forward(actions, timesteps, conditioning_vec))


@pytest.mark.parametrize("dct_coe_num", [-1, 5])
def test_dct_coe_num_validation_rejects_out_of_range_values(dct_coe_num: int):
    with pytest.raises(ValueError, match="dct_coe_num"):
        MultiTaskDiTConfig(horizon=4, dct_coe_num=dct_coe_num)


@pytest.mark.parametrize("dct_coe_num", [0, 4])
def test_dct_coe_num_validation_accepts_boundaries(dct_coe_num: int):
    config = MultiTaskDiTConfig(horizon=4, dct_coe_num=dct_coe_num)
    assert config.dct_coe_num == dct_coe_num


@pytest.mark.parametrize("phy_loss_weight", [-1.0, float("nan"), float("inf")])
def test_phy_loss_weight_validation_rejects_invalid_values(phy_loss_weight: float):
    with pytest.raises(ValueError, match="phy_loss_weight"):
        MultiTaskDiTConfig(phy_loss_weight=phy_loss_weight)


def test_positive_phy_loss_weight_requires_flow_matching_and_accepts_dct_k0():
    with pytest.raises(ValueError, match="flow_matching"):
        MultiTaskDiTConfig(phy_loss_weight=1.0, dct_coe_num=4)

    config = MultiTaskDiTConfig(
        objective="flow_matching",
        horizon=4,
        phy_loss_weight=1.0,
        dct_coe_num=0,
    )
    assert config.phy_loss_weight == 1.0


@pytest.mark.parametrize(
    ("objective", "lambda_flow_k", "dct_coe_num", "expected_indices"),
    [
        ("flow_matching", 1.0, 4, [-1, 0, 1, 2]),
        ("flow_matching", 1.0, 0, [-2, -1, 0, 1, 2, 3]),
        ("flow_matching", 0.0, 0, [-1, 0, 1, 2]),
        ("diffusion", 1.0, 0, [-1, 0, 1, 2]),
    ],
)
def test_action_delta_indices_use_branch_specific_frame_count(
    objective: str,
    lambda_flow_k: float,
    dct_coe_num: int,
    expected_indices: list[int],
):
    config = MultiTaskDiTConfig(
        horizon=4,
        n_obs_steps=2,
        objective=objective,
        lambda_flow_k=lambda_flow_k,
        dct_coe_num=dct_coe_num,
    )

    assert config.action_delta_indices == expected_indices


@pytest.mark.parametrize(
    ("mode", "expected_indices"),
    [
        ("reverse", [-2, -1, 0]),
        ("forward", [-1, 0, 1]),
        ("central", [-2, -1, 0, 1]),
    ],
)
def test_observation_delta_indices_match_conditioning_derivative_stencil(
    mode: str,
    expected_indices: list[int],
):
    config = MultiTaskDiTConfig(
        n_obs_steps=2,
        objective="flow_matching",
        lambda_flow_k=1.0,
        conditioning_derivative_mode=mode,
    )

    assert config.observation_delta_indices == expected_indices


def test_central_conditioning_derivative_differentiates_every_policy_observation():
    objective = _make_objective(conditioning_derivative_mode="central")
    fps = torch.tensor([10.0, 4.0]).reshape(2, 1, 1)
    derivative_stencil = torch.tensor(
        [
            [[0.0, 4.0], [1.0, 7.0], [4.0, 10.0], [9.0, 13.0]],
            [[2.0, -3.0], [4.0, 1.0], [8.0, 7.0], [14.0, 15.0]],
        ]
    )
    conditioning_steps = derivative_stencil[:, 1:3]
    expected = torch.cat(
        (
            (derivative_stencil[:, 2] - derivative_stencil[:, 0]) * fps[:, 0] * 0.5,
            (derivative_stencil[:, 3] - derivative_stencil[:, 1]) * fps[:, 0] * 0.5,
        ),
        dim=1,
    )

    actual = objective._conditioning_dot(conditioning_steps, fps, derivative_stencil)

    torch.testing.assert_close(actual, expected)


def test_zero_dct_coefficients_preserve_central_difference_exactly():
    horizon = 6
    objective = _make_objective(horizon=horizon, dct_coe_num=0)
    actions = torch.randn(3, horizon + 2, 4)
    fps = torch.tensor([7.0, 11.0, 19.0]).reshape(3, 1, 1)

    expected = (actions[:, 2 : horizon + 2] - actions[:, :horizon]) * (fps * 0.5)

    assert torch.equal(objective._action_dot(actions, fps), expected)


def test_dct_action_derivative_rejects_extra_action_frame():
    horizon = 6
    objective = _make_objective(horizon=horizon, dct_coe_num=horizon)
    actions = torch.randn(2, horizon + 1, 3, dtype=torch.float64)
    fps = torch.tensor(10.0, dtype=actions.dtype)

    with pytest.raises(ValueError, match=r"requires exactly horizon \(6\).+got 7"):
        objective._action_dot(actions, fps)


def test_central_difference_rejects_missing_neighbor_action_frame():
    horizon = 6
    objective = _make_objective(horizon=horizon, dct_coe_num=0)
    actions = torch.randn(2, horizon + 1, 3)

    with pytest.raises(ValueError, match=r"requires exactly horizon \+ 2 \(8\).+got 7"):
        objective._action_dot(actions, torch.tensor(10.0))


@pytest.mark.parametrize("num_modes", [1, 7])
def test_constant_sequence_has_zero_dct_derivative(num_modes: int):
    horizon = 7
    objective = _make_objective(horizon=horizon, dct_coe_num=num_modes)
    actions = torch.full((2, horizon, 3), 1.75, dtype=torch.float64)

    actual = objective._action_dot(actions, torch.tensor(13.0, dtype=actions.dtype))

    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=1e-12, rtol=0)


def test_single_dct_mode_matches_closed_form_derivative():
    horizon = 8
    mode = 3
    coefficients = torch.tensor([1.25, -0.75], dtype=torch.float64)
    batch_scales = torch.tensor([1.0, 0.4], dtype=torch.float64)
    fps = torch.tensor([5.0, 17.0], dtype=torch.float64).reshape(2, 1, 1)
    objective = _make_objective(
        horizon=horizon,
        action_dim=coefficients.numel(),
        dct_coe_num=mode + 1,
    )
    actions = _single_mode_actions(horizon, mode, coefficients, batch_scales)
    expected = _single_mode_derivative(
        horizon,
        mode,
        coefficients,
        fps,
        batch_scales,
    )

    actual = objective._action_dot(actions, fps)

    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_dct_high_frequency_truncation_uses_coefficient_count():
    horizon = 7
    retained_modes = horizon - 1
    omitted_mode = retained_modes
    coefficients = torch.tensor([0.8, -1.1], dtype=torch.float64)
    fps = torch.tensor(9.0, dtype=torch.float64)
    actions = _single_mode_actions(horizon, omitted_mode, coefficients)

    truncated = _make_objective(
        horizon=horizon,
        action_dim=coefficients.numel(),
        dct_coe_num=retained_modes,
    )._action_dot(actions, fps)
    included = _make_objective(
        horizon=horizon,
        action_dim=coefficients.numel(),
        dct_coe_num=retained_modes + 1,
    )._action_dot(actions, fps)
    expected = _single_mode_derivative(horizon, omitted_mode, coefficients, fps)

    torch.testing.assert_close(truncated, torch.zeros_like(truncated), atol=1e-12, rtol=0)
    torch.testing.assert_close(included, expected, atol=1e-12, rtol=1e-12)
    assert torch.count_nonzero(included) > 0


def test_scalar_and_per_batch_fps_broadcast_with_expected_shape():
    horizon = 5
    objective = _make_objective(horizon=horizon, action_dim=2, dct_coe_num=4, sample_frequency=3.0)
    actions = torch.randn(2, horizon, 2, dtype=torch.float64)
    unit_fps = torch.tensor(1.0, dtype=actions.dtype)
    unit_derivative = objective._action_dot(actions, unit_fps)

    scalar_fps = objective._sample_frequency({}, actions[:, :horizon])
    batch_fps = objective._sample_frequency(
        {"sample_frequency": torch.tensor([2.0, 5.0])},
        actions[:, :horizon],
    )

    assert scalar_fps.shape == torch.Size([])
    assert batch_fps.shape == (2, 1, 1)
    assert objective._action_dot(actions, scalar_fps).shape == (2, horizon, 2)
    torch.testing.assert_close(objective._action_dot(actions, scalar_fps), unit_derivative * 3.0)
    torch.testing.assert_close(
        objective._action_dot(actions, batch_fps),
        unit_derivative * batch_fps,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_dct_action_derivative_preserves_dtype_and_device(dtype: torch.dtype):
    horizon = 5
    objective = _make_objective(horizon=horizon, action_dim=2, dct_coe_num=4)
    actions = torch.randn(2, horizon, 2, dtype=dtype)
    fps = torch.tensor(10.0, dtype=dtype)

    actual = objective._action_dot(actions, fps)

    assert actual.shape == (2, horizon, 2)
    assert actual.dtype == dtype
    assert actual.device == actions.device
    assert torch.isfinite(actual).all()


def test_dct_action_derivative_rejects_short_sequences():
    horizon = 5
    objective = _make_objective(horizon=horizon, dct_coe_num=3)
    actions = torch.randn(2, horizon - 1, 3)

    with pytest.raises(ValueError, match=r"requires exactly horizon \(5\).+got 4"):
        objective._action_dot(actions, torch.tensor(10.0))


def test_dct_and_central_difference_use_branch_specific_padding_frames():
    horizon = 4
    data = torch.zeros(1, horizon, 3)
    conditioning_steps = torch.zeros(1, 2, 2)

    dct_valid = _make_objective(horizon=horizon, dct_coe_num=3)._kinematic_valid_mask(
        {"action_is_pad": torch.tensor([[False, True, False, False]])},
        data,
        conditioning_steps,
    )
    finite_difference_valid = _make_objective(
        horizon=horizon,
        dct_coe_num=0,
    )._kinematic_valid_mask(
        {"action_is_pad": torch.tensor([[False, True, False, False, False, True]])},
        data,
        conditioning_steps,
    )

    assert torch.equal(dct_valid, torch.tensor([[True, False, True, True]]))
    assert torch.equal(finite_difference_valid, torch.tensor([[False, False, True, False]]))


@pytest.mark.parametrize(
    ("dct_coe_num", "action_is_pad"),
    [
        (3, torch.zeros(2, 5, dtype=torch.bool)),
        (3, torch.zeros(1, 4, dtype=torch.bool)),
        (0, torch.zeros(2, 5, dtype=torch.bool)),
        (0, torch.zeros(2, 7, dtype=torch.bool)),
    ],
)
def test_kinematic_padding_mask_rejects_wrong_shape(
    dct_coe_num: int,
    action_is_pad: Tensor,
):
    horizon = 4
    objective = _make_objective(horizon=horizon, dct_coe_num=dct_coe_num)

    with pytest.raises(ValueError, match="action_is_pad must have shape"):
        objective._kinematic_valid_mask(
            {"action_is_pad": action_is_pad},
            torch.zeros(2, horizon, 3),
            torch.zeros(2, 2, 2),
        )


@pytest.mark.parametrize(
    ("dct_coe_num", "num_action_frames"),
    [
        (4, 5),
        (0, 4),
    ],
)
def test_compute_loss_rejects_wrong_action_frame_count_before_kinematic_warmup(
    dct_coe_num: int,
    num_action_frames: int,
):
    horizon = 4
    action_dim = 3
    conditioning_dim = 6
    objective = _make_objective(
        horizon=horizon,
        action_dim=action_dim,
        dct_coe_num=dct_coe_num,
        pre_train_steps=100,
    )

    with pytest.raises(ValueError, match="Flow-matching actions must have shape"):
        objective.compute_loss(
            TinyFlowModel(action_dim, conditioning_dim),
            {ACTION: torch.randn(2, num_action_frames, action_dim)},
            torch.randn(2, conditioning_dim),
            train_step=0,
        )


def test_central_difference_keeps_flow_chunk_and_padding_mask_centered_during_warmup(
    monkeypatch,
):
    horizon = 4
    action_dim = 2
    conditioning_dim = 6
    objective = _make_objective(
        horizon=horizon,
        action_dim=action_dim,
        dct_coe_num=0,
        do_mask_loss_for_padding=True,
        pre_train_steps=100,
    )
    monkeypatch.setattr(
        objective,
        "_sample_timesteps",
        lambda batch_size, device: torch.ones(batch_size, device=device),
    )
    model = RecordingFlowModel(action_dim, conditioning_dim)
    actions = torch.arange(
        (horizon + 2) * action_dim,
        dtype=torch.float32,
    ).reshape(1, horizon + 2, action_dim)
    action_is_pad = torch.tensor([[True, False, True, False, False, True]])
    captured_pad = []
    original_flow_loss = objective._flow_loss

    def record_flow_loss(predicted_velocity, target_velocity, action_is_pad=None):
        captured_pad.append(action_is_pad.detach().clone())
        return original_flow_loss(predicted_velocity, target_velocity, action_is_pad)

    monkeypatch.setattr(objective, "_flow_loss", record_flow_loss)
    loss, metrics = objective.compute_loss(
        model,
        {ACTION: actions, "action_is_pad": action_is_pad},
        torch.randn(1, conditioning_dim),
        train_step=0,
    )

    assert torch.isfinite(loss)
    assert model.last_actions is not None
    torch.testing.assert_close(model.last_actions, actions[:, 1 : horizon + 1])
    assert len(captured_pad) == 1
    torch.testing.assert_close(captured_pad[0], action_is_pad[:, 1 : horizon + 1])
    assert metrics["lambda_flow_k"] == 0.0
    assert metrics["physical_loss"] == 0.0  # t=1 makes t(1-t) vanish, including in warmup.
    assert metrics["weighted_physical_loss"] == 0.0


def test_dct_physical_loss_matches_value_level_formula_and_total(monkeypatch):
    horizon = 5
    action_dim = 2
    weight = 2.5
    objective = _make_objective(
        horizon=horizon,
        action_dim=action_dim,
        dct_coe_num=horizon,
        lambda_flow_k=0.0,
        phy_loss_weight=weight,
    )
    timesteps = torch.tensor([0.25, 0.75])
    monkeypatch.setattr(
        objective,
        "_sample_timesteps",
        lambda batch_size, device: timesteps.to(device),
    )
    monkeypatch.setattr(torch, "randn_like", lambda data: torch.zeros_like(data))
    actions = _single_mode_actions(
        horizon,
        mode=2,
        coefficients=torch.tensor([1.5, -0.75]),
        batch_scales=torch.tensor([1.0, 0.5]),
    )
    fps = torch.tensor([10.0, 20.0])
    model = ScaledActionFlowModel(scale=0.2)

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
    expected_flow_loss = flow_error.square().mean()
    expected_total = expected_flow_loss + weight * expected_physical_loss

    torch.testing.assert_close(loss, expected_total)
    assert metrics["physical_loss"] == pytest.approx(expected_physical_loss.item())
    assert metrics["weighted_physical_loss"] == pytest.approx((weight * expected_physical_loss).item())
    assert metrics["phy_loss_weight"] == weight
    loss.backward()
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad)


def test_dct_k0_physical_derivative_is_second_order_accurate_at_boundaries():
    horizon = 6
    objective = _make_objective(horizon=horizon, action_dim=2, dct_coe_num=0)
    sample_index = torch.arange(horizon, dtype=torch.float64)
    flow_error = torch.stack((sample_index.square(), 3.0 * sample_index + 2.0), dim=-1)
    flow_error = flow_error.unsqueeze(0)
    fps = torch.tensor(10.0, dtype=torch.float64)

    actual = objective._physical_action_dot(flow_error, fps)

    expected = torch.stack((2.0 * sample_index, torch.full_like(sample_index, 3.0)), dim=-1)
    torch.testing.assert_close(actual, expected.unsqueeze(0) * fps)


def test_physical_weight_only_changes_final_aggregation_not_fm_or_jvp(monkeypatch):
    torch.manual_seed(0)
    horizon = 4
    action_dim = 3
    conditioning_dim = 6
    weight = 0.7
    timesteps = torch.tensor([0.3, 0.8])
    objectives = [
        _make_objective(
            horizon=horizon,
            action_dim=action_dim,
            dct_coe_num=horizon,
            lambda_flow_k=0.2,
            phy_loss_weight=phy_loss_weight,
        )
        for phy_loss_weight in (0.0, weight)
    ]
    for objective in objectives:
        monkeypatch.setattr(
            objective,
            "_sample_timesteps",
            lambda batch_size, device: timesteps.to(device),
        )
    monkeypatch.setattr(torch, "randn_like", lambda data: torch.zeros_like(data))
    actions = torch.randn(2, horizon, action_dim)
    conditioning_steps = torch.randn(2, 2, conditioning_dim // 2)
    derivative_stencil = _reverse_conditioning_stencil(conditioning_steps)
    conditioning_vec = conditioning_steps.flatten(start_dim=1)
    model = TinyFlowModel(action_dim, conditioning_dim)

    baseline_loss, baseline_metrics = objectives[0].compute_loss(
        model,
        {ACTION: actions},
        conditioning_vec,
        conditioning_steps=conditioning_steps,
        derivative_conditioning_steps=derivative_stencil,
    )
    physical_loss, physical_metrics = objectives[1].compute_loss(
        model,
        {ACTION: actions},
        conditioning_vec,
        conditioning_steps=conditioning_steps,
        derivative_conditioning_steps=derivative_stencil,
    )

    for metric in (
        "flow_loss",
        "kinematic_loss",
        "physical_loss",
        "kinematic_state_jvp_rms",
        "kinematic_action_jvp_rms",
    ):
        assert physical_metrics[metric] == pytest.approx(baseline_metrics[metric])
    assert baseline_metrics["physical_loss"] > 0
    torch.testing.assert_close(
        physical_loss,
        baseline_loss + weight * torch.tensor(physical_metrics["physical_loss"]),
    )


@pytest.mark.parametrize("phy_loss_weight", [0.0, 0.5])
@pytest.mark.parametrize("lambda_flow_k", [0.0, 0.2])
def test_physical_metrics_are_reported_with_and_without_kinematic_loss(
    monkeypatch,
    phy_loss_weight: float,
    lambda_flow_k: float,
):
    torch.manual_seed(1)
    horizon = 4
    action_dim = 3
    conditioning_dim = 6
    objective = _make_objective(
        horizon=horizon,
        action_dim=action_dim,
        dct_coe_num=horizon,
        lambda_flow_k=lambda_flow_k,
        phy_loss_weight=phy_loss_weight,
    )
    monkeypatch.setattr(
        objective,
        "_sample_timesteps",
        lambda batch_size, device: torch.full((batch_size,), 0.4, device=device),
    )
    monkeypatch.setattr(torch, "randn_like", lambda data: torch.zeros_like(data))
    conditioning_steps = torch.randn(2, 2, conditioning_dim // 2)

    loss, metrics = objective.compute_loss(
        TinyFlowModel(action_dim, conditioning_dim),
        {ACTION: torch.randn(2, horizon, action_dim)},
        conditioning_steps.flatten(start_dim=1),
        conditioning_steps=conditioning_steps,
        derivative_conditioning_steps=_reverse_conditioning_stencil(conditioning_steps),
    )

    assert metrics["physical_loss"] > 0
    assert metrics["weighted_physical_loss"] == pytest.approx(phy_loss_weight * metrics["physical_loss"])
    if lambda_flow_k > 0:
        assert "kinematic_residual_time_weight_mean" in metrics
        assert "kinematic_loss_time_weight_mean" in metrics
    assert loss.item() == pytest.approx(
        metrics["flow_loss"]
        + metrics["lambda_flow_k"] * metrics["kinematic_loss"]
        + metrics["weighted_physical_loss"]
    )


@pytest.mark.parametrize("phy_loss_weight", [0.0, 0.5])
def test_dct_k0_logs_real_physical_loss_for_zero_and_positive_weight(
    monkeypatch,
    phy_loss_weight: float,
):
    horizon = 4
    objective = _make_objective(
        horizon=horizon,
        action_dim=2,
        dct_coe_num=0,
        lambda_flow_k=0.0,
        phy_loss_weight=phy_loss_weight,
    )
    timesteps = torch.full((2,), 0.5)
    monkeypatch.setattr(
        objective,
        "_sample_timesteps",
        lambda batch_size, device: timesteps.to(device),
    )
    monkeypatch.setattr(torch, "randn_like", lambda data: torch.zeros_like(data))
    monkeypatch.setattr(
        objective,
        "_action_dot",
        lambda *args, **kwargs: pytest.fail("K0 physical loss must not use the H+2 JVP stencil"),
    )
    model = ScaledActionFlowModel(scale=0.3)
    sample_index = torch.arange(horizon, dtype=torch.float32)
    actions = torch.stack((sample_index.square(), sample_index), dim=-1)
    actions = actions.unsqueeze(0).repeat(2, 1, 1)

    loss, metrics = objective.compute_loss(
        model,
        {ACTION: actions},
        conditioning_vec=torch.zeros(2, 1),
    )

    assert torch.isfinite(loss)
    assert metrics["physical_loss"] > 0
    assert metrics["weighted_physical_loss"] == pytest.approx(phy_loss_weight * metrics["physical_loss"])
    if phy_loss_weight > 0:
        loss.backward()
        assert model.scale.grad is not None
        assert torch.isfinite(model.scale.grad)


def test_physical_masked_mean_reuses_flow_padding_semantics():
    objective = _make_objective(
        horizon=3,
        action_dim=2,
        dct_coe_num=3,
        do_mask_loss_for_padding=True,
    )
    values = torch.tensor([[[1.0, 3.0], [100.0, 200.0], [5.0, 7.0]]])
    action_is_pad = torch.tensor([[False, True, False]])

    actual = objective._masked_action_mean(values, action_is_pad)

    torch.testing.assert_close(actual, torch.tensor(4.0))


def test_gripper_is_masked_before_action_jvp(monkeypatch):
    torch.manual_seed(0)
    horizon = 4
    action_dim = 3
    conditioning_dim = 6
    objective = _make_objective(
        horizon=horizon,
        action_dim=action_dim,
        dct_coe_num=2,
        gripper_first=False,
    )
    model = TinyFlowModel(action_dim, conditioning_dim)
    coefficients = torch.tensor([0.7, -1.2, 2.0])
    actions = _single_mode_actions(horizon, 1, coefficients)
    data = actions[:, :horizon]
    x_t = torch.randn_like(data)
    timesteps = torch.full((1,), 0.5)
    conditioning_steps = torch.randn(1, 2, conditioning_dim // 2)
    conditioning_vec = conditioning_steps.flatten(start_dim=1)
    captured_action_tangents = []

    original_jvp = torch.func.jvp

    def record_jvp(func, primals, tangents, *args, **kwargs):
        captured_action_tangents.extend(
            tangent.detach().clone()
            for tangent in tangents
            if isinstance(tangent, Tensor) and tangent.ndim == 3
        )
        return original_jvp(func, primals, tangents, *args, **kwargs)

    monkeypatch.setattr(torch.func, "jvp", record_jvp)

    objective._compute_kinematic_loss(
        model=model,
        batch={ACTION: actions},
        data=data,
        action_sequence=actions,
        x_t=x_t,
        t=timesteps,
        conditioning_vec=conditioning_vec,
        conditioning_steps=conditioning_steps,
        derivative_conditioning_steps=_reverse_conditioning_stencil(conditioning_steps),
    )

    assert len(captured_action_tangents) == 1
    tangent = captured_action_tangents[0]
    assert torch.count_nonzero(tangent[..., :-1]) > 0
    assert torch.count_nonzero(tangent[..., -1]) == 0


def test_full_jvp_reports_one_minus_k_time_weights():
    horizon = 4
    action_dim = 3
    conditioning_dim = 6
    model = TinyFlowModel(action_dim, conditioning_dim)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    slopes = torch.tensor([1.0, 2.0])
    action_steps = torch.arange(horizon + 2, dtype=torch.float32)
    actions = slopes[:, None, None] * action_steps[None, :, None]
    actions = actions.expand(-1, -1, action_dim).clone()
    data = actions[:, 1 : horizon + 1]
    x_t = torch.randn_like(data)
    timesteps = torch.tensor([0.25, 0.5])
    conditioning_steps = torch.randn(2, 2, conditioning_dim // 2)
    conditioning_vec = conditioning_steps.flatten(start_dim=1)

    objective = _make_objective(
        horizon=horizon,
        action_dim=action_dim,
    )
    _, kinematic_loss, _, metrics = objective._compute_kinematic_loss(
        model=model,
        batch={ACTION: actions},
        data=data,
        action_sequence=actions,
        x_t=x_t,
        t=timesteps,
        conditioning_vec=conditioning_vec,
        conditioning_steps=conditioning_steps,
        derivative_conditioning_steps=_reverse_conditioning_stencil(conditioning_steps),
    )

    # The full-JVP residual multiplier is (1-k), so the loss uses (1-k)^2.
    torch.testing.assert_close(kinematic_loss, torch.tensor(78.125))
    torch.testing.assert_close(metrics["kinematic_residual_time_weight_mean"], torch.tensor(0.625))
    torch.testing.assert_close(metrics["kinematic_loss_time_weight_mean"], torch.tensor(0.40625))


def test_kinematic_loss_backpropagates_through_action_and_conditioning_jvps():
    horizon = 4
    action_dim = 3
    conditioning_step_dim = 3
    conditioning_dim = 2 * conditioning_step_dim
    action_steps = torch.arange(horizon + 2, dtype=torch.float32)
    actions = action_steps[None, :, None].expand(1, -1, action_dim).clone()
    data = actions[:, 1 : horizon + 1]
    x_t = torch.randn_like(data)
    timesteps = torch.full((1,), 0.5)
    conditioning_steps = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]],
    )
    conditioning_vec = conditioning_steps.flatten(start_dim=1)

    model = TinyFlowModel(action_dim, conditioning_dim)
    with torch.no_grad():
        model.action_projection.weight.copy_(torch.eye(action_dim))
        model.conditioning_projection.weight.fill_(0.01)
        model.time_scale.zero_()
    objective = _make_objective(
        horizon=horizon,
        action_dim=action_dim,
    )
    _, kinematic_loss, _, _ = objective._compute_kinematic_loss(
        model=model,
        batch={ACTION: actions},
        data=data,
        action_sequence=actions,
        x_t=x_t,
        t=timesteps,
        conditioning_vec=conditioning_vec,
        conditioning_steps=conditioning_steps,
        derivative_conditioning_steps=_reverse_conditioning_stencil(conditioning_steps),
    )
    kinematic_loss.backward()
    action_grad = model.action_projection.weight.grad
    conditioning_grad = model.conditioning_projection.weight.grad
    assert action_grad is not None
    assert torch.isfinite(action_grad).all()
    assert torch.count_nonzero(action_grad) > 0
    assert conditioning_grad is not None
    assert torch.isfinite(conditioning_grad).all()
    assert torch.count_nonzero(conditioning_grad) > 0


@pytest.mark.parametrize("action_jvp_grad_scale", [0.1, 0.25])
def test_action_jvp_grad_scale_preserves_forward_and_scales_action_gradient(
    action_jvp_grad_scale: float,
):
    horizon = 4
    action_dim = 3
    conditioning_step_dim = 3
    conditioning_dim = 2 * conditioning_step_dim
    action_steps = torch.arange(horizon + 2, dtype=torch.float32)
    actions = action_steps[None, :, None].expand(1, -1, action_dim).clone()
    data = actions[:, 1 : horizon + 1]
    x_t = torch.randn_like(data)
    timesteps = torch.full((1,), 0.5)
    conditioning_steps = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]],
    )
    conditioning_vec = conditioning_steps.flatten(start_dim=1)

    losses = []
    gradients = []
    for grad_scale in (1.0, action_jvp_grad_scale):
        model = TinyFlowModel(action_dim, conditioning_dim)
        with torch.no_grad():
            model.action_projection.weight.copy_(torch.eye(action_dim))
            model.conditioning_projection.weight.fill_(0.01)
            model.time_scale.zero_()
        objective = _make_objective(
            horizon=horizon,
            action_dim=action_dim,
            action_jvp_grad_scale=grad_scale,
        )
        _, kinematic_loss, _, _ = objective._compute_kinematic_loss(
            model=model,
            batch={ACTION: actions},
            data=data,
            action_sequence=actions,
            x_t=x_t,
            t=timesteps,
            conditioning_vec=conditioning_vec,
            conditioning_steps=conditioning_steps,
            derivative_conditioning_steps=_reverse_conditioning_stencil(conditioning_steps),
        )
        kinematic_loss.backward()
        losses.append(kinematic_loss.detach())
        gradients.append(
            (
                model.action_projection.weight.grad.clone(),
                model.conditioning_projection.weight.grad.clone(),
            )
        )

    torch.testing.assert_close(losses[1], losses[0])
    full_action_grad, full_conditioning_grad = gradients[0]
    scaled_action_grad, scaled_conditioning_grad = gradients[1]
    torch.testing.assert_close(
        scaled_action_grad,
        action_jvp_grad_scale * full_action_grad,
    )
    torch.testing.assert_close(scaled_conditioning_grad, full_conditioning_grad)


@pytest.mark.parametrize("action_jvp_grad_scale", [-0.1, 1.1, float("nan"), float("inf")])
def test_action_jvp_grad_scale_validation_rejects_invalid_values(
    action_jvp_grad_scale: float,
):
    with pytest.raises(ValueError, match="action_jvp_grad_scale"):
        MultiTaskDiTConfig(action_jvp_grad_scale=action_jvp_grad_scale)


@pytest.mark.parametrize("dropout", [0.0, 0.5])
def test_nonstochastic_jvp_terms_share_dropout_realization(
    monkeypatch,
    dropout: float,
):
    torch.manual_seed(123)
    horizon = 4
    action_dim = 3
    conditioning_step_dim = 3
    conditioning_dim = 2 * conditioning_step_dim
    objective = _make_objective(
        horizon=horizon,
        action_dim=action_dim,
        dct_coe_num=horizon,
    )
    model = DropoutFlowModel(action_dim, conditioning_dim, dropout=dropout).train()
    with torch.no_grad():
        model.action_projection.weight.fill_(0.2)
        model.conditioning_projection.weight.fill_(0.1)
        model.time_scale.fill_(0.3)

    actions = torch.arange(
        1,
        1 + horizon * action_dim,
        dtype=torch.float32,
    ).reshape(1, horizon, action_dim)
    data = actions.clone()
    x_t = actions * 0.25
    timesteps = torch.full((1,), 0.5)
    conditioning_steps = torch.tensor(
        [[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]],
    )
    conditioning_vec = conditioning_steps.flatten(start_dim=1)

    jvp_primals = []
    original_jvp = torch.func.jvp

    def record_jvp(func, primals, tangents, *args, **kwargs):
        primal, tangent = original_jvp(func, primals, tangents, *args, **kwargs)
        jvp_primals.append(primal.detach().clone())
        return primal, tangent

    monkeypatch.setattr(torch.func, "jvp", record_jvp)
    rng_before_jvps = torch.get_rng_state()
    objective._compute_kinematic_loss(
        model=model,
        batch={ACTION: actions},
        data=data,
        action_sequence=actions,
        x_t=x_t,
        t=timesteps,
        conditioning_vec=conditioning_vec,
        conditioning_steps=conditioning_steps,
        derivative_conditioning_steps=_reverse_conditioning_stencil(conditioning_steps),
    )
    rng_after_shared_jvps = torch.get_rng_state()

    assert len(jvp_primals) == 2
    torch.testing.assert_close(jvp_primals[1], jvp_primals[0])

    # The shared-mask pair should consume the same random stream as one vector-field
    # evaluation, rather than either consuming two masks or restoring the stream entirely.
    torch.set_rng_state(rng_before_jvps)
    conditioning_dot_vec = objective._conditioning_dot(
        conditioning_steps,
        torch.tensor(objective.config.sample_frequency),
        derivative_conditioning_steps=_reverse_conditioning_stencil(conditioning_steps),
    )
    original_jvp(
        lambda cond: model(x_t, timesteps, conditioning_vec=cond),
        (conditioning_vec,),
        (conditioning_dot_vec,),
    )
    rng_after_one_jvp = torch.get_rng_state()
    assert torch.equal(rng_after_shared_jvps, rng_after_one_jvp)
    torch.set_rng_state(rng_after_shared_jvps)


def test_dct_kinematic_loss_supports_forward_and_backward():
    torch.manual_seed(0)
    batch_size = 2
    horizon = 4
    action_dim = 3
    conditioning_step_dim = 3
    conditioning_dim = 2 * conditioning_step_dim
    objective = _make_objective(
        horizon=horizon,
        action_dim=action_dim,
        dct_coe_num=horizon,
    )
    model = TinyFlowModel(action_dim, conditioning_dim)
    batch = {ACTION: torch.randn(batch_size, horizon, action_dim)}
    conditioning_steps = torch.randn(
        batch_size,
        2,
        conditioning_step_dim,
        requires_grad=True,
    )
    conditioning_vec = conditioning_steps.flatten(start_dim=1)

    loss, output = objective.compute_loss(
        model,
        batch,
        conditioning_vec,
        conditioning_steps=conditioning_steps,
        derivative_conditioning_steps=_reverse_conditioning_stencil(conditioning_steps),
        train_step=0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert math.isfinite(output["kinematic_loss"])
    assert math.isfinite(output["kinematic_state_jvp_rms"])
    assert math.isfinite(output["kinematic_action_jvp_rms"])
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
