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

from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import FlowMatchingObjective
from lerobot.utils.constants import ACTION


def _make_objective(
    horizon: int = 7,
    action_dim: int = 3,
    **overrides,
) -> FlowMatchingObjective:
    values = {
        "dct_coe_num": 0,
        "conditioning_derivative_mode": "reverse",
        "enable_stochastic": False,
        "gripper_first": True,
        "image_only_condition_jvp": False,
        "lambda_flow_k": 1.0,
        "pre_train_steps": 0,
        "sample_frequency": 10.0,
        "sigma_min": 0.0,
        "timestep_sampling_strategy": "uniform",
        "use_1_k": False,
        "use_jvp_ak": False,
    }
    values.update(overrides)
    return FlowMatchingObjective(
        SimpleNamespace(**values),
        action_dim=action_dim,
        horizon=horizon,
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


@pytest.mark.parametrize("dct_coe_num", [-1, 5])
def test_dct_coe_num_validation_rejects_out_of_range_values(dct_coe_num: int):
    with pytest.raises(ValueError, match="dct_coe_num"):
        MultiTaskDiTConfig(horizon=4, dct_coe_num=dct_coe_num)


@pytest.mark.parametrize("dct_coe_num", [0, 4])
def test_dct_coe_num_validation_accepts_boundaries(dct_coe_num: int):
    config = MultiTaskDiTConfig(horizon=4, dct_coe_num=dct_coe_num)
    assert config.dct_coe_num == dct_coe_num


def test_action_delta_indices_keep_extra_frame_for_forward_difference_compatibility():
    config = MultiTaskDiTConfig(horizon=4, n_obs_steps=2, dct_coe_num=4)

    assert config.action_delta_indices == [-1, 0, 1, 2, 3]


def test_zero_dct_coefficients_preserve_forward_difference_exactly():
    horizon = 6
    objective = _make_objective(horizon=horizon, dct_coe_num=0)
    actions = torch.randn(3, horizon + 1, 4)
    fps = torch.tensor([7.0, 11.0, 19.0]).reshape(3, 1, 1)

    expected = (actions[:, 1 : horizon + 1] - actions[:, :horizon]) * fps

    assert torch.equal(objective._action_dot(actions, fps), expected)


def test_dct_action_derivative_ignores_extra_action_frame():
    horizon = 6
    objective = _make_objective(horizon=horizon, dct_coe_num=horizon)
    actions = torch.randn(2, horizon + 1, 3, dtype=torch.float64)
    changed_extra_action = actions.clone()
    changed_extra_action[:, -1] = 1_000.0
    fps = torch.tensor(10.0, dtype=actions.dtype)

    expected = objective._action_dot(actions, fps)
    actual = objective._action_dot(changed_extra_action, fps)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("num_modes", [1, 7])
def test_constant_sequence_has_zero_dct_derivative(num_modes: int):
    horizon = 7
    objective = _make_objective(horizon=horizon, dct_coe_num=num_modes)
    actions = torch.full((2, horizon + 1, 3), 1.75, dtype=torch.float64)

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
    actions = torch.randn(2, horizon + 1, 2, dtype=torch.float64)
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
    actions = torch.randn(2, horizon + 1, 2, dtype=dtype)
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

    with pytest.raises(ValueError, match=r"requires at least horizon \(5\).+got 4"):
        objective._action_dot(actions, torch.tensor(10.0))


def test_dct_padding_mask_ignores_extra_frame_while_forward_difference_uses_it():
    horizon = 4
    data = torch.zeros(1, horizon, 3)
    conditioning_steps = torch.zeros(1, 2, 2)
    action_is_pad = torch.tensor([[False, True, False, False, True]])
    batch = {"action_is_pad": action_is_pad}

    dct_valid = _make_objective(horizon=horizon, dct_coe_num=3)._kinematic_valid_mask(
        batch,
        data,
        conditioning_steps,
    )
    finite_difference_valid = _make_objective(
        horizon=horizon,
        dct_coe_num=0,
    )._kinematic_valid_mask(
        batch,
        data,
        conditioning_steps,
    )

    assert torch.equal(dct_valid, torch.tensor([[True, False, True, True]]))
    assert torch.equal(finite_difference_valid, torch.tensor([[False, False, True, False]]))


@pytest.mark.parametrize("enable_stochastic", [False, True])
def test_gripper_is_masked_before_action_jvp(monkeypatch, enable_stochastic: bool):
    torch.manual_seed(0)
    horizon = 4
    action_dim = 3
    conditioning_dim = 6
    objective = _make_objective(
        horizon=horizon,
        action_dim=action_dim,
        dct_coe_num=2,
        enable_stochastic=enable_stochastic,
        gripper_first=False,
        use_jvp_ak=True,
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

    if enable_stochastic:
        original_jvp = torch.autograd.functional.jvp

        def record_jvp(func, inputs, v=None, *args, **kwargs):
            tangents = v if isinstance(v, tuple) else (v,)
            captured_action_tangents.extend(
                tangent.detach().clone()
                for tangent in tangents
                if isinstance(tangent, Tensor) and tangent.ndim == 3
            )
            return original_jvp(func, inputs, v, *args, **kwargs)

        monkeypatch.setattr(torch.autograd.functional, "jvp", record_jvp)
    else:
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


@pytest.mark.parametrize("enable_stochastic", [False, True])
def test_use_1_k_scales_state_jvp_by_one_minus_flow_time(enable_stochastic: bool):
    horizon = 4
    action_dim = 3
    conditioning_dim = 6
    model = TinyFlowModel(action_dim, conditioning_dim)
    actions = torch.zeros(1, horizon + 1, action_dim)
    data = actions[:, :horizon]
    x_t = torch.randn_like(data)
    timesteps = torch.full((1,), 0.25)
    conditioning_steps = torch.randn(1, 2, conditioning_dim // 2)
    conditioning_vec = conditioning_steps.flatten(start_dim=1)

    losses = []
    for use_1_k in (False, True):
        objective = _make_objective(
            horizon=horizon,
            action_dim=action_dim,
            enable_stochastic=enable_stochastic,
            use_1_k=use_1_k,
            use_jvp_ak=False,
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
        losses.append(kinematic_loss)

    torch.testing.assert_close(losses[1], losses[0] * 0.75**2)


@pytest.mark.parametrize("use_jvp_ak", [False, True])
@pytest.mark.parametrize("enable_stochastic", [False, True])
def test_dct_kinematic_branches_support_forward_and_backward(
    use_jvp_ak: bool,
    enable_stochastic: bool,
):
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
        enable_stochastic=enable_stochastic,
        use_jvp_ak=use_jvp_ak,
    )
    model = TinyFlowModel(action_dim, conditioning_dim)
    batch = {ACTION: torch.randn(batch_size, horizon + 1, action_dim)}
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
    if not enable_stochastic:
        assert math.isfinite(output["kinematic_state_jvp_rms"])
        if use_jvp_ak:
            assert math.isfinite(output["kinematic_action_jvp_rms"])
        else:
            assert "kinematic_action_jvp_rms" not in output
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
