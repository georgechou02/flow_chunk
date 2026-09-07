import math

import pytest
import torch
from torch import Tensor, nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.flow.configuration_flow import FlowConfig
from lerobot.policies.flow.modeling_flow import FlowPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_STATE


class _TinyVectorField(nn.Module):
    def __init__(self, action_dim: int, conditioning_dim: int):
        super().__init__()
        self.action_projection = nn.Linear(action_dim, action_dim, bias=False)
        self.conditioning_projection = nn.Linear(conditioning_dim, action_dim, bias=False)

    def forward(
        self,
        actions: Tensor,
        timesteps: Tensor,
        global_cond: Tensor | None = None,
    ) -> Tensor:
        assert global_cond is not None
        return (
            self.action_projection(actions)
            + self.conditioning_projection(global_cond)[:, None]
            + timesteps[:, None, None]
        )


class _ScaledActionVectorField(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(
        self,
        actions: Tensor,
        timesteps: Tensor,
        global_cond: Tensor | None = None,
    ) -> Tensor:
        assert global_cond is not None
        return self.scale * actions


def _make_config(**overrides) -> FlowConfig:
    values = {
        "device": "cpu",
        "input_features": {
            OBS_STATE: PolicyFeature(FeatureType.STATE, (3,)),
            OBS_ENV_STATE: PolicyFeature(FeatureType.ENV, (4,)),
        },
        "output_features": {ACTION: PolicyFeature(FeatureType.ACTION, (2,))},
        "n_obs_steps": 2,
        "horizon": 8,
        "n_action_steps": 4,
        "down_dims": (16, 32),
        "n_groups": 8,
        "diffusion_step_embed_dim": 16,
        "kernel_size": 3,
        "pretrained_backbone_weights": None,
        "interpolation_mode": "bspline",
        "dct_coe_num": 5,
        "bspline_degree": 2,
        "bspline_coe_num": 5,
        "conditioning_derivative_mode": "central",
    }
    values.update(overrides)
    return FlowConfig(**values)


def test_first_order_defaults_match_training_script():
    config = FlowConfig()

    assert config.lambda_flow_k == 0.01
    assert config.pre_train_steps == 0
    assert config.phy_loss_weight == 0.0
    assert config.interpolation_mode == "bspline"
    assert config.dct_coe_num == 48
    assert config.bspline_degree == 2
    assert config.bspline_coe_num == 48
    assert config.conditioning_derivative_mode == "central"
    assert config.image_only_condition_jvp is False
    assert config.observation_delta_indices == [-2, -1, 0, 1]


@pytest.mark.parametrize("phy_loss_weight", [-1.0, float("nan"), float("inf")])
def test_physical_loss_weight_rejects_invalid_values(phy_loss_weight: float):
    with pytest.raises(ValueError, match="phy_loss_weight"):
        _make_config(phy_loss_weight=phy_loss_weight)


@pytest.mark.parametrize("pre_train_steps", [-1, 1.5, True])
def test_pre_train_steps_rejects_invalid_values(pre_train_steps):
    with pytest.raises(ValueError, match="pre_train_steps"):
        _make_config(pre_train_steps=pre_train_steps)


def test_pretraining_disables_kinematic_loss_until_global_step_boundary():
    config = _make_config(
        lambda_flow_k=0.2,
        pre_train_steps=2,
        phy_loss_weight=0.0,
        clean_action_log_freq=0,
    )
    policy = FlowPolicy(config).train()
    policy.flow.unet = _TinyVectorField(action_dim=2, conditioning_dim=14)
    assert config.observation_delta_indices == [-2, -1, 0, 1]
    batch = {
        OBS_STATE: torch.randn(2, 4, 3),
        OBS_ENV_STATE: torch.randn(2, 4, 4),
        ACTION: torch.randn(2, 8, 2),
        "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
        f"{OBS_STATE}_is_pad": torch.zeros(2, 4, dtype=torch.bool),
    }

    for step in (0, 1):
        policy.set_train_step(step)
        loss, metrics = policy(batch)

        assert metrics["train_step"] == step
        assert metrics["pre_train_steps"] == 2
        assert metrics["lambda_flow_k_config"] == pytest.approx(0.2)
        assert metrics["lambda_flow_k"] == 0.0
        assert metrics["kinematic_loss"] == 0.0
        assert loss.item() == pytest.approx(metrics["flow_loss"])

    policy.set_train_step(2)
    loss, metrics = policy(batch)

    assert metrics["train_step"] == 2
    assert metrics["lambda_flow_k"] == pytest.approx(0.2)
    assert metrics["kinematic_loss"] > 0
    assert loss.item() == pytest.approx(metrics["flow_loss"] + 0.2 * metrics["kinematic_loss"])


@pytest.mark.parametrize("clean_action_log_freq", [-1, 1.5, True])
def test_clean_action_log_frequency_rejects_invalid_values(clean_action_log_freq):
    with pytest.raises(ValueError, match="clean_action_log_freq"):
        _make_config(clean_action_log_freq=clean_action_log_freq)


def test_clean_action_metrics_follow_cadence_and_restore_training_state(monkeypatch):
    config = _make_config(
        lambda_flow_k=0.0,
        clean_action_log_freq=2,
        num_inference_steps=1,
    )
    policy = FlowPolicy(config).train()
    policy.flow.unet = _TinyVectorField(action_dim=2, conditioning_dim=14)
    batch = {
        OBS_STATE: torch.randn(2, 2, 3),
        OBS_ENV_STATE: torch.randn(2, 2, 4),
        ACTION: torch.randn(2, 8, 2),
        "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
    }
    batch["action_is_pad"][0, 2] = True
    sampler_calls = 0

    def fake_generate_actions(sample_batch, noise=None):
        nonlocal sampler_calls
        sampler_calls += 1
        assert not policy.training
        assert noise is not None
        return sample_batch[ACTION][:, 1:5] + 2.0

    monkeypatch.setattr(policy.flow, "generate_actions", fake_generate_actions)

    policy.set_train_step(0)
    _, first_metrics = policy(batch)
    assert "clean_action_mse" not in first_metrics
    assert sampler_calls == 0

    policy.set_train_step(1)
    _, second_metrics = policy(batch)
    assert second_metrics["clean_action_mse"] == pytest.approx(4.0)
    assert second_metrics["clean_action_valid_ratio"] == pytest.approx(7 / 8)
    assert sampler_calls == 1
    assert policy.training
    assert policy.flow.unet.training


@pytest.mark.parametrize(
    ("mode", "expected_indices"),
    [
        ("reverse", [-2, -1, 0]),
        ("forward", [-1, 0, 1]),
        ("central", [-2, -1, 0, 1]),
    ],
)
def test_observation_indices_cover_conditioning_derivative_stencil(
    mode: str,
    expected_indices: list[int],
):
    config = _make_config(conditioning_derivative_mode=mode)

    assert config.observation_delta_indices == expected_indices


def test_image_only_condition_jvp_masks_state_and_environment_tangents():
    input_features = {
        OBS_STATE: PolicyFeature(FeatureType.STATE, (3,)),
        OBS_IMAGE: PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
        OBS_ENV_STATE: PolicyFeature(FeatureType.ENV, (4,)),
    }
    policy = FlowPolicy(
        _make_config(
            input_features=input_features,
            image_only_condition_jvp=True,
            use_separate_rgb_encoder_per_camera=False,
        )
    )
    image_slice = policy.flow.image_feature_slice
    assert image_slice is not None
    assert image_slice.start == 3
    assert image_slice.stop == 3 + policy.flow.rgb_encoder.feature_dim

    feature_dim = 3 + policy.flow.rgb_encoder.feature_dim + 4
    derivative_steps = torch.arange(4 * feature_dim, dtype=torch.float64).reshape(1, 4, feature_dim)
    conditioning_steps = derivative_steps[:, 1:3]
    fps = torch.tensor(10.0, dtype=torch.float64)

    actual = policy.flow._conditioning_dot(conditioning_steps, derivative_steps, fps).reshape(
        1, 2, feature_dim
    )
    expected = (derivative_steps[:, 2:4] - derivative_steps[:, :2]) * (fps * 0.5)
    expected[..., : image_slice.start] = 0
    expected[..., image_slice.stop :] = 0

    torch.testing.assert_close(actual, expected)


def test_image_only_condition_jvp_requires_an_image_feature():
    policy = FlowPolicy(_make_config(image_only_condition_jvp=True))
    derivative_steps = torch.randn(1, 4, 7)

    with pytest.raises(ValueError, match="requires at least one image feature"):
        policy.flow._conditioning_dot(
            derivative_steps[:, 1:3],
            derivative_steps,
            torch.tensor(10.0),
        )


def test_dct_zero_mode_requests_neighboring_action_frames():
    config = _make_config(interpolation_mode="dct", dct_coe_num=0)

    assert config.action_delta_indices == list(range(-2, 8))


def test_gripper_can_be_excluded_from_first_order_supervision():
    policy = FlowPolicy(_make_config(gripper_first=False))

    mask = policy.flow._kinematic_action_mask(torch.empty(0))

    torch.testing.assert_close(mask, torch.tensor([True, False]))


def test_dct_action_derivative_matches_single_mode():
    horizon = 8
    mode = 2
    fps = torch.tensor(10.0, dtype=torch.float64)
    config = _make_config(interpolation_mode="dct", dct_coe_num=4)
    policy = FlowPolicy(config)
    sample_points = torch.arange(horizon, dtype=torch.float64) + 0.5
    alpha = math.sqrt(2.0 / horizon)
    actions = (alpha * torch.cos(math.pi * mode * sample_points / horizon))[None, :, None]
    actions = actions.expand(-1, -1, 2).clone()
    expected = (-alpha * (math.pi * mode / horizon) * torch.sin(math.pi * mode * sample_points / horizon))[
        None, :, None
    ]
    expected = expected.expand_as(actions) * fps

    actual = policy.flow._action_dot(actions, fps)

    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_bspline_action_derivative_matches_quadratic_trajectory():
    horizon = 8
    fps = torch.tensor(10.0, dtype=torch.float64)
    config = _make_config()
    policy = FlowPolicy(config)
    sample_u = torch.linspace(0.0, 1.0, horizon, dtype=torch.float64)
    actions = torch.stack((sample_u.square(), 3 * sample_u - 2), dim=-1).unsqueeze(0)
    expected_du = torch.stack((2 * sample_u, torch.full_like(sample_u, 3)), dim=-1)
    expected = expected_du.unsqueeze(0) * fps / (horizon - 1)

    actual = policy.flow._action_dot(actions, fps)

    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)


def test_first_order_loss_runs_both_jvps_and_backpropagates():
    torch.manual_seed(0)
    config = _make_config(lambda_flow_k=0.2, phy_loss_weight=0.3)
    policy = FlowPolicy(config).train()
    policy.flow.unet = _TinyVectorField(action_dim=2, conditioning_dim=14)
    batch = {
        OBS_STATE: torch.randn(2, 4, 3),
        OBS_ENV_STATE: torch.randn(2, 4, 4),
        ACTION: torch.randn(2, 8, 2),
        "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
        f"{OBS_STATE}_is_pad": torch.zeros(2, 4, dtype=torch.bool),
    }

    loss, metrics = policy(batch)
    loss.backward()

    assert metrics["kinematic_loss"] > 0
    assert metrics["kinematic_state_jvp_rms"] > 0
    assert metrics["kinematic_action_jvp_rms"] > 0
    assert metrics["kinematic_valid_ratio"] == 1.0
    assert metrics["physical_loss"] > 0
    assert metrics["total_loss"] == pytest.approx(
        metrics["flow_loss"]
        + config.lambda_flow_k * metrics["kinematic_loss"]
        + metrics["weighted_physical_loss"]
    )
    assert all(parameter.grad is not None for parameter in policy.flow.unet.parameters())


def test_lambda_zero_preserves_plain_flow_batch_shapes():
    config = _make_config(lambda_flow_k=0.0)
    policy = FlowPolicy(config).train()
    policy.flow.unet = _TinyVectorField(action_dim=2, conditioning_dim=14)
    batch = {
        OBS_STATE: torch.randn(2, 2, 3),
        OBS_ENV_STATE: torch.randn(2, 2, 4),
        ACTION: torch.randn(2, 8, 2),
        "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
    }

    loss, metrics = policy(batch)

    assert loss.item() == pytest.approx(metrics["flow_loss"])
    assert metrics["kinematic_loss"] == 0.0
    assert metrics["physical_loss"] > 0
    assert metrics["weighted_physical_loss"] == 0.0


def test_dct_physical_loss_matches_value_level_formula_and_backpropagates(monkeypatch):
    torch.manual_seed(1)
    weight = 1.75
    config = _make_config(
        lambda_flow_k=0.0,
        phy_loss_weight=weight,
        interpolation_mode="dct",
        dct_coe_num=8,
    )
    policy = FlowPolicy(config).train()
    model = _ScaledActionVectorField(scale=0.2)
    policy.flow.unet = model
    timesteps = torch.tensor([0.25, 0.75])
    monkeypatch.setattr(
        policy.flow,
        "_sample_timesteps",
        lambda batch_size, device, dtype: timesteps.to(device=device, dtype=dtype),
    )
    monkeypatch.setattr(torch, "randn_like", lambda data: torch.zeros_like(data))
    actions = torch.randn(2, 8, 2)
    fps = torch.tensor([10.0, 20.0])
    batch = {
        OBS_STATE: torch.randn(2, 2, 3),
        OBS_ENV_STATE: torch.randn(2, 2, 4),
        ACTION: actions,
        "sample_frequency": fps,
    }

    loss, metrics = policy(batch)

    t_expanded = timesteps.view(-1, 1, 1)
    predicted_velocity = model.scale * t_expanded * actions
    target_velocity = actions
    flow_error = predicted_velocity - target_velocity
    g_flow_error = policy.flow._action_dot(flow_error, fps.view(-1, 1, 1))
    physical_residual = t_expanded * (1 - t_expanded) * g_flow_error
    expected_physical_loss = physical_residual.square().mean()
    expected_total = flow_error.square().mean() + weight * expected_physical_loss

    torch.testing.assert_close(loss, expected_total)
    assert metrics["physical_loss"] == pytest.approx(expected_physical_loss.item())
    assert metrics["weighted_physical_loss"] == pytest.approx((weight * expected_physical_loss).item())
    assert metrics["phy_loss_weight"] == weight
    loss.backward()
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad)


def test_dct_zero_mode_physical_derivative_uses_second_order_boundaries():
    config = _make_config(
        lambda_flow_k=0.0,
        interpolation_mode="dct",
        dct_coe_num=0,
    )
    policy = FlowPolicy(config)
    sample_index = torch.arange(config.horizon, dtype=torch.float64)
    flow_error = torch.stack((sample_index.square(), 3.0 * sample_index + 2.0), dim=-1).unsqueeze(0)
    fps = torch.tensor(10.0, dtype=torch.float64)

    actual = policy.flow._physical_action_dot(flow_error, fps)

    expected = torch.stack((2.0 * sample_index, torch.full_like(sample_index, 3.0)), dim=-1)
    torch.testing.assert_close(actual, expected.unsqueeze(0) * fps)


def test_bspline_physical_loss_excludes_discrete_gripper(monkeypatch):
    config = _make_config(
        lambda_flow_k=0.0,
        phy_loss_weight=1.0,
        gripper_first=False,
    )
    policy = FlowPolicy(config).train()
    policy.flow.unet = _ScaledActionVectorField(scale=0.0)
    monkeypatch.setattr(
        policy.flow,
        "_sample_timesteps",
        lambda batch_size, device, dtype: torch.full((batch_size,), 0.5, device=device, dtype=dtype),
    )
    monkeypatch.setattr(torch, "randn_like", lambda data: torch.zeros_like(data))
    actions = torch.zeros(1, 8, 2)
    actions[..., -1] = torch.tensor([-1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    batch = {
        OBS_STATE: torch.randn(1, 2, 3),
        OBS_ENV_STATE: torch.randn(1, 2, 4),
        ACTION: actions,
    }

    loss, metrics = policy(batch)

    torch.testing.assert_close(loss, actions.square().mean())
    assert metrics["physical_loss"] == 0.0
    assert metrics["weighted_physical_loss"] == 0.0


def test_bspline_physical_loss_excludes_entire_padded_chunk():
    policy = FlowPolicy(_make_config(lambda_flow_k=0.0, phy_loss_weight=1.0))
    values = torch.stack((torch.ones(8, 2), torch.full((8, 2), 100.0)))
    action_is_pad = torch.tensor([[False] * 8, [False, False, True, False, False, False, False, False]])

    actual = policy.flow._physical_residual_mean(values, action_is_pad)

    torch.testing.assert_close(actual, torch.tensor(1.0))
