#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
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
"""Flow matching action chunk policy based on the Diffusion Policy architecture."""

import math
from collections import deque
from contextlib import nullcontext
from numbers import Integral

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from ..diffusion.modeling_diffusion import DiffusionConditionalUnet1d, DiffusionRgbEncoder
from ..pretrained import PreTrainedPolicy
from ..utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from .configuration_flow import FlowConfig


def _disabled_autocast_context(device: torch.device):
    """Keep the higher-order autodiff path in full precision."""
    if device.type in {"cuda", "cpu", "xpu", "hpu", "mps"}:
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


def _parameter_averaged_bspline_knots(
    degree: int,
    coefficient_count: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> Tensor:
    """Build a clamped B-spline knot vector from a uniform pseudo-grid."""
    if degree < 1 or coefficient_count <= degree:
        raise ValueError(
            "B-spline knots require degree >= 1 and coefficient_count > degree, "
            f"got degree={degree}, coefficient_count={coefficient_count}."
        )
    pseudo_grid = torch.linspace(0.0, 1.0, coefficient_count, dtype=dtype, device=device)
    internal_values = [
        pseudo_grid[index : index + degree].mean() for index in range(1, coefficient_count - degree)
    ]
    internal = torch.stack(internal_values) if internal_values else torch.empty(0, dtype=dtype, device=device)
    endpoints = torch.zeros(degree + 1, dtype=dtype, device=device)
    return torch.cat((endpoints, internal, torch.ones_like(endpoints)))


def _bspline_basis_matrix(query: Tensor, knots: Tensor, degree: int) -> Tensor:
    """Evaluate every B-spline basis function using Cox--de Boor recursion."""
    if degree < 0:
        raise ValueError(f"B-spline degree must be non-negative, got {degree}.")
    query = query.reshape(-1).to(device=knots.device, dtype=knots.dtype)
    basis = ((query[:, None] >= knots[:-1]) & (query[:, None] < knots[1:])).to(knots.dtype)

    endpoint_rows = query == knots[-1]
    if torch.any(endpoint_rows):
        positive_final_spans = torch.nonzero(
            (knots[:-1] < knots[-1]) & (knots[1:] == knots[-1]), as_tuple=False
        ).flatten()
        if not len(positive_final_spans):
            raise ValueError("B-spline knot vector has no positive-width final span.")
        endpoint_basis = torch.zeros_like(basis)
        endpoint_basis[:, positive_final_spans[-1]] = 1
        basis = torch.where(endpoint_rows[:, None], endpoint_basis, basis)

    for order in range(1, degree + 1):
        num_basis = len(knots) - order - 1
        left_denominator = knots[order : order + num_basis] - knots[:num_basis]
        right_denominator = knots[order + 1 : order + num_basis + 1] - knots[1 : num_basis + 1]
        left_nonzero = left_denominator != 0
        right_nonzero = right_denominator != 0
        safe_left = torch.where(left_nonzero, left_denominator, torch.ones_like(left_denominator))
        safe_right = torch.where(right_nonzero, right_denominator, torch.ones_like(right_denominator))
        left_weight = (query[:, None] - knots[:num_basis]) / safe_left
        right_weight = (knots[order + 1 : order + num_basis + 1] - query[:, None]) / safe_right
        basis = (
            left_weight * basis[:, :num_basis] * left_nonzero
            + right_weight * basis[:, 1 : num_basis + 1] * right_nonzero
        )
    return basis


def _bspline_design_and_derivative(
    horizon: int,
    degree: int,
    coefficient_count: int,
) -> tuple[Tensor, Tensor]:
    """Return the B-spline sample design and analytic d/du basis matrices."""
    knots = _parameter_averaged_bspline_knots(degree, coefficient_count)
    sample_u = torch.linspace(0.0, 1.0, horizon, dtype=torch.float64)
    design = _bspline_basis_matrix(sample_u, knots, degree)
    lower_basis = _bspline_basis_matrix(sample_u, knots, degree - 1)

    left_denominator = knots[degree : degree + coefficient_count] - knots[:coefficient_count]
    right_denominator = knots[degree + 1 : degree + coefficient_count + 1] - knots[1 : coefficient_count + 1]
    left_scale = torch.where(
        left_denominator != 0,
        degree / torch.where(left_denominator != 0, left_denominator, torch.ones_like(left_denominator)),
        torch.zeros_like(left_denominator),
    )
    right_scale = torch.where(
        right_denominator != 0,
        degree / torch.where(right_denominator != 0, right_denominator, torch.ones_like(right_denominator)),
        torch.zeros_like(right_denominator),
    )
    derivative = (
        lower_basis[:, :coefficient_count] * left_scale
        - lower_basis[:, 1 : coefficient_count + 1] * right_scale
    )
    return design, derivative


class FlowPolicy(PreTrainedPolicy):
    """Flow matching policy for action chunk prediction."""

    config_class = FlowConfig
    name = "flow"

    def __init__(
        self,
        config: FlowConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.register_buffer("_train_step", torch.zeros((), dtype=torch.long), persistent=False)

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.flow = FlowModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        return self.flow.parameters()

    def set_train_step(self, step: int) -> None:
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        self._train_step.fill_(step)

    def _maybe_increment_train_step(self) -> None:
        if self.training and torch.is_grad_enabled():
            self._train_step.add_(1)

    def _should_compute_clean_action_metrics(self) -> bool:
        frequency = self.config.clean_action_log_freq
        return (
            self.training
            and torch.is_grad_enabled()
            and frequency > 0
            and (int(self._train_step.item()) + 1) % frequency == 0
        )

    @torch.no_grad()
    def _compute_clean_action_metrics(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """Measure inference-ODE actions against the normalized expert execution window."""
        uses_central_action_stencil = (
            self.config.lambda_flow_k > 0 and self.flow._uses_central_action_stencil()
        )
        data_start = int(uses_central_action_stencil)
        execution_start = self.config.n_obs_steps - 1
        execution_end = execution_start + self.config.n_action_steps
        target = batch[ACTION][
            :, data_start + execution_start : data_start + execution_end
        ]

        # A fixed Gaussian prior makes this sparse metric comparable over time and
        # avoids advancing the RNG stream used by the training objective.
        generator = torch.Generator(device="cpu").manual_seed(0)
        model_dtype = get_dtype_from_parameters(self.flow)
        noise = torch.randn(
            (target.shape[0], self.config.horizon, target.shape[-1]),
            generator=generator,
            dtype=torch.float32,
        ).to(device=target.device, dtype=model_dtype)

        training_states = {module: module.training for module in self.modules()}
        try:
            self.eval()
            predicted = self.flow.generate_actions(batch, noise=noise)
        finally:
            for module, training in training_states.items():
                module.training = training

        squared_error = (predicted.float() - target.float()).square()
        if "action_is_pad" in batch:
            valid_steps = ~batch["action_is_pad"][
                :, data_start + execution_start : data_start + execution_end
            ].to(device=squared_error.device, dtype=torch.bool)
        else:
            valid_steps = torch.ones(
                squared_error.shape[:2], dtype=torch.bool, device=squared_error.device
            )
        valid = valid_steps.unsqueeze(-1).expand_as(squared_error)
        clean_action_mse = (squared_error * valid).sum() / valid.sum().clamp_min(1)
        return {
            "clean_action_mse": clean_action_mse.item(),
            "clean_action_valid_ratio": valid_steps.float().mean().item(),
        }

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Predict a chunk of actions given environment observations.

        Supports two modes:
        - Online (queues populated via select_action): stacks observations from internal queues.
        - Offline (empty queues, e.g. dataloader batch): uses the batch directly.
        """
        queues_populated = any(len(q) > 0 for q in self._queues.values())
        if queues_populated:
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        else:
            batch = dict(batch)
            if self.config.image_features:
                for key in self.config.image_features:
                    if batch[key].ndim == 4:
                        batch[key] = batch[key].unsqueeze(1)
                batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        actions = self.flow.generate_actions(batch, noise=noise)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying flow model. See `DiffusionPolicy.select_action` for the action chunking scheme.
        """
        # NOTE: for offline evaluation, we have action in the batch, so we need to pop it out
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        # NOTE: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            for key in self.config.image_features:
                if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        train_step = int(self._train_step.item()) if self.training else None
        loss, output_dict = self.flow.compute_loss(batch, train_step=train_step)
        if self._should_compute_clean_action_metrics():
            output_dict.update(self._compute_clean_action_metrics(batch))
        self._maybe_increment_train_step()
        return loss, output_dict


class FlowModel(nn.Module):
    def __init__(self, config: FlowConfig):
        super().__init__()
        self.config = config

        # Build observation encoders (depending on which observations are provided).
        global_cond_dim = self.config.robot_state_feature.shape[0]
        self.image_feature_slice: slice | None = None
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                image_feature_dim = encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                image_feature_dim = self.rgb_encoder.feature_dim * num_images
            self.image_feature_slice = slice(global_cond_dim, global_cond_dim + image_feature_dim)
            global_cond_dim += image_feature_dim
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        if not config.gripper_first and config.action_feature.shape[0] < 2:
            raise ValueError("gripper_first=False requires at least one non-gripper action dimension.")

        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)

        if config.compile_model:
            # Compile the U-Net. "reduce-overhead" is preferred for the small-batch repetitive loops
            # common in flow inference.
            self.unet = torch.compile(self.unet, mode=config.compile_mode)

        self.num_inference_steps = config.num_inference_steps

        self._bspline_matrix_cache: dict[tuple[torch.device, torch.dtype], tuple[Tensor, Tensor]] = {}
        self._bspline_analysis_cpu: Tensor | None = None
        self._bspline_derivative_basis_cpu: Tensor | None = None
        if config.interpolation_mode == "bspline":
            degree = config.bspline_degree
            coefficient_count = config.bspline_coe_num
            if (
                isinstance(degree, bool)
                or not isinstance(degree, Integral)
                or isinstance(coefficient_count, bool)
                or not isinstance(coefficient_count, Integral)
            ):
                raise ValueError("bspline_degree and bspline_coe_num must be integers.")
            design, derivative_basis = _bspline_design_and_derivative(
                config.horizon,
                int(degree),
                int(coefficient_count),
            )
            pinv_rtol = 1e-13
            singular_values = torch.linalg.svdvals(design)
            effective_rank = int(torch.count_nonzero(singular_values > singular_values[0] * pinv_rtol).item())
            if effective_rank != coefficient_count:
                raise ValueError(
                    "B-spline collocation matrix is numerically rank deficient for "
                    f"degree={degree}, coefficient_count={coefficient_count}, horizon={config.horizon}."
                )
            analysis = torch.linalg.pinv(design, rtol=pinv_rtol)
            coefficient_identity = torch.eye(coefficient_count, dtype=design.dtype)
            coefficient_recovery_error = torch.max(torch.abs(analysis @ design - coefficient_identity)).item()
            if coefficient_recovery_error > 1e-8:
                raise ValueError(
                    "B-spline collocation matrix is too ill-conditioned for stable coefficient recovery."
                )
            if coefficient_count == config.horizon:
                sample_identity = torch.eye(config.horizon, dtype=design.dtype)
                sample_reconstruction_error = torch.max(torch.abs(design @ analysis - sample_identity)).item()
                if sample_reconstruction_error > 1e-8:
                    raise ValueError("B-spline full-coefficient fit is not numerically lossless.")
            # Keep these as CPU float64 tensors so policy-wide dtype conversions do not
            # degrade the fixed least-squares analysis operator.
            self._bspline_analysis_cpu = analysis
            self._bspline_derivative_basis_cpu = derivative_basis

    # ========= inference  ============
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior.
        sample = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )

        dt = 1.0 / self.num_inference_steps
        for step in range(self.num_inference_steps):
            t = torch.full(
                sample.shape[:1],
                step / self.num_inference_steps,
                dtype=sample.dtype,
                device=sample.device,
            )
            velocity = self.unet(sample, t, global_cond=global_cond)
            sample = sample + dt * velocity

        return sample

    def _prepare_conditioning_steps(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode every observation while preserving its time-step dimension."""
        batch_size, num_observation_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]
        # Extract image features.
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                # Combine batch and sequence dims while rearranging to make the camera index dimension first.
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                # Separate batch and sequence dims back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features_list,
                    "(n b s) ... -> b s (n ...)",
                    b=batch_size,
                    s=num_observation_steps,
                )
            else:
                # Combine batch, sequence, and "which camera" dims before passing to shared encoder.
                img_features = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features,
                    "(b s n) ... -> b s (n ...)",
                    b=batch_size,
                    s=num_observation_steps,
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        return torch.cat(global_cond_feats, dim=-1)

    def _conditioning_window(
        self,
        derivative_conditioning_steps: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Select policy observations from the possibly extended derivative stencil."""
        num_derivative_steps = derivative_conditioning_steps.shape[1]
        if num_derivative_steps == self.config.n_obs_steps:
            conditioning_start = 0
        else:
            mode = self.config.conditioning_derivative_mode
            expected_steps = self.config.n_obs_steps + (2 if mode == "central" else 1)
            if num_derivative_steps != expected_steps:
                raise ValueError(
                    f"{mode} conditioning differences require {expected_steps} observation steps, "
                    f"got {num_derivative_steps}."
                )
            conditioning_start = 1 if mode in {"reverse", "central"} else 0
        conditioning_steps = derivative_conditioning_steps[
            :, conditioning_start : conditioning_start + self.config.n_obs_steps
        ]
        return conditioning_steps, conditioning_steps.flatten(start_dim=1)

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode and flatten the policy's configured observation window."""
        derivative_conditioning_steps = self._prepare_conditioning_steps(batch)
        _, global_cond = self._conditioning_window(derivative_conditioning_steps)
        return global_cond

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)
        }
        """
        batch_size = batch[OBS_STATE].shape[0]

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = self.config.n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def _sample_timesteps(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.config.timestep_sampling_strategy == "uniform":
            return torch.rand(batch_size, device=device, dtype=dtype)
        elif self.config.timestep_sampling_strategy == "beta":
            beta_dist = torch.distributions.Beta(
                self.config.timestep_sampling_alpha, self.config.timestep_sampling_beta
            )
            u = beta_dist.sample((batch_size,)).to(device=device, dtype=dtype)
            return self.config.timestep_sampling_s * (1.0 - u)
        else:
            raise ValueError(f"Unknown timestep strategy: {self.config.timestep_sampling_strategy}")

    def _uses_central_action_stencil(self) -> bool:
        return self.config.interpolation_mode == "dct" and self.config.dct_coe_num == 0

    def _bspline_matrices(self, reference: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if self._bspline_analysis_cpu is None or self._bspline_derivative_basis_cpu is None:
            raise RuntimeError("B-spline matrices requested outside interpolation_mode='bspline'.")
        key = (reference.device, dtype)
        matrices = self._bspline_matrix_cache.get(key)
        if matrices is None:
            matrices = (
                self._bspline_analysis_cpu.to(device=reference.device, dtype=dtype),
                self._bspline_derivative_basis_cpu.to(device=reference.device, dtype=dtype),
            )
            self._bspline_matrix_cache[key] = matrices
        return matrices

    def _sample_frequency(self, batch: dict[str, Tensor], reference: Tensor) -> Tensor:
        fps = batch.get("sample_frequency", batch.get("fps", self.config.sample_frequency))
        fps_tensor = torch.as_tensor(fps, device=reference.device, dtype=reference.dtype)
        if fps_tensor.ndim == 0:
            return fps_tensor
        if fps_tensor.shape[0] != reference.shape[0]:
            raise ValueError(
                f"sample_frequency batch dimension {fps_tensor.shape[0]} does not match "
                f"batch size {reference.shape[0]}."
            )
        return fps_tensor.reshape(reference.shape[0], *([1] * (reference.ndim - 1)))

    def _action_dot(self, action_sequence: Tensor, fps: Tensor) -> Tensor:
        """Differentiate the action chunk in physical time."""
        if self._uses_central_action_stencil():
            expected_frames = self.config.horizon + 2
            if action_sequence.shape[1] != expected_frames:
                raise ValueError(
                    "Central-difference action derivative requires horizon + 2 frames "
                    f"({expected_frames}), got {action_sequence.shape[1]}."
                )
            return (
                action_sequence[:, 2 : self.config.horizon + 2] - action_sequence[:, : self.config.horizon]
            ) * (fps * 0.5)

        if action_sequence.shape[1] != self.config.horizon:
            representation = "B-spline" if self.config.interpolation_mode == "bspline" else "DCT"
            raise ValueError(
                f"{representation} action derivative requires horizon frames "
                f"({self.config.horizon}), got {action_sequence.shape[1]}."
            )

        input_dtype = action_sequence.dtype
        compute_dtype = torch.float32 if input_dtype in {torch.float16, torch.bfloat16} else input_dtype
        with _disabled_autocast_context(action_sequence.device):
            actions = action_sequence.to(dtype=compute_dtype)
            if self.config.interpolation_mode == "bspline":
                analysis, derivative_basis = self._bspline_matrices(actions, compute_dtype)
                coefficients = torch.einsum("mh,bhd->bmd", analysis, actions)
                action_dot = torch.einsum("hm,bmd->bhd", derivative_basis, coefficients)
                action_dot = action_dot * (fps.to(dtype=compute_dtype) / (self.config.horizon - 1))
                return action_dot.to(dtype=input_dtype)

            modes = torch.arange(
                self.config.dct_coe_num,
                device=actions.device,
                dtype=compute_dtype,
            )
            sample_points = (
                torch.arange(self.config.horizon, device=actions.device, dtype=compute_dtype) + 0.5
            )
            alpha = torch.full_like(modes, math.sqrt(2.0 / self.config.horizon))
            alpha[0] = math.sqrt(1.0 / self.config.horizon)
            phase = math.pi * modes[:, None] * sample_points[None, :] / self.config.horizon
            basis = alpha[:, None] * torch.cos(phase)
            coefficients = torch.einsum("bnd,kn->bkd", actions, basis)
            derivative_basis = (
                -alpha[:, None] * (math.pi * modes[:, None] / self.config.horizon) * torch.sin(phase)
            )
            action_dot = torch.einsum("bkd,kh->bhd", coefficients, derivative_basis)
            action_dot = action_dot * fps.to(dtype=compute_dtype)
        return action_dot.to(dtype=input_dtype)

    def _physical_action_dot(self, flow_error: Tensor, fps: Tensor) -> Tensor:
        """Differentiate an H-frame flow error without changing the JVP action stencil."""
        if not self._uses_central_action_stencil():
            return self._action_dot(flow_error, fps)
        if flow_error.shape[1] != self.config.horizon:
            raise ValueError(
                "Physical central-difference derivative requires horizon frames "
                f"({self.config.horizon}), got {flow_error.shape[1]}."
            )
        if self.config.horizon == 1:
            return torch.zeros_like(flow_error)
        if self.config.horizon == 2:
            slope = (flow_error[:, 1:2] - flow_error[:, :1]) * fps
            return slope.expand(-1, 2, -1)

        first = (
            -1.5 * flow_error[:, :1]
            + 2.0 * flow_error[:, 1:2]
            - 0.5 * flow_error[:, 2:3]
        ) * fps
        interior = (flow_error[:, 2:] - flow_error[:, :-2]) * (fps * 0.5)
        last = (
            0.5 * flow_error[:, -3:-2]
            - 2.0 * flow_error[:, -2:-1]
            + 1.5 * flow_error[:, -1:]
        ) * fps
        return torch.cat((first, interior, last), dim=1)

    def _conditioning_dot(
        self,
        conditioning_steps: Tensor,
        derivative_conditioning_steps: Tensor,
        fps: Tensor,
    ) -> Tensor:
        """Differentiate encoded conditioning with the configured finite-difference stencil."""
        mode = self.config.conditioning_derivative_mode
        image_only = self.config.image_only_condition_jvp
        num_conditioning_steps = conditioning_steps.shape[1]
        expected_steps = num_conditioning_steps + (2 if mode == "central" else 1)
        if derivative_conditioning_steps.shape[1] != expected_steps:
            raise ValueError(
                f"{mode} conditioning differences for {num_conditioning_steps} policy observations "
                f"require {expected_steps} derivative observations, got "
                f"{derivative_conditioning_steps.shape[1]}."
            )

        conditioning_start = 1 if mode in {"reverse", "central"} else 0
        current_steps = derivative_conditioning_steps[
            :, conditioning_start : conditioning_start + num_conditioning_steps
        ]
        conditioning_fps = fps.reshape(fps.shape[0], 1, 1) if fps.ndim > 0 else fps
        if mode == "reverse":
            previous_steps = derivative_conditioning_steps[
                :, conditioning_start - 1 : conditioning_start + num_conditioning_steps - 1
            ]
            conditioning_dot_steps = (current_steps - previous_steps) * conditioning_fps
        elif mode == "forward":
            next_steps = derivative_conditioning_steps[
                :, conditioning_start + 1 : conditioning_start + num_conditioning_steps + 1
            ]
            conditioning_dot_steps = (next_steps - current_steps) * conditioning_fps
        else:
            previous_steps = derivative_conditioning_steps[
                :, conditioning_start - 1 : conditioning_start + num_conditioning_steps - 1
            ]
            next_steps = derivative_conditioning_steps[
                :, conditioning_start + 1 : conditioning_start + num_conditioning_steps + 1
            ]
            conditioning_dot_steps = (next_steps - previous_steps) * (conditioning_fps * 0.5)

        if image_only:
            if self.image_feature_slice is None:
                raise ValueError("image_only_condition_jvp=True requires at least one image feature.")
            image_mask = conditioning_dot_steps.new_zeros(conditioning_dot_steps.shape[-1])
            image_mask[self.image_feature_slice] = 1
            conditioning_dot_steps = conditioning_dot_steps * image_mask

        return conditioning_dot_steps.flatten(start_dim=1)

    def _kinematic_action_mask(self, reference: Tensor) -> Tensor:
        mask = torch.ones(
            self.config.action_feature.shape[0],
            dtype=torch.bool,
            device=reference.device,
        )
        if not self.config.gripper_first:
            mask[-1] = False
        return mask

    def _kinematic_valid_mask(
        self,
        batch: dict[str, Tensor],
        data: Tensor,
        derivative_conditioning_steps: Tensor,
    ) -> Tensor:
        valid = torch.ones(data.shape[:2], dtype=torch.bool, device=data.device)
        if "action_is_pad" in batch:
            action_is_pad = batch["action_is_pad"].to(device=data.device, dtype=torch.bool)
            expected_frames = self.config.horizon + 2 * int(self._uses_central_action_stencil())
            expected_shape = (data.shape[0], expected_frames)
            if action_is_pad.shape != expected_shape:
                raise ValueError(
                    f"action_is_pad must have shape {expected_shape}, got {tuple(action_is_pad.shape)}."
                )
            if self._uses_central_action_stencil():
                action_valid = ~action_is_pad[:, 1 : self.config.horizon + 1]
                action_valid &= ~action_is_pad[:, : self.config.horizon]
                action_valid &= ~action_is_pad[:, 2 : self.config.horizon + 2]
            elif self.config.interpolation_mode == "bspline":
                action_valid = (~action_is_pad.any(dim=1))[:, None].expand(-1, self.config.horizon)
            else:
                action_valid = ~action_is_pad[:, : self.config.horizon]
            valid &= action_valid

        observation_pad_key = f"{OBS_STATE}_is_pad"
        if observation_pad_key in batch:
            observation_is_pad = batch[observation_pad_key].to(device=data.device, dtype=torch.bool)
            if observation_is_pad.shape[:2] != derivative_conditioning_steps.shape[:2]:
                raise ValueError(
                    f"{observation_pad_key} shape {tuple(observation_is_pad.shape)} does not match "
                    f"conditioning steps {tuple(derivative_conditioning_steps.shape[:2])}."
                )
            valid &= (~observation_is_pad.any(dim=1))[:, None]
        return valid

    def _compute_kinematic_loss(
        self,
        batch: dict[str, Tensor],
        data: Tensor,
        action_sequence: Tensor,
        x_t: Tensor,
        t: Tensor,
        global_cond: Tensor,
        conditioning_steps: Tensor,
        derivative_conditioning_steps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        fps = self._sample_frequency(batch, data)
        action_mask = self._kinematic_action_mask(data)
        action_dot = self._action_dot(action_sequence, fps)
        action_dot = action_dot * action_mask.to(dtype=action_dot.dtype)
        conditioning_dot = self._conditioning_dot(
            conditioning_steps,
            derivative_conditioning_steps,
            fps,
        )

        def vector_field_condition(condition: Tensor) -> Tensor:
            return self.unet(x_t, t, global_cond=condition)

        with _disabled_autocast_context(x_t.device):
            # Replay the same dropout realization for both JVP terms
            # so their sum differentiates one stochastic vector field.
            jvp_rng_devices = [] if x_t.device.type == "cpu" else [x_t.device]
            with torch.random.fork_rng(devices=jvp_rng_devices, device_type=x_t.device.type):
                predicted_velocity, conditioning_jvp = torch.func.jvp(
                    vector_field_condition,
                    (global_cond,),
                    (conditioning_dot,),
                )

            def vector_field_action(actions: Tensor) -> Tensor:
                return self.unet(actions, t, global_cond=global_cond)

            _, action_jvp = torch.func.jvp(
                vector_field_action,
                (x_t,),
                (action_dot,),
            )
            residual = (1 - t.view(-1, 1, 1)) * (
                conditioning_jvp + t.view(-1, 1, 1) * action_jvp - action_dot
            )

        valid = self._kinematic_valid_mask(batch, data, derivative_conditioning_steps)
        valid_float = valid.to(dtype=residual.dtype)
        loss_per_step = residual[..., action_mask].square().mean(dim=-1)
        num_valid = valid_float.sum()
        kinematic_loss = (loss_per_step * valid_float).sum() / num_valid.clamp_min(1)
        valid_ratio = valid_float.mean()

        def masked_rms(value: Tensor) -> Tensor:
            squared_per_step = value.detach()[..., action_mask].float().square().mean(dim=-1)
            return torch.sqrt((squared_per_step * valid_float.float()).sum() / num_valid.float().clamp_min(1))

        metrics = {
            "kinematic_state_jvp_rms": masked_rms(conditioning_jvp),
            "kinematic_action_jvp_rms": masked_rms(action_jvp),
        }
        return predicted_velocity, kinematic_loss, valid_ratio, metrics

    def _flow_loss(
        self,
        predicted_velocity: Tensor,
        target_velocity: Tensor,
        action_is_pad: Tensor | None,
    ) -> Tensor:
        loss = F.mse_loss(predicted_velocity, target_velocity, reduction="none")
        if self.config.do_mask_loss_for_padding:
            if action_is_pad is None:
                raise ValueError("You need to provide 'action_is_pad' when do_mask_loss_for_padding=True.")
            mask = ~action_is_pad.to(device=loss.device, dtype=torch.bool).unsqueeze(-1)
            num_valid = mask.sum() * loss.shape[-1]
            return (loss * mask).sum() / num_valid.clamp_min(1)
        return loss.mean()

    def _masked_action_mean(
        self,
        values: Tensor,
        action_is_pad: Tensor | None,
    ) -> Tensor:
        if self.config.do_mask_loss_for_padding and action_is_pad is not None:
            expected_shape = values.shape[:2]
            if action_is_pad.shape != expected_shape:
                raise ValueError(
                    f"Physical action_is_pad must have shape {expected_shape}, "
                    f"got {tuple(action_is_pad.shape)}."
                )
            mask = ~action_is_pad.to(device=values.device, dtype=torch.bool).unsqueeze(-1)
            num_valid = mask.sum() * values.shape[-1]
            return (values * mask).sum() / num_valid.clamp_min(1)
        return values.mean()

    def _physical_residual_mean(
        self,
        values: Tensor,
        action_is_pad: Tensor | None,
    ) -> Tensor:
        """Average physical residuals over valid continuous-action chunks."""
        if self.config.interpolation_mode != "bspline" or action_is_pad is None:
            return self._masked_action_mean(values, action_is_pad)

        expected_shape = values.shape[:2]
        if action_is_pad.shape != expected_shape:
            raise ValueError(
                f"Physical action_is_pad must have shape {expected_shape}, "
                f"got {tuple(action_is_pad.shape)}."
            )

        # A B-spline derivative at every query point depends on the full fitted
        # action chunk, so one padded action invalidates the complete chunk.
        valid_chunks = ~action_is_pad.to(device=values.device, dtype=torch.bool).any(dim=1)
        valid = valid_chunks[:, None, None].to(dtype=values.dtype)
        num_valid_values = valid.sum() * values.shape[1] * values.shape[2]
        return (values * valid).sum() / num_valid_values.clamp_min(1)

    def _effective_lambda_flow_k(self, train_step: int | None) -> float:
        if train_step is not None and train_step < self.config.pre_train_steps:
            return 0.0
        return float(self.config.lambda_flow_k)

    def compute_loss(
        self,
        batch: dict[str, Tensor],
        train_step: int | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({OBS_STATE, ACTION})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        uses_central_action_stencil = self.config.lambda_flow_k > 0 and self._uses_central_action_stencil()
        expected_action_frames = self.config.horizon + 2 * int(uses_central_action_stencil)
        action_sequence = batch[ACTION]
        expected_action_shape = (
            action_sequence.shape[0],
            expected_action_frames,
            self.config.action_feature.shape[0],
        )
        if action_sequence.shape != expected_action_shape:
            raise ValueError(
                f"Flow actions must have shape {expected_action_shape}, got {tuple(action_sequence.shape)}."
            )
        if "action_is_pad" in batch and batch["action_is_pad"].shape != expected_action_shape[:2]:
            raise ValueError(
                f"action_is_pad must have shape {expected_action_shape[:2]}, "
                f"got {tuple(batch['action_is_pad'].shape)}."
            )

        derivative_conditioning_steps = self._prepare_conditioning_steps(batch)
        conditioning_steps, global_cond = self._conditioning_window(derivative_conditioning_steps)

        # Flow matching.
        data_start = int(uses_central_action_stencil)
        trajectory = action_sequence[:, data_start : data_start + self.config.horizon]
        noise = torch.randn_like(trajectory)
        t = self._sample_timesteps(trajectory.shape[0], trajectory.device, trajectory.dtype)
        t_expanded = t.view(-1, 1, 1)

        x_t = t_expanded * trajectory + (1 - (1 - self.config.sigma_min) * t_expanded) * noise
        target_velocity = trajectory - (1 - self.config.sigma_min) * noise

        effective_lambda_flow_k = self._effective_lambda_flow_k(train_step)
        if effective_lambda_flow_k > 0 and torch.is_grad_enabled():
            pred_velocity, kinematic_loss, valid_ratio, jvp_metrics = self._compute_kinematic_loss(
                batch=batch,
                data=trajectory,
                action_sequence=action_sequence,
                x_t=x_t,
                t=t,
                global_cond=global_cond,
                conditioning_steps=conditioning_steps,
                derivative_conditioning_steps=derivative_conditioning_steps,
            )
        else:
            pred_velocity = self.unet(x_t, t, global_cond=global_cond)
            kinematic_loss = trajectory.new_zeros(())
            valid_ratio = trajectory.new_zeros(())
            jvp_metrics = {}

        flow_action_is_pad = None
        if "action_is_pad" in batch:
            flow_action_is_pad = batch["action_is_pad"][:, data_start : data_start + self.config.horizon]
        flow_loss = self._flow_loss(pred_velocity, target_velocity, flow_action_is_pad)
        phy_loss_weight = float(self.config.phy_loss_weight)
        flow_error = pred_velocity - target_velocity
        physical_flow_error = flow_error if phy_loss_weight > 0 else flow_error.detach()
        fps = self._sample_frequency(batch, trajectory)
        physical_grad_context = nullcontext() if phy_loss_weight > 0 else torch.no_grad()
        with physical_grad_context:
            g_flow_error = self._physical_action_dot(physical_flow_error, fps)
            physical_residual = t_expanded * (1 - t_expanded) * g_flow_error
            physical_action_mask = self._kinematic_action_mask(trajectory)
            physical_loss = self._physical_residual_mean(
                physical_residual[..., physical_action_mask].square(),
                flow_action_is_pad,
            )
        weighted_physical_loss = (
            phy_loss_weight * physical_loss if phy_loss_weight > 0 else trajectory.new_zeros(())
        )
        total_loss = flow_loss + effective_lambda_flow_k * kinematic_loss + weighted_physical_loss

        output_dict = {
            "flow_loss": flow_loss.detach().float().item(),
            "kinematic_loss": kinematic_loss.detach().float().item(),
            "lambda_flow_k": effective_lambda_flow_k,
            "lambda_flow_k_config": float(self.config.lambda_flow_k),
            "pre_train_steps": int(self.config.pre_train_steps),
            "train_step": int(train_step) if train_step is not None else -1,
            "gripper_first": float(self.config.gripper_first),
            "physical_loss": physical_loss.detach().float().item(),
            "phy_loss_weight": phy_loss_weight,
            "weighted_physical_loss": weighted_physical_loss.detach().float().item(),
            "kinematic_valid_ratio": valid_ratio.detach().float().item(),
            "total_loss": total_loss.detach().float().item(),
        }
        output_dict.update({name: value.detach().float().item() for name, value in jvp_metrics.items()})
        return total_loss, output_dict
