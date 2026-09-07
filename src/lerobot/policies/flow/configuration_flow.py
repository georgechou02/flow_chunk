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
import math
from dataclasses import dataclass, field
from numbers import Integral

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamConfig
from lerobot.optim.schedulers import CosineAnnealingWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("flow")
@dataclass
class FlowConfig(PreTrainedConfig):
    """Configuration class for FlowPolicy.

    This policy follows the same action chunking and observation conditioning structure as DiffusionPolicy,
    but trains a flow matching velocity field and samples actions with Euler ODE integration.
    """

    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 64
    n_action_steps: int = 32

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # The original Diffusion Policy implementation doesn't sample frames for the last 7 steps,
    # which avoids excessive padding and leads to improved training results.
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # Architecture / modeling.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    use_group_norm: bool = False
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = True
    # Unet.
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # Flow matching.
    num_inference_steps: int = 100
    timestep_sampling_strategy: str = "uniform"
    timestep_sampling_alpha: float = 1.5
    timestep_sampling_beta: float = 1.0
    timestep_sampling_s: float = 0.999
    sigma_min: float = 0.0

    # First-order (kinematic) supervision always includes conditioning and action-input JVPs.
    lambda_flow_k: float = 0.01
    # Number of initial global optimizer steps trained with the kinematic
    # weight forced to zero. The configured non-zero lambda is retained so the
    # dataloader requests a stable derivative stencil across the switch.
    pre_train_steps: int = 0
    phy_loss_weight: float = 0.0  # Weight for the value-level physical residual loss.
    gripper_first: bool = True  # Include the final gripper action in first-order supervision.
    sample_frequency: float = 10.0
    interpolation_mode: str = "bspline"
    dct_coe_num: int = 48
    bspline_degree: int = 2
    bspline_coe_num: int = 48
    conditioning_derivative_mode: str = "central"
    image_only_condition_jvp: bool = False  # Keep only image features in the conditioning JVP tangent.

    # Optimization
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    # Loss computation
    do_mask_loss_for_padding: bool = False
    # Periodically solve the inference ODE and log its action-space MSE. Set to 0 to disable.
    clean_action_log_freq: int = 100

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        if self.resize_shape is not None and (
            len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)
        ):
            raise ValueError(f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}.")
        if not (0 < self.crop_ratio <= 1.0):
            raise ValueError(f"`crop_ratio` must be in (0, 1]. Got {self.crop_ratio}.")

        if self.resize_shape is not None:
            if self.crop_ratio < 1.0:
                self.crop_shape = (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
            else:
                # Explicitly disable cropping for resize+ratio path when crop_ratio == 1.0.
                self.crop_shape = None
        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(f"`crop_shape` must have positive dimensions. Got {self.crop_shape}.")

        # Check that the horizon size and U-Net downsampling is compatible.
        # U-Net downsamples by 2 with each stage.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )

        if self.num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive. Got {self.num_inference_steps}.")
        if (
            isinstance(self.clean_action_log_freq, bool)
            or not isinstance(self.clean_action_log_freq, Integral)
            or self.clean_action_log_freq < 0
        ):
            raise ValueError(
                "`clean_action_log_freq` must be a non-negative integer. "
                f"Got {self.clean_action_log_freq}."
            )

        supported_timestep_sampling_strategies = ["uniform", "beta"]
        if self.timestep_sampling_strategy not in supported_timestep_sampling_strategies:
            raise ValueError(
                "`timestep_sampling_strategy` must be one of "
                f"{supported_timestep_sampling_strategies}. Got {self.timestep_sampling_strategy}."
            )
        if not (0.0 <= self.sigma_min <= 1.0):
            raise ValueError(f"`sigma_min` must be in [0, 1]. Got {self.sigma_min}.")
        if not math.isfinite(self.lambda_flow_k) or self.lambda_flow_k < 0:
            raise ValueError(f"`lambda_flow_k` must be finite and non-negative. Got {self.lambda_flow_k}.")
        if (
            isinstance(self.pre_train_steps, bool)
            or not isinstance(self.pre_train_steps, Integral)
            or self.pre_train_steps < 0
        ):
            raise ValueError(
                "`pre_train_steps` must be a non-negative integer. "
                f"Got {self.pre_train_steps}."
            )
        if not math.isfinite(self.phy_loss_weight) or self.phy_loss_weight < 0:
            raise ValueError(
                f"`phy_loss_weight` must be finite and non-negative. Got {self.phy_loss_weight}."
            )
        if not math.isfinite(self.sample_frequency) or self.sample_frequency <= 0:
            raise ValueError(f"`sample_frequency` must be finite and positive. Got {self.sample_frequency}.")

        self.interpolation_mode = self.interpolation_mode.lower()
        if self.interpolation_mode not in {"dct", "bspline"}:
            raise ValueError(
                f"`interpolation_mode` must be 'dct' or 'bspline'. Got {self.interpolation_mode}."
            )
        if (
            isinstance(self.dct_coe_num, bool)
            or not isinstance(self.dct_coe_num, Integral)
            or not 0 <= self.dct_coe_num <= self.horizon
        ):
            raise ValueError(
                "`dct_coe_num` must be an integer in [0, horizon]. "
                f"Got {self.dct_coe_num} for horizon={self.horizon}."
            )
        if self.interpolation_mode == "bspline":
            if (
                isinstance(self.bspline_degree, bool)
                or not isinstance(self.bspline_degree, Integral)
                or isinstance(self.bspline_coe_num, bool)
                or not isinstance(self.bspline_coe_num, Integral)
            ):
                raise ValueError("`bspline_degree` and `bspline_coe_num` must be integers.")
            if not 2 <= self.bspline_degree < self.bspline_coe_num <= self.horizon:
                raise ValueError(
                    "B-spline interpolation requires 2 <= bspline_degree < bspline_coe_num <= horizon. "
                    f"Got degree={self.bspline_degree}, coefficients={self.bspline_coe_num}, "
                    f"horizon={self.horizon}."
                )

        self.conditioning_derivative_mode = self.conditioning_derivative_mode.lower()
        if self.conditioning_derivative_mode not in {"reverse", "forward", "central"}:
            raise ValueError(
                "`conditioning_derivative_mode` must be 'reverse', 'forward', or 'central'. "
                f"Got {self.conditioning_derivative_mode}."
            )
        if not (0.0 < self.timestep_sampling_s <= 1.0):
            raise ValueError(f"`timestep_sampling_s` must be in (0, 1]. Got {self.timestep_sampling_s}.")
        if self.timestep_sampling_alpha <= 0:
            raise ValueError("`timestep_sampling_alpha` must be positive.")
        if self.timestep_sampling_beta <= 0:
            raise ValueError("`timestep_sampling_beta` must be positive.")
        if self.scheduler_name != "cosine":
            raise ValueError(f"`scheduler_name` must be 'cosine'. Got {self.scheduler_name}.")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> CosineAnnealingWithWarmupSchedulerConfig:
        return CosineAnnealingWithWarmupSchedulerConfig(num_warmup_steps=self.scheduler_warmup_steps)

    def validate_features(self) -> None:
        if self.robot_state_feature is None:
            raise ValueError("You must provide 'observation.state' among the inputs.")

        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if self.resize_shape is None and self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the image shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for `{key}`."
                    )

        # Check that all input images have the same shape.
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )

    @property
    def observation_delta_indices(self) -> list:
        indices = list(range(1 - self.n_obs_steps, 1))
        if self.lambda_flow_k > 0:
            if self.conditioning_derivative_mode in {"reverse", "central"}:
                indices.insert(0, -self.n_obs_steps)
            if self.conditioning_derivative_mode in {"forward", "central"}:
                indices.append(1)
        return indices

    @property
    def action_delta_indices(self) -> list:
        start = 1 - self.n_obs_steps
        uses_central_difference_stencil = (
            self.lambda_flow_k > 0 and self.interpolation_mode == "dct" and self.dct_coe_num == 0
        )
        if uses_central_difference_stencil:
            return list(range(start - 1, start + self.horizon + 1))
        return list(range(start, start + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
