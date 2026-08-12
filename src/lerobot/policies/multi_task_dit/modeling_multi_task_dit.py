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

"""Multi-Task Diffusion Transformer (DiT) Policy

Transformer-based diffusion policy for multi-task robot learning with text and vision conditioning.
Supports both diffusion and flow matching objectives for action generation.

References:
- https://arxiv.org/abs/2507.05331
- https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/
- https://brysonkjones.substack.com/p/dissecting-and-open-sourcing-multitask-diffusion-transformer-policy
"""

import math
from collections import deque
from contextlib import nullcontext
from typing import TYPE_CHECKING

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:
    SDPBackend = None
    sdpa_kernel = None

from lerobot.utils.import_utils import _diffusers_available, _transformers_available, require_package

from .configuration_multi_task_dit import MultiTaskDiTConfig

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers import CLIPTextModel, CLIPVisionModel
else:
    CLIPTextModel = None
    CLIPVisionModel = None

if TYPE_CHECKING or _diffusers_available:
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
else:
    DDIMScheduler = None
    DDPMScheduler = None
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from ..pretrained import PreTrainedPolicy
from ..utils import get_output_shape, populate_queues


def _sdpa_math_kernel_context():
    """Force math SDPA for higher-order AD paths that are unsupported by flash kernels."""
    if sdpa_kernel is not None and SDPBackend is not None:
        return sdpa_kernel(SDPBackend.MATH)
    if torch.cuda.is_available() and hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True,
        )
    return nullcontext()


def _disabled_autocast_context(device: torch.device):
    """Run higher-order AD in full precision to avoid AMP dtype mismatches during backward."""
    if device.type in {"cuda", "cpu", "xpu", "hpu", "mps"}:
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


# -- Policy --


class MultiTaskDiTPolicy(PreTrainedPolicy):
    config_class = MultiTaskDiTConfig
    name = "multi_task_dit"

    def __init__(self, config: MultiTaskDiTConfig, **kwargs):
        if config.vision_encoder_type == "clip" or not config.single_task:
            require_package("transformers", extra="multi_task_dit")
        require_package("diffusers", extra="multi_task_dit")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.register_buffer("_train_step", torch.zeros((), dtype=torch.long), persistent=True)

        self._queues = None

        self.observation_encoder = ObservationEncoder(config)
        conditioning_dim = self.observation_encoder.conditioning_dim
        self.noise_predictor = DiffusionTransformer(config, conditioning_dim=conditioning_dim)

        action_dim = config.action_feature.shape[0]
        horizon = config.horizon

        if config.is_diffusion:
            self.objective = DiffusionObjective(
                config,
                action_dim=action_dim,
                horizon=horizon,
                do_mask_loss_for_padding=config.do_mask_loss_for_padding,
            )
        elif config.is_flow_matching:
            self.objective = FlowMatchingObjective(
                config,
                action_dim=action_dim,
                horizon=horizon,
                do_mask_loss_for_padding=config.do_mask_loss_for_padding,
                image_feature_slice=self.observation_encoder.image_feature_slice,
            )
        else:
            raise ValueError(f"Unsupported objective: {config.objective}")

        self.reset()

    def get_optim_params(self) -> list:
        """Returns parameter groups with different learning rates for vision vs non-vision parameters"""
        non_vision_params = []
        vision_encoder_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if name.startswith("observation_encoder.vision_encoder.") or name.startswith(
                "observation_encoder.vision_encoders."
            ):
                vision_encoder_params.append(param)
            else:
                non_vision_params.append(param)

        return [
            {"params": non_vision_params},
            {
                "params": vision_encoder_params,
                "lr": self.config.optimizer_lr * self.config.vision_encoder_lr_multiplier,
            },
        ]

    def set_train_step(self, step: int) -> None:
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        self._train_step.fill_(step)

    def _maybe_increment_train_step(self) -> None:
        if self.training and torch.is_grad_enabled():
            self._train_step.add_(1)

    def _generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        conditioning_vec = self.observation_encoder.encode(batch)
        actions = self.objective.conditional_sample(self.noise_predictor, batch_size, conditioning_vec)

        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]
        return actions

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations"""
        self.eval()

        for k in batch:
            if k in self._queues:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        actions = self._generate_actions(batch)
        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Prepare batch by stacking image features if needed."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy to avoid modifying original
            camera_images = [batch[key] for key in self.config.image_features]
            first_shape = camera_images[0].shape
            if any(image.shape != first_shape for image in camera_images[1:]):
                if not self.observation_encoder.do_resize:
                    raise ValueError("Mixed-resolution camera inputs require image_resize_shape.")
                resized_camera_images = []
                for image in camera_images:
                    leading_shape = image.shape[:-3]
                    flattened_image = image.reshape(-1, *image.shape[-3:])
                    resized_image = self.observation_encoder.resize(flattened_image)
                    resized_camera_images.append(
                        resized_image.reshape(*leading_shape, *resized_image.shape[-3:])
                    )
                camera_images = resized_camera_images
            batch[OBS_IMAGES] = torch.stack(camera_images, dim=-4)

        return batch

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations"""
        if ACTION in batch:
            batch = dict(batch)  # shallow copy to avoid modifying original
            batch.pop(ACTION)

        batch = self._prepare_batch(batch)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Run the batch through the model and compute the loss for training"""
        batch = self._prepare_batch(batch)

        derivative_conditioning_steps = self.observation_encoder.encode_steps(batch)
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
        conditioning_vec = conditioning_steps.flatten(start_dim=1)

        train_step = int(self._train_step.item()) if self.training else None
        if self.config.is_flow_matching:
            loss, output_dict = self.objective.compute_loss(
                self.noise_predictor,
                batch,
                conditioning_vec,
                conditioning_steps=conditioning_steps,
                derivative_conditioning_steps=derivative_conditioning_steps,
                train_step=train_step,
            )
            self._maybe_increment_train_step()
            return loss, output_dict

        loss = self.objective.compute_loss(self.noise_predictor, batch, conditioning_vec)
        self._maybe_increment_train_step()
        return loss, None


# -- Observation Encoders --


class CLIPVisionEncoder(nn.Module):
    """CLIP vision encoder using the CLS token for global image representation."""

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.model = CLIPVisionModel.from_pretrained(self.model_name)
        self.num_non_spatial_tokens = 1
        self.embed_dim = self.model.config.hidden_size

    def forward(self, x: Tensor) -> Tensor:
        """Encode RGB image to CLS token."""
        outputs = self.model(pixel_values=x, output_hidden_states=False)
        cls_token = outputs.last_hidden_state[:, 0]
        b, embed_dim = cls_token.shape
        return cls_token.reshape(b, embed_dim, 1, 1)

    def get_output_shape(self) -> tuple:
        return (self.embed_dim, 1, 1)


class SpatialSoftmax(nn.Module):
    """Spatial soft argmax that converts feature maps into keypoint coordinates."""

    def __init__(self, input_shape: tuple[int, int, int], num_kp: int | None = None):
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be (C, H, W), got {input_shape}")

        self._in_c, self._in_h, self._in_w = input_shape
        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self._in_h),
            torch.linspace(-1.0, 1.0, self._in_w),
            indexing="ij",
        )
        pos_grid = torch.stack([pos_x.reshape(-1), pos_y.reshape(-1)], dim=1)
        self.register_buffer("pos_grid", pos_grid)

    def forward(self, features: Tensor) -> Tensor:
        if self.nets is not None:
            features = self.nets(features)

        features = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)
        expected_xy = attention @ self.pos_grid
        return expected_xy.view(-1, self._out_c, 2)


class ResNetVisionEncoder(nn.Module):
    """Lightweight ResNet vision encoder for MultiTaskDiT conditioning."""

    def __init__(self, config: MultiTaskDiTConfig):
        super().__init__()
        backbone_factory = getattr(torchvision.models, config.vision_backbone)
        backbone_model = backbone_factory(weights=config.pretrained_backbone_weights)
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))

        images_shape = next(iter(config.image_features.values())).shape
        if config.image_crop_shape is not None:
            dummy_shape_h_w = config.image_crop_shape
        elif config.image_resize_shape is not None:
            dummy_shape_h_w = config.image_resize_shape
        else:
            dummy_shape_h_w = images_shape[1:]

        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(self.feature_dim, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        features = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        features = self.relu(self.out(features))
        b = features.shape[0]
        return features.reshape(b, self.feature_dim, 1, 1)

    def get_output_shape(self) -> tuple[int, int, int]:
        return (self.feature_dim, 1, 1)


class CLIPTextEncoder(nn.Module):
    """CLIP text encoder with frozen weights and a learnable projection layer.

    Accepts pre-tokenized inputs (input_ids and attention_mask) from the processor pipeline. See the processor
    pipeline to see how the tokenization is handled.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch16", projection_dim: int = 512):
        super().__init__()
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_embed_dim = self.text_encoder.config.hidden_size
        self.projection = nn.Linear(self.text_embed_dim, projection_dim)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Encode pre-tokenized text to feature vectors."""
        # Ensure inputs are on the same device as the model
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            clip_features = outputs.pooler_output

        return self.projection(clip_features)


class ObservationEncoder(nn.Module):
    """Handles all observation processing for the conditioning vector."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.image_resize_shape is not None:
            self.do_resize = True
            self.resize = torchvision.transforms.Resize(
                size=config.image_resize_shape,
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )
        else:
            self.do_resize = False

        if config.image_crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.image_crop_shape)
            if config.image_crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.image_crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        if config.image_features:
            self.num_cameras = len(config.image_features)
            self.camera_names = list(config.image_features.keys())

            if config.use_separate_rgb_encoder_per_camera:
                self.vision_encoders = nn.ModuleList(
                    [self._make_vision_encoder() for _ in self.camera_names]
                )
                self.vision_encoder = None
            else:
                self.vision_encoder = self._make_vision_encoder()
                self.vision_encoders = None
        else:
            self.vision_encoder = None
            self.vision_encoders = None
            self.camera_names = []
            self.num_cameras = 0

        if hasattr(config, "robot_state_feature") and config.robot_state_feature:
            self.robot_state_dim = config.robot_state_feature.shape[0]
        else:
            self.robot_state_dim = 0

        self.text_encoder = None
        if not config.single_task:
            self.text_encoder = CLIPTextEncoder(
                model_name=config.text_encoder_name,
                projection_dim=config.hidden_dim,
            )

        total_dim = self.robot_state_dim
        image_feature_dim = 0
        if self.vision_encoder is not None or self.vision_encoders is not None:
            encoder_to_check = self.vision_encoder or next(iter(self.vision_encoders))
            c, h, w = encoder_to_check.get_output_shape()
            image_feature_dim = c * h * w * self.num_cameras
            total_dim += image_feature_dim
        self.image_feature_slice = (
            slice(self.robot_state_dim, self.robot_state_dim + image_feature_dim)
            if image_feature_dim > 0
            else None
        )
        if self.text_encoder is not None:
            total_dim += config.hidden_dim
        self.conditioning_dim = total_dim * config.n_obs_steps

    def _make_vision_encoder(self) -> nn.Module:
        if self.config.vision_encoder_type == "clip":
            return CLIPVisionEncoder(model_name=self.config.vision_encoder_name)
        if self.config.vision_encoder_type == "resnet":
            return ResNetVisionEncoder(self.config)
        raise ValueError(f"Unsupported vision_encoder_type: {self.config.vision_encoder_type}")

    def _apply_preprocessing(self, images: Tensor) -> Tensor:
        if self.do_resize:
            images = self.resize(images)
        if self.do_crop:
            images = self.maybe_random_crop(images) if self.training else self.center_crop(images)
        return images

    def encode_steps(self, batch: dict) -> Tensor:
        """Encode observations to per-step conditioning features."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        conditioning_feats = []

        conditioning_feats.append(batch[OBS_STATE])

        if self.vision_encoder is not None or self.vision_encoders is not None:
            images = batch[OBS_IMAGES]

            if len(images.shape) == 5:
                images = images.unsqueeze(1)

            if self.config.use_separate_rgb_encoder_per_camera:
                camera_features = []
                for cam_idx in range(self.num_cameras):
                    cam_images = images[:, :, cam_idx]
                    cam_images_flat = einops.rearrange(cam_images, "b s c h w -> (b s) c h w")
                    cam_images_flat = self._apply_preprocessing(cam_images_flat)
                    cam_features = self.vision_encoders[cam_idx](cam_images_flat)
                    cam_visual_features = cam_features.flatten(start_dim=1)
                    cam_features_reshaped = einops.rearrange(
                        cam_visual_features, "(b s) f -> b s f", b=batch_size, s=n_obs_steps
                    )
                    camera_features.append(cam_features_reshaped)
                img_features = torch.cat(camera_features, dim=-1)
                conditioning_feats.append(img_features)
            else:
                images_flat = einops.rearrange(images, "b s n c h w -> (b s n) c h w")
                images_flat = self._apply_preprocessing(images_flat)
                visual_features = self.vision_encoder(images_flat).flatten(start_dim=1)
                img_features = einops.rearrange(
                    visual_features, "(b s n) f -> b s (n f)", b=batch_size, s=n_obs_steps, n=self.num_cameras
                )
                conditioning_feats.append(img_features)

        if self.text_encoder is not None:
            input_ids = batch[OBS_LANGUAGE_TOKENS]  # [batch_size, seq_length]
            attention_mask = batch[OBS_LANGUAGE_ATTENTION_MASK]  # [batch_size, seq_length]

            text_features = self.text_encoder(input_ids, attention_mask)

            text_features = text_features.unsqueeze(1).expand(-1, n_obs_steps, -1)
            conditioning_feats.append(text_features)

        combined_features = torch.cat(conditioning_feats, dim=-1)
        return combined_features

    def encode(self, batch: dict) -> Tensor:
        """Encode observations to flattened conditioning vector format."""
        return self.encode_steps(batch).flatten(start_dim=1)


# -- Transformer Components --


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Modulate input with shift and scale for AdaLN-Zero."""
    return x * (1 + scale) + shift


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for transformers."""

    def __init__(self, head_dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._precompute_cache(max_seq_len)

    def _precompute_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("_sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def _rotate_half(self, x: Tensor) -> Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        seq_len = q.shape[2]
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}.")

        cos = self._cos_cached[:, :, :seq_len, :].to(q.dtype)
        sin = self._sin_cached[:, :, :seq_len, :].to(q.dtype)

        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)
        return q_rotated, k_rotated


class RoPEAttention(nn.Module):
    """Multi-head self-attention with Rotary Position Embedding (RoPE)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 512,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.rope = RotaryPositionalEmbedding(head_dim=self.head_dim, max_seq_len=max_seq_len, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape  # noqa: N806

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, k = self.rope(q, k)

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout.p if isinstance(self.dropout, nn.Dropout) and self.training else 0.0,
        )

        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.hidden_size)
        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    """DiT-style transformer block with AdaLN-Zero."""

    def __init__(
        self,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_features: int = 128,
        dropout: float = 0.0,
        use_rope: bool = False,
        max_seq_len: int = 512,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.use_rope = use_rope

        if use_rope:
            self.attn = RoPEAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
            )
        else:
            self.multihead_attn = nn.MultiheadAttention(
                hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout
            )

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(num_features, 6 * hidden_size, bias=True))

    def forward(self, x: Tensor, features: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            features
        ).chunk(6, dim=1)

        attn_input = modulate(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1))

        if self.use_rope:
            attn_out = self.attn(attn_input)
        else:
            attn_out, _ = self.multihead_attn(attn_input, attn_input, attn_input)

        x = x + gate_msa.unsqueeze(1) * attn_out

        mlp_input = modulate(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        mlp_out = self.mlp(mlp_input)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class DiffusionTransformer(nn.Module):
    """Transformer-based diffusion noise prediction model."""

    def __init__(self, config, conditioning_dim: int):
        super().__init__()
        self.config = config
        self.conditioning_dim = conditioning_dim

        self.action_dim = config.action_feature.shape[0]
        self.horizon = config.horizon
        self.hidden_size = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        self.use_rope = config.use_rope

        self.timestep_embed_dim = config.timestep_embed_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.timestep_embed_dim),
            nn.Linear(self.timestep_embed_dim, 2 * self.timestep_embed_dim),
            nn.GELU(),
            nn.Linear(2 * self.timestep_embed_dim, self.timestep_embed_dim),
            nn.GELU(),
        )

        self.cond_dim = self.timestep_embed_dim + conditioning_dim
        self.input_proj = nn.Linear(self.action_dim, self.hidden_size)

        if config.use_positional_encoding:
            self.pos_embedding = nn.Parameter(
                torch.empty(1, self.horizon, self.hidden_size).normal_(std=0.02)
            )
        else:
            self.pos_embedding = None

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    num_features=self.cond_dim,
                    dropout=self.dropout,
                    use_rope=self.use_rope,
                    max_seq_len=self.horizon,
                    rope_base=config.rope_base,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.output_proj = nn.Linear(self.hidden_size, self.action_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for block in self.transformer_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, x: Tensor, timestep: Tensor, conditioning_vec: Tensor) -> Tensor:
        _, seq_len, _ = x.shape

        timestep_features = self.time_mlp(timestep)
        cond_features = torch.cat([timestep_features, conditioning_vec], dim=-1)

        hidden_seq = self.input_proj(x)

        if self.pos_embedding is not None:
            hidden_seq = hidden_seq + self.pos_embedding[:, :seq_len, :]

        for block in self.transformer_blocks:
            hidden_seq = block(hidden_seq, cond_features)

        return self.output_proj(hidden_seq)


# -- Objectives --


class DiffusionObjective(nn.Module):
    """Standard diffusion (DDPM/DDIM) objective implementation."""

    def __init__(self, config, action_dim: int, horizon: int, do_mask_loss_for_padding: bool = False):
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        self.horizon = horizon
        self.do_mask_loss_for_padding = do_mask_loss_for_padding

        scheduler_kwargs = {
            "num_train_timesteps": config.num_train_timesteps,
            "beta_start": config.beta_start,
            "beta_end": config.beta_end,
            "beta_schedule": config.beta_schedule,
            "clip_sample": config.clip_sample,
            "clip_sample_range": config.clip_sample_range,
            "prediction_type": config.prediction_type,
        }

        if config.noise_scheduler_type == "DDPM":
            self.noise_scheduler: DDPMScheduler | DDIMScheduler = DDPMScheduler(**scheduler_kwargs)
        elif config.noise_scheduler_type == "DDIM":
            self.noise_scheduler = DDIMScheduler(**scheduler_kwargs)
        else:
            raise ValueError(f"Unsupported noise scheduler type {config.noise_scheduler_type}")

        self.num_inference_steps = (
            config.num_inference_steps
            if config.num_inference_steps is not None
            else self.noise_scheduler.config.num_train_timesteps
        )

    def compute_loss(self, model: nn.Module, batch: dict[str, Tensor], conditioning_vec: Tensor) -> Tensor:
        clean_actions = batch[ACTION]
        noise = torch.randn_like(clean_actions)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(clean_actions.shape[0],),
            device=clean_actions.device,
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(clean_actions, noise, timesteps)

        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "sample":
            target = clean_actions
        else:
            raise ValueError(f"Unsupported prediction type: {prediction_type}")

        predicted = model(noisy_actions, timesteps, conditioning_vec=conditioning_vec)
        loss = F.mse_loss(predicted, target, reduction="none")

        if self.do_mask_loss_for_padding and "action_is_pad" in batch:
            mask = ~batch["action_is_pad"].unsqueeze(-1)
            num_valid = mask.sum() * loss.shape[-1]
            return (loss * mask).sum() / num_valid.clamp_min(1)

        return loss.mean()

    def conditional_sample(self, model: nn.Module, batch_size: int, conditioning_vec: Tensor) -> Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        sample = torch.randn(
            size=(batch_size, self.horizon, self.action_dim),
            dtype=dtype,
            device=device,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            model_output = model(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                conditioning_vec=conditioning_vec,
            )
            sample = self.noise_scheduler.step(model_output, t, sample).prev_sample

        return sample


class FlowMatchingObjective(nn.Module):
    """Flow matching objective: trains a model to predict velocity fields."""

    def __init__(
        self,
        config,
        action_dim: int,
        horizon: int,
        do_mask_loss_for_padding: bool = False,
        image_feature_slice: slice | None = None,
    ):
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        self.horizon = horizon
        self.do_mask_loss_for_padding = do_mask_loss_for_padding
        self.image_feature_slice = image_feature_slice
        if not self.config.gripper_first and self.action_dim < 2:
            raise ValueError("gripper_first=False requires at least one non-gripper action dimension.")

    def _effective_lambda_flow_k(self, train_step: int | None) -> float:
        if train_step is not None and train_step < self.config.pre_train_steps:
            return 0.0
        return float(self.config.lambda_flow_k)

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        if self.config.timestep_sampling_strategy == "uniform":
            return torch.rand(batch_size, device=device)
        elif self.config.timestep_sampling_strategy == "beta":
            beta_dist = torch.distributions.Beta(
                self.config.timestep_sampling_alpha, self.config.timestep_sampling_beta
            )
            u = beta_dist.sample((batch_size,)).to(device)
            return self.config.timestep_sampling_s * (1.0 - u)
        else:
            raise ValueError(f"Unknown timestep strategy: {self.config.timestep_sampling_strategy}")

    def compute_loss(
        self,
        model: nn.Module,
        batch: dict[str, Tensor],
        conditioning_vec: Tensor,
        conditioning_steps: Tensor | None = None,
        train_step: int | None = None,
        derivative_conditioning_steps: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        action_sequence = batch[ACTION]
        data = action_sequence[:, : self.horizon]
        batch_size = data.shape[0]
        device = data.device

        noise = torch.randn_like(data)
        t = self._sample_timesteps(batch_size, device)
        t_expanded = t.view(-1, 1, 1)
        x_t = t_expanded * data + (1 - (1 - self.config.sigma_min) * t_expanded) * noise

        target_velocity = data - (1 - self.config.sigma_min) * noise
        effective_lambda_flow_k = self._effective_lambda_flow_k(train_step)
        use_kinematic_loss = effective_lambda_flow_k > 0 and torch.is_grad_enabled()
        if use_kinematic_loss:
            if conditioning_steps is None:
                raise ValueError("conditioning_steps is required when lambda_flow_k > 0.")
            if derivative_conditioning_steps is None:
                raise ValueError(
                    "derivative_conditioning_steps is required when lambda_flow_k > 0."
                )
            predicted_velocity, kinematic_loss, kinematic_valid_ratio, kinematic_jvp_metrics = (
                self._compute_kinematic_loss(
                    model=model,
                    batch=batch,
                    data=data,
                    action_sequence=action_sequence,
                    x_t=x_t,
                    t=t,
                    conditioning_vec=conditioning_vec,
                    conditioning_steps=conditioning_steps,
                    derivative_conditioning_steps=derivative_conditioning_steps,
                )
            )
        else:
            predicted_velocity = model(x_t, t, conditioning_vec=conditioning_vec)
            kinematic_loss = data.new_zeros(())
            kinematic_valid_ratio = data.new_zeros(())
            kinematic_jvp_metrics = {}

        flow_loss = self._flow_loss(predicted_velocity, target_velocity, batch)
        total_loss = flow_loss + effective_lambda_flow_k * kinematic_loss

        output_dict = {
            "flow_loss": flow_loss.detach().float().item(),
            "kinematic_loss": kinematic_loss.detach().float().item(),
            "lambda_flow_k": effective_lambda_flow_k,
            "lambda_flow_k_config": float(self.config.lambda_flow_k),
            "pre_train_steps": int(self.config.pre_train_steps),
            "train_step": int(train_step) if train_step is not None else -1,
            "use_jvp_ak": float(self.config.use_jvp_ak),
            "use_1_k": float(self.config.use_1_k),
            "gripper_first": float(self.config.gripper_first),
            "enable_stochastic": float(self.config.enable_stochastic),
            "total_loss": total_loss.detach().float().item(),
            "kinematic_valid_ratio": kinematic_valid_ratio.detach().float().item(),
        }
        output_dict.update(
            {name: value.detach().float().item() for name, value in kinematic_jvp_metrics.items()}
        )
        return total_loss, output_dict

    def _flow_loss(self, predicted_velocity: Tensor, target_velocity: Tensor, batch: dict[str, Tensor]) -> Tensor:
        loss = F.mse_loss(predicted_velocity, target_velocity, reduction="none")

        if self.do_mask_loss_for_padding and "action_is_pad" in batch:
            mask = ~batch["action_is_pad"][:, : loss.shape[1]].to(device=loss.device, dtype=torch.bool).unsqueeze(-1)
            num_valid = mask.sum() * loss.shape[-1]
            return (loss * mask).sum() / num_valid.clamp_min(1)

        return loss.mean()

    def _sample_frequency(self, batch: dict[str, Tensor], reference: Tensor) -> Tensor:
        fps = batch.get("sample_frequency")
        if fps is None:
            fps = batch.get("fps")
        if fps is None:
            fps = self.config.sample_frequency

        fps_tensor = torch.as_tensor(fps, device=reference.device, dtype=reference.dtype)
        if fps_tensor.ndim == 0:
            return fps_tensor
        if fps_tensor.shape[0] != reference.shape[0]:
            raise ValueError(
                f"sample_frequency batch dimension {fps_tensor.shape[0]} does not match batch size {reference.shape[0]}"
            )
        return fps_tensor.reshape(reference.shape[0], *([1] * (reference.ndim - 1)))

    def _action_dot(self, action_sequence: Tensor, fps: Tensor) -> Tensor:
        if self.config.dct_coe_num == 0:
            return (action_sequence[:, 1 : self.horizon + 1] - action_sequence[:, : self.horizon]) * fps

        num_samples = self.horizon
        if action_sequence.shape[1] < num_samples:
            raise ValueError(
                f"DCT action derivative requires at least horizon ({num_samples}) action frames, "
                f"got {action_sequence.shape[1]}"
            )

        num_modes = self.config.dct_coe_num
        input_dtype = action_sequence.dtype
        compute_dtype = torch.float32 if input_dtype in {torch.float16, torch.bfloat16} else input_dtype
        with _disabled_autocast_context(action_sequence.device):
            actions = action_sequence[:, :num_samples].to(dtype=compute_dtype)
            modes = torch.arange(num_modes, device=actions.device, dtype=compute_dtype)
            sample_points = torch.arange(num_samples, device=actions.device, dtype=compute_dtype) + 0.5
            alpha = torch.full_like(modes, math.sqrt(2.0 / num_samples))
            alpha[0] = math.sqrt(1.0 / num_samples)

            phase = math.pi * modes[:, None] * sample_points[None, :] / num_samples
            basis = alpha[:, None] * torch.cos(phase)
            coefficients = torch.einsum("bnd,kn->bkd", actions, basis)

            derivative_basis = -alpha[:, None] * (math.pi * modes[:, None] / num_samples) * torch.sin(phase)
            action_dot = torch.einsum("bkd,kh->bhd", coefficients, derivative_basis)
            action_dot = action_dot * fps.to(dtype=compute_dtype)

        return action_dot.to(dtype=input_dtype)

    def _kinematic_action_mask(self, reference: Tensor) -> Tensor:
        mask = torch.ones(self.action_dim, dtype=torch.bool, device=reference.device)
        if not self.config.gripper_first:
            mask[-1] = False
        return mask

    def _conditioning_dot(
        self,
        conditioning_steps: Tensor,
        fps: Tensor,
        derivative_conditioning_steps: Tensor | None = None,
    ) -> Tensor:
        mode = getattr(self.config, "conditioning_derivative_mode", "reverse")
        image_only = getattr(self.config, "image_only_condition_jvp", False)
        if derivative_conditioning_steps is None:
            raise ValueError(f"{mode} conditioning differences require the complete derivative stencil.")

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
        elif mode == "central":
            previous_steps = derivative_conditioning_steps[
                :, conditioning_start - 1 : conditioning_start + num_conditioning_steps - 1
            ]
            next_steps = derivative_conditioning_steps[
                :, conditioning_start + 1 : conditioning_start + num_conditioning_steps + 1
            ]
            conditioning_dot_steps = (next_steps - previous_steps) * (conditioning_fps * 0.5)
        else:
            raise ValueError(f"Unsupported conditioning derivative mode: {mode}")

        if image_only:
            if self.image_feature_slice is None:
                raise ValueError("image_only_condition_jvp=True requires at least one image feature.")
            image_mask = conditioning_dot_steps.new_zeros(conditioning_dot_steps.shape[-1])
            image_mask[self.image_feature_slice] = 1
            conditioning_dot_steps = conditioning_dot_steps * image_mask

        return conditioning_dot_steps.flatten(start_dim=1)

    def _kinematic_valid_mask(
        self,
        batch: dict[str, Tensor],
        data: Tensor,
        conditioning_steps: Tensor,
        derivative_conditioning_steps: Tensor | None = None,
    ) -> Tensor:
        kinematic_valid = torch.ones(data.shape[:2], dtype=torch.bool, device=data.device)

        if "action_is_pad" in batch:
            action_is_pad = batch["action_is_pad"].to(device=data.device, dtype=torch.bool)
            action_valid = ~action_is_pad[:, : data.shape[1]]
            if self.config.dct_coe_num == 0:
                action_valid &= ~action_is_pad[:, 1 : data.shape[1] + 1]
            kinematic_valid &= action_valid

        obs_pad_key = f"{OBS_STATE}_is_pad"
        if obs_pad_key in batch:
            if derivative_conditioning_steps is None:
                raise ValueError("Observation padding validation requires the complete derivative stencil.")
            obs_is_pad = batch[obs_pad_key].to(device=data.device, dtype=torch.bool)
            if obs_is_pad.shape[:2] != derivative_conditioning_steps.shape[:2]:
                raise ValueError(
                    f"{obs_pad_key} shape {tuple(obs_is_pad.shape)} does not match conditioning steps "
                    f"{tuple(derivative_conditioning_steps.shape[:2])}"
                )
            obs_valid = ~obs_is_pad.any(dim=1)
            kinematic_valid &= obs_valid[:, None]
        # If no observation padding mask is present, validity is determined by action padding.

        return kinematic_valid

    def _compute_kinematic_loss(
        self,
        model: nn.Module,
        batch: dict[str, Tensor],
        data: Tensor,
        action_sequence: Tensor,
        x_t: Tensor,
        t: Tensor,
        conditioning_vec: Tensor,
        conditioning_steps: Tensor,
        derivative_conditioning_steps: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        fps = self._sample_frequency(batch, data)
        kinematic_action_mask = self._kinematic_action_mask(data)
        a_dot_data = self._action_dot(action_sequence, fps)
        a_dot_data = a_dot_data * kinematic_action_mask.to(dtype=a_dot_data.dtype)
        conditioning_dot_vec = self._conditioning_dot(
            conditioning_steps,
            fps,
            derivative_conditioning_steps=derivative_conditioning_steps,
        )
        state_jvp = None
        action_jvp = None
        if self.config.enable_stochastic:
            horizon_index = torch.randint(self.horizon, ()).item()
            horizon_slice = slice(horizon_index, horizon_index + 1)
        else:
            horizon_slice = slice(None)

        def flow_vector_field_cond(cond: Tensor) -> Tensor:
            return model(x_t, t, conditioning_vec=cond)[:, horizon_slice]

        if self.config.enable_stochastic:
            # Keep the primal flow pass eligible for Flash SDPA under the outer AMP context.
            predicted_velocity = model(x_t, t, conditioning_vec=conditioning_vec)

        with _disabled_autocast_context(x_t.device), _sdpa_math_kernel_context():
            if self.config.enable_stochastic:
                if self.config.use_jvp_ak:

                    def flow_vector_field(cond: Tensor, actions: Tensor) -> Tensor:
                        return model(actions, t, conditioning_vec=cond)[:, horizon_slice]

                    _, kinematic_dot = torch.autograd.functional.jvp(
                        flow_vector_field,
                        (conditioning_vec, x_t),
                        (conditioning_dot_vec, t.view(-1, 1, 1) * a_dot_data),
                        create_graph=True,
                    )
                else:
                    _, kinematic_dot = torch.autograd.functional.jvp(
                        flow_vector_field_cond,
                        conditioning_vec,
                        conditioning_dot_vec,
                        create_graph=True,
                    )
                    state_jvp = kinematic_dot
                    if self.config.use_1_k:
                        kinematic_dot = (1 - t.view(-1, 1, 1)) * kinematic_dot
                residual = kinematic_dot - a_dot_data[:, horizon_slice]
            else:
                predicted_velocity, v_s_dot = torch.func.jvp(
                    flow_vector_field_cond,
                    (conditioning_vec,),
                    (conditioning_dot_vec,),
                )
                state_jvp = v_s_dot

                if self.config.use_jvp_ak:

                    def flow_vector_field_action(actions: Tensor) -> Tensor:
                        return model(actions, t, conditioning_vec=conditioning_vec)

                    _, v_a_k_dot = torch.func.jvp(
                        flow_vector_field_action,
                        (x_t,),
                        (a_dot_data,),
                    )
                    action_jvp = v_a_k_dot
                    residual = v_s_dot + t.view(-1, 1, 1) * v_a_k_dot - a_dot_data
                else:
                    if self.config.use_1_k:
                        v_s_dot = (1 - t.view(-1, 1, 1)) * v_s_dot
                    residual = v_s_dot - a_dot_data

        kinematic_valid = self._kinematic_valid_mask(
            batch,
            data,
            conditioning_steps,
            derivative_conditioning_steps=derivative_conditioning_steps,
        )[:, horizon_slice]
        kinematic_loss_per_step = torch.mean(residual[..., kinematic_action_mask] ** 2, dim=-1)
        valid = kinematic_valid.to(dtype=kinematic_loss_per_step.dtype)
        num_valid = valid.sum()
        kinematic_loss = (kinematic_loss_per_step * valid).sum() / num_valid.clamp_min(1)
        kinematic_valid_ratio = valid.mean()

        # JVPs are vector-valued. Log their masked RMS magnitudes so the two
        # kinematic terms are comparable as scalar WandB curves. Detaching here
        # keeps observability from extending the training autograd graph.
        def masked_jvp_rms(jvp: Tensor) -> Tensor:
            squared_per_step = torch.mean(
                jvp.detach()[..., kinematic_action_mask].float().square(),
                dim=-1,
            )
            return torch.sqrt((squared_per_step * valid.float()).sum() / num_valid.float().clamp_min(1))

        jvp_metrics = {}
        if state_jvp is not None:
            jvp_metrics["kinematic_state_jvp_rms"] = masked_jvp_rms(state_jvp)
        if action_jvp is not None:
            jvp_metrics["kinematic_action_jvp_rms"] = masked_jvp_rms(action_jvp)

        return predicted_velocity, kinematic_loss, kinematic_valid_ratio, jvp_metrics

    def conditional_sample(self, model: nn.Module, batch_size: int, conditioning_vec: Tensor) -> Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        x = torch.randn((batch_size, self.horizon, self.action_dim), dtype=dtype, device=device)

        num_steps = self.config.num_integration_steps
        time_grid = torch.linspace(0, 1, num_steps + 1, device=device)

        if self.config.integration_method == "euler":
            x = self._euler_integrate(model, x, time_grid, conditioning_vec)
        elif self.config.integration_method == "rk4":
            x = self._rk4_integrate(model, x, time_grid, conditioning_vec)
        else:
            raise ValueError(f"Unknown integration method: {self.config.integration_method}")

        return x

    def _euler_integrate(
        self, model: nn.Module, x_init: Tensor, time_grid: Tensor, conditioning_vec: Tensor
    ) -> Tensor:
        x = x_init
        for i in range(len(time_grid) - 1):
            t_scalar = time_grid[i].item()
            dt = (time_grid[i + 1] - time_grid[i]).item()
            t_batch = torch.full((x.shape[0],), t_scalar, dtype=x.dtype, device=x.device)
            with torch.no_grad():
                velocity = model(x, t_batch, conditioning_vec=conditioning_vec)
            x = x + dt * velocity
        return x

    def _rk4_integrate(
        self, model: nn.Module, x_init: Tensor, time_grid: Tensor, conditioning_vec: Tensor
    ) -> Tensor:
        x = x_init

        def dynamics(x_val: Tensor, t_scalar: float) -> Tensor:
            t_batch = torch.full((x_val.shape[0],), t_scalar, dtype=x_val.dtype, device=x_val.device)
            with torch.no_grad():
                return model(x_val, t_batch, conditioning_vec=conditioning_vec)

        for i in range(len(time_grid) - 1):
            t = time_grid[i].item()
            dt = (time_grid[i + 1] - time_grid[i]).item()

            k1 = dynamics(x, t)
            k2 = dynamics(x + dt * k1 / 2, t + dt / 2)
            k3 = dynamics(x + dt * k2 / 2, t + dt / 2)
            k4 = dynamics(x + dt * k3, t + dt)

            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return x
