# 完成标注

- `src/lerobot/policies/multi_task_dit/configuration_multi_task_dit.py`: 第 72、74-76、138-153 行为本次新增/改动的 ResNet vision 配置和校验。
- `src/lerobot/policies/multi_task_dit/modeling_multi_task_dit.py`: 第 65、118-120、227-293、341-347、365-370 行为本次新增/改动的 ResNet encoder、选择逻辑和 optimizer 分组。
- `prompt/lerobot_multi_task_dit_light_resnet_prompt.md`: 第 1-6 行为本次完成标注。

# MultiTaskDiT Flow 轻量 ResNet Vision 改造 Prompt

请在 `/home/zhouzhi/code_store/higher-order/lerobot` 中对 `multi_task_dit` 做小改，目标是让单任务 `flow_matching` 训练更轻量。不要重构无关 policy。

## 目标

- 保留 `multi_task_dit` 的 flow matching action chunk 逻辑。
- 将 vision encoder 从默认 CLIP 改为可选轻量 ResNet。
- Text 部分不要改结构：继续使用现有 `CLIPTextEncoder`，保持 frozen。
- 保持单任务训练可用，优先降低训练时间和显存。

## 主要改动

1. 在 `MultiTaskDiTConfig` 增加 vision 选项：

```python
vision_encoder_type: str = "clip"  # "clip" or "resnet"
vision_backbone: str = "resnet18"
pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
spatial_softmax_num_keypoints: int = 32
```

`vision_encoder_name` 保留给 CLIP；只有 `vision_encoder_type == "clip"` 时才校验名字包含 `clip`。

2. 在 `modeling_multi_task_dit.py` 增加 `ResNetVisionEncoder`。

实现方式参考 diffusion policy 的 `DiffusionRgbEncoder`，但要适配 MultiTaskDiT 当前接口：

```python
forward(x) -> Tensor  # (B, feature_dim, 1, 1)
get_output_shape() -> tuple[int, int, int]  # (feature_dim, 1, 1)
```

其中：

```python
feature_dim = spatial_softmax_num_keypoints * 2
```

默认 `32` 个 keypoints 时，每个相机输出 `64` 维。

3. 在 `ObservationEncoder` 中按配置选择 encoder：

```python
if config.vision_encoder_type == "clip":
    encoder = CLIPVisionEncoder(...)
elif config.vision_encoder_type == "resnet":
    encoder = ResNetVisionEncoder(...)
```

`use_separate_rgb_encoder_per_camera` 的逻辑保持一致：既支持共享 encoder，也支持每个相机一个 encoder。

4. 注意 preprocessing 不要重复做。

MultiTaskDiT 现在已经在 `ObservationEncoder._apply_preprocessing()` 里做 resize/crop。新的 `ResNetVisionEncoder` 不要再单独 resize/crop，直接吃已经预处理后的 `(B, C, H, W)`。

5. 维度必须自动计算，不要硬编码。

`ObservationEncoder._setup_vector_output()` 继续通过：

```python
feature_map_shape = encoder_to_check.get_output_shape()
c, h, w = feature_map_shape
spatial_feature_dim = c * h * w
```

计算 vision conditioning 维度。ResNet 输出 `(64, 1, 1)` 后，两个相机就是 `128` 维，再加 state/text，然后乘 `n_obs_steps`。

Text 维度继续保留：

```python
self.text_dim = config.hidden_dim
total_dim += self.text_dim
```

不要移除 language tokens，也不要改 `CLIPTextEncoder` 的 frozen 逻辑。

6. 更新 optimizer 参数分组。

当前只匹配 `observation_encoder.vision_encoder`，需要确保共享 encoder 和 `vision_encoders` 都能进入 vision lr multiplier 参数组。可以按名字前缀匹配：

```python
"observation_encoder.vision_encoder" in name
or "observation_encoder.vision_encoders" in name
```

## 推荐快速验证配置

```text
policy.type=multi_task_dit
policy.objective=flow_matching
policy.vision_encoder_type=resnet
policy.vision_backbone=resnet18
policy.spatial_softmax_num_keypoints=32
policy.hidden_dim=256
policy.num_layers=3
policy.num_heads=4
policy.timestep_embed_dim=128
policy.horizon=16
policy.n_action_steps=8
policy.n_obs_steps=1 或 2
policy.num_integration_steps=8
```

先跑单任务 `TASK_IDS=[0]`，训练 `2k-5k` steps，少量 eval episodes 即可验证趋势。
最后，在完成任务之后，你需要在这个文档的开头标注一下你改了哪些地方的第几行是你新加入的。
