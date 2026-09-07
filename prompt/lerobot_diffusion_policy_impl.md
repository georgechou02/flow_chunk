# LeRobot Diffusion Policy 实现梳理

本文档解释当前仓库里的 `diffusion` policy 是怎么实现的。对应代码主要在：

- `src/lerobot/policies/diffusion/configuration_diffusion.py`
- `src/lerobot/policies/diffusion/modeling_diffusion.py`
- `src/lerobot/policies/diffusion/processor_diffusion.py`
- `src/lerobot/datasets/factory.py`
- `src/lerobot/datasets/dataset_reader.py`
- `src/lerobot/scripts/lerobot_train.py`

## 总体结构

`diffusion` policy 是一个 action chunk diffusion policy。它不是一步输出一个 action，而是根据一段历史 observation 条件，一次生成长度为 `horizon` 的 action trajectory，然后从里面取 `n_action_steps` 个动作执行。

代码层次是：

```text
DiffusionConfig
  定义输入窗口、action chunk 长度、U-Net 参数、scheduler 参数、normalization 等

DiffusionPolicy
  LeRobot policy wrapper，负责 queue、select_action、forward、processor 对接

DiffusionModel
  真正的模型逻辑：观测编码、U-Net、diffusion scheduler、训练 loss、采样

DiffusionRgbEncoder
  图像编码器：ResNet backbone + SpatialSoftmax + Linear

DiffusionConditionalUnet1d
  条件 1D U-Net：在 action trajectory 时间轴上做卷积，用 diffusion timestep 和 observation condition 做 FiLM 调制
```

在 factory 里，`policy.type=diffusion` 会映射到 `DiffusionConfig` 和 `DiffusionPolicy`：

- `get_policy_class("diffusion") -> DiffusionPolicy`
- `make_policy_config("diffusion") -> DiffusionConfig`
- processor 会走 `make_diffusion_pre_post_processors`

## 配置含义

`DiffusionConfig` 默认的核心结构是：

```python
n_obs_steps = 2
horizon = 64
n_action_steps = 32
```

含义：

- `n_obs_steps`: 输入给 policy 的 observation 历史长度，默认 2。也就是当前帧和上一帧。
- `horizon`: diffusion model 生成的完整 action trajectory 长度，默认 64。
- `n_action_steps`: 实际拿去执行的 action 数量，默认 32。

注意这里的 `horizon` 不是“从当前时刻开始的 64 步”。因为 observation window 里包含历史帧，所以 action window 的索引是相对 observation window 对齐的。

配置里：

```python
observation_delta_indices = list(range(1 - n_obs_steps, 1))
action_delta_indices = list(range(1 - n_obs_steps, 1 - n_obs_steps + horizon))
```

默认 `n_obs_steps=2, horizon=64` 时：

```text
observation_delta_indices = [-1, 0]
action_delta_indices      = [-1, 0, 1, ..., 62]
```

所以 dataloader 会为一个中心 timestep `t` 返回：

- observation: `[t-1, t]`
- action: `[t-1, t, t+1, ..., t+62]`

推理时模型生成完整 `horizon` 后，只取：

```python
start = n_obs_steps - 1
end = start + n_action_steps
actions = actions[:, start:end]
```

默认就是从完整 action trajectory 的 index 1 开始取 32 个动作，即对应当前时刻 `t` 到 `t+31`。

`drop_n_last_frames` 默认是 7，注释里写的是：

```text
horizon - n_action_steps - n_obs_steps + 1
```

默认 `64 - 32 - 2 + 1 = 31`，但当前文件默认值是 7，脚本里也可以覆盖。它不在 model 内部用，而是在 `EpisodeAwareSampler` 里丢掉每个 episode 最后的若干中心帧，减少末尾 action window 大量 padding 的情况。

## 数据窗口是怎么来的

训练数据不是 policy 自己手动取历史帧，而是 dataset 根据 config 的 delta indices 取窗口。

流程：

1. `lerobot_train.py` 创建 dataset。
2. `datasets/factory.py` 调 `resolve_delta_timestamps(cfg.trainable_config, ds_meta)`。
3. `resolve_delta_timestamps` 读取 policy config 的：
   - `observation_delta_indices`
   - `action_delta_indices`
   - `reward_delta_indices`
4. 它把 frame index offset 除以 dataset fps，变成 delta timestamps。
5. `LeRobotDataset` / `DatasetReader` 根据这些 offsets 查询对应帧。

`DatasetReader._get_query_indices` 会对 episode 边界做 clamp：

```python
query_idx = max(ep_start, min(ep_end - 1, abs_idx + delta))
```

同时生成 padding mask：

```python
"action_is_pad": abs_idx + delta < ep_start or abs_idx + delta >= ep_end
```

所以训练 batch 里典型形状是：

```text
observation.state        (B, n_obs_steps, state_dim)
observation.images       (B, n_obs_steps, num_cameras, C, H, W)
observation.env_state    (B, n_obs_steps, env_dim)   可选
action                   (B, horizon, action_dim)
action_is_pad            (B, horizon)
```

如果某些 action 是 episode 边界外 clamp 出来的，`action_is_pad=True`。训练时只有 `do_mask_loss_for_padding=True` 才会屏蔽这些位置。

## Processor 做了什么

`processor_diffusion.py` 定义 preprocessor 和 postprocessor。

preprocessor 顺序：

```text
RenameObservationsProcessorStep(rename_map={})
AddBatchDimensionProcessorStep()
DeviceProcessorStep(device=config.device)
NormalizerProcessorStep(features=input+output, norm_map=normalization_mapping, stats=dataset_stats)
```

postprocessor 顺序：

```text
UnnormalizerProcessorStep(features=output_features, norm_map=normalization_mapping, stats=dataset_stats)
DeviceProcessorStep(device="cpu")
```

默认 normalization：

```python
VISUAL -> MEAN_STD
STATE  -> MIN_MAX
ACTION -> MIN_MAX
```

也就是说模型内部训练和采样的 action 通常是在归一化空间里。推理输出会经过 postprocessor unnormalize 回机器人/环境动作空间。

## DiffusionPolicy wrapper

`DiffusionPolicy` 继承 `PreTrainedPolicy`，主要负责三件事。

第一，初始化：

```python
require_package("diffusers", extra="diffusion")
config.validate_features()
self.diffusion = DiffusionModel(config)
self.reset()
```

第二，维护推理队列：

```python
self._queues = {
    OBS_STATE: deque(maxlen=n_obs_steps),
    ACTION: deque(maxlen=n_action_steps),
}
```

如果有 image 或 env_state，也会为它们建 observation queue。

第三，提供三个主要接口：

```python
forward(batch)
predict_action_chunk(batch, noise=None)
select_action(batch, noise=None)
```

### forward

训练时调用：

```python
loss = self.diffusion.compute_loss(batch)
return loss, None
```

如果有多个 camera image，`forward` 会先把各 camera key stack 成统一的 `OBS_IMAGES`：

```python
batch[OBS_IMAGES] = torch.stack([batch[key] for key in image_features], dim=-4)
```

### predict_action_chunk

离线 batch 或手动传入 batch 时，如果 queue 为空，就直接使用 batch；如果 queue 已经有 rollout 历史，则把 queue 里的 observation stack 成：

```text
(B, n_obs_steps, ...)
```

然后调用：

```python
actions = self.diffusion.generate_actions(batch, noise=noise)
```

返回的是已经截取后的 `(B, n_action_steps, action_dim)`。

### select_action

在线 rollout 时每个 env step 调一次。

逻辑：

1. 如果 batch 里有 `action`，先 pop 掉，因为 evaluation batch 可能带 label，但推理不需要。
2. 把当前 observation 放进 queue。
3. 如果 action queue 空了：
   - 调 `predict_action_chunk`
   - 得到 `(B, n_action_steps, action_dim)`
   - transpose 成 `(n_action_steps, B, action_dim)` 放入 queue
4. 每次 `popleft()` 一个 action 返回。

这就是 action chunking 的执行方式：模型不是每步都重采样，而是 action queue 空了才重采样一次。

## DiffusionModel 的 observation conditioning

`DiffusionModel.__init__` 会根据输入特征构造 observation encoder。

条件向量由这些东西拼起来：

```text
robot state features
image features          可选
environment state       可选
```

每个 observation step 都会编码，最后沿时间展平。

具体逻辑在 `_prepare_global_conditioning`：

```python
global_cond_feats = [batch[OBS_STATE]]
...
global_cond_feats.append(img_features)
...
global_cond_feats.append(batch[OBS_ENV_STATE])
return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)
```

如果单步 condition dim 是 `D_cond`，那么最终传给 U-Net 的 `global_cond` 是：

```text
(B, n_obs_steps * D_cond)
```

这就是为什么 `DiffusionConditionalUnet1d` 初始化时用：

```python
global_cond_dim = per_step_global_cond_dim * config.n_obs_steps
```

## 图像编码器

`DiffusionRgbEncoder` 是：

```text
optional resize
optional random/center crop
torchvision ResNet backbone 去掉最后 avgpool/fc
SpatialSoftmax
Linear
ReLU
```

默认 backbone 是 `resnet18`，预训练权重是 `ResNet18_Weights.IMAGENET1K_V1`。

多相机时有两种模式：

1. `use_separate_rgb_encoder_per_camera=True` 默认：
   - 每个 camera 一个独立 ResNet encoder。
   - 每个相机输出 feature 后 concat。
2. `False`：
   - 所有 camera 共享同一个 encoder。

`SpatialSoftmax` 把 ResNet feature map 转成 keypoint-like feature。默认 `spatial_softmax_num_keypoints=32`，输出维度是：

```text
32 keypoints * 2 coordinates = 64
```

随后 `Linear(64, 64) + ReLU`。

## 条件 U-Net

核心网络是 `DiffusionConditionalUnet1d`。

输入：

```text
x             (B, T, action_dim)
timestep      (B,)
global_cond   (B, global_cond_dim)
```

输出：

```text
(B, T, action_dim)
```

这里 `T = horizon`。U-Net 在 action trajectory 的时间轴上做 1D convolution，不是在视频时间轴或 diffusion step 轴上卷积。

forward 里先把：

```python
x: (B, T, D) -> (B, D, T)
```

因为 PyTorch `Conv1d` 使用 `(B, C, T)`。

然后 diffusion timestep 先过 sinusoidal embedding 和 MLP：

```text
timestep -> DiffusionSinusoidalPosEmb -> Linear -> Mish -> Linear
```

再和 `global_cond` concat：

```python
global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
```

这个 `global_feature` 会传入每个 residual block，用 FiLM 调制卷积特征。

U-Net 结构：

```text
down path:
  [ResBlock, ResBlock, Downsample] x len(down_dims)

middle:
  ResBlock
  ResBlock

up path:
  concat skip
  ResBlock
  ResBlock
  Upsample

final:
  Conv1dBlock
  Conv1d -> action_dim
```

默认 `down_dims=(512, 1024, 2048)`，所以 horizon 必须能被 `2 ** len(down_dims)` 整除。默认 `horizon=64`，可以被 8 整除。

FiLM residual block 是 `DiffusionConditionalResidualBlock1d`：

```text
Conv1dBlock
condition MLP -> bias 或 scale+bias
Conv1dBlock
residual connection
```

默认 `use_film_scale_modulation=True`，所以 condition 会输出 scale 和 bias。

## 训练 loss

训练入口是：

```python
DiffusionPolicy.forward(batch)
  -> DiffusionModel.compute_loss(batch)
```

`compute_loss` 要求：

```text
observation.state      (B, n_obs_steps, state_dim)
observation.images     (B, n_obs_steps, num_cameras, C, H, W) 或 env_state
action                 (B, horizon, action_dim)
action_is_pad          (B, horizon)
```

训练步骤：

1. 编码 observation：

```python
global_cond = self._prepare_global_conditioning(batch)
```

2. 取 clean action trajectory：

```python
trajectory = batch[ACTION]
```

3. 随机采样噪声：

```python
eps = torch.randn(trajectory.shape, device=trajectory.device)
```

4. 每个 batch item 随机采样 diffusion timestep：

```python
timesteps = torch.randint(
    low=0,
    high=num_train_timesteps,
    size=(B,),
).long()
```

5. 用 diffusers scheduler 加噪：

```python
noisy_trajectory = noise_scheduler.add_noise(trajectory, eps, timesteps)
```

6. U-Net 预测：

```python
pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)
```

7. 目标取决于 `prediction_type`：

```python
if prediction_type == "epsilon":
    target = eps
elif prediction_type == "sample":
    target = batch[ACTION]
```

默认是 `epsilon`，也就是预测噪声。

8. MSE：

```python
loss = F.mse_loss(pred, target, reduction="none")
```

9. 如果 `do_mask_loss_for_padding=True`，用 `action_is_pad` 屏蔽 padding：

```python
mask = (~batch["action_is_pad"]).unsqueeze(-1)
loss = (loss * mask).sum() / (mask.sum() * action_dim).clamp_min(1)
```

否则直接：

```python
loss.mean()
```

训练脚本里 `update_policy` 是标准 PyTorch/Accelerate 流程：

```text
with accelerator.autocast():
    loss, output_dict = policy.forward(batch)
accelerator.backward(loss)
clip grad
optimizer.step()
optimizer.zero_grad()
lr_scheduler.step()
```

这个 `DiffusionPolicy.forward` 返回的是 `(loss, None)`，所以没有额外 loss log dict。

## 推理采样

推理调用链：

```text
select_action
  -> predict_action_chunk
    -> DiffusionModel.generate_actions
      -> _prepare_global_conditioning
      -> conditional_sample
      -> 截取 current 对齐的 n_action_steps
```

`conditional_sample` 流程：

1. 从标准高斯初始化 action trajectory：

```python
sample = torch.randn((B, horizon, action_dim))
```

也可以通过参数 `noise` 固定初始噪声，方便 debug 或可重复比较。

2. 设置 scheduler timesteps：

```python
noise_scheduler.set_timesteps(num_inference_steps)
```

如果 `num_inference_steps=None`，默认用 `num_train_timesteps`。脚本里常设成更小的值，比如 32。

3. 反向 denoising loop：

```python
for t in noise_scheduler.timesteps:
    model_output = unet(sample, t_batch, global_cond)
    sample = noise_scheduler.step(model_output, t, sample).prev_sample
```

4. 返回完整 `horizon` action trajectory。

5. `generate_actions` 再截取：

```python
start = n_obs_steps - 1
end = start + n_action_steps
actions = actions[:, start:end]
```

默认 `n_obs_steps=2`，所以从 index 1 开始取。

## 时间对齐例子

默认：

```text
n_obs_steps = 2
horizon = 64
n_action_steps = 32
```

以当前 timestep `t` 为中心：

```text
输入 observation:
  obs[t-1], obs[t]

训练 target action trajectory:
  a[t-1], a[t], a[t+1], ..., a[t+62]

模型生成完整 trajectory:
  a_hat[t-1], a_hat[t], ..., a_hat[t+62]

实际执行:
  a_hat[t], a_hat[t+1], ..., a_hat[t+31]
```

这解释了为什么 `horizon` 里包含历史 action 位置。模型训练时见过 action trajectory 和 observation window 的完整对齐关系；执行时只使用当前及未来 action。

## 和 ACT 的区别

这个仓库里的 ACT 强制 `n_obs_steps == 1`，没有 observation history 输入。Diffusion policy 则默认支持历史输入，并且 dataloader 会真的返回 `(B, n_obs_steps, ...)`。

ACT:

```text
observation: 当前帧
output: action chunk
loss: L1 + optional VAE KL
```

Diffusion:

```text
observation: 历史窗口，默认 2 帧
output: action trajectory denoising
loss: diffusion MSE on noise/sample
```

## 和 FlowPolicy 的关系

当前仓库里 `flow` policy 是从 diffusion policy 结构改出来的。两者共享很多设计：

- 都用 action chunk / horizon。
- 都用 observation history 做 global condition。
- 都用 `DiffusionRgbEncoder` 和 `DiffusionConditionalUnet1d` 风格的网络。
- 都用 `action_is_pad` 处理 episode 边界。

关键区别：

Diffusion policy 训练：

```text
x_t = scheduler.add_noise(action, eps, timestep)
target = eps 或 clean action
model  = denoising prediction
loss   = MSE(model(x_t, timestep, cond), target)
```

Flow policy 训练：

```text
x_t = t * action + (1 - t) * noise   具体实现可能带 sigma_min
target_velocity = action - noise
model = velocity field
loss = MSE(model(x_t, t, cond), target_velocity)
```

所以如果目标是加 `bc_only.py` 那类 kinematic loss，`flow` 比 `diffusion` 更直接，因为它本来就学习 velocity field。Diffusion policy 的 U-Net 默认预测 `epsilon`，不是物理意义上的 action velocity field；要加 kinetic loss 需要先定义它约束的是 denoising score/noise prediction、clean sample prediction，还是某个转换后的 velocity，这会比 flow policy 绕。

## 对 kinetic loss 实验的启示

如果后续想在 LeRobot 里验证 action chunk kinetic loss：

1. Diffusion policy 有历史 observation 输入，数据窗口和 action chunk 对齐比较完整。
2. 但 diffusion 默认目标是 noise prediction，不是 flow velocity。
3. `action` batch 默认只有 `horizon` 个 action，并没有额外的 `next_actions` 字段。
4. 可以在 action chunk 内部近似 `a_dot`：

```text
delta_a[:, :-1] = action[:, 1:] - action[:, :-1]
```

但最后一个 action 没有 next action，必须丢掉、置零或扩展 action window 到 `horizon + 1`。

5. 如果对 observation 做 JVP，直接对 pixel/image 做 JVP 不一定有清晰物理意义；更可控的是对 encoded/global condition 或 proprio state 部分做。
6. 如果使用 `use_jvp_ak=True` 的 action-input JVP，`delta_a` 的噪声会直接进入 JVP 方向，真实机器人 action chunk 上需要特别注意平滑和 episode boundary mask。

因此，若只是理解历史输入和 action chunk，diffusion policy 很合适；若要最小成本验证 `bc_only.py` 那种 kinematic residual，当前仓库的 `flow` policy 是更自然的起点。

## 当前 LIBERO diffusion 脚本

仓库里有 `train_diffusion_libero10_task01.sh`。它的核心配置是：

```bash
--policy.type=diffusion
--policy.device=cuda
--policy.horizon="${POLICY_HORIZON}"              # 默认 64
--policy.n_action_steps="${POLICY_N_ACTION_STEPS}" # 默认 32
--policy.down_dims="${POLICY_DOWN_DIMS}"           # 默认 [512,1024,2048]
--policy.num_train_timesteps="${POLICY_NUM_TRAIN_TIMESTEPS}"       # 默认 100
--policy.num_inference_steps="${POLICY_NUM_INFERENCE_STEPS}"       # 默认 32
--policy.noise_scheduler_type=DDIM
--policy.do_mask_loss_for_padding=true
--policy.drop_n_last_frames="${POLICY_DROP_N_LAST_FRAMES}"
```

这里把训练/推理 scheduler 设为 DDIM，训练 timestep 默认 100，推理只跑 32 步 denoising。`do_mask_loss_for_padding=true` 对 LIBERO 这种 episode window 很重要，否则末尾 clamp/pad 出来的 action 会进入 loss。
