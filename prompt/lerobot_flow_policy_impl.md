# LeRobot Flow Policy 实现梳理

本文档解释当前仓库里的 `flow` policy 是怎么实现的，并单独对比它和 `diffusion` policy 的不同。对应代码主要在：

- `src/lerobot/policies/flow/configuration_flow.py`
- `src/lerobot/policies/flow/modeling_flow.py`
- `src/lerobot/policies/flow/processor_flow.py`
- `src/lerobot/policies/diffusion/modeling_diffusion.py`
- `src/lerobot/datasets/factory.py`
- `src/lerobot/datasets/dataset_reader.py`
- `src/lerobot/scripts/lerobot_train.py`

## 总体结构

`flow` policy 是一个 action chunk flow-matching policy。它沿用了 `DiffusionPolicy` 的 action chunk、observation history、RGB encoder 和 1D U-Net 结构，但训练目标不是 diffusion denoising，而是直接学习从 noise action trajectory 积分到 data action trajectory 的 velocity field。

代码层次是：

```text
FlowConfig
  定义输入窗口、action chunk 长度、U-Net 参数、flow matching 参数、normalization 等

FlowPolicy
  LeRobot policy wrapper，负责 queue、select_action、forward、processor 对接

FlowModel
  真正的 flow matching 模型逻辑：观测编码、U-Net velocity field、训练 loss、Euler 采样

DiffusionRgbEncoder
  复用 diffusion policy 的图像编码器：ResNet backbone + SpatialSoftmax + Linear

DiffusionConditionalUnet1d
  复用 diffusion policy 的条件 1D U-Net：输入 action trajectory + time + observation condition，输出 velocity
```

在 factory 里，`policy.type=flow` 会映射到：

- `get_policy_class("flow") -> FlowPolicy`
- `make_policy_config("flow") -> FlowConfig`
- processor 走 `make_flow_pre_post_processors`

## 配置含义

`FlowConfig` 默认核心结构是：

```python
n_obs_steps = 2
horizon = 64
n_action_steps = 32
```

这和 `DiffusionConfig` 一样。含义：

- `n_obs_steps`: 输入给 policy 的 observation 历史长度，默认 2。
- `horizon`: 模型内部生成的完整 action trajectory 长度，默认 64。
- `n_action_steps`: rollout 时实际执行的 action 数量，默认 32。

数据窗口也和 diffusion policy 一致：

```python
observation_delta_indices = list(range(1 - n_obs_steps, 1))
action_delta_indices = list(range(1 - n_obs_steps, 1 - n_obs_steps + horizon))
```

默认 `n_obs_steps=2, horizon=64` 时：

```text
observation_delta_indices = [-1, 0]
action_delta_indices      = [-1, 0, 1, ..., 62]
```

所以以当前 timestep `t` 为中心，dataloader 会返回：

```text
observation: obs[t-1], obs[t]
action:      a[t-1], a[t], a[t+1], ..., a[t+62]
```

推理时先生成完整 `horizon`，再截取：

```python
start = n_obs_steps - 1
end = start + n_action_steps
actions = actions[:, start:end]
```

默认就是执行 `a_hat[t]` 到 `a_hat[t+31]`。

## Flow matching 专有配置

`FlowConfig` 里和 flow matching 直接相关的字段是：

```python
num_inference_steps = 100
timestep_sampling_strategy = "uniform"
timestep_sampling_alpha = 1.5
timestep_sampling_beta = 1.0
timestep_sampling_s = 0.999
sigma_min = 0.0
```

含义：

- `num_inference_steps`: 推理时 Euler ODE 积分步数。
- `timestep_sampling_strategy`: 训练时采样 flow time `t` 的策略，支持 `"uniform"` 和 `"beta"`。
- `timestep_sampling_alpha/beta/s`: beta 采样策略参数。
- `sigma_min`: flow bridge 末端保留的最小噪声系数，默认 0。

当前实现支持两种训练 time 采样：

```python
uniform: t ~ Uniform(0, 1)
beta:    u ~ Beta(alpha, beta), t = s * (1 - u)
```

默认是 uniform。

## 数据窗口和 padding

数据窗口生成方式与 diffusion policy 相同：

1. `lerobot_train.py` 创建 dataset。
2. `datasets/factory.py` 调 `resolve_delta_timestamps(cfg.trainable_config, ds_meta)`。
3. 读取 `FlowConfig` 的 `observation_delta_indices` 和 `action_delta_indices`。
4. `DatasetReader` 按 episode 内绝对 index 查询历史 observation 和 action chunk。
5. episode 边界外的 query 会 clamp，同时生成 `action_is_pad`。

典型训练 batch 形状：

```text
observation.state        (B, n_obs_steps, state_dim)
observation.images       (B, n_obs_steps, num_cameras, C, H, W)
observation.env_state    (B, n_obs_steps, env_dim)   可选
action                   (B, horizon, action_dim)
action_is_pad            (B, horizon)
```

如果 `do_mask_loss_for_padding=True`，`FlowModel.compute_loss` 会用 `~action_is_pad` 屏蔽 episode 边界 padding。

## Processor 做了什么

`processor_flow.py` 和 `processor_diffusion.py` 基本一致。

preprocessor：

```text
RenameObservationsProcessorStep(rename_map={})
AddBatchDimensionProcessorStep()
DeviceProcessorStep(device=config.device)
NormalizerProcessorStep(features=input+output, norm_map=normalization_mapping, stats=dataset_stats)
```

postprocessor：

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

因此 flow 模型训练、采样、Euler 积分都发生在归一化 action 空间里；输出动作会被 postprocessor unnormalize。

## FlowPolicy wrapper

`FlowPolicy` 继承 `PreTrainedPolicy`，结构和 `DiffusionPolicy` 很接近。

初始化：

```python
config.validate_features()
self.flow = FlowModel(config)
self.reset()
```

和 diffusion policy 的一个小区别：`FlowPolicy` 不需要 `require_package("diffusers")`，因为它不依赖 DDPM/DDIM scheduler。

推理队列：

```python
self._queues = {
    OBS_STATE: deque(maxlen=n_obs_steps),
    ACTION: deque(maxlen=n_action_steps),
}
```

如果有 image 或 env_state，也会维护对应 observation queue。

主要接口：

```python
forward(batch)
predict_action_chunk(batch, noise=None)
select_action(batch, noise=None)
```

### forward

训练时：

```python
loss = self.flow.compute_loss(batch)
return loss, None
```

如果有多个 camera image，会先 stack 成统一的 `OBS_IMAGES`：

```python
batch[OBS_IMAGES] = torch.stack([batch[key] for key in image_features], dim=-4)
```

### predict_action_chunk

如果 rollout queue 已经有历史 observation，就从 queue stack 出 `(B, n_obs_steps, ...)`；如果 queue 为空，就直接使用传入 batch。

然后调用：

```python
actions = self.flow.generate_actions(batch, noise=noise)
```

返回截取后的 `(B, n_action_steps, action_dim)`。

### select_action

在线执行时：

1. 如果 batch 里有 label action，先 pop 掉。
2. 把当前 observation 放进 queue。
3. 如果 action queue 空了：
   - 调 `predict_action_chunk`
   - 得到 `(B, n_action_steps, action_dim)`
   - transpose 成 `(n_action_steps, B, action_dim)` 放入 queue
4. 每步 `popleft()` 一个 action。

这和 diffusion policy 的 action chunk 执行方式一致。

## FlowModel 的 observation conditioning

`FlowModel` 复用了 diffusion policy 的 observation conditioning 设计。

每个 observation step 的 condition 由这些东西拼起来：

```text
robot state
image features          可选
environment state       可选
```

然后沿 `n_obs_steps` 展平：

```python
return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)
```

最终传给 U-Net 的 `global_cond` 形状是：

```text
(B, n_obs_steps * per_step_condition_dim)
```

`FlowConfig.validate_features` 明确要求 `observation.state` 存在：

```python
if self.robot_state_feature is None:
    raise ValueError("You must provide 'observation.state' among the inputs.")
```

同时要求至少有 image 或 env_state。实际上 `FlowModel` 也直接使用 `batch[OBS_STATE]`，所以这个要求是合理的。

## 网络结构

`FlowModel` 的网络是：

```python
self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)
```

也就是说 flow policy 没有单独实现新的 flow network，而是直接复用 diffusion policy 的条件 1D U-Net。

输入：

```text
x_t          (B, horizon, action_dim)
t            (B,)
global_cond  (B, global_cond_dim)
```

输出：

```text
pred_velocity (B, horizon, action_dim)
```

虽然类名里叫 `DiffusionConditionalUnet1d`，但这里它被当作 flow velocity field 使用。`diffusion_step_embed_dim` 在 flow 里也继续作为 time embedding 维度使用；`t` 是连续 float，而不是 diffusion scheduler 的整数 timestep。底层 `DiffusionSinusoidalPosEmb` 对 float tensor 也能工作。

U-Net 在 action trajectory 的 horizon 维度上做 1D convolution：

```text
(B, horizon, action_dim)
  -> rearrange to (B, action_dim, horizon)
  -> conditional down/mid/up U-Net
  -> (B, horizon, action_dim)
```

condition 通过 timestep embedding 和 observation `global_cond` concat 后，用 FiLM 注入每个 residual block。

## 训练目标

训练入口：

```text
FlowPolicy.forward(batch)
  -> FlowModel.compute_loss(batch)
```

`compute_loss` 要求：

```text
observation.state      (B, n_obs_steps, state_dim)
observation.images     (B, n_obs_steps, num_cameras, C, H, W) 或 env_state
action                 (B, horizon, action_dim)
action_is_pad          (B, horizon)
```

核心步骤：

1. 编码 observation：

```python
global_cond = self._prepare_global_conditioning(batch)
```

2. clean action trajectory：

```python
trajectory = batch[ACTION]
```

3. 采样 noise：

```python
noise = torch.randn_like(trajectory)
```

4. 采样 flow time：

```python
t = self._sample_timesteps(B, device, dtype)
t_expanded = t.view(-1, 1, 1)
```

5. 构造 flow bridge 上的中间点：

```python
x_t = t * trajectory + (1 - (1 - sigma_min) * t) * noise
```

如果 `sigma_min=0`，公式退化为：

```text
x_t = t * action + (1 - t) * noise
```

即：

```text
t=0: x_t = noise
t=1: x_t = action
```

6. velocity target：

```python
target_velocity = trajectory - (1 - sigma_min) * noise
```

如果 `sigma_min=0`：

```text
target_velocity = action - noise
```

这就是直线路径 `x_t = (1-t) noise + t action` 的常速度。

7. U-Net 预测 velocity：

```python
pred_velocity = self.unet(x_t, t, global_cond=global_cond)
```

8. MSE：

```python
loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")
```

9. 可选 padding mask：

```python
mask = (~batch["action_is_pad"]).unsqueeze(-1)
loss = (loss * mask).sum() / (mask.sum() * action_dim).clamp_min(1)
```

否则直接 `loss.mean()`。

## 推理采样

推理入口：

```text
select_action
  -> predict_action_chunk
    -> FlowModel.generate_actions
      -> _prepare_global_conditioning
      -> conditional_sample
      -> 截取 current 对齐的 n_action_steps
```

`conditional_sample` 是显式 Euler ODE 积分，没有 DDPM/DDIM scheduler。

1. 初始化：

```python
sample = torch.randn((B, horizon, action_dim))
```

或者外部传入固定 `noise`，便于 debug 和复现实验。

2. Euler step：

```python
dt = 1.0 / num_inference_steps
for step in range(num_inference_steps):
    t = step / num_inference_steps
    velocity = self.unet(sample, t, global_cond=global_cond)
    sample = sample + dt * velocity
```

默认 `num_inference_steps=100`。

3. 返回完整 `horizon` trajectory。

4. `generate_actions` 截取实际执行段：

```python
start = n_obs_steps - 1
end = start + n_action_steps
actions = actions[:, start:end]
```

默认从 index 1 取 32 步，也就是当前时刻及未来动作。

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

flow 起点:
  noise trajectory, shape 与 action trajectory 相同

flow 终点:
  action trajectory

实际执行:
  a_hat[t], a_hat[t+1], ..., a_hat[t+31]
```

## 和 DiffusionPolicy 的相同点

两者相同点很多，因为 `flow` 明确是基于 diffusion policy 结构改出来的：

1. 都是 action chunk policy。
2. 默认都是 `n_obs_steps=2, horizon=64, n_action_steps=32`。
3. 数据窗口和时间对齐方式相同。
4. 都用 `action_is_pad` 标记 episode 边界 padding。
5. 都用相同 normalization mapping：

```text
VISUAL -> MEAN_STD
STATE  -> MIN_MAX
ACTION -> MIN_MAX
```

6. 都复用 `DiffusionRgbEncoder`。
7. 都用 `DiffusionConditionalUnet1d` 在 action horizon 维度上建模。
8. 都用 observation history 编成全局 condition，然后 flatten 成 `(B, n_obs_steps * D)`。
9. 推理时都维护 observation queue 和 action queue。
10. 都一次预测 chunk，执行 `n_action_steps` 个 action 后再重采样。

## 和 DiffusionPolicy 的关键不同

### 1. 学习目标不同

Diffusion policy 学的是 denoising：

```text
clean action trajectory -> scheduler 加噪 -> noisy trajectory
U-Net(noisy trajectory, diffusion timestep, condition)
预测 epsilon 或 clean sample
loss = MSE(pred, epsilon 或 clean action)
```

Flow policy 学的是 velocity field：

```text
noise trajectory 和 action trajectory 之间插值出 x_t
U-Net(x_t, continuous t, condition)
预测 dx_t/dt
loss = MSE(pred_velocity, action - noise)
```

所以 flow policy 的输出有直接的“速度场”含义；diffusion policy 的默认输出是 noise prediction。

### 2. time 的语义不同

Diffusion policy：

```text
timestep 是 scheduler 的离散 integer diffusion timestep
范围大致是 0 ... num_train_timesteps-1
```

Flow policy：

```text
t 是连续 flow time
范围是 [0, 1]
```

虽然两者都喂给同一个 sinusoidal embedding 模块，但语义不同。

### 3. 采样方式不同

Diffusion policy 采样依赖 `diffusers` scheduler：

```python
noise_scheduler.set_timesteps(num_inference_steps)
for t in noise_scheduler.timesteps:
    model_output = unet(sample, t, cond)
    sample = noise_scheduler.step(model_output, t, sample).prev_sample
```

Flow policy 采样是手写 Euler ODE：

```python
dt = 1 / num_inference_steps
for step in range(num_inference_steps):
    t = step / num_inference_steps
    velocity = unet(sample, t, cond)
    sample = sample + dt * velocity
```

所以 flow policy 不需要 `diffusers` 依赖，也没有 DDPM/DDIM scheduler 的 beta schedule、prediction type、clip sample 等概念。

### 4. 配置项不同

DiffusionConfig 有这些 diffusion scheduler 字段：

```python
noise_scheduler_type
num_train_timesteps
beta_schedule
beta_start
beta_end
prediction_type
clip_sample
clip_sample_range
num_inference_steps
```

FlowConfig 替换成 flow matching 字段：

```python
num_inference_steps
timestep_sampling_strategy
timestep_sampling_alpha
timestep_sampling_beta
timestep_sampling_s
sigma_min
```

### 5. loss target 不同

Diffusion 默认：

```python
target = eps
```

Flow 默认：

```python
target_velocity = action - noise
```

这点对 kinetic loss 很关键。`bc_only.py` 那类 kinematic residual 是约束 velocity field 的导数，和 flow policy 的 `pred_velocity` 更自然对应；diffusion policy 要先决定约束 noise prediction、sample prediction，还是把它转换成 velocity。

### 6. 是否有直接 velocity field

Flow policy 有：

```python
pred_velocity = self.unet(x_t, t, global_cond)
```

Diffusion policy 没有直接 velocity field，默认是：

```python
pred = self.unet(noisy_trajectory, timesteps, global_cond)
target = eps
```

因此如果目标是验证 action chunk 的 kinetic loss，flow policy 是更直接的落点。

### 7. 推理步数默认不同

当前配置：

```text
DiffusionConfig.num_inference_steps = None
FlowConfig.num_inference_steps = 100
```

Diffusion 如果不指定推理步数，会默认等于 scheduler 的 `num_train_timesteps`。Flow 必须是正整数，默认 100。

### 8. validate_features 有细微差异

FlowConfig 明确要求 `observation.state`：

```python
if self.robot_state_feature is None:
    raise ValueError(...)
```

DiffusionConfig 的注释也说需要 `observation.state`，而 model 实现也直接访问 `batch[OBS_STATE]`，但当前 `validate_features` 里主要检查 image/env_state。实际使用上，两者都应提供 `observation.state`。

## 对 kinetic loss 的意义

你关心的是在 flow policy 的 action chunk 里加入 `bc_only.py` 那种 kinematic loss。当前 `flow` policy 比 `diffusion` policy 更合适，原因是：

1. 它已经直接学习 action chunk 上的 velocity field。
2. 训练中已经构造了：

```python
x_t = t * action + (1 - t) * noise
pred_velocity = unet(x_t, t, global_cond)
```

3. 这和 `bc_only.py` 的 flow 部分结构同源：

```python
v_pred = actor_flow(observation, x_t, t)
```

4. 如果要加入 kinematic loss，大概率需要对 `pred_velocity` 做 JVP：

```text
d v / d condition  * condition_dot
d v / d x_t        * x_t_dot
```

5. LeRobot flow 当前没有 `next_actions` 或 `delta_a` 字段。可以从 action chunk 内部构造：

```text
delta_a[:, :-1] = action[:, 1:] - action[:, :-1]
```

但最后一个位置没有 next action；需要丢掉、置零，或扩展 action window 到 `horizon + 1`。

6. `use_jvp_ak=True` 时，action difference 同时作为 target 和 JVP 方向，真实机器人动作噪声会更敏感。更稳的第一版是先做 state/global-condition 方向 JVP，即类似 `use_jvp_ak=False`。

7. 图像输入的 tangent 不好定义。更可控的实现方式是对 `global_cond` 或 proprio/state 子向量求 JVP，而不是直接对 pixel 做 JVP。

## 最小实验建议

如果要在这个 flow policy 上做 kinetic loss 验证，建议路线：

1. 先不碰网络结构，保留 `DiffusionConditionalUnet1d`。
2. 在 `FlowModel.compute_loss` 中，在原 flow MSE 外新增 `kinematic_loss`。
3. 第一版只做 action chunk 内差分：

```text
a_dot = action[:, 1:] - action[:, :-1]
```

并只在前 `horizon - 1` 个 action token 上算 kinematic loss。

4. mask 同时考虑：

```text
~action_is_pad[:, :-1]
~action_is_pad[:, 1:]
```

避免跨 episode padding 的差分。

5. 先实现 `lambda_kinematic` 和 `use_action_jvp=False`。
6. 跑 `lambda_kinematic=0` 对照，确认完全复现 baseline。
7. 再测小权重，例如 `0.01, 0.1, 1.0`。
8. 如果要开 action-input JVP，优先配合平滑后的 `delta_a` 或更严格的 episode 内 mask。

## 一句话总结

当前 `flow` policy 可以看成：

```text
DiffusionPolicy 的 action chunk / history / U-Net 框架
+ flow matching velocity objective
+ Euler ODE sampling
- diffusers scheduler
- epsilon/sample denoising target
```

所以它和 diffusion policy 的工程骨架非常接近，但数学目标不同。对你想验证的 kinematic loss 来说，`flow` 是更自然的切入点，`diffusion` 更适合作为架构参考和 baseline，不适合作为第一版 kinetic loss 的直接承载对象。

