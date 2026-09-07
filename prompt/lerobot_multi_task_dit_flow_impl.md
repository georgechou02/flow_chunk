# LeRobot Multi-Task DiT Flow Matching 实现梳理

本文档解释当前仓库里的 `multi_task_dit` policy 在 `objective="flow_matching"` 时是怎么实现的。这里只看 flow matching 部分，不展开 diffusion objective。

对应代码主要在：

- `src/lerobot/policies/multi_task_dit/configuration_multi_task_dit.py`
- `src/lerobot/policies/multi_task_dit/modeling_multi_task_dit.py`
- `src/lerobot/policies/multi_task_dit/processor_multi_task_dit.py`
- `src/lerobot/policies/factory.py`
- 对比代码：`/home/zhouzhi/code_store/higher-order/higher-order/agents/bc_only.py`
- 对比数据采样：`/home/zhouzhi/code_store/higher-order/higher-order/utils/datasets.py`
- 对比网络：`/home/zhouzhi/code_store/higher-order/higher-order/utils/networks.py`

## 总体结构

`multi_task_dit` 是一个 action chunk policy。它的模型主体是 transformer，支持两个 objective：

```text
objective="diffusion"       -> DiffusionObjective
objective="flow_matching"   -> FlowMatchingObjective
```

flow 部分不是单独的 policy 类，而是同一个 `MultiTaskDiTPolicy` 在初始化时根据 config 选择 `FlowMatchingObjective`：

```text
MultiTaskDiTConfig
  定义 observation/action 窗口、flow matching 参数、transformer 参数、
  CLIP vision/text encoder 参数、normalization、optimizer 等

MultiTaskDiTPolicy
  LeRobot policy wrapper，负责 processor 对接、observation/action queue、
  forward、select_action、predict_action_chunk

ObservationEncoder
  把 robot state、camera images、language tokens 编成一个全局 conditioning vector

DiffusionTransformer
  名字叫 DiffusionTransformer，但在 flow_matching 下它预测 velocity field
  v_theta(x_t, t, condition)

FlowMatchingObjective
  训练时构造 noise-data interpolation，监督 velocity
  推理时从 Gaussian noise 出发，用 Euler 或 RK4 积分到 action chunk
```

在 factory 里，`policy.type=multi_task_dit` 会映射到：

- `get_policy_class("multi_task_dit") -> MultiTaskDiTPolicy`
- `make_policy_config("multi_task_dit") -> MultiTaskDiTConfig`
- processor 会走 `make_multi_task_dit_pre_post_processors`

注意：`multi_task_dit` 的 flow 模式还是同一个 policy type，只是 config 里要设置：

```python
objective = "flow_matching"
```

## 配置含义

`MultiTaskDiTConfig` 默认核心窗口是：

```python
n_obs_steps = 2
horizon = 32
n_action_steps = 24
```

含义：

- `n_obs_steps`: policy 条件里使用多少帧 observation 历史，默认 2。
- `horizon`: 模型一次生成的完整 action trajectory 长度，默认 32。
- `n_action_steps`: 每次 action queue 真正拿去执行的动作数量，默认 24。

和 diffusion policy 一样，action window 会和 observation window 对齐：

```python
observation_delta_indices = list(range(1 - n_obs_steps, 1))
action_delta_indices = list(range(1 - n_obs_steps, 1 - n_obs_steps + horizon))
```

默认 `n_obs_steps=2, horizon=32` 时：

```text
observation_delta_indices = [-1, 0]
action_delta_indices      = [-1, 0, 1, ..., 30]
```

所以 dataloader 会以中心 timestep `t` 返回：

```text
observation: [t-1, t]
action:      [t-1, t, t+1, ..., t+30]
```

推理时完整生成 `(B, horizon, action_dim)` 后，只取：

```python
start = n_obs_steps - 1
end = start + n_action_steps
actions = actions[:, start:end]
```

默认就是从 index 1 开始取 24 个动作，也就是当前时刻 `t` 到 `t+23`。完整 horizon 里的 index 0 是和历史 observation 对齐的上一时刻动作。

`drop_n_last_frames` 默认自动计算：

```python
horizon - n_action_steps - n_obs_steps + 1
```

默认值是：

```text
32 - 24 - 2 + 1 = 7
```

它不是模型内部逻辑，而是训练采样时用来减少 episode 末尾 action window padding 的配置。

## Flow Matching 相关配置

flow matching 专用参数是：

```python
sigma_min = 0.0
num_integration_steps = 100
integration_method = "euler"       # "euler" or "rk4"
timestep_sampling_strategy = "beta" # "uniform" or "beta"

timestep_sampling_s = 0.999
timestep_sampling_alpha = 1.5
timestep_sampling_beta = 1.0
```

训练时采样连续时间 `t`：

- `uniform`: 直接从 `[0, 1]` 均匀采样。
- `beta`: 先采样 `u ~ Beta(alpha, beta)`，再设 `t = s * (1 - u)`。

推理时从 `t=0` 积分到 `t=1`：

- `num_integration_steps`: ODE 积分步数，默认 100。
- `integration_method`: 支持 Euler 和 RK4。

## 数据窗口是怎么来的

`multi_task_dit` 复用 LeRobot 的 policy config delta indices 机制。policy 自己不手动去 dataset 里取历史帧，而是 config 提供：

```text
observation_delta_indices
action_delta_indices
reward_delta_indices = None
```

dataset/factory 会把这些 frame offset 转成 delta timestamps，然后 `LeRobotDataset` / `DatasetReader` 按窗口返回 batch。

训练 batch 的典型形状是：

```text
observation.state                 (B, n_obs_steps, state_dim)
observation.images                (B, n_obs_steps, num_cameras, C, H, W)  可选
observation.language.tokens       (B, seq_len)                            可选但当前模型通常需要
observation.language.attention_mask (B, seq_len)                          可选但当前模型通常需要
action                            (B, horizon, action_dim)
action_is_pad                     (B, horizon)
```

`action_is_pad` 用来标记 episode 边界外 clamp/padding 出来的 action。`FlowMatchingObjective` 只有在：

```python
do_mask_loss_for_padding=True
```

并且 batch 里有 `action_is_pad` 时，才会用它 mask loss。默认配置里 `do_mask_loss_for_padding=False`。

## Processor 做了什么

`processor_multi_task_dit.py` 定义 preprocessor 和 postprocessor。

preprocessor 顺序：

```text
RenameObservationsProcessorStep(rename_map={})
AddBatchDimensionProcessorStep()
TokenizerProcessorStep(...)
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

所以 flow matching 训练和 ODE 采样都发生在归一化 action 空间里。`select_action` 返回的动作会在 postprocessor 里 unnormalize 回真实动作空间。

和 diffusion policy 相比，`multi_task_dit` preprocessor 多了一个关键步骤：

```text
TokenizerProcessorStep
```

它使用 `config.text_encoder_name` 对任务语言做 CLIP tokenizer，生成：

```text
OBS_LANGUAGE_TOKENS
OBS_LANGUAGE_ATTENTION_MASK
```

## MultiTaskDiTPolicy wrapper

`MultiTaskDiTPolicy.__init__` 做几件事：

```python
require_package("transformers", extra="multi_task_dit")
require_package("diffusers", extra="multi_task_dit")
config.validate_features()

self.observation_encoder = ObservationEncoder(config)
conditioning_dim = self.observation_encoder.conditioning_dim
self.noise_predictor = DiffusionTransformer(config, conditioning_dim=conditioning_dim)

if config.is_flow_matching:
    self.objective = FlowMatchingObjective(...)
```

这里 `noise_predictor` 这个名字来自 diffusion 场景。在 `flow_matching` 下，它实际预测的是 velocity，不是 noise。

### queue

`reset()` 会创建：

```python
self._queues = {
    OBS_STATE: deque(maxlen=n_obs_steps),
    ACTION: deque(maxlen=n_action_steps),
}
```

如果有 image features，还会加：

```python
self._queues[OBS_IMAGES] = deque(maxlen=n_obs_steps)
```

language tokens 没有进 queue。推理时 language 直接来自当前 batch，`ObservationEncoder` 会把同一个 text feature expand 到每个 observation step。

### forward

训练入口：

```python
forward(batch)
  -> _prepare_batch(batch)
  -> conditioning_vec = observation_encoder.encode(batch)
  -> loss = objective.compute_loss(noise_predictor, batch, conditioning_vec)
```

如果有多个 camera key，`_prepare_batch` 会先把它们 stack 成统一的 `OBS_IMAGES`：

```python
batch[OBS_IMAGES] = torch.stack([batch[key] for key in image_features], dim=-4)
```

然后 `FlowMatchingObjective.compute_loss` 只看：

```text
batch[ACTION]
conditioning_vec
action_is_pad   可选
```

### select_action

在线 rollout 时每个 env step 调一次：

1. 如果 batch 里有 `action` label，先移除。
2. `_prepare_batch` stack image。
3. `populate_queues` 把当前 observation 放进 queue。
4. 如果 action queue 空了：
   - 用 queue 里的 observation history 调 `predict_action_chunk`
   - 得到 `(B, n_action_steps, action_dim)`
   - transpose 成 `(n_action_steps, B, action_dim)` 放入 queue
5. 每次 `popleft()` 一个 action 返回。

也就是说它不是每个 env step 都重新采样 action chunk，而是 action queue 用完后才重新生成一段。

## ObservationEncoder

`ObservationEncoder` 把多模态 observation 编成一个扁平 conditioning vector。

### state

state 直接进入 conditioning：

```python
conditioning_feats.append(batch[OBS_STATE])
```

形状：

```text
(B, n_obs_steps, state_dim)
```

### vision

vision encoder 使用 HuggingFace `CLIPVisionModel`：

```text
CLIPVisionModel.from_pretrained(config.vision_encoder_name)
```

`CLIPVisionEncoder.forward` 取 CLIP 最后一层的 CLS token：

```python
outputs = self.model(pixel_values=x)
cls_token = outputs.last_hidden_state[:, 0]
return cls_token.reshape(b, embed_dim, 1, 1)
```

然后 flatten 成单个 image feature。

多相机有两种模式：

1. `use_separate_rgb_encoder_per_camera=True`
   - 每个 camera 一个独立 CLIP vision encoder。
2. `False` 默认
   - 所有 camera 共享同一个 CLIP vision encoder。

image 在进 CLIP 之前会按 config 做可选 resize/crop：

```text
image_resize_shape
image_crop_shape
image_crop_is_random
```

训练时如果 `image_crop_is_random=True`，用 random crop；eval 时用 center crop。

### text

text encoder 使用 HuggingFace `CLIPTextModel`：

```python
self.text_encoder = CLIPTextModel.from_pretrained(model_name)
```

CLIP text encoder 参数被冻结：

```python
for param in self.text_encoder.parameters():
    param.requires_grad = False
```

后面接一个可训练 projection：

```python
self.projection = nn.Linear(text_embed_dim, config.hidden_dim)
```

`encode` 里如果 batch 有 language tokens：

```python
text_features = self.text_encoder(input_ids, attention_mask)
text_features = text_features.unsqueeze(1).expand(-1, n_obs_steps, -1)
conditioning_feats.append(text_features)
```

也就是说同一个任务语言 feature 会复制到每个 observation step。

注意当前实现里 `_setup_vector_output` 会无条件把 `text_dim` 加进 `conditioning_dim`：

```python
total_dim += self.text_dim
```

所以这个模型通常假设 batch 里有 language tokens。如果实际 batch 没有 language tokens，`encode()` 拼出来的 conditioning vector 维度会比 `DiffusionTransformer` 初始化时期望的维度小。

### conditioning vector 形状

每个 observation step 的 feature 是：

```text
state feature
+ image feature        可选
+ text feature         通常需要
```

最后：

```python
combined_features = torch.cat(conditioning_feats, dim=-1)
return combined_features.flatten(start_dim=1)
```

所以输出形状是：

```text
(B, n_obs_steps * per_step_condition_dim)
```

## DiffusionTransformer 在 flow 里做什么

`DiffusionTransformer` 是 action-sequence transformer。它的输入输出是：

```text
input x            (B, horizon, action_dim)
input timestep     (B,)
input condition    (B, conditioning_dim)

output             (B, horizon, action_dim)
```

在 diffusion objective 下，output 可以是 predicted noise；在 flow matching 下，output 是 predicted velocity：

```text
v_theta(x_t, t, condition)
```

### action token

每个 action step 是一个 token。先用 linear 投影到 transformer hidden dim：

```python
hidden_seq = self.input_proj(x)
```

形状：

```text
(B, horizon, action_dim) -> (B, horizon, hidden_dim)
```

如果 `use_positional_encoding=True`，会加可学习 absolute position embedding。默认是 `False`。

默认 `use_rope=True`，所以 transformer attention 里用 RoPE 表示 action token 的相对位置。

### timestep 和 condition

flow matching 的 `t` 是 `[0, 1]` 上的连续浮点数。它会过 sinusoidal embedding 和 MLP：

```python
timestep_features = self.time_mlp(timestep)
```

然后和 observation conditioning 拼起来：

```python
cond_features = torch.cat([timestep_features, conditioning_vec], dim=-1)
```

这个 `cond_features` 不是作为 token 进入 transformer，而是作为每层 transformer block 的 AdaLN-Zero 调制条件。

### TransformerBlock

每层是 DiT-style block：

```text
LayerNorm without affine
AdaLN modulation from cond_features
self-attention over action tokens
residual with gate
LayerNorm without affine
AdaLN modulation from cond_features
MLP
residual with gate
```

`adaLN_modulation` 输出 6 组参数：

```python
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
```

这些参数都来自：

```python
nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * hidden_size))
```

初始化时把最后一层 AdaLN modulation 的 weight 和 bias 置零：

```python
nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
```

这是 AdaLN-Zero 的常见初始化方式，让 block 初始时更接近 identity/residual 结构。

## FlowMatchingObjective 训练 loss

训练入口：

```text
MultiTaskDiTPolicy.forward
  -> FlowMatchingObjective.compute_loss
```

核心代码：

```python
data = batch[ACTION]
noise = torch.randn_like(data)
t = self._sample_timesteps(batch_size, device)
t_expanded = t.view(-1, 1, 1)

x_t = t_expanded * data + (1 - (1 - sigma_min) * t_expanded) * noise
target_velocity = data - (1 - sigma_min) * noise

predicted_velocity = model(x_t, t, conditioning_vec=conditioning_vec)
loss = F.mse_loss(predicted_velocity, target_velocity, reduction="none")
```

设：

```text
x_0 = noise
x_1 = data
```

当前实现的 interpolation path 是：

```text
x_t = t * x_1 + (1 - (1 - sigma_min) * t) * x_0
```

对 `t` 求导，得到 velocity target：

```text
dx_t / dt = x_1 - (1 - sigma_min) * x_0
```

当默认 `sigma_min=0.0` 时：

```text
x_t = (1 - t) * noise + t * data
target_velocity = data - noise
```

这就是标准的 rectified-flow/flow-matching BC 形式：模型学习在任意中间点 `x_t` 上，把噪声 action trajectory 推向数据 action trajectory 的速度场。

### padding mask

如果打开：

```python
do_mask_loss_for_padding=True
```

并且 batch 里有：

```python
action_is_pad
```

则 loss 会屏蔽 padding action：

```python
mask = ~batch["action_is_pad"].unsqueeze(-1)
num_valid = mask.sum() * loss.shape[-1]
loss = (loss * mask).sum() / num_valid.clamp_min(1)
```

否则直接：

```python
loss.mean()
```

### 这里没有什么

`multi_task_dit` 的 flow loss 只有 velocity matching。当前实现没有：

- action MSE BC loss
- `delta_a` supervision
- `next_actions`
- state JVP
- action JVP
- kinematic residual
- value/Q/advantage 相关项

行为克隆的作用来自 flow matching 本身：数据端点 `x_1` 就是 expert action chunk。

## FlowMatchingObjective 推理采样

推理入口：

```text
select_action
  -> predict_action_chunk
    -> _generate_actions
      -> observation_encoder.encode
      -> FlowMatchingObjective.conditional_sample
      -> slice current-aligned n_action_steps
```

`conditional_sample` 先从标准高斯采样完整 action trajectory：

```python
x = torch.randn((batch_size, horizon, action_dim), dtype=dtype, device=device)
```

然后构造时间网格：

```python
time_grid = torch.linspace(0, 1, num_steps + 1, device=device)
```

如果 `integration_method="euler"`：

```python
for i in range(len(time_grid) - 1):
    t_batch = torch.full((B,), t_scalar)
    velocity = model(x, t_batch, conditioning_vec=conditioning_vec)
    x = x + dt * velocity
```

如果 `integration_method="rk4"`，则标准四阶 Runge-Kutta：

```text
k1 = f(x, t)
k2 = f(x + dt*k1/2, t + dt/2)
k3 = f(x + dt*k2/2, t + dt/2)
k4 = f(x + dt*k3, t + dt)
x = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
```

返回的是完整：

```text
(B, horizon, action_dim)
```

之后 policy wrapper 再截取：

```python
actions = actions[:, n_obs_steps - 1 : n_obs_steps - 1 + n_action_steps]
```

最后这些 action 被放进 action queue，逐步执行。

当前 `FlowMatchingObjective` 自己不做 `[-1, 1]` clip。动作尺度主要依赖 processor 的 action normalization 和 postprocessor 的 unnormalization。

## 和 `bc_only.py` flow policy chunk 的区别

`multi_task_dit` 的 flow matching 和 `bc_only.py` 里的 `policy_type="flow"` + `action_chunking=True` 都是在学从噪声到 action chunk 的 flow，但实现目标和假设差别很大。

### 1. 框架和使用场景不同

`multi_task_dit` 是 LeRobot policy：

```text
PyTorch
PreTrainedPolicy
processor pipeline
LeRobotDataset window
CLIP vision/text conditioning
action queue
```

`bc_only.py` 是 higher-order 目录里的 JAX/Flax offline RL/BC agent：

```text
JAX/Flax
ml_collections config
ReplayBuffer/Dataset.sample_sequence
low-dimensional continuous-control observation
optional custom visual encoder
manual evaluation action queue
```

### 2. flow 的网络结构不同

`multi_task_dit`：

```text
input:  (B, horizon, action_dim)
model:  transformer over horizon action tokens
output: (B, horizon, action_dim)
```

它保留 action chunk 的时间维度，transformer self-attention 直接建模 chunk 内不同 action step 的关系。

`bc_only.py` chunk flow：

```python
batch_actions = jnp.reshape(batch["actions"], (batch_size, -1))
full_action_dim = horizon_length * action_dim
```

网络是 `ActorVectorField`：

```text
input:  concat(observation, flattened_action_chunk, t)
model:  MLP
output: flattened velocity, shape (B, horizon_length * action_dim)
```

它把 action chunk flatten 成一个大向量。chunk 内时间结构只体现在向量维度顺序上，网络本身不是 temporal transformer。

### 3. action window 对齐不同

`multi_task_dit` 默认：

```text
observation: [t-1, t]
action:      [t-1, t, ..., t+30]
execute:     [t, ..., t+23]
```

因为它有 `n_obs_steps`，完整 horizon 包含和历史 observation 对齐的旧 action，再从 `n_obs_steps - 1` 开始截取当前动作。

`bc_only.py` 的 chunk 数据来自 `Dataset.sample_sequence(batch_size, horizon_length)`：

```text
observation: s_t
actions:     [a_t, a_{t+1}, ..., a_{t+H-1}]
```

没有 `n_obs_steps` 的历史 observation 对齐，也没有额外包含 `a_{t-1}`。它的 action chunk 是从当前采样 index 直接开始的未来动作段。

### 4. observation 条件不同

`multi_task_dit` conditioning 是：

```text
state history
+ CLIP image features
+ CLIP text feature
flatten over n_obs_steps
```

形状是：

```text
(B, n_obs_steps * per_step_condition_dim)
```

然后通过 AdaLN-Zero 调制每层 transformer。

`bc_only.py` flow chunk 通常是：

```text
observation s_t
```

可选 encoder 后，直接和 flattened action chunk、time concat 进 MLP：

```python
inputs = jnp.concatenate([observations, actions, times], axis=-1)
```

没有 language conditioning，也没有 LeRobot 那套多帧 state/image processor。

### 5. 训练目标不同

`multi_task_dit` flow loss 是纯 flow matching：

```text
L = mean || v_theta(x_t, t, c) - (data - (1 - sigma_min) * noise) ||^2
```

最多加一个 padding mask。

`bc_only.py` flow chunk loss 是两部分：

```text
actor_loss = bc_weight * bc_flow_loss + lambda_flow_k * kinematic_loss
```

其中 `bc_flow_loss` 类似基础 flow matching：

```python
x_0 = normal_noise
x_1 = flattened_action_chunk
t = uniform(0, 1)
x_t = (1 - t) * x_0 + t * x_1
vel_target = x_1 - x_0
v_pred = actor_flow(observation, x_t, t)
```

如果 `action_chunking=True`，它会 reshape 回 `(B, H, action_dim)` 后用 `valid` mask：

```python
bc_flow_loss = mean(reshape((v_pred - vel_target)^2, (B, H, A)) * valid[..., None])
```

然后额外有 high-order/kinematic supervision。

### 6. `bc_only.py` 多了 kinematic/JVP loss

`bc_only.py` 会先构造 action dynamics target：

```python
if "delta_a" in batch:
    a_dot_data = batch["delta_a"]
elif "next_actions" in batch:
    a_dot_data = batch["next_actions"] - batch["actions"]
else:
    a_dot_data = zeros_like(actions)
```

chunk 模式下：

```python
s_dot = batch["next_observations"][:, 0] - batch["observations"]
a_dot_data = reshape(delta_a, (B, H * action_dim))
valid = batch["valid"]
not_done = valid * batch["masks"]
```

它对 flow vector field 关于 state 做 JVP：

```python
v_pred, v_s_dot = jax.jvp(flow_vector_field_fn, (observations,), (s_dot,))
```

如果 `use_jvp_ak=True`，还会对 action 输入做 JVP：

```python
_, v_a_k_dot = jax.jvp(flow_vector_field_action_fn, (x_t,), (a_dot_data,))
kinematic_residual = v_s_dot + t * v_a_k_dot - a_dot_data
```

否则：

```python
kinematic_residual = v_s_dot - a_dot_data
```

最后：

```python
kinematic_loss = mean(sum(kinematic_residual^2 over action_dim) * not_done)
```

`multi_task_dit` 完全没有这部分。它不会使用 `delta_a`、`next_actions`、`next_observations - observations`，也不会约束 velocity field 对 state/action 的导数。

这是两者最核心的区别：
`multi_task_dit` 是标准条件生成式 flow matching；`bc_only.py` 是 flow BC 再叠加一个和 action dynamics 一致性有关的高阶监督项。

### 7. sequence mask 和 episode 边界处理不同

`multi_task_dit` 依赖 LeRobot dataset 的 `action_is_pad`。只有 `do_mask_loss_for_padding=True` 时才在 loss 里屏蔽 padding。

`bc_only.py` 的 chunk batch 来自 `Dataset.sample_sequence`，里面显式构造：

```text
valid    (B, H)
masks    (B, H)
terminals (B, H)
delta_a  (B, H, action_dim)
```

`valid` 会在 sequence 跨过 episode 结束后变成 0，`delta_a` 在 terminal/timeout 位置也会置 0。chunk flow loss 和 kinematic loss 都会用这些 mask 避免跨 episode 的错误监督。

### 8. 推理积分不同

`multi_task_dit`：

```text
initial x: N(0, I), shape (B, horizon, action_dim)
time grid: linspace(0, 1, num_integration_steps + 1)
solver: Euler or RK4
default steps: 100
clip: FlowMatchingObjective 内部不 clip
```

`bc_only.py`：

```text
initial x: N(0, I), shape (B, horizon_length * action_dim)
solver: Euler only
default flow_steps: 10
clip: compute_flow_actions 末尾 jnp.clip(actions, -1, 1)
```

`bc_only.py` 在 eval 时会把 flat action chunk reshape 回：

```python
action_chunk = np.array(action_chunk).reshape(-1, action_dim)
```

然后用 evaluation 里的 `action_queue` 逐个执行。这个 queue 是 evaluation 代码手写的，不是 LeRobot `PreTrainedPolicy` 的内置 queue。

## 简短对照表

| 维度                   | `multi_task_dit` flow                                   | `bc_only.py` flow chunk                      |
| ---------------------- | ------------------------------------------------------- | -------------------------------------------- |
| 框架                   | PyTorch + LeRobot `PreTrainedPolicy`                    | JAX/Flax agent                               |
| policy 开关            | `policy.type=multi_task_dit`, `objective=flow_matching` | `policy_type="flow"`, `action_chunking=True` |
| action 表示            | `(B, horizon, action_dim)`                              | flatten 成 `(B, H * action_dim)`             |
| 网络                   | DiT transformer over action tokens                      | `ActorVectorField` MLP                       |
| 条件                   | state history + CLIP image + CLIP text                  | 当前 observation，可选 encoder               |
| 时间窗口               | observation/action delta indices 对齐，包含历史 offset  | `sample_sequence` 从当前 index 取未来 H 步   |
| 基础 loss              | flow matching velocity MSE                              | flow matching velocity MSE                   |
| 额外 loss              | 无                                                      | kinematic/JVP high-order loss                |
| action dynamics target | 不使用                                                  | 使用 `delta_a` 或 `next_actions - actions`   |
| state/action JVP       | 无                                                      | 有 state JVP，可选 action JVP                |
| padding/valid mask     | 可选 `action_is_pad`                                    | `valid * masks` 显式进入 loss                |
| 推理 solver            | Euler 或 RK4                                            | Euler                                        |
| 默认采样步数           | 100                                                     | 10                                           |
| 输出执行               | LeRobot policy action queue                             | evaluation 代码手写 action queue             |

## 可以这样理解

`multi_task_dit` flow 部分实现的是一个多模态条件 action chunk generator：

```text
condition = encode(state history, images, language)
noise action trajectory -> ODE flow -> normalized expert-like action trajectory
```

它关注的是“给定多模态任务条件，生成一段动作”。

`bc_only.py` 的 flow chunk 更像一个低维连续控制实验里的 flow BC agent：

```text
observation + noisy flattened action chunk + time -> flattened velocity
```

并且它额外把 action chunk 随 state 演化的一阶变化纳入训练：

```text
state tangent s_dot
action tangent delta_a
JVP of vector field
kinematic residual
```

所以两者虽然都可以“一次生成 action chunk”，但 `multi_task_dit` 是标准条件 flow matching 生成模型，`bc_only.py` 是 flow matching BC 加高阶动力学一致性正则的 JAX 实验实现。
