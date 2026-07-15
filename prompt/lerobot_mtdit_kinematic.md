# 完成标注

- `src/lerobot/policies/multi_task_dit/configuration_multi_task_dit.py`: 第 60-62、177-180 行新增 kinematic/JVP 配置与校验。
- `src/lerobot/policies/multi_task_dit/modeling_multi_task_dit.py`: 第 30-44、75-92 行新增 SDPA math-only context helper 和 disabled-autocast context helper，用于避免 `torch.func.jvp` 触发不支持 forward AD 的 flash attention kernel，以及 AMP 半精度反传 dtype 不一致。
- `src/lerobot/policies/multi_task_dit/modeling_multi_task_dit.py`: 第 221-227、445-495、853-1012 行新增 `encode_steps`、flow kinematic/JVP loss、mask、日志输出、no-grad 兼容逻辑，并在 `_compute_kinematic_loss()` 的 JVP 分支临时关闭 autocast 且切到 math attention。
- `train_multitask_dit_flow_resnet.sh`: 第 92-94、172-174、205-207 行补入训练脚本的 kinematic 参数开关与 LIBERO `sample_frequency=10.0` 默认值。
- `prompt/lerobot_mtdit_kinematic.md`: 第 1-7 行为本次完成标注。

# Prompt: 给 `multi_task_dit` flow matching 引入 kinematic loss

你要修改的代码在：

- `/home/zhouzhi/code_store/higher-order/lerobot/src/lerobot/policies/multi_task_dit/configuration_multi_task_dit.py`
- `/home/zhouzhi/code_store/higher-order/lerobot/src/lerobot/policies/multi_task_dit/modeling_multi_task_dit.py`
- 必要时少量调整测试或训练日志相关代码

参考实现是：

- `/home/zhouzhi/code_store/higher-order/higher-order/agents/bc_only.py`
- 重点看 `BCOnlyAgent.actor_loss_flow`

目标：把 `bc_only.py` 里 flow BC 的 kinematic/JVP loss 思路引入 LeRobot 的 `multi_task_dit`，只作用在 `objective="flow_matching"` 路径上。不要改 diffusion objective 的行为。

## 先明确设计选择

当前 `multi_task_dit` 已经支持：

```python
vision_encoder_type = "clip" | "resnet"
```

ResNet 路径是：

```text
ResNetVisionEncoder
  torchvision ResNet backbone
  SpatialSoftmax
  Linear + ReLU
```

action head 实际吃的不是 raw image，而是：

```text
ObservationEncoder.encode(batch) -> conditioning_vec
DiffusionTransformer(x_t, t, conditioning_vec)
```

所以这里的 kinematic loss 里的 `s` 不取 raw pixels，也不只取 proprio raw state，而是取“输入 action head 的 encoded observation / conditioning state”。

具体说：

1. `ObservationEncoder` 先得到每个 observation step 的 encoded feature：

```text
conditioning_steps: (B, n_obs_steps, per_step_condition_dim)
```

这个 per-step feature 包含：

```text
normalized robot state
+ encoded image feature    如果有 image
+ encoded text feature     如果有 language token；因为 text 被 expand 到每个 obs step，finite difference 通常为 0
```

2. `conditioning_vec` 仍然是原来的 flatten：

```text
conditioning_vec = conditioning_steps.flatten(start_dim=1)
shape: (B, n_obs_steps * per_step_condition_dim)
```

3. kinematic loss 的 `s_dot` 用 action head 输入域里的 finite difference：

```text
s_dot_step = finite_difference(conditioning_steps[:, -1], conditioning_steps[:, -2])
```

然后构造成和 `conditioning_vec` 同 shape 的 JVP tangent：

```text
conditioning_dot_steps = zeros_like(conditioning_steps)
conditioning_dot_steps[:, -1] = s_dot_step
conditioning_dot_vec = conditioning_dot_steps.flatten(start_dim=1)
```

也就是只让当前 observation slot 沿 `s_dot` 方向扰动，历史 observation slot 固定。这更接近 `bc_only.py` 里“当前 state 的 tangent”，而不是把整个 history window 当成时间平移窗口。

如果后面你想改成整个 history window 的 shift tangent，需要额外拿到未来 encoded observation，这次不要做。

## 关于我之前说的两个配置名

不要新增下面这些复杂配置：

```text
kinematic_loss_target
kinematic_state_tangent
kinematic_loss_range
```

它们的含义只是设计维度，不是这次要加的 config：

- `kinematic_loss_target`: 指 `a_dot_data` 从哪里来，是从 action window 内部差分，还是 dataset 直接提供 `delta_a` / `next_actions`。
- `kinematic_state_tangent`: 指 `s_dot` 在哪个空间里算，是 raw proprio/image，还是 image/text encode 之后，还是完整 action-head conditioning vector。

这次固定选择：

- `a_dot_data`: 直接从当前 normalized action window 做 finite difference。
- `s_dot`: 从 action head 输入域的 encoded per-step observation 做 finite difference。

因此只新增最少配置：

```python
lambda_flow_k: float = 0.0
use_jvp_ak: bool = False
```

`lambda_flow_k <= 0` 时 kinematic loss 关闭，行为必须和当前 flow matching 完全一致。

采样频率不要设计成 kinematic source/range 之类的大配置。优先复用现有 dataset/batch/fps 来源；如果在 `FlowMatchingObjective` 里确实拿不到 dataset fps，再加一个很窄的字段，比如：

```python
sample_frequency: float = 30.0
```

或者 `fps`，但先确认项目里是否已有统一字段可复用。不要为了这个引入一套复杂配置。

注意量纲：这里按真正的时间导数实现。dataset metadata 里给的是 `fps`，所以：

```text
dt = 1 / fps
dot = finite_difference / dt = finite_difference * fps
```

不要写成除以 `fps`。如果实现里新增字段，字段名应明确是 `fps` 还是 `dt`，避免混淆。

## 需要改的结构

### 1. `MultiTaskDiTConfig`

在 flow matching 配置附近新增：

```python
lambda_flow_k: float = 0.0
use_jvp_ak: bool = False
```

如果无法从现有 batch/dataset metadata 拿到采样频率，再新增一个非常窄的字段，例如：

```python
sample_frequency: float = 30.0
```

校验：

- `lambda_flow_k >= 0`
- `sample_frequency > 0`，如果新增了这个字段
- `use_jvp_ak` 仅在 `objective="flow_matching"` 下有意义，但不用强行报错

不要新增 `use_kinematic_loss`，直接用 `lambda_flow_k > 0` 作为开关。

### 2. `ObservationEncoder`

当前 `encode(batch)` 直接返回 flatten 后的 `conditioning_vec`。需要做一个小重构，避免重复编码逻辑：

```text
encode_steps(batch) -> conditioning_steps: (B, n_obs_steps, per_step_condition_dim)
encode(batch)       -> encode_steps(batch).flatten(start_dim=1)
```

命名不强制，但要满足：

- 不要复制一份完整 image/text encoding 代码。
- 主 loss 和 kinematic loss 使用同一次 forward 里得到的 encoded features。
- 这样 random crop / image augmentation 不会因为重复 encode 产生两份不同 augmentation 噪声。

如果想在 kinematic loss 上 freeze image encoder / text encoder，不加额外 config。直接在构造 kinematic tangent 的地方手动 `detach()` 相应 encoded features 或 `conditioning_dot`。默认可以先不 detach，或者明确留一个局部注释说明可在这里 detach。

建议实现方式：

```text
conditioning_steps = observation_encoder.encode_steps(batch)
conditioning_vec = conditioning_steps.flatten(start_dim=1)
```

在 `MultiTaskDiTPolicy.forward` 里把 `conditioning_steps` 一起传给 flow objective。diffusion objective 可以继续只用 `conditioning_vec`。

### 3. `FlowMatchingObjective.compute_loss`

当前 flow matching loss 是：

```text
data = batch[ACTION]                         # (B, horizon, action_dim)
noise = randn_like(data)
t = sample_timesteps(B)                      # (B,)
x_t = t * data + (1 - (1 - sigma_min) * t) * noise
target_velocity = data - (1 - sigma_min) * noise
predicted_velocity = model(x_t, t, conditioning_vec)
flow_loss = mse(predicted_velocity, target_velocity)
```

保留这个主 loss。

新增 kinematic loss 时，逻辑是：

```text
total_loss = flow_loss + lambda_flow_k * kinematic_loss
```

当 `lambda_flow_k == 0`，不要跑 JVP，直接走旧逻辑。

### 4. 构造 `a_dot_data`

固定从 normalized action window 内部做 finite difference：

```text
data:       (B, H, A)
a_dot_pair = finite_difference(data[:, 1:], data[:, :-1])
```

然后 pad 回 `(B, H, A)`，最后一个 action 没有 next action，最后一位 mask 掉：

```text
a_dot_data[:, :-1] = a_dot_pair
a_dot_data[:, -1] = 0
```

按采样间隔缩放：

```text
dt = 1 / fps
a_dot = (a_{i+1} - a_i) / dt = (a_{i+1} - a_i) * fps
```

`fps` 从 dataset metadata 取。LIBERO 当前 metadata 里 action/state/image 都是 `10.0 Hz`。

这个差分在 Normalizer 之后的 action 空间里做，也就是 normalized action 空间。

### 5. 构造 `s_dot`

从 `conditioning_steps` 做：

```text
conditioning_steps: (B, n_obs_steps, D)
s_dot_step = conditioning_steps[:, -1] - conditioning_steps[:, -2]
```

按同一个采样间隔缩放：

```text
dt = 1 / fps
s_dot = (s_t - s_{t-1}) / dt = (s_t - s_{t-1}) * fps
```

然后：

```text
conditioning_dot_steps = zeros_like(conditioning_steps)
conditioning_dot_steps[:, -1] = s_dot_step
conditioning_dot_vec = conditioning_dot_steps.flatten(start_dim=1)
```

要求：

- `n_obs_steps >= 2` 才能启用 kinematic loss；否则报清楚错误或跳过并记录。
- text feature 因为每步相同，finite difference 应自然为 0。
- image feature 使用已经 encode 后的 feature，不对 raw pixel 做差。
- 如果要 freeze image/text encoder 对 kinematic loss 的梯度，不加配置，在这里手动对相关 encoded feature 或 `conditioning_dot_vec` 做 `detach()`。

### 6. JVP 形式

目标对齐 `bc_only.py`：

```text
不使用 action JVP:
  residual = v_s_dot - a_dot_data

使用 action JVP:
  residual = v_s_dot + t * v_a_k_dot - a_dot_data
```

在 PyTorch 里用 `torch.func.jvp` 或等价的可反传 JVP API。

建议先对 conditioning input 做 JVP：

```text
f_cond(cond) = model(x_t, t, conditioning_vec=cond)
predicted_velocity, v_s_dot = jvp(
    f_cond,
    primals=(conditioning_vec,),
    tangents=(conditioning_dot_vec,),
)
```

这样 `predicted_velocity` 可以直接用于主 flow loss，避免重复 forward。

如果 `use_jvp_ak=True`，再对 action input `x_t` 做 JVP：

```text
f_action(x) = model(x, t, conditioning_vec=conditioning_vec)
_, v_a_k_dot = jvp(
    f_action,
    primals=(x_t,),
    tangents=(a_dot_data,),
)
residual = v_s_dot + t[:, None, None] * v_a_k_dot - a_dot_data
```

如果 `use_jvp_ak=False`：

```text
residual = v_s_dot - a_dot_data
```

注意：

- 不能在 `torch.no_grad()` 下算训练 loss 的 JVP。
- JVP 要能对 model 参数反传。
- AMP 下要测一下 dtype，必要时对 JVP 分支禁用 autocast 或保持和主 forward 一致。
- 不要让 inference sampling 路径跑 JVP。

### 7. mask 和 loss 归一化

主 flow loss 继续支持当前 `action_is_pad` 逻辑。

kinematic loss 需要自己的 valid mask：

```text
pair_valid[:, :-1] = True
pair_valid[:, -1] = False
```

还要额外处理 observation 边界 padding。`n_obs_steps=2` 时，`s_dot` 来自 `conditioning_steps[:, -1] - conditioning_steps[:, -2]`。如果 `observation.state_is_pad[:, -2]` 或 `observation.state_is_pad[:, -1]` 为 true，说明这个 `s_dot` 用到了 episode 边界 clamp，当前样本的 kinematic loss 应整体无效：

```text
obs_pair_valid = ~observation.state_is_pad[:, -2] & ~observation.state_is_pad[:, -1]
pair_valid &= obs_pair_valid[:, None]
```

如果 batch 没有 `observation.state_is_pad`，但有其他 observation padding key，可以用任一可靠 observation padding mask；如果完全没有 observation padding mask，至少保留 action padding mask，并在实现注释里说明缺口。

如果 batch 里有 `action_is_pad`：

```text
pair_valid[:, :-1] &= ~action_is_pad[:, :-1] & ~action_is_pad[:, 1:]
pair_valid[:, -1] = False
```

然后：

```text
kinematic_loss_per_step = sum(residual ** 2, dim=-1)  # (B, H)
kinematic_loss = masked_mean(kinematic_loss_per_step, pair_valid)
```

不要把最后一个 action 的 padded `a_dot=0` 纳入 loss。

这次不新增 `kinematic_loss_range`。默认对 full horizon 里所有有效 adjacent pair 做 kinematic loss，包括历史对齐位。如果后续实验想只约束实际执行的 `[n_obs_steps - 1 : n_obs_steps - 1 + n_action_steps]`，再单独改。

### 8. 日志输出

当前 `MultiTaskDiTPolicy.forward` 返回：

```python
return loss, None
```

加 kinematic loss 后，flow matching 路径应返回一个 output dict，至少包含：

```text
flow_loss
kinematic_loss
lambda_flow_k
use_jvp_ak
total_loss
kinematic_valid_ratio
```

diffusion objective 可以继续返回 `None`，但 policy forward 要能兼容 flow objective 返回 dict。

注意不要破坏 `lerobot_train.py` 里的：

```python
loss, output_dict = policy.forward(batch)
```

### 9. 保持行为兼容

必须满足：

1. `objective="diffusion"` 行为不变。
2. `objective="flow_matching"` 且 `lambda_flow_k=0` 时，行为和当前实现等价：
   - 不计算 JVP
   - 不要求 `n_obs_steps >= 2`
   - loss 数值只来自原 flow matching loss
3. `lambda_flow_k>0` 时才要求 `conditioning_steps` 和 kinematic tangent。
4. `use_jvp_ak=True` 只增加 action-input JVP，不改变基础 flow matching 目标。

## 测试建议

至少做这些验证：

1. 构造一个小 batch，`objective="flow_matching"`, `lambda_flow_k=0`，确认 forward 能跑，loss 是标量，output dict 合理或为空。
2. `lambda_flow_k>0`, `use_jvp_ak=False`，确认 forward/backward 能跑。
3. `lambda_flow_k>0`, `use_jvp_ak=True`，确认 forward/backward 能跑。
4. `action_is_pad` 存在时，确认 kinematic mask 不把最后一位和 padding pair 算进 loss。
5. `vision_encoder_type="resnet"` 时确认 shape 对齐：
   - `conditioning_steps: (B, n_obs_steps, D)`
   - `conditioning_dot_vec: same shape as conditioning_vec`
   - model output/residual: `(B, horizon, action_dim)`
6. 如果测试环境不能下载 HuggingFace CLIP 权重，优先用 resnet/no-pretrained 或 monkeypatch encoder，不要让测试依赖网络。

## 最容易出错的点

- 不要对 raw image 做 finite difference；这里的 `s_dot` 是 action head 输入域里的 encoded observation finite difference。
- 不要重复 encode 两次，否则 random crop 会让 `conditioning_vec` 和 `conditioning_dot` 来自不同 augmentation。
- JVP tangent 的 shape 必须和 primal 完全一致：
  - conditioning JVP: `(B, n_obs_steps * D)`
  - action JVP: `(B, horizon, action_dim)`
- `a_dot_data` 来自 normalized action window，不是 unnormalized action。
- 最后一个 horizon step 没有 `a_{t+1}`，必须 mask 掉。
- observation history 的开头可能被 episode 边界 clamp；如果 `observation.state_is_pad` 表示 `[-1, 0]` 里某个 step 是 padding，当前样本的 kinematic loss 要 mask 掉。
- 如果加入采样频率字段，要说明它表示 fps 还是 dt；这里应使用 `dot = finite_difference / dt = finite_difference * fps`。
- 不要新增一堆 source/range config。这次只做固定设计：encoded conditioning finite diff + action window finite diff。

## 预期数学形式

基础 flow matching：

```text
x_0 = noise
x_1 = action chunk
x_t = t x_1 + (1 - (1 - sigma_min)t) x_0
u_t = x_1 - (1 - sigma_min)x_0
flow_loss = ||v_theta(x_t, t, c) - u_t||^2
```

kinematic：

```text
dt = 1 / fps
c_dot = finite_difference(encoded_conditioning) / dt
a_dot = finite_difference(action_chunk) / dt
```

不使用 action JVP：

```text
v_s_dot = J_c v_theta(x_t, t, c) [c_dot]
kinematic_residual = v_s_dot - a_dot
```

使用 action JVP：

```text
v_a_dot = J_x v_theta(x_t, t, c) [a_dot]
kinematic_residual = v_s_dot + t * v_a_dot - a_dot
```

总 loss：

```text
loss = flow_loss + lambda_flow_k * kinematic_loss
```

其中 `kinematic_loss` 是 residual 在 action dim 上求平方和，再对 valid horizon step 做 masked mean。

在你完成这些之后。需要在这个文档的最开头说明你修改了哪里。修改了哪个文件的哪行。
