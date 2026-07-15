# LeRobot Diffusion Policy 改 Flow Matching 提示词

请在 `/home/zhouzhi/code_store/higher-order/lerobot` 当前框架下，增量式新增一版 flow matching action chunk policy。新 policy 放在：

```text
src/lerobot/policies/flow
```

要求代码风格尽量贴近原本 diffusion policy。能从 `src/lerobot/policies/diffusion` 复制迁移的代码就复制迁移，能直接复用的已有模块就复用，不要做大规模重构。

## 目标

基于现有 `DiffusionPolicy / DiffusionModel` 的结构，在 `policies/flow` 下实现一个 flow matching 版本：

- 输入仍然是 observation history。
- 输出仍然是 action chunk。
- 复用现有 image encoder、state/env_state conditioning、action queue、normalization、processor、训练入口。
- 将 DDPM/DDIM 的 noise scheduler 训练/采样逻辑替换为 flow matching velocity 训练和 ODE 积分采样。
- 保持增量开发：不要直接把 `policies/diffusion` 改坏，原 diffusion policy 应该继续可用。

## 代码原则

- 保持当前 LeRobot 代码风格、命名、函数拆分方式和注释风格。
- 优先复制迁移 `DiffusionPolicy`、`DiffusionModel`、`DiffusionRgbEncoder`、`DiffusionConditionalUnet1d`、`_prepare_global_conditioning()`、`select_action()`、`predict_action_chunk()` 的已有结构。
- 对稳定且可直接复用的模块，优先 import 复用；如果复制后改动更少、更贴近现有 policy 目录组织，也可以复制迁移。
- 不要引入新的大抽象、复杂封装或无关依赖。
- 不要改动 ACT、SmolVLA、Pi0 等无关 policy。
- 不要直接覆盖 `diffusion` 注册名。新增 policy 注册名建议为 `flow`。

## 参考位置

- 当前 diffusion policy：
  - `src/lerobot/policies/diffusion/modeling_diffusion.py`
  - `src/lerobot/policies/diffusion/configuration_diffusion.py`
  - `src/lerobot/policies/diffusion/processor_diffusion.py`
- 仓库内 flow matching 参考：
  - `src/lerobot/policies/multi_task_dit/modeling_multi_task_dit.py` 里的 `FlowMatchingObjective`
  - `src/lerobot/policies/vla_jepa/action_head.py` 的 velocity loss 和 Euler sampling

## 实现要求

### 1. 配置

新增必要 flow matching 配置，保持字段简洁：

- `num_inference_steps`
- `timestep_sampling_strategy`
- `timestep_sampling_alpha`
- `timestep_sampling_beta`
- `timestep_sampling_s`
- `sigma_min`

新增 policy 类型注册名用 `flow`。新增文件建议包括：

- `src/lerobot/policies/flow/__init__.py`
- `src/lerobot/policies/flow/configuration_flow.py`
- `src/lerobot/policies/flow/modeling_flow.py`
- `src/lerobot/policies/flow/processor_flow.py`

必要时同步更新 policy factory/registry 相关位置，让 `--policy.type=flow` 可用。

### 2. 训练 loss

保持 batch/action shape 与 diffusion policy 一致：`action` 为 `(B, horizon, action_dim)`。

训练逻辑改为：

```python
noise = torch.randn_like(trajectory)
t = sample_t(batch_size)
x_t = t * trajectory + (1 - (1 - sigma_min) * t) * noise
target_velocity = trajectory - (1 - sigma_min) * noise
pred_velocity = self.unet(x_t, t, global_cond=global_cond)
loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")
```

保留原有 `do_mask_loss_for_padding` 的 mask 逻辑。

### 3. 推理采样

从高斯噪声初始化整段 action trajectory：

```python
x = torch.randn(batch_size, horizon, action_dim)
```

用 Euler 积分从 `t=0` 到 `t=1`：

```python
for step in range(num_inference_steps):
    t = torch.full((batch_size,), step / num_inference_steps, device=x.device, dtype=x.dtype)
    velocity = self.unet(x, t, global_cond=global_cond)
    x = x + dt * velocity
```

采样完成后，仍然按现有 diffusion policy 的方式截取：

```python
start = n_obs_steps - 1
end = start + n_action_steps
actions = trajectory[:, start:end]
```

### 4. timestep 兼容

现有 `DiffusionConditionalUnet1d` 的 timestep embedding 可以复用，但它当前偏向整数 diffusion step。请做最小改动，让它能接收 flow matching 的连续 `t`，不要重写整个 U-Net。

### 5. 验证

至少做这些检查：

- `python -m py_compile` 对改动文件通过。
- 用小配置跑一个 forward/loss smoke test，确认输出 loss 是标量。
- 用 `predict_action_chunk()` smoke test，确认输出 shape 是 `(B, n_action_steps, action_dim)`。

## 注意

这个任务重点是做一个干净、可读、容易合并的 flow matching baseline，不追求一次性实现所有 scheduler/solver 变体。先把 Euler flow matching 路径跑通。
