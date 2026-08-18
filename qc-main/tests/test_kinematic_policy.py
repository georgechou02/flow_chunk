"""Sanity checks for ACFQL high-order (DCT / forward-difference) supervision."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

QC_ROOT = Path(__file__).resolve().parents[1]
if str(QC_ROOT) not in sys.path:
    sys.path.insert(0, str(QC_ROOT))

from agents.acfql import ACFQLAgent, get_config
from utils.action_derivatives import action_dot, action_dot_dct, action_dot_forward
from utils.datasets import Dataset


def _assert_close(actual, expected, atol, msg):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    max_err = np.max(np.abs(actual - expected))
    print(f"  {msg}: max_err={max_err:.6g} (atol={atol})")
    if not np.allclose(actual, expected, atol=atol, rtol=1e-5):
        raise AssertionError(f"{msg} failed: max_err={max_err} > atol={atol}")


def _single_mode_actions(horizon, mode, coefficients):
    alpha = math.sqrt((1.0 if mode == 0 else 2.0) / horizon)
    sample_points = np.arange(horizon, dtype=np.float32) + 0.5
    basis = alpha * np.cos(math.pi * mode * sample_points / horizon)
    return basis[None, :, None] * coefficients[None, None, :]


def _single_mode_derivative(horizon, mode, coefficients, fps):
    alpha = math.sqrt((1.0 if mode == 0 else 2.0) / horizon)
    sample_points = np.arange(horizon, dtype=np.float32) + 0.5
    basis_dot = -alpha * (math.pi * mode / horizon) * np.sin(math.pi * mode * sample_points / horizon)
    return (basis_dot[None, :, None] * coefficients[None, None, :]) * fps


def test_forward_and_dct_derivatives():
    print("\n[1] Action-chunk derivatives")
    horizon, action_dim, fps = 8, 3, 10.0
    rng = np.random.default_rng(0)
    actions = rng.normal(size=(4, horizon, action_dim)).astype(np.float32)
    next_actions = rng.normal(size=(4, horizon, action_dim)).astype(np.float32)

    forward = np.asarray(action_dot_forward(jnp.asarray(actions), jnp.asarray(next_actions), fps=fps))
    _assert_close(forward, (next_actions - actions) * fps, 1e-6, "forward difference matches (next-cur)*fps")

    dispatched = np.asarray(
        action_dot(jnp.asarray(actions), next_actions=jnp.asarray(next_actions), fps=fps, dct_coe_num=0)
    )
    _assert_close(dispatched, forward, 1e-6, "dct_coe_num=0 dispatches to forward difference")

    coefficients = np.array([[0.4, -0.2, 1.1]], dtype=np.float32)
    mode = 3
    cosine_actions = _single_mode_actions(horizon, mode, coefficients[0])
    expected = _single_mode_derivative(horizon, mode, coefficients[0], fps)
    dct = np.asarray(action_dot_dct(jnp.asarray(cosine_actions), fps=fps, num_modes=horizon))
    _assert_close(dct, expected, 1e-5, f"DCT derivative of cosine mode {mode}")

    truncated = np.asarray(action_dot_dct(jnp.asarray(cosine_actions), fps=fps, num_modes=mode))
    _assert_close(truncated, np.zeros_like(expected), 1e-5, "dropping the active DCT mode zeros the derivative")

    ramp = np.broadcast_to(np.arange(horizon, dtype=np.float32)[None, :, None], (2, horizon, 1))
    ramp_next = ramp + 1.0
    ramp_dot = np.asarray(action_dot_forward(jnp.asarray(ramp), jnp.asarray(ramp_next), fps=1.0))
    _assert_close(ramp_dot, np.ones_like(ramp), 1e-6, "forward difference of a unit ramp is 1")
    print("  PASS")


def _tiny_config(horizon, lambda_flow_k, dct_coe_num, use_jvp_ak=False):
    config = get_config()
    config.horizon_length = horizon
    config.action_chunking = True
    config.lambda_flow_k = lambda_flow_k
    config.dct_coe_num = dct_coe_num
    config.use_jvp_ak = use_jvp_ak
    config.stop_gradient_jvp_ak = False
    config.use_1_k = False
    config.sample_frequency = 1.0
    config.actor_hidden_dims = (32, 32)
    config.value_hidden_dims = (32, 32)
    config.actor_type = "best-of-n"
    config.actor_num_samples = 4
    config.encoder = None
    config.batch_size = 8
    return config


def _synthetic_batch(obs_dim, action_dim, horizon, batch_size=8, constant_obs=False, constant_actions=False, seed=0):
    rng = np.random.default_rng(seed)
    observations = rng.normal(size=(batch_size, obs_dim)).astype(np.float32)
    if constant_obs:
        next_observations = np.repeat(observations[:, None, :], horizon, axis=1)
    else:
        next_observations = rng.normal(size=(batch_size, horizon, obs_dim)).astype(np.float32)
        next_observations[:, 0] = observations + rng.normal(size=observations.shape).astype(np.float32) * 0.1

    if constant_actions:
        actions = np.zeros((batch_size, horizon, action_dim), dtype=np.float32)
        next_actions = np.zeros_like(actions)
    else:
        actions = rng.normal(size=(batch_size, horizon, action_dim)).astype(np.float32)
        next_actions = rng.normal(size=(batch_size, horizon, action_dim)).astype(np.float32)

    return dict(
        observations=observations,
        next_observations=next_observations,
        actions=actions,
        next_actions=next_actions,
        rewards=np.zeros((batch_size, horizon), dtype=np.float32),
        masks=np.ones((batch_size, horizon), dtype=np.float32),
        terminals=np.zeros((batch_size, horizon), dtype=np.float32),
        valid=np.ones((batch_size, horizon), dtype=np.float32),
    )


def _make_agent(obs_dim, action_dim, config):
    ex_obs = np.zeros((obs_dim,), dtype=np.float32)
    ex_act = np.zeros((action_dim,), dtype=np.float32)
    return ACFQLAgent.create(seed=0, ex_observations=ex_obs, ex_actions=ex_act, config=config)


def test_kinematic_identities():
    print("\n[2] Kinematic residual identities (s_dot=0, use_jvp_ak=False)")
    obs_dim, action_dim, horizon = 6, 3, 5
    rng = jax.random.PRNGKey(1)

    for name, dct_coe_num in (("forward", 0), ("dct", horizon)):
        config = _tiny_config(horizon, lambda_flow_k=1.0, dct_coe_num=dct_coe_num, use_jvp_ak=False)
        agent = _make_agent(obs_dim, action_dim, config)
        batch = _synthetic_batch(
            obs_dim, action_dim, horizon, constant_obs=True, constant_actions=False, seed=1
        )
        _, info = agent.actor_loss(batch, agent.network.params, rng)

        a_dot = action_dot(
            jnp.asarray(batch["actions"]),
            next_actions=jnp.asarray(batch["next_actions"]),
            fps=1.0,
            dct_coe_num=dct_coe_num,
        )
        expected = jnp.mean(jnp.square(a_dot), axis=-1).mean()
        _assert_close(
            info["kinematic_state_jvp_rms"],
            0.0,
            1e-5,
            f"{name}: state JVP along s_dot=0 is 0",
        )
        _assert_close(
            info["kinematic_loss"],
            expected,
            1e-5,
            f"{name}: residual collapses to -a_dot so loss = mean(||a_dot||^2)",
        )
        assert float(info["a_dot_rms"]) > 0.0, f"{name}: a_dot_rms should be > 0"

        const_batch = _synthetic_batch(
            obs_dim, action_dim, horizon, constant_obs=True, constant_actions=True, seed=2
        )
        _, const_info = agent.actor_loss(const_batch, agent.network.params, rng)
        _assert_close(const_info["kinematic_loss"], 0.0, 1e-6, f"{name}: zero a_dot and s_dot => loss 0")
        _assert_close(const_info["a_dot_rms"], 0.0, 1e-6, f"{name}: constant actions => a_dot_rms 0")

    off_config = _tiny_config(horizon, lambda_flow_k=0.0, dct_coe_num=0)
    off_agent = _make_agent(obs_dim, action_dim, off_config)
    batch = _synthetic_batch(obs_dim, action_dim, horizon, constant_obs=True, seed=3)
    _, off_info = off_agent.actor_loss(batch, off_agent.network.params, rng)
    _assert_close(off_info["kinematic_loss"], 0.0, 1e-8, "lambda_flow_k=0 reports kinematic_loss=0")
    _assert_close(off_info["actor_loss"], off_info["bc_flow_loss"], 1e-6, "lambda_flow_k=0 actor_loss == bc_flow_loss")
    print("  PASS")


def test_update_and_jvp_ak():
    print("\n[3] One optimizer step for both supervision modes")
    obs_dim, action_dim, horizon = 6, 3, 5
    batch = _synthetic_batch(obs_dim, action_dim, horizon, seed=4)

    for name, dct_coe_num in (("forward", 0), ("dct", horizon)):
        config = _tiny_config(horizon, lambda_flow_k=1.0, dct_coe_num=dct_coe_num)
        agent = _make_agent(obs_dim, action_dim, config)
        new_agent, info = agent.update(batch)
        kin = float(info["actor/kinematic_loss"])
        total = float(info["actor/actor_loss"])
        print(
            f"  {name}: kinematic_loss={kin:.6g}  bc_flow_loss={float(info['actor/bc_flow_loss']):.6g}  "
            f"a_dot_rms={float(info['actor/a_dot_rms']):.6g}  "
            f"state_jvp_rms={float(info['actor/kinematic_state_jvp_rms']):.6g}  "
            f"action_jvp_rms={float(info['actor/kinematic_action_jvp_rms']):.6g}"
        )
        if not math.isfinite(kin) or not math.isfinite(total):
            raise AssertionError(f"{name} produced non-finite losses")
        if kin <= 0:
            raise AssertionError(f"{name} expected a positive kinematic_loss on a random batch")
        before = jax.tree_util.tree_map(lambda x: np.array(x), agent.network.params)
        after = jax.tree_util.tree_map(lambda x: np.array(x), new_agent.network.params)
        changed = jax.tree_util.tree_reduce(
            lambda acc, xs: acc or np.any(xs[0] != xs[1]),
            jax.tree_util.tree_map(lambda a, b: (a, b), before, after),
            False,
        )
        if not changed:
            raise AssertionError(f"{name} update did not change parameters")
    print("  PASS")


def _terminals_from_d4rl(dataset):
    terminals = np.zeros_like(dataset["rewards"], dtype=np.float32)
    for i in range(len(terminals) - 1):
        jumped = np.linalg.norm(dataset["observations"][i + 1] - dataset["next_observations"][i]) > 1e-6
        terminals[i] = float(jumped or dataset["terminals"][i] == 1.0)
    terminals[-1] = 1.0
    return terminals


def test_d4rl_hopper_batch():
    print("\n[4] Real D4RL hopper-medium-v2 batch")
    os.environ.setdefault("MUJOCO_GL", "egl")
    import d4rl  # noqa: F401
    import gym

    env = gym.make("hopper-medium-v2")
    raw = d4rl.qlearning_dataset(env)
    terminals = _terminals_from_d4rl(raw)
    dataset = Dataset.create(
        observations=raw["observations"].astype(np.float32),
        actions=np.clip(raw["actions"].astype(np.float32), -1 + 1e-5, 1 - 1e-5),
        next_observations=raw["next_observations"].astype(np.float32),
        terminals=terminals,
        rewards=raw["rewards"].astype(np.float32),
        masks=(1.0 - raw["terminals"]).astype(np.float32),
    )
    horizon = 5
    batch = dataset.sample_sequence(16, sequence_length=horizon, discount=0.99)
    obs_dim = batch["observations"].shape[-1]
    action_dim = batch["actions"].shape[-1]
    print(f"  obs_dim={obs_dim} action_dim={action_dim} chunk={batch['actions'].shape}")

    forward = np.asarray(
        action_dot(
            jnp.asarray(batch["actions"]),
            next_actions=jnp.asarray(batch["next_actions"]),
            fps=1.0,
            dct_coe_num=0,
        )
    )
    dct = np.asarray(
        action_dot(
            jnp.asarray(batch["actions"]),
            next_actions=jnp.asarray(batch["next_actions"]),
            fps=1.0,
            dct_coe_num=horizon,
        )
    )
    print(f"  forward a_dot rms={np.sqrt(np.mean(forward**2)):.6g}")
    print(f"  DCT a_dot rms    ={np.sqrt(np.mean(dct**2)):.6g}")
    _assert_close(forward, batch["next_actions"] - batch["actions"], 1e-6, "D4RL batch forward difference")
    if np.allclose(dct, 0.0):
        raise AssertionError("DCT a_dot should be non-zero on hopper action chunks")

    for name, dct_coe_num in (("forward", 0), ("dct", horizon)):
        config = _tiny_config(horizon, lambda_flow_k=1.0, dct_coe_num=dct_coe_num)
        agent = _make_agent(obs_dim, action_dim, config)
        _, info = agent.update(batch)
        print(
            f"  {name} update: kinematic_loss={float(info['actor/kinematic_loss']):.6g}  "
            f"bc_flow_loss={float(info['actor/bc_flow_loss']):.6g}  "
            f"valid_ratio={float(info['actor/kinematic_valid_ratio']):.6g}"
        )
        if not math.isfinite(float(info["actor/kinematic_loss"])):
            raise AssertionError(f"{name} D4RL update produced a non-finite kinematic_loss")
        if float(info["actor/kinematic_loss"]) <= 0:
            raise AssertionError(f"{name} D4RL kinematic_loss should be positive")
    print("  PASS")


def test_ogbench_like_readme(env_name, require_dataset=True):
    """Load an OGBench singletask env the same way as README/main.py."""
    print(f"\n[5] OGBench {env_name}")
    from envs.env_utils import make_env_and_datasets

    try:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(env_name)
    except Exception as exc:
        if require_dataset:
            raise
        print(f"  skip: {type(exc).__name__}: {exc}")
        return

    example = train_dataset.sample(())
    horizon = 5
    batch = train_dataset.sample_sequence(8, sequence_length=horizon, discount=0.99)
    obs_dim = int(np.asarray(example["observations"]).shape[-1])
    action_dim = int(np.asarray(example["actions"]).shape[-1])
    print(
        f"  obs_dim={obs_dim} action_dim={action_dim} "
        f"dataset_size={train_dataset.size} seq_actions={batch['actions'].shape}"
    )

    if batch["observations"].shape[-1] != batch["next_observations"].shape[-1]:
        raise AssertionError(
            f"state tangent shape mismatch: obs {batch['observations'].shape} vs "
            f"next_obs {batch['next_observations'].shape}"
        )

    # README QC uses best-of-n; README QC-FQL uses distill-ddpg.
    for actor_type, dct_coe_num, name in (
        ("best-of-n", 0, "QC forward"),
        ("best-of-n", horizon, "QC dct"),
        ("distill-ddpg", 0, "QC-FQL forward"),
        ("distill-ddpg", horizon, "QC-FQL dct"),
    ):
        config = _tiny_config(horizon, lambda_flow_k=1.0, dct_coe_num=dct_coe_num)
        config.actor_type = actor_type
        agent = ACFQLAgent.create(
            seed=0,
            ex_observations=example["observations"],
            ex_actions=example["actions"],
            config=config,
        )
        _, info = agent.update(batch)
        kin = float(info["actor/kinematic_loss"])
        print(
            f"  {name}: kinematic_loss={kin:.6g}  "
            f"bc_flow_loss={float(info['actor/bc_flow_loss']):.6g}  "
            f"distill_loss={float(info['actor/distill_loss']):.6g}  "
            f"valid_ratio={float(info['actor/kinematic_valid_ratio']):.6g}"
        )
        if not math.isfinite(kin) or kin <= 0:
            raise AssertionError(f"{name} on {env_name} produced invalid kinematic_loss={kin}")

    env.close()
    eval_env.close()
    print("  PASS")


def main():
    print("JAX devices:", jax.devices())
    test_forward_and_dct_derivatives()
    test_kinematic_identities()
    test_update_and_jvp_ak()
    test_d4rl_hopper_batch()
    # Local OGBench datasets covering the README domains (skip cube-triple; the npz is too large).
    test_ogbench_like_readme("cube-double-play-singletask-task2-v0")
    test_ogbench_like_readme("scene-play-singletask-task1-v0")
    test_ogbench_like_readme("puzzle-3x3-play-singletask-task1-v0")
    print("\nAll kinematic-policy checks passed.")


if __name__ == "__main__":
    main()
