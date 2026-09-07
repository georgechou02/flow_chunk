"""Sanity checks for QC-IFQL (chunked IQL + flow BC + rejection sampling)."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

QC_ROOT = Path(__file__).resolve().parents[1]
if str(QC_ROOT) not in sys.path:
    sys.path.insert(0, str(QC_ROOT))

from agents.acifql import ACIFQLAgent, get_config


def _tiny_config(horizon, lambda_flow_k=0.0, **overrides):
    config = get_config()
    config.horizon_length = horizon
    config.action_chunking = True
    config.lambda_flow_k = lambda_flow_k
    config.actor_hidden_dims = (32, 32)
    config.value_hidden_dims = (32, 32)
    config.batch_size = 8
    config.num_samples = 4
    config.encoder = None
    for key, value in overrides.items():
        config[key] = value
    return config


def _synthetic_batch(obs_dim, action_dim, horizon, batch_size=8, seed=0):
    rng = np.random.default_rng(seed)
    observations = rng.normal(size=(batch_size, obs_dim)).astype(np.float32)
    next_observations = rng.normal(size=(batch_size, horizon, obs_dim)).astype(np.float32)
    prev_observations = rng.normal(size=(batch_size, obs_dim)).astype(np.float32)
    actions = rng.normal(size=(batch_size, horizon, action_dim)).astype(np.float32)
    next_actions = rng.normal(size=(batch_size, horizon, action_dim)).astype(np.float32)
    return dict(
        observations=observations,
        prev_observations=prev_observations,
        prev_valid=np.ones((batch_size,), dtype=np.float32),
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
    return ACIFQLAgent.create(seed=0, ex_observations=ex_obs, ex_actions=ex_act, config=config)


def test_create_update_sample():
    obs_dim, action_dim, horizon = 6, 3, 5
    batch = _synthetic_batch(obs_dim, action_dim, horizon)
    config = _tiny_config(horizon, lambda_flow_k=0.0)
    agent = _make_agent(obs_dim, action_dim, config)
    assert "modules_value" in agent.network.params
    assert "modules_actor_flow" in agent.network.params
    assert "modules_actor_onestep_flow" not in agent.network.params

    new_agent, info = agent.update(batch)
    assert float(info["actor/lambda_flow_k"]) == 0.0
    assert float(info["actor/kinematic_loss"]) == 0.0
    assert np.isfinite(float(info["actor/bc_flow_loss"]))
    assert np.isfinite(float(info["value/value_loss"]))
    assert np.isfinite(float(info["critic/critic_loss"]))

    chunk = new_agent.sample_actions(batch["observations"][0], rng=jax.random.PRNGKey(0))
    assert chunk.shape == (horizon * action_dim,)
    batched = new_agent.sample_actions(batch["observations"], rng=jax.random.PRNGKey(1))
    assert batched.shape == (batch["observations"].shape[0], horizon * action_dim)


def test_kinematic_on_when_lambda_positive():
    obs_dim, action_dim, horizon = 6, 3, 5
    batch = _synthetic_batch(obs_dim, action_dim, horizon, seed=4)
    config = _tiny_config(horizon, lambda_flow_k=1.0, derivative_kind="forward", dct_coe_num=0)
    agent = _make_agent(obs_dim, action_dim, config)
    _, info = agent.update(batch)
    kin = float(info["actor/kinematic_loss"])
    assert np.isfinite(kin) and kin > 0.0


def test_physical_on_when_weight_positive():
    obs_dim, action_dim, horizon = 6, 3, 5
    batch = _synthetic_batch(obs_dim, action_dim, horizon, seed=5)
    off = _tiny_config(horizon, lambda_flow_k=0.0, phy_loss_weight=0.0)
    on = _tiny_config(horizon, lambda_flow_k=0.0, phy_loss_weight=0.5, derivative_kind="forward")
    off_agent = _make_agent(obs_dim, action_dim, off)
    on_agent = _make_agent(obs_dim, action_dim, on)
    _, off_info = off_agent.update(batch)
    _, on_info = on_agent.update(batch)
    assert float(off_info["actor/weighted_physical_loss"]) == 0.0
    phy = float(on_info["actor/physical_loss"])
    weighted = float(on_info["actor/weighted_physical_loss"])
    assert np.isfinite(phy) and phy > 0.0
    assert np.isclose(weighted, 0.5 * phy, rtol=1e-5, atol=1e-5)
    assert np.isclose(
        float(on_info["actor/actor_loss"]),
        float(on_info["actor/bc_flow_loss"]) + weighted,
        rtol=1e-5,
        atol=1e-5,
    )


if __name__ == "__main__":
    test_create_update_sample()
    test_kinematic_on_when_lambda_positive()
    test_physical_on_when_weight_positive()
    print("PASS")
