"""QC baselines plus the full, time-weighted LeRobot first-order constraint."""

import sys
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

QC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QC_ROOT))

from agents.acfql import ACFQLAgent, get_config as fql_config
from agents.acifql import ACIFQLAgent, get_config as ifql_config


AGENTS = [(ACFQLAgent, fql_config), (ACIFQLAgent, ifql_config)]


class LinearField:
    def select(self, name):
        def field(observations, actions, times, params):
            return params["action"] * actions + params["state"] * observations[:, :1]

        return field


@pytest.mark.parametrize("agent_class,config_factory", AGENTS)
@pytest.mark.parametrize("grad_scale", [0.0, 0.3, 1.0])
def test_fos_value_and_parameter_gradients(agent_class, config_factory, grad_scale):
    config = config_factory()
    config.horizon_length = 3
    config.action_dim = 1
    config.ob_dims = (1,)
    config.derivative_kind = "forward"
    config.action_jvp_grad_scale = grad_scale
    agent = agent_class(
        rng=jax.random.PRNGKey(0), network=LinearField(),
        config=flax.core.FrozenDict(config.to_dict()),
    )
    # Central state velocity is 3; expert action velocity varies by horizon step.
    actions = jnp.zeros((4, 3, 1))
    action_dot = jnp.broadcast_to(jnp.array([1.0, 2.0, 3.0])[None, :, None], actions.shape)
    batch = dict(
        observations=jnp.ones((4, 1)), prev_observations=-jnp.ones((4, 1)),
        next_observations=jnp.full((4, 3, 1), 5.0),
        actions=actions, next_actions=action_dot,
        valid=jnp.ones((4, 3)), masks=jnp.ones((4, 3)), prev_valid=jnp.ones(4),
    )
    time = jnp.array([0.0, 0.25, 0.75, 1.0])[:, None]
    params = dict(action=jnp.array(0.4), state=jnp.array(0.7))
    x_t = jnp.ones((4, 3))

    def loss(p):
        return agent._compute_kinematic_loss(batch, x_t, time, p)[1]

    dot = action_dot[..., 0]
    residual = (1 - time) * (3 * params["state"] + time * params["action"] * dot - dot)
    np.testing.assert_allclose(loss(params), jnp.square(residual).mean(), rtol=1e-6)
    gradients = jax.grad(loss)(params)
    np.testing.assert_allclose(
        gradients["action"],
        grad_scale * (2 * residual * (1 - time) * time * dot).mean(), atol=1e-6,
    )
    np.testing.assert_allclose(
        gradients["state"], (2 * residual * (1 - time) * 3).mean(), atol=1e-6,
    )


def baseline_actor_loss(agent, batch, params, rng):
    """Original QC flow BC, plus QC-FQL distillation/Q terms where applicable."""
    actions = jnp.asarray(batch["actions"]).reshape(batch["actions"].shape[0], -1)
    rng, noise_rng, time_rng = jax.random.split(rng, 3)
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.uniform(time_rng, (actions.shape[0], 1))
    field_name = "actor_bc_flow" if isinstance(agent, ACFQLAgent) else "actor_flow"
    prediction = agent.network.select(field_name)(
        batch["observations"], (1 - time) * noise + time * actions, time, params=params,
    )
    loss = (jnp.square(prediction - (actions - noise)).reshape(batch["actions"].shape)
            * batch["valid"][..., None]).mean()
    if isinstance(agent, ACFQLAgent) and agent.config["actor_type"] == "distill-ddpg":
        _, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, actions.shape)
        target = agent.compute_flow_actions(batch["observations"], noises)
        actor = agent.network.select("actor_onestep_flow")(
            batch["observations"], noises, params=params,
        )
        distillation = jnp.square(actor - target).mean()
        q = agent.network.select("critic")(
            batch["observations"], actions=jnp.clip(actor, -1, 1),
        ).mean(axis=0)
        loss = loss + agent.config["alpha"] * distillation - q.mean()
    return loss


@pytest.mark.parametrize("variant", ["qc", "qc-fql", "qc-ifql"])
def test_zero_lambda_recovers_baseline_loss_and_gradients(variant):
    agent_class, factory = AGENTS[int(variant == "qc-ifql")]
    config = factory()
    config.horizon_length = 3
    config.actor_hidden_dims = (8,)
    config.value_hidden_dims = (8,)
    if variant != "qc-ifql":
        config.actor_type = "best-of-n" if variant == "qc" else "distill-ddpg"
    agent = agent_class.create(0, np.zeros(2, np.float32), np.zeros(1, np.float32), config)
    rng = np.random.default_rng(4)
    batch = dict(
        observations=jnp.asarray(rng.normal(size=(2, 2)), dtype=jnp.float32),
        prev_observations=jnp.zeros((2, 2)), next_observations=jnp.ones((2, 3, 2)),
        actions=jnp.asarray(rng.normal(size=(2, 3, 1)), dtype=jnp.float32),
        next_actions=jnp.ones((2, 3, 1)), valid=jnp.ones((2, 3)),
        masks=jnp.ones((2, 3)), prev_valid=jnp.ones(2),
    )
    key = jax.random.PRNGKey(7)
    actual, actual_grad = jax.value_and_grad(
        lambda p: agent.actor_loss(batch, p, key)[0]
    )(agent.network.params)
    expected, expected_grad = jax.value_and_grad(
        lambda p: baseline_actor_loss(agent, batch, p, key)
    )(agent.network.params)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    for got, want in zip(jax.tree_util.tree_leaves(actual_grad), jax.tree_util.tree_leaves(expected_grad)):
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)

    augmented = agent.replace(config=agent.config.copy({"lambda_flow_k": 0.2}))
    loss, info = augmented.actor_loss(batch, augmented.network.params, key)
    assert float(info["kinematic_loss"]) > 0
    assert float(info["weighted_physical_loss"]) == 0
    np.testing.assert_allclose(loss, expected + 0.2 * info["kinematic_loss"], rtol=1e-6)


@pytest.mark.parametrize("agent_class,config_factory", AGENTS)
def test_qc_default_hyperparameters_are_preserved(agent_class, config_factory):
    config = config_factory()
    expected = dict(
        lr=3e-4, batch_size=256, actor_hidden_dims=(512,) * 4,
        value_hidden_dims=(512,) * 4, discount=0.99, tau=0.005,
        flow_steps=10, layer_norm=True, actor_layer_norm=False,
        lambda_flow_k=0.0, phy_loss_weight=0.0, sample_frequency=1.0,
        dct_coe_num=0, derivative_kind="auto", state_derivative_mode="central",
        action_jvp_grad_scale=1.0,
    )
    if agent_class is ACFQLAgent:
        expected.update(alpha=100.0, q_agg="mean", actor_num_samples=32, actor_type="distill-ddpg")
    else:
        expected.update(expectile=0.9, q_agg="min", num_samples=32)
    for key, value in expected.items():
        assert config[key] == value, key
