import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.action_derivatives import action_dot
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value

class ACFQLAgent(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent with action chunking. 
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""

        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :] # take the first action
        
        # TD loss
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng=sample_rng)

        next_qs = self.network.select(f'target_critic')(batch['next_observations'][..., -1, :], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)
        
        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch_actions, params=grad_params)
        
        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def _action_chunk(self, batch):
        """Return the action chunk that the flow actor actually models."""
        actions = batch["actions"]
        next_actions = batch.get("next_actions")
        if actions.ndim == 2:
            actions = actions[:, None, :]
            if next_actions is not None:
                next_actions = next_actions[:, None, :]
        if not self.config["action_chunking"]:
            actions = actions[:, :1]
            if next_actions is not None:
                next_actions = next_actions[:, :1]
        return actions, next_actions

    def _state_dot(self, batch, fps):
        """Forward-difference state tangent at the current observation."""
        observations = batch["observations"]
        next_observations = batch["next_observations"]
        if next_observations.ndim == observations.ndim + 1:
            next_obs0 = next_observations[:, 0]
        else:
            next_obs0 = next_observations
        return (next_obs0 - observations) * jnp.asarray(fps, dtype=observations.dtype)

    def _kinematic_valid_mask(self, batch, num_steps):
        """Per-step mask for kinematic supervision."""
        if "valid" in batch:
            kinematic_valid = batch["valid"][:, :num_steps]
            if "masks" in batch:
                kinematic_valid = kinematic_valid * batch["masks"][:, :num_steps]
        else:
            batch_size = batch["actions"].shape[0]
            kinematic_valid = jnp.ones((batch_size, num_steps), dtype=jnp.float32)

        # DCT couples every frame in the chunk, so drop samples that cross an
        # episode boundary rather than differentiating a mixed trajectory.
        if self.config["dct_coe_num"] > 0:
            all_valid = jnp.all(kinematic_valid > 0, axis=-1, keepdims=True)
            kinematic_valid = kinematic_valid * all_valid.astype(kinematic_valid.dtype)
        return kinematic_valid

    def _flow_actions(self, observations, noises, params=None, clip=True):
        """Integrate the flow vector field into an action chunk (inference ODE)."""
        is_encoded = False
        if self.config["encoder"] is not None:
            observations = self.network.select("actor_bc_flow_encoder")(
                observations, params=params
            )
            is_encoded = True

        actions = noises
        n_steps = self.config["flow_steps"]
        for i in range(n_steps):
            t = jnp.full((*observations.shape[:-1], 1), i / n_steps)
            vels = self.network.select("actor_bc_flow")(
                observations, actions, t, is_encoded=is_encoded, params=params
            )
            actions = actions + vels / n_steps
        if clip:
            actions = jnp.clip(actions, -1, 1)
        return actions

    def _compute_kinematic_loss(self, batch, x_t, t, grad_params):
        """High-order constraint on the interpolant vector field ``v_θ(s, x_t, t)``.

        Matches IFQL actor and multi-task DiT: JVP of the flow velocity on
        ``x_t = (1-t)z + t a``, not of the Euler-integrated policy. ``ȧ`` is the
        chunk derivative of the expert actions (forward difference or DCT).
        Residual is ``v_s_dot - ȧ``, or DiT's ``(1-t)(v_s_dot + t·v_a_k_dot - ȧ)``
        when ``use_jvp_ak`` is set.
        """
        fps = self.config["sample_frequency"]
        actions, next_actions = self._action_chunk(batch)
        a_dot = action_dot(
            actions,
            next_actions=next_actions,
            fps=fps,
            dct_coe_num=self.config["dct_coe_num"],
        )
        a_dot_flat = jnp.reshape(a_dot, (a_dot.shape[0], -1))
        s_dot = self._state_dot(batch, fps)
        observations = batch["observations"]

        def flow_vector_field_obs(obs):
            return self.network.select("actor_bc_flow")(obs, x_t, t, params=grad_params)

        pred, v_s_dot = jax.jvp(flow_vector_field_obs, (observations,), (s_dot,))

        v_a_k_dot = jnp.zeros_like(v_s_dot)
        if self.config["use_jvp_ak"]:

            def flow_vector_field_action(actions_t):
                return self.network.select("actor_bc_flow")(
                    observations, actions_t, t, params=grad_params
                )

            _, v_a_k_dot = jax.jvp(flow_vector_field_action, (x_t,), (a_dot_flat,))
            if self.config["stop_gradient_jvp_ak"]:
                v_a_k_dot = jax.lax.stop_gradient(v_a_k_dot)
            residual = (1.0 - t) * (v_s_dot + t * v_a_k_dot - a_dot_flat)
        else:
            if self.config["use_1_k"]:
                v_s_dot = (1.0 - t) * v_s_dot
            residual = v_s_dot - a_dot_flat

        kinematic_valid = self._kinematic_valid_mask(batch, a_dot.shape[1])
        residual_chunk = jnp.reshape(residual, a_dot.shape)
        per_step = jnp.mean(jnp.square(residual_chunk), axis=-1)
        num_valid = jnp.maximum(kinematic_valid.sum(), jnp.asarray(1.0, dtype=per_step.dtype))
        kinematic_loss = (per_step * kinematic_valid).sum() / num_valid

        def masked_rms(values):
            values_chunk = jnp.reshape(values, a_dot.shape)
            per_step_sq = jnp.mean(jnp.square(values_chunk), axis=-1)
            return jnp.sqrt((per_step_sq * kinematic_valid).sum() / num_valid)

        info = {
            "kinematic_loss": kinematic_loss,
            "kinematic_valid_ratio": kinematic_valid.mean(),
            "a_dot_rms": masked_rms(a_dot_flat),
            "kinematic_state_jvp_rms": masked_rms(v_s_dot),
            "kinematic_action_jvp_rms": masked_rms(v_a_k_dot),
        }
        return pred, kinematic_loss, info

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))  # fold in horizon_length together with action_dim
        else:
            batch_actions = batch["actions"][..., 0, :] # take the first one
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        lambda_flow_k = self.config["lambda_flow_k"]
        if lambda_flow_k > 0:
            pred, kinematic_loss, kinematic_info = self._compute_kinematic_loss(
                batch, x_t, t, grad_params
            )
        else:
            pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
            kinematic_loss = jnp.zeros((), dtype=pred.dtype)
            kinematic_info = {
                "kinematic_loss": kinematic_loss,
                "kinematic_valid_ratio": jnp.zeros((), dtype=pred.dtype),
                "a_dot_rms": jnp.zeros((), dtype=pred.dtype),
                "kinematic_state_jvp_rms": jnp.zeros((), dtype=pred.dtype),
                "kinematic_action_jvp_rms": jnp.zeros((), dtype=pred.dtype),
            }

        # only bc on the valid chunk indices
        if self.config["action_chunking"]:
            bc_flow_loss = jnp.mean(
                jnp.reshape(
                    (pred - vel) ** 2, 
                    (batch_size, self.config["horizon_length"], self.config["action_dim"]) 
                ) * batch["valid"][..., None]
            )
        else:
            bc_flow_loss = jnp.mean(jnp.square(pred - vel))

        if self.config["actor_type"] == "distill-ddpg":
            # Distillation loss.
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
            actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)
            
            # Q loss.
            actor_actions = jnp.clip(actor_actions, -1, 1)

            qs = self.network.select(f'critic')(batch['observations'], actions=actor_actions)
            q = jnp.mean(qs, axis=0)
            q_loss = -q.mean()
        else:
            distill_loss = jnp.zeros(())
            q_loss = jnp.zeros(())

        # Total loss.
        actor_loss = (
            bc_flow_loss
            + lambda_flow_k * kinematic_loss
            + self.config['alpha'] * distill_loss
            + q_loss
        )

        info = {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'lambda_flow_k': jnp.asarray(lambda_flow_k, dtype=actor_loss.dtype),
        }
        info.update(kinematic_info)
        return actor_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)
    
    @jax.jit
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        # update_size = batch["observations"].shape[0]
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)
    
    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        
        if self.config["actor_type"] == "distill-ddpg":
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                    self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1),
                ),
            )
            actions = self.network.select(f'actor_onestep_flow')(observations, noises)
            actions = jnp.clip(actions, -1, 1)

        elif self.config["actor_type"] == "best-of-n":
            action_dim = self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1)
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                    self.config["actor_num_samples"], action_dim
                ),
            )
            observations = jnp.repeat(observations[..., None, :], self.config["actor_num_samples"], axis=-2)
            actions = self.compute_flow_actions(observations, noises)
            actions = jnp.clip(actions, -1, 1)
            if self.config["q_agg"] == "mean":
                q = self.network.select("critic")(observations, actions).mean(axis=0)
            else:
                q = self.network.select("critic")(observations, actions).min(axis=0)
            indices = jnp.argmax(q, axis=-1)

            bshape = indices.shape
            indices = indices.reshape(-1)
            bsize = len(indices)
            actions = jnp.reshape(actions, (-1, self.config["actor_num_samples"], action_dim))[jnp.arange(bsize), indices, :].reshape(
                bshape + (action_dim,))

        return actions

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        return self._flow_actions(observations, noises, params=None, clip=True)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
        )

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        
        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, full_actions)),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
        )
        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config["weight_decay"] > 0.:
            network_tx = optax.adamw(learning_rate=config['lr'], weight_decay=config["weight_decay"])
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params

        params[f'modules_target_critic'] = params[f'modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        if config["lambda_flow_k"] < 0:
            raise ValueError(f"lambda_flow_k must be >= 0, got {config['lambda_flow_k']}")
        if config["sample_frequency"] <= 0:
            raise ValueError(f"sample_frequency must be > 0, got {config['sample_frequency']}")
        if config["dct_coe_num"] < 0:
            raise ValueError(f"dct_coe_num must be >= 0, got {config['dct_coe_num']}")
        if config["action_chunking"] and config["dct_coe_num"] > config["horizon_length"]:
            raise ValueError(
                f"dct_coe_num must be in [0, horizon_length] "
                f"(got {config['dct_coe_num']} for horizon_length={config['horizon_length']})"
            )

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():

    config = ml_collections.ConfigDict(
        dict(
            agent_name='acfql',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            alpha=100.0,  # BC coefficient (need to be tuned for each environment).
            num_qs=2, # critic ensemble size
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=None,  # Visual encoder name (None, 'impala_small', etc.).
            horizon_length=ml_collections.config_dict.placeholder(int), # will be set
            action_chunking=True,  # False means n-step return
            actor_type="distill-ddpg",
            actor_num_samples=32,  # for actor_type="best-of-n" only
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
            lambda_flow_k=0.0,  # Weight for high-order / kinematic JVP loss (0 disables it).
            dct_coe_num=0,  # 0: forward-difference a_dot; >0: truncated DCT a_dot.
            use_jvp_ak=False,  # Add action-input JVP term to the kinematic residual.
            stop_gradient_jvp_ak=False,  # Stop gradients through the action-input JVP.
            use_1_k=False,  # Scale the state JVP by (1 - t) when use_jvp_ak is disabled.
            sample_frequency=1.0,  # Control frequency in Hz for action/state derivatives.
        )
    )
    return config
