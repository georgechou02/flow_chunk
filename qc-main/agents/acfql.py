import copy
import math
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.action_derivatives import action_dot, mixes_frames, physical_action_dot, resolve_kind
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
        """State tangent at the current observation (DiT ``conditioning_derivative_mode``).

        Retain QC's central ``(s_{t+1} - s_{t-1}) * fps / 2`` default. Forward uses ``next_observations``;
        reverse / central also need ``prev_observations``.
        """
        observations = batch["observations"]
        next_observations = batch["next_observations"]
        if next_observations.ndim == observations.ndim + 1:
            next_obs0 = next_observations[:, 0]
        else:
            next_obs0 = next_observations
        fps = jnp.asarray(fps, dtype=observations.dtype)
        mode = self.config.get("state_derivative_mode", "central")
        if mode == "forward":
            return (next_obs0 - observations) * fps
        if "prev_observations" not in batch:
            raise ValueError(f"{mode} state derivatives require prev_observations in the batch")
        prev_observations = batch["prev_observations"]
        if mode == "reverse":
            return (observations - prev_observations) * fps
        if mode == "central":
            return (next_obs0 - prev_observations) * (fps * jnp.asarray(0.5, dtype=observations.dtype))
        raise ValueError(
            "state_derivative_mode must be 'reverse', 'forward', or 'central', "
            f"got {mode!r}"
        )

    def _kinematic_valid_mask(self, batch, num_steps):
        """Per-step mask for kinematic supervision."""
        if "valid" in batch:
            kinematic_valid = batch["valid"][:, :num_steps]
            if "masks" in batch:
                kinematic_valid = kinematic_valid * batch["masks"][:, :num_steps]
        else:
            batch_size = batch["actions"].shape[0]
            kinematic_valid = jnp.ones((batch_size, num_steps), dtype=jnp.float32)

        # Every estimator except forward differences couples multiple frames, so
        # drop samples that cross an episode boundary rather than differentiating
        # a mixed trajectory. Savitzky-Golay only couples within its window, but
        # the chunk-wide mask is a cheap over-approximation at these horizons.
        if mixes_frames(self.config["derivative_kind"], self.config["dct_coe_num"]):
            all_valid = jnp.all(kinematic_valid > 0, axis=-1, keepdims=True)
            kinematic_valid = kinematic_valid * all_valid.astype(kinematic_valid.dtype)
        if (
            self.config.get("state_derivative_mode", "central") != "forward"
            and "prev_valid" in batch
        ):
            kinematic_valid = kinematic_valid * batch["prev_valid"][:, None].astype(
                kinematic_valid.dtype
            )
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

        Matches QC's flow actor and multi-task DiT: JVP of the flow velocity on
        ``x_t = (1-t)z + t a``, not of the Euler-integrated policy. ``ȧ`` is the
        chunk derivative of the expert actions (see ``utils.action_derivatives``).
        Residual is always ``(1-t)(v_s_dot + t·v_a_k_dot - ȧ)``,
        matching the current LeRobot main algorithm.
        """
        fps = self.config["sample_frequency"]
        actions, next_actions = self._action_chunk(batch)
        a_dot = action_dot(
            actions,
            next_actions=next_actions,
            fps=fps,
            dct_coe_num=self.config["dct_coe_num"],
            kind=self.config["derivative_kind"],
            savgol_window=self.config["savgol_window"],
            savgol_polyorder=self.config["savgol_polyorder"],
            bspline_num_control_points=(
                self.config.get("bspline_coe_num", 0)
                or self.config["bspline_num_control_points"]
                or None
            ),
            bspline_degree=self.config["bspline_degree"],
            chebyshev_num_modes=self.config["chebyshev_num_modes"] or None,
        )
        a_dot_flat = jnp.reshape(a_dot, (a_dot.shape[0], -1))
        s_dot = self._state_dot(batch, fps)
        observations = batch["observations"]

        def flow_vector_field_obs(obs):
            return self.network.select("actor_bc_flow")(obs, x_t, t, params=grad_params)

        pred, v_s_dot = jax.jvp(flow_vector_field_obs, (observations,), (s_dot,))

        def flow_vector_field_action(actions_t):
            return self.network.select("actor_bc_flow")(
                observations, actions_t, t, params=grad_params
            )

        _, v_a_k_dot = jax.jvp(flow_vector_field_action, (x_t,), (a_dot_flat,))
        action_jvp_grad_scale = self.config["action_jvp_grad_scale"]
        if action_jvp_grad_scale == 0.0:
            v_a_k_dot_for_loss = jax.lax.stop_gradient(v_a_k_dot)
        elif action_jvp_grad_scale == 1.0:
            v_a_k_dot_for_loss = v_a_k_dot
        else:
            detached = jax.lax.stop_gradient(v_a_k_dot)
            v_a_k_dot_for_loss = detached + action_jvp_grad_scale * (v_a_k_dot - detached)
        # Match LeRobot main: both JVPs are present in the forward residual.
        # The time factor is applied BEFORE squaring; grad_scale only changes
        # the direct kinematic gradient through the action-input JVP.
        residual = (1.0 - t) * (v_s_dot + t * v_a_k_dot_for_loss - a_dot_flat)

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
            "kinematic_residual_time_weight_mean": (1.0 - t).mean(),
            "kinematic_loss_time_weight_mean": jnp.square(1.0 - t).mean(),
            "action_jvp_grad_scale": jnp.asarray(self.config["action_jvp_grad_scale"]),
            "kinematic_valid_ratio": kinematic_valid.mean(),
            "a_dot_rms": masked_rms(a_dot_flat),
            "kinematic_state_jvp_rms": masked_rms(v_s_dot),
            "kinematic_action_jvp_rms": masked_rms(v_a_k_dot),
        }
        return pred, kinematic_loss, info

    def _flow_error_chunk(self, flow_error):
        """Reshape a flattened flow error into ``(B, H, A)``."""
        if self.config["action_chunking"]:
            return jnp.reshape(
                flow_error,
                (flow_error.shape[0], self.config["horizon_length"], self.config["action_dim"]),
            )
        return flow_error[:, None, :]

    def _compute_physical_loss(self, batch, flow_error, t):
        """Value-level residual ``t(1-t) g(v_θ - (a - z))`` on the action chunk."""
        error_chunk = self._flow_error_chunk(flow_error)
        g_flow_error = physical_action_dot(
            error_chunk,
            fps=self.config["sample_frequency"],
            dct_coe_num=self.config["dct_coe_num"],
            kind=self.config["derivative_kind"],
            savgol_window=self.config["savgol_window"],
            savgol_polyorder=self.config["savgol_polyorder"],
            bspline_num_control_points=(
                self.config.get("bspline_coe_num", 0)
                or self.config["bspline_num_control_points"]
                or None
            ),
            bspline_degree=self.config["bspline_degree"],
            chebyshev_num_modes=self.config["chebyshev_num_modes"] or None,
        )
        t_expanded = t[:, :, None]
        physical_residual = t_expanded * (1.0 - t_expanded) * g_flow_error
        physical_valid = self._kinematic_valid_mask(batch, error_chunk.shape[1])
        per_step = jnp.mean(jnp.square(physical_residual), axis=-1)
        num_valid = jnp.maximum(physical_valid.sum(), jnp.asarray(1.0, dtype=per_step.dtype))
        physical_loss = (per_step * physical_valid).sum() / num_valid
        return physical_loss

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

        phy_loss_weight = float(self.config.get("phy_loss_weight", 0.0))
        flow_error = pred - vel
        physical_flow_error = flow_error if phy_loss_weight > 0 else jax.lax.stop_gradient(flow_error)
        physical_loss = self._compute_physical_loss(batch, physical_flow_error, t)
        weighted_physical_loss = (
            phy_loss_weight * physical_loss
            if phy_loss_weight > 0
            else jnp.zeros((), dtype=pred.dtype)
        )

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
            + weighted_physical_loss
            + self.config['alpha'] * distill_loss
            + q_loss
        )

        info = {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'lambda_flow_k': jnp.asarray(lambda_flow_k, dtype=actor_loss.dtype),
            'physical_loss': physical_loss,
            'phy_loss_weight': jnp.asarray(phy_loss_weight, dtype=actor_loss.dtype),
            'weighted_physical_loss': weighted_physical_loss,
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
        action_jvp_grad_scale = float(config["action_jvp_grad_scale"])
        if not math.isfinite(action_jvp_grad_scale) or not 0.0 <= action_jvp_grad_scale <= 1.0:
            raise ValueError("action_jvp_grad_scale must be finite and in [0, 1]")
        phy_loss_weight = float(config.get("phy_loss_weight", 0.0))
        if not math.isfinite(phy_loss_weight) or phy_loss_weight < 0:
            raise ValueError(f"phy_loss_weight must be finite and >= 0, got {phy_loss_weight}")
        if config.get("state_derivative_mode", "central") not in {"reverse", "forward", "central"}:
            raise ValueError(
                "state_derivative_mode must be 'reverse', 'forward', or 'central', "
                f"got {config.get('state_derivative_mode')!r}"
            )
        if config["sample_frequency"] <= 0:
            raise ValueError(f"sample_frequency must be > 0, got {config['sample_frequency']}")
        if config["dct_coe_num"] < 0:
            raise ValueError(f"dct_coe_num must be >= 0, got {config['dct_coe_num']}")
        if config["action_chunking"] and config["dct_coe_num"] > config["horizon_length"]:
            raise ValueError(
                f"dct_coe_num must be in [0, horizon_length] "
                f"(got {config['dct_coe_num']} for horizon_length={config['horizon_length']})"
            )

        # Fail here rather than inside the jitted loss, where the traceback would
        # point at the stencil construction instead of the offending config.
        derivative_kind = resolve_kind(config["derivative_kind"], config["dct_coe_num"])
        chunk_length = config["horizon_length"] if config["action_chunking"] else 1
        if derivative_kind == "savgol":
            if config["savgol_polyorder"] < 1:
                raise ValueError(
                    f"savgol_polyorder must be >= 1, got {config['savgol_polyorder']}"
                )
            if config["savgol_window"] <= config["savgol_polyorder"]:
                raise ValueError(
                    f"savgol_window must exceed savgol_polyorder, got "
                    f"{config['savgol_window']} <= {config['savgol_polyorder']}"
                )
            if config["savgol_window"] > chunk_length:
                raise ValueError(
                    f"savgol_window must be <= chunk length {chunk_length}, "
                    f"got {config['savgol_window']}"
                )
        elif derivative_kind == "bspline":
            if chunk_length < 2:
                raise ValueError("bspline derivatives require a chunk length of at least 2")
            if config["bspline_degree"] < 2:
                raise ValueError(
                    f"bspline_degree must be >= 2, got {config['bspline_degree']}"
                )
            num_control_points = (
                config.get("bspline_coe_num", 0)
                or config["bspline_num_control_points"]
                or chunk_length
            )
            if not config["bspline_degree"] < num_control_points <= chunk_length:
                raise ValueError(
                    "bspline requires 2 <= bspline_degree < M <= chunk length "
                    f"(got p={config['bspline_degree']}, M={num_control_points}, H={chunk_length})"
                )
        elif derivative_kind == "chebyshev":
            if chunk_length < 2:
                raise ValueError("chebyshev derivatives require a chunk length of at least 2")
            num_modes = config["chebyshev_num_modes"] or chunk_length
            if not 1 <= num_modes <= chunk_length:
                raise ValueError(
                    f"chebyshev_num_modes must be in [1, {chunk_length}], got {num_modes}"
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
            phy_loss_weight=0.0,  # Weight for the value-level physical residual loss.
            dct_coe_num=0,  # 0: forward-difference a_dot; >0: truncated DCT a_dot.
            # a_dot estimator: "auto" defers to dct_coe_num, or force one of
            # "forward" / "dct" / "savgol" / "bspline" / "chebyshev".
            derivative_kind="auto",
            savgol_window=5,  # Samples per local polynomial fit.
            savgol_polyorder=2,  # Local polynomial degree; must be < savgol_window.
            bspline_num_control_points=0,  # 0: M = chunk length (full fit, like DiT M=H).
            bspline_coe_num=0,  # v2 name for M; if set, overrides bspline_num_control_points.
            bspline_degree=2,  # Quadratic by default (DiT training script). Requires 2 <= p < M.
            chebyshev_num_modes=0,  # 0: keep all H Chebyshev modes.
            action_jvp_grad_scale=1.0,  # Full action-JVP gradient, matching LeRobot main.
            sample_frequency=1.0,  # Control frequency in Hz for action/state derivatives.
            # State tangent at the chunk start; retain the QC derivative default.
            state_derivative_mode="central",
        )
    )
    return config
