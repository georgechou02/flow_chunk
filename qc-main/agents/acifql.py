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


class ACIFQLAgent(flax.struct.PyTreeNode):
    """QC-IFQL: implicit flow Q-learning on a chunked action space.

    This is Park et al.'s IFQL (IQL expectile V/Q, flow-matching BC, best-of-N
    rejection sampling) with Q-chunking: the flow policy, Q, and V all operate
    on action chunks, and the critic uses an n-step backup of length ``H``.
    It is not QC (TD critic + best-of-N) and not QC-FQL (distilled one-step actor).
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def _batch_actions(self, batch):
        """Flatten or trim the action chunk the flow actor actually models."""
        actions = batch["actions"]
        if actions.ndim == 2:
            actions = actions[:, None, :]
        if self.config["action_chunking"]:
            return jnp.reshape(actions, (actions.shape[0], -1))
        return actions[:, 0, :]

    def _action_chunk(self, batch):
        """Return the action chunk used for kinematic ``ȧ``."""
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

        Retain QC's central ``(s_{t+1} - s_{t-1}) * fps / 2`` default.
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

    def _next_observation(self, batch):
        """Observation at the end of the n-step / chunk backup."""
        observations = batch["observations"]
        next_observations = batch["next_observations"]
        if next_observations.ndim == observations.ndim + 1:
            return next_observations[..., -1, :]
        return next_observations

    def _last_step_valid(self, batch):
        if "valid" not in batch:
            return jnp.ones(batch["actions"].shape[:1], dtype=jnp.float32)
        return batch["valid"][..., -1]

    def _kinematic_valid_mask(self, batch, num_steps):
        """Per-step mask for kinematic supervision."""
        if "valid" in batch:
            kinematic_valid = batch["valid"][:, :num_steps]
            if "masks" in batch:
                kinematic_valid = kinematic_valid * batch["masks"][:, :num_steps]
        else:
            batch_size = batch["actions"].shape[0]
            kinematic_valid = jnp.ones((batch_size, num_steps), dtype=jnp.float32)

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

    def value_loss(self, batch, grad_params):
        """IQL expectile V loss on the chunked Q."""
        batch_actions = self._batch_actions(batch)
        q1, q2 = self.network.select("target_critic")(
            batch["observations"], actions=batch_actions
        )
        q = jnp.minimum(q1, q2)
        v = self.network.select("value")(batch["observations"], params=grad_params)
        valid = self._last_step_valid(batch)
        per_sample = self.expectile_loss(q - v, q - v, self.config["expectile"])
        value_loss = (per_sample * valid).mean()

        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    def critic_loss(self, batch, grad_params):
        """IQL critic loss with an n-step / chunk backup through V(s')."""
        batch_actions = self._batch_actions(batch)
        next_v = self.network.select("value")(self._next_observation(batch))
        target_q = (
            batch["rewards"][..., -1]
            + (self.config["discount"] ** self.config["horizon_length"])
            * batch["masks"][..., -1]
            * next_v
        )
        q1, q2 = self.network.select("critic")(
            batch["observations"], actions=batch_actions, params=grad_params
        )
        valid = self._last_step_valid(batch)
        td = (q1 - target_q) ** 2 + (q2 - target_q) ** 2
        critic_loss = (td * valid).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": target_q.mean(),
            "q_max": target_q.max(),
            "q_min": target_q.min(),
        }

    def _flow_actions(self, observations, noises, params=None, clip=True):
        """Integrate the flow vector field into an action chunk (inference ODE)."""
        is_encoded = False
        if self.config["encoder"] is not None:
            observations = self.network.select("actor_flow_encoder")(
                observations, params=params
            )
            is_encoded = True

        actions = noises
        n_steps = self.config["flow_steps"]
        for i in range(n_steps):
            t = jnp.full((*observations.shape[:-1], 1), i / n_steps)
            vels = self.network.select("actor_flow")(
                observations, actions, t, is_encoded=is_encoded, params=params
            )
            actions = actions + vels / n_steps
        if clip:
            actions = jnp.clip(actions, -1, 1)
        return actions

    def _compute_kinematic_loss(self, batch, x_t, t, grad_params):
        """High-order constraint on the interpolant vector field ``v_θ(s, x_t, t)``.

        Same residual as QC's flow actor / multi-task DiT. Disabled when
        ``lambda_flow_k == 0``.
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
            return self.network.select("actor_flow")(obs, x_t, t, params=grad_params)

        pred, v_s_dot = jax.jvp(flow_vector_field_obs, (observations,), (s_dot,))

        def flow_vector_field_action(actions_t):
            return self.network.select("actor_flow")(
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
        """Flow-matching BC on the action chunk, plus optional kinematic JVP."""
        batch_actions = self._batch_actions(batch)
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

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
            pred = self.network.select("actor_flow")(
                batch["observations"], x_t, t, params=grad_params
            )
            kinematic_loss = jnp.zeros((), dtype=pred.dtype)
            kinematic_info = {
                "kinematic_loss": kinematic_loss,
                "kinematic_valid_ratio": jnp.zeros((), dtype=pred.dtype),
                "a_dot_rms": jnp.zeros((), dtype=pred.dtype),
                "kinematic_state_jvp_rms": jnp.zeros((), dtype=pred.dtype),
                "kinematic_action_jvp_rms": jnp.zeros((), dtype=pred.dtype),
            }

        if self.config["action_chunking"]:
            bc_flow_loss = jnp.mean(
                jnp.reshape(
                    (pred - vel) ** 2,
                    (batch_size, self.config["horizon_length"], self.config["action_dim"]),
                )
                * batch["valid"][..., None]
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

        actor_loss = bc_flow_loss + lambda_flow_k * kinematic_loss + weighted_physical_loss
        info = {
            "actor_loss": actor_loss,
            "bc_flow_loss": bc_flow_loss,
            "lambda_flow_k": jnp.asarray(lambda_flow_k, dtype=actor_loss.dtype),
            "physical_loss": physical_loss,
            "phy_loss_weight": jnp.asarray(phy_loss_weight, dtype=actor_loss.dtype),
            "weighted_physical_loss": weighted_physical_loss,
        }
        info.update(kinematic_info)
        return actor_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng = jax.random.split(rng)

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f"value/{k}"] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = value_loss + critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @staticmethod
    def _update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, "critic")
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def compute_flow_actions(self, observations, noises):
        """Compute actions from the BC flow model using the Euler method."""
        return self._flow_actions(observations, noises, params=None, clip=True)

    @jax.jit
    def sample_actions(self, observations, rng=None):
        """Best-of-N rejection sampling from the chunked flow policy."""
        action_dim = self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )
        noises = jax.random.normal(
            rng,
            (
                *observations.shape[: -len(self.config["ob_dims"])],
                self.config["num_samples"],
                action_dim,
            ),
        )
        sampled_observations = jnp.repeat(
            observations[..., None, :], self.config["num_samples"], axis=-2
        )
        actions = self.compute_flow_actions(sampled_observations, noises)
        actions = jnp.clip(actions, -1, 1)
        if self.config["q_agg"] == "mean":
            q = self.network.select("critic")(sampled_observations, actions).mean(axis=0)
        else:
            q = self.network.select("critic")(sampled_observations, actions).min(axis=0)
        indices = jnp.argmax(q, axis=-1)

        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        actions = jnp.reshape(actions, (-1, self.config["num_samples"], action_dim))[
            jnp.arange(bsize), indices, :
        ].reshape(bshape + (action_dim,))
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new agent."""
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

        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["value"] = encoder_module()
            encoders["critic"] = encoder_module()
            encoders["actor_flow"] = encoder_module()

        value_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=1,
            encoder=encoders.get("value"),
        )
        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=2,
            encoder=encoders.get("critic"),
        )
        actor_flow_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("actor_flow"),
        )

        network_info = dict(
            value=(value_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            actor_flow=(actor_flow_def, (ex_observations, full_actions, ex_times)),
        )
        if encoders.get("actor_flow") is not None:
            network_info["actor_flow_encoder"] = (encoders.get("actor_flow"), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_critic"] = params["modules_critic"]

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim
        _validate_kinematic_config(config)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def _validate_kinematic_config(config):
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


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="acifql",
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            expectile=0.9,  # IQL expectile; Park / QC-IFQL default.
            q_agg="min",  # Rejection sampling uses min over the Q ensemble.
            num_samples=32,  # Best-of-N samples; Park / QC-IFQL default.
            flow_steps=10,
            encoder=None,
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=True,
            lambda_flow_k=0.0,  # 0 is vanilla QC-IFQL; >0 adds the kinematic JVP.
            phy_loss_weight=0.0,  # Weight for the value-level physical residual loss.
            dct_coe_num=0,
            derivative_kind="auto",
            savgol_window=5,
            savgol_polyorder=2,
            bspline_num_control_points=0,
            bspline_coe_num=0,
            bspline_degree=2,
            chebyshev_num_modes=0,
            action_jvp_grad_scale=1.0,  # Full action-JVP gradient, matching LeRobot main.
            sample_frequency=1.0,
            state_derivative_mode="central",
        )
    )
    return config
