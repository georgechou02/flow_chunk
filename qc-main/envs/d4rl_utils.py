import os

import d4rl
import gym
import gymnasium
import numpy as np

from envs.env_utils import EpisodeMonitor
from utils.datasets import Dataset

DEFAULT_SMALL_SAMPLES_DIR = "/home/chengpeng/small_samples"


class GymToGymnasium(gymnasium.Env):
    """Adapt a classic gym D4RL env to the gymnasium API used by QC."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env):
        super().__init__()
        self._gym_env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    @property
    def unwrapped(self):
        inner = self._gym_env
        while hasattr(inner, "env"):
            inner = inner.env
        return inner

    def reset(self, *, seed=None, options=None):
        if seed is not None and hasattr(self._gym_env, "seed"):
            self._gym_env.seed(seed)
        observation = self._gym_env.reset()
        if isinstance(observation, tuple):
            observation, info = observation[0], (observation[1] if len(observation) > 1 else {})
        else:
            info = {}
        return observation, info

    def step(self, action):
        step_res = self._gym_env.step(action)
        if len(step_res) == 5:
            return step_res
        observation, reward, done, info = step_res
        return observation, reward, bool(done), False, info

    def render(self):
        return self._gym_env.render(mode="rgb_array")

    def close(self):
        return self._gym_env.close()

    def get_normalized_score(self, returns):
        return self.unwrapped.get_normalized_score(returns)


def make_env(env_name):
    """Make D4RL environment."""
    env = gym.make(env_name)
    env = GymToGymnasium(env)
    env = EpisodeMonitor(env)
    return env


def _load_qlearning_dataset(env, env_name, ratio, small_samples_dir):
    """Load a D4RL q-learning dict, optionally from a subsampled npy."""
    if ratio == 1:
        return d4rl.qlearning_dataset(env.unwrapped)

    partial_path = os.path.join(small_samples_dir, f"{env_name}-ratio-{ratio}-seed-111.npy")
    if not os.path.isfile(partial_path):
        raise FileNotFoundError(
            f"D4RL small-sample file not found: {partial_path}. "
            "Use --ratio=1 for the full dataset, or pass --small_samples_dir."
        )
    dataset = np.load(partial_path, allow_pickle=True)[0]
    return {
        "observations": np.asarray(dataset["observations"]),
        "actions": np.asarray(dataset["actions"]),
        "next_observations": np.asarray(dataset["next_observations"]),
        "rewards": np.asarray(dataset["rewards"]),
        "terminals": np.asarray(dataset["terminals"]),
    }


def get_dataset(
    env,
    env_name,
    ratio=1,
    small_samples_dir=DEFAULT_SMALL_SAMPLES_DIR,
):
    """Make D4RL dataset.

    Args:
        env: Environment instance.
        env_name: Name of the environment.
        ratio: 1 uses the full D4RL dataset; otherwise load the subsampled npy
            at ``{small_samples_dir}/{env_name}-ratio-{ratio}-seed-111.npy``.
        small_samples_dir: Directory containing subsampled D4RL datasets.
    """
    dataset = _load_qlearning_dataset(env, env_name, ratio, small_samples_dir)

    terminals = np.zeros_like(dataset['rewards'])  # Indicate the end of an episode.
    masks = np.zeros_like(dataset['rewards'])  # Indicate whether we should bootstrap from the next state.
    rewards = dataset['rewards'].copy().astype(np.float32)
    raw_terminals = np.asarray(dataset['terminals']).astype(np.float32)
    if 'antmaze' in env_name:
        for i in range(len(terminals) - 1):
            terminals[i] = float(
                np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6
            )
            masks[i] = 1 - raw_terminals[i]
        rewards = rewards - 1.0
    else:
        for i in range(len(terminals) - 1):
            if (
                np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6
                or raw_terminals[i] == 1.0
            ):
                terminals[i] = 1
            else:
                terminals[i] = 0
            masks[i] = 1 - raw_terminals[i]
    masks[-1] = 1 - raw_terminals[-1]
    terminals[-1] = 1

    return Dataset.create(
        observations=dataset['observations'].astype(np.float32),
        actions=dataset['actions'].astype(np.float32),
        next_observations=dataset['next_observations'].astype(np.float32),
        terminals=terminals.astype(np.float32),
        rewards=rewards,
        masks=masks,
    )
