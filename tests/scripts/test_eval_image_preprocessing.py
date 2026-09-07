"""Check that ordinary suite evaluation reaches the existing exact image path."""

from unittest.mock import Mock

import draccus
import pytest

from lerobot.configs.default import EvalConfig
from lerobot.scripts import lerobot_eval


@pytest.mark.parametrize("device", [None, "cuda:0"])
def test_suite_evaluation_forwards_image_device_without_changing_rollouts(monkeypatch, device):
    calls = []

    def evaluate(**kwargs):
        calls.append(kwargs)
        return {
            "per_episode": [
                {"sum_reward": i % 2, "max_reward": i % 2, "success": bool(i % 2)}
                for i in range(kwargs["n_episodes"])
            ],
        }

    monkeypatch.setattr(lerobot_eval, "eval_policy", evaluate)
    envs = [Mock(), Mock()]
    for env in envs:
        del env._ensure
    policy = object()
    result = lerobot_eval.eval_policy_all(
        envs={"libero_10": dict(enumerate(envs))}, policy=policy,
        env_preprocessor=None, env_postprocessor=None,
        preprocessor=None, postprocessor=None, n_episodes=50,
        start_seed=1000, image_preprocessing_device=device,
    )
    assert result["overall"]["n_episodes"] == 100
    assert result["overall"]["pc_success"] == 50
    assert [r["task_id"] for r in result["per_task"]] == [0, 1]
    for env, call in zip(envs, calls, strict=True):
        assert call["env"] is env and call["policy"] is policy
        assert call["image_preprocessing_device"] == device
        assert call["start_seed"] == 1000 and call["n_episodes"] == 50
        env.close.assert_called_once()


def test_image_preprocessing_is_opt_in_and_parses_from_eval_config():
    assert not EvalConfig(batch_size=10).fast_image_preprocessing
    with draccus.config_type("json"):
        cfg = draccus.decode(EvalConfig, {"batch_size": 10, "fast_image_preprocessing": True})
    assert cfg.fast_image_preprocessing and cfg.batch_size == 10 and cfg.n_episodes == 50
