"""Compatibility checks for the optional LIBERO-Plus evaluation fast path.

Run in the LIBERO-Plus environment with its repository on PYTHONPATH. The
reference branch remains executable to check pixels and RNG, not just metrics.
"""

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from lerobot.envs.utils import _LazyAsyncVectorEnv, preprocess_observation


@pytest.fixture(scope="module")
def corruption_module():
    pytest.importorskip("numba")
    pytest.importorskip("wand.api")
    return pytest.importorskip("libero.libero.envs.env_wrapper")


def assert_rng_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


@pytest.mark.parametrize("severity", range(1, 11))
@pytest.mark.parametrize("shape,seed", [((48, 53, 3), 1000), ((24, 32, 3), 17)])
def test_glass_pixels_and_rng(corruption_module, monkeypatch, severity, shape, seed):
    pixels = np.random.default_rng(7).integers(0, 256, size=shape, dtype=np.uint8)
    monkeypatch.setenv("LIBERO_PLUS_FAST_CORRUPTIONS", "0")
    np.random.seed(seed)
    # Include a cached Gaussian variate in the global RNG state.
    np.random.normal()
    reference = corruption_module.glass_blur(pixels, severity)
    reference_rng = np.random.get_state()

    monkeypatch.setenv("LIBERO_PLUS_FAST_CORRUPTIONS", "1")
    np.random.seed(seed)
    np.random.normal()
    actual = corruption_module.glass_blur(pixels, severity)
    np.testing.assert_array_equal(actual, reference)
    assert_rng_equal(np.random.get_state(), reference_rng)


@pytest.mark.parametrize("severity", [1, 5, 10])
def test_glass_full_resolution(corruption_module, monkeypatch, severity):
    test_glass_pixels_and_rng(corruption_module, monkeypatch, severity, (256, 256, 3), 1000)


def test_compilation_does_not_change_rng(corruption_module):
    from libero.libero.envs.fast_corruptions import warmup

    np.random.seed(1000)
    before = np.random.get_state()
    warmup()
    assert_rng_equal(np.random.get_state(), before)


def test_static_metadata_does_not_spawn_workers():
    def fail_factory():
        raise AssertionError("Static metadata must not construct an environment")

    env = _LazyAsyncVectorEnv(
        [fail_factory, fail_factory],
        observation_space=Box(0, 1, (2,)),
        action_space=Box(-1, 1, (1,)),
        metadata={},
        static_attributes={"task_description": ["task a", "task b"]},
    )
    assert env.call("task_description") == ("task a", "task b")
    assert env._env is None

    class LiveEnv:
        def call(self, name, *args, **kwargs):
            return name, args, kwargs

    env._env = LiveEnv()
    assert env.call("dynamic") == ("dynamic", (), {})
    assert env.call("task_description", "arg") == ("task_description", ("arg",), {})


def test_static_metadata_requires_one_value_per_env():
    with pytest.raises(ValueError, match="one value per environment"):
        _LazyAsyncVectorEnv([lambda: None], static_attributes={"task": []})


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("batched", [False, True])
def test_preprocess_images_exact(device, batched):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required for GPU equivalence check")
    # Every possible uint8 value, both camera streams, state and layout.
    pixels = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    if batched:
        pixels = np.stack([pixels, pixels[::-1]])
    state = np.zeros((2, 7) if batched else (7,), dtype=np.float64)
    obs = {"pixels": {"front": pixels, "wrist": pixels.copy()}, "agent_pos": state}
    reference = preprocess_observation(obs)
    actual = preprocess_observation(obs, image_device=device)
    assert actual.keys() == reference.keys()
    for key, value in actual.items():
        assert torch.equal(value.cpu(), reference[key]), key
        assert value.dtype == reference[key].dtype
        assert value.is_contiguous()
        assert value.device.type == (device if "images" in key else "cpu")
