"""Unit tests for classes in main.py."""  # :contentReference[oaicite:2]{index=2}

import os
import ast
import torch
import numpy as np
import gymnasium as gym
from tensordict import TensorDict
from torchrl.envs.transforms import Transform
from typing import Callable


# Dynamically load class definitions from main.py without running its scripts
_main_path = os.path.join(os.path.dirname(__file__), "../../../examples/PPO/main.py")
_main_src = open(_main_path).read()
_tree = ast.parse(_main_src)
_namespace: dict[str, object] = {
    "torch": torch,
    "gym": gym,
    "TensorDict": TensorDict,
    "Transform": Transform,
    "Callable": Callable,
}
for node in _tree.body:
    if isinstance(node, ast.ClassDef) and node.name in (
        "RewardPredictorTransform",
        "NNRewardWrapper",
    ):
        _src = ast.get_source_segment(_main_src, node)  # type: ignore
        exec(_src, _namespace)

RewardPredictorTransform = _namespace["RewardPredictorTransform"]  # type: ignore
NNRewardWrapper = _namespace["NNRewardWrapper"]  # type: ignore


def test_transform_sets_reward() -> None:
    """Test RewardPredictorTransform sets next reward in TensorDict."""

    def model(obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Return sum of obs and act as reward."""
        return obs.sum(dim=-1) + act.sum(dim=-1)

    trans = RewardPredictorTransform(model)
    td = TensorDict({}, batch_size=[2])
    obs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    act = torch.tensor([[0.5, 0.5], [1.0, 1.0]])
    td.set(("next", "observation"), obs)
    td.set("action", act)

    out_td = trans._call(td)
    reward = out_td.get(("next", "reward"))
    expected = (obs.sum(dim=-1) + act.sum(dim=-1)).unsqueeze(-1)
    assert torch.allclose(reward, expected)


def test_transform_returns_unmodified_if_missing() -> None:
    """Test transform returns unchanged dict if keys are missing."""
    trans = RewardPredictorTransform(lambda o, a: torch.zeros(1))
    td = TensorDict({}, batch_size=[1])
    result = trans._call(td)
    assert result is td
    # TensorDict doesn’t have .contains(); use the `in` operator instead
    assert ("next", "reward") not in result


def test_nnrewardwrapper_replaces_reward() -> None:
    """Test NNRewardWrapper replaces env reward with model output."""

    class DummyEnv(gym.Env):
        """Dummy env returning fixed values."""

        def __init__(self) -> None:
            """Initialize dummy environment."""
            super().__init__()

        def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
            """Return fixed observation, original reward, flags, and info."""
            obs = np.array([0.1, -0.2], dtype=np.float32)
            orig_r = 5.0
            terminated = False
            truncated = True
            info: dict = {"ok": True}
            return obs, orig_r, terminated, truncated, info

    def reset(**kwargs: object) -> tuple[np.ndarray, dict]:
        """Reset environment (not used in this test)."""
        return np.array([0.1, -0.2], dtype=np.float32), {}

    def model(obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Return tensor with value 42.0."""
        return torch.tensor([42.0])

    wrapper = NNRewardWrapper(DummyEnv(), model, torch.device("cpu"))
    action = np.array([1.0, -1.0], dtype=np.float32)
    obs_out, nn_r, term, trunc, info = wrapper.step(action)

    assert np.allclose(obs_out, np.array([0.1, -0.2], dtype=np.float32))
    assert nn_r == 42.0
    assert term is False
    assert trunc is True
    assert info == {"ok": True}
