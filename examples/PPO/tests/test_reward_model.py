"""Pytest for examples/PPO/test_reward_model.py"""

# tests/test_reward_model.py
# run with: pytest -vv test_reward_model.py

import numpy as np
import torch
import pytest
import sys

sys.path.append("../")
import train_reward_model as reward_script
from train_reward_model import RewardPredictor

from _pytest.monkeypatch import MonkeyPatch


def test_numpy_bool8_alias_exists() -> None:
    """Verify that numpy.bool8 exists and aliases numpy.bool_."""
    assert hasattr(np, "bool8"), "np.bool8 should be defined."
    assert np.bool8 is np.bool_, "np.bool8 should alias np.bool_."


def test_init_default_hidden_sizes() -> None:
    """Test that default hidden_sizes produces correct layer dimensions."""
    state_dim = 3
    action_dim = 2
    model = RewardPredictor(state_dim=state_dim, action_dim=action_dim)
    # dims = [state_dim + action_dim, 64, 64, 1]
    seq = list(model.net)
    # Expect layers: Linear(5->64), ReLU, Linear(64->64), ReLU, Linear(64->1)
    assert isinstance(seq[0], torch.nn.Linear)
    assert seq[0].in_features == state_dim + action_dim
    assert seq[0].out_features == 64
    assert isinstance(seq[1], torch.nn.ReLU)
    assert isinstance(seq[2], torch.nn.Linear)
    assert seq[2].in_features == 64
    assert seq[2].out_features == 64
    assert isinstance(seq[3], torch.nn.ReLU)
    assert isinstance(seq[4], torch.nn.Linear)
    assert seq[4].in_features == 64
    assert seq[4].out_features == 1
    # No extra layers
    assert len(seq) == 5


def test_init_custom_hidden_sizes() -> None:
    """Test that custom hidden_sizes are reflected in the network."""
    state_dim = 4
    action_dim = 1
    hidden = [10, 20, 30]
    model = RewardPredictor(
        state_dim=state_dim, action_dim=action_dim, hidden_sizes=hidden
    )
    seq = list(model.net)
    # There should be len(hidden)*2 (Linear+ReLU) + final Linear
    expected_layers = len(hidden) * 2 + 1
    assert len(seq) == expected_layers
    # Check first linear matches dims[0] -> hidden[0]
    first = seq[0]
    assert isinstance(first, torch.nn.Linear)
    assert first.in_features == state_dim + action_dim
    assert first.out_features == hidden[0]
    # Check last linear matches hidden[-1] -> 1
    last = seq[-1]
    assert isinstance(last, torch.nn.Linear)
    assert last.in_features == hidden[-1]
    assert last.out_features == 1


def test_forward_output_shape() -> None:
    """Verify forward returns correct shape for a batch of inputs."""
    state_dim = 2
    action_dim = 2
    batch = 7
    model = RewardPredictor(state_dim=state_dim, action_dim=action_dim)
    # random input
    s = torch.randn(batch, state_dim)
    a = torch.randn(batch, action_dim)
    out = model(s, a)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (batch,)


def test_forward_single_sample() -> None:
    """Verify forward handles batch size of one correctly."""
    state_dim = 5
    action_dim = 3
    model = RewardPredictor(state_dim=state_dim, action_dim=action_dim)
    s = torch.randn(1, state_dim)
    a = torch.randn(1, action_dim)
    out = model(s, a)
    assert out.shape == (1,)
    # Value should be a Python float when detached and item() is called
    assert isinstance(out.detach().item(), float)


def test_save_and_load_state_dict(tmp_path: pytest.TempPathFactory) -> None:
    """Test that saving and loading the model state dict recreates weights."""
    state_dim = 3
    action_dim = 4
    model = RewardPredictor(state_dim=state_dim, action_dim=action_dim)
    # change weights to non-default for test
    for param in model.parameters():
        torch.nn.init.constant_(param, 0.123)
    save_file = tmp_path / "test_model.pth"
    torch.save(model.state_dict(), str(save_file))
    # Load into new model
    new_model = RewardPredictor(state_dim=state_dim, action_dim=action_dim)
    state = torch.load(str(save_file))
    new_model.load_state_dict(state)
    # Compare every parameter tensor
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)


@pytest.fixture(autouse=True)
def clean_reward_dir(tmp_path: pytest.TempPathFactory, monkeypatch: MonkeyPatch) -> None:
    """Redirect reward_model_dir to a temporary path before running tests."""
    import sys

    sys.path.append("../")
    monkeypatch.setattr(reward_script, "reward_model_dir", str(tmp_path))
    # ensure directory creation logic runs
    reward_script.os.makedirs(str(tmp_path), exist_ok=True)
