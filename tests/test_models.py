"""Unit tests for RewardPredictor class in models.py."""

import pytest
import torch
from torch import nn
import sys

sys.path.append("../")
from examples.PPO.models import RewardPredictor


class TestRewardPredictor:
    """Test suite for the RewardPredictor class."""

    def test_layer_structure_default(self) -> None:
        """Test default hidden sizes and layer structure."""
        state_dim = 2
        action_dim = 3
        model = RewardPredictor(state_dim, action_dim)
        # dims: [5, 64, 64, 1] => two hidden layers => 5 modules
        assert len(model.net) == 5
        # Validate that Linear and ReLU alternate, ending with Linear
        expected_dims = [state_dim + action_dim, 64, 64, 1]
        modules = list(model.net)
        # Linear modules are at indices 0, 2, 4
        linear_indices = [0, 2, 4]
        for idx in linear_indices:
            assert isinstance(modules[idx], nn.Linear)
        # ReLU modules are at indices 1, 3
        for idx in [1, 3]:
            assert isinstance(modules[idx], nn.ReLU)
        # Check Linear layer shapes
        in_out_pairs = [
            (model.net[i].in_features, model.net[i].out_features) for i in linear_indices
        ]
        assert in_out_pairs == [
            (expected_dims[0], expected_dims[1]),
            (expected_dims[1], expected_dims[2]),
            (expected_dims[2], expected_dims[3]),
        ]

    def test_layer_structure_custom(self) -> None:
        """Test custom hidden sizes and layer structure."""
        state_dim = 4
        action_dim = 1
        hidden_sizes = [10, 20, 30]
        model = RewardPredictor(state_dim, action_dim, hidden_sizes)
        expected_dims = [state_dim + action_dim] + hidden_sizes + [1]
        # Each hidden layer adds Linear+ReLU, plus final Linear
        expected_module_count = len(hidden_sizes) * 2 + 1
        assert len(model.net) == expected_module_count
        # Validate dimensions of each Linear layer
        linear_modules = [m for m in model.net if isinstance(m, nn.Linear)]
        dims_pairs = list(zip(expected_dims[:-1], expected_dims[1:]))
        for lm, (in_f, out_f) in zip(linear_modules, dims_pairs):
            assert lm.in_features == in_f
            assert lm.out_features == out_f

    def test_forward_output_shape(self) -> None:
        """Test that forward returns correct output shape."""
        batch_size, state_dim, action_dim = 4, 3, 2
        model = RewardPredictor(state_dim, action_dim)
        s = torch.randn(batch_size, state_dim)
        a = torch.randn(batch_size, action_dim)
        out = model(s, a)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (batch_size,)

    def test_forward_zero_input_zero_weights(self) -> None:
        """Test forward with zero inputs and zeroed weights yields zero output."""
        batch_size, state_dim, action_dim = 5, 2, 2
        model = RewardPredictor(state_dim, action_dim)
        # Zero out all Linear weights and biases
        for m in model.net:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        s = torch.zeros(batch_size, state_dim)
        a = torch.zeros(batch_size, action_dim)
        out = model(s, a)
        assert torch.allclose(out, torch.zeros(batch_size))

    def test_forward_mismatched_batch_raises(self) -> None:
        """Test forward with mismatched batch sizes raises RuntimeError."""
        s = torch.randn(3, 2)
        a = torch.randn(4, 2)
        model = RewardPredictor(2, 2)
        with pytest.raises(RuntimeError):
            _ = model(s, a)
