"""Neural networks for examples/PPO/main.py."""

import torch
from torch import nn


class RewardPredictor(nn.Module):
    """Reward network."""

    def __init__(
        self, state_dim: int, action_dim: int, hidden_sizes: list[int] = [64, 64]
    ) -> None:
        """Initialize the RewardPredictor.

        Args:
            state_dim (int): Dimensionality of the state vector.
            action_dim (int): Dimensionality of the action vector.
            hidden_sizes (list[int]): Sizes of the hidden layers.
        """
        super().__init__()

        # Build a list of layer sizes: [state+action, *hidden_sizes, 1]
        dims: list[int] = [state_dim + action_dim] + hidden_sizes + [1]

        layers: list[nn.Module] = []
        # Add hidden layers with ReLU activations
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())

        # Final linear layer to produce a single scalar reward
        layers.append(nn.Linear(dims[-2], dims[-1]))

        # Combine into a single sequential model
        self.net: nn.Sequential = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict reward for given states and actions.

        Args:
            s (torch.Tensor): Tensor of states, shape (batch_size, state_dim).
            a (torch.Tensor): Tensor of actions, shape (batch_size, action_dim).

        Returns:
            torch.Tensor: Predicted rewards, shape (batch_size,).
        """
        x: torch.Tensor = torch.cat([s, a], dim=-1)
        return self.net(x).squeeze(-1)
