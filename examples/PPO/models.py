"""Neural networks for examples/PPO_custom_env/main.py."""

import torch
from torch import nn


class EnvironmentPredictor(nn.Module):
    """Environment network.

    A two-hidden-layer MLP that, given (obs_t, action_t), predicts (obs_{t+1}, reward_t).
    """

    def __init__(
        self, obs_dim: int, action_dim: int, hidden_dims: list[int] = [128, 128]
    ) -> None:
        """Initialize the EnvironmentPredictor.

        Args:
            obs_dim (int): Dimensionality of the observation/state vector.
            action_dim (int): Dimensionality of the action vector.
            hidden_dims (list[int]): Sizes of the hidden layers.
        """
        super().__init__()
        input_dim = obs_dim + action_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )
        self.next_state_head = nn.Linear(hidden_dims[1], obs_dim)
        self.reward_head = nn.Linear(hidden_dims[1], 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict reward for given states and actions.

        Args:
            obs (torch.Tensor): Tensor of states, shape (batch_size, state_dim).
            act (torch.Tensor): Tensor of actions, shape (batch_size, action_dim).

        Returns:
            torch.Tensor: Predicted rewards, shape (batch_size,).
        """
        x = torch.cat([obs, act], dim=-1)
        h = self.net(x)
        next_obs = self.next_state_head(h)
        reward = self.reward_head(h).squeeze(-1)
        return next_obs, reward


class ValuePredictor(nn.Module):
    """Value network."""

    def __init__(self, state_dim: int, hidden_dim: int = 128) -> None:
        """Initialize the ValuePredictor.

        Args:
            state_dim (int): Dimensionality of the state vector.
            hidden_dim (int): Sizes of the hidden layers.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict value for given states.

        Args:
            s (torch.Tensor): Tensor of states, shape (batch_size, state_dim).

        Returns:
            torch.Tensor: Predicted value, shape (batch_size,).
        """
        return self.net(s)


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
        """Forward pass: predict value for given states and actions.

        Args:
            s (torch.Tensor): Tensor of states, shape (batch_size, state_dim).
            a (torch.Tensor): Tensor of actions, shape (batch_size, action_dim).

        Returns:
            torch.Tensor: Predicted rewards, shape (batch_size,).
        """
        x: torch.Tensor = torch.cat([s, a], dim=-1)
        return self.net(x).squeeze(-1)
