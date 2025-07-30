"""Pretrain a value network on a gym environment."""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from models import ValueNetwork


def pretrain_value_network(
    env_name: str = "InvertedDoublePendulum-v4",
    hidden_dim: int = 128,
    lr: float = 1e-3,
    gamma: float = 0.99,
    num_episodes: int = 1000,
    normalize_returns: bool = True,
) -> ValueNetwork:
    """Pretrain a value network.

    Fit it to Monte Carlo returns from random rollouts.

    Args:
        env_name (str):           Gym environment ID to use.
        hidden_dim (int):         Number of hidden units in the MLP.
        lr (float):               Learning rate for the Adam optimizer.
        gamma (float):            Discount factor for return calculation.
        num_episodes (int):       How many random episodes to sample.
        normalize_returns (bool): Whether to zero‑mean/unit‑variance normalize returns.

    Returns:
        ValueNetwork: The network after pretraining on random‑policy rollouts.
    """
    # 1) Create env and network
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    value_net = ValueNetwork(state_dim, hidden_dim)
    optimizer = optim.Adam(value_net.parameters(), lr=lr)
    mse = nn.MSELoss()

    for ep in range(1, num_episodes + 1):
        # 2) Rollout with random policy
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            state, _ = reset_out
        else:
            state = reset_out

        # --- inside each episode ---
        states, rewards = [], []
        done = False
        while not done:
            action = env.action_space.sample()
            step_out = env.step(action)

            next_state, reward, terminated, truncated, info = step_out
            done = terminated or truncated

            if isinstance(next_state, tuple):
                next_state, _ = next_state

            states.append(state)
            rewards.append(reward)
            state = next_state

        # 3) Compute Monte Carlo returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        states = torch.tensor(states, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)

        # 4) (Optional) normalize returns to stabilize training
        if normalize_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 5) Fit value network
        preds = value_net(states)
        loss = mse(preds, returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 6) Logging
        if ep % 100 == 0:
            print(f"[Pretrain] Episode {ep}/{num_episodes}  Loss: {loss.item():.4f}")

    env.close()
    return value_net


if __name__ == "__main__":
    # Example: pretrain for 500 episodes
    pretrained_value = pretrain_value_network(
        env_name="InvertedDoublePendulum-v4",
        hidden_dim=256,
        lr=3e-4,
        gamma=0.98,
        num_episodes=500,
    )
    # Save model
    torch.save(
        pretrained_value.state_dict(), "./trained_models/pretrained_value_net.pth"
    )
