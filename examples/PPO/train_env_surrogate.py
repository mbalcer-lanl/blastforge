"""Train a surrogate model to emulate a gym environment."""

import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import EnvironmentPredictor


def collect_data(
    env: gym.Env, num_episodes: int, max_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Roll out a random policy to collect (obs, action, next_obs, reward) tuples.

    Args:
        env (gym.Env):
            The Gym environment to sample from.
        num_episodes (int):
            Number of episodes to collect.
        max_steps (int):
            Maximum steps per episode before truncating.

    Returns:
        obs_arr (np.ndarray):
            Array of observations, shape (N, obs_dim).
        act_arr (np.ndarray):
            Array of actions taken, shape (N, action_dim).
        next_obs_arr (np.ndarray):
            Array of next observations, shape (N, obs_dim).
        rew_arr (np.ndarray):
            Array of rewards, shape (N,).
    """
    buffer = []
    for ep in range(num_episodes):
        # Gym >=0.26: reset() returns (obs, info)
        obs, _ = env.reset()
        for t in range(max_steps):
            act = env.action_space.sample()
            # step() returns (obs, reward, terminated, truncated, info)
            next_obs, reward, terminated, truncated, _ = env.step(act)
            done = terminated or truncated

            buffer.append((obs, act, next_obs, reward))
            obs = next_obs
            if done:
                break

    obs_arr, act_arr, next_obs_arr, rew_arr = zip(*buffer)
    return (
        np.array(obs_arr, dtype=np.float32),  # shape (N, obs_dim)
        np.array(act_arr, dtype=np.float32),  # shape (N, action_dim)
        np.array(next_obs_arr, dtype=np.float32),
        np.array(rew_arr, dtype=np.float32),
    )


def train_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    dataloader: torch.utils.data.DataLoader[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    epochs: int,
    device: torch.device,
) -> None:
    """Train the dynamics model using MSE loss on both next-state and reward predictions.

    Args:
        model (nn.Module):
            A PyTorch module which, given (obs, action) returns
            (predicted_next_obs, predicted_reward).
        optimizer (optim.Optimizer):
            The optimizer (e.g., Adam) used to update model parameters.
        dataloader (DataLoader[obs, act, next_obs, reward]):
            Yields batches of (obs_b, act_b, next_obs_b, rew_b).
        epochs (int):
            Number of full passes over the dataloader.
        device (torch.device):
            Device on which to perform training (CPU or GPU).
    """
    criterion = nn.MSELoss()
    model.to(device)
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for obs_b, act_b, next_obs_b, rew_b in dataloader:
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)
            next_obs_b = next_obs_b.to(device)
            rew_b = rew_b.to(device)

            pred_next, pred_rew = model(obs_b, act_b)
            loss_ns = criterion(pred_next, next_obs_b)
            loss_rw = criterion(pred_rew, rew_b)
            loss = loss_ns + loss_rw

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * obs_b.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch:3d}/{epochs:3d} — Loss: {avg_loss:.6f}")


"""Main training loop."""

parser = argparse.ArgumentParser(
    description="Train a dynamics model for InvertedDoublePendulum-v4"
)
parser.add_argument(
    "--episodes", type=int, default=100, help="Number of rollouts for data collection"
)
parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per rollout")
parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument(
    "--save_path",
    type=str,
    default="trained_models/env_network.pth",
    help="Path to save trained model",
)
parser.add_argument(
    "--no-cuda", action="store_true", help="Disable CUDA even if available"
)
args = parser.parse_args()

device = torch.device(
    "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
)
print(f"Using device: {device}")

env = gym.make("InvertedDoublePendulum-v4")
obs_dim = env.observation_space.shape[0]  # 11
action_dim = env.action_space.shape[0]  # 1
print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")

# 1) Data collection
print("Collecting data with random policy...")
obs_np, act_np, next_obs_np, rew_np = collect_data(env, args.episodes, args.max_steps)
print(f"Collected {obs_np.shape[0]} transitions.")

# 2) Build dataset & loader
dataset = TensorDataset(
    torch.from_numpy(obs_np),
    torch.from_numpy(act_np),
    torch.from_numpy(next_obs_np),
    torch.from_numpy(rew_np),
)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# 3) Instantiate model & optimizer
model = EnvironmentPredictor(obs_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# 4) Train
print("Starting training...")
train_model(model, optimizer, loader, args.epochs, device)

# 5) Save
torch.save(model.state_dict(), args.save_path)
print(f"Model parameters saved to '{args.save_path}'")
