"""Pretrain reward network."""

import os
import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import random

import gymnasium as gym
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from models import RewardPredictor

# Save parameters
reward_model_dir = "./trained_models/"  # path to save reward model
reward_model_filename = "reward_network"  # name of saved reward network
os.makedirs(reward_model_dir, exist_ok=True)

# Parameters
n_episodes = 100  # how many rollouts to collect
max_steps = 1000  # max steps per episode

# 1. Create the environment
env = gym.make("InvertedDoublePendulum-v4")

# 2. Set seeds
seed = 0

# seed the Env’s RNG via reset
obs, info = env.reset(seed=seed)
# seed the action & observation spaces (if you sample from them)
env.action_space.seed(seed)
env.observation_space.seed(seed)
# seed other PRNGs you’re using
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# Now `obs` is the initial state, deterministically generated.

# 4. Storage for data
states = []
actions = []
rewards = []

for ep in range(n_episodes):
    obs, info = env.reset(seed=seed + ep)  # <-- unpack here
    for t in range(max_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # append *only* the array
        states.append(obs)
        actions.append(action)
        rewards.append(reward)

        obs = next_obs
        if done:
            break

# convert to arrays
states = np.stack(states, axis=0)  # shape: (total_steps, obs_dim)
actions = np.stack(actions, axis=0)  # shape: (total_steps, act_dim)
rewards = np.array(rewards)  # shape: (total_steps,)


# convert NumPy arrays to PyTorch tensors
tensor_s = torch.from_numpy(states).float()
tensor_a = torch.from_numpy(actions).float()
tensor_r = torch.from_numpy(rewards).float()

dataset = TensorDataset(tensor_s, tensor_a, tensor_r)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
# hyperparameters
lr = 1e-3
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# init model, loss, optimizer
model = RewardPredictor(state_dim=states.shape[1], action_dim=actions.shape[1]).to(
    device
)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_s, batch_a, batch_r in loader:
        batch_s = batch_s.to(device)
        batch_a = batch_a.to(device)
        batch_r = batch_r.to(device)

        # forward
        pred_r = model(batch_s, batch_a)
        loss = criterion(pred_r, batch_r)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_s.size(0)

    epoch_loss /= len(dataset)
    print(f"Epoch {epoch + 1}/{epochs} — MSE: {epoch_loss:.4f}")

# save reward network
save_path = os.path.join(*[reward_model_dir, reward_model_filename + ".pth"])
torch.save(model.state_dict(), save_path)


# Example: run one fresh episode
obs, _ = env.reset(seed=42)
env.action_space.seed(42)
env.observation_space.seed(42)
np.random.seed(42)
torch.manual_seed(42)

total_err = 0.0
count = 0

with torch.no_grad():
    while True:
        # sample an action
        a = env.action_space.sample()

        # step and unpack correctly
        next_obs, true_r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        # now obs is guaranteed to be a numpy array
        s_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        a_tensor = torch.from_numpy(a).float().unsqueeze(0).to(device)

        # predict
        pred_r = model(s_tensor, a_tensor).item()
        print("pred_r =", pred_r)

        total_err += (pred_r - true_r) ** 2
        count += 1

        obs = next_obs  # advance

        if done:
            break

print("Test MSE:", total_err / count)
