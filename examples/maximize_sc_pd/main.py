#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a PPO policy (TorchRL) on a custom Gymnasium env that calls a
user-provided simulation function with 10 inputs and returns a single scalar.
The immediate reward equals that scalar.

Requirements (tested with TorchRL 0.9+):
    pip install "torch>=2.3" "torchrl>=0.9" gymnasium numpy tqdm

If you use CUDA:
    pip install "torch==<matching CUDA build>"

References:
- TorchRL PPO tutorial & API (Collector, GAE, ClipPPOLoss, ProbabilisticActor).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

import numpy as np
import torch
from torch import nn

import gymnasium as gym
from gymnasium import spaces

# TensorDict / TorchRL core
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from tensordict import TensorDict

from torchrl.envs import GymWrapper, TransformedEnv, DoubleToFloat
from torchrl.envs.utils import (
    ExplorationType,
    set_exploration_type,
    check_env_specs,
)

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE


# -----------------------------------------------
# 2) Custom Gymnasium Env wrapping the simulation
# -----------------------------------------------

class Sim10Env(gym.Env):
    """
    A single-step bandit-like env:
      - action: R^10 (bounded) --> fed to simulation
      - observation: a 1-D dummy array (zeros) by default
      - reward: simulation output (scalar)
      - episode ends every step (terminated=True)

    If you want longer episodes, set max_episode_steps>1 and define your own
    termination logic. PPO with GAE will still work.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        sim_fn: Callable[[np.ndarray], float],
        action_low: float | np.ndarray = -1.0,
        action_high: float | np.ndarray = 1.0,
        max_episode_steps: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.sim_fn = sim_fn

        # Bounds for the 10D action (broadcast scalars if needed)
        low = np.full((10,), action_low, dtype=np.float32) if np.isscalar(action_low) else np.asarray(action_low, dtype=np.float32)
        high = np.full((10,), action_high, dtype=np.float32) if np.isscalar(action_high) else np.asarray(action_high, dtype=np.float32)
        assert low.shape == (10,) and high.shape == (10,), "action_low/high must be length-10 or scalar."
        assert np.all(high > low), "Each action_high must be > action_low."

        self.action_space = spaces.Box(low=low, high=high, shape=(10,), dtype=np.float32)

        # Minimal observation; PPO needs something to read.
        # Using a constant 1-D observation is fine for a stateless bandit.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        self.max_episode_steps = max_episode_steps
        self._step_count = 0
        self._last_sim_out = 0.0

        # Seeding
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._step_count = 0
        self._last_sim_out = 0.0
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        self._step_count += 1

        # Enforce bounds to be safe, even though the policy is bounded.
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        sim_out = float(self.sim_fn(action))  # scalar
        if not np.isfinite(sim_out):
            raise RuntimeError(f"sim_fn returned non-finite {sim_out} for action {action}")
        self._last_sim_out = sim_out

        reward = sim_out
        # Single-step termination by default (bandit); change if you want longer episodes
        terminated = True if self._step_count >= self.max_episode_steps else False
        truncated = False

        # Observation can remain a dummy vector; you can put diagnostics here if you like.
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {"sim_output": sim_out}

        return obs, reward, terminated, truncated, info

    def render(self):
        # No rendering; place holder if you want to visualize.
        pass


# ----------------------------
# 3) PPO Training configuration
# ----------------------------

@dataclass
class PPOConfig:
    # Environment / simulation
    action_low: float = -1.0
    action_high: float = 1.0
    max_episode_steps: int = 1  # 1 => stateless bandit; increase if needed
    seed: int = 0

    # Network
    hidden_units: int = 256
    device: str = "cuda" if (torch.cuda.is_available()) else "cpu"

    # Data collection
    frames_per_batch: int = 2048     # increase for more stable gradient estimates
    total_frames: int = 100_000      # overall training interactions

    # PPO optimization
    ppo_epochs: int = 10
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 1e-4
    lr: float = 3e-4
    max_grad_norm: float = 1.0

    # Saving/eval
    save_path: str = "ppo_sim_actor.pt"
    eval_every_n_batches: int = 10


# ----------------------------
# 4) Build policy / value nets
# ----------------------------

def build_actor_critic(env: TransformedEnv, cfg: PPOConfig):
    """
    Builds:
      - ProbabilisticActor with TanhNormal bounded by env.action_spec
      - ValueOperator for state value V(s)
    """
    device = torch.device(cfg.device)
    action_dim = int(np.prod(env.action_spec.shape))

    # Policy backbone -> (loc, scale) for a Normal
    policy_backbone = nn.Sequential(
        nn.Linear(1,            cfg.hidden_units),
        nn.ReLU(),
        nn.Linear(cfg.hidden_units,   cfg.hidden_units),
        nn.ReLU(),
        nn.Linear(cfg.hidden_units,   cfg.hidden_units),
        nn.ReLU(),
        nn.Linear(cfg.hidden_units,   2 * action_dim),  # e.g., mean + log_std
    ).to(device)
    policy_head = NormalParamExtractor()  # splits to ("loc", "scale")

    policy_param_module = TensorDictModule(
        module=nn.Sequential(policy_backbone, policy_head),
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )

    # Bounded stochastic actor: TanhNormal with env action bounds
    # safe=True + spec ensures outputs respect bounds even in edge cases.
    actor = ProbabilisticActor(
        module=policy_param_module,
        in_keys=["loc", "scale"],
        spec=env.action_spec,
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec.low,
            "high": env.action_spec.high,
        },
        return_log_prob=True,
        safe=True,
    ).to(device)

    # Value network V(s)
    value_net = nn.Sequential(
        nn.Linear(1,          cfg.hidden_units),
        nn.ReLU(),
        nn.Linear(cfg.hidden_units, cfg.hidden_units),
        nn.ReLU(),
        nn.Linear(cfg.hidden_units, cfg.hidden_units),
        nn.ReLU(),
        nn.Linear(cfg.hidden_units, 1),   # V(s) scalar
    ).to(device)    

    
    value = ValueOperator(module=value_net, in_keys=["observation"]).to(device)

    return actor, value


# ----------------------------
# 5) Make TorchRL environment
# ----------------------------

def make_env(sim_fn: Callable[[np.ndarray], float], cfg: PPOConfig) -> TransformedEnv:
    """
    Wrap our Gymnasium env with TorchRL's GymWrapper and a simple float transform.
    """
    base = Sim10Env(
        sim_fn=sim_fn,
        action_low=cfg.action_low,
        action_high=cfg.action_high,
        max_episode_steps=cfg.max_episode_steps,
        seed=cfg.seed,
    )
    env = TransformedEnv(
        GymWrapper(base, device=cfg.device),
        DoubleToFloat(),  # keep everything as float32
    )
    check_env_specs(env)  # sanity check
    return env


# ----------------------------
# 6) Training loop (PPO)
# ----------------------------

def train(sim_fn: Callable[[np.ndarray], float], cfg: PPOConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)
    env = make_env(sim_fn, cfg)

    actor, value = build_actor_critic(env, cfg)

    # Collector: batches trajectories on-policy
    collector = SyncDataCollector(
        lambda: make_env(sim_fn, cfg),   # <- factory that returns a new EnvBase
        policy=actor,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        device=device,
        exploration_type=ExplorationType.RANDOM,
    )

    # Buffer to shuffle data into PPO minibatches
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=cfg.frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    # Advantage (GAE) + PPO losses
    advantage = GAE(
        gamma=cfg.gamma, lmbda=cfg.gae_lambda, value_network=value, average_gae=True
    )
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=value,
        clip_epsilon=cfg.clip_eps,
        entropy_bonus=bool(cfg.entropy_coef),
        entropy_coef=cfg.entropy_coef,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    ).to(device)

    optim = torch.optim.Adam(loss_module.parameters(), lr=cfg.lr)

    print(f"[PPO] device={device}, frames_per_batch={cfg.frames_per_batch}, total_frames={cfg.total_frames}")

    running_reward = None
    batches = 0

    for tensordict in collector:
        batches += 1
        # Compute advantages (adds "advantage" and "value_target")
        advantage(tensordict)

        # Flatten batch of transitions
        data_view = tensordict.reshape(-1)
        replay_buffer.extend(data_view.cpu())

        # PPO inner loop over shuffled minibatches
        num_minibatches = math.ceil(cfg.frames_per_batch / cfg.minibatch_size)
        for _ in range(cfg.ppo_epochs):
            for _ in range(num_minibatches):
                batch = replay_buffer.sample(cfg.minibatch_size).to(device)
                loss_dict = loss_module(batch)
                loss = (
                    loss_dict["loss_objective"]
                    + loss_dict["loss_critic"]
                    + loss_dict["loss_entropy"]
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.max_grad_norm)
                optim.step()
                optim.zero_grad()

        # Simple training log
        batch_reward = tensordict["next", "reward"].mean().item()
        running_reward = (
            batch_reward if running_reward is None else 0.95 * running_reward + 0.05 * batch_reward
        )
        print(f"Batch {batches:4d} | avg reward (batch): {batch_reward:+.5f} | ema: {running_reward:+.5f}")

        # Optional quick evaluation (deterministic)
        if (batches % cfg.eval_every_n_batches) == 0:
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                actor.eval()
                td_eval = env.rollout(
                    policy=actor,
                    max_steps=cfg.max_episode_steps,
                    auto_reset=True,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                ret = td_eval.get(("next", "reward")).sum().item()
                print(f"  [eval] return over {cfg.max_episode_steps} step(s): {ret:+.5f}")
                actor.train()

    # Save trained actor weights
    torch.save(actor.state_dict(), cfg.save_path)
    print(f"Saved actor to: {cfg.save_path}")


# ----------------------------
# 7) Run
# ----------------------------

if __name__ == "__main__":
    cfg = PPOConfig()
    
    n_points_liner = 10
    
    alpha0 = 60
    z_liner_offset0 = 1.5
    r_shell0 = np.linspace(0.0, 0.85, n_points_liner)
    x0 = r_shell0*np.tan((90-alpha0)*(np.pi/180)) + z_liner_offset0
    
    # If your simulator needs different action bounds, do:
    cfg.action_low = x0 - 0.5 * x0
    cfg.action_high = x0 + 0.108
    
    cfg.device = "cpu"
    cfg.seed = 42
    cfg.gamma = 0.99
    cfg.gae_lambda = 0.95
    
    cfg.frames_per_batch = 2048 # 1
    cfg.minibatch_size = 256
    cfg.ppo_epochs = 10
    cfg.max_grad_norm = 1.0
    cfg.eval_every_n_batches = 10
    cfg.max_episode_steps = 1
    
    import torch, numpy as np
    from typing import Callable

    class MLP(torch.nn.Module):
        def __init__(self, in_dim, hidden_dims=(32,32), dropout=0.1, out_dim=1):
            super().__init__()
            layers, last = [], in_dim
            for h in hidden_dims:
                layers += [torch.nn.Linear(last, h), torch.nn.ReLU()]
                if dropout > 0:
                    layers += [torch.nn.Dropout(p=dropout)]
                last = h
            layers += [torch.nn.Linear(last, out_dim)]
            self.net = torch.nn.Sequential(*layers)
        def forward(self, x): return self.net(x)

    def load_surrogate(ckpt_path: str, device: str | None = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        meta = ckpt["meta"]

        model = MLP(
            in_dim=meta["input_dim"],
            hidden_dims=tuple(meta["hidden_dims"]),
            dropout=meta["dropout"],
            out_dim=meta["output_dim"],
        ).to(device).eval()
        
        with torch.no_grad():
            _ = model(torch.zeros(1, meta["input_dim"], dtype=torch.float32, device=device))


        state = ckpt["model_state_dict"]
        if any(k.startswith("module.") for k in state):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        return model, meta, device

    def make_sim_fn_from_ckpt(ckpt_path: str, device: str | None = None) -> Callable[[np.ndarray], float]:
        model, meta, device = load_surrogate(ckpt_path, device)

        x_mean = torch.from_numpy(np.asarray(meta["x_mean"], dtype=np.float32)).to(device)
        x_std  = torch.from_numpy(np.asarray(meta["x_std"],  dtype=np.float32)).to(device)
        norm_tgt = bool(meta.get("normalize_target", False))
        if norm_tgt:
            y_mean = torch.from_numpy(np.asarray(meta["y_mean"], dtype=np.float32)).to(device)
            y_std  = torch.from_numpy(np.asarray(meta["y_std"],  dtype=np.float32)).to(device)

        eps = torch.finfo(torch.float32).eps

        @torch.inference_mode()
        def sim_fn(x: np.ndarray) -> float:
            x = np.asarray(x, dtype=np.float32)
            if x.ndim != 1:
                raise ValueError(f"sim_fn expects a 1D feature vector; got shape {x.shape}")
            if x.shape[0] != meta["input_dim"]:
                raise ValueError(f"Expected input_dim={meta['input_dim']}, got {x.shape[0]}")

            xt = torch.from_numpy(x).unsqueeze(0).to(device)              # (1, D)
            xs = (xt - x_mean) / torch.clamp(x_std, min=eps)              # avoid /0
            xs = torch.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0)    # sanitize

            y_scaled = model(xs)
            y = (y_scaled * y_std + y_mean) if norm_tgt else y_scaled
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)      # sanitize

            return float(y.squeeze().item())

        return sim_fn

    sim_fn = make_sim_fn_from_ckpt("./train_emulator/surrogate_out/surrogate.pt", device="cpu")
    
    run_train = True
    if run_train:
        train(sim_fn, cfg)
    else:
        state = torch.load(cfg.save_path, map_location=cfg.device)
    
    # Rebuild env & actor exactly as in training, then load weights
    env = make_env(sim_fn, cfg)
    actor, _ = build_actor_critic(env, cfg)
    actor.load_state_dict(state)
    actor.to(cfg.device).eval()

    # Deterministic (noise-free) action from the TanhNormal policy
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        td = TensorDict({'observation': torch.zeros(1, 1, device=cfg.device)}, batch_size=[1])
        td = actor(td)  # adds "action"
        a_star = td['action'][0].cpu().numpy()  # shape (10,)

    r_star = float(sim_fn(a_star))
    print("Policy's deterministic action:\n", a_star)
    print("Simulator reward for that action:", r_star)

