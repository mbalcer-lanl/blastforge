#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a PPO policy (TorchRL) on a custom Gymnasium env that calls a
user-provided neural network.
The immediate reward is the -MSE(target, emulator output density field).

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

from models import tCNNsurrogate, hybrid2vectorCNN
from utils import mse_2d, load_model_and_optimizer_hdf5

import matplotlib.pyplot as plt
# -----------------------------------------------
# 2) Custom Gymnasium Env wrapping the simulation
# -----------------------------------------------

class SimEnv(gym.Env):
    """
    A single-step bandit-like env:
      - action: R^10 (bounded) --> fed to simulation
      - observation: a 1-D dummy array (zeros) by default
      - reward: simulation output (scalar)
      - episode ends every step (terminated=True)

    If you want longer episodes, set max_steps>1 and define your own
    termination logic. PPO with GAE will still work.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        sim_fn,                              # callable: np.ndarray (D,) -> np.ndarray / torch.Tensor
        action_low, action_high,             # arrays for Box bounds
        output_hw=(1120, 800),               # (H, W)
        channels=1,                          # set to 3 for RGB, etc.
        channels_first=True,                 # True: (C,H,W), False: (H,W,C)
        max_steps=200,
        target = None,
        seed = 42
        # reward_fn=None                       # callable: obs -> float
    ):
        super().__init__()
        self.sim_fn = sim_fn
        self.H, self.W = output_hw
        self.C = channels
        self.channels_first = channels_first
        self.max_steps = max_steps
        self.step_count = 0

        # Spaces
        self.action_space = gym.spaces.Box(
            low=np.asarray(action_low, dtype=np.float32),
            high=np.asarray(action_high, dtype=np.float32),
            dtype=np.float32
        )
        self.y_dim = int(np.prod(self.action_space.shape))
        
        # Observation shapes
        img_shape = (self.C, self.H, self.W) if channels_first else (self.H, self.W, self.C)

        # ---- target image -> (C,H,W) float32
        self.target = np.asarray(target, dtype=np.float32) if target is not None else np.zeros((self.H, self.W), np.float32)
        if self.target.ndim == 2:                       # (H,W)
            self.target = self.target[None, ...]        # (1,H,W)
        elif self.target.ndim == 3 and not channels_first:  # (H,W,C) -> (C,H,W)
            self.target = np.moveaxis(self.target, -1, 0)
        assert self.target.shape == img_shape, f"target shape {self.target.shape} != expected {img_shape}"

        # Observation space is a Dict[y, h1, h2]
        self.observation_space = spaces.Dict({
            "y":  spaces.Box(low=-np.inf, high=np.inf, shape=(self.y_dim,), dtype=np.float32),
            "h1": spaces.Box(low=-np.inf, high=np.inf, shape=img_shape, dtype=np.float32),
            "h2": spaces.Box(low=-np.inf, high=np.inf, shape=img_shape, dtype=np.float32),
        })
        
    def _coerce_img(self, sim_out):
        # -> (C,H,W) float32; same as your _coerce_obs but only for images
        if torch.is_tensor(sim_out):
            sim_out = sim_out.detach().cpu().numpy()
        img = np.asarray(sim_out, dtype=np.float32)
        expected = (self.C, self.H, self.W) if self.channels_first else (self.H, self.W, self.C)

        if img.ndim == 1:
            if img.size != np.prod(expected): raise ValueError(f"Flat output size {img.size} cannot reshape to {expected}")
            img = img.reshape(expected)
        elif img.ndim == 2:
            if self.C != 1 or img.shape != (self.H, self.W): raise ValueError(f"Got 2D {img.shape}, expected {(self.H,self.W)}")
            img = img[None, ...] if self.channels_first else img[..., None]
        elif img.ndim == 3:
            if self.channels_first:
                if img.shape == (self.H, self.W, self.C): img = np.moveaxis(img, -1, 0)
                elif img.shape != (self.C, self.H, self.W): raise ValueError(f"Unexpected {img.shape}")
            else:
                if img.shape == (self.C, self.H, self.W): img = np.moveaxis(img, 0, -1)
                elif img.shape != (self.H, self.W, self.C): raise ValueError(f"Unexpected {img.shape}")
        else:
            raise ValueError(f"Expected 1D/2D/3D image; got {img.ndim}D")

        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        return img
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        print('Resetting environment ...')
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.step_count = 0
        # At t=0 the policy will see (y=0, h1=0 image, h2=target)
        y0  = np.zeros((self.y_dim,), dtype=np.float32)
        h10 = np.zeros((self.C, self.H, self.W), dtype=np.float32)
        obs = {"y": y0, "h1": h10, "h2": self.target.copy()}
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        print('Conducting environment step ...')
        self.step_count += 1

        # Enforce bounds to be safe, even though the policy is bounded.
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 1) Run the simulator/model → image-like output (no scalar cast!)
        sim_out = self.sim_fn(action)              # model → image
        img = self._coerce_img(sim_out)            # (C,H,W) float32
        
        ## fig, ax = plt.subplots()
        ## im = ax.imshow(sim_out[0,:,:], origin="lower", extent=None)  # default colormap
        ## cbar = fig.colorbar(im, ax=ax)
        ## cbar.set_label("Value")
        ## plt.show()
        
        # Normalize to a numpy array and check finiteness
        if torch.is_tensor(sim_out):
            finite = torch.isfinite(sim_out).all().item()
            sim_out = sim_out.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            sim_out = np.asarray(sim_out, dtype=np.float32)
            finite = np.isfinite(sim_out).all()

        if not finite:
            # Optional: count what's wrong for easier debugging
            n_nan = int(np.isnan(sim_out).sum())
            n_inf = int(np.isinf(sim_out).sum())
            raise RuntimeError(
                f"sim_fn returned non-finite values (nan={n_nan}, inf={n_inf}) "
                f"for action {action}"
            )

        # 2) Convert to an observation that matches observation_space
        obs = {
            "y":  action.astype(np.float32, copy=False).reshape(self.y_dim),
            "h1": img,
            "h2": self.target.copy(),
        }
        # print('obs.ndim =', obs.ndim)
        # print('obs.shape =', obs.shape) # (1, 1120, 800)
        
        self._last_sim_out = sim_out
        
        ## reward = sim_out
        
        reward = -1*mse_2d(self.target[0,:,:], img[0,:,:])
        print('reward =', reward)
        # Single-step termination by default (bandit); change if you want longer episodes
        ## terminated = True if self.step_count >= self.max_steps else False
        ## truncated = False
        
        terminated = False
        truncated = self.step_count >= self.max_steps  # time-limit


        # Observation can remain a dummy vector; you can put diagnostics here if you like.
        ## obs = np.zeros(self.observation_space.shape, dtype=np.float32)
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
    max_steps: int = 1  # 1 => stateless bandit; increase if needed
    seed: int = 0
    nvar: int = 28

    # Network
    hidden_units: int = 256
    device: str = "cuda" if (torch.cuda.is_available()) else "cpu"

    # Data collection
    frames_per_batch: int = 64     # increase for more stable gradient estimates
    total_frames: int = 1000      # overall training interactions

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
    
    # Target
    target: float = 1.0


# ----------------------------
# 4) Build policy / value nets
# ----------------------------

def build_actor_critic(env: TransformedEnv, cfg: PPOConfig):
    """
    Builds:
      - ProbabilisticActor with TanhNormal bounded by env.action_spec
      - ValueOperator for state value V(s)
    """
    print('Building actor and critic networks ...')
    device = torch.device(cfg.device)
    action_dim = int(np.prod(env.action_spec.shape))
    
    
    ## class CNNBackbone(nn.Module):
    ##     def __init__(self, c, h, w, target_hw=(112, 80), hidden=256):
    ##         super().__init__()
    ##         self.resize = nn.AdaptiveAvgPool2d(target_hw)  # shrink any HxW to fixed H'xW'
    ##         self.fe = nn.Sequential(
    ##             nn.Flatten(start_dim=-3),
    ##             nn.Conv2d(c, 16, kernel_size=8, stride=4), nn.ReLU(),
    ##             nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
    ##             nn.Conv2d(32, 64, kernel_size=3, stride=1), nn.ReLU(),
    ##             nn.Flatten(start_dim=1)  # flattens C,H,W together
    ##         )
    ##         with torch.no_grad():
    ##             dummy = torch.zeros(1, c, h, w)
    ##             feat_dim = self.fe(self.resize(dummy)).shape[-1]  # <-- matches runtime
    ##         self.head = nn.Sequential(
    ##             nn.Linear(feat_dim, hidden), nn.ReLU(),
    ##         )
## 
    ##     def forward(self, x):
    ##         x = self.resize(x)   # ensure fixed resolution
    ##         x = self.fe(x)       # now shape (B, feat_dim)
    ##         return self.head(x)
    
    # Policy backbone -> (loc, scale) for a Normal
    ## policy_backbone = nn.Sequential(
    ##     nn.Flatten(start_dim=-3),
    ##     nn.Linear(28,            cfg.hidden_units),
    ##     nn.ReLU(),
    ##     nn.Linear(cfg.hidden_units,   cfg.hidden_units),
    ##     nn.ReLU(),
    ##     nn.Linear(cfg.hidden_units,   cfg.hidden_units),
    ##     nn.ReLU(),
    ##     nn.Linear(cfg.hidden_units,   2 * action_dim),  # e.g., mean + log_std
    ## ).to(device)
    ## policy_head = NormalParamExtractor()  # splits to ("loc", "scale")
    
    ## target_hw=(112, 80)
    ## C, H, W = env.observation_space.shape      # channels-first
    ## act_dim = env.action_space.shape[0]         # e.g., 28
    ## backbone = CNNBackbone(C, H, W, target_hw=target_hw, hidden=cfg.hidden_units)
    ## net = nn.Sequential(
    ##     backbone,
    ##     nn.Linear(cfg.hidden_units, 2 * act_dim),
    ##     NormalParamExtractor(),                 # -> loc, scale
    ## )
    
    
    class ImageMLPBackbone(nn.Module):
        def __init__(self, hidden=256):
            super().__init__()
            # Flatten ONLY the last 3 dims (C,H,W), keep any leading batch/time dims intact
            self.flatten_chw = nn.Flatten(start_dim=-3)
            self.net = nn.Sequential(
                self.flatten_chw,
                nn.LazyLinear(hidden),  # <-- infers in_features on first forward (ignores batch dims)
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.net(x)
    
    hidden = cfg.hidden_units
    backbone = ImageMLPBackbone(hidden=hidden)
    head = nn.Sequential(
        backbone,
        nn.Linear(hidden, 2 * action_dim),
        NormalParamExtractor(),  # -> loc, scale
    )
    
    policy_param_module = TensorDictModule(
        module=head, # nn.Sequential(policy_backbone, policy_head), # net,# 
        in_keys=["h1"],
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
    ## value_net = nn.Sequential(
    ##     nn.Linear(1,          cfg.hidden_units),
    ##     nn.ReLU(),
    ##     nn.Linear(cfg.hidden_units, cfg.hidden_units),
    ##     nn.ReLU(),
    ##     nn.Linear(cfg.hidden_units, cfg.hidden_units),
    ##     nn.ReLU(),
    ##     nn.Linear(cfg.hidden_units, 1),   # V(s) scalar
    ## ).to(device)
    
    value_model_args = {
        "img_size": (1, 1120, 800),
        "input_vector_size": 28,
        "output_dim": 1,
        "features": 12,
        "depth": 4,
        "kernel": 3,
        "img_embed_dim": 32,
        "vector_embed_dim": 32,
        "size_reduce_threshold": (16, 16),
        "vector_feature_list": (4, 4, 4, 4),
        "output_feature_list": (4, 4, 4, 4),
        "act_layer": nn.GELU,
        "norm_layer": nn.LayerNorm
        }
    value_net = hybrid2vectorCNN(**value_model_args).to(device)
    
    def disable_bn_running_stats_(module: nn.Module):
        for m in module.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
                # keep m.training = True globally; with track_running_stats=False
                # BN uses batch stats in both train/eval

    # After you build the nets:
    # disable_bn_running_stats_(actor)      # if your policy has BN
    disable_bn_running_stats_(value_net)  # your hybrid2vectorCNN has BN in its CNN blocks
    
    value = ValueOperator(
        module=value_net,
        in_keys=["y", "h1", "h2"],      # <— three inputs
        out_keys=["state_value"],       # optional; default is "state_value"
    ).to(device)
    
    return actor, value


# ----------------------------
# 5) Make TorchRL environment
# ----------------------------

def make_env(sim_fn: Callable[[np.ndarray], float], cfg: PPOConfig) -> TransformedEnv:
    """
    Wrap our Gymnasium env with TorchRL's GymWrapper and a simple float transform.
    """
    print('Making environment ...')
    base = SimEnv(
        sim_fn=sim_fn,
        action_low=cfg.action_low,
        action_high=cfg.action_high,
        max_steps=cfg.max_steps,
        # nvar=cfg.nvar,
        target=cfg.target,
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
    
    # --- metrics storage ---
    logs = {
        "epoch_actor": [],     # per inner epoch
        "epoch_critic": [],
        "epoch_entropy": [],
        "epoch_total": [],
        "batch_idx": [],       # outer batch index (collector iteration)
        "batch_reward": [],    # mean reward of the outer batch (already computed)
        "eval_return": [],    # return of predicted action of current batch
        "best_action": [],        # (num_evals, action_dim)
    }
    
    print('Starting training ...')
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
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optim, total_frames // frames_per_batch, 0.0
    # )
    print(f"[PPO] device={device}, frames_per_batch={cfg.frames_per_batch}, total_frames={cfg.total_frames}")

    running_reward = None
    batches = 0

    for tensordict in collector:
        batches += 1
        print('Batch #', batches)
        # Compute advantages (adds "advantage" and "value_target")
        print('Computing advantage ...')
        advantage(tensordict)
        
        adv = tensordict["advantage"]              # tensor on your device
        print("adv shape:", tuple(adv.shape))
        print("adv mean:", adv.mean().item())
        print("adv std: ", adv.std().item())
        print("adv min/max:", adv.min().item(), adv.max().item())
        
        # Flatten batch of transitions
        data_view = tensordict.reshape(-1)
        replay_buffer.extend(data_view.cpu())

        # PPO inner loop over shuffled minibatches
        num_minibatches = math.ceil(cfg.frames_per_batch / cfg.minibatch_size)
        print('num_minibatches =', num_minibatches)
        for i in range(cfg.ppo_epochs):
            print('epoch =', i+1)
            
            epoch_actor = 0.0
            epoch_critic = 0.0
            epoch_entropy = 0.0
            
            for k in range(num_minibatches):
                print('minibatch =', k+1)
                print('Sampling minibatch from replay buffer ...')
                batch = replay_buffer.sample(cfg.minibatch_size).to(device)
                print('Computing loss ...')
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
                
                # detach to Python floats
                epoch_actor   += float(loss_dict["loss_objective"].item())
                epoch_critic  += float(loss_dict["loss_critic"].item())
                epoch_entropy += float(loss_dict["loss_entropy"].item())

            # mean over minibatches = "epoch" statistic
            epoch_actor   /= num_minibatches
            epoch_critic  /= num_minibatches
            epoch_entropy /= num_minibatches
            logs["epoch_actor"].append(epoch_actor)
            logs["epoch_critic"].append(epoch_critic)
            logs["epoch_entropy"].append(epoch_entropy)
            logs["epoch_total"].append(epoch_actor + epoch_critic + epoch_entropy)
            print('total loss epoch i =', epoch_actor + epoch_critic + epoch_entropy)
            logs["batch_idx"].append(batches)

        # Simple training log
        batch_reward = tensordict["next", "reward"].mean().item()
        logs["batch_reward"].append(batch_reward)
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
                    max_steps=cfg.max_steps,
                    auto_reset=True,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                # Action tensor has leading time (and maybe env) dims.
                a = td_eval.get("action")             # shape [T, N_env, A] or [T, A]
                a_last = a[-1]                        # last timestep
                if a_last.ndim == 2:                  # [N_env, A]
                    a_star = a_last[0]                # take env 0 (or mean over envs)
                else:                                 # [A]
                    a_star = a_last
                best_action = a_star.detach().cpu().numpy()
                logs["best_action"].append(best_action)

                ret = td_eval.get(("next", "reward")).sum().item()
                logs["eval_return"].append(ret)
                print(f"  [eval] deterministic action (first 5): {best_action[:5]!r} | return: {ret:+.5f}")
                actor.train()
                
                
                
                
                # 1) Reset and set the action
                td = env.reset().to(cfg.device)
                a = torch.as_tensor(a_star, dtype=torch.float32, device=cfg.device).unsqueeze(0)  # (1, A)
                td.set("action", a)

                # 2) Step once (bandit: one step is enough)
                td_next = env.step(td)

                # 3) Pull the image out of the tensordict
                # If your observation is a Dict with keys y/h1/h2 (as we wired earlier):
                if ("next", "h1") in td_next.keys(True):
                    img_t = td_next.get(("next", "h1"))[0]       # torch, shape (C,H,W)
                # If your observation is a single Box image:
                else:
                    img_t = td_next.get(("next", "observation"))[0]  # torch, shape (C,H,W)

                # Optionally, you can also fetch the raw sim output from Gym info:
                # (GymWrapper puts info under "_extra")
                if ("next","_extra","sim_output") in td_next.keys(True):
                    img_raw = td_next.get(("next","_extra","sim_output"))[0]  # same content
                
                # plot result from best action at current batch
                fig, ax = plt.subplots()
                img = img_t.detach().cpu().numpy()
                to_show = img[0] if img.ndim == 3 else img
                im = ax.imshow(to_show, origin="lower", extent=None)  # default colormap
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label("Density")
                plt.savefig('./figures/simulated_best_output_batch{0}.png'.format(batches), dpi=150, transparent=True)
                
                # plt.show()
                
        
        # Save trained actor weights
        save_path_i = "ppo_sim_actor_batch{0}.pt".format(batches)
        torch.save(actor.state_dict(), save_path_i)
        print(f"Saved actor to: {save_path_i}")
        
    # Save trained actor weights
    torch.save(actor.state_dict(), cfg.save_path)
    print(f"Saved actor to: {cfg.save_path}")
    
    # Optional: save raw logs for later
    np.savez("training_logs.npz", **logs)
    
    return logs

# ----------------------------
# 7) Run
# ----------------------------

if __name__ == "__main__":
    
    run_train = True
    
    cfg = PPOConfig()
    
    rv_mean = np.array([
       4.5, # liner position
       5.0,
       5.5,
       7.0,
       8.5,
       10.0,
       10.5,
       0.5, # thicknesses of Cu
       0.25,
       0.25,
       0.25,
       0.25,
       0.3,
       0.3,
       0.4, # thicknesses of Al
       0.35,
       0.25,
       0.25,
       0.2,
       0.1,
       0.1,
       0.1, # thicknesses of Sy
       0.1,
       0.1,
       0.1,
       0.15,
       0.15,
       0.2,
       # 25.0, # radius of shell
       ])
    
    rv_lb = np.array([
       4.5-2, # liner position
       5.0-2,
       5.5-2,
       7.0-2,
       8.5-2,
       10.0-2,
       10.5-2,
       0, # thicknesses of Cu
       0,
       0,
       0,
       0,
       0,
       0,
       0, # thicknesses of Al
       0,
       0,
       0,
       0,
       0,
       0,
       0, # thicknesses of Sy
       0,
       0,
       0,
       0,
       0,
       0,
       # 25.0, # radius of shell
       ])
    
    rv_ub = np.array([
       4.5+2, # liner position
       5.0+2,
       5.5+2,
       7.0+2,
       8.5+2,
       10.0+2,
       10.5+2,
       1, # thicknesses of Cu
       1,
       1,
       1,
       1,
       1,
       1,
       1, # thicknesses of Al
       1,
       1,
       1,
       1,
       1,
       1,
       1, # thicknesses of Sy
       1,
       1,
       1,
       1,
       1,
       1,
       # 25.0, # radius of shell
       ])
    
    # define target
    # cfg.target = np.load('./data/target/target1.npy')
    cfg.target = np.zeros((1120, 800))
    cfg.target[500:700, 0:20] = 8.9
    cfg.target[500:700, 780:-1] = 8.9
    
    fig, ax = plt.subplots()
    im = ax.imshow(cfg.target, origin="lower", extent=None)  # default colormap
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Density")
    plt.savefig('./figures/target.png', dpi=150, transparent=True)
    plt.show()
    
    cfg.nvar = len(rv_mean)
    
    # bounds
    cfg.action_low = rv_lb
    cfg.action_high = rv_ub
    
    cfg.device = "cpu"
    cfg.seed = 42
    cfg.gamma = 0.99
    cfg.gae_lambda = 0.95
    
    cfg.total_frames = 2048
    cfg.frames_per_batch = 256
    cfg.minibatch_size = 16
    cfg.ppo_epochs = 5
    cfg.max_grad_norm = 1.0
    cfg.eval_every_n_batches = 1
    cfg.max_steps = 1
    

    def load_surrogate(ckpt_path: str, device: str | None = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ## ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ## meta = ckpt["meta"]

        ## model = MLP(
        ##     in_dim=meta["input_dim"],
        ##     hidden_dims=tuple(meta["hidden_dims"]),
        ##     dropout=meta["dropout"],
        ##     out_dim=meta["output_dim"],
        ## ).to(device).eval()
        featureList = [768,512,256,128,64]
        linearFeatures = 768
        model = tCNNsurrogate(
            input_size=29,
            linear_features=(7, 5, linearFeatures),
            initial_tconv_kernel=(5, 5),
            initial_tconv_stride=(5, 5),
            initial_tconv_padding=(0, 0),
            initial_tconv_outpadding=(0, 0),
            initial_tconv_dilation=(1, 1),
            kernel=(3, 3),
            nfeature_list=featureList,
            output_image_size=(1120, 800),
            act_layer=nn.GELU,
        )
        
        ## with torch.no_grad():
        ##     _ = model(torch.zeros(1, meta["input_dim"], dtype=torch.float32, device=device))
        
        
        ## state = ckpt["model_state_dict"]
        ## if any(k.startswith("module.") for k in state):
        ##     state = {k.replace("module.", ""): v for k, v in state.items()}
        ## model.load_state_dict(state, strict=True)
        initial_learningrate = 5.00E-03
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=initial_learningrate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.01,
        )
        
        filepath = "./data/emulator/study011_modelState_epoch0100.hdf5"
        
        epoch = load_model_and_optimizer_hdf5(model,optimizer,filepath)
        
        return model, device

    def make_sim_fn_from_ckpt(ckpt_path: str, device: str | None = None) -> Callable[[np.ndarray], float]:
        model, device = load_surrogate(ckpt_path, device)
        
        ## x_mean = torch.from_numpy(np.asarray(meta["x_mean"], dtype=np.float32)).to(device)
        ## x_std  = torch.from_numpy(np.asarray(meta["x_std"],  dtype=np.float32)).to(device)
        ## norm_tgt = bool(meta.get("normalize_target", False))
        ## if norm_tgt:
        ##     y_mean = torch.from_numpy(np.asarray(meta["y_mean"], dtype=np.float32)).to(device)
        ##     y_std  = torch.from_numpy(np.asarray(meta["y_std"],  dtype=np.float32)).to(device)

        ## eps = torch.finfo(torch.float32).eps

        @torch.inference_mode()
        def sim_fn(x_in: np.ndarray) -> np.ndarray:
            """
            Runs the model on a 1D input vector and returns an image prediction
            as a numpy array. Removes only the batch dimension; preserves channels.
            """
            x_in = np.asarray(x_in, dtype=np.float32)
            x = np.zeros(28+1, dtype=np.float32)
            x[:28] = x_in
            x[-1] = 25.0
            if x.ndim != 1:
                raise ValueError(f"sim_img expects a 1D feature vector; got shape {x.shape}")
            ## if x.shape[0] != meta["input_dim"]:  # e.g., 28
            ##     raise ValueError(f"Expected input_dim={meta['input_dim']}, got {x.shape[0]}")

            xt = torch.from_numpy(x).unsqueeze(0).to(device)                 # (1, D)
            # xs = (xt - x_mean) / torch.clamp(x_std, min=eps)                 # normalize inputs
            xs = torch.nan_to_num(xt, nan=0.0, posinf=0.0, neginf=0.0)

            model.eval()
            with torch.no_grad():
                y = model(xs)    # xs                                     # (1, C, H, W) or (1, H, W) or (1, N)

            # If you normalized targets during training, make sure y_mean/y_std are broadcastable to y_scaled.
            ## y = (y_scaled * y_std + y_mean) if norm_tgt else y_scaled
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

            # Drop only the batch dimension; DO NOT squeeze all dims or you might drop channel=1 by accident.
            y = y.squeeze(0)                                                 # now (C,H,W) or (H,W) or (N,)

            # (Optional) If the model returns a flat vector, reshape to your image size:
            # e.g., output_image_size = (1120, 800); optionally a channel count.
            h, w = (1120, 800)
            if y.ndim == 1:
                num = y.numel()
                if num == h * w:
                    y = y.view(h, w)
                elif num % (h * w) == 0:
                    c = num // (h * w)
                    y = y.view(c, h, w)

            return y.detach().cpu().numpy()

        return sim_fn
    
    sim_fn = make_sim_fn_from_ckpt("./data/emulator/study011_modelState_epoch0100.hdf5", device="cpu")
    
    
    if run_train:
        logs = train(sim_fn, cfg)
        
    else:
        state = torch.load(cfg.save_path, map_location=cfg.device)
        with np.load("training_logs.npz") as f:
            logs = {k: f[k] for k in f.files}
    
    import torch
    from torchrl.envs.utils import ExplorationType, set_exploration_type

    # 1) Rebuild env and actor exactly as in training
    env = make_env(sim_fn, cfg)                       # your factory
    actor, _ = build_actor_critic(env, cfg)

    # 2) Load weights
    state = torch.load(cfg.save_path, map_location=cfg.device)
    actor.load_state_dict(state)
    actor.to(cfg.device).eval()

    # 3) Deterministic single-step rollout (works for bandit: max_steps=1)
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        td = env.rollout(
            policy=actor,
            max_steps=cfg.max_steps,      # 1 for your bandit; >1 if episodic
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
        )
        # td["action"] has a time (and maybe env) dimension
        a = td.get("action")              # shape [T, N_env, A] or [T, A]
        a_star = a[-1]                    # last time step
        if a_star.ndim == 2:              # [N_env, A]
            a_star = a_star[0]            # take env 0 (or mean across envs)
        a_star = a_star.detach().cpu().numpy()

    print("Deterministic action:", a_star)
    
    
    # def eval_env(a_star):
    # 1) Reset and set the action
    td = env.reset().to(cfg.device)
    a = torch.as_tensor(a_star, dtype=torch.float32, device=cfg.device).unsqueeze(0)  # (1, A)
    td.set("action", a)

    # 2) Step once (bandit: one step is enough)
    td_next = env.step(td)

    # 3) Pull the image out of the tensordict
    # If your observation is a Dict with keys y/h1/h2 (as we wired earlier):
    if ("next", "h1") in td_next.keys(True):
        img_t = td_next.get(("next", "h1"))[0]       # torch, shape (C,H,W)
    # If your observation is a single Box image:
    else:
        img_t = td_next.get(("next", "observation"))[0]  # torch, shape (C,H,W)

    # Optionally, you can also fetch the raw sim output from Gym info:
    # (GymWrapper puts info under "_extra")
    if ("next","_extra","sim_output") in td_next.keys(True):
        img_raw = td_next.get(("next","_extra","sim_output"))[0]  # same content

    # 4) Convert and visualize (assumes single-channel)
    img = img_t.detach().cpu().numpy()
    to_show = img[0] if img.ndim == 3 else img
    plt.imshow(to_show, origin="lower")
    plt.colorbar()
    plt.savefig('simulated_best_output.png', dpi=150)
    plt.show()
    

    # If you also want the reward:
    reward = td_next.get(("next","reward")).item()
    print("reward at a_star:", reward)
    
    
    
    
    
    
    
    import matplotlib.pyplot as plt
    import numpy as np

    def moving_avg(x, k=10):
        if len(x) < 2 or k <= 1:
            return np.asarray(x, dtype=float)
        k = min(k, len(x))
        w = np.ones(k, dtype=float) / k
        return np.convolve(np.asarray(x, dtype=float), w, mode="valid")

    # --- Loss curves per PPO inner epoch ---
    plt.figure()
    plt.plot(logs["epoch_total"], label="total")
    plt.plot(logs["epoch_actor"], label="actor/objective")
    plt.plot(logs["epoch_critic"], label="critic")
    plt.plot(logs["epoch_entropy"], label="entropy")
    plt.title("PPO losses per inner epoch")
    plt.xlabel("Inner epoch (across all outer batches)")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ppo_losses.png", dpi=150)

    # --- Reward per outer batch (collector iteration) ---
    plt.figure()
    plt.plot(logs["batch_reward"], label="batch mean reward")
    # optional smoothing for readability
    ma = moving_avg(logs["batch_reward"], k=5)
    x = np.arange(len(ma)) + (len(logs["batch_reward"]) - len(ma))
    plt.plot(x, ma, label="moving avg (k=5)")
    plt.title("Mean reward per outer batch")
    plt.xlabel("Outer batch index")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ppo_batch_reward.png", dpi=150)
    plt.show()
