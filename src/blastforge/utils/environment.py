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

import matplotlib.pyplot as plt
import os




class SimEnv(gym.Env):
    """
    A single-step bandit-like env:
      - action: R^d (bounded) --> fed to simulation
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
    
    def evaluate(self, action: np.ndarray):
        print('Conducting environment step ...')
        self.step_count += 1

        # Enforce bounds to be safe, even though the policy is bounded.
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 1) Run the simulator/model → image-like output (no scalar cast!)
        sim_out = self.sim_fn(action)              # model → image
        img = self._coerce_img(sim_out)            # (C,H,W) float32
        return img

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
    
    def close(self):
        if hasattr(self, "_last_sim_out"):
            self._last_sim_out = None
        if hasattr(self.sim_fn, "close") and callable(self.sim_fn.close):
            self.sim_fn.close()
        try:
            super().close()
        except Exception:
            pass



# ----------------------------
# Make TorchRL environment
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


# Immediate reward metric (-MSE) you want to maximize

def mse_2d(y_true, y_pred, *, mask: Optional[np.ndarray] = None, nan_safe: bool = False) -> float:
    """
    Compute the Mean Squared Error (MSE) between two arrays of 2D field values.

    Parameters
    ----------
    y_true, y_pred : array-like
        Arrays with identical shape (e.g., (H, W) or (T, H, W)).
    mask : np.ndarray, optional
        Boolean array of the same shape as inputs. True values are INCLUDED in the MSE;
        False values are ignored. If provided, NaNs at masked-in points are still handled
        according to `nan_safe`.
    nan_safe : bool, default False
        If True, ignores NaNs using `np.nanmean`. If False, any NaN will propagate.

    Returns
    -------
    float
        The mean squared error over all included elements.
    """
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Shapes must match, got {a.shape} and {b.shape}")

    diff2 = (a - b) ** 2

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != a.shape:
            raise ValueError(f"Mask shape must match inputs, got {m.shape} vs {a.shape}")
        # Only keep masked-in elements
        diff2 = diff2[m]

    if nan_safe:
        return float(np.nanmean(diff2))
    else:
        return float(np.mean(diff2))

