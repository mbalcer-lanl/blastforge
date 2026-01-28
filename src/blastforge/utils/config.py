

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

import numpy as np
import torch
from torch import nn

import gymnasium as gym
from gymnasium import spaces

import os


# ----------------------------
# Configuration of user inputs
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
    
    # Model file paths
    emulator_filepath: str = "./src/blastforge/models/emulator/study012_modelState_epoch0100.hdf5"
    value_pretrain_filepath: str = './data/value/reward_regular_run_opt_fix_lr5e-4/runs/study_001/study001_modelState_epoch0100.pth'
