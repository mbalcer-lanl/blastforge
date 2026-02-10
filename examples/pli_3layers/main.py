"""
run in blastforge environment with:
python main.py

cd /mnt/c/Users/349957/Documents/1Research/lanl/blastforge/git/fork/pretrained_action_NN/blastforge/examples/pli_3layers
conda activate bf_fork
python main.py

Train a PPO policy (TorchRL) on a custom Gymnasium env that calls a
user-provided emulator of a PLI density field of copper at t=25\mu s 
with 28 inputs and returns a single scalar.
The immediate reward is the Mean Squared Error (MSE) of the density field 
between the target and emulator output.
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

import matplotlib.pyplot as plt
import os

from blastforge.utils.environment import make_env # SimEnv
from blastforge.utils.config_ch import PPOConfig
from blastforge.RL.ppo import train
from blastforge.utils.build_models import make_sim_fn_from_ckpt, build_policy_network_gaussian_cnn

import multiprocessing as mp # do we need?




if __name__ == "__main__":
    
    mp.set_start_method("fork", force=True) # do we need?
    
    # Flag to run the training loop of the policy network
    run_train = True
    
    # get default command line arguments
    cfg = PPOConfig()
    
    # absolute path to main blastforge directory
    bf_dir = '/users/mbalcer/blastforge/pretrain_policy/blastforge/'
    
    # filepaths to models
    cfg.emulator_filepath = bf_dir+"src/blastforge/models/emulator/study012_modelState_epoch0100.hdf5"
    cfg.value_pretrain_filepath = bf_dir+'src/blastforge/models/value/value_NN.pth'
    cfg.policy_pretrain_filepath = bf_dir+'src/blastforge/models/policy/study001_modelState_epoch0080.pth'
    
    cfg.norm_file = bf_dir+'src/blastforge/models/policy/lsc240420_Bspline_norms.npz'
    

    
    
    # define bounds of geometric parameters
    # lower bounds
    cfg.action_low = np.array([
       # liner position
       3.0333559 , 3.13851665, 3.54830177, 4.22973636, 5.18432918,
       6.41485328, 8.00023045,
       0.05, # thicknesses of Cu
       0.05,
       0.05,
       0.05,
       0.05,
       0.05,
       0.05,
       0.05, # thicknesses of Al
       0.05,
       0.05,
       0.05,
       0.05,
       0.05,
       0.05,
       0.05, # thicknesses of Sy
       0.05,
       0.05,
       0.05,
       0.05,
       0.05,
       0.05,
       # 25.0, # radius of shell
       ])
    # upper bounds
    cfg.action_high = np.array([
       # liner position
       15, 15, 15, 15, 15, 15, 15,
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
    cfg.target[500:700, 399-20:399+20] = 8.93
    
    
    cfg.nvar = len(cfg.action_low)
    
    # PPO arguments
    cfg.device = "cpu"
    cfg.seed = 42
    cfg.gamma = 0.99
    cfg.gae_lambda = 0.95
    
    cfg.total_frames = 256
    cfg.frames_per_batch = 16
    cfg.minibatch_size = int(cfg.total_frames/cfg.frames_per_batch) # total_frames = frames_per_batch*minibatch_size
    cfg.ppo_epochs = 5
    cfg.max_grad_norm = 1.0
    cfg.eval_every_n_batches = 1
    cfg.max_steps = 1

    case_name = '{0}_{1}'.format(cfg.total_frames,cfg.frames_per_batch)
    
    cfg.data_path = './'+case_name+'/data/'
    cfg.fig_path = './'+case_name+'/figures/'
    cfg.save_path = './'+case_name+'/data/ppo_sim_actor.pt'
    # create data and figures directory if it does not exist
    os.makedirs(cfg.fig_path, exist_ok=True)
    os.makedirs(cfg.save_path, exist_ok=True)
    
    # plot target if requested
    plt_target = True
    if plt_target:
        fig, ax = plt.subplots()
        im = ax.imshow(cfg.target, origin="lower", vmin=0.0, vmax=9.0)  # default colormap
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Density")
        plt.savefig(cfg.fig_path+'target.png', dpi=150, transparent=True)
        plt.show()

    # run training loop (PPO)
    if run_train:
        logs = train(cfg)
        
    else:
        state = torch.load(cfg.save_path, map_location=cfg.device)
        with np.load("training_logs.npz") as f:
            logs = {k: f[k] for k in f.files}
    
    
    ## Post-process
    sim_fn = make_sim_fn_from_ckpt(cfg.emulator_filepath, cfg, device="cpu")
    
    # Rebuild env and actor exactly as in training
    env = make_env(sim_fn, cfg)                       # your factory
    actor = build_policy_network_gaussian_cnn(env, cfg)

    # Load weights
    state = torch.load(cfg.save_path, map_location=cfg.device)
    actor.load_state_dict(state)
    actor.to(cfg.device).eval()

    # Deterministic single-step rollout (works for bandit: max_steps=1)
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
    
    
    # Reset and set the action
    td = env.reset().to(cfg.device)
    a = torch.as_tensor(a_star, dtype=torch.float32, device=cfg.device).unsqueeze(0)  # (1, A)
    td.set("action", a)

    # Step once (bandit: one step is enough)
    td_next = env.step(td)

    # Pull the image out of the tensordict
    # if observation is a Dict with keys y/h1/h2:
    if ("next", "h1") in td_next.keys(True):
        img_t = td_next.get(("next", "h1"))[0]       # torch, shape (C,H,W)
    # if observation is a single Box image:
    else:
        img_t = td_next.get(("next", "observation"))[0]  # torch, shape (C,H,W)

    # Optionally, you can also fetch the raw sim output from Gym info:
    # (GymWrapper puts info under "_extra")
    if ("next","_extra","sim_output") in td_next.keys(True):
        img_raw = td_next.get(("next","_extra","sim_output"))[0]  # same content

    # Convert and visualize (assumes single-channel)
    img = img_t.detach().cpu().numpy()
    to_show = img[0] if img.ndim == 3 else img
    plt.imshow(to_show, origin="lower", vmin=0.0, vmax=9.0)
    plt.colorbar()
    plt.savefig(cfg.fig_path+'simulated_best_output.png', dpi=150)
    plt.show()
    

    # Get reward:
    reward = td_next.get(("next","reward")).item()
    print("reward at a_star:", reward)
    
    # plot
    
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
    plt.savefig(cfg.fig_path+"ppo_losses.png", dpi=150)

    # --- Reward per outer batch (collector iteration) ---
    plt.figure()
    plt.plot(logs["eval_return"]) #, label="batch reward")
    #x = np.arange(len(ma)) + (len(logs["eval_return"]) - len(ma))
    plt.title("Reward of predicted action per batch")
    plt.xlabel("Outer batch index")
    plt.ylabel("Reward")
    #plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.fig_path+"ppo_batch_reward.png", dpi=150)
    plt.show()
