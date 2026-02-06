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
from blastforge.utils.config import PPOConfig
from blastforge.utils.build_models import build_policy_network, build_value_network, build_policy_network_gaussian_cnn, make_sim_fn_from_ckpt

import multiprocessing as mp # do we need?



# ----------------------------
# Training loop (PPO)
# ----------------------------

def train(cfg: PPOConfig):
    
    # initialize current state
    sim_fn = make_sim_fn_from_ckpt(cfg.emulator_filepath, cfg, device="cpu")
    
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

    # actor = build_policy_network(env, cfg)
    actor = build_policy_network_gaussian_cnn(env, cfg)
    value = build_value_network(env, cfg)

    # Collector: batches trajectories on-policy
    collector = SyncDataCollector(
        lambda: make_env(sim_fn, cfg),   # <- factory that returns a new EnvBase
        policy=actor,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        device=device,
        exploration_type=ExplorationType.RANDOM,
        use_buffers=False,
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
    
    try:
        for tensordict in collector:
            batches += 1
            print('Batch #', batches)
            # Compute advantages (adds "advantage" and "value_target")
            print('Computing advantage ...')
            with torch.no_grad():
                advantage(tensordict)
            
            adv = tensordict["advantage"]              # tensor on your device
            print("adv shape:", tuple(adv.shape))
            print("adv mean:", adv.mean().item())
            print("adv std: ", adv.std().item())
            print("adv min/max:", adv.min().item(), adv.max().item())
            
            # Flatten batch of transitions
            data_view = tensordict.reshape(-1)
            replay_buffer.extend(data_view)

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
                    batch = replay_buffer.sample(cfg.minibatch_size) # .to(device)
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
                eval_env = make_env(sim_fn, cfg)
                try:
                    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                        actor.eval()
                        td_eval = eval_env.rollout(
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
                        im = ax.imshow(to_show, origin="lower", vmin=0.0, vmax=9.0)  # default colormap
                        cbar = fig.colorbar(im, ax=ax)
                        cbar.set_label("Density")
                        plt.tight_layout()
                        plt.savefig('./figures/simulated_best_output_batch{0}.png'.format(batches), dpi=150, transparent=True)
                        
                        # plt.show()
                finally:
                    eval_env.close()
    finally:
        try:
            collector.shutdown(close_env=True)   # preferred if available
        except Exception:
            pass
        try:
            collector.close()      # some versions expose close() instead
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass
        
    # Save trained actor weights
    save_path_i = "ppo_sim_actor_batch{0}.pt".format(batches)
    torch.save(actor.state_dict(), save_path_i)
    print(f"Saved actor to: {save_path_i}")
    
    # save log file for current batch
    np.savez("training_logs.npz", **logs)
    
    # Save trained actor weights
    torch.save(actor.state_dict(), cfg.save_path)
    print(f"Saved actor to: {cfg.save_path}")
    
    # save raw logs for later
    # np.savez("training_logs.npz", **logs)
    
    
    return logs
