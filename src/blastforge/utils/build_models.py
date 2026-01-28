from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

import numpy as np
import torch
from torch import nn

import h5py

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
from blastforge.models.models import tCNNsurrogate, hybrid2vectorCNN

# ----------------------------
# Build policy NN
# ----------------------------

def build_policy_network(env: TransformedEnv, cfg: PPOConfig):
    """
    Builds:
      - ProbabilisticActor with TanhNormal bounded by env.action_spec
      - ValueOperator for state value V(s)
    """
    print('Building policy network ...')
    device = torch.device(cfg.device)
    action_dim = int(np.prod(env.action_spec.shape))
    
    
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
        module=head, # nn.Sequential(policy_backbone, policy_head), # net,
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
    
    return actor

# ----------------------------
# Build value NN
# ----------------------------

def build_value_network(env: TransformedEnv, cfg: PPOConfig):
    """
    Builds:
      - ProbabilisticActor with TanhNormal bounded by env.action_spec
      - ValueOperator for state value V(s)
    """
    # Value network V(s)
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
    
    value_net = hybrid2vectorCNN(**value_model_args).to(cfg.device)
    
    def disable_bn_running_stats_(module: nn.Module):
        for m in module.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
                # keep m.training = True globally; with track_running_stats=False
                # BN uses batch stats in both train/eval
    
    # hybrid2vectorCNN has BN in its CNN blocks
    disable_bn_running_stats_(value_net)
    
    state = torch.load(cfg.value_pretrain_filepath, map_location=cfg.device, weights_only=False)
    # from torch.serialization import safe_globals
    # with safe_globals([torch.nn.modules.activation.GELU]):
    #     state = torch.load(cfg.value_pretrain_filepath, map_location=device, weights_only=False)
    
    # (Optional) unwrap common checkpoint formats
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    
    # (Optional) remove 'module.' prefix if saved with DataParallel/DistributedDataParallel
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    
    # 3) Load into the model
    missing, unexpected = value_net.load_state_dict(state, strict=False)
    print("missing:", missing)
    print("unexpected:", unexpected)
    
    value = ValueOperator(
        module=value_net,
        in_keys=["y", "h1", "h2"],      # <— three inputs
        out_keys=["state_value"],       # optional; default is "state_value"
    ).to(cfg.device)
    
    return value

# ----------------------------
# Build environment emulator NN
# ----------------------------

def load_model_and_optimizer_hdf5(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, filepath: str
) -> int:
    """Loads state of model and optimizer stored in an hdf5 format.

    Args:
        model (torch.nn.Module): Pytorch model to load state into.
        optimizer (torch.optim.Optimizer): Pytorch optimizer to load state into.
        filepath (str): Path to the hdf5 checkpoint file.

    Returns:
        epoch (int): Epoch associated with training

    """
    # If model is wrapped in DataParallel, access the underlying module
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    with h5py.File(filepath, "r") as h5f:
        # Get epoch number
        epoch = h5f.attrs["epoch"]

        # Load model parameters and buffers
        for name in h5f.get("model/parameters", []):  # Get the group
            if isinstance(h5f["model/parameters/" + name], h5py.Dataset):
                data = torch.from_numpy(h5f["model/parameters/" + name][:])
            else:
                data = torch.tensor(h5f.attrs["model/parameters/" + name])

            name_list = name.split(".")
            param_name = name_list.pop()
            submod_name = ".".join(name_list)

            model.get_submodule(submod_name)._parameters[param_name].data.copy_(data)

        for name in h5f.get("model/buffers", []):
            if isinstance(h5f["model/buffers/" + name], h5py.Dataset):
                buffer = torch.from_numpy(h5f["model/buffers/" + name][:])
            else:
                buffer = torch.tensor(h5f.attrs["model/buffers/" + name])

            name_list = name.split(".")
            param_name = name_list.pop()
            submod_name = ".".join(name_list)
            model.get_submodule(submod_name)._buffers[param_name].data.copy_(buffer)

        # Rebuild optimizer state (need to call this before loading state)
        optimizer_state = optimizer.state_dict()

        # Load optimizer parameter groups
        for k in h5f.attrs:
            if "optimizer/group" in k:
                # print('k-string:', k)
                idx, param = k.split("/")[1:]
                optimizer_state["param_groups"][int(idx.lstrip("group"))][param] = (
                    h5f.attrs[k]
                )

        # Load state values, like momentums
        for name, group in h5f.items():
            if "optimizer/state" in name:
                state_idx = int(name.split("state")[1])
                param_idx, param_state = list(optimizer_state["state"].items())[
                    state_idx
                ]
                for k in group:
                    optimizer_state["state"][param_idx][k] = torch.from_numpy(
                        group[k][:]
                    )

        # Load optimizer state
        optimizer.load_state_dict(optimizer_state)

    return epoch
'''

def make_sim_fn_from_ckpt(device: str | None = None) -> Callable[[np.ndarray], float]: # ckpt_path: str, 
    
    # hyperparameters
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
    
    initial_learningrate = 5.00E-03
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=initial_learningrate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
    )
    
    epoch = load_model_and_optimizer_hdf5(model,optimizer,cfg.emulator_filepath)
    
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
    
'''


def load_surrogate(ckpt_path: str, cfg: PPOConfig, device: str | None = None):
    # device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
    
    # filepath = "./data/emulator/study012_modelState_epoch0100.hdf5"
    
    epoch = load_model_and_optimizer_hdf5(model,optimizer,cfg.emulator_filepath)
    
    return model# , device

def make_sim_fn_from_ckpt(ckpt_path: str, cfg = PPOConfig, device: str | None = None) -> Callable[[np.ndarray], float]:
    model = load_surrogate(ckpt_path, cfg)
    
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

        xt = torch.from_numpy(x).unsqueeze(0).to(cfg.device)                 # (1, D)
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



'''

# ----------------------------
# Build policy / value NN
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
    
    print('Building actor NN ...')
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
    
    print('Building value NN ...')
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
    
    
    state = torch.load(cfg.value_pretrain_filepath, map_location=device, weights_only=False)
    from torch.serialization import safe_globals
    # with safe_globals([torch.nn.modules.activation.GELU]):
    #     state = torch.load(cfg.value_pretrain_filepath, map_location=device, weights_only=False)
    
    # (Optional) unwrap common checkpoint formats
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    
    # (Optional) remove 'module.' prefix if saved with DataParallel/DistributedDataParallel
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    
    # 3) Load into the model
    missing, unexpected = value_net.load_state_dict(state, strict=False)
    print("missing:", missing)
    print("unexpected:", unexpected)

    value = ValueOperator(
        module=value_net,
        in_keys=["y", "h1", "h2"],      # <— three inputs
        out_keys=["state_value"],       # optional; default is "state_value"
    ).to(device)
    
    return actor, value


'''
