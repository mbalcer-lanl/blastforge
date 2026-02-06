from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

import numpy as np
import torch
from torch import nn

import torch.distributed as dist

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

from torchrl.modules import SafeProbabilisticModule

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, SafeProbabilisticModule
from torch.distributions import MultivariateNormal
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

import matplotlib.pyplot as plt
import os


from blastforge.utils.environment import make_env # SimEnv
from blastforge.utils.config import PPOConfig
from blastforge.models.models import tCNNsurrogate, hybrid2vectorCNN

from blastforge.models.policy.policy_model import gaussian_policyCNN, gaussian_Image2VectorCNN




# ----------------------------
# Build policy NN
# ----------------------------

def build_policy_network(env: TransformedEnv, cfg: PPOConfig):
    """
    Builds:
      - ProbabilisticActor with TanhNormal bounded by env.action_spec
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



def build_policy_network_gaussian_cnn(env, cfg, *, cnn_kwargs=None):
    """
    Builds a bounded stochastic policy actor (TanhNormal) from a
    gaussian_Image2VectorCNN backbone.

    The gaussian_Image2VectorCNN returns a torch.distributions.MultivariateNormal
    with (mean, full covariance). Since TorchRL's TanhNormal is parameterized by
    (loc, scale) (diagonal std), we convert full covariance -> diagonal std via
    sqrt(diag(cov)).

    Args:
        env: TransformedEnv (must expose env.action_spec with .shape/.low/.high)
        cfg: PPOConfig (must expose cfg.device)
        cnn_kwargs: optional dict forwarded to gaussian_Image2VectorCNN(...)

    Returns:
        actor: ProbabilisticActor producing bounded actions in env.action_spec.
    """
    print("Building policy network (gaussian_Image2VectorCNN) ...")

    device = torch.device(cfg.device)
    action_dim = int(np.prod(env.action_spec.shape))

    cnn_kwargs = {} if cnn_kwargs is None else dict(cnn_kwargs)
    
    cnn_kwargs = {
        'img_size': (1, 1120, 800), 
        'output_dim': 28, 
        'size_threshold': (12, 12), 
        'kernel': 3, 
        'features': 32, 
        'interp_depth': 20, 
        'conv_onlyweights': True, 
        'batchnorm_onlybias': True, 
        'hidden_features': 64, 
        'final_activation': nn.Identity
        }

    # Ensure the CNN output matches the action dimension unless the caller overrides.
    # (If caller passes output_dim explicitly, this will respect it.)
    cnn_kwargs.setdefault("output_dim", action_dim)

    # --- Your provided network class must be in scope ---
    # from your_module import gaussian_Image2VectorCNN
    backbone = gaussian_Image2VectorCNN(**cnn_kwargs).to(device)

    # Hard check: output_dim must match action_dim for a valid policy.
    if getattr(backbone.i2v_cnn, "output_dim", None) is not None:
        if int(backbone.i2v_cnn.output_dim) != action_dim:
            raise ValueError(
                f"gaussian_Image2VectorCNN output_dim ({backbone.i2v_cnn.output_dim}) "
                f"!= action_dim ({action_dim}). Pass cnn_kwargs={{'output_dim': {action_dim}}}."
            )

    class MVNToLocScale(nn.Module):
        """Wrap gaussian_Image2VectorCNN to output loc/scale suitable for TanhNormal."""
        def __init__(self, mvn_net: nn.Module, min_scale: float = 1e-6):
            super().__init__()
            self.mvn_net = mvn_net
            self.min_scale = float(min_scale)

        def forward(self, h1: torch.Tensor):
            # mvn_net returns torch.distributions.MultivariateNormal
            dist = self.mvn_net(h1)
            loc = dist.mean  # (N, action_dim)

            # Convert full covariance -> diagonal std for TanhNormal
            cov = dist.covariance_matrix  # (N, action_dim, action_dim)
            var = cov.diagonal(dim1=-2, dim2=-1)  # (N, action_dim)
            scale = torch.sqrt(torch.clamp(var, min=self.min_scale))  # (N, action_dim)

            return loc, scale

    policy_head = MVNToLocScale(backbone)

    policy_param_module = TensorDictModule(
        module=policy_head,
        in_keys=["h1"],
        out_keys=["loc", "scale"],
    )

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
        in_keys=["x", "h1", "h2"],      # <— three inputs
        out_keys=["state_value"],       # optional; default is "state_value"
    ).to(cfg.device)
    
    return value



# ----------------------------
# Build pretrained policy NN
# ----------------------------

class DistToParamsWrapper(torch.nn.Module):
    def __init__(self, base_net, jitter=1e-6):
        super().__init__()
        self.base = base_net
        self.jitter = jitter

    def forward(self, x_curr, h2):
        # x_curr=h1
        h_target=h2
        dist = self.base(h_target)                       # MultivariateNormal
        if dist is None:
            raise RuntimeError("base_net returned None instead of a Distribution")
        
        print('dist =', dist.mean)
        # extract tensors needed by the distribution constructor
        loc = dist.mean - x_curr                          # (batch, action_dim)
        print
        # preferred stable param: scale_tril if present
        if hasattr(dist, "scale_tril") and dist.scale_tril is not None:
            scale_tril = dist.scale_tril
        else:
            cov = dist.covariance_matrix
            eye = torch.eye(cov.size(-1), device=cov.device, dtype=cov.dtype)
            scale_tril = torch.linalg.cholesky(cov + self.jitter * eye)

        # sanity checks: raise informative errors early
        if loc is None:
            raise RuntimeError("extracted loc is None (check base_net output)")
        if scale_tril is None:
            raise RuntimeError("extracted scale_tril is None (check base_net output)")

        return loc, scale_tril # {"loc": loc, "scale_tril": scale_tril}

def build_pretrained_policy_network(env: TransformedEnv, cfg: PPOConfig):
    """
    Builds:
      - ProbabilisticActor with TanhNormal bounded by env.action_spec
    """
    img_h = 1120
    img_w = 800
    input_vector_size = 29
    output_dim = 29
    
    policy_model_args = {
        'img_size': (1, 1120, 800), 
        'output_dim': 29, 
        'size_threshold': (12, 12), 
        'kernel': 3, 
        'features': 32, 
        'interp_depth': 20, 
        'conv_onlyweights': True, 
        'batchnorm_onlybias': True, 
        'hidden_features': 64, 
        'final_activation': nn.Identity
        }
    
    print('initializing pretrained policy NN')
    # policy_net = gaussian_Image2VectorCNN(**policy_model_args).to(cfg.device)
    
    available_models = {
        "gaussian_Image2VectorCNN": gaussian_Image2VectorCNN
    }
    
    # , optimizer, starting_epoch
    print('loading pretrained policy NN')
    policy_net = load_model_and_optimizer(
            cfg.policy_pretrain_filepath,
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={
                "lr": 1e-3,
                "betas": (0.9, 0.999),
                "eps": 1e-08,
                "weight_decay": 0.0, # 0.01, zero weight decay for only mean_mlp
            },
            available_models=available_models,
            device=cfg.device,
        )
    
    
    def disable_bn_running_stats_(module: nn.Module):
        for m in module.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
                # keep m.training = True globally; with track_running_stats=False
                # BN uses batch stats in both train/eval
    
    # gaussian_policyCNN has BN in its CNN blocks
    disable_bn_running_stats_(policy_net)
    
    state = torch.load(cfg.policy_pretrain_filepath, map_location=cfg.device, weights_only=False)
    # from torch.serialization import safe_globals
    # with safe_globals([torch.nn.modules.activation.GELU]):
    #     state = torch.load(cfg.value_pretrain_filepath, map_location=device, weights_only=False)
    
    # print(state.keys())          # sometimes contains 'state_dict' and 'args' / 'config'
    # if 'model_args' in state:
    #     print(state['model_args'])
    
    # missing, unexpected = policy_net.load_state_dict(state, strict=False)
    # print("missing:", missing)
    # print("unexpected:", unexpected)
    
    wrapped = TensorDictModule(
        module=policy_net, # DistToParamsWrapper(),
        in_keys=["h1"], # ["x_curr", "h_target"],
        out_keys=["loc", "scale"],   # keys your wrapper returns
    )
    ## from tensordict import TensorDict
    ## test_input = np.zeros((1,1,1120, 800), dtype=np.float32)
    ## 
    ## # test_input = torch.from_numpy(test_input).float()
    ## td = TensorDict({"x_curr": np.zeros(29, dtype=np.float32), "h2": test_input})
    ## td_out = wrapped(td)   # wrapped is your TensorDictModule
## 
    ## print("TD keys after wrapped:", list(td_out.keys()))
    ## print("loc value:", td_out.get("loc"))           # should not be None
    ## print("scale_tril value:", td_out.get("scale_tril"))
    
    # policy = SafeProbabilisticModule(
    #     in_keys={"loc": "loc", "scale_tril": "scale_tril"},  # map dist kwarg -> td key
    #     out_keys=["action"],          # where sampled action will be written
    #     distribution_class=MultivariateNormal,
    #     return_log_prob=True,
    #     spec=env.action_spec,
    # ).to(cfg.device)
    
    policy = ProbabilisticActor(
        module=wrapped,
        in_keys=["loc", "scale"],
        spec=env.action_spec,
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec.low,
            "high": env.action_spec.high,
        },
        return_log_prob=True,
        safe=True,
    ).to(cfg.device)
    
    return policy


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



def load_model_and_optimizer(
    filepath: str,
    optimizer_class: type,
    optimizer_kwargs: dict,
    available_models: dict,
    device: str = "cuda",
) -> tuple[torch.nn.Module]: # , torch.optim.Optimizer, int
    """Dynamically load model & optimizer state from checkpoint.

    NOTE: This function only works while loading checkpoints created by
    `save_model_and_optimizer`

    - Working for both DDP and non-DDP training.
    - Loads the checkpoint only on rank 0 when in DDP.
    - If using DDP, broadcasts the checkpoint to all other ranks.
    - Handles models both inside and outside of `DistributedDataParallel`.

    Args:
        filepath (str): Checkpoint filename.
        optimizer_class (type): Torch optimizer class
        optimizer_kwargs (dict): Dictionary of optimizer parameters.
        available_models (dict): Dictionary mapping class names to class references.
        device (torch.device): String or device specifier.

    """
    # Get rank if in DDP, else assume single process
    if dist.is_initialized():
        load_rank = dist.get_rank()
    else:
        load_rank = 0

    checkpoint = None

    if load_rank == 0:
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        epochIDX = checkpoint["epoch"]
        print(f"[Rank {load_rank}] Loaded checkpoint from epoch {epochIDX}")

    # If in DDP, broadcast checkpoint to all ranks
    if dist.is_initialized():
        checkpoint_list = [checkpoint]
        dist.broadcast_object_list(checkpoint_list, src=0)
        checkpoint = checkpoint_list[0]  # Unpack checkpoint on all ranks

    # Retrieve model class and arguments
    model_class_name = checkpoint["model_class"]
    model_args = checkpoint["model_args"]

    # Ensure model class exists
    if model_class_name not in available_models:
        raise ValueError(
            f"Unknown model class: {model_class_name}. Add it to `available_models`."
        )

    # Dynamically create the model
    model = available_models[model_class_name](**model_args)

    # Load state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Move model to GPU if necessary
    model.to(device)

    # Initialize optimizer and move to device
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Move optimizer state to GPU if necessary
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)

    # Synchronize all processes in DDP
    if dist.is_initialized():
        dist.barrier()

    return model # , optimizer, checkpoint["epoch"]




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




def make_sim_fn_from_ckpt(ckpt_path: str, cfg: PPOConfig, device: str | None = None) -> Callable[[np.ndarray], float]:
    
    model = load_surrogate(ckpt_path, cfg)
    dev = torch.device(device or cfg.device)
    model = model.to(dev)
    model.eval()
    
    H, W = 1120, 800

    @torch.inference_mode()
    def sim_fn(x_in: np.ndarray) -> np.ndarray:
        x_in = np.asarray(x_in, dtype=np.float32)
        # Accept x shaped (D,), (1, D), or (B, D)
        if x_in.ndim == 2:
            if x_in.shape[0] == 1:
                x_in = x_in[0]                 # (1, D) -> (D,)
            else:
                # If you want to support batching, handle it explicitly (see below)
                raise ValueError(f"sim_fn got batch x with shape {x_in.shape}; expected (D,) or (1,D)")
        elif x_in.ndim != 1:
            raise ValueError(f"sim_fn expects 1D x; got {x_in.shape}")

        x = np.zeros(28 + 1, dtype=np.float32)
        x[:28] = x_in
        x[-1] = 25.0

        xt = torch.from_numpy(x).unsqueeze(0).to(dev)
        xt = torch.nan_to_num(xt, nan=0.0, posinf=0.0, neginf=0.0)

        y = model(xt)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = y.squeeze(0)

        if y.ndim == 1:
            num = y.numel()
            if num == H * W:
                y = y.view(H, W)
            elif num % (H * W) == 0:
                c = num // (H * W)
                y = y.view(c, H, W)

        return y.detach().cpu().numpy()

    return sim_fn
