"""Reinforcement Learning (PPO) with TorchRL Tutorial."""

import warnings

from collections import defaultdict

import os
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
from torch import multiprocessing
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
    Transform,
)
from torchrl.envs.libs.gym import GymEnv, GymWrapper
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
from typing import Callable
from models import RewardPredictor

# TorchRL prefers spawn method, that restricts creation of
# ``~torchrl.envs.ParallelEnv`` inside
# `__main__` method call, but for the easy of reading the code switch to fork
# which is also a default spawn method in Google's Colaboratory
try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass

warnings.filterwarnings("ignore")

######################################################################
# If you are running this in Google Colab, make sure you install the following
# dependencies:
#
# .. code-block:: bash
#
#    !pip3 install torchrl
#    !pip3 install gym[mujoco]
#    !pip3 install tqdm
#
# Proximal Policy Optimization (PPO) is a policy-gradient algorithm where a
# batch of data is being collected and directly consumed to train the policy to maximise
# the expected return given some proximality constraints. You can think of it
# as a sophisticated version of `REINFORCE
# <https://link.springer.com/content/pdf/10.1007/BF00992696.pdf>`_,
# the foundational policy-optimization algorithm. For more information, see the
# `Proximal Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`_ paper.
#
# PPO is usually regarded as a fast and efficient method for online, on-policy
# reinforcement algorithm. TorchRL provides a loss-module that does all the work
# for you, so that you can rely on this implementation and focus on solving your
# problem rather than re-inventing the wheel every time you want to train a policy.
#
# For completeness, here is a brief overview of what the loss computes, even though
# this is taken care of by our :class:`~torchrl.objectives.ClipPPOLoss`
# module—the algorithm works as follows:
# 1. we will sample a batch of data by playing the
# policy in the environment for a given number of steps.
# 2. Then, we will perform a given number of optimization steps with random
# sub-samples of this batch using
# a clipped version of the REINFORCE loss.
# 3. The clipping will put a pessimistic bound on our loss: lower return estimates will
# be favored compared to higher ones.
# The precise formula of the loss is:
#
# .. math::
#
#     L(s,a,\theta_k,\theta) = \min\left(
#     \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a), \;\;
#     g(\epsilon, A^{\pi_{\theta_k}}(s,a))
#     \right),
#
# There are two components in that loss: in the first part of the minimum operator,
# we simply compute an importance-weighted version of the REINFORCE loss (for example, a
# REINFORCE loss that we have corrected for the fact that the current policy
# configuration lags the one that was used for the data collection).
# The second part of that minimum operator is a similar loss where we have clipped
# the ratios when they exceeded or were below a given pair of thresholds.
#
# This loss ensures that whether the advantage is positive or negative, policy
# updates that would produce significant shifts from the previous configuration
# are being discouraged.

######################################################################
# Define Hyperparameters
# ----------------------
#
# We set the hyperparameters for our algorithm. Depending on the resources
# available, one may choose to execute the policy on GPU or on another
# device.
# The ``frame_skip`` will control how for how many frames is a single
# action being executed. The rest of the arguments that count frames
# must be corrected for this value (since one environment step will
# actually return ``frame_skip`` frames).
#

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0) if torch.cuda.is_available() and not is_fork else torch.device("cpu")
)
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

######################################################################
# Data collection parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# When collecting data, we will be able to choose how big each batch will be
# by defining a ``frames_per_batch`` parameter. We will also define how many
# frames (such as the number of interactions with the simulator) we will allow
# ourselves to
# use. In general, the goal of an RL algorithm is to learn to solve the task
# as fast as it can in terms of environment interactions: the lower the ``total_frames``
# the better.
#
frames_per_batch = 1000
# For a complete training, bring the number of frames up to 1M
total_frames = 50_000

######################################################################
# PPO parameters
# ~~~~~~~~~~~~~~
#
# At each data collection (or batch collection) we will run the optimization
# over a certain number of *epochs*, each time consuming the entire data we just
# acquired in a nested training loop. Here, the ``sub_batch_size`` is different from the
# ``frames_per_batch`` here above: recall that we are working with a "batch of data"
# coming from our collector, which size is defined by ``frames_per_batch``, and that
# we will further split in smaller sub-batches during the inner training loop.
# The size of these sub-batches is controlled by ``sub_batch_size``.
#
sub_batch_size = 64  # cardinality of the sub-samples gathered from the current
# data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

######################################################################
# Define an environment
# ---------------------
#
# In RL, an *environment* is usually the way we refer to a simulator or a
# control system. Various libraries provide simulation environments for reinforcement
# learning, including Gymnasium (previously OpenAI Gym), DeepMind control suite, and
# many others.
# As a general library, TorchRL's goal is to provide an interchangeable interface
# to a large panel of RL simulators, allowing you to easily swap one environment
# with another. For example, creating a wrapped gym environment can be achieved
# with few characters:
#

base_env = GymEnv("InvertedDoublePendulum-v4", device=device)


######################################################################
# Load reward pre-trained network


# load trained reward network
reward_model = RewardPredictor(
    state_dim=11,  # states.shape[1],
    action_dim=1,  # actions.shape[1],
    hidden_sizes=[64, 64],
)

# Load the saved parameters from train_reward_model.py
reward_model_dir = "./trained_models/"  # path to save reward model
reward_model_filename = "reward_network"  # name of saved reward network
load_reward_model_path = os.path.join(
    *[reward_model_dir, reward_model_filename + ".pth"]
)
reward_model.load_state_dict(torch.load(load_reward_model_path))

# If you’ll be doing inference, switch to eval mode:
reward_model.eval()


######################################################################
# There are a few things to notice in this code: first, we created
# the environment by calling the ``GymEnv`` wrapper. If extra keyword arguments
# are passed, they will be transmitted to the ``gym.make`` method, hence covering
# the most common environment construction commands.
# Alternatively, one could also directly create a gym environment using
# ``gym.make(env_name, **kwargs)``
# and wrap it in a `GymWrapper` class.
#
# Also the ``device`` argument: for gym, this only controls the device where
# input action and observed states will be stored, but the execution will always
# be done on CPU. The reason for this is simply that gym does not support on-device
# execution, unless specified otherwise. For other libraries, we have control over
# the execution device and, as much as we can, we try to stay consistent in terms of
# storing and execution backends.
#
# Transforms
# ~~~~~~~~~~
#
# We will append some transforms to our environments to prepare the data for
# the policy. In Gym, this is usually achieved via wrappers. TorchRL takes a
# different
# approach, more similar to other pytorch domain libraries, through the use of
# transforms.
# To add transforms to an environment, one should simply wrap it in a
# :class:`~torchrl.envs.transforms.TransformedEnv`
# instance and append the sequence of transforms to it. The transformed
# environment will inherit
# the device and meta-data of the wrapped environment, and transform these
# depending on the sequence
# of transforms it contains.
#
# Normalization
# ~~~~~~~~~~~~~
#
# The first to encode is a normalization transform.
# As a rule of thumb, it is preferable to have data that loosely
# match a unit Gaussian distribution: to obtain this, we will
# run a certain number of random steps in the environment and compute
# the summary statistics of these observations.
#
# We'll append two other transforms: the
# :class:`~torchrl.envs.transforms.DoubleToFloat` transform will
# convert double entries to single-precision numbers, ready to be read by the
# policy. The :class:`~torchrl.envs.transforms.StepCounter` transform will be
# used to count the steps before
# the environment is terminated. We will use this measure as a supplementary
# measure
# of performance.
#
# As we will see later, many of the TorchRL's classes rely on
# :class:`~tensordict.TensorDict`
# to communicate. You could think of it as a python dictionary with some extra
# tensor features. In practice, this means that many modules we will be working
# with need to be told what key to read (``in_keys``) and what key to write
# (``out_keys``) in the ``tensordict`` they will receive. Usually, if ``out_keys``
# is omitted, it is assumed that the ``in_keys`` entries will be updated
# in-place. For our transforms, the only entry we are interested in is referred
# to as ``"observation"`` and our transform layers will be told to modify this
# entry and this entry only:
#


class RewardPredictorTransform(Transform):
    """Reward network.

    A Transform that applies a learned reward model to (observation, action) pairs
    stored in a TensorDict.
    Reads keys "observation" and "action", writes key "reward".

    Args:
        Transform (torchrl.envs.transforms.Transform):
            Base class that manages which TensorDict keys are read and written.
            Here it’s configured with in_keys=["observation", "action"] and
            out_keys=["reward"].

    """

    def __init__(
        self, model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> None:
        """Initialize.

        Args:
            model (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
                A callable (e.g., nn.Module) taking (observation, action) tensors
                and returning a 1D reward tensor of shape (batch,).
        """
        super().__init__(
            in_keys=[("next", "observation"), "action"],
            out_keys=[("next", "reward")],
        )
        self.model = model

    def _call(self, td: TensorDict) -> TensorDict:
        """Apply the reward predictor to the TensorDict.

        Args:
            td (TensorDict):
                Must contain keys "observation" and "action", each mapping to a Tensor
                of shape (batch_size, dim).

        Returns:
            TensorDict:
                The same TensorDict, with "reward" set to a Tensor of shape
                (batch_size, 1).
        """
        obs_next = td.get(("next", "observation"))
        act = td.get("action")
        if obs_next is None or act is None:
            return td

        # 2) now obs and act should be bona-fide tensors, so we can concat
        with torch.no_grad():
            pred_r = self.model(obs_next, act).unsqueeze(-1)
        td.set(("next", "reward"), pred_r)
        return td


class NNRewardWrapper(gym.Wrapper):
    """Environment wrapper.

    Environment wrapper that replaces the environment's reward with predictions
    from a pretrained reward network.

    A Gym Wrapper that intercepts the step() call, applies a learned reward model
    to each (observation, action) pair, and returns the model’s output in place
    of the environment’s original reward.

    Args:
        env (gym.Env):
            The underlying environment to wrap.
        reward_model (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
            A callable (e.g., an nn.Module) taking an observation tensor and an
            action tensor and returning a single-element tensor with the predicted
            reward.
        device (torch.device):
            The device on which to perform model inference (e.g., "cpu" or "cuda").
    """

    def __init__(
        self,
        env: gym.Env,
        reward_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device,
    ) -> None:
        """Initialize the NNRewardWrapper.

        Stores the environment, reward model, and device for inference.

        Args:
            env (gym.Env): The environment to wrap.
            reward_model (Callable): Pretrained reward network.
            device (torch.device): Device on which to run the reward model.
        """
        super().__init__(env)
        self.reward_model = reward_model
        self.device = device

    def step(self, action: gym.Space) -> tuple:
        """Take a step in the environment, then replace its reward.

        1) Execute `env.step(action)` to get (obs, original_reward, done flags, info).
        2) Convert the NumPy observation and action into torch.Tensors on the
           target device.
        3) Run the reward model in no-grad mode to predict a scalar reward.
        4) Return the same tuple as the wrapped environment, substituting in the
           network’s reward.

        Args:
            action (np.ndarray):
                The action to apply in the underlying environment.

        Returns:
            obs (np.ndarray):
                Next observation from the environment.
            nn_r (float):
                Reward predicted by the neural network.
            terminated (bool):
                Whether the episode ended by reaching a terminal state.
            truncated (bool):
                Whether the episode ended by time truncation.
            info (dict):
                Additional information from the environment.
        """
        # 1) Step the environment
        obs, _orig_r, terminated, truncated, info = self.env.step(action)

        # 2) Prepare tensors for the model: add batch dimension and move to device
        obs_t = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        act_t = torch.from_numpy(action).float().to(self.device).unsqueeze(0)

        # 3) Compute the reward with the pretrained model (no gradient tracking)
        with torch.no_grad():
            nn_r = self.reward_model(obs_t, act_t).item()

        # 4) Return everything in the original Gym step format, substituting the reward
        return obs, nn_r, terminated, truncated, info


gym_env = gym.make("InvertedDoublePendulum-v4")
nn_wrapped = NNRewardWrapper(gym_env, reward_model, device)
base_env = GymWrapper(env=nn_wrapped, device=device)

# Build and wrap
all_transforms = Compose(
    ObservationNorm(in_keys=["observation"]),
    DoubleToFloat(),
    StepCounter(),
    RewardPredictorTransform(reward_model),  # <- injects reward network
)

# Wrap everything in a TorchRL TransformedEnv
env = TransformedEnv(base_env, all_transforms)

# set normalization parameters of the environment
env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

# check for the shape of our summary stats:
print("normalization constant shape:", env.transform[0].loc.shape)

print("observation_spec:", env.observation_spec)
print("reward_spec:", env.reward_spec)
print("input_spec:", env.input_spec)
print("action_spec (as defined by input_spec):", env.action_spec)

# check environment specifications
check_env_specs(env)

# rollout the environment
rollout = env.rollout(3)
print("rollout of three steps:", rollout)
print("Shape of the rollout TensorDict:", rollout.batch_size)

######################################################################
# Policy
# ------
#
# PPO utilizes a stochastic policy to handle exploration. This means that our
# neural network will have to output the parameters of a distribution, rather
# than a single value corresponding to the action taken.
#
# As the data is continuous, we use a Tanh-Normal distribution to respect the
# action space boundaries. TorchRL provides such distribution, and the only
# thing we need to care about is to build a neural network that outputs the
# right number of parameters for the policy to work with (a location, or mean,
# and a scale):
#
# .. math::
#
#     f_{\theta}(\text{observation}) = \mu_{\theta}(\text{observation}),
#     \sigma^{+}_{\theta}(\text{observation})
#
# The only extra-difficulty that is brought up here is to split our output in
# two equal parts and map the second to a strictly positive space.
#
# We design the policy in three steps:
#
# 1. Define a neural network ``D_obs`` -> ``2 * D_action``. Indeed,
# our ``loc`` (mu) and ``scale`` (sigma) both have dimension ``D_action``.
#
# 2. Append a :class:`~tensordict.nn.distributions.NormalParamExtractor` to
# extract a location and a scale (for example, splits the input in two equal
# parts and applies a positive transformation to the scale parameter).
#
# 3. Create a probabilistic :class:`~tensordict.nn.TensorDictModule` that can
# generate this distribution and sample from it.
#

actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
    NormalParamExtractor(),
)

######################################################################
# To enable the policy to "talk" with the environment through the ``tensordict``
# data carrier, we wrap the ``nn.Module`` in a
# :class:`~tensordict.nn.TensorDictModule`. This
# class will simply ready the ``in_keys`` it is provided with and write the
# outputs in-place at the registered ``out_keys``.
#
policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
)

######################################################################
# We now need to build a distribution out of the location and scale of our
# normal distribution. To do so, we instruct the
# :class:`~torchrl.modules.tensordict_module.ProbabilisticActor`
# class to build a :class:`~torchrl.modules.TanhNormal` out of the location and
# scale
# parameters. We also provide the minimum and maximum values of this
# distribution, which we gather from the environment specs.
#
# The name of the ``in_keys`` (and hence the name of the ``out_keys`` from
# the :class:`~tensordict.nn.TensorDictModule` above) cannot be set to any value
# one may
# like, as the :class:`~torchrl.modules.TanhNormal` distribution constructor
# will expect the
# ``loc`` and ``scale`` keyword arguments. That being said,
# :class:`~torchrl.modules.tensordict_module.ProbabilisticActor` also accepts
# ``Dict[str, str]`` typed ``in_keys`` where the key-value pair indicates
# what ``in_key`` string should be used for every keyword argument that is to
# be used.
#
policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": env.action_spec.space.low,
        "high": env.action_spec.space.high,
    },
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
)

######################################################################
# Value network
# -------------
#
# The value network is a crucial component of the PPO algorithm, even though it
# won't be used at inference time. This module will read the observations and
# return an estimation of the discounted return for the following trajectory.
# This allows us to amortize learning by relying on the some utility estimation
# that is learned on-the-fly during training. Our value network share the same
# structure as the policy, but for simplicity we assign it its own set of
# parameters.
#
value_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(1, device=device),
)

value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)

######################################################################
# let's try our policy and value modules. As we said earlier, the usage of
# :class:`~tensordict.nn.TensorDictModule` makes it possible to directly
# read the output
# of the environment to run these modules, as they know what information to read
# and where to write it:
#
print("Running policy:", policy_module(env.reset()))
print("Running value:", value_module(env.reset()))

######################################################################
# Data collector
# --------------
#
# TorchRL provides a set of `DataCollector classes
# <https://pytorch.org/rl/reference/collectors.html>`__.
# Briefly, these classes execute three operations: reset an environment,
# compute an action given the latest observation, execute a step in the environment,
# and repeat the last two steps until the environment signals a stop (or reaches
# a done state).
#
# They allow you to control how many frames to collect at each iteration
# (through the ``frames_per_batch`` parameter),
# when to reset the environment (through the ``max_frames_per_traj`` argument),
# on which ``device`` the policy should be executed, etc. They are also
# designed to work efficiently with batched and multiprocessed environments.
#
# The simplest data collector is the
# :class:`~torchrl.collectors.collectors.SyncDataCollector`:
# it is an iterator that you can use to get batches of data of a given length, and
# that will stop once a total number of frames (``total_frames``) have been
# collected.
# Other data collectors (:class:`~torchrl.collectors.collectors.MultiSyncDataCollector`
# and
# :class:`~torchrl.collectors.collectors.MultiaSyncDataCollector`) will execute
# the same operations in synchronous and asynchronous manner over a
# set of multiprocessed workers.
#
# As for the policy and environment before, the data collector will return
# :class:`~tensordict.TensorDict` instances with a total number of elements that will
# match ``frames_per_batch``. Using :class:`~tensordict.TensorDict` to pass data to the
# training loop allows you to write data loading pipelines
# that are 100% oblivious to the actual specificities of the rollout content.
#
collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

######################################################################
# Replay buffer
# -------------
#
# Replay buffers are a common building piece of off-policy RL algorithms.
# In on-policy contexts, a replay buffer is refilled every time a batch of
# data is collected, and its data is repeatedly consumed for a certain number
# of epochs.
#
# TorchRL's replay buffers are built using a common container
# :class:`~torchrl.data.ReplayBuffer` which takes as argument the components
# of the buffer: a storage, a writer, a sampler and possibly some transforms.
# Only the storage (which indicates the replay buffer capacity) is mandatory.
# We also specify a sampler without repetition to avoid sampling multiple times
# the same item in one epoch.
# Using a replay buffer for PPO is not mandatory and we could simply
# sample the sub-batches from the collected batch, but using these classes
# make it easy for us to build the inner training loop in a reproducible way.
#

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

######################################################################
# Loss function
# -------------
#
# The PPO loss can be directly imported from TorchRL for convenience using the
# :class:`~torchrl.objectives.ClipPPOLoss` class. This is the easiest way of
# utilizing PPO:
# it hides away the mathematical operations of PPO and the control flow that
# goes with it.
#
# PPO requires some "advantage estimation" to be computed. In short, an advantage
# is a value that reflects an expectancy over the return value while dealing with
# the bias / variance tradeoff.
# To compute the advantage, one just needs to (1) build the advantage module, which
# utilizes our value operator, and (2) pass each batch of data through it before each
# epoch.
# The GAE module will update the input ``tensordict`` with new ``"advantage"`` and
# ``"value_target"`` entries.
# The ``"value_target"`` is a gradient-free tensor that represents the empirical
# value that the value network should represent with the input observation.
# Both of these will be used by :class:`~torchrl.objectives.ClipPPOLoss` to
# return the policy and value losses.
#

advantage_module = GAE(
    gamma=gamma,
    lmbda=lmbda,
    value_network=value_module,
    average_gae=True,
    device=device,
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)

######################################################################
# Training loop
# -------------
# We now have all the pieces needed to code our training loop.
# The steps include:
#
# * Collect data
#
#   * Compute advantage
#
#     * Loop over the collected to compute loss values
#     * Back propagate
#     * Optimize
#     * Repeat
#
#   * Repeat
#
# * Repeat
#


logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
for i, tensordict_data in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.
    for _ in range(num_epochs):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # Optimization: backward, grad clipping and optimization step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    if i % 10 == 0:
        # We evaluate the policy once every 10 batches of data.
        # Evaluation is rather simple: execute the policy without exploration
        # (take the expected value of the action distribution) for a given
        # number of steps (1000, which is our ``env`` horizon).
        # The ``rollout`` method of the ``env`` can take a policy as argument:
        # it will then execute this policy at each step.
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = env.rollout(1000, policy_module)
            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(eval_rollout["next", "reward"].sum().item())
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {logs['eval step_count'][-1]}"
            )
            del eval_rollout
    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()

######################################################################
# Results
# -------
#
# Before the 1M step cap is reached, the algorithm should have reached a max
# step count of 1000 steps, which is the maximum number of steps before the
# trajectory is truncated.
#
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.plot(logs["reward"])
plt.title("training rewards (average)")
plt.subplot(2, 2, 2)
plt.plot(logs["step_count"])
plt.title("Max step count (training)")
plt.subplot(2, 2, 3)
plt.plot(logs["eval reward (sum)"])
plt.title("Return (test)")
plt.subplot(2, 2, 4)
plt.plot(logs["eval step_count"])
plt.title("Max step count (test)")
plt.show()
