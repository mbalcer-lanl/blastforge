# PPO of the InvertedDoublePendulum-v4 Environment using TorchRL

This example is based on the TorchRL PPO tutorial (https://docs.pytorch.org/tutorials/intermediate/reinforcement_ppo.html). A reward network was added to this tutorial, which learns the reward from a pretraining script.

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Usage](#usage)  

---

## Overview

This example trains a Reinforcement Learning (RL) model using the Proximal Policy Optimization (PPO) algorithm. A reward network is pretrained to learn the reward from the environment and used when training the policy network in the PPO algorithm.

---

## Prerequisites

Assuming Blast Forge is installed, this example requires gymnasium[mujoco] to be installed, with:

```bash
pip install gymnasium[mujoco]
```

---

## Usage

Pretrain the reward model first by:

```bash
python train_reward_model.py
```

This trains the reward network and saves it in `reward_model_dir` with the file name `reward_model_filename`.

Train the policy network following the PPO algorithm by running:

```bash
python run.py
```
