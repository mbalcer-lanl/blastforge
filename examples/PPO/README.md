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

This example trains a Reinforcement Learning (RL) model using the Proximal Policy Optimization (PPO) algorithm. 
A surrogate model is trained
A value network is pretrained to learn the value from the environment and used to initialize the weights of the value network in the PPO algorithm.

---

## Prerequisites

Assuming Blast Forge is installed, this example requires gymnasium[mujoco] to be installed, with:

```bash
pip install gymnasium[mujoco]
```

---

## Usage

1. Train the surrogate model that emulates the environment:

```bash
python train_env_surrogate.py
```

This trains the reward network and saves it in `env_model_dir` with the file name `env_model_filename`.


2. Pretrain the value model first by:

```bash
python train_value_model.py
```

This trains the reward network and saves it in `value_model_dir` with the file name `value_model_filename`.


3. Train the policy network following the PPO algorithm by running:

```bash
python main.py
```
