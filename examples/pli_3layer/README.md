# Design of Perturbed Liner Interface (PLI) using PPO

---

## Table of Contents

- [Overview](#overview)  
- [Prerequisites](#prerequisites)  
- [Usage](#usage)  

---

## Overview

The user supplies a "target" shape of a copper jet at $$t=25$$ $$\mu\text{s}$$. The RL algorithm will find an initial geometry of the 3-layered liner. The 3 layers of the liner consist of a polymer, aluminum, and copper, where the polymer touches the HE. An ensemble of 2D axisymmetric PLI simulations were used to train a NN emulator. This emulator is called during the RL algorithm as the environment.

---

## Prerequisites

Installation of Blastforge.

---

## Usage

1. cd to examples/pli_3layer directory.

2. In main.py, supply the target density field of the copper jet at $$t=25$$ $$\mu\text{s}$$, in the variable "cfg.target" or supply a .npy file with the "--target" argurment from the command line. The target should be size (1120, 800) pixels, which defines the entire density field of copper at $$t=25$$ $$\mu$$s. The target supplied in this example is a rectangle with values of 8.93 (density of copper at initial conditions) and values of zero elsewhere.

3. In main.py, to train the model, the variable "run_train" should be set to True. Otherwise it will plot the results of the final iteration.

4. Run the algorithm with:

```bash
python main.py
```
