"""Test MPS torch.

Also, make sure torchRL and Gymnasium are installed.

"""

import torch
import torchrl
import gymnasium


print('Torch version:', torch.__version__)
print('Torch-RL version:', torchrl.__version__)
print('Gymnasium version:', gymnasium.__version__)

# Check if MPS is available
if torch.backends.mps.is_available():
    print('MPS backend available!')

    # Test a tensor operation
    x = torch.randn(3, 3, device='mps')
    print('Tensor created on MPS:', x)
else:
    print('MPS backend NOT available!')
