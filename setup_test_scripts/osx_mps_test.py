"""Test MPS torch."""

import torch


# Check if MPS is available
if torch.backends.mps.is_available():
    print('MPS backend available!')

    # Test a tensor operation
    x = torch.randn(3, 3, device='mps')
    print('Tensor created on MPS:', x)
else:
    print('MPS backend NOT available!')
