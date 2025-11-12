import numpy as np
from typing import Optional
import torch
import h5py

def mse_2d(y_true, y_pred, *, mask: Optional[np.ndarray] = None, nan_safe: bool = False) -> float:
    """
    Compute the Mean Squared Error (MSE) between two arrays of 2D field values.

    Parameters
    ----------
    y_true, y_pred : array-like
        Arrays with identical shape (e.g., (H, W) or (T, H, W)).
    mask : np.ndarray, optional
        Boolean array of the same shape as inputs. True values are INCLUDED in the MSE;
        False values are ignored. If provided, NaNs at masked-in points are still handled
        according to `nan_safe`.
    nan_safe : bool, default False
        If True, ignores NaNs using `np.nanmean`. If False, any NaN will propagate.

    Returns
    -------
    float
        The mean squared error over all included elements.
    """
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Shapes must match, got {a.shape} and {b.shape}")

    diff2 = (a - b) ** 2

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != a.shape:
            raise ValueError(f"Mask shape must match inputs, got {m.shape} vs {a.shape}")
        # Only keep masked-in elements
        diff2 = diff2[m]

    if nan_safe:
        return float(np.nanmean(diff2))
    else:
        return float(np.mean(diff2))


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