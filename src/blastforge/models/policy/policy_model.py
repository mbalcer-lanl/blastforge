"""Probabilistic CNN modules for RL policy networks."""
import math

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributions import MultivariateNormal

# from yoke.utils.parameters import count_torch_params
# 
# from yoke.models.CNNmodules import CNN_Interpretability_Module
# from yoke.models.CNNmodules import CNN_Reduction_Module
# from yoke.models.CNNmodules import Image2VectorCNN
# from yoke.models.cnn_utils import generalMLP


class gaussian_policyCNN(nn.Module):
    """Vector-and-Image to Gaussian distribution.

    Convolutional Neural Network Module that maps a triple (y, H1, H2) to a
    Gaussian distribution, N(x, C). Here, y is a 1D-tensor, H1 and H2 are
    2D-tensors. The mean x is a 1D-tensor and C is a 2D-tensor satisfying the
    symmetry and positive-definite properties of a covariance.

    Each input is first processed through an independent branch before
    concatenation to two forks of dense networks to estimate the mean and
    covariance.

    Args:
        img_size (tuple[int, int, int]): (C, H, W) dimensions of H1 and H2.
        input_vector_size (int): Size of input vector
        output_dim (int): Dimension of Guassian mean.
        min_variance (float): Minimum variance in diagonal covariance entries.
        features (int): Number of output channels/features for each convolutional layer.
        depth (int): Number of convolutional layers in each image processing branch.
        kernel (int): Size of symmetric convolutional kernels
        img_embed_dim (int): Number of features in MLP output from image embeddings.
        vector_embed_dim (int): Number of features in MLP output from image embeddings.
        vector_feature_list (tuple[int, ...]): Number of features in each hidden layer
                                               of vector-MLP.
        output_feature_list (tuple[int, ...]): Number of features in each hidden layer
                                               of final/output-MLP.
        act_layer (nn.Module): torch neural network layer class to use as activation
        norm_layer (nn.Module): torch neural network layer class to use as normalization
                                between MLP layers.

    """

    def __init__(
        self,
        img_size: tuple[int, int, int] = (1, 1120, 400),
        input_vector_size: int = 28,
        output_dim: int = 28,
        min_variance: float = 1e-6,
        features: int = 12,
        depth: int = 12,
        kernel: int = 3,
        img_embed_dim: int = 32,
        vector_embed_dim: int = 32,
        size_reduce_threshold: tuple[int, int] = (8, 8),
        vector_feature_list: tuple[int, ...] = (32, 32, 64, 64),
        output_feature_list: tuple[int, ...] = (64, 128, 128, 64),
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        """Initialization for hybrid CNN."""
        super().__init__()

        self.img_size = img_size
        _, H, W = self.img_size
        self.kernel = kernel
        self.features = features
        self.img_embed_dim = img_embed_dim
        self.vector_embed_dim = vector_embed_dim
        self.vector_feature_list = vector_feature_list
        self.output_feature_list = output_feature_list
        self.depth = depth
        self.size_reduce_threshold = size_reduce_threshold
        self.input_vector_size = input_vector_size
        self.output_dim = output_dim
        self.min_variance = min_variance
        self.act_layer = act_layer
        self.norm_layer = norm_layer

        # CNN processing branch for H1
        self.interpH1 = CNN_Interpretability_Module(
            img_size=self.img_size,
            kernel=self.kernel,
            features=self.features,
            depth=self.depth,
            conv_onlyweights=True,
            batchnorm_onlybias=True,
            act_layer=self.act_layer,
        )

        self.reduceH1 = CNN_Reduction_Module(
            img_size=(self.features, H, W),
            size_threshold=self.size_reduce_threshold,
            kernel=self.kernel,
            stride=2,
            features=self.features,
            conv_onlyweights=True,
            batchnorm_onlybias=True,
            act_layer=self.act_layer,
        )

        self.finalW_h1 = self.reduceH1.finalW
        self.finalH_h1 = self.reduceH1.finalH

        # Linear embedding H1
        self.lin_embed_h1 = generalMLP(
            input_dim=self.finalH_h1 * self.finalW_h1 * self.features,
            output_dim=self.img_embed_dim,
            hidden_feature_list=(2 * self.img_embed_dim,),
            act_layer=self.act_layer,
            norm_layer=self.norm_layer,
        )

        # Image embed will end with a GELU activation
        self.h1_embed_act = self.act_layer()

        # CNN processing branch for H2
        self.interpH2 = CNN_Interpretability_Module(
            img_size=self.img_size,
            kernel=self.kernel,
            features=self.features,
            depth=self.depth,
            conv_onlyweights=True,
            batchnorm_onlybias=True,
            act_layer=self.act_layer,
        )

        self.reduceH2 = CNN_Reduction_Module(
            img_size=(self.features, H, W),
            size_threshold=self.size_reduce_threshold,
            kernel=self.kernel,
            stride=2,
            features=self.features,
            conv_onlyweights=True,
            batchnorm_onlybias=True,
            act_layer=self.act_layer,
        )

        self.finalW_h2 = self.reduceH2.finalW
        self.finalH_h2 = self.reduceH2.finalH

        # Linear embedding H2
        self.lin_embed_h2 = generalMLP(
            input_dim=self.finalH_h2 * self.finalW_h2 * self.features,
            output_dim=self.img_embed_dim,
            hidden_feature_list=(2 * self.img_embed_dim,),
            act_layer=self.act_layer,
            norm_layer=self.norm_layer,
        )

        # Image embed will end with a GELU activation
        self.h2_embed_act = self.act_layer()

        # MLP for processing vector input
        self.vector_mlp = generalMLP(
            input_dim=self.input_vector_size,
            output_dim=self.vector_embed_dim,
            hidden_feature_list=self.vector_feature_list,
            act_layer=self.act_layer,
            norm_layer=self.norm_layer,
        )

        self.vector_embed_act = self.act_layer()

        # Mean MLP
        #
        # NOTE: Final activation is just identity.
        cat_size = self.vector_embed_dim + 2 * self.img_embed_dim
        self.mean_mlp = generalMLP(
            input_dim=cat_size,
            output_dim=self.output_dim,
            hidden_feature_list=self.output_feature_list,
            act_layer=self.act_layer,
            norm_layer=nn.Identity,
        )

        # Covariance MLP
        self.num_cov_elements = self.output_dim * (self.output_dim + 1) // 2
        self.cov_mlp = generalMLP(
            input_dim=cat_size,
            output_dim=self.num_cov_elements,
            hidden_feature_list=self.output_feature_list,
            act_layer=self.act_layer,
            norm_layer=nn.Identity,
        )
        self._init_cov_mlp(self.cov_mlp, self.output_dim, self.min_variance)

    def _init_cov_mlp(
        self, mlp: nn.Module, output_dim: int, min_var: float = 1e-6
    ) -> None:
        """Initialize covariance layer to output identity.

        Initializes the final layer of the MLP such that the predicted Cholesky
        factor L results in a covariance matrix close to identity.

        Args:
            mlp (nn.Module): Multi-layer perceptron module. Must have last layer linear.
            output_dim (int): Dimension of network output.
            min_var (float): Minimum variance on covariance diagonal

        """
        tril_indices = torch.tril_indices(output_dim, output_dim)

        # Find the last linear layer (which has no activation)
        last_layer = mlp.LayerList[-1][0]  # Should be the "linear" layer
        err_msg = "Expected final MLP layer to be nn.Linear"
        assert isinstance(last_layer, nn.Linear), err_msg

        with torch.no_grad():
            # Start with all zeros
            last_layer.weight.zero_()
            last_layer.bias.zero_()

            # Set diagonal entries of L such that softplus(bias) + min_var ~ 1
            target_diag = 1.0 - min_var
            # softplus inverse
            init_bias_val = torch.log(torch.exp(torch.tensor(target_diag)) - 1.0)
            init_bias_val = init_bias_val.item()

            for idx, (row, col) in enumerate(zip(tril_indices[0], tril_indices[1])):
                if row == col:
                    last_layer.bias[idx] = init_bias_val  # Diagonal element
                else:
                    last_layer.bias[idx] = 0.0  # Off-diagonal

    def forward(
        self,
        y: torch.Tensor,
        h1: torch.Tensor,
        h2: torch.Tensor,
    ) -> torch.Tensor:
        """Forward method for hybrid CNN."""
        # Process first image
        h1_out = self.interpH1(h1)
        h1_out = self.reduceH1(h1_out)
        h1_out = torch.flatten(h1_out, start_dim=1)
        h1_out = self.lin_embed_h1(h1_out)
        h1_out = self.h1_embed_act(h1_out)

        # Process second image
        h2_out = self.interpH2(h2)
        h2_out = self.reduceH2(h2_out)
        h2_out = torch.flatten(h2_out, start_dim=1)
        h2_out = self.lin_embed_h2(h2_out)
        h2_out = self.h2_embed_act(h2_out)

        # Process vector
        y_out = self.vector_mlp(y)
        y_out = self.vector_embed_act(y_out)

        # Concatenate outputs and send on to mean and covariance prediction.
        cat = torch.cat((y_out, h1_out, h2_out), dim=1)

        # Predict the mean.
        mean = self.mean_mlp(cat)

        # Predict the covariance
        L_params = self.cov_mlp(cat)

        # Use Cholesky decomposition to ensure positive definite covariance.
        triIDXs = torch.tril_indices(self.output_dim, self.output_dim)
        L = torch.zeros(y.size(0), self.output_dim, self.output_dim, device=y.device)
        L[:, triIDXs[0], triIDXs[1]] = L_params

        # Ensure positive diagonal with softplus(z) = log(1+exp(z))
        diagIDXs = torch.arange(self.output_dim)
        L[:, diagIDXs, diagIDXs] = nn.functional.softplus(L[:, diagIDXs, diagIDXs])

        # Ensure a minimum variance
        L[:, diagIDXs, diagIDXs] += self.min_variance

        # Reconstruct covariance matrix assuming Cholesky factorization.
        cov_matrix = torch.matmul(L, L.transpose(-1, -2))

        # Return a MultivariateNormal distribution
        return MultivariateNormal(mean, covariance_matrix=cov_matrix)


class gaussian_Image2VectorCNN(nn.Module):
    """Probabilistic image-to-vector CNN.

    Convolutional Neural Network Model that maps an image to a multivariate normal
    distribution. Constructed using a deterministic image-to-vector CNN fed into two MLP
    structures, one to estimate the mean and a second MLP to estimate the covariance of
    the multivariate normal.

    Args:
        img_size (tuple[int, int, int]): size of input (channels, height, width)
        output_dim (int): size of output vector
        min_variance (float): Minimum variance in diagonal covariance entries.
        size_threshold (tuple[int, int]): (approximate) size of reduced image
                                          (height, width)
        kernel (int): size of square convolutional kernel
        features (int): number of features in the convolutional layers
        interp_depth (int): number of interpretability blocks
        conv_onlyweights (bool): determines if convolutional layers learn only
                                 weights or weights and bias
        batchnorm_onlybias (bool): determines if the batch normalization layers
                                   learn only bias or weights and bias
        act_layer (nn.modules.activation): torch neural network layer class to
                                           use as activation
        norm_layer (nn.modules.normalization): torch neural network layer class
        hidden_features (int): number of hidden features in the fully connected
                               dense layer
        final_activation (nn.modules.activation): torch neural network layer class to use
                                                  for the final output.

    """

    def __init__(
        self,
        img_size: tuple[int, int, int] = (1, 1120, 400),
        output_dim: int = 29,
        min_variance: float = 1e-6,
        size_threshold: tuple[int, int] = (12, 12),
        kernel: int = 3,
        features: int = 16,
        interp_depth: int = 12,
        conv_onlyweights: bool = True,
        batchnorm_onlybias: bool = True,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        hidden_features: int = 32,
        final_activation: nn.Module = nn.Identity,
    ) -> None:
        """Initialization for probabilistic image-to-vector CNN."""
        super().__init__()
        self.min_variance = min_variance

        # Main CNN branch
        self.i2v_cnn = Image2VectorCNN(img_size=img_size,
                                       output_dim=output_dim,
                                       size_threshold=size_threshold,
                                       kernel=kernel,
                                       features=features,
                                       interp_depth=interp_depth,
                                       conv_onlyweights=conv_onlyweights,
                                       batchnorm_onlybias=batchnorm_onlybias,
                                       act_layer=act_layer,
                                       norm_layer=norm_layer,
                                       hidden_features=hidden_features,
                                       final_activation=final_activation,
                                       return_hidden=True,
                                       )

        # Covariance MLP
        self.num_cov_elements = output_dim * (output_dim + 1) // 2
        self.cov_mlp = generalMLP(
            input_dim=self.i2v_cnn.hidden_features,
            output_dim=self.num_cov_elements,
            hidden_feature_list=(2 * self.i2v_cnn.hidden_features,),  # Could expand.
            act_layer=act_layer,
            norm_layer=nn.Identity,
        )
        self._init_cov_mlp(self.cov_mlp, self.i2v_cnn.output_dim, self.min_variance)

    def _init_cov_mlp(
        self, mlp: nn.Module, output_dim: int, min_var: float = 1e-6
    ) -> None:
        """Initialize covariance layer to output identity.

        Initializes the final layer of the MLP such that the predicted Cholesky
        factor L results in a covariance matrix close to identity.

        Args:
            mlp (nn.Module): Multi-layer perceptron module. Must have last layer linear.
            output_dim (int): Dimension of network output.
            min_var (float): Minimum variance on covariance diagonal

        """
        tril_indices = torch.tril_indices(output_dim, output_dim)

        # Find the last linear layer (which has no activation)
        last_layer = mlp.LayerList[-1][0]  # Should be the "linear" layer
        err_msg = "Expected final MLP layer to be nn.Linear"
        assert isinstance(last_layer, nn.Linear), err_msg

        with torch.no_grad():
            # Start with all zeros
            last_layer.weight.zero_()
            last_layer.bias.zero_()

            # Set diagonal entries of L such that softplus(bias) + min_var ~ 1
            target_diag = 1.0 - min_var
            # softplus inverse
            init_bias_val = torch.log(torch.exp(torch.tensor(target_diag)) - 1.0)
            init_bias_val = init_bias_val.item()

            for idx, (row, col) in enumerate(zip(tril_indices[0], tril_indices[1])):
                if row == col:
                    last_layer.bias[idx] = init_bias_val  # Diagonal element
                else:
                    last_layer.bias[idx] = 0.0  # Off-diagonal

    def forward(
        self,
        h1: torch.Tensor, # target image
    ) -> torch.Tensor:
    
        # ---- normalize input shape to (N, C, H, W) ----
        # try to infer expected channel count if model exposes img_size
        expected_C = None
        if hasattr(self.i2v_cnn, "img_size"):
            # img_size often stored as (C, H, W)
            try:
                expected_C = int(self.i2v_cnn.img_size[0])
            except Exception:
                expected_C = None
# 
        # ensure tensor
        if not torch.is_tensor(h1):
            raise TypeError("h1 must be a torch.Tensor")
# 
        # now canonicalize shapes
        if h1.ndim == 2:
            # (H, W) -> (1, 1, H, W)
            h1 = h1.unsqueeze(0).unsqueeze(0)
        elif h1.ndim == 3:
            # Could be (C, H, W)  OR  (N, H, W)
            if expected_C is not None and h1.shape[0] == expected_C:
                # (C, H, W) -> (1, C, H, W)
                h1 = h1.unsqueeze(0)
            else:
                # treat as (N, H, W) -> (N, 1, H, W)
                h1 = h1.unsqueeze(1)
        elif h1.ndim == 4:
            # could be (N, C, H, W) or (N, H, W, C)
            if expected_C is not None and h1.shape[-1] == expected_C and h1.shape[1] not in (1, 3):
                # (N, H, W, C) -> (N, C, H, W)
                h1 = h1.permute(0, 3, 1, 2)
            # otherwise assume (N, C, H, W) already correct
        else:
            raise ValueError(f"unexpected input dims for h1: {tuple(h1.shape)}")
# 
        # optional: sanity-check channels match model expectation
        if expected_C is not None and h1.shape[1] != expected_C:
            # don't crash — just warn (or raise if you prefer strictness)
            import warnings
            warnings.warn(f"input channels ({h1.shape[1]}) != model expected ({expected_C})")

    
        """Forward method for probabilistic Image-to-Vector CNN."""
        # Predict the mean.
        mean_out, hidden_out = self.i2v_cnn(h1)
        L_params = self.cov_mlp(hidden_out)

        triIDXs = torch.tril_indices(self.i2v_cnn.output_dim, self.i2v_cnn.output_dim)
        N = h1.size(0)
        D = self.i2v_cnn.output_dim

        # Build lower-triangular factor
        L = torch.zeros(N, D, D, device=h1.device, dtype=mean_out.dtype)
        L[:, triIDXs[0], triIDXs[1]] = L_params

        # Force strictly-positive diagonal (and keep it away from 0)
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        diag = nn.functional.softplus(diag) + self.min_variance  # strictly > 0
        L = L.clone()
        L[:, torch.arange(D), torch.arange(D)] = diag

        # Extra safety: clamp diagonal to avoid tiny/huge values early in training
        # (tune these if you want)
        min_diag = 1e-4
        max_diag = 10.0
        L[:, torch.arange(D), torch.arange(D)] = torch.clamp(
            L[:, torch.arange(D), torch.arange(D)], min=min_diag, max=max_diag
        )

        # Last-resort numerical stabilization: add tiny jitter to diagonal
        # (this changes L slightly, but keeps PD)
        jitter = 1e-6
        L[:, torch.arange(D), torch.arange(D)] = L[:, torch.arange(D), torch.arange(D)] + jitter

        # IMPORTANT: return MVN using scale_tril so it doesn't do cholesky on covariance_matrix
        return MultivariateNormal(mean_out, scale_tril=L)


class DifferenceMVN(nn.Module):
    """Difference of multi-variate normal prediction nets.

    Combine two trained gaussian_Image2VectorCNN models to predict the MVN
    of the difference (vec1 - vec2) given two images.

    Assumes independence between the two predictive distributions, so
    Cov(Y1 - Y2) = Cov(Y1) + Cov(Y2).

    Args:
        mvn1_args (dict): kwarg dictionary to initialize `gaussian_Image2VectorCNN`.
        mvn2_args (dict): kwarg dictionary to initialize `gaussian_Image2VectorCNN`.
        freeze (bool): Flag to enable fine tuning.
    """

    def __init__(
            self,
            mvn1_args: dict,
            mvn2_args: dict,
            freeze: bool = False,
            ) -> None:
        """Initialization with two networks."""
        super().__init__()
        self.mvn1 = gaussian_Image2VectorCNN(**mvn1_args)
        self.mvn2 = gaussian_Image2VectorCNN(**mvn2_args)

        # Sanity check on output dims (optional but helpful)
        try:
            self.out_dim1 = self.mvn1.i2v_cnn.output_dim
            self.out_dim2 = self.mvn2.i2v_cnn.output_dim
        except AttributeError:
            err_msg = ('Expected gaussian_Image2VectorCNN-like modules with '
                       'i2v_cnn.output_dim.')
            raise ValueError(err_msg)

        if self.out_dim1 != self.out_dim2:
            err_msg = f"Output dims must match, got {self.out_dim1} vs {self.out_dim2}"
            raise ValueError(err_msg)

        if freeze:
            for p in self.mvn1.parameters():
                p.requires_grad = False
            for p in self.mvn2.parameters():
                p.requires_grad = False
            self.mvn1.eval()
            self.mvn2.eval()

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> MultivariateNormal:
        """Difference MVN forward method."""
        # Each net returns a MultivariateNormal
        dist1: MultivariateNormal = self.mvn1(img1)
        dist2: MultivariateNormal = self.mvn2(img2)

        # Means: (B, D)
        mu1 = dist1.mean
        mu2 = dist2.mean

        # Covariances: (B, D, D)
        # Prefer covariance_matrix because it's guaranteed PSD from your construction.
        # If only scale_tril is available, you can reconstruct via L @ L^T.
        if dist1.covariance_matrix is None:
            L1 = dist1.scale_tril
            Sigma1 = L1 @ L1.transpose(-1, -2)
        else:
            Sigma1 = dist1.covariance_matrix

        if dist2.covariance_matrix is None:
            L2 = dist2.scale_tril
            Sigma2 = L2 @ L2.transpose(-1, -2)
        else:
            Sigma2 = dist2.covariance_matrix

        # Difference distribution parameters
        mu_delta = mu1 - mu2
        Sigma_delta = Sigma1 + Sigma2  # assumes independence

        # Return an MVN for the difference
        return MultivariateNormal(loc=mu_delta, covariance_matrix=Sigma_delta)

    @classmethod
    def from_two_gaussian_checkpoints(
        cls,
        mvn1_ckpt: str,
        mvn2_ckpt: str,
        device: str = "cuda",
        ) -> nn.Module:
        """Instantiation method from two network checkpoints.

        Args:
            mvn1_ckpt (str): Filename of checkpoint to load for mvn1
            mvn2_ckpt (str): Filename of checkpoint to load for mvn2
            device (str): Device name to load objects to
        """
        # Load args + weights for each subnet (DDP-safe)
        mvn1_args, mvn1_sd = load_gaussian_ckpt_light(mvn1_ckpt)
        mvn2_args, mvn2_sd = load_gaussian_ckpt_light(mvn2_ckpt)

        # Build composite
        model = cls(mvn1_args=mvn1_args, mvn2_args=mvn2_args)
        model.mvn1.load_state_dict(mvn1_sd, strict=True)
        model.mvn2.load_state_dict(mvn2_sd, strict=True)
        model.to(device)

        return model


def load_gaussian_ckpt_light(ckpt_path: str) -> tuple[dict, dict]:
    """Light loader specifically for initialization of difference MVN.

    Load a `gaussian_Image2VectorCNN` checkpoint written by Yoke checkpointer,
    but only return (model_args, model_state_dict). DDP-safe via broadcast.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    ckpt = None

    if rank == 0:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt["model_class"] != "gaussian_Image2VectorCNN":
            verr = f"Expected gaussian_Image2VectorCNN, got {ckpt['model_class']}"
            raise ValueError(verr)

    if dist.is_initialized():
        lst = [ckpt]
        dist.broadcast_object_list(lst, src=0)
        ckpt = lst[0]

    model_args = ckpt["model_args"]
    model_state_dict = ckpt["model_state_dict"]

    return model_args, model_state_dict




####################################
# Interpretability Module
####################################
class CNN_Interpretability_Module(nn.Module):
    """Interpretability module.

    Convolutional Neural Network Module that creates the "interpretability
    layers" Sequence of Conv2D, Batch Normalization, and Activation. The key
    idea is to keep the size of the image approximately equal throughout the
    network.

    Args:
        img_size (tuple[int, int, int]): size of input (channels, height, width)
        kernel (int): size of square convolutional kernel
        features (int): number of features in the convolutional layers
        depth (int): number of interpretability blocks
        conv_onlyweights (bool): determines if convolutional layers learn
                                 only weights or weights and bias
        batchnorm_onlybias (bool): determines if the batch normalization
                                   layers learn only bias or weights and bias
        act_layer(nn.modules.activation): torch neural network layer class
                                          to use as activation

    """

    def __init__(
        self,
        img_size: tuple[int, int, int] = (1, 1700, 500),
        kernel: int = 5,
        features: int = 12,
        depth: int = 12,
        conv_onlyweights: bool = True,
        batchnorm_onlybias: bool = True,
        act_layer: nn.Module = nn.GELU,
    ) -> None:
        """Initialization for interpretability CNN."""
        super().__init__()

        self.img_size = img_size
        C, _, _ = self.img_size
        self.kernel = kernel
        self.features = features
        self.depth = depth
        self.conv_weights = True
        self.conv_bias = not conv_onlyweights
        self.batchnorm_weights = not batchnorm_onlybias
        self.batchnorm_bias = True

        # Input Layers
        self.inConv = nn.Conv2d(
            in_channels=C,
            out_channels=self.features,
            kernel_size=self.kernel,
            stride=1,
            padding="same",  # pads the input so the output
            # has the shape as the input,
            # stride=1 only
            bias=self.conv_bias,
        )

        normLayer = nn.BatchNorm2d(features)

        if not self.batchnorm_weights:
            nn.init.constant_(normLayer.weight, 1)
            normLayer.weight.requires_grad = False

        self.inNorm = normLayer
        self.inActivation = act_layer()

        # Module list to hold interpretability layers
        self.InterpConvList = nn.ModuleList()
        for i in range(self.depth - 1):
            interpLayer = nn.Conv2d(
                in_channels=self.features,
                out_channels=self.features,
                kernel_size=self.kernel,
                stride=1,
                padding="same",  # pads the input so the
                # output has the shape of
                # the input, stride=1 only
                bias=self.conv_bias,
            )

            normLayer = nn.BatchNorm2d(features)

            # Necessary step to turn off only the scaling.
            if not self.batchnorm_weights:
                nn.init.constant_(normLayer.weight, 1)
                normLayer.weight.requires_grad = False

            # Make list of small sequential modules. Then we'll use enumerate
            # in forward method.
            interp_dict = OrderedDict(
                [
                    (f"interp{i:02d}", interpLayer),
                    (f"bnorm{i:02d}", normLayer),
                    (f"act{i:02d}", act_layer()),
                ]
            )
            self.InterpConvList.append(nn.Sequential(interp_dict))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for interpretable CNN."""
        # Input Layers
        x = self.inConv(x)
        x = self.inNorm(x)
        x = self.inActivation(x)

        # Interpretability Layers
        for i, interp_conv in enumerate(self.InterpConvList):
            x = interp_conv(x)

        return x


####################################
# Reduction Module
####################################
class CNN_Reduction_Module(nn.Module):
    """Reduction CNN.

    Convolutional Neural Network Module that creates the "reduction layers"
    Sequence of Conv2D, Batch Normalization, and Activation. Key idea is to
    halve the image size at each layer using double-strided convolutions.

    Args:
        img_size (tuple[int, int, int]): size of input
                                         (channels, height, width)
        size_threshold (tuple[int, int]): (approximate) size of final,
                                          reduced image (height, width)
        kernel (int): size of square convolutional kernel
        stride (int): size of base stride for convolutional kernel
        features (int): number of features in the convolutional layers
        conv_onlyweights (bool): determines if convolutional layers learn
                                 only weights or weights and bias
        batchnorm_onlybias (bool): determines if the batch normalization layers
                                   learn only bias or weights and bias
        act_layer(nn.modules.activation): torch neural network layer class to
                                          use as activation

    """

    def __init__(
        self,
        img_size: tuple[int, int, int] = (1, 1700, 500),
        size_threshold: tuple[int, int] = (8, 8),
        kernel: int = 5,
        stride: int = 2,
        features: int = 12,
        conv_onlyweights: bool = True,
        batchnorm_onlybias: bool = True,
        act_layer: nn.Module = nn.GELU,
    ) -> None:
        """Initialization for reduction CNN."""
        super().__init__()

        self.img_size = img_size
        C, H, W = self.img_size
        self.size_threshold = size_threshold
        H_lim, W_lim = self.size_threshold
        self.kernel = kernel
        self.stride = stride
        self.features = features
        self.depth = 0  # initialize depth
        self.conv_weights = True
        self.conv_bias = not conv_onlyweights
        self.batchnorm_weights = not batchnorm_onlybias
        self.batchnorm_bias = True

        # Input Layers
        self.inConv = nn.Conv2d(
            in_channels=C,
            out_channels=self.features,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.stride,
            padding_mode="zeros",
            bias=self.conv_bias,
        )

        normLayer = nn.BatchNorm2d(features)

        # Necessary step to turn off only the scaling.
        if not self.batchnorm_weights:
            nn.init.constant_(normLayer.weight, 1)
            normLayer.weight.requires_grad = False

        self.inNorm = normLayer
        self.inActivation = act_layer()

        W, H, _ = conv2d_shape(
            w=W,
            h=H,
            k=self.kernel,
            s_w=self.stride,
            s_h=self.stride,
            p_w=self.stride,
            p_h=self.stride,
        )

        self.depth += 1

        # Module list to hold reduction layers
        self.ReduceConvList = nn.ModuleList()
        while W > W_lim or H > H_lim:
            # Set Stride & Padding
            if W > W_lim:
                w_stride = self.stride
            else:
                w_stride = 1
            if H > H_lim:
                h_stride = self.stride
            else:
                h_stride = 1

            w_pad = 2 * w_stride
            h_pad = 2 * h_stride

            # Define Layers
            reduceLayer = nn.Conv2d(
                in_channels=self.features,
                out_channels=self.features,
                kernel_size=self.kernel,
                stride=(h_stride, w_stride),
                padding=(h_pad, w_pad),
                padding_mode="zeros",
                bias=self.conv_bias,
            )

            normLayer = nn.BatchNorm2d(features)

            if not self.batchnorm_weights:
                nn.init.constant_(normLayer.weight, 1)
                normLayer.weight.requires_grad = False

            # Make list of small sequential modules. Then we'll use enumerate
            # in forward method.
            self.depth += 1
            reduce_dict = OrderedDict(
                [
                    (f"reduce{self.depth:02d}", reduceLayer),
                    (f"bnorm{self.depth:02d}", normLayer),
                    (f"act{self.depth:02d}", act_layer()),
                ]
            )
            self.ReduceConvList.append(nn.Sequential(reduce_dict))

            # Recalculate Size
            W, H, _ = conv2d_shape(
                w=W, h=H, k=self.kernel, s_w=w_stride, s_h=h_stride, p_w=w_pad, p_h=h_pad
            )

        # Define final size
        self.finalW = W
        self.finalH = H

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for reduction CNN."""
        # Input Layers
        x = self.inConv(x)
        x = self.inNorm(x)
        x = self.inActivation(x)

        # Reduction Layers
        for i, reduce_conv in enumerate(self.ReduceConvList):
            x = reduce_conv(x)

        return x



class Image2VectorCNN(nn.Module):
    """Image to vector CNN.

    Convolutional Neural Network Model that maps an image to a vector. Constructed using
    both an interpretability block, defined above, and a reduction block.

    Args:
        img_size (tuple[int, int, int]): size of input (channels, height, width)
        output_dim (int): size of output vector
        size_threshold (tuple[int, int]): (approximate) size of reduced image
                                          (height, width)
        kernel (int): size of square convolutional kernel
        features (int): number of features in the convolutional layers
        interp_depth (int): number of interpretability blocks
        conv_onlyweights (bool): determines if convolutional layers learn only
                                 weights or weights and bias
        batchnorm_onlybias (bool): determines if the batch normalization layers
                                   learn only bias or weights and bias
        act_layer (nn.modules.activation): torch neural network layer class to
                                           use as activation
        norm_layer (nn.modules.normalization): torch neural network layer class
        hidden_features (int): number of hidden features in the fully connected
                               dense layer
        final_activation (nn.modules.activation): torch neural network layer class to use
                                                  for the final output.
        return_hidden (bool): Flag to return output from CNN layers prior to MLP outputs

    """

    def __init__(
        self,
        img_size: tuple[int, int, int] = (1, 1120, 400),
        output_dim: int = 29,
        size_threshold: tuple[int, int] = (12, 12),
        kernel: int = 3,
        features: int = 16,
        interp_depth: int = 12,
        conv_onlyweights: bool = True,
        batchnorm_onlybias: bool = True,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        hidden_features: int = 32,
        final_activation: nn.Module = nn.Identity,
        return_hidden: bool = False,
    ) -> None:
        """Initialization for image-to-vector CNN."""
        super().__init__()
        self.img_size = img_size
        _, H, W = self.img_size
        self.output_dim = output_dim
        self.size_threshold = size_threshold
        self.kernel = kernel
        self.features = features
        self.interp_depth = interp_depth
        self.hidden_features = hidden_features
        self.final_activation = final_activation()
        self.return_hidden = return_hidden

        self.conv_onlyweights = conv_onlyweights
        self.conv_weights = True
        self.conv_bias = not self.conv_onlyweights
        self.batchnorm_onlybias = batchnorm_onlybias
        self.batchnorm_weights = not self.batchnorm_onlybias
        self.batchnorm_bias = True

        self.interp_module = CNN_Interpretability_Module(
            img_size=self.img_size,
            kernel=self.kernel,
            features=self.features,
            depth=self.interp_depth,
            conv_onlyweights=self.conv_onlyweights,
            batchnorm_onlybias=self.batchnorm_onlybias,
            act_layer=act_layer,
        )

        self.reduction_module = CNN_Reduction_Module(
            img_size=(self.features, H, W),
            size_threshold=self.size_threshold,
            kernel=self.kernel,
            features=self.features,
            conv_onlyweights=self.conv_onlyweights,
            batchnorm_onlybias=self.batchnorm_onlybias,
            act_layer=act_layer,
        )

        self.reduction_depth = self.reduction_module.depth
        self.finalW = self.reduction_module.finalW
        self.finalH = self.reduction_module.finalH

        self.endConv = nn.Conv2d(
            in_channels=self.features,
            out_channels=self.features,
            kernel_size=self.kernel,
            stride=1,
            padding="same",  # pads the input so the output
            # has the shape as the input,
            # stride=1 only
            bias=self.conv_bias,
        )
        self.endConvActivation = act_layer()

        # Hidden Layer (Equivalent to a matched convolutional layer?)
        self.hidden = nn.Linear(
            self.finalH * self.finalW * self.features, self.hidden_features
        )
        self.hiddenActivation = act_layer()

        # MLP Output Layer
        self.out_mlp = generalMLP(
            input_dim=self.hidden_features,
            output_dim=self.output_dim,
            hidden_feature_list=(2 * self.hidden_features,),
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for image-to-scalar CNN."""
        x = self.interp_module(x)
        x = self.reduction_module(x)

        # Final Convolution
        x = self.endConv(x)
        x = self.endConvActivation(x)
        x = torch.flatten(x, start_dim=1)

        # Hidden Layer
        hidden_out = self.hidden(x)
        hidden_out = self.hiddenActivation(hidden_out)

        # MLP Output Layer
        out = self.out_mlp(hidden_out)

        # Final activation
        out = self.final_activation(out)

        if self.return_hidden:
            return out, hidden_out
        else:
            return out


####################################
# Get Conv2D Shape
####################################
def conv2d_shape(
    w: int, h: int, k: int, s_w: int, s_h: int, p_w: int, p_h: int
) -> tuple[int, int, int]:
    """Function to calculate the new dimension of an image after a nn.Conv2d.

    Args:
        w (int): starting width
        h (int): starting height
        k (int): kernel size
        s_w (int): stride size along the width
        s_h (int): stride size along the height
        p_w (int): padding size along the width
        p_h (int): padding size along the height

    Returns:
        new_w (int): number of pixels along the width
        new_h (int): number of pixels along the height
        total (int): total number of pixels in new image

    See Also:
    Formula taken from
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    Assuming a 2D input and dilation = 1

    """
    new_w = int(math.floor(((w + 2 * p_w - (k - 1) - 1) / s_w) + 1))
    new_h = int(math.floor(((h + 2 * p_h - (k - 1) - 1) / s_h) + 1))
    total = new_w * new_h

    return new_w, new_h, total


def convtranspose2d_shape(
    w: int,
    h: int,
    k_w: int,
    k_h: int,
    s_w: int,
    s_h: int,
    p_w: int,
    p_h: int,
    op_w: int,
    op_h: int,
    d_w: int,
    d_h: int,
) -> tuple[int, int, int]:
    """Calculate the dimension of an image after a nn.ConvTranspose2d.

    This assumes *groups*, *dilation*, and *ouput_padding* are all default
    values.

    Args:
        w (int): starting width
        h (int): starting height
        k_w (int): kernel width size
        k_h (int): kernel height size
        s_w (int): stride size along the width
        s_h (int): stride size along the height
        p_w (int): padding size along the width
        p_h (int): padding size along the height
        op_w (int): output padding size along the width
        op_h (int): output padding size along the height
        d_w (int): dilation size along the width
        d_h (int): dilation size along the height

    Returns:
        new_w (int): number of pixels along the width
        new_h (int): number of pixels along the height
        total (int): total number of pixels in new image

    See Also:
    Formula taken from
    https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

    """
    new_w = (w - 1) * s_w - 2 * p_w + d_w * (k_w - 1) + op_w + 1
    new_h = (h - 1) * s_h - 2 * p_h + d_h * (k_h - 1) + op_h + 1
    total = new_w * new_h

    return new_w, new_h, total


class generalMLP(nn.Module):
    """A general multi-layer perceptron structure.

    Consists of stacked linear layers, normalizing layers, and
    activations. This is meant to be reused as a highly customizeable, but
    standardized, MLP structure.

    Args:
        input_dim (int): Dimension of input
        output_dim (int): Dimension of output
        hidden_feature_list (tuple[int, ...]): List of number of features in each layer.
                                               Length determines number of layers.
        act_layer (nn.modules.activation): torch neural network layer class to
                                           use as activation
        norm_layer (nn.Module): Normalization layer.

    """

    def __init__(
        self,
        input_dim: int = 64,
        output_dim: int = 16,
        hidden_feature_list: tuple[int, ...] = (16, 32, 32, 16),
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        """Initialization for MLP."""
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_feature_list = hidden_feature_list
        self.act_layer = act_layer
        self.norm_layer = norm_layer

        # Create full feature list without mutating input
        self.feature_list = (input_dim,) + hidden_feature_list + (output_dim,)

        # Module list to hold linear, normalization, and activation layers.
        self.LayerList = nn.ModuleList()
        # Create transpose convolutional layer for each entry in feature list.
        for i in range(len(self.feature_list) - 1):
            linear = nn.Linear(self.feature_list[i], self.feature_list[i + 1])

            normalize = self.norm_layer(self.feature_list[i + 1])
            activation = self.act_layer()

            # Make list of small sequential modules. Then we'll use enumerate
            # in forward method.
            #
            # Don't attach an activation to the final layer
            if i == len(self.feature_list) - 2:
                cmpd_dict = OrderedDict(
                    [
                        (f"linear{i:02d}", linear),
                    ]
                )
            else:
                cmpd_dict = OrderedDict(
                    [
                        (f"linear{i:02d}", linear),
                        (f"norm{i:02d}", normalize),
                        (f"act{i}", activation),
                    ]
                )

            self.LayerList.append(nn.Sequential(cmpd_dict))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for MLP."""
        # enumeration of nn.moduleList is supported under `torch.jit.script`
        for i, ll_layer in enumerate(self.LayerList):
            x = ll_layer(x)

        return x



def count_torch_params(model: nn.Module, trainable: bool = True) -> int:
    """Count parameters in a pytorch model.

    Args:
        model (nn.Module): Model to count parameters for.
        trainable (bool): If TRUE, count only trainable parameters.

    """
    plist = []
    for p in model.parameters():
        if trainable:
            if p.requires_grad:
                plist.append(p.numel())
            else:
                pass
        else:
            plist.append(p.numel())

    return sum(plist)


def freeze_torch_params(model: nn.Module) -> None:
    """Freeze all parameters in a PyTorch model in place.

    Args:
        model (nn.Module): model to freeze.

    """
    for p in model.parameters():
        if hasattr(p, "requires_grad"):
            p.requires_grad = False

if __name__ == "__main__":
    """For testing and debugging.

    """

    # Excercise model setup
    batch_size = 2
    img_h = 1120
    img_w = 800
    input_vector_size = 28
    output_dim = 28
    y = torch.rand(batch_size, input_vector_size)
    H1 = torch.rand(batch_size, 1, img_h, img_w)
    H2 = torch.rand(batch_size, 1, img_h, img_w)

    model_args_large = {
        "img_size": (1, img_h, img_w),
        "input_vector_size": input_vector_size,
        "output_dim": output_dim,
        "min_variance": 1e-6,
        "features": 12,
        "depth": 12,
        "kernel": 3,
        "img_embed_dim": 32,
        "vector_embed_dim": 32,
        "size_reduce_threshold": (8, 8),
        "vector_feature_list": (32, 32, 64, 64),
        "output_feature_list": (64, 128, 128, 64),
    }

    model_args_medium = {
        "img_size": (1, img_h, img_w),
        "input_vector_size": input_vector_size,
        "output_dim": output_dim,
        "min_variance": 1e-6,
        "features": 12,
        "depth": 15,
        "kernel": 3,
        "img_embed_dim": 32,
        "vector_embed_dim": 32,
        "size_reduce_threshold": (16, 16),
        "vector_feature_list": (16, 64, 64, 16),
        "output_feature_list": (16, 64, 64, 16),
    }

    model_args_small = {
        "img_size": (1, img_h, img_w),
        "input_vector_size": input_vector_size,
        "output_dim": output_dim,
        "min_variance": 1e-6,
        "features": 4,
        "depth": 6,
        "kernel": 3,
        "img_embed_dim": 16,
        "vector_embed_dim": 16,
        "size_reduce_threshold": (24, 24),
        "vector_feature_list": (8, 16, 16, 8),
        "output_feature_list": (8, 16, 16, 8),
    }

    policy_model = gaussian_policyCNN(**model_args_medium)

    policy_model.eval()
    policy_distribution = policy_model(y, H1, H2)
    print("Initial mean:", policy_distribution.mean)
    print("Initial covariance:", policy_distribution.covariance_matrix)
    print("Initial mean shape:", policy_distribution.mean.shape)
    print("Initial covariance shape:", policy_distribution.covariance_matrix.shape)
    print(
        "Number of trainable parameters in value network:",
        count_torch_params(policy_model, trainable=True),
    )
