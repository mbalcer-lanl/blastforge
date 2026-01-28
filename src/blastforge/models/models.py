import torch
import torch.nn as nn
from collections import OrderedDict
import math

class tCNNsurrogate(nn.Module):
    """Vector-to-Image CNN.

    Convolutional Neural Network Module that creates a scalar-to-image
    surrogate using a sequence of ConvTranspose2D, Batch Normalization, and
    Activation layers.

    This architecture is meant to reproduce the architecture described in
    Jekel et. al. 2022 *Using conservation laws to infer deep learning
    model accuracy of Richtmyer-Meshkov instabilities.*

    However, image sizes are not always square powers of 2. Therefore, we
    allow a transpose convolution with specified parameters to resize the
    initial image stack to something that can be upsized to the output
    image size by multiplying by a power of 2. Unlike the jekelCNNsurrogate
    class which dealt with resizing by interpolation in the last layer. It
    is confusing because it is...

    WARNING!!!

    If the linear_features, intial convolution parameters, and feature list are
    not set up carefully then the output will be different than the expected
    output image size. A helper function should be constructed to aid in
    checking consistency but is not available now.

    WARNING!!!

    Args:
        input_size (int): Size of input
        linear_features (tuple[int, int, int]): Window size and number of features
                                                scalar parameters are originally
                                                mapped into
        initial_tconv_kernel (tuple[int, int]): Kernel size of initial tconv2d
        initial_tconv_stride (tuple[int, int]): Stride size of initial tconv2d
        initial_tconv_padding (tuple[int, int]): Padding size of initial tconv2d
        initial_tconv_outpadding (tuple[int, int]): Outout padding size of
                                                    initial tconv2d
        initial_tconv_dilation (tuple[int, int]): Dilation size of initial tconv2d
        kernel (tuple[int, int]): Size of transpose-convolutional kernel
        nfeature_list (list[int]): List of number of features in each
                                   T-convolutional layer
        output_image_size (tuple[int, int]): Image size to output, (H, W).
        output_image_channels (int): Number of output image channels.
        act_layer(nn.modules.activation): torch neural network layer class
                                          to use as activation

    """

    def __init__(
        self,
        input_size: int = 29,
        linear_features: tuple[int, int] = (7, 5, 256),
        initial_tconv_kernel: tuple[int, int] = (5, 5),
        initial_tconv_stride: tuple[int, int] = (5, 5),
        initial_tconv_padding: tuple[int, int] = (0, 0),
        initial_tconv_outpadding: tuple[int, int] = (0, 0),
        initial_tconv_dilation: tuple[int, int] = (1, 1),
        kernel: tuple[int, int] = (3, 3),
        nfeature_list: list[int] = [256, 128, 64, 32, 16],
        output_image_size: tuple[int, int] = (1120, 800),
        output_image_channels: int = 1,
        act_layer: nn.Module = nn.GELU,
    ) -> None:
        """Initialization for the t-CNN surrogate."""
        super().__init__()

        self.input_size = input_size
        self.output_image_size = output_image_size  # This argument is not used currently
        self.output_image_channels = output_image_channels
        self.linear_features = linear_features
        self.initial_tconv_kernel = initial_tconv_kernel
        self.initial_tconv_stride = initial_tconv_stride
        self.initial_tconv_padding = initial_tconv_padding
        self.initial_tconv_outpadding = initial_tconv_outpadding
        self.initial_tconv_dilation = initial_tconv_dilation
        self.nfeature_list = nfeature_list
        self.kernel = kernel
        self.nConvT = len(self.nfeature_list)

        # First linear remap
        out_features = (
            self.linear_features[0] * self.linear_features[1] * self.linear_features[2]
        )
        self.dense_expand = nn.Linear(
            in_features=self.input_size, out_features=out_features, bias=False
        )

        normLayer = nn.BatchNorm2d(self.linear_features[2])
        nn.init.constant_(normLayer.weight, 1)
        normLayer.weight.requires_grad = False

        self.inNorm = normLayer
        self.inActivation = act_layer()

        # Initial tconv2d layer to prepare for doubling layers
        self.initTConv = nn.ConvTranspose2d(
            in_channels=self.linear_features[2],
            out_channels=self.nfeature_list[0],
            kernel_size=self.initial_tconv_kernel,
            stride=self.initial_tconv_stride,
            padding=self.initial_tconv_padding,
            output_padding=self.initial_tconv_outpadding,
            dilation=self.initial_tconv_dilation,
            bias=False,
        )

        normLayer = nn.BatchNorm2d(self.linear_features[2])
        nn.init.constant_(normLayer.weight, 1)
        normLayer.weight.requires_grad = False

        self.initTconvNorm = normLayer
        self.initTconvActivation = act_layer()

        # Module list to hold transpose convolutions
        self.CompoundConvTList = nn.ModuleList()
        # Create transpose convolutional layer for each entry in feature list.
        for i in range(self.nConvT - 1):
            tconv = nn.ConvTranspose2d(
                in_channels=self.nfeature_list[i],
                out_channels=self.nfeature_list[i + 1],
                kernel_size=self.kernel,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            )

            normLayer = nn.BatchNorm2d(self.nfeature_list[i + 1])
            nn.init.constant_(normLayer.weight, 1)
            normLayer.weight.requires_grad = False

            # Make list of small sequential modules. Then we'll use enumerate
            # in forward method.
            cmpd_dict = OrderedDict(
                [
                    (f"tconv{i:02d}", tconv),
                    (f"bnorm{i:02d}", normLayer),
                    (f"act{i:02d}", act_layer()),
                ]
            )
            self.CompoundConvTList.append(nn.Sequential(cmpd_dict))

        # Final Transpose Conv layer followed by hyperbolic tanh activation
        self.final_tconv = nn.ConvTranspose2d(
            in_channels=self.nfeature_list[-1],
            out_channels=self.output_image_channels,
            kernel_size=self.kernel,
            stride=2,
            padding=1,
            output_padding=1,
            bias=True,
        )

        # If normalizing to [-1, 1]
        # self.final_act = nn.Tanh()

        # Else...
        self.final_act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for the t-CNN surrogate."""
        # Input Layers
        x = self.dense_expand(x)
        # Reshape to a 2D block with channels
        # NOTE: -1 infers batch size
        x = x.view(
            -1, self.linear_features[2], self.linear_features[0], self.linear_features[1]
        )

        x = self.inNorm(x)
        x = self.inActivation(x)
        # print('After dense-map shape:', x.shape)

        # Initial resize tConv layer
        x = self.initTConv(x)
        x = self.initTconvNorm(x)
        x = self.initTconvActivation(x)
        # print('After initTconv shape:', x.shape)

        # enumeration of nn.moduleList is supported under `torch.jit.script`
        for i, cmpdTconv in enumerate(self.CompoundConvTList):
            x = cmpdTconv(x)

        # Final ConvT
        x = self.final_tconv(x)
        x = self.final_act(x)
        # print('After final convT shape:', x.shape)

        return x


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


class hybrid2vectorCNN(nn.Module):
    """Vector-and-Image to Vector-and-Image CNNs.

    Convolutional Neural Network Module that maps a triple (y, H1, H2) to a
    vector, R. Here, y is a 1D-tensor, H1 and H2 are 2D-tensors, and R is a
    1D-tensor. Each input is first processed through an independent branch
    before concatenation to a dense network.

    Args:
        img_size (tuple[int, int, int]): (C, H, W) dimensions of H1 and H2.
        input_vector_size (int): Size of input vector
        output_dim (int): Dimension of vector output.
        features (int): Number of output channels/features for each convolutional layer.
        depth (int): Number of convolutional layers in each image processing branch.
        kernel (int): Size of symmetric convolutional kernels
        img_embed_dim (int): Number of features in MLP output from image embeddings.
        vector_embed_dim (int): Number of features in MLP output from image embeddings.
        vector_feature_list (tuple[int, ...]): Number of features in each hidden layer
                                               of vector-MLP.
        output_feature_list (tuple[int, ...]): Number of features in each hidden layer
                                               of final/output-MLP.
        act_layer(nn.Module): torch neural network layer class to use as activation
        norm_layer(nn.Module): torch neural network layer class to use as normalization
                               between MLP layers.

    """

    def __init__(
        self,
        img_size: tuple[int, int, int] = (1, 1120, 400),
        input_vector_size: int = 28,
        output_dim: int = 1,
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

        # Final MLP
        #
        # NOTE: Final activation is just identity.
        cat_size = self.vector_embed_dim + 2 * self.img_embed_dim
        self.final_mlp = generalMLP(
            input_dim=cat_size,
            output_dim=self.output_dim,
            hidden_feature_list=self.output_feature_list,
            act_layer=self.act_layer,
            norm_layer=nn.Identity,
        )

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

        # Concatenate outputs and send through final MLP layer.
        cat = torch.cat((y_out, h1_out, h2_out), dim=1)
        out = self.final_mlp(cat)

        return out


## if __name__ == "__main__":
##     """For testing and debugging.
## 
##     """
## 
##     # Excercise model setup
##     batch_size = 4
##     img_h = 1120
##     img_w = 400
##     input_vector_size = 28
##     output_dim = 5
##     y = torch.rand(batch_size, input_vector_size)
##     H1 = torch.rand(batch_size, 1, img_h, img_w)
##     H2 = torch.rand(batch_size, 1, img_h, img_w)
## 
##     value_model = hybrid2vectorCNN(
##         img_size=(1, img_h, img_w),
##         input_vector_size=input_vector_size,
##         output_dim=output_dim,
##         features=12,
##         img_embed_dim=32,
##         vector_embed_dim=32,
##         vector_feature_list=(32, 32, 64, 64),
##         depth=12,
##         kernel=3,
##         size_reduce_threshold=(8, 8),
##         act_layer=nn.GELU,
##     )
## 
##     value_model.eval()
##     value_pred = value_model(y, H1, H2)
##     print("value_pred shape:", value_pred.shape)
##     print(
##         "Number of trainable parameters in value network:",
##         count_torch_params(value_model, trainable=True),
##     )