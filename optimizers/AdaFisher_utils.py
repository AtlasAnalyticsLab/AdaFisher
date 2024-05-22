from torch.nn.functional import pad
from typing import Dict, Tuple
from torch import (Tensor, cat, einsum, sum)
from math import prod
from torch.nn import Module, Linear, Conv2d, BatchNorm2d, LayerNorm


def MinMaxNormalization(tensor: Tensor, epsilon: float = 1e-12) -> Tensor:
    min_tensor = tensor.min()
    max_tensor = tensor.max()
    range_tensor = max_tensor - min_tensor
    return tensor.add_(-min_tensor).div_(range_tensor + epsilon)


def update_running_avg(new: Tensor, current: Dict[Module, Tensor], gammas: list):
    """
    Update the running average of parameters with a new value using a specified beta3 coefficient.

    This function is designed to update the running average of model parameters in a neural network training loop.
    It utilizes a form of exponential moving average, adjusted by the beta3 coefficient, to blend the current parameter
    values with new ones.

    Parameters:
    - new (Tensor): The new value(s) to be incorporated into the running average. This tensor should have the same
    dimensions as the parameter values it's updating.
    - current (Dict[Module, Tensor]): A dictionary mapping from PyTorch modules to their parameters that are to be
    updated. The running average calculation is applied directly to these parameters.
    - gammas (list): The coefficients used for exponential smoothing, controlling the rate at which the running average
    forgets previous values. They must be between 0 and 1, where values closer to 1 make the average more stable over time.

    Returns:
    - None: The function updates the `current` dictionary in-place, modifying the parameter values directly.

    Note:
    - This function modifies the `current` dictionary in-place, so it does not return anything.
    Ensure that this behavior is intended in your use case.
    """
    current *= (1 - gammas[0])
    current += new * gammas[1]


def _extract_patches(x: Tensor, kernel_size: Tuple[int], stride: Tuple[int], padding: Tuple[int]) -> Tensor:
    """
    Extract patches from input feature maps given a specified kernel size, stride, and padding.

    This function applies a sliding window approach to input feature maps to extract patches according to the defined
    kernel size, stride, and padding. It is useful for operations that require localized portions of the input, such
    as convolutional layers in neural networks. The function handles padding by extending the input feature maps if
    needed, then extracts overlapping or non-overlapping patches based on the stride and rearranges the output to a s
    uitable format for further processing.

    Parameters:
    - x (Tensor): The input feature maps with dimensions (batch_size, in_channels, height, width).
    - kernel_size (Tuple[int]): The height and width of the kernel (filter) as a tuple (kernel_height, kernel_width).
    - stride (Tuple[int]): The stride of the convolution operation as a tuple (stride_height, stride_width). Determines
    the step size for moving the kernel across the input.
    - padding (Tuple[int]): The amount of padding added to the height and width of the input feature maps as a tuple
    (padding_height, padding_width).

    Returns:
    - Tensor: The extracted patches with dimensions (batch_size, output_height, output_width,
    in_channels * kernel_height * kernel_width), where `output_height` and `output_width` are computed based on the
    input dimensions, kernel size, stride, and padding.

    The function automatically adjusts the input feature maps with padding if specified, and then uses the unfold
    operation to extract patches. The output is rearranged to ensure compatibility with downstream processes or layers.
    """
    if padding[0] + padding[1] > 0:
        x = pad(x, (padding[1], padding[1], padding[0],
                    padding[0])).data
    x = x.unfold(dimension=2, size=kernel_size[0], step=stride[0])
    x = x.unfold(dimension=3, size=kernel_size[1], step=stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))
    return x


class Compute_H_bar_D:
    """
    Computes the diagonal elements of the covariance matrix of activations ('H') for various layer types in a neural
    network.

    This class is particularly useful in scenarios where only the variance of each activation is needed, rather than
    the full covariance matrix. This can significantly reduce computational complexity and memory usage in large
    networks or for specific applications like certain optimization algorithms or initialization strategies.
    """

    @classmethod
    def compute_H_bar_D(cls, h, layer) -> Tensor:
        """
        Computes the diagonal of the covariance matrix for the activations of a given layer.

        Parameters:
        - a (Tensor): The input activations to the layer. This tensor should have dimensions that match what the layer
        expects.
        - layer (Module): A PyTorch layer instance, such as Linear, Conv2d, BatchNorm2d, or LayerNorm.

        Returns:
        - Tensor: A tensor containing the diagonal elements of the covariance matrix of the activations.
        """
        return cls.__call__(h, layer)

    @classmethod
    def __call__(cls, h: Tensor, layer: Module) -> Tensor:
        """
        Directly calls the instance to compute the diagonal of the covariance matrix by delegating to layer-specific
        methods.

        Parameters:
        - h (Tensor): Input activations, same as in `compute_H_bar_D`.
        - layer (Module): The PyTorch layer instance, same as in `compute_H_bar_D`.

        Returns:
        - Tensor: The diagonal of the covariance matrix of the activations.
        """
        if isinstance(layer, Linear):
            H_bar_D = cls.linear(h, layer)
        elif isinstance(layer, Conv2d):
            H_bar_D = cls.conv2d(h, layer)
        elif isinstance(layer, BatchNorm2d):
            H_bar_D = cls.batchnorm2d(h, layer)
        elif isinstance(layer, LayerNorm):
            H_bar_D = cls.layernorm(h, layer)
        else:
            raise NotImplementedError

        return H_bar_D

    @staticmethod
    def conv2d(h: Tensor, layer: Conv2d) -> Tensor:
        """
        Computes the diagonal of the covariance matrix for activations from a Conv2d layer.

        Parameters:
        - h (Tensor): Input activations with shape (batch_size, in_channels, height, width).
        - layer (Conv2d): The convolutional layer from `torch.nn`.

        Returns:
        - Tensor: The diagonal of the covariance matrix of the activations.
        """
        batch_size = h.size(0)
        h = _extract_patches(h, layer.kernel_size, layer.stride, layer.padding)
        spatial_size = h.size(2) * h.size(3)
        h = h.reshape(-1, h.size(-1))
        if layer.bias is not None:
            h_bar = cat([h, h.new(h.size(0), 1).fill_(1)], 1)
        return einsum('ij,ij->j', h_bar, h_bar) / (batch_size * spatial_size) if layer.bias is not None else einsum('ij,ij->j', h, h) / (batch_size * spatial_size)

    @staticmethod
    def linear(h: Tensor, layer: Linear) -> Tensor:
        """
        Computes the diagonal of the covariance matrix for activations from a Linear layer.

        Parameters:
        - h (Tensor): Input activations, possibly flattened for fully connected layers.
        - layer (Linear): The linear layer from `torch.nn`.

        Returns:
        - Tensor: The diagonal of the covariance matrix of the activations.
        """
        if len(h.shape) > 2:
            h = h.reshape(-1, h.shape[-1])
        batch_size = h.size(0)
        if layer.bias is not None:
            h_bar = cat([h, h.new(h.size(0), 1).fill_(1)], 1)
        return einsum('ij,ij->j', h_bar, h_bar) / batch_size if layer.bias is not None else einsum('ij,ij->j', h, h)

    @staticmethod
    def batchnorm2d(h: Tensor, layer: BatchNorm2d) -> Tensor:
        """
        Computes the diagonal of the covariance matrix for activations from a BatchNorm2d layer.

        Parameters:
        - h (Tensor): Input activations with shape suitable for batch normalization.
        - layer (BatchNorm2d): The batch normalization layer from `torch.nn`.

        Returns:
        - Tensor: The diagonal of the covariance matrix of the activations.
        """
        batch_size, spatial_size = h.size(0), h.size(2) * h.size(3)
        sum_h = sum(h, dim=(0, 2, 3)).unsqueeze(1) / (spatial_size ** 2)
        h_bar = cat([sum_h, sum_h.new(sum_h.size(0), 1).fill_(1)], 1)
        return einsum('ij,ij->j', h_bar, h_bar) / (batch_size ** 2)

    @staticmethod
    def layernorm(h: Tensor, layer: LayerNorm) -> Tensor:
        """
        Computes the diagonal of the covariance matrix for activations from a LayerNorm layer.

        Parameters:
        - h (Tensor): Input activations, which can have any shape as layer normalization is flexible.
        - layer (LayerNorm): The layer normalization from `torch.nn`.

        Returns:
        - Tensor: The diagonal of the covariance matrix of the activations.
        """
        dim_to_reduce = [d for d in range(h.ndim) if d != 1]
        batch_size, dim_norm = h.shape[0], prod([h.shape[dim] for dim in dim_to_reduce if dim != 0])
        sum_h = sum(h, dim=dim_to_reduce).unsqueeze(1) / (dim_norm ** 2)
        h_bar = cat([sum_h, sum_h.new(sum_h.size(0), 1).fill_(1)], 1)
        return einsum('ij,ij->j', h_bar, h_bar) / (batch_size ** 2)


class Compute_S_D:
    """
    Computes the diagonal elements of the gradient covariance matrix ('S') for various layer types in a neural network.

    This class supports operations on gradients from Conv2d, Linear, BatchNorm2d, and LayerNorm layers, providing
    insights into the gradient distribution's variance across different parameters. Such computations are crucial
    for gradient-based optimization and understanding model behavior during training.
    """

    @classmethod
    def compute_S_D(cls, s: Tensor, layer: Module) -> Tensor:
        """
        Computes the diagonal of the gradient covariance matrix for the gradients of a given layer.

        Parameters:
        - s (Tensor): The gradients of the layer's output with respect to some loss function.
        - layer (Module): A PyTorch layer instance (Conv2d, Linear, BatchNorm2d, LayerNorm).

        Returns:
        - Tensor: A tensor containing the diagonal elements of the gradient covariance matrix.
        """
        return cls.__call__(s, layer)

    @classmethod
    def __call__(cls, s: Tensor, layer: Module) -> Tensor:
        """
        Directly calls the instance to compute the diagonal of the gradient covariance matrix by delegating to
        layer-specific methods.

        Parameters:
        - s (Tensor): Gradients, same as in `compute_S_D`.
        - layer (Module): The PyTorch layer instance, same as in `compute_S_D`.

        Returns:
        - Tensor: The diagonal of the gradient covariance matrix.
        """
        if isinstance(layer, Conv2d):
            S_D = cls.conv2d(s, layer)
        elif isinstance(layer, Linear):
            S_D = cls.linear(s, layer)
        elif isinstance(layer, BatchNorm2d):
            S_D = cls.batchnorm2d(s, layer)
        elif isinstance(layer, LayerNorm):
            S_D = cls.layernorm(s, layer)
        else:
            raise NotImplementedError
        return S_D

    @staticmethod
    def conv2d(s: Tensor, layer: Conv2d) -> Tensor:
        """
        Computes the diagonal of the gradient covariance matrix for a Conv2d layer.

        Parameters:
        - s (Tensor): Gradients of the Conv2d layer outputs with respect to the loss function, with shape
         (batch_size, n_filters, out_h, out_w).
        - layer (Conv2d): The convolutional layer from `torch.nn`.

        Returns:
        - Tensor: The diagonal of the gradient covariance matrix.
        """
        batch_size = s.shape[0]
        spatial_size = s.size(2) * s.size(3)
        s = s.transpose(1, 2).transpose(2, 3)
        s = s.reshape(-1, s.size(-1))
        return einsum('ij,ij->j', s, s) / (batch_size * spatial_size)

    @staticmethod
    def linear(s: Tensor, layer: Linear) -> Tensor:
        """
        Computes the diagonal of the gradient covariance matrix for a Linear layer.

        Parameters:
        - s (Tensor): Gradients of the Linear layer outputs, possibly reshaped if originally multi-dimensional.
        - layer (Linear): The linear layer from `torch.nn`.

        Returns:
        - Tensor: The diagonal of the gradient covariance matrix.
        """
        if len(s.shape) > 2:
            s = s.reshape(-1, s.shape[-1])
        batch_size = s.size(0)
        return einsum('ij,ij->j', s, s) / batch_size

    @staticmethod
    def batchnorm2d(s: Tensor, layer: BatchNorm2d) -> Tensor:
        """
        Computes the diagonal of the gradient covariance matrix for a BatchNorm2d layer.

        Parameters:
        - s (Tensor): Gradients of the BatchNorm2d layer outputs.
        - layer (BatchNorm2d): The batch normalization layer from `torch.nn`.

        Returns:
        - Tensor: The diagonal of the gradient covariance matrix.
        """
        batch_size = s.size(0)
        sum_s = sum(s, dim=(0, 2, 3))
        return einsum('i,i->i', sum_s, sum_s) / batch_size

    @staticmethod
    def layernorm(s: Tensor, layer: LayerNorm) -> Tensor:
        """
        Computes the diagonal of the gradient covariance matrix for a LayerNorm layer.

        Parameters:
        - s (Tensor): Gradients of the LayerNorm layer outputs.
        - layer (LayerNorm): The layer normalization from `torch.nn`.

        Returns:
        - Tensor: The diagonal of the gradient covariance matrix.
        """
        batch_size = s.size(0)
        sum_s = sum(s, dim=tuple(range(s.ndim - 1)))
        return einsum('i,i->i', sum_s, sum_s) / batch_size