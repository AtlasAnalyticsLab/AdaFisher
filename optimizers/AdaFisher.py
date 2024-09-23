"""Implementation of AdaFisher/AdaFisherW"""

from typing import Callable, Dict, List, Union, Tuple, Type
from torch import (Tensor, kron, is_grad_enabled, no_grad, zeros_like,
                   preserve_format, ones_like, cat, einsum, sum, inf)
from torch.optim import Optimizer
import torch.distributed as dist
from torch.nn.functional import pad
from math import prod
from torch.nn import Module, Linear, Conv2d, BatchNorm2d, LayerNorm, Parameter


def smart_detect_inf(tensor: Tensor) -> Tensor:
    """
    Replaces positive infinity in the tensor with 1. and negative infinity with 0..
    
    Parameters:
    tensor (torch.Tensor): Input tensor that can have any dimension.
    
    Returns:
    torch.Tensor: A tensor with the same shape, dtype, and device as the input, where
                  positive infinities are replaced by 1. and negative infinities by 0..
    """
    result_tensor = tensor.clone()
    result_tensor[tensor == inf] = 1.
    result_tensor[tensor == -inf] = 0.
    return result_tensor


def MinMaxNormalization(tensor: Tensor, epsilon: float = 1e-6) -> Tensor:
    """
    Normalizes a tensor using the Min-Max scaling technique with an option to adjust for infinities.
    This scaling shrinks the range of the feature data to be between 0 and 1. An epsilon is added
    to the denominator for numerical stability to avoid division by zero.

    Parameters:
    tensor (torch.Tensor): The input tensor to normalize, which can have any shape.
    epsilon (float, optional): A small value added to the denominator to prevent division by zero. 
                               Defaults to 1e-12.

    Returns:
    torch.Tensor: A tensor of the same shape as the input, with all elements scaled to the range [0, 1].
    """
    tensor = smart_detect_inf(tensor)
    min_tensor = tensor.min()
    max_tensor = tensor.max()
    range_tensor = max_tensor - min_tensor
    return tensor.add_(-min_tensor).div_(range_tensor + epsilon)


def update_running_avg(new: Tensor, current: Tensor, gammas: list):
    """
    Update the running average of parameters with a new value using a specified beta3 coefficient.

    This function is designed to update the running average of model parameters in a neural network training loop.
    It utilizes a form of exponential moving average, adjusted by the beta3 coefficient, to blend the current parameter
    values with new ones.

    Parameters:
    - new (Tensor): The new values to be incorporated into the running average. This tensor should have the same
    dimensions as the parameter values it's updating.
    - current Tensor: The previous values that are to be updated. The running average calculation is applied directly
    to these parameters.
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


def _extract_patches(x: Tensor, kernel_size: Tuple[int],
                     stride: Tuple[int],
                     padding: Tuple[int],
                     groups: int) -> Tensor:

    """
    Extract patches from input feature maps given a specified kernel size, stride, padding, and groups.

    This function applies a sliding window approach to input feature maps to extract patches according to the defined
    kernel size, stride, and padding, while respecting the groups parameter. It is useful for operations that require
    localized portions of the input, such as convolutional layers in neural networks. The function handles padding by
    extending the input feature maps if needed, then extracts overlapping or non-overlapping patches based on the
    stride and rearranges the output to a suitable format for further processing.

    Parameters:
    - x (Tensor): The input feature maps with dimensions (batch_size, in_channels, height, width).
    - kernel_size (Tuple[int]): The height and width of the kernel (filter) as a tuple (kernel_height, kernel_width).
    - stride (Tuple[int]): The stride of the convolution operation as a tuple (stride_height, stride_width). Determines
    the step size for moving the kernel across the input.
    - padding (Tuple[int]): The amount of padding added to the height and width of the input feature maps as a tuple
    (padding_height, padding_width).
    - groups (int): The number of groups for grouped convolution.

    Returns:
    - Tensor: The extracted patches with dimensions (batch_size, output_height, output_width,
    in_channels * kernel_height * kernel_width), where `output_height` and `output_width` are computed based on the
    input dimensions, kernel size, stride, and padding.

    The function automatically adjusts the input feature maps with padding if specified, and then uses the unfold
    operation to extract patches. The output is rearranged to ensure compatibility with downstream processes or layers.
    """

    if padding[0] + padding[1] > 0:
        x = pad(x, (padding[1], padding[1], padding[0], padding[0]))
    batch_size, in_channels, height, width = x.size()
    x = x.view(batch_size, groups, in_channels // groups, height, width)
    x = x.unfold(3, kernel_size[0], stride[0])
    x = x.unfold(4, kernel_size[1], stride[1])
    x = x.permute(0, 1, 3, 4, 2, 5, 6).contiguous()
    x = x.view(batch_size, groups, -1, in_channels // groups * kernel_size[0] * kernel_size[1])
    x = x.view(batch_size, -1, x.size(2), x.size(3))
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
        h = _extract_patches(h, layer.kernel_size, layer.stride, layer.padding, layer.groups)
        spatial_size = h.size(2) * h.size(3)
        h = h.reshape(-1, h.size(-1))
        if layer.bias is not None:
            h_bar = cat([h, h.new(h.size(0), 1).fill_(1)], 1)
        return einsum('ij,ij->j', h_bar, h_bar) / (batch_size * spatial_size) if layer.bias is not None \
            else einsum('ij,ij->j', h, h) / (batch_size * spatial_size)

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
        return einsum('ij,ij->j', h_bar, h_bar) / batch_size if layer.bias is not None \
            else einsum('ij,ij->j', h, h) / batch_size

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

class AdaFisherBackBone(Optimizer):
    """
    The AdaFisherBackBone class serves as the base class for optimizers that adjust model parameters 
    based on the Fisher Information Matrix to more efficiently navigate the curvature of the loss landscape. 
    This class is designed to work with neural network models and supports specific module types 
    for which Fisher Information can be effectively computed.

    The class initializes with model-related parameters and configurations for the optimizer, 
    including learning rate adjustments and regularization techniques based on the second-order 
    information. It integrates advanced techniques such as dynamic preconditioning with the Fisher 
    Information to stabilize and speed up the convergence of the training process.

    Attributes:
        SUPPORTED_MODULES (Tuple[Type[str], ...]): A tuple listing the types of model modules (layers) 
        supported by this optimizer. Typical supported types include "Linear", "Conv2d", 
        "BatchNorm2d", and "LayerNorm".

    Args:
        model (Module): The neural network model to optimize.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        beta (float, optional): The beta parameter for the optimizer, used in calculations 
                                like those in momentum. Defaults to 0.9.
        Lambda (float, optional): Regularization parameter. Defaults to 1e-3.
        gammas (List[float], optional): A list containing gamma parameters used in the 
                                        running average computations. Defaults to [0.92, 0.008].
        TCov (int, optional): Interval in steps for recalculating the covariance matrices. Defaults to 100.
        weight_decay (float, optional): Weight decay coefficient to help prevent overfitting. Defaults to 0.
        dist (bool, optional): True when using a Mutli-GPUs environment, otherwise False.

    Raises:
        ValueError: If any of the parameters are out of their expected ranges.
    """ 
    
    SUPPORTED_MODULES: Tuple[Type[str], ...] = ("Linear", "Conv2d", "BatchNorm2d", "LayerNorm")

    def __init__(self,
                 model: Module,
                 lr: float = 1e-3,
                 beta: float = 0.9,
                 Lambda: float = 1e-3,
                 gammas: List = [0.92, 0.008],
                 TCov: int = 100,
                 weight_decay: float = 0,
                 dist: bool = False
                 ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if not 0.0 <= gammas[0] < 1.0:
            raise ValueError(f"Invalid gamma parameter at index 0: {gammas[0]}")
        if not 0.0 <= gammas[1] < 1.0:
            raise ValueError(f"Invalid gamma parameter at index 1: {gammas[1]}")
        if not TCov > 0:
            raise ValueError(f"Invalid TCov parameter: {TCov}")
        defaults = dict(lr=lr, beta=beta,
                        weight_decay=weight_decay)

        self.gammas = gammas
        self.Lambda = Lambda
        self.model = model
        self.TCov = TCov
        self.dist = dist
        self.steps = 0

        # store vectors for the pre-conditioner
        self.H_bar_D: Dict[Module, Tensor] = {}
        self.S_D: Dict[Module, Tensor] = {}
        # save the layers of the network where FIM can be computed
        self.modules: List[Module] = []
        self.Compute_H_bar_D = Compute_H_bar_D()
        self.Compute_S_D = Compute_S_D()
        self._prepare_model()
        super(AdaFisherBackBone, self).__init__(model.parameters(), defaults)

    def _save_input(self, module: Module, input: Tensor, output: Tensor):
        """
        Captures and updates the diagonal elements of the activation covariance matrix for a given
        module at specified intervals. This method is part of the AdaFisher optimizer's mechanism to
        incorporate curvature information from the model's activations into the optimization process.

        Specifically, it computes the diagonal of the activation covariance matrix for the current
        input to the module. This computation is performed at intervals defined by the `TCov` attribute
        of the optimizer. The method then updates a running average of these diagonal elements,
        maintaining this information as part of the optimizer's internal state. This updated state
        is later used to adjust the optimization strategy based on the geometry of the loss surface.

        Parameters:
            - module (Module): The module (layer) of the neural network for which the activations are
                               being analyzed. This module should be compatible with the operations
                               defined in the `Compute_H_bar_D` method.
            - input (Tensor): The input tensor to the `module` during the forward pass. Only the first
                              element of the tuple is used, which is expected to be a tensor representing
                              the data input to the module.
            - output (Tensor): The output tensor from the `module` during the forward pass. This parameter
                               is not used within the method but is required to match the signature for
                               PyTorch hook functions.

        Note:
        This method should be attached as a forward hook to the relevant PyTorch modules within the
        model being optimized. It plays a crucial role in the AdaFisher optimization process by
        leveraging real-time curvature information derived from the model's activations. The periodic
        computation and updating mechanism ensures an efficient balance between capturing accurate
        curvature information and maintaining computational performance.
        """
        if is_grad_enabled() and self.steps % self.TCov == 0:
            H_bar_D_i = self.Compute_H_bar_D(input[0].data, module)
            if self.steps == 0:
                self.H_bar_D[module] = H_bar_D_i.new(H_bar_D_i.size(0)).fill_(1)
            update_running_avg(MinMaxNormalization(H_bar_D_i), self.H_bar_D[module], self.gammas)
            

    def _save_grad_output(self, module: Module, grad_input: Tensor, grad_output: Tensor):
        """
        Updates the optimizer's internal state by capturing and maintaining the diagonal elements
        of the gradient covariance matrix for a given module. This operation is conducted at
        intervals specified by the `TCov` attribute, focusing on the gradients of the output
        with respect to the module's parameters.

        At each specified interval, this method computes the diagonal of the gradient covariance
        matrix based on the gradients of the module's output. This information is crucial for
        adapting the optimizer's behavior according to the curvature of the loss surface, as
        reflected by the gradients' distribution. The computed diagonal elements are then used to
        update a running average stored in the optimizer's internal state, ensuring that the
        optimizer's adjustments are based on up-to-date curvature information.

        Parameters:
            - module (Module): The neural network module (layer) for which gradient information is
                               being analyzed. This should be a module for which gradients are
                               computed during the backward pass.
            - grad_input (Tensor): The gradient of the loss with respect to the input of the `module`.
                                   This parameter is not directly used within the method but is included
                                   to match the signature required for PyTorch backward hooks.
            - grad_output (Tensor): The gradient of the loss with respect to the output of the `module`.
                                    Only the first element of this tuple is used, which is expected to
                                    be a tensor representing the gradient data.

        Note:
        This method is designed to be attached as a backward hook to the relevant PyTorch modules
        within the model being optimized. By capturing real-time information on the distribution of
        gradients, it enables the AdaFisher optimizer to make more informed adjustments to the model
        parameters, factoring in the curvature of the loss surface for more efficient optimization.
        """
        if self.steps % self.TCov == 0:
            S_D_i = self.Compute_S_D(grad_output[0].data, module)
            if self.steps == 0:
                self.S_D[module] = S_D_i.new(S_D_i.size(0)).fill_(1)
            update_running_avg(MinMaxNormalization(S_D_i), self.S_D[module], self.gammas)

    def _prepare_model(self):
        """
        Prepares the model for optimization by registering forward and backward hooks on supported
        modules. These hooks are crucial for capturing activation and gradient information necessary
        for the AdaFisher optimizer to adjust its parameters based on the curvature of the loss
        surface.

        This method iterates through all the modules in the model, identifying those that are
        supported based on a predefined list of module class names (`SUPPORTED_MODULES`). It then
        registers forward hooks to capture activation information and backward hooks to capture
        gradient information. These hooks are attached to functions within the optimizer that
        compute and update the running averages of the activation and gradient covariance matrices.

        Note:
            - When defining the model to be optimized with AdaFisher, avoid using in-place operations
              (e.g., `relu(inplace=True)` or '*=') within the modules. In-place operations can disrupt the proper functioning
              of the forward and backward hooks, leading to incorrect computations or even runtime errors.
              This limitation arises because in-place operations modify the data directly, potentially
              bypassing or altering the data flow that the hooks rely on to capture activation and gradient
              information accurately.
            - Only modules whose class names are in the `SUPPORTED_MODULES` list will have hooks registered.
              It is essential to ensure that the model's critical components for capturing curvature
              information are supported to leverage the full capabilities of the AdaFisher optimizer.

        This preparation step is a critical prerequisite to using the AdaFisher optimizer effectively,
        enabling it to dynamically adjust optimization strategies based on real-time insights into
        the model's behavior.
        """
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.SUPPORTED_MODULES:
                self.modules.append(module)
                module.register_forward_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad_output)
    
    def aggregate_kronecker_factors(self, module: Module):
        """
        Aggregates Kronecker factors across multiple GPUs in a distributed setting.

        This function performs an all-reduce operation to sum the Kronecker factors 
        `H_bar_D` and `S_D` across all GPUs involved in the training. It then averages 
        these factors by dividing the summed values by the total number of GPUs (`world_size`).

        Args:
            module (Module): The neural network module for which the Kronecker factors 
                            `H_bar_D` and `S_D` are being aggregated.
        """
        dist.all_reduce(self.H_bar_D[module], op=dist.ReduceOp.SUM)
        dist.all_reduce(self.S_D[module], op=dist.ReduceOp.SUM)

        self.H_bar_D[module] /= dist.get_world_size()
        self.S_D[module] /= dist.get_world_size()

    def _get_F_tilde(self, module: Module):
        """
        Computes the FIM for a given module's parameters. This method is called internally by the AdaFisher optimizer
        during the optimization process to adjust the parameters of the model in a way that considers both
        the curvature of the loss surface and a damping term.

        The Fisher Information Matrix is approximated using the Kronecker product of the diagonal
        elements of the activation and gradient covariance matrices, captured for the module through
        the optimizer's forward and backward hooks. This approximation is then regularized by adding
        a scalar (Lambda) to its diagonal to ensure numerical stability and to incorporate prior
        knowledge or assumptions about the parameter distribution.

        Parameters:
            - module (Module): The module for which the parameter update is being computed. The method
                               checks if the module has a bias term and adjusts the computation
                               accordingly.

        Returns:
            - A tensor or list of tensors representing the update to be applied to the module's
              parameters. For modules with bias, the update is returned as a list containing separate
              updates for the weights and the bias. For modules without bias, a single tensor F_tilde
              is returned.

        Note:
        This method directly manipulates the gradients of the module's parameters based on the
        computed Fisher Information and is a key component of the AdaFisher optimizer's strategy
        to leverage curvature information for more efficient and effective optimization.
        """
        # Fisher Computation
        if self.dist:
            self.aggregate_kronecker_factors(module = module)
        F_tilde = kron(self.H_bar_D[module].unsqueeze(1), self.S_D[module].unsqueeze(0)).t() + self.Lambda
        if module.bias is not None:
            F_tilde = [F_tilde[:, :-1], F_tilde[:, -1:]]
            F_tilde[0] = F_tilde[0].view(*module.weight.grad.data.size())
            F_tilde[1] = F_tilde[1].view(*module.bias.grad.data.size())
            return F_tilde
        else:
            return F_tilde.reshape(module.weight.grad.data.size())

    def _check_dim(self, param: List[Parameter], idx_module: int, idx_param: int) -> bool:
        """
        Checks if the dimensions of a given parameter match the dimensions of its corresponding module's weights or bias.
        This function is crucial for ensuring that the Fisher information is available for a specific module.

        Parameters:
            - param (List[Parameter]): A list of all parameters within the model.
            - idx_module (int): The index of the module within the model's modules list.
            - idx_param (int): The index of the parameter within the list of parameters being checked.

        Returns:
            - bool: True if the parameter's dimensions match those of the corresponding module's weights or bias
            (Fisher information is available);
            False otherwise.
        """
        params = param[idx_param]
        module = self.modules[idx_module]
        param_size = params.data.size()
        return param_size == module.weight.data.size() or (module.bias is not None and param_size == module.bias.data.size())


class AdaFisher(AdaFisherBackBone):
    """AdaFisher Optimizer: An adaptive learning rate optimizer that leverages Fisher Information for parameter updates.

       This class AdaFisher optimizer extends traditional optimization techniques by incorporating Fisher Information to
       adaptively adjust the learning rates based on the curvature of the loss landscape. This approach aims to enhance
       convergence stability and speed by scaling updates according to the inverse curvature of the parameter space,
       making it particularly suited for deep learning models where the loss landscape can be highly non-convex.

       Key Features:
           - Adaptive Learning Rates: Adjusts learning rates based on Fisher Information, potentially leading to faster
           convergence by taking more informed steps.
           - Curvature-aware Updates: Utilizes the curvature of the loss landscape to modulate the update steps, aiming
           to improve optimization efficiency.
           - Support for Custom Modules: Designed to work with a wide range of neural network architectures by allowing
           flexible mappings between parameters and modules.

       Usage:
       AdaFisher should be used similarly to other PyTorch optimizers, with the additional step of preparing the model
       by registering necessary hooks to compute Fisher Information:

       ```python
       import torch
       model = MyModel()
       optimizer = AdaFisher(model, lr=1e-3, beta=0.9, gamma=[0.92, 0.008], Lambda=1e-3, weight_decay=0)

       for input, target in dataset:
           optimizer.zero_grad()
           output = model(input)
           loss = loss_fn(output, target)
           loss.backward()
           optimizer.step()
       ```
       """
    def __init__(self, 
                 model: Module,
                 lr: float = 1e-3,
                 beta: float = 0.9,
                 Lambda: float = 1e-3,
                 gammas: List = [0.92, 0.008],
                 TCov: int = 100,
                 weight_decay: float = 0,
                 dist: bool = False
                ):
        super(AdaFisher, self).__init__(model,
                                        lr = lr,
                                        beta = beta,
                                        Lambda = Lambda,
                                        gammas = gammas,
                                        TCov = TCov,
                                        dist = dist,
                                        weight_decay = weight_decay)
        
    @no_grad()
    def _step(self, hyperparameters: Dict[str, float], param: Parameter, F_tilde: Tensor):
        """
        Performs a single optimization step for one parameter tensor, applying updates calculated
        from gradient and Fisher information.

        This method applies the AdaFisher optimization logic to update a given parameter based on its
        gradient, the Fisher information, and the optimizer's current state. It computes an adapted
        learning rate for the parameter by taking into account the curvature of the loss surface
        (approximated by the Fisher information) and applies the F_tilde in a way that aims to
        efficiently minimize the loss.

        Parameters:
        - hyperparameters (dict): A dictionary containing optimization hyperparameters such as
                                  learning rate (`lr`), betas for the moving averages (`betas`),
                                  and weight decay (`weight_decay`).
        - param (Parameter): The parameter tensor to be updated.
        - F_tilde (Tensor): The F_tilde tensor computed based on the Fisher information for the
                           parameter.

        Note:
        This method modifies the parameter tensor in-place. It initializes the optimizer's state
        for the parameter if it has not been initialized yet. This state tracks the first and
        second moment of the gradients and the Fisher information. The method then computes decayed
        moving averages for these moments and uses them to adjust the update step. It accounts for
        weight decay if specified in the hyperparameters.

        The adaptation of the learning rate is based on the square root of the second moment (the
        Fisher information) with bias correction applied, ensuring that the updates are scaled
        appropriately as the optimization progresses. This method is decorated with `@no_grad()` to
        ensure that these operations do not track gradients, which is essential for updating the
        parameters without affecting the computational graph of the model's forward pass.
        """
        grad = param.grad
        state = self.state[param]
        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = zeros_like(
                param, memory_format=preserve_format)
        exp_avg = state['exp_avg']
        beta = hyperparameters['beta']
        state['step'] += 1
        bias_correction1 = 1 - beta ** state['step']
        if hyperparameters['weight_decay'] != 0:
            grad = grad.add(param, alpha=hyperparameters['weight_decay'])
        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta).add_(grad, alpha=1 - beta)
        step_size = hyperparameters['lr'] / bias_correction1
        # Update Rule
        param.addcdiv_(exp_avg, F_tilde, value=-step_size)

    @no_grad()
    def step(self, closure: Union[None, Callable[[], Tensor]] = None):
        """
        Performs a single optimization step across all parameter groups of the model.
        This method iterates over each parameter group, updating parameters based on their gradients, a set of
        hyperparameters, and update tensors calculated for each corresponding module. The update process carefully
        tracks the relationship between parameters and their corresponding modules, using index tracking and dimension
        checks to ensure updates are applied accurately.

        Arguments:
            - closure (callable, optional): A closure that reevaluates the model and returns the loss. Currently,
            the use of closure is not supported, and attempting to provide one will raise a NotImplementedError.

        The update process involves:
            1. Iterating through each parameter group and its corresponding hyperparameters.
            2. For each parameter, checking if a gradient is available. If not, it skips to the next parameter.
            3. Using index tracking to associate parameters with their corresponding modules, considering cases
            where parameters do not have gradients or are part of buffer counts.
            4. Performing a dimensionality check to ensure compatibility between the parameter and module,
            then calculating the update curvature tensor.
            5. Applying the update curvature using a custom step function that incorporates the calculated update
            tensor and hyperparameters.
        """
        if closure is not None:
            raise NotImplementedError("Closure not supported.")
        for group in self.param_groups:
            idx_param, idx_module, buffer_count = 0, 0, 0
            param, hyperparameters = group['params'], {"weight_decay": group['weight_decay'], "beta": group['beta'], "lr": group['lr']}
            for _ in range(len(self.modules)):
                if param[idx_param].grad is None:
                    idx_param += 1
                    if param[idx_param].ndim > 1:
                        idx_module += 1
                    else:
                        buffer_count += 1
                    if buffer_count == 2:
                        idx_module += 1
                        buffer_count = 0
                    continue
                m = self.modules[idx_module]
                if self._check_dim(param, idx_module, idx_param):
                    F_tilde = self._get_F_tilde(m)
                    idx_module += 1
                else:
                    F_tilde = ones_like(param[idx_param]) 
                if isinstance(F_tilde, list):
                    for F_tilde_i in F_tilde:
                        self._step(hyperparameters, param[idx_param], F_tilde_i)
                        idx_param += 1
                else:
                    self._step(hyperparameters, param[idx_param], F_tilde)
                    idx_param += 1
        self.steps += 1


class AdaFisherW(AdaFisherBackBone):
    """AdaFisherW Optimizer: An adaptive learning rate optimizer that leverages Fisher Information for parameter updates.

   The AdaFisherW optimizer extends traditional optimization techniques by incorporating Fisher Information to
   adaptively adjust the learning rates based on the curvature of the loss landscape. This approach aims to enhance
   convergence stability and speed by scaling updates according to the inverse curvature of the parameter space,
   making it particularly suited for deep learning models where the loss landscape can be highly non-convex.

   Key Features:
       - Adaptive Learning Rates: Adjusts learning rates based on Fisher Information, potentially leading to faster
       convergence by taking more informed steps.
       - Curvature-aware Updates: Utilizes the curvature of the loss landscape to modulate the update steps, aiming
       to improve optimization efficiency.
       - Support for Custom Modules: Designed to work with a wide range of neural network architectures by allowing
       flexible mappings between parameters and modules.

   Usage:
   AdaFisherW should be used similarly to other PyTorch optimizers, with the additional step of preparing the model
   by registering necessary hooks to compute Fisher Information:

   ```python
   import torch
   model = MyModel()
   optimizer = AdaFisherW(model, lr=1e-3, beta=0.9, gamma=[0.98, 0.008], Lambda=1e-3, weight_decay=0)

   for input, target in dataset:
       optimizer.zero_grad()
       output = model(input)
       loss = loss_fn(output, target)
       loss.backward()
       optimizer.step()
   ```
    """
    def __init__(self, 
                 model: Module,
                 lr: float = 1e-3,
                 beta: float = 0.9,
                 Lambda: float = 1e-3,
                 gammas: List = [0.92, 0.008],
                 TCov: int = 100,
                 weight_decay: float = 0,
                 dist: bool = False
                ):
        super(AdaFisherW, self).__init__(model,
                                        lr = lr,
                                        beta = beta,
                                        Lambda = Lambda,
                                        gammas = gammas,
                                        TCov = TCov,
                                        dist = dist,
                                        weight_decay = weight_decay)
    
    @no_grad()
    def _step(self, hyperparameters: Dict[str, float], param: Parameter, F_tilde: Tensor):
        """
        Performs a single optimization step for one parameter tensor, applying updates according
        to the AdaFisher algorithm and hyperparameters provided.

        This method integrates the curvature information into the update rule and utilizes a weight
        decay approach similar to AdamW, directly applying the decay to the parameters before the
        gradient update. This approach decouples weight decay from the optimization steps, allowing
        for more direct control over regularization independently from the learning rate.

        Parameters:
        - hyperparameters (dict): A dictionary containing optimization hyperparameters, including
                                  'lr' for learning rate, 'beta' for the exponential moving average
                                  coefficient, and 'eps' for numerical stability.
        - param (Parameter): The parameter tensor to be updated.
        - F_tilde (Tensor): The pre-computed update based on the Fisher information and the gradient
                           information of the parameter.

        Note:
        The method initializes state for each parameter during the first call, storing the first
        and second moment estimators as well as a step counter to adjust the bias correction terms.
        The weight decay is applied in a manner similar to AdamW, affecting the parameter directly
        before the gradient update, which improves regularization by decoupling it from the scale of
        the gradients.

        The update rule incorporates curvature information through the 'F_tilde' tensor, which
        influences the step size and direction based on the estimated Fisher Information Matrix,
        providing a more informed update step that accounts for the loss surface's curvature.
        """
        grad = param.grad
        state = self.state[param]
        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = zeros_like(
                param, memory_format=preserve_format)
        exp_avg = state['exp_avg']
        beta1 = hyperparameters['beta']
        state['step'] += 1
        bias_correction1 = 1 - beta1 ** state['step']
        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        param.data -= hyperparameters['lr'] * (exp_avg / bias_correction1 / F_tilde + hyperparameters['weight_decay'] * param.data)


    @no_grad()
    def step(self, closure: Union[None, Callable[[], Tensor]] = None):
        """
        Performs a single optimization step across all parameter groups of the model.
        This method iterates over each parameter group, updating parameters based on their gradients, a set of
        hyperparameters, and update tensors calculated for each corresponding module. The update process carefully
        tracks the relationship between parameters and their corresponding modules, using index tracking and dimension
        checks to ensure updates are applied accurately.

        Arguments:
            - closure (callable, optional): A closure that reevaluates the model and returns the loss. Currently,
            the use of closure is not supported, and attempting to provide one will raise a NotImplementedError.

        The update process involves:
            1. Iterating through each parameter group and its corresponding hyperparameters.
            2. For each parameter, checking if a gradient is available. If not, it skips to the next parameter.
            3. Using index tracking to associate parameters with their corresponding modules, considering cases
            where parameters do not have gradients or are part of buffer counts.
            4. Performing a dimensionality check to ensure compatibility between the parameter and module,
            then calculating the update curvature tensor.
            5. Applying the update curvature using a custom step function that incorporates the calculated update
            tensor and hyperparameters.
        """
        if closure is not None:
            raise NotImplementedError("Closure not supported.")
        for group in self.param_groups:
            idx_param, idx_module, buffer_count = 0, 0, 0
            param, hyperparameters = group['params'], {"weight_decay": group['weight_decay'], "beta": group['beta'], "lr": group['lr']}
            for _ in range(len(self.modules)):
                if param[idx_param].grad is None:
                    idx_param += 1
                    if param[idx_param].ndim > 1:
                        idx_module += 1
                    else:
                        buffer_count += 1
                    if buffer_count == 2:
                        idx_module += 1
                        buffer_count = 0
                    continue
                m = self.modules[idx_module]
                if self._check_dim(param, idx_module, idx_param):
                    F_tilde = self._get_F_tilde(m)
                    idx_module += 1
                else:
                    F_tilde = ones_like(param[idx_param])
                if isinstance(F_tilde, list):
                    for F_tilde_i in F_tilde:
                        self._step(hyperparameters, param[idx_param], F_tilde_i)
                        idx_param += 1
                else:
                    self._step(hyperparameters, param[idx_param], F_tilde)
                    idx_param += 1
        self.steps += 1