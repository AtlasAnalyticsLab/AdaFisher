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
    Scales tensor values to range [0,1] using min-max normalization.

    Args:
        tensor: Input tensor
        epsilon: Small value to prevent division by zero (default: 1e-6)
    
    Returns:
        Normalized tensor with values in [0,1]
    """
    tensor = smart_detect_inf(tensor)
    min_tensor = tensor.min()
    max_tensor = tensor.max()
    range_tensor = max_tensor - min_tensor
    return tensor.add_(-min_tensor).div_(range_tensor + epsilon)


def update_running_avg(new: Tensor, current: Tensor, gamma: float):
    """
    Updates exponential moving average of parameters in-place.

    Args:
        new: New parameter values
        current: Current parameter values to update
        gamma: Smoothing coefficient (0-1)
    """
    current *= gamma * 1e-1
    current += (gamma * 1e-2) * new


def _extract_patches(x: Tensor, kernel_size: Tuple[int],
                     stride: Tuple[int],
                     padding: Tuple[int],
                     groups: int) -> Tensor:

    """
    Extracts sliding window patches from input feature maps for convolution operations.

    Args:
        x: Input tensor (batch_size, in_channels, height, width)
        kernel_size: Height and width of kernel (h, w)
        stride: Step size for sliding window (h, w)
        padding: Input padding (h, w)
        groups: Number of groups for grouped convolution

    Returns:
        Reshaped patches tensor (batch_size, output_h, output_w, in_channels * kernel_h * kernel_w)
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


class Compute_H_D:
    """
    Computes diagonal elements of activation covariance matrices for different neural network layers.
    """

    @classmethod
    def compute_H_D(cls, h, layer) -> Tensor:
        """
        Computes diagonal of activation covariance matrix.

        Args:
            h: Input activations
            layer: PyTorch layer (Linear, Conv2d, BatchNorm2d, or LayerNorm)
        
        Returns:
            Diagonal elements of covariance matrix
        """
        return cls.__call__(h, layer)

    @classmethod
    def __call__(cls, h: Tensor, layer: Module) -> Tensor:
        """
        Delegates computation to layer-specific methods.

        Args:
            h: Input activations
            layer: PyTorch layer
        
        Returns:
            Covariance matrix diagonal
        """
        if isinstance(layer, Linear):
            H_D = cls.linear(h, layer)
        elif isinstance(layer, Conv2d):
            H_D = cls.conv2d(h, layer)
        elif isinstance(layer, BatchNorm2d):
            H_D = cls.batchnorm2d(h, layer)
        elif isinstance(layer, LayerNorm):
            H_D = cls.layernorm(h, layer)
        else:
            raise NotImplementedError

        return H_D

    @staticmethod
    def conv2d(h: Tensor, layer: Conv2d) -> Tensor:
        """
        Computes covariance diagonal for Conv2d layer.

        Args:
            h: Input (batch_size, in_channels, height, width)
            layer: Conv2d layer
        
        Returns:
            Covariance diagonal
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
        Computes covariance diagonal for Linear layer.

        Args:
            h: Input activations
            layer: Linear layer
        
        Returns:
            Covariance diagonal
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
        Computes covariance diagonal for BatchNorm2d layer.

        Args:
            h: Input activations
            layer: BatchNorm2d layer
        
        Returns:
            Covariance diagonal
        """
        batch_size, spatial_size = h.size(0), h.size(2) * h.size(3)
        sum_h = sum(h, dim=(0, 2, 3)).unsqueeze(1) / (spatial_size ** 2)
        h_bar = cat([sum_h, sum_h.new(sum_h.size(0), 1).fill_(1)], 1) # 1st column for H of beta parameter
        return einsum('ij,ij->j', h_bar, h_bar) / (batch_size ** 2)

    @staticmethod
    def layernorm(h: Tensor, layer: LayerNorm) -> Tensor:
        """
        Computes covariance diagonal for LayerNorm layer.

        Args:
            h: Input activations
            layer: LayerNorm layer
        
        Returns:
            Covariance diagonal
        """
        dim_to_reduce = [d for d in range(h.ndim) if d != 1] # T in the paper
        batch_size, dim_norm = h.shape[0], prod([h.shape[dim] for dim in dim_to_reduce if dim != 0])
        sum_h = sum(h, dim=dim_to_reduce).unsqueeze(1) / (dim_norm ** 2)
        h_bar = cat([sum_h, sum_h.new(sum_h.size(0), 1).fill_(1)], 1) # 1st column for H of beta parameter
        return einsum('ij,ij->j', h_bar, h_bar) / (batch_size ** 2)


class Compute_S_D:
    """
    Computes diagonal elements of gradient covariance matrices for different neural network layers.
    """

    @classmethod
    def compute_S_D(cls, s: Tensor, layer: Module) -> Tensor:
        """
        Computes gradient covariance matrix diagonal.

        Args:
            s: Output gradients
            layer: PyTorch layer (Conv2d, Linear, BatchNorm2d, or LayerNorm)
        
        Returns:
            Diagonal elements of gradient covariance matrix
        """
        return cls.__call__(s, layer)

    @classmethod
    def __call__(cls, s: Tensor, layer: Module) -> Tensor:
        """
        Delegates computation to layer-specific methods.

        Args:
            s: Output gradients
            layer: PyTorch layer
        
        Returns:
            Covariance matrix diagonal
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
        Computes gradient covariance diagonal for Conv2d.

        Args:
            s: Gradients (batch_size, n_filters, out_h, out_w)
            layer: Conv2d layer
        
        Returns:
            Covariance diagonal
        """
        batch_size = s.shape[0]
        spatial_size = s.size(2) * s.size(3)
        s = s.transpose(1, 2).transpose(2, 3)
        s = s.reshape(-1, s.size(-1))
        return einsum('ij,ij->j', s, s) / (batch_size * spatial_size)

    @staticmethod
    def linear(s: Tensor, layer: Linear) -> Tensor:
        """
        Computes gradient covariance diagonal for Linear layer.

        Args:
            s: Output gradients
            layer: Linear layer
        
        Returns:
            Covariance diagonal
        """
        if len(s.shape) > 2:
            s = s.reshape(-1, s.shape[-1])
        batch_size = s.size(0)
        return einsum('ij,ij->j', s, s) / batch_size

    @staticmethod
    def batchnorm2d(s: Tensor, layer: BatchNorm2d) -> Tensor:
        """
        Computes gradient covariance diagonal for BatchNorm2d.

        Args:
            s: Output gradients
            layer: BatchNorm2d layer
        
        Returns:
            Covariance diagonal
        """
        batch_size = s.size(0)
        sum_s = sum(s, dim=(0, 2, 3))
        return einsum('i,i->i', sum_s, sum_s) / batch_size

    @staticmethod
    def layernorm(s: Tensor, layer: LayerNorm) -> Tensor:
        """
        Computes gradient covariance diagonal for LayerNorm.

        Args:
            s: Output gradients
            layer: LayerNorm layer
        
        Returns:
            Covariance diagonal
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
                 gamma: float = 0.8,
                 TCov: int = 100,
                 weight_decay: float = 0,
                 dist: bool = False
                 ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma parameter: {gamma}")
        if not TCov > 0:
            raise ValueError(f"Invalid TCov parameter: {TCov}")
        defaults = dict(lr=lr, beta=beta,
                        weight_decay=weight_decay)

        self.gamma = gamma
        self.Lambda = Lambda
        self.model = model
        self.TCov = TCov
        self.dist = dist
        self.steps = 0

        # store vectors for the pre-conditioner
        self.H_D: Dict[Module, Tensor] = {}
        self.S_D: Dict[Module, Tensor] = {}
        # save the layers of the network where FIM can be computed
        self.modules: List[Module] = []
        self.Compute_H_D = Compute_H_D()
        self.Compute_S_D = Compute_S_D()
        self._prepare_model()
        super(AdaFisherBackBone, self).__init__(model.parameters(), defaults)

    def _save_input(self, module: Module, input: Tensor, output: Tensor):
        """
        Updates diagonal elements of activation covariance matrix periodically.

        Args:
            module: Neural network layer
            input: Layer input tensor (first element used)
            output: Layer output tensor (unused)

         Note:
            Attaches as forward hook to track layer activations for AdaFisher optimization.
        """
        if is_grad_enabled() and self.steps % self.TCov == 0:
            H_D_i = self.Compute_H_D(input[0].data, module)
            if self.steps == 0:
                self.H_D[module] = H_D_i.new(H_D_i.size(0)).fill_(1)
            update_running_avg(MinMaxNormalization(H_D_i), self.H_D[module], self.gamma)
            

    def _save_grad_output(self, module: Module, grad_input: Tensor, grad_output: Tensor):
        """
        Updates diagonal elements of gradient covariance matrix periodically.

        Args:
            module: Neural network layer
            grad_input: Input gradients (unused)
            grad_output: Output gradients (first element used)

        Note:
            Attaches as backward hook to track layer gradients for AdaFisher optimization.
        """
        if self.steps % self.TCov == 0:
            S_D_i = self.Compute_S_D(grad_output[0].data, module)
            if self.steps == 0:
                self.S_D[module] = S_D_i.new(S_D_i.size(0)).fill_(1)
            update_running_avg(MinMaxNormalization(S_D_i), self.S_D[module], self.gamma)

    def _prepare_model(self):
        """
        Registers hooks on supported layers to track activations and gradients.

        Warning:
            Avoid in-place operations (e.g., relu(inplace=True) or '*=') in model layers 
            as they can interfere with hook functionality.
        """
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.SUPPORTED_MODULES:
                self.modules.append(module)
                module.register_forward_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad_output)
                
    def state_dict(self):
        """Save the Kronecker factors in the dictionary state of the optimizer

        Returns:
            Dict[str:List[Tensor]]: Dictionnary containing the states of the optimizer 
        """
        state_dict = super().state_dict()
        # Save H_D and S_D as lists ordered by self.modules
        state_dict['H_D'] = [self.H_D[module] for module in self.modules]
        state_dict['S_D'] = [self.S_D[module] for module in self.modules]
        state_dict['steps'] = self.steps
        return state_dict

    def load_state_dict(self, state_dict):
        """Load the Kronecker factors from the dictionary state of the optimizer

        Args:
            state_dict (Dict[str:List[Tensor]]): Dictionnary containing the states of the optimizer 
        """
        H_D_list = state_dict.pop('H_D', [])
        S_D_list = state_dict.pop('S_D', [])
        self.steps = state_dict.pop('steps', 0)
        # Load standard parameter states
        super().load_state_dict(state_dict)
        # Rebuild H_D and S_D dictionaries
        self.H_D = {}
        self.S_D = {}
        for module, h_bar, s in zip(self.modules, H_D_list, S_D_list):
            self.H_D[module] = h_bar
            self.S_D[module] = s
    
    def aggregate_kronecker_factors(self, module: Module):
        """
        Aggregates Kronecker factors across multiple GPUs in a distributed setting.

        This function performs an all-reduce operation to sum the Kronecker factors 
        `H_D` and `S_D` across all GPUs involved in the training. It then averages 
        these factors by dividing the summed values by the total number of GPUs (`world_size`).

        Args:
            module (Module): The neural network module for which the Kronecker factors 
                            `H_D` and `S_D` are being aggregated.
        """
        dist.all_reduce(self.H_D[module], op=dist.ReduceOp.SUM)
        dist.all_reduce(self.S_D[module], op=dist.ReduceOp.SUM)

        self.H_D[module] /= dist.get_world_size()
        self.S_D[module] /= dist.get_world_size()

    def _get_F_tilde(self, module: Module):
        """
        Computes Fisher Information Matrix with regularization for parameter updates.

        Args:
            module: Neural network layer
        
        Returns:
            - For layers with bias: [weight_update, bias_update]
            - For layers without bias: weight_update tensor

        Note:
            Uses Kronecker product of activation and gradient covariance diagonals.
        """
        # Fisher Computation
        if self.dist:
            self.aggregate_kronecker_factors(module = module)
        F_tilde = kron(self.H_D[module].unsqueeze(1), self.S_D[module].unsqueeze(0)).t() + self.Lambda
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
                 gamma: float = 0.8,
                 TCov: int = 100,
                 weight_decay: float = 0,
                 dist: bool = False
                ):
        super(AdaFisher, self).__init__(model,
                                        lr = lr,
                                        beta = beta,
                                        Lambda = Lambda,
                                        gamma = gamma,
                                        TCov = TCov,
                                        dist = dist,
                                        weight_decay = weight_decay)
        
    @no_grad()
    def _step(self, hyperparameters: Dict[str, float], param: Parameter, F_tilde: Tensor):
        """
        Updates a single parameter using AdaFisher optimization step.

        Args:
            hyperparameters: Dict containing 'lr', 'beta', and 'weight_decay'
            param: Parameter tensor to update
            F_tilde: Fisher information tensor for parameter
        
        Note:
            Uses bias-corrected moving averages and Fisher information for adaptive updates.
            Modifies parameter in-place.
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
        bias_correction = 1 - beta ** state['step']
        if hyperparameters['weight_decay'] != 0:
            grad = grad.add(param, alpha=hyperparameters['weight_decay'])
        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta).add_(grad, alpha= 1 - beta)
        step_size = hyperparameters['lr'] / bias_correction
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
                 gamma: float = 0.8,
                 TCov: int = 100,
                 weight_decay: float = 0,
                 dist: bool = False
                ):
        super(AdaFisherW, self).__init__(model,
                                        lr = lr,
                                        beta = beta,
                                        Lambda = Lambda,
                                        gamma = gamma,
                                        TCov = TCov,
                                        dist = dist,
                                        weight_decay = weight_decay)
    
    @no_grad()
    def _step(self, hyperparameters: Dict[str, float], param: Parameter, F_tilde: Tensor):
        """
        Updates a single parameter using AdaFisherW optimization step.

        Args:
            hyperparameters: Dict containing 'lr', 'beta', and 'weight_decay'
            param: Parameter tensor to update
            F_tilde: Fisher information tensor for parameter
        
        Note:
            Uses bias-corrected moving averages and Fisher information for adaptive updates.
            Modifies parameter in-place.
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
        bias_correction = 1 - beta ** state['step']
        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta).add_(grad, alpha= 1 - beta)
        param.data -= hyperparameters['lr'] * (exp_avg / bias_correction / F_tilde + hyperparameters['weight_decay'] * param.data)


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