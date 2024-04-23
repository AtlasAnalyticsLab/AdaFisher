"""Implements AdaFisherW"""

from typing import Callable, Dict, List, Union, Tuple, Type
from torch import (Tensor, kron, is_grad_enabled, no_grad, zeros_like,
                   preserve_format, ones_like)
from torch.optim import Optimizer
from torch.nn import Module, Parameter
from optimizers.AdaFisher_utils import (Compute_A_Diag, Compute_G_Diag, update_running_avg, MinMaxNormalization)

__all__ = ['AdaFisherW']


class AdaFisherW(Optimizer):
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
   optimizer = AdaFisherW(model, lr=1e-3, beta=0.9, gamma=0.91, Lambda=1e-3, eps=1e-8, TCov=10, weight_decay=0)

   for input, target in dataset:
       optimizer.zero_grad()
       output = model(input)
       loss = loss_fn(output, target)
       loss.backward()
       optimizer.step()
   ```
    """
    SUPPORTED_MODULES: Tuple[Type[str], ...] = ("Linear", "Conv2d", "BatchNorm2d", "LayerNorm")

    def __init__(self,
                 model: Module,
                 lr: float = 1e-3,
                 beta: float = 0.9,
                 Lambda: float = 1e-3,
                 gammas: list = [0.98, 0.0008],
                 eps: float = 1e-8,
                 TCov: int = 10,
                 weight_decay: float = 0
                 ):
        """Initializes the AdaFisherW optimizer.

        Parameters:
            - model (Module): The neural network model to optimize.
            - lr (float, optional): Learning rate. Default is 1e-3.
            - beta (float, optional): The beta1 parameter in the optimizer, controlling the moving average of
            gradients. Default is 0.9.
            - gamma (float, optional): The gamma parameter in the optimizer, controlling the moving average of
            kronecker factors A and G. Default is 0.91.
            - Lambda (float, optional): Tikhonov Damping term added to the Fisher Information Matrix (FIM)
            to stabilize inversion. Default is 1e-3.
            - eps (float, optional): Term added to the denominator to improve numerical stability. Default is 1e-8.
            - TCov (int, optional): Time interval for updating the covariance matrices. Default is 10.
            - weight_decay (float, optional): Weight decay coefficient. Default is 0.

        Raises:
            - ValueError: If any of the provided parameters are out of their expected ranges.

        This optimizer extends the traditional optimization methods by incorporating Fisher Information to
        adaptively adjust the learning rates based on the curvature of the loss landscape. It specifically
        tracks the covariance of activation and gradient of the pre-activation functions across layers, aiming
        to enhance convergence stability and speed.

        The optimizer prepares the model for computing the Fisher Information Matrix by identifying relevant
        layers and initializing necessary data structures for tracking the covariance of activations and
        gradients. It then sets itself up by inheriting from a base optimizer class with the specified parameters.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if not 0.0 <= gammas[0] < 1.0:
            raise ValueError(f"Invalid gamma parameter at index 0: {gammas[0]}")
        if not 0.0 <= gammas[1] < 1.0:
            raise ValueError(f"Invalid gamma parameter at index 1: {gammas[1]}")
        if not TCov > 0:
            raise ValueError(f"Invalid TCov parameter: {TCov}")
        defaults = dict(lr=lr, beta=beta, eps=eps,
                        weight_decay=weight_decay)

        self.gammas = gammas
        self.Lambda = Lambda
        self.model = model
        self.TCov = TCov
        self.steps = 0

        # store vectors for the pre-conditioner
        self.diag_A: Dict[str, Tensor] = {}
        self.diag_G: Dict[str, Tensor] = {}
        # save the layers of the network where FIM can be computed
        self.modules: List[Module] = []
        self.Cov_AMatrix = Compute_A_Diag()
        self.Cov_GMatrix = Compute_G_Diag()
        self._prepare_model()
        super(AdaFisherW, self).__init__(model.parameters(), defaults)

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
                               defined in the `Cov_AMatrix` method.
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
            diag_A = self.Cov_AMatrix(input[0].data, module)
            if self.steps == 0:
                self.diag_A[module] = diag_A.new(diag_A.size(0)).fill_(1)
            update_running_avg(MinMaxNormalization(diag_A), self.diag_A[module], self.gammas)

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
            diag_G = self.Cov_GMatrix(grad_output[0].data, module)
            if self.steps == 0:
                self.diag_G[module] = diag_G.new(diag_G.size(0)).fill_(1)
            update_running_avg(MinMaxNormalization(diag_G, 10), self.diag_G[module], self.gammas)

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
              (e.g., `relu(inplace=True)` or '*=') within the modules. In-place operations can disrupt the proper
              functioning of the forward and backward hooks, leading to incorrect computations or even runtime errors.
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

    def _get_update(self, module: Module):
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
              updates for the weights and the bias. For modules without bias, a single tensor update
              is returned.

        Note:
        This method directly manipulates the gradients of the module's parameters based on the
        computed Fisher Information and is a key component of the AdaFisher optimizer's strategy
        to leverage curvature information for more efficient and effective optimization.
        """
        # Fisher Computation
        H = kron(self.diag_A[module].unsqueeze(1), self.diag_G[module].unsqueeze(0)).t() + self.Lambda
        if module.bias is not None:
            H = [H[:, :-1], H[:, -1:]]
            H[0] = H[0].view(*module.weight.grad.data.size())
            H[1] = H[1].view(*module.bias.grad.data.size())
            return H
        else:
            return H.reshape(module.weight.grad.data.size())

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
        module_weight = self.modules[idx_module].weight
        module_bias = self.modules[idx_module].bias
        if module_bias is not None:
            if params.data.size() != module_weight.data.size() and params.data.size() != module_bias.data.size():
                return False
            else:
                return True
        else:
            if params.data.size() != module_weight.data.size():
                return False
            else:
                return True

    @no_grad()
    def _step(self, hyperparameters: Dict[str, float], param: Parameter, update: Tensor):
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
        - update (Tensor): The pre-computed update based on the Fisher information and the gradient
                           information of the parameter.

        Note:
        The method initializes state for each parameter during the first call, storing the first
        and second moment estimators as well as a step counter to adjust the bias correction terms.
        The weight decay is applied in a manner similar to AdamW, affecting the parameter directly
        before the gradient update, which improves regularization by decoupling it from the scale of
        the gradients.

        The update rule incorporates curvature information through the 'update' tensor, which
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
        param.data -= hyperparameters['lr'] * (exp_avg / bias_correction1 / update + hyperparameters['weight_decay'] * param.data)


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
            param, hyperparameters = group['params'], {"weight_decay": group['weight_decay'], "eps": group['eps'],
                                                       "beta": group['beta'], "lr": group['lr']}
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
                    update = self._get_update(m)
                    idx_module += 1
                else:
                    update = ones_like(param[idx_param])
                if isinstance(update, list):
                    for u in update:
                        self._step(hyperparameters, param[idx_param], u)
                        idx_param += 1
                else:
                    self._step(hyperparameters, param[idx_param], update)
                    idx_param += 1
        self.steps += 1