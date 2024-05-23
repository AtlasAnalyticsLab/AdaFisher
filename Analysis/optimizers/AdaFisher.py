"""Implementation of AdaFisher"""

from typing import Callable, Dict, List, Union, Tuple, Type
from torch import (Tensor, kron, is_grad_enabled, no_grad, zeros_like,
                   preserve_format, ones_like, diag)
from torch.optim import Optimizer
from torch.nn import Module, Parameter
from optimizers.AdaFisher_utils import (Compute_H_bar, Compute_S, update_running_avg, MinMaxNormalization)
import pickle
__all__ = ['AdaFisher']


class AdaFisher(Optimizer):
    """AdaFisher Optimizer: An adaptive learning rate optimizer that leverages Fisher Information for parameter updates.

       The AdaFisher optimizer extends traditional optimization techniques by incorporating Fisher Information to
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
    SUPPORTED_MODULES: Tuple[Type[str], ...] = ("Linear", "Conv2d", "BatchNorm2d", "LayerNorm")

    def __init__(self,
                 model: Module,
                 lr: float = 1e-3,
                 beta: float = 0.9,
                 Lambda: float = 1e-3,
                 gammas: list = [0.92, 0.008],
                 TCov: int = 100,
                 weight_decay: float = 0
                 ):
        """Initializes the AdaFisher optimizer.

                Parameters:
                - model (Module): The neural network model to optimize.
                - lr (float, optional): Learning rate. Default is 1e-3.
                - beta (float, optional): The beta1 parameter in the optimizer, controlling the moving average of
                gradients. Default is 0.9.
                - gammas (list, optional): The gammas parameter in the optimizer, controlling the moving average of
                kronecker factors H and S. Default is [0.92, 0.008].
                - Lambda (float, optional): Tikhonov Damping term added to the Fisher Information Matrix (FIM)
                to stabilize inversion. Default is 1e-3.
                - TCov (int, optional): Time interval for updating the covariance matrices. Default is 100.
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
        self.steps = 0

        # store vectors for the pre-conditioner
        self.H_bar_D: Dict[Module, Tensor] = {}
        self.S_D: Dict[Module, Tensor] = {}
        # Here we store the full Kronecker factors for analysis
        self.H_bar: Dict[str, Tensor] = {}
        self.S: Dict[str, Tensor] = {}
        # save the layers of the network where FIM can be computed
        self.modules: List[Module] = []
        self.Compute_H_bar = Compute_H_bar()
        self.Compute_S = Compute_S()
        self._prepare_model()
        super(AdaFisher, self).__init__(model.parameters(), defaults)


    def _save_input(self, module: Module, input: Tensor, output: Tensor):
        """
        Captures and updates the diagonal and full matrix elements of the activation covariance matrix for a given
        module at specified intervals. This method is part of the AdaFisher optimizer's mechanism to
        incorporate curvature information from the model's activations into the optimization process.

        Specifically, it computes the diagonal and full matrix of the activation covariance matrix for the current
        input to the module. This computation is performed at intervals defined by the `TCov` attribute
        of the optimizer. The method then updates a running average of these matrices,
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
            H_bar_i, H_bar_D_i = self.Compute_H_bar(input[0].data, module)
            if self.steps == 0:
                self.H_bar_D[module] = H_bar_D_i.new(H_bar_D_i.size(0)).fill_(1)
                self.H_bar[module] = diag(H_bar_i.new(H_bar_i.size(0)).fill_(1))
            update_running_avg(MinMaxNormalization(H_bar_D_i), self.H_bar_D[module], self.gammas)
            update_running_avg(MinMaxNormalization(H_bar_i), self.H_bar[module], self.gammas)

    def _save_grad_output(self, module: Module, grad_input: Tensor, grad_output: Tensor):
        """
        Updates the optimizer's internal state by capturing and maintaining the diagonal and full matrix elements
        of the gradient covariance matrix for a given module. This operation is conducted at
        intervals specified by the `TCov` attribute, focusing on the gradients of the output
        with respect to the module's parameters.

        At each specified interval, this method computes the diagonal and full matrix of the gradient covariance
        matrix based on the gradients of the module's output. This information is crucial for
        adapting the optimizer's behavior according to the curvature of the loss surface, as
        reflected by the gradients' distribution. The computed diagonal and full matrix elements are then used to
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
            S_i, S_D_i = self.Compute_S(grad_output[0].data, module)
            if self.steps == 0:
                self.S_D[module] = S_D_i.new(S_D_i.size(0)).fill_(1)
                self.S[module] = diag(S_D_i.new(S_D_i.size(0)).fill_(1))
            update_running_avg(MinMaxNormalization(S_D_i), self.S_D[module], self.gammas)
            update_running_avg(MinMaxNormalization(S_i), self.S[module], self.gammas)
            

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
        if self.steps % 50 == 0:
            with open(f'H_bar_resnet/H_bar_{self.steps}.pkl', 'wb') as f:
                pickle.dump(self.H_bar, f)
            with open(f'S_resnet/S_{self.steps}.pkl', 'wb') as f:
                pickle.dump(self.S, f)
        self.steps += 1