import torch
import torch.optim as optim
import math
from optimizers.AdaFisher_utils import (Compute_A_Matrix, Compute_G_Matrix, update_running_avg)

"""
Begining of the AdaFisher algorithm. 

"""
__all__ = ['AdaFisher']


class AdaFisher(optim.Optimizer):
    def __init__(self, model, lr: float = 1e-3, betas: tuple = (0.9, 0.999),
                 Lambda: float = 1e-3, beta3=0.95, eps: float = 1e-8,
                 TCov: int = 10, weight_decay: float = 0, kappa: float = 0.5):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= beta3 < 1.0:
            raise ValueError(f"Invalid beta3 parameter: {beta3}")
        if not 0.0 <= kappa <= 1.0:
            raise ValueError("Invalid Fisher power value: {}".format(kappa))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, kappa=kappa)

        self.beta3 = beta3
        self.Lambda = Lambda
        self.model = model
        self.TCov = TCov
        self.known_layers = ("Linear", "Conv2d", "BatchNorm2d")
        self.steps = 0

        self.cov_a_bar, self.cov_g = {}, {}  # here we want to save in a dictionary the covariance activation and
        # gradient of pre-activation function.
        self.modules = []  # we save the layers of the network where FIM can be computed

        self.Cov_AMatrix = Compute_A_Matrix()
        self.Cov_GMatrix = Compute_G_Matrix()
        self._prepare_model()
        super(AdaFisher, self).__init__(model.parameters(), defaults)

    def _save_input(self, module, input, output):  # the aim of this function is to save the activation function
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            cov_a_bar = self.Cov_AMatrix(input[0].data, module)
            if self.steps == 0:
                self.cov_a_bar[module] = cov_a_bar.new(cov_a_bar.size(0)).fill_(1)
            update_running_avg(cov_a_bar, self.cov_a_bar[module], self.beta3)

    def _save_grad_output(self, module, grad_input, grad_output):  # the aim of this function is to save the gradient
        # of the pre-activation function
        cov_g = self.Cov_GMatrix(grad_output[0].data, module)  # the grad_input corresponds to the gradient of the
        # pre-activation function for this layer. Initialize buffers                                   # the
        # grad_output corresponds to the gradient of the activation function for this layer.
        if self.steps % self.TCov == 0:
            if self.steps == 0:
                self.cov_g[module] = cov_g.new(cov_g.size(0)).fill_(1)
            update_running_avg(cov_g, self.cov_g[module], self.beta3)

    def _prepare_model(self):
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_layers:  # if the layer name is among the known layers (Linear, Conv2d and Batchnorm2d)
                self.modules.append(module)
                module.register_forward_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad_output)

    def _get_update(self, m, Lambda):
        v = self.cov_g[m].unsqueeze(1) * self.cov_a_bar[m].unsqueeze(0) + Lambda
        if m.bias is not None:
            v = [v[:, :-1], v[:, -1:]]
            v[0].reshape(m.weight.grad.data.size()); v[1].reshape(m.bias.grad.data.size())
            v[0].squeeze_(1); v[1].squeeze_(1)
            return v
        else:
            return v.reshape(m.weight.grad.data.size())

    @torch.no_grad()
    def _step(self, group, p, j):
        grad = p.grad
        state = self.state[p]
        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(
                p, memory_format=torch.preserve_format)
            # Exponential moving average of squared of the Curvature information
            state['exp_avg_Fisher'] = torch.zeros_like(
                p, memory_format=torch.preserve_format)
        exp_avg, exp_avg_Fisher = state['exp_avg'], state['exp_avg_Fisher']
        beta1, beta2 = group['betas']
        state['step'] += 1
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        if group['weight_decay'] != 0 and self.steps >= 20 * self.TCov:
            grad = grad.add(p, alpha=group['weight_decay'])
        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_Fisher.mul_(beta2).addcmul_(j, j, value=1 - beta2)
        k = group['kappa']
        denom = ((exp_avg_Fisher.sqrt() ** k) /
                 math.sqrt(bias_correction2) ** k).add_(group['eps'])
        step_size = group['lr'] / bias_correction1
        p.addcdiv_(exp_avg, denom, value=-step_size)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            idx_p = 0
            for m_idx in range(len(self.modules)):
                if group['params'][idx_p] is None:
                    continue
                m = self.modules[m_idx]
                v = self._get_update(m, self.Lambda)
                if isinstance(v, list): # bias and weight
                    for wb in v:
                        p = group['params'][idx_p]
                        self._step(group, p, wb)
                        idx_p += 1
                else:
                    p = group['params'][idx_p]
                    self._step(group, p, v)
                    idx_p += 1
        self.steps += 1
        return loss
