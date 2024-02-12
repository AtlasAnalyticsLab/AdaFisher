import torch.nn as nn
import torch
import torch.nn.functional as F


def update_running_avg(new, current, alpha):
    """Compute running average of matrix in-place

    current = alpha*new + (1-alpha)*current
    """
    current *= alpha / (1 - alpha)
    current += new
    current *= (1 - alpha)


def _extract_patches(x, kernel_size, stride, padding):
    """
    :param x: The input feature maps.  (batch_size, in_c, h, w)
    :param kernel_size: the kernel size of the conv filter (tuple of two elements)
    :param stride: the stride of conv operation  (tuple of two elements)
    :param padding: number of paddings. be a tuple of two elements
    :return: (batch_size, out_h, out_w, in_c*kh*kw)
    """
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data
    x = x.unfold(dimension=2, size=kernel_size[0], step=stride[0])
    x = x.unfold(dimension=3, size=kernel_size[1], step=stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))
    return x


class Compute_A_Matrix:
    @classmethod
    def compute_cov_a(cls, a, layer):
        return cls.__call__(a, layer)

    @classmethod
    def __call__(cls, a, layer):
        if isinstance(layer, nn.Linear):
            cov_a = cls.linear(a, layer)
        elif isinstance(layer, nn.Conv2d):
            cov_a = cls.conv2d(a, layer)
        elif isinstance(layer, nn.BatchNorm2d):
            cov_a = cls.batchnorm2d(a, layer)
        else:
            raise NotImplementedError

        return cov_a

    @staticmethod
    def conv2d(a, layer):
        batch_size = a.size(0)
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
        spatial_size = a.size(2) * a.size(3)
        a = a.reshape(-1, a.size(-1))
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        return torch.einsum('ij,ij->j', a, a) / (spatial_size * batch_size)


    def linear(a, layer):
        if len(a.shape) > 2:
             a = a.reshape(-1, a.shape[-1])
        batch_size = a.size(0)
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        return torch.einsum('ij,ij->j', a, a) / batch_size

    @staticmethod
    def batchnorm2d(a, layer):
        batch_size = a.size(0)
        sum_a = torch.sum(a, dim=(0, 2, 3)).unsqueeze(1)
        sum_a = torch.cat([sum_a, sum_a.new(sum_a.size(0), 1).fill_(1)], 1)
        return torch.einsum('ij,ij->j', sum_a, sum_a) / batch_size


class Compute_G_Matrix:

    @classmethod
    def compute_cov_g(cls, g, layer):
        """
        :param g: gradient
        :param layer: the corresponding layer
        :return:
        """
        # batch_size = g.size(0)
        return cls.__call__(g, layer)

    @classmethod
    def __call__(cls, g, layer):
        if isinstance(layer, nn.Conv2d):
            cov_g = cls.conv2d(g, layer)
        elif isinstance(layer, nn.Linear):
            cov_g = cls.linear(g, layer)
        elif isinstance(layer, nn.BatchNorm2d):
            cov_g = cls.batchnorm2d(g, layer)
        else:
            cov_g = None

        return cov_g

    @staticmethod
    def conv2d(g, layer):
        # g: batch_size * n_filters * out_h * out_w
        # n_filters is actually the output dimension (analogous to Linear layer)
        spatial_size = g.size(2) * g.size(3)
        batch_size = g.shape[0]
        g = g.transpose(1, 2).transpose(2, 3)
        g = g.reshape(-1, g.size(-1))
        return torch.einsum('ij,ij->j', g, g) / (spatial_size * batch_size)

    @staticmethod
    def linear(g, layer):
        # g: batch_size * out_dim
        if len(g.shape) > 2:
            g = g.reshape(-1, g.shape[-1])
        batch_size = g.size(0)
        return torch.einsum('ij,ij->j', g, g) / batch_size

    @staticmethod
    def batchnorm2d(g, layer):
        batch_size = g.size(0)
        spatial_dim = g.size(2) * g.size(3)
        sum_g = torch.sum(g, dim=(0, 2, 3))
        return torch.einsum('i,i->i', sum_g, sum_g) / (batch_size * spatial_dim)