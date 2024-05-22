import torch
import numpy as np


def seed_experiment(seed):
    """Seed the pseudorandom number generator, for repeatability.

    Args:
        seed (int): random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def to_device(tensors, device):
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, dict):
        return dict(
            (key, to_device(tensor, device)) for (key, tensor) in tensors.items()
        )
    else:
        raise NotImplementedError("Unknown type {0}".format(type(tensors)))
