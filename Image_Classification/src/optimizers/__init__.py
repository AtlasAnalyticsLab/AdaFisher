from typing import Any, List
from optimizers.Adam import Adam
from optimizers.AdamW import AdamW
from optimizers.AdaHessian import Adahessian
from optimizers.AdaFisher import AdaFisher
from optimizers.AdaFisherW import AdaFisherW
from optimizers.Apollo import Apollo
from optimizers.sam import SAM
from optimizers.sgd import SGD
from optimizers.kfac import KFACOptimizer
from optimizers.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR, OneCycleLR, MultiStepLR, LinearLR
import torch


def get_optimizer_scheduler(
        optim_method: str,
        lr_scheduler: str,
        init_lr: float,
        net: Any,
        listed_params: List[Any],
        train_loader_len: int,
        mini_batch_size: int,
        max_epochs: int,
        optimizer_kwargs=dict(),
        scheduler_kwargs=dict()) -> torch.nn.Module:
    optimizer = None
    scheduler = None
    optim_processed_kwargs = {
        k: v for k, v in optimizer_kwargs.items() if v is not None}
    scheduler_processed_kwargs = {
        k: v for k, v in scheduler_kwargs.items() if v is not None}
    if optim_method == "AdaFisher":
        optimizer = AdaFisher(model=net, lr=init_lr,
                              **optim_processed_kwargs)
    elif optim_method == "AdaFisherW":
        optimizer = AdaFisherW(model=net, lr=init_lr,
                              **optim_processed_kwargs)
    elif optim_method == 'SGD':
        if 'momentum' not in optim_processed_kwargs.keys() or \
                'weight_decay' not in optim_processed_kwargs.keys():
            raise ValueError(
                "'momentum' and 'weight_decay' need to be specified for"
                " SGD optimizer in config.yaml::**kwargs")
        optimizer = SGD(
            net.parameters(), lr=init_lr,
            **optim_processed_kwargs)
    elif optim_method == 'Adam':
        optimizer = Adam(net.parameters(), lr=init_lr,
                         **optim_processed_kwargs)
    elif optim_method == 'AdamW':
        optimizer = AdamW(net.parameters(), lr=init_lr,
                            **optim_processed_kwargs)
    elif optim_method == 'AdaHessian':
        optimizer = Adahessian(net.parameters(), lr=init_lr,
                            **optim_processed_kwargs)
    elif optim_method == 'Apollo':
        optimizer = Apollo(net.parameters(), lr=init_lr,
                             **optim_processed_kwargs)
    elif optim_method == 'SAM':
        optimizer = SAM(net.parameters(), SGD, lr=init_lr,
                           **optim_processed_kwargs)
    elif optim_method == 'kfac':
        optimizer = KFACOptimizer(net, lr=init_lr,
                                  **optim_processed_kwargs)
    elif optim_method in ['Shampoo']:  # , 'kfac']:
        optimizer = SGD(
            net.parameters(),
            lr=init_lr,
            weight_decay=optim_processed_kwargs["weight_decay"],
            momentum=optim_processed_kwargs["momentum"]
        )
    else:
        raise ValueError(f"Warning: Unknown optimizer {optim_method}")
    if lr_scheduler == 'StepLR':
        if 'step_size' not in scheduler_processed_kwargs.keys() or \
                'gamma' not in scheduler_processed_kwargs.keys():
            raise ValueError(
                "'step_size' and 'gamma' need to be specified for"
                "StepLR scheduler in config.yaml::**kwargs")
        scheduler = StepLR(
            optimizer,
            **scheduler_processed_kwargs)
    elif lr_scheduler == "MultiStepLR":
        if 'milestones' not in scheduler_processed_kwargs.keys():
            raise ValueError("You need to specify milestones parameters")
        scheduler = MultiStepLR(
            optimizer,
            **scheduler_processed_kwargs
        )
    elif lr_scheduler == "CosineAnnealingLR":
        if 'T_max' not in scheduler_processed_kwargs.keys():
            raise ValueError("You need to specify the maximum number "
                             "of iterations or epochs for the cosine annealing schedule")
        scheduler = CosineAnnealingLR(
            optimizer,
            **scheduler_processed_kwargs
        )
    elif lr_scheduler == 'CosineAnnealingWarmRestarts':
        # first_restart_epochs = 25
        # increasing_factor = 1
        if 'T_0' not in scheduler_processed_kwargs.keys() or \
                'T_mult' not in scheduler_processed_kwargs.keys():
            raise ValueError(
                "'first_restart_epochs' and 'increasing_factor' need to be "
                "specified for CosineAnnealingWarmRestarts scheduler in "
                "config.yaml::**kwargs")
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_processed_kwargs['T_0'],
            T_mult=scheduler_processed_kwargs['T_mult'],
            **scheduler_processed_kwargs)
    elif lr_scheduler == 'OneCycleLR':
        scheduler = OneCycleLR(
            optimizer, max_lr=init_lr,
            steps_per_epoch=train_loader_len, epochs=max_epochs,
            **scheduler_processed_kwargs)
    elif lr_scheduler == 'LinearLR':
        scheduler = LinearLR(
            optimizer
        )
    elif lr_scheduler not in ['None', '']:
        print(f"Warning: Unknown LR scheduler {lr_scheduler}")

    return (optimizer, scheduler)
