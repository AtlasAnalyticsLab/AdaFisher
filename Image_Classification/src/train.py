from argparse import Namespace as APNamespace, _SubParsersAction, \
    ArgumentParser
from typing import Tuple, Dict, Any, List
from datetime import datetime
from pathlib import Path
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from optimizers.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, OneCycleLR, CosineAnnealingLR
# import logging
import warnings
import time
import os
import json
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group
import numpy as np
import torch
from torch.cuda.amp import GradScaler
import yaml

try:
    import nvidia_smi

    MEM_TRACKING = True
except ModuleNotFoundError:
    MEM_TRACKING = False
from optimizers import get_optimizer_scheduler
from optimizers.AdaHessian import Adahessian
from asdl.precondition import PreconditioningConfig, ShampooGradientMaker, KfacGradientMaker
from utils.early_stop import EarlyStop
from models import get_network
from utils.utils import parse_config
from utils.data import get_data
from torchvision.models import (resnet50, ResNet50_Weights, mobilenet_v3_large, MobileNet_V3_Large_Weights,
                                densenet121, DenseNet121_Weights, resnet101, ResNet101_Weights)


def args(sub_parser: _SubParsersAction):
    sub_parser.add_argument(
        '--seed', dest='seed',
        default=42, type=int,
        help="Set the random seed: Default = 42")
    sub_parser.add_argument(
        '--config', dest='config',
        default='config.yaml', type=str,
        help="Set configuration file path: Default = 'config.yaml'")
    sub_parser.add_argument(
        '--root', dest='root',
        default='./../', type=str,
        help="Set root path of project that parents all others: Default = './../'")
    sub_parser.add_argument(
        '--data', dest='data',
        default='data', type=str,
        help="Set data directory path: Default = 'data'")
    sub_parser.add_argument(
        '--output', dest='output',
        default='logs', type=str,
        help="Set output directory path: Default = 'logs'")
    sub_parser.add_argument(
        '--checkpoint', dest='checkpoint',
        default='checkpoint', type=str,
        help="Set checkpoint directory path: Default = 'checkpoint'")
    sub_parser.add_argument(
        '--resume', dest='resume',
        default=None, type=str,
        help="Set checkpoint resume path: Default = None")
    sub_parser.add_argument(
        '--save-freq', default=25, type=int,
        help='Checkpoint epoch save frequency: Default = 25')
    sub_parser.add_argument(
        '--cpu', action='store_true',
        dest='cpu',
        help="Flag: CPU bound training: Default = False")
    sub_parser.set_defaults(cpu=False)
    sub_parser.add_argument(
        '--dist', action='store_true',
        dest='dist',
        help="Distributed training: Default = False")
    sub_parser.set_defaults(dist=False)
    sub_parser.add_argument(
        '--pretrained', action='store_true',
        dest='pretrained',
        help="Pretrained weights for (Resnet50, Resnet101, DenseNet121, MobileNetV3): Default = False")


class TrainingAgent:
    config: Dict[str, Any] = None
    train_loader: DataLoader = None
    test_loader: DataLoader = None
    num_classes: int = None
    network: torch.nn.Module = None
    optimizer: torch.optim.Optimizer = None
    scheduler: torch.optim.lr_scheduler.LRScheduler = None
    criterion: torch.nn.Module = None
    output_filename: Path = None
    checkpoint = None
    early_stop = None
    gm = None

    def __init__(
            self,
            config_path: Path,
            device: str,
            output_path: Path,
            data_path: Path,
            checkpoint_path: Path,
            resume: Path = None,
            save_freq: int = 25,
            dist: bool = False,
            autocast: bool = False,
            pretrained: bool = False) -> None:

        self.dist = dist
        if self.dist:
            self.gpu = int(os.environ["LOCAL_RANK"])
        else:
            self.gpu = device
        self.best_acc1 = 0.
        self.start_epoch = 0
        self.start_trial = 0
        self.device = device
        self.autocast = autocast
        self.data_path = data_path
        self.output_path = output_path
        self.checkpoint_path = checkpoint_path
        self.resume = resume
        self.save_freq = save_freq
        self.pretrained = pretrained

        self.load_config(config_path, data_path)
        print("Experiment Configuration")
        print("-" * 45)
        for k, v in self.config.items():
            if isinstance(v, list) or isinstance(v, dict):
                print(f"    {k:<20} {v}")
            else:
                print(f"    {k:<20} {v:<20}")
        print("-" * 45)

    def load_config(self, config_path: Path, data_path: Path) -> None:
        with config_path.open() as f:
            self.config = config = parse_config(
                yaml.load(f, Loader=yaml.Loader))
        if self.device == 'cpu':
            warnings.warn("Using CPU will be slow")
        self.train_loader, self.test_loader, self.num_classes = get_data(
            name=config['dataset'], root=data_path, mini_batch_size=config['mini_batch_size'],
            num_workers=config['num_workers'], aug=config['aug'], cutout=config['cutout'], n_holes=config['n_holes'],
            length=config['cutout_length'], dist=self.dist)
        self.criterion = torch.nn.CrossEntropyLoss()
        if np.less(float(config['early_stop_threshold']), 0):
            print("Notice: early stop will not be used as it was " +
                  f"set to {config['early_stop_threshold']}, " +
                  "training till completion")
        self.early_stop = EarlyStop(
            patience=int(config['early_stop_patience']),
            threshold=float(config['early_stop_threshold']))
        cudnn.benchmark = True  # This command is time consuming for the first epochs
        self.checkpoint_path = self.create_output_dir(checkpoint=True)
        if self.resume is not None:
            if self.gpu is None:
                self.checkpoint = torch.load(str(self.resume))
            else:
                self.checkpoint = torch.load(
                    str(self.resume),
                    map_location=self.gpu)
            self.start_epoch = self.checkpoint['epoch']
            self.start_trial = self.checkpoint['trial']
            self.best_acc1 = self.checkpoint['best_acc1']
            print(f'Resuming config for trial {self.start_trial} at ' +
                  f'epoch {self.start_epoch}')

    def get_pretrained_model(self):
        if self.config['network'] == "resnet50Cifar":
            standard_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            standard_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            num_features = standard_model.fc.in_features
            standard_model.fc = nn.Linear(num_features, self.num_classes)
            self.network.load_state_dict(standard_model.state_dict(), strict=False)
        elif self.config['network'] == "resnet101Cifar":
            standard_model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
            standard_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            num_features = standard_model.fc.in_features
            standard_model.fc = nn.Linear(num_features, self.num_classes)
            self.network.load_state_dict(standard_model.state_dict(), strict=False)
        elif self.config['network'] == "mobilenetv3":
            standard_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
            standard_model.classifier[3] = torch.nn.Linear(standard_model.classifier[3].in_features, self.num_classes)
            self.network.load_state_dict(standard_model.state_dict(), strict=False)
        elif self.config['network'] == "densenet121Cifar":
            standard_model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            num_features = standard_model.classifier.in_features
            standard_model.classifier = nn.Linear(num_features, self.num_classes)
            self.network.load_state_dict(standard_model.state_dict(), strict=False)
        else:
            raise NotImplementedError(f"{self.config['network']} does not support pretrained weights yet")

    def reset(self, learning_rate: float) -> None:
        self.network = get_network(name=self.config['network'], num_classes=self.num_classes)
        for layer in self.network.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()
        if self.pretrained:
            self.get_pretrained_model()
        if self.device == 'cpu':
            print("Resetting cpu-based network")
        elif self.device == 'mps':
            self.network = self.network.to(self.gpu)
        elif self.dist:
            self.network.to(self.gpu)
            self.network = torch.nn.parallel.DistributedDataParallel(
                self.network, device_ids=[self.gpu])
        else:
            self.network = self.network.cuda(self.gpu)

        self.optimizer, self.scheduler = get_optimizer_scheduler(
            optim_method=self.config['optimizer'],
            lr_scheduler=self.config['scheduler'],
            init_lr=learning_rate,
            net=self.network,
            listed_params=list(self.network.parameters()),
            train_loader_len=len(self.train_loader),
            mini_batch_size=self.config['mini_batch_size'],
            max_epochs=self.config['max_epochs'],
            optimizer_kwargs=self.config['optimizer_kwargs'],
            scheduler_kwargs=self.config['scheduler_kwargs'])
        if self.config['optimizer'] in ['Shampoo', 'kfac']:
            damping = self.config['optimizer_kwargs']['damping']
            curvature_update_interval = self.config['optimizer_kwargs']['curvature_update_interval']
            ema_decay = self.config['optimizer_kwargs']['ema_decay']
            config = PreconditioningConfig(data_size=self.config['mini_batch_size'],
                                           damping=damping,
                                           preconditioner_upd_interval=curvature_update_interval,
                                           curvature_upd_interval=curvature_update_interval,
                                           ema_decay=ema_decay,
                                           ignore_modules=[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                                           nn.LayerNorm])
            if self.config['optimizer'] == "Shampoo":
                self.gm = ShampooGradientMaker(self.network, config)
            elif self.config['optimizer'] == "kfac":
                self.gm = KfacGradientMaker(self.network, config)
            else:
                raise ValueError(f"This optimizer is not reconized: {self.config['optimizer']}")
        self.early_stop.reset()

    def create_output_dir(self, checkpoint: bool = False):
        def create_dir(path: Path):
            if not path.exists():
                print(f"Info: Directory {path} does not exist, building")
                path.mkdir(exist_ok=True, parents=True)
            return path

        opt = self.config['optimizer']
        net = self.config['network']

        base_path = self.checkpoint_path if checkpoint else self.output_path
        output_path_opt = create_dir(base_path / Path(opt).expanduser())
        output_path_net = create_dir(output_path_opt / Path(net).expanduser())
        return output_path_net

    def train(self) -> None:
        learning_rate = self.config['init_lr']
        for trial in range(self.start_trial,
                           self.config['n_trials']):
            self.reset(learning_rate)
            if trial == self.start_trial and self.resume is not None:
                print("Resuming Network/Optimizer")
                self.network.load_state_dict(
                    self.checkpoint['state_dict_network'])
                self.optimizer.load_state_dict(
                    self.checkpoint['state_dict_optimizer'])
                self.scheduler.load_state_dict(
                    self.checkpoint['state_dict_scheduler'])
                epochs = range(self.start_epoch, self.config['max_epochs'])
                output_path_net = self.create_output_dir()
                self.output_filename = self.checkpoint['output_filename']

            else:
                epochs = range(0, self.config['max_epochs'])
                output_path_net = self.create_output_dir()
                self.output_filename = Path(
                    (f"date={datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_results_trial={trial}_"
                     f"{self.config['network']}_{self.config['dataset']}_{self.config['optimizer']}_")
                    + '_'.join([f"{k}={v}" for k, v in self.config['optimizer_kwargs'].items()]) +
                    f"_{self.config['scheduler']}" +
                    '_'.join([f"{k}={v}" for k, v in self.config['scheduler_kwargs'].items()]) +
                    f"_LR={learning_rate}")

            lr_output_path = output_path_net / self.output_filename
            lr_output_path.mkdir(exist_ok=True, parents=True)
            self.output_filename = lr_output_path
            self.run_epochs(trial, epochs)

    def on_time_results(self, results: dict, epoch: int) -> None:
        if epoch == 0:
            with open(os.path.join(self.output_filename, "config.json"), "w") as f:
                json.dump(self.config, f, indent=2)
        files_to_keys = {
            "train_loss.txt": 'train_loss',
            "train_accuracy1.txt": 'train_acc1',
            "train_accuracy5.txt": 'train_acc5',
            "test_loss.txt": 'test_loss',
            "test_accuracy1.txt": 'test_acc1',
            "test_accuracy5.txt": 'test_acc5',
            "tot_time.txt": 'time'
        }
        for filename, key in files_to_keys.items():
            mode = 'w' if epoch == 0 else 'a'
            with open(os.path.join(self.output_filename, filename), mode) as f:
                separator = '' if epoch == 0 else '\n'
                f.write(f"{separator}{results[key]}")
        if epoch == 0:
            with open(os.path.join(self.output_filename, "config.json"), "w") as f:
                json.dump(self.config, f, indent=2)
        if MEM_TRACKING:
            with open(os.path.join(self.output_filename, "mem_used.txt"), "a") as f:
                f.write(str(results['mem_used'])) if epoch == 0 else \
                    f.write('\n' + str(results['mem_used']))

    @staticmethod
    def track_gpu_memory_usage():
        nvidia_smi.nvmlInit()
        deviceCount = nvidia_smi.nvmlDeviceGetCount()
        s = ""
        for i in range(deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            used = (info.used / info.total) * 100
            if deviceCount == 0:
                s += f"\nDevice: {i} | Name: {nvidia_smi.nvmlDeviceGetName(handle)} | Mem used:{used:.2f}"
            else:
                s += f"Device: {i} | Name: {nvidia_smi.nvmlDeviceGetName(handle)} | Mem used:{used:.2f}"
        nvidia_smi.nvmlShutdown()
        return s

    def run_epochs(self, trial: int, epochs: List[int]) -> None:
        for epoch in epochs:
            start_time = time.time()
            train_loss, train_acc1, train_acc5 = self.epoch_iteration(epoch)
            mem_used = self.track_gpu_memory_usage() if MEM_TRACKING else None
            test_loss, test_acc1, test_acc5 = self.validate()
            end_time = time.time()
            tot_time_epoch = end_time - start_time
            results = {"train_loss": train_loss, "train_acc1": train_acc1, "train_acc5": train_acc5,
                       "test_loss": test_loss, "test_acc1": test_acc1, "test_acc5": test_acc5, "time": tot_time_epoch,
                       "mem_used": mem_used}
            self.on_time_results(results, epoch)
            if isinstance(self.scheduler, StepLR) or isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()
            total_time = time.time()
            scheduler_string = f" w/ {self.config['scheduler']}" if \
                self.scheduler is not None else ''
            print(
                f"{self.config['optimizer']}{scheduler_string} " +
                f"on {self.config['dataset']}: " +
                f"T {trial + 1}/{self.config['n_trials']} | " +
                f"E {epoch + 1}/{epochs[-1] + 1} Ended | " +
                "E Time: {:.3f}s | ".format(end_time - start_time) +
                "~Time Left: {:.3f}s | ".format(
                    (total_time - start_time) * (epochs[-1] - epoch)),
                "Train Loss: {:.4f}% | Train Acc. {:.4f}% | ".format(
                    train_loss,
                    train_acc1) +
                "Test Loss: {:.4f}% | Test Acc. {:.4f}%".format(
                    test_loss,
                    test_acc1))

            if self.early_stop(train_loss):
                print("Info: Early stop activated.")
                break
            if not self.dist:
                data = {'epoch': epoch + 1,
                        'trial': trial,
                        'config': self.config,
                        'state_dict_network': self.network.state_dict(),
                        'state_dict_optimizer': self.optimizer.state_dict(),
                        'state_dict_scheduler': self.scheduler.state_dict()
                        if self.scheduler is not None else None,
                        'best_acc1': self.best_acc1,
                        'output_filename': Path(self.output_filename).name}
                if epoch % self.save_freq == 0:
                    filename = f'trial_{trial}_epoch_{epoch}.pth.tar'
                    torch.save(data, str(self.checkpoint_path / filename))
                if torch.greater(test_acc1, self.best_acc1):
                    self.best_acc1 = test_acc1
                    torch.save(
                        data, str(self.checkpoint_path / 'best.pth.tar'))
                torch.save(data, str(self.checkpoint_path / 'last.pth.tar'))

    def epoch_iteration(self, epoch: int):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        precision = torch.float16 if self.config['precision'] == 'fp16' else torch.float32
        scaler = GradScaler(enabled=True) if self.autocast else None
        # switch to train mode
        self.network.train()
        for i, (inputs, targets) in enumerate(self.train_loader):
            targets = targets.to(self.gpu)
            inputs = inputs.to(self.gpu)
            if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                self.scheduler.step(epoch + i / len(self.train_loader))
            self.optimizer.zero_grad()
            if self.config['optimizer'] in ["Shampoo", "kfac"]:
                dummy_y = self.gm.setup_model_call(self.network, inputs)
                self.gm.setup_loss_call(self.criterion, dummy_y, targets)
                outputs, loss = self.gm.forward_and_backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                               self.config['optimizer_kwargs']['clipping_norm'])
                self.optimizer.step()
            else:
                if self.autocast:
                    with torch.autocast(device_type=self.gpu, dtype=precision, enabled=True):
                        outputs = self.network(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.network(inputs)
                    loss = self.criterion(outputs, targets)
                if isinstance(self.optimizer, Adahessian):
                    scaler.scale(loss).backward(create_graph=True) if self.autocast else loss.backward(
                        create_graph=True)
                else:
                    scaler.scale(loss).backward() if self.autocast else loss.backward()
                scaler.step(self.optimizer) if self.autocast else self.optimizer.step()
                if self.autocast:
                    scaler.update()

            prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
        return losses.avg, top1.avg, top5.avg

    def validate(self):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.network.eval()
        for i, (inputs, targets) in enumerate(self.test_loader):
            targets = targets.to(self.gpu)
            inputs = inputs.to(self.gpu)
            with torch.no_grad():
                # compute output
                outputs = self.network(inputs)
                loss = self.criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
        return losses.avg, top1.avg, top5.avg


class AverageMeter:
    """Computes and stores the average and current value"""
    val: float
    avg: float
    sum: float
    count: int

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def setup_dirs(args: APNamespace) -> Tuple[Path, Path, Path, Path, Path | None]:
    root_path = Path(args.root).expanduser()
    config_path = root_path / Path(args.config).expanduser()
    data_path = root_path / Path(args.data).expanduser()
    output_path = root_path / Path(args.output).expanduser()
    checkpoint_path = root_path / Path(args.checkpoint).expanduser()
    if not config_path.exists():
        raise ValueError(f"Info: Config path {config_path} does not exist")
    if not data_path.exists():
        print(f"Info: Data dir {data_path} does not exist, building")
        data_path.mkdir(exist_ok=True, parents=True)
    if not output_path.exists():
        print(f"Info: Output dir {output_path} does not exist, building")
        output_path.mkdir(exist_ok=True, parents=True)
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(exist_ok=True, parents=True)
    if args.resume is not None:
        if not Path(args.resume).exists():
            raise ValueError("Resume path does not exist")
    return config_path, output_path, data_path, checkpoint_path, Path(args.resume) if args.resume is not None else None


def main(args: APNamespace):
    print("Argument Parser Options")
    print("-" * 45)
    for arg in vars(args):
        attr = getattr(args, arg)
        attr = attr if attr is not None else "None"
        print(f"    {arg:<20}: {attr:<40}")
    print("-" * 45)
    args.config_path, args.output_path, args.data_path, args.checkpoint_path, args.resume = setup_dirs(args)
    if args.dist:
        init_process_group(backend='nccl')
        main_worker(args)
        destroy_process_group()
    else:
        main_worker(args)


def main_worker(args: APNamespace):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.cpu:
        device = 'cuda'
        autocast = True
    elif torch.backends.mps.is_available() and not args.cpu:
        device = 'mps'
        autocast = False
    else:
        device = 'cpu'
        autocast = False
    training_agent = TrainingAgent(
        config_path=args.config_path,
        device=device,
        output_path=args.output_path,
        data_path=args.data_path,
        resume=args.resume,
        save_freq=args.save_freq,
        checkpoint_path=args.checkpoint_path,
        dist=args.dist,
        autocast=autocast,
        pretrained=args.pretrained)
    print(f"Info: Pytorch device is set to {training_agent.device}")
    training_agent.train()


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    args(parser)
    args = parser.parse_args()
    main(args)
