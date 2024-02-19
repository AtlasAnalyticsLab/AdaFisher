from argparse import Namespace as APNamespace, _SubParsersAction, \
    ArgumentParser
from typing import Tuple, Dict, Any, List
from datetime import datetime
from pathlib import Path
from optimizers.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, OneCycleLR, CosineAnnealingLR
# import logging
import warnings
import time
import os
import json

import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from sklearn import metrics
import numpy as np
import torch
import yaml
import nvidia_smi
from optimizers import get_optimizer_scheduler
from optimizers.AdaHessian import Adahessian
from optimizers.sam import SAM
from utils.early_stop import EarlyStop
from models import get_network
from utils.utils import parse_config
from utils.data import get_data


def args(sub_parser: _SubParsersAction):
    sub_parser.add_argument(
        '--config', dest='config',
        default='config.yaml', type=str,
        help="Set configuration file path: Default = 'config.yaml'")
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
        '--root', dest='root',
        default='./../', type=str,
        help="Set root path of project that parents all others: Default = './../'")
    sub_parser.add_argument(
        '--save-freq', default=25, type=int,
        help='Checkpoint epoch save frequency: Default = 25')
    # sub_parser.set_defaults(resume=False)
    sub_parser.add_argument(
        '--cpu', action='store_true',
        dest='cpu',
        help="Flag: CPU bound training: Default = False")
    sub_parser.set_defaults(cpu=False)
    sub_parser.add_argument(
        '--gpu', default=0, type=int,
        help='GPU id to use: Default = 0')
    sub_parser.add_argument(
        '--multiprocessing-distributed', action='store_true',
        dest='mpd',
        help='Use multi-processing distributed training to launch '
        'N processes per node, which has N GPUs. This is the '
        'fastest way to use PyTorch for either single node or '
        'multi node data parallel training: Default = False')
    sub_parser.set_defaults(mpd=False)
    sub_parser.add_argument(
        '--dist-url', default='tcp://127.0.0.1:23456', type=str,
        help="url used to set up distributed training:" +
             "Default = 'tcp://127.0.0.1:23456'")
    sub_parser.add_argument(
        '--dist-backend', default='nccl', type=str,
        help="distributed backend: Default = 'nccl'")
    sub_parser.add_argument(
        '--world-size', default=-1, type=int,
        help='Number of nodes for distributed training: Default = -1')
    sub_parser.add_argument(
        '--rank', default=-1, type=int,
        help='Node rank for distributed training: Default = -1')


class TrainingAgent:
    config: Dict[str, Any] = None
    train_loader = None
    test_loader = None
    train_sampler = None
    num_classes: int = None
    network: torch.nn.Module = None
    optimizer: torch.optim.Optimizer = None
    scheduler = None
    loss = None
    output_filename: Path = None
    checkpoint = None

    def __init__(
            self,
            config_path: Path,
            device: str,
            output_path: Path,
            data_path: Path,
            checkpoint_path: Path,
            resume: Path = None,
            save_freq: int = 25,
            gpu: int = None,
            ngpus_per_node: int = 0,
            world_size: int = -1,
            rank: int = -1,
            dist: bool = False,
            mpd: bool = False,
            dist_url: str = None,
            dist_backend: str = None) -> None:

        self.gpu = gpu
        self.mpd = mpd
        self.dist = dist
        self.rank = rank
        self.best_acc1 = 0.
        self.start_epoch = 0
        self.start_trial = 0
        self.device = device
        self.resume = resume
        self.dist_url = dist_url
        self.save_freq = save_freq
        self.world_size = world_size
        self.dist_backend = dist_backend
        self.ngpus_per_node = ngpus_per_node

        self.data_path = data_path
        self.output_path = output_path
        self.checkpoint_path = checkpoint_path

        self.load_config(config_path, data_path)
        print("Experiment Configuration")
        print("-"*45)
        for k, v in self.config.items():
            if isinstance(v, list) or isinstance(v, dict):
                print(f"    {k:<20} {v}")
            else:
                print(f"    {k:<20} {v:<20}")
        print("-"*45)

    def load_config(self, config_path: Path, data_path: Path) -> None:
        with config_path.open() as f:
            self.config = config = parse_config(
                yaml.load(f, Loader=yaml.Loader))
        if self.device == 'cpu':
            warnings.warn("Using CPU will be slow")
        elif self.dist:
            if self.gpu is not None:
                config['mini_batch_size'] = int(
                    config['mini_batch_size'] / self.ngpus_per_node)
                config['num_workers'] = int(
                    (config['num_workers'] + self.ngpus_per_node - 1) /
                    self.ngpus_per_node)
        self.train_loader, self.train_sampler, self.test_loader, self.num_classes = get_data(
                name=config['dataset'], root=data_path, mini_batch_size=config['mini_batch_size'],
                num_workers=config['num_workers'], cutout=config['cutout'], n_holes=config['n_holes'],
                length=config['cutout_length'], dist=self.dist)
        self.criterion = torch.nn.CrossEntropyLoss().cuda(self.gpu) if \
            config['loss'] == 'cross_entropy' else None
        if np.less(float(config['early_stop_threshold']), 0):
            print("Notice: early stop will not be used as it was " +
                  f"set to {config['early_stop_threshold']}, " +
                  "training till completion")
        self.early_stop = EarlyStop(
            patience=int(config['early_stop_patience']),
            threshold=float(config['early_stop_threshold']))
        # cudnn.benchmark = True # This command is time consuming for the first epochs
        if self.resume is not None:
            if self.gpu is None:
                self.checkpoint = torch.load(str(self.resume))
            else:
                self.checkpoint = torch.load(
                    str(self.resume),
                    map_location=f'cuda:{self.gpu}')
            self.start_epoch = self.checkpoint['epoch']
            self.start_trial = self.checkpoint['trial']
            self.best_acc1 = self.checkpoint['best_acc1']
            print(f'Resuming config for trial {self.start_trial} at ' +
                  f'epoch {self.start_epoch}')

    def reset(self, learning_rate: float) -> None:
        self.network = get_network(name=self.config['network'],
                                   num_classes=self.num_classes)
        if self.device == 'cpu':
            print("Resetting cpu-based network")
        elif self.dist:
            if self.gpu is not None:
                torch.cuda.set_device(self.gpu)
                self.network.cuda(self.gpu)
                self.network = torch.nn.parallel.DistributedDataParallel(
                    self.network,
                    device_ids=[self.gpu])
            else:
                self.network.cuda()
                self.network = torch.nn.parallel.DistributedDataParallel(
                    self.network)
        elif self.gpu is not None:
            torch.cuda.set_device(self.gpu)
            self.network = self.network.cuda(self.gpu)
        else:
            self.network = torch.nn.DataParallel(self.network)
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
        self.early_stop.reset()

    def train(self) -> None:
        if not isinstance(self.config['init_lr'], list):
            list_lr = [self.config['init_lr']]
        else:
            list_lr = self.config['init_lr']
        for learning_rate in list_lr:
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
                    self.output_filename = self.checkpoint['output_filename']
                else:
                    epochs = range(0, self.config['max_epochs'])
                    self.output_filename = f"date={datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_" +\
                        "results_" + f"trial={trial}_" +\
                        f"{self.config['network']}_" + f"{self.config['dataset']}_" +\
                        f"{self.config['optimizer']}_" + '_'.join([f"{k}={v}" for k, v in
                        self.config['optimizer_kwargs'].items()]) + f"_{self.config['scheduler']}" +\
                        '_'.join([f"{k}={v}" for k, v in self.config['scheduler_kwargs'].items()]) +\
                        f"_LR={learning_rate}"
                    self.checkpoint_filename = f"date={datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_" +\
                        "checkpoint_" + f"trial={trial}_" + \
                       f"{self.config['network']}_" + f"{self.config['dataset']}_" + \
                       f"{self.config['optimizer']}_" + '_'.join([f"{k}={v}" for k, v in
                        self.config['optimizer_kwargs'].items()]) + f"_{self.config['scheduler']}" + \
                       '_'.join([f"{k}={v}" for k, v in self.config['scheduler_kwargs'].items()]) + \
                       f"_LR={learning_rate}"

                lr_output_path = self.output_path / self.output_filename
                lr_checkpoint_path = self.checkpoint_path / self.checkpoint_filename
                lr_output_path.mkdir(exist_ok=True, parents=True)
                lr_checkpoint_path.mkdir(exist_ok=True, parents=True)
                self.output_filename = lr_output_path
                self.checkpoint_filename = lr_checkpoint_path
                self.run_epochs(trial, epochs)

    def on_time_results(self, results: dict, epoch: int) -> None:
        if epoch == 0:
            with open(os.path.join(self.output_filename, "config.json"), "w") as f:
                json.dump(self.config, f, indent=2)

        # Log train, val, and test losses and perplexities
        with open(os.path.join(self.output_filename, "train_loss.txt"), "a") as f:
            f.write(str(results['train_loss'])) if epoch == 0 else \
            f.write('\n' + str(results['train_loss']))
        with open(os.path.join(self.output_filename, "train_accuracy1.txt"), "a") as f:
            f.write(str(results['train_acc1'])) if epoch == 0 else \
            f.write('\n' + str(results['train_acc1']))
        with open(os.path.join(self.output_filename, "train_accuracy5.txt"), "a") as f:
            f.write(str(results['train_acc5'])) if epoch == 0 else \
            f.write('\n' + str(results['train_acc5']))
        with open(os.path.join(self.output_filename, "test_loss.txt"), "a") as f:
            f.write(str(results['test_loss'])) if epoch == 0 else \
            f.write('\n' + str(results['test_loss']))
        with open(os.path.join(self.output_filename, "test_accuracy1.txt"), "a") as f:
            f.write(str(results['test_acc1'])) if epoch == 0 else \
            f.write('\n' + str(results['test_acc1']))
        with open(os.path.join(self.output_filename, "test_accuracy5.txt"), "a") as f:
            f.write(str(results['test_acc5'])) if epoch == 0 else \
            f.write('\n' + str(results['test_acc5']))
        with open(os.path.join(self.output_filename, "tot_time.txt"), "a") as f:
            f.write(str(results['time'])) if epoch == 0 else \
            f.write('\n' + str(results['time']))

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
            if self.dist:
                self.train_sampler.set_epoch(epoch)
            start_time = time.time()
            train_loss, (train_acc1, train_acc5) = self.epoch_iteration(
                trial, epoch)
            mem_used = self.track_gpu_memory_usage()
            test_loss, (test_acc1, test_acc5) = self.validate(epoch)
            end_time = time.time()
            tot_time_epoch = end_time - start_time
            results = {"train_loss": train_loss, "train_acc1": train_acc1, "train_acc5": train_acc5,
                       "test_loss": test_loss, "test_acc1": test_acc1, "test_acc5":test_acc5, "time": tot_time_epoch,
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
                    train_acc1 * 100) +
                "Test Loss: {:.4f}% | Test Acc. {:.4f}%".format(
                    test_loss,
                    test_acc1 * 100))

            if self.early_stop(train_loss):
                print("Info: Early stop activated.")
                break
            if not self.mpd or \
                    (self.mpd and self.rank % self.ngpus_per_node == 0):
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
                    filename = f'epoch_{epoch}.pth.tar'
                    torch.save(data, str(self.checkpoint_filename / filename))
                if np.greater(test_acc1, self.best_acc1):
                    self.best_acc1 = test_acc1
                    torch.save(
                        data, str(self.checkpoint_filename / 'best.pth.tar'))

    def epoch_iteration(self, trial: int, epoch: int):
        self.network.train()
        train_loss = 0
        top1 = AverageMeter()
        top5 = AverageMeter()
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            if self.gpu is not None:
                inputs = inputs.cuda(self.gpu, non_blocking=True)
            if self.device == 'cuda':
                targets = targets.cuda(self.gpu, non_blocking=True)

            if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                self.scheduler.step(epoch + batch_idx / len(self.train_loader))
            self.optimizer.zero_grad()
            if isinstance(self.optimizer, SAM):
                outputs = self.network(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                outputs = self.network(inputs)
                self.criterion(outputs, targets).backward()
                self.optimizer.second_step(zero_grad=True)
            else:
                outputs = self.network(inputs)
                loss = self.criterion(outputs, targets)
                if isinstance(self.optimizer, Adahessian):
                    loss.backward(create_graph=True)
                else:
                    loss.backward()
                self.optimizer.step()

            train_loss += loss.item()
            acc1, acc5 = accuracy(
                outputs, targets, (1, min(self.num_classes, 5)),
                aoc=self.num_classes == 2)
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

        return train_loss / (batch_idx + 1), (top1.avg.cpu().item() / 100.,
                                              top5.avg.cpu().item() / 100.)

    def validate(self, epoch: int):
        self.network.eval()
        test_loss = 0
        top1 = AverageMeter()
        top5 = AverageMeter()
        tgts = list()
        preds = list()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                if self.gpu is not None:
                    inputs = inputs.cuda(self.gpu, non_blocking=True)
                if self.device == 'cuda':
                    targets = targets.cuda(self.gpu, non_blocking=True)
                outputs = self.network(inputs)
                if self.num_classes == 2:
                    tgts.extend(targets.tolist())
                    preds.extend(outputs[:, 1].tolist())
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                acc1, acc5 = accuracy(outputs, targets, topk=(
                    1, min(self.num_classes, 5)),
                    aoc=self.num_classes == 2)
                top1.update(acc1[0], inputs.size(0))
                top5.update(acc5[0], inputs.size(0))

        return test_loss / (batch_idx + 1), (top1.avg.cpu().item() / 100,
                                             top5.avg.cpu().item() / 100)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(outputs, targets, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.contiguous().view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def setup_dirs(args: APNamespace) -> Tuple[Path, Path, Path, Path]:
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
    return config_path, output_path, data_path, checkpoint_path,\
        Path(args.resume) if args.resume is not None else None


def main(args: APNamespace):
    print("Argument Parser Options")
    print("-"*45)
    for arg in vars(args):
        attr = getattr(args, arg)
        attr = attr if attr is not None else "None"
        print(f"    {arg:<20}: {attr:<40}")
    print("-"*45)
    args.config_path, args.output_path, \
        args.data_path, args.checkpoint_path, \
        args.resume = setup_dirs(args)
    ngpus_per_node = torch.cuda.device_count()
    args.distributed = args.mpd or args.world_size > 1
    if args.mpd:
        args.world_size *= ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu: int, ngpus_per_node: int, args: APNamespace):
    args.gpu = gpu
    if args.distributed:
        if args.mpd:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    training_agent = TrainingAgent(
        config_path=args.config_path,
        device=device,
        output_path=args.output_path,
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        resume=args.resume,
        save_freq=args.save_freq,
        gpu=args.gpu,
        ngpus_per_node=ngpus_per_node,
        world_size=args.world_size,
        rank=args.rank,
        dist=args.distributed,
        mpd=args.mpd,
        dist_url=args.dist_url,
        dist_backend=args.dist_backend)
    print(f"Info: Pytorch device is set to {training_agent.device}")
    training_agent.train()


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    args(parser)
    args = parser.parse_args()
    main(args)
