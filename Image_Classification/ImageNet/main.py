import argparse
import os
import time

import torch
import torch.nn as nn
import torch.utils.data
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from optimizers.AdaFisher import AdaFisher
from optimizers.Adam import Adam
from optimizers.sgd import SGD
import torch.backends.cudnn as cudnn
from ..src.models.resnet import resnet50
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from data_loader import data_loader
from helper import AverageMeter, save_checkpoint, accuracy
from asdl.precondition import PreconditioningConfig, ShampooGradientMaker, KfacGradientMaker

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_true',
                    help='use pin memory')
parser.add_argument('--print-freq', '-f', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoitn, (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--optimizer', default='Adam', type=str,
                    help='Optimizer to use (default: Adam)')

best_prec1 = 0.0


def main():
    global args, best_prec1
    args = parser.parse_args()
    output_filename = create_output_dir(args)
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'resnet50':
        model = resnet50(num_classes=1000)
    else:
        raise NotImplementedError

    # use cuda
    model.cuda()
    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    gm = None
    if args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=args.lr,
                         weight_decay=args.weight_decay)
    elif args.optimizer == 'AdaFisher':
        optimizer = AdaFisher(model, lr=args.lr,
                          weight_decay=args.weight_decay)
    elif args.optimizer in ['Shampoo', 'kfac']:
        optimizer = SGD(model.parameters(), momentum=args.momentum, lr=args.lr, weight_decay=args.weight_decay)
        damping = 1e-12 if args.optimizer == 'Shampoo' else 1e-3
        curvature_update_interval = 1  if args.optimizer == 'Shampoo' else 10
        ema_decay = -1
        config = PreconditioningConfig(data_size=args.batch_size,
                                        damping=damping,
                                        preconditioner_upd_interval=curvature_update_interval,
                                        curvature_upd_interval=curvature_update_interval,
                                        ema_decay=ema_decay,
                                        ignore_modules=[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                                        nn.LayerNorm])
        if args.optimizer == "Shampoo":
            gm = ShampooGradientMaker(model, config)
        elif args.optimizer == "kfac":
            gm = KfacGradientMaker(model, config)
    else:
        raise NotImplementedError(f"{args.optimizer} is not yet implemented")
    
    scheduler = CosineAnnealingLR(optimizer, T_max = args.epochs)
    # optionlly resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True
    # Data loading
    train_loader, val_loader = data_loader(args.data, args.batch_size, args.workers, args.pin_memory)

    if args.evaluate:
        validate(val_loader, model, criterion, args.print_freq)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        start = time.time()
        train_top1, train_top5, train_losses = train(train_loader, model, criterion, optimizer, epoch, args, gm)
        scheduler.step()
        # evaluate on validation set
        val_top1, val_top5, val_losses = validate(val_loader, model, criterion, args.print_freq)
        end = time.time()
        results = {
            'train_loss': train_losses,
            'train_acc1': train_top1,
            'train_acc5': train_top5,
            'test_loss': val_losses,
            'test_acc1': val_top1,
            'test_acc5': val_top5,
            'time': end-start,
        }
        on_time_results(output_filename= output_filename, results=results, epoch=epoch)

        # remember the best prec@1 and save checkpoint
        is_best = val_top1 > best_prec1
        best_prec1 = max(val_top1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best, f"{args.arch}_{args.optimizer}.pth")


def create_output_dir(args):
    opt = args.optimizer
    net = args.arch
    output_path_opt = Path("logs") / Path(opt).expanduser()
    if not output_path_opt.exists():
        print(f"Info: Output dir {output_path_opt} does not exist, building")
        output_path_opt.mkdir(exist_ok=True, parents=True)
    output_path = output_path_opt / Path(net).expanduser()
    if not output_path.exists():
        print(f"Info: Output dir {output_path} does not exist, building")
        output_path.mkdir(exist_ok=True, parents=True)
    output_filename = Path(output_path) / Path(f"date={datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{net}_ImageNet_{opt}").expanduser()
    output_filename.mkdir(exist_ok=True, parents=True)
    return output_filename
        
def on_time_results(output_filename, results: dict, epoch: int) -> None:
    # Log train, val, and test losses and perplexities
    with open(f"{output_filename}/train_loss.txt", "a") as f:
        f.write(str(results['train_loss'])) if epoch == 0 else \
            f.write('\n' + str(results['train_loss']))
    with open(f"{output_filename}/train_accuracy1.txt", "a") as f:
        f.write(str(results['train_acc1'])) if epoch == 0 else \
            f.write('\n' + str(results['train_acc1']))
    with open(f"{output_filename}/train_accuracy5.txt", "a") as f:
        f.write(str(results['train_acc5'])) if epoch == 0 else \
            f.write('\n' + str(results['train_acc5']))
    with open(f"{output_filename}/test_loss.txt", "a") as f:
        f.write(str(results['test_loss'])) if epoch == 0 else \
            f.write('\n' + str(results['test_loss']))
    with open(f"{output_filename}/test_accuracy1.txt", "a") as f:
        f.write(str(results['test_acc1'])) if epoch == 0 else \
            f.write('\n' + str(results['test_acc1']))
    with open(f"{output_filename}/test_accuracy5.txt", "a") as f:
        f.write(str(results['test_acc5'])) if epoch == 0 else \
            f.write('\n' + str(results['test_acc5']))
    with open(f"{output_filename}/tot_time.txt", "a") as f:
        f.write(str(results['time'])) if epoch == 0 else \
            f.write('\n' + str(results['time']))

def train(train_loader, model, criterion, optimizer, epoch, args, gm):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input = input.cuda()
        optimizer.zero_grad()
        # compute output
        if args.optimizer in ['Shampoo', 'kfac']:
            dummy_y = gm.setup_model_call(model, input)
            gm.setup_loss_call(criterion, dummy_y, target)
            output, loss = gm.forward_and_backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        else:
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
    print('Epoch: {} | Loss: {loss.avg:.4f} | Prec@1 {top1.avg:.3f} | Prec@5 {top5.avg:.3f}'.format(epoch, loss=losses, top1=top1, top5=top5))
    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, criterion, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg

if __name__ == '__main__':
    main()
