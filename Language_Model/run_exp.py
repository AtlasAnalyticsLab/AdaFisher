import math
import os
import time
import torch
import urllib.request
import numpy as np
from asdl.precondition import PreconditioningConfig, ShampooGradientMaker, KfacGradientMaker
from torch.utils.data import DataLoader
from tqdm import tqdm
from GPT1 import MiniGPT1
import torch.nn as nn
from utils.wikitext2 import Wikitext2
from utils.torch_utils import seed_experiment, to_device
from utils.data_utils import save_logs, track_memory_gpu
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from optimizers.AdamW import AdamW
from optimizers.AdaFisher import AdaFisherW
from optimizers.AdaHessian import Adahessian
from optimizers.sgd import SGD

def train(epoch, model, dataloader, optimizer, args, gm = None):
    model.train()
    losses = []
    total_iters = 0
    start_time = time.time()
    for idx, batch in enumerate(
        tqdm(
            dataloader, desc="Epoch {0}".format(epoch), disable=(not args.progress_bar)
        )
    ):
        batch = to_device(batch, args.device)
        if args.optimizer in ["Shampoo", "kfac"]:
            dummy_y = gm.setup_model_call(model, batch["source"])
            gm.setup_loss_call(model.loss, dummy_y, batch["target"], batch["mask"])
            outputs, loss = gm.forward_and_backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       args.clip_norm)
        else:
            optimizer.zero_grad()
            log_probas = model(batch["source"])
            loss = model.loss(log_probas, batch["target"], batch["mask"])
            losses.append(loss.item() * batch["mask"].sum().item())
            if args.optimizer == 'AdaHessian':
                loss.backward(create_graph=True)
            else:
                loss.backward()
        optimizer.step()
        total_iters += 1
        if idx % args.print_every == 0:
            tqdm.write(f"[TRAIN] Epoch: {epoch}, Iter : {idx} / {len(dataloader)}, Loss: {loss.item():.5f}")

    mean_loss = np.mean(losses)
    mean_loss /= args.batch_size * dataloader.dataset.max_length
    perplexity = math.exp(mean_loss)
    tqdm.write(f"== [TRAIN] Epoch: {epoch}, Perplexity: {perplexity:.3f} ==>")
    return mean_loss, perplexity, time.time() - start_time


def evaluate(epoch, model, dataloader, args, mode="val"):
    model.eval()
    losses = []
    total_loss = 0.0
    total_iters = 0
    start_time = time.time()
    with torch.no_grad():
        for idx, batch in enumerate(
            tqdm(dataloader, desc="Evaluation", disable=(not args.progress_bar))
        ):
            batch = to_device(batch, args.device)
            log_probas = model(batch["source"])

            loss = model.loss(log_probas, batch["target"], batch["mask"])
            losses.append(loss.item() * batch["mask"].sum().item())

            total_loss += loss.item()
            total_iters += batch["source"].shape[1]

            if idx % args.print_every == 0:
                tqdm.write(
                    f"[{mode.upper()}] Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.5f}"
                )

        mean_loss = np.mean(losses)
        mean_loss /= args.batch_size * dataloader.dataset.max_length
        perplexity = math.exp(mean_loss)
        tqdm.write(
            f"=== [{mode.upper()}] Epoch: {epoch}, Iter: {idx}, Perplexity: {perplexity:.3f} ===>"
        )

    return mean_loss, perplexity, time.time() - start_time


def main(args):
    # Seed the experiment, for repeatability
    seed_experiment(args.seed)
    # Dataloaders
    train_dataset = Wikitext2(args.data_folder, split="train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_dataset = Wikitext2(args.data_folder, split="validation")
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_dataset = Wikitext2(args.data_folder, split="test")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Download the embeddings
    if not os.path.isfile(args.embeddings):
        print("Downloading embeddings...")
        urllib.request.urlretrieve(EMBEDDINGS_URL, args.embeddings)

    # Model

    model = MiniGPT1.load_embeddings_from(
        args.embeddings, num_heads=12, num_layers=args.layers
    )
    model.to(args.device)

    # Optimizer
    if args.optimizer == "adamw":
        optimizer = AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "AdaFisherW":
        optimizer = AdaFisherW(model, lr=args.lr, gammas=[args.gamma1, args.gamma2], TCov=args.curvature_update_interval,
                               weight_decay=args.weight_decay, Lambda=args.damping
                               )
    elif args.optimizer == "AdaHessian":
        optimizer = Adahessian(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer in ['Shampoo', 'kfac']:
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum
        )

    print(
        f"Initialized GPT1 model with {sum(p.numel() for p in model.parameters())} "
        f"total parameters, of which {sum(p.numel() for p in model.parameters() if p.requires_grad)} are learnable."
    )
    if args.optimizer in ['Shampoo', 'kfac']:
        config = PreconditioningConfig(data_size=args.batch_size,
                                       damping=args.damping,
                                       preconditioner_upd_interval=args.curvature_update_interval,
                                       curvature_upd_interval=args.curvature_update_interval,
                                       ema_decay=args.ema_decay,
                                       ignore_modules=[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                                       nn.LayerNorm])
        if args.optimizer == "Shampoo":
            gm = ShampooGradientMaker(model, config)
        elif args.optimizer == "kfac":
            gm = KfacGradientMaker(model, config)
    else:
        gm = None

    tot_params = sum(p.numel() for p in model.parameters())
    tot_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_losses, valid_losses = [], []
    train_ppls, valid_ppls = [], []
    train_times, valid_times = [], []
    for epoch in range(args.epochs):

        tqdm.write(f"====== Epoch {epoch} ======>")

        loss, ppl, wall_time = train(epoch, model, train_dataloader, optimizer, args, gm)
        train_losses.append(loss)
        train_ppls.append(ppl)
        train_times.append(wall_time)
        track_memory_gpu(args, epoch)
        
        loss, ppl, wall_time = evaluate(epoch, model, valid_dataloader, args)
        valid_losses.append(loss)
        valid_ppls.append(ppl)
        valid_times.append(wall_time)

    test_loss, test_ppl, test_time = evaluate(
        epoch, model, test_dataloader, args, mode="test"
    )

    print(f"===== Best validation perplexity: {min(valid_ppls):.3f} at epoch: {valid_ppls.index(min(valid_ppls))} =====>")
    logs = (
        train_losses,
        train_ppls,
        train_times,
        valid_losses,
        valid_ppls,
        valid_times,
        test_loss,
        test_ppl,
        test_time,
        tot_params,
        tot_learnable_params,
    )
    if args.log == 1:
        save_logs(args, *logs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run an experiment for assignment 2.")

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--data_folder",
        type=str,
        default="./data",
        help="path to the data folder (default: %(default)s).",
    )
    data.add_argument(
        "--batch_size", type=int, default=2, help="batch size (default: %(default)s)."
    )

    model = parser.add_argument_group("Model")
    model.add_argument(
        "--embeddings",
        type=str,
        default="./data/embeddings.npz",
        help="path to the embeddings file (default: %(default)s).",
    )
    model.add_argument(
        "--layers",
        type=int,
        default=1,
        help="number of layers in the model (default: %(default)s).",
    )

    optimization = parser.add_argument_group("Optimization")
    optimization.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="number of epochs for training (default: %(default)s).",
    )
    optimization.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["AdaFisherW", "AdaFisher", "adam", "adamw", 'kfac', 'Shampoo', "AdaHessian"],
        help="choice of optimizer (default: %(default)s).",
    )
    optimization.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate for Adam optimizer (default: %(default)s).",
    )
    optimization.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="momentum for SGD optimizer (default: %(default)s).",
    )
    optimization.add_argument(
        "--clip_norm",
        type=int,
        default=10,
        help="Clip norm for SGD optimizer (default: %(default)s).",
    )
    optimization.add_argument(
        "--ema_decay",
        type=int,
        default=-1,
        help="ema_decay for SGD optimizer (default: %(default)s).",
    )
    optimization.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="weight decay (default: %(default)s).",
    )
    optimization.add_argument(
        "--damping",
        type=float,
        default=1e-3,
        help="Damping parameter (default: %(default)s).",
    )
    optimization.add_argument(
        "--curvature_update_interval",
        type=float,
        default=10,
        help="curvature_update_interval parameter (default: %(default)s).",
    )
    optimization.add_argument(
        "--gamma1",
        type=float,
        default=0.92,
        help="gamma1 parameter (default: %(default)s).",
    )
    optimization.add_argument(
        "--gamma2",
        type=float,
        default=0.008,
        help="gamma2 parameter (default: %(default)s).",
    )
    exp = parser.add_argument_group("Experiment config")
    exp.add_argument(
        "--exp_id",
        type=str,
        default="debug",
        help="unique experiment identifier (default: %(default)s).",
    )
    exp.add_argument(
        "--log",
        type = int,
        default = 1,
        help="whether or not to log data from the experiment.",
    )
    exp.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="directory to log results to (default: %(default)s).",
    )
    exp.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for repeatability (default: %(default)s).",
    )

    misc = parser.add_argument_group("Miscellaneous")
    misc.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="number of processes to use for data loading (default: %(default)s).",
    )
    misc.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cuda",
        help="device to store tensors on (default: %(default)s).",
    )
    misc.add_argument(
        "--progress_bar", action="store_true", help="show tqdm progress bar."
    )
    misc.add_argument(
        "--print_every",
        type=int,
        default=10,
        help="number of minibatches after which to print loss (default: %(default)s).",
    )
    args = parser.parse_args()
    main(args)
