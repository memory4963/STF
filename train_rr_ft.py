# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import math
import random
import shutil
import sys
from time import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models
import compressai.utils as utils


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }
    freezed_parameters = {
        n
        for n, p in net.named_parameters()
        if not p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters | freezed_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, args
):
    clip_max_norm = args.clip_max_norm
    model.train()
    device = next(model.parameters()).device
    start_time = time()
    pre_step = -1

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        if args.distributed:
            aux_loss = model.module.aux_loss()
        else:
            aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 100 == 0 and utils.is_main_process():
            now_time = time()
            left_time = int((now_time-start_time)/(i-pre_step)*(len(train_dataloader)-i))
            start_time = now_time
            pre_step = i
            print(
                f'Train epoch {epoch}: ['
                f'{i*len(d)}/{len(train_dataloader.dataset)}'
                f' ({100. * i / len(train_dataloader):.0f}%)]'
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item() * 255 ** 2 / 3:.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f'\tAux loss: {aux_loss.item():.2f}'
                f'\tETA: {left_time//3600}:{(left_time%3600)//60}:{left_time%60}'
            )


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg * 255 ** 2 / 3:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )
    return loss.avg


def save_checkpoint(state, is_best, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-8]+"_best"+filename[-8:])


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="stf",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--num_slices", type=int, default=10, help="Num Slices (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--save_path", type=str, default="ckpt/model.pth.tar", help="Where to Save model"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--pretrained", type=str, help="Path to a pretrained ckpt")
    parser.add_argument("--freeze_main", action="store_true", help="whether freeze main path")
    parser.add_argument("--freeze_lrp", action="store_true", help="whether freeze lrp")
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument( "--local_rank", default=0, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')

    # debug
    parser.add_argument('--rd_cost', action='store_true')

    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    utils.init_distributed_mode(args)

    if args.seed is not None:
        # seed = args.seed + utils.get_rank()
        seed = args.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    if utils.is_main_process():
        print(args)

    world_size = utils.get_world_size()
    global_rank = utils.get_rank()

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=world_size, rank=global_rank)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler=train_sampler,
            pin_memory=(device == "cuda"),
        )

        # test不用分布式
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=(device == "cuda"),
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=(device == "cuda"),
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=(device == "cuda"),
        )

    if args.pretrained and os.path.exists(args.pretrained):
        if utils.is_main_process():
            print("Loading", args.pretrained)
        ckpt = torch.load(args.pretrained)
        deps = ckpt['deps']
        print(deps)
        state_dict = ckpt['state_dict']
        net = models[args.model].from_state_dict(state_dict, deps)
    elif args.checkpoint and os.path.exists(args.checkpoint):
        if utils.is_main_process():
            print("Loading", args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location=device)
        deps = ckpt['deps']
        print(deps)
        state_dict = ckpt['state_dict']
        net = models[args.model].from_state_dict(state_dict, deps)
    else:
        raise Exception("Must start from a pretrained model or a ckpt!")
    net = net.to(device)

    # freeze main path
    if args.freeze_main:
        for n, p in net.named_parameters():
            if n.startswith('g_a'):
                p.requires_grad = False
            if n.startswith('g_s'):
                p.requires_grad = False
    if args.freeze_lrp:
        for n, p in net.named_parameters():
            if n.startswith('lrp_transforms'):
                p.requires_grad = False

    net_without_ddp = net
    if args.distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
        net_without_ddp = net.module

    optimizer, aux_optimizer = configure_optimizers(net, args)
    # CC的settings # by LUO
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 25, 30, 33], gamma=1/3)
    criterion = RateDistortionLoss()

    last_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):  # load from previous checkpoint
        ckpt = torch.load(args.checkpoint, map_location=device)
        last_epoch = ckpt["epoch"] + 1
        optimizer.load_state_dict(ckpt["optimizer"])
        aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

    best_loss = float("inf")
    if args.rd_cost:
        criterion.lmbda = args.lmbda
        loss = test_epoch(0, test_dataloader, net_without_ddp, criterion)
        print(loss)
        exit()
    for epoch in range(last_epoch, args.epochs):
        # CC中是这么做的 # by LUO
        if epoch < args.epochs // 2:
            criterion.lmbda = 2*args.lmbda
        else:
            criterion.lmbda = args.lmbda

        if utils.is_main_process():
            print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args
        )
        lr_scheduler.step()

        if utils.is_main_process():
            loss = test_epoch(epoch, test_dataloader, net_without_ddp, criterion)

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if args.save:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "deps": deps,
                    },
                    is_best,
                    args.save_path,
                )
        if args.distributed:
            dist.barrier()


if __name__ == "__main__":
    main(sys.argv[1:])
