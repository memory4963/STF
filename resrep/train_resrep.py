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
from compressai.models import rr_utils
from compressai.zoo import models
import compressai.utils as utils
from resrep.rr_builder import CompactorLayer, RRBuilder


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


def configure_optimizers(model, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in model.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in model.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }
    freezed_parameters = {
        n
        for n, p in model.named_parameters()
        if not p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(model.named_parameters())
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


def train_one_step(model, d, criterion, optimizer, aux_optimizer, lasso_strength, distributed, clip_max_norm):
    optimizer.zero_grad()
    aux_optimizer.zero_grad()

    out_net = model(d)

    out_criterion = criterion(out_net, d)
    out_criterion["loss"].backward()
    aux_loss = model.module.aux_loss() if distributed else model.aux_loss()
    aux_loss.backward()

    if clip_max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

    for group in model.compactors:
        for compactor in group:
            # mask掉的通道只会继续被降低，丢掉正常的loss
            compactor.mask_weight_grad()
            compactor.after_backward()
            compactor.add_lasso_penalty(lasso_strength)

    optimizer.step()
    aux_optimizer.step()

    return out_criterion, aux_loss


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


def save_checkpoint(state, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename)


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
    parser.add_argument("--pretrained", type=str, help="Path to a pretrained model")

    parser.add_argument("--resrep_warmup_step", type=int, default=1000, help="step num before resrep pruning")
    parser.add_argument("--mask_interval", type=int, default=1500, help="step num before resrep pruning")
    parser.add_argument("--flops_target", type=float, default=0.3, help="fraction of keeping flops")
    parser.add_argument("--num_per_mask", type=int, default=20, help="channels to prune each time")
    parser.add_argument("--lasso_strength", type=float, default=1e-9, help="penalty of lasso")
    parser.add_argument("--least_remain_channel", type=int, default=5, help="least remaining channel of each layer")
    parser.add_argument("--threshold", type=float, default=1e-5, help="penalty of lasso")
    parser.add_argument("--slow_start", type=int, default=100, help="slow start for pruning to avoid performance crash")
    parser.add_argument("--freeze_main", action="store_true", help="whether freeze main")
    parser.add_argument("--y_excluded", action="store_true", help="whether exclude y when calculate compactor score")
    parser.add_argument("--norm", action="store_true", help="calculate the normed score of channels, conflict with y_excluded")
    parser.add_argument("--score_mode", default="resrep", choices=[
        "resrep",
        "fisher_mask",
        "fisher_gate",
        "gate_decorator"
    ], help="choose how to calculate the importance of channels.")
    parser.add_argument("--grad_ratio", type=float, default=1., help="power of gradient in gate decorator mode.")

    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_true')

    # debug
    parser.add_argument('--rd_cost', action='store_true')

    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    utils.init_distributed_mode(args)

    if args.seed is not None:
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
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

    if args.norm and args.y_excluded:
        raise "norm and y_excluded conflict with each other."

    model = models[args.model](RRBuilder(args.score_mode, args.grad_ratio), num_slices=args.num_slices, y_excluded=args.y_excluded, score_norm=args.norm)
    model = model.to(device)
    ori_deps = model.cal_deps(min_channel=args.least_remain_channel)
    print(ori_deps)

    main_prune = 'main' in args.model

    # freeze main path
    if args.freeze_main:
        for n, p in model.named_parameters():
            if n.startswith('g_a'):
                p.requires_grad = False
            if n.startswith('g_s'):
                p.requires_grad = False
            # if n.startswith('g_a') and not n.startswith('g_a.6'):
            #     p.requires_grad = False
            # if n.startswith('g_s') and not n.startswith('g_s.0'):
            #     p.requires_grad = False

    net_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        net_without_ddp = model.module

    optimizer, aux_optimizer = configure_optimizers(model, args)
    # Settings of ResRep
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*len(train_dataloader))
    criterion = RateDistortionLoss(args.lmbda)

    last_epoch = 0
    if args.pretrained and os.path.exists(args.pretrained):
        if utils.is_main_process():
            print("Loading", args.pretrained)
        state_dict = torch.load(args.pretrained, map_location=device)['state_dict']
        model.load_pretrained(state_dict)

    if args.checkpoint and os.path.exists(args.checkpoint):  # load from previous checkpoint
        if utils.is_main_process():
            print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    step = last_epoch*len(train_dataloader)

    num_per_mask = args.num_per_mask
    args.num_per_mask = num_per_mask // 2
    mask_interval = args.mask_interval 
    args.mask_interval = mask_interval * 2
    if args.rd_cost:
        loss = test_epoch(0, test_dataloader, net_without_ddp, criterion)
        print(loss)
        exit()
    for epoch in range(last_epoch, args.epochs):
        if utils.is_main_process():
            print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        model.train()
        device = next(model.parameters()).device
        start_time = time()
        pre_step = -1

        # debug
        # pruned_save_name = args.save_path[:-8] + '_pruned_' + str(epoch) + args.save_path[-8:]
        # save_checkpoint(rr_utils.cc_model_prune(model, ori_deps, args.threshold, enhanced_resrep='enhance' in args.model, without_y='without_y' in args.model, min_channel=args.least_remain_channel, main_prune=main_prune), pruned_save_name)
        # exit()

        for i, d in enumerate(train_dataloader):
            resrep_step = step - args.resrep_warmup_step
            if resrep_step == 0:
                model.reset_grad_records() # start to record grads after warmup
            if resrep_step > 0 and resrep_step % args.mask_interval == 0:

                # 慢启动, 一开始1/4的剪枝速度，然后1/2，然后正常速度
                if resrep_step // (mask_interval*2) > args.slow_start:
                    args.num_per_mask = num_per_mask
                    args.mask_interval = mask_interval
                elif resrep_step // (mask_interval*2) > args.slow_start // 2:
                    args.num_per_mask = num_per_mask
                    args.mask_interval = mask_interval * 2
                else:
                    args.num_per_mask = num_per_mask // 2
                    args.mask_interval = mask_interval * 2

                print(f'update mask at step {step}')
                model.resrep_masking(ori_deps, args)
                masked_deps = model.cal_mask_deps()
                print(masked_deps)
                print(rr_utils.cal_cc_flops(masked_deps, main_prune)/rr_utils.cal_cc_flops(ori_deps, main_prune))
            d = d.to(device)
            out_criterion, aux_loss = train_one_step(model, d, criterion, optimizer, aux_optimizer, args.lasso_strength, args.distributed, args.clip_max_norm)

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

            lr_scheduler.step()
            step += 1

        if utils.is_main_process() and ((epoch+1) % 1 == 0 or epoch == args.epochs-1):
            loss = test_epoch(epoch, test_dataloader, net_without_ddp, criterion)

            if args.save:
                pruned_model = rr_utils.cc_model_prune(model, ori_deps, args.threshold, enhanced_resrep='enhance' in args.model, without_y='without_y' in args.model, min_channel=args.least_remain_channel, main_prune=main_prune)
                if args.save_path.endswith('.pth.tar'):
                    save_name = args.save_path[:-8] + '_' + str(epoch) + args.save_path[-8:]
                    pruned_save_name = args.save_path[:-8] + '_pruned_' + str(pruned_model['keep_portion'])[:5] + '_' + str(epoch) + args.save_path[-8:]
                else:
                    save_name = args.save_path.rsplit('.', 1)[0] + '_' + str(epoch) + args.save_path.rsplit('.', 1)[1]
                    pruned_save_name = args.save_path.rsplit('.', 1)[0] + '_pruned_' + str(pruned_model['keep_portion'])[:5] + '_' + str(epoch) + args.save_path.rsplit('.', 1)[1]
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "deps": model.cal_deps(min_channel=args.least_remain_channel),
                        "args": str(args)
                    },
                    save_name
                )
                save_checkpoint(
                    pruned_model,
                    pruned_save_name)
        if args.distributed:
            dist.barrier()


if __name__ == "__main__":
    main(sys.argv[1:])
