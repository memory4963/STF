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

import argparse
import math
import random
import shutil
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, lpips=False, yuv=False):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.lpips = lpips
        self.yuv = yuv
        self.uv_ratio = 0.5

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

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

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
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)

        panelty = 0.
        for gd in model.gds:
            panelty += gd.gate.abs().sum()
        out_criterion["loss"] += model.sparse_lambda * panelty

        out_criterion["loss"].backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )


def tick(
    model, criterion, train_dataloader, epoch, clip_max_norm, args, tick_round=10, subset_scale=0.1, num=5
):
    model.train()

    # set new optimizer
    for n, p in model.named_parameters():
        if 'g_s.9' in n or 'gate' in n:
            continue
        p.requires_grad = False

    parameters = {
        n
        for n, p in model.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    params_dict = dict(model.named_parameters())
    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    for gd in model.gds:
        gd.reset_score()

    device = next(model.parameters()).device

    for _ in range(tick_round):
        for gd in model.gds:
            gd.reset_score()
        for i, d in enumerate(train_dataloader):
            d = d.to(device)

            optimizer.zero_grad()

            out_net = model(d)

            out_criterion = criterion(out_net, d)

            out_criterion["loss"].backward()

            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

            # calculate score
            for gd in model.gds:
                gd.cal_score()

            optimizer.step()

            if i % 10 == 0:
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                )
            if i >= subset_scale*len(train_dataloader):
                break
        scores = torch.empty([0]).to(device)
        for gd in model.gds:
            idx = torch.where(gd.mask > 0.)[1]
            scores = torch.cat((scores, gd.get_score()[idx]))
        threshold = torch.sort(scores)[0][num]

        for gd in model.gds:
            score = gd.get_score()
            hard_threashold = torch.sort(score)[0][-gd.minimal]
            hard_mask = score >= hard_threashold
            soft_mask = score > threshold
            mask = hard_mask + soft_mask
            print(sum(mask))
            gd.mask.set_(mask.to(torch.float32).to(device).view(1, -1, 1, 1) * gd.mask)
    for n, p in model.named_parameters():
        p.requires_grad = True


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
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f} |"
    )

    return loss.avg


def save_checkpoint(state, is_best, epoch, save_path="output"):
    save_file = os.path.join(save_path, "checkpoint.pth.tar")
    torch.save(state, save_file)
    if is_best:
        shutil.copyfile(save_file, os.path.join(save_path, "checkpoint_best_loss.pth.tar"))
    if epoch % 10 == 0:
        shutil.copyfile(save_file, os.path.join(save_path, "checkpoint_{}.pth.tar".format(epoch)))



def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-q", "--quality", type=int, required=False, default=4, help="Quality of network"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--tick_freq",
        default=10,
        type=int,
        help="Number of tock epochs between tick (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-sl",
        "--sparse_lambda",
        default=0.001,
        type=float,
        help="sparse lambda for hit pruning (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=30,
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
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
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
        "--save_path", type=str, default="output/"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--pretrained", action="store_true", default=False, help="Whether use pretrained")
    parser.add_argument(
        "--tick_round",
        type=int,
        default=10,
        help="tick round (default: %(default)s)",
    )
    parser.add_argument(
        "--flops_target",
        type=float,
        default=0.3,
        help="fraction of flops to keep (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    return args


def get_con_flops(input_deps, output_deps, h, w=None, kernel_size=3, groups=1):
    if w is None:
        w = h
    rtn = input_deps * output_deps * h * w * kernel_size * kernel_size // groups
    return rtn.data.cpu().numpy()


def calculate_cai_bmshj_main_flops(deps):
    result = []
    result.append(get_con_flops(3, deps[0], 128, 128, 5))
    result.append(get_con_flops(deps[0], deps[1], 64, 64, 5))
    result.append(get_con_flops(deps[1], deps[2], 32, 32, 5))
    result.append(get_con_flops(deps[2], deps[3], 16, 16, 5))
    result.append(get_con_flops(deps[3], deps[4], 32, 32, 5))
    result.append(get_con_flops(deps[4], deps[5], 64, 64, 5))
    result.append(get_con_flops(deps[5], deps[6], 128, 128, 5))
    result.append(get_con_flops(deps[6], 3, 256, 25, 5))
    return np.sum(result)


def cal_deps(net):
    return [sum((net.g_a[1].mask > 0.).view(-1)), sum((net.g_a[4].mask > 0.).view(-1)), sum((net.g_a[7].mask > 0.).view(-1)), net.g_a[9].weight.shape[0],
            sum((net.g_s[1].mask > 0.).view(-1)), sum((net.g_s[4].mask > 0.).view(-1)), sum((net.g_s[7].mask > 0.).view(-1))]


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

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

    net = models[args.model](sparse_lambda=args.sparse_lambda)
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, (800,))
    criterion = RateDistortionLoss(lmbda=args.lmbda, lpips=args.lpips, yuv=args.yuv_loss)

    ori_flops = calculate_cai_bmshj_main_flops(cal_deps(net))

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"], True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        if (epoch + 1) % args.tick_freq == 0:
            print('tick', epoch)
            tick(
                net,
                criterion,
                train_dataloader,
                epoch,
                args.clip_max_norm,
                args,
                tick_round=args.tick_round
            )
            if calculate_cai_bmshj_main_flops(cal_deps(net)) < args.flops_target * ori_flops:
                break

        loss = test_epoch(epoch, test_dataloader, net, criterion)
        # lr_scheduler.step(loss)
        lr_scheduler.step(epoch)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                epoch,
                save_path=args.save_path
            )
    prune_model(net, args.save_path)


def prune_model(model, save_path):
    save_dict = model.state_dict()
    deps = []
    for mask_name, weight_names in model.mask_weight_pairs.items():
        if 'g_a' in mask_name: deconv = False
        else: deconv = True

        weight_name, bias_name, beta_name, gamma_name, suc_weight_name, gate_name = weight_names
        weight, bias, beta, gamma, suc_weight, gate = map(lambda x: save_dict[x], weight_names)

        mask_idx = torch.where(save_dict[mask_name] > 0.)[1]
        deps.append(mask_idx.shape[0])
        if deconv:
            weight = torch.einsum('ijkv,ajbc->ijkv', weight, gate)
            weight = weight[:, mask_idx]
            suc_weight = suc_weight[mask_idx]
        else:
            weight = torch.einsum('jikv,ajbc->jikv', weight, gate)
            weight = weight[mask_idx]
            suc_weight = suc_weight[:, mask_idx]
        bias = bias*gate.view(-1)
        bias = bias[mask_idx]
        beta = beta[mask_idx]
        gamma = gamma[mask_idx]
        gamma = gamma[:, mask_idx]

        save_dict[weight_name] = weight
        save_dict[bias_name] = bias
        save_dict[beta_name] = beta
        save_dict[gamma_name] = gamma
        save_dict[suc_weight_name] = suc_weight
    for k, v in model.KEY_TABLE.items():
        save_dict[k] = save_dict.pop(v)
    for k in model.mask_weight_pairs:
        save_dict.pop(k)
    for k in model.to_be_pop:
        save_dict.pop(k)
    deps.insert(3, save_dict['g_a.6.weight'].shape[0])
    torch.save({'state_dict': save_dict, 'deps': deps}, os.path.join(save_path, 'pruned_model.pth'))
    print('save_path: ' + os.path.join(save_path, 'pruned_model.pth'))


if __name__ == "__main__":
    main(sys.argv[1:])
