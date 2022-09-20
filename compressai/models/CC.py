import math
from typing import Iterable
import torch
import torch.nn as nn

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from compressai.models import CC_tables
from .utils import conv, deconv, update_registered_buffers
from compressai.ops import ste_round
from compressai.layers import conv3x3, subpel_conv3x3, Win_noShift_Attention
from .base import CompressionModel
from .rr_utils import *

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class CC(CompressionModel):
    """Channel-wise Context model"""

    def __init__(self, N=192, M=320, num_slices=10, max_support_slices=-1, **kwargs):
        super().__init__(**kwargs)
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv3x3(320, 320),
            nn.ReLU(),
            conv(320, 256, stride=2),
            nn.ReLU(),
            conv(256, 192, stride=2),
        )

        self.h_mean_s = nn.Sequential(
            deconv(192, 192, stride=2),
            nn.ReLU(),
            deconv(192, 256, stride=2),
            nn.ReLU(),
            conv3x3(256, 320),
        )

        self.h_scale_s = nn.Sequential(
            deconv(192, 192, stride=2),
            nn.ReLU(),
            deconv(192, 256, stride=2),
            nn.ReLU(),
            conv3x3(256, 320),
        )
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv3x3(320 + 320//self.num_slices*i, 224 + 320//self.num_slices*i*2//3),
                nn.ReLU(),
                conv3x3(224 + 320//self.num_slices*i*2//3, 128 + 320//self.num_slices*i*1//3),
                nn.ReLU(),
                conv3x3(128 + 320//self.num_slices*i*1//3, 320//self.num_slices),
            ) for i in range(self.num_slices)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv3x3(320 + 320//self.num_slices*i, 224 + 320//self.num_slices*i*2//3),
                nn.ReLU(),
                conv3x3(224 + 320//self.num_slices*i*2//3, 128 + 320//self.num_slices*i*1//3),
                nn.ReLU(),
                conv3x3(128 + 320//self.num_slices*i*1//3, 320//self.num_slices),
            ) for i in range(self.num_slices)
            )
        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv3x3(320 + 320//self.num_slices*(i+1), 224 + 320//self.num_slices*i*2//3),
                nn.ReLU(),
                conv3x3(224 + 320//self.num_slices*i*2//3, 128 + 320//self.num_slices*i*1//3),
                nn.ReLU(),
                conv3x3(128 + 320//self.num_slices*i*1//3, 320//self.num_slices),
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)


    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated


    def forward(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = self.split_slices(y)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
    
    def split_slices(self, y):
        return y.chunk(self.num_slices, 1)

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        # N = state_dict["g_a.0.weight"].size(0)
        # M = state_dict["g_a.6.weight"].size(0)
        # net = cls(N, M)
        net = cls(192, 320)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())


            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}


class CC_uneven(CC):
    def __init__(self, N=192, M=320, num_slices=5, max_support_slices=-1, slices=[10,20,40,80,170], **kwargs):
        super().__init__(N, M, num_slices, max_support_slices, **kwargs)
        self.slices = slices
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv3x3(320 + sum(self.slices[:i]), 224 + sum(self.slices[:i])*2//3),
                nn.ReLU(),
                conv3x3(224 + sum(self.slices[:i])*2//3, 128 + sum(self.slices[:i])*1//3),
                nn.ReLU(),
                conv3x3(128 + sum(self.slices[:i])*1//3, self.slices[i]),
            ) for i in range(self.num_slices)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv3x3(320 + sum(self.slices[:i]), 224 + sum(self.slices[:i])*2//3),
                nn.ReLU(),
                conv3x3(224 + sum(self.slices[:i])*2//3, 128 + sum(self.slices[:i])*1//3),
                nn.ReLU(),
                conv3x3(128 + sum(self.slices[:i])*1//3, self.slices[i]),
            ) for i in range(self.num_slices)
            )
        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv3x3(320 + sum(self.slices[:i]) + self.slices[i], 224 + sum(self.slices[:i])*2//3),
                nn.ReLU(),
                conv3x3(224 + sum(self.slices[:i])*2//3, 128 + sum(self.slices[:i])*1//3),
                nn.ReLU(),
                conv3x3(128 + sum(self.slices[:i])*1//3, self.slices[i]),
            ) for i in range(self.num_slices)
        )

    def split_slices(self, y):
        return y.split(self.slices, dim=1)


class CC_RandomSplit(CC):
    """Channel-wise Context model"""

    def __init__(self, N=192, M=320, **kwargs):
        super().__init__(N, M, **kwargs)
        # TODO 第一个slice也过transform
        self.cc_mean_transforms = nn.ModuleList(
            [nn.Sequential(
                conv3x3(320, 224),
                nn.ReLU(),
                conv3x3(224, 128),
                nn.ReLU(),
                conv3x3(128, 320),
            ), nn.Sequential(
                conv3x3(320 + 320, 224 + 320*2//3),
                nn.ReLU(),
                conv3x3(224 + 320*2//3, 128 + 320*1//3),
                nn.ReLU(),
                conv3x3(128 + 320*1//3, 320),
            )]
        )
        self.cc_scale_transforms = nn.ModuleList(
            [nn.Sequential(
                conv3x3(320, 224),
                nn.ReLU(),
                conv3x3(224, 128),
                nn.ReLU(),
                conv3x3(128, 320),
            ), nn.Sequential(
                conv3x3(320 + 320, 224 + 320*2//3),
                nn.ReLU(),
                conv3x3(224 + 320*2//3, 128 + 320*1//3),
                nn.ReLU(),
                conv3x3(128 + 320*1//3, 320),
            )]
        )
        self.lrp_transforms = nn.ModuleList(
            [nn.Sequential(
                conv3x3(320 + 320, 224),
                nn.ReLU(),
                conv3x3(224, 128),
                nn.ReLU(),
                conv3x3(128, 320),
            ), nn.Sequential(
                conv3x3(320 + 320 + 320, 224 + 320*2//3),
                nn.ReLU(),
                conv3x3(224 + 320*2//3, 128 + 320*1//3),
                nn.ReLU(),
                conv3x3(128 + 320*1//3, 320),
        )]
        )


    def forward(self, x, split_pos):
        return self.train_forward if split_pos is int else self.eval_forward(x, split_pos)
    
    def train_forward(self, x, split_pos):
        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        anchor_means = self.cc_mean_transforms[0](latent_means)
        anchor_scales = self.cc_scale_transforms[0](latent_scales)

        _, y_hat_anchor_likelihood = self.gaussian_conditional(y[:, :split_pos], anchor_scales[:, :split_pos], anchor_means[:, :split_pos]) 
        y_hat_anchor = ste_round(y[:, :split_pos] - anchor_means[:, :split_pos]) + anchor_means[:, :split_pos]
        anchor_slice = torch.zeros_like(latent_means)
        anchor_slice[:, :split_pos] = y_hat_anchor
        anchor_lrp = self.lrp_transforms[0](torch.cat((latent_means, anchor_slice), dim=1))
        y_hat_anchor += 0.5 * torch.tanh(anchor_lrp[:, :split_pos])

        support_slice = torch.zeros_like(latent_means)
        support_slice[:, :split_pos] = y_hat_anchor
        nonanchor_means = self.cc_mean_transforms[1](torch.cat((latent_means, support_slice), dim=1))
        nonanchor_scales = self.cc_scale_transforms[1](torch.cat((latent_means, support_slice), dim=1))

        _, y_hat_nonanchor_likelihood = self.gaussian_conditional(y[:, split_pos:], nonanchor_scales[:, split_pos:], nonanchor_means[:, split_pos:])
        y_hat_nonanchor = ste_round(y[:, split_pos:] - nonanchor_means[:, split_pos:]) + nonanchor_means[:, split_pos:]

        y_hat_nonanchor_fullchannel = torch.zeros_like(latent_means)
        y_hat_nonanchor_fullchannel[:, split_pos:] = y_hat_nonanchor
        lrp = self.lrp_transforms[1](torch.cat((latent_means, support_slice, y_hat_nonanchor_fullchannel), dim=1))
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_nonanchor_fullchannel += lrp
        y_hat_nonanchor = y_hat_nonanchor_fullchannel[:, split_pos:]

        y_hat = torch.cat((y_hat_anchor, y_hat_nonanchor), dim=1)
        y_likelihoods = torch.cat((y_hat_anchor_likelihood, y_hat_nonanchor_likelihood), dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def eval_forward(self, x, split_pos):
        self.slices = split_pos + [320]

        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = self.split_slices(y)
        y_hat_slices = []
        y_likelihood = []

        anchor_means = self.cc_mean_transforms[0](latent_means)
        anchor_scales = self.cc_scale_transforms[0](latent_scales)

        _, y_hat_anchor_likelihood = self.gaussian_conditional(y[:, :self.slices[0]], anchor_scales[:, :self.slices[0]], anchor_means[:, :self.slices[0]]) 
        y_likelihood.append(y_hat_anchor_likelihood)
        y_hat_anchor = ste_round(y[:, :self.slices[0]] - anchor_means[:, :self.slices[0]]) + anchor_means[:, :self.slices[0]]
        anchor_slice = torch.zeros_like(latent_means)
        anchor_slice[:, :self.slices[0]] = y_hat_anchor
        anchor_lrp = self.lrp_transforms[0](torch.cat((latent_means, anchor_slice), dim=1))
        y_hat_anchor += 0.5 * torch.tanh(anchor_lrp[:, :self.slices[0]])
        y_hat_slices.append(y_hat_anchor)

        for slice_index, y_slice in enumerate(y_slices[1:]):
            support_slices = y_hat_slices
            support_slices = torch.zeros_like(latent_means)
            support_slices[:, :self.slices[slice_index]] = torch.cat(y_hat_slices, dim=1)
            mean_support = torch.cat([latent_means, support_slices], dim=1)
            mu = self.cc_mean_transforms[1](mean_support)
            mu = mu[:, self.slices[slice_index]:self.slices[slice_index+1]]

            scale_support = torch.cat([latent_scales, support_slices], dim=1)
            scale = self.cc_scale_transforms[1](scale_support)
            scale = scale[:, self.slices[slice_index]:self.slices[slice_index+1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            y_hat_slice_full = torch.zeros_like(latent_means)
            y_hat_slice_full[:, self.slices[slice_index]:self.slices[slice_index+1]] = y_hat_slice
            lrp_support = torch.cat([mean_support, y_hat_slice_full], dim=1)
            lrp = self.lrp_transforms[1](lrp_support)[:, self.slices[slice_index]:self.slices[slice_index+1]]
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def split_slices(self, y):
        splits = [self.slices[0]]
        splits += [self.slices[i+1]-self.slices[i] for i in range(len(self.slices)-1)]
        return y.split(splits, dim=1)


class CCResRep(CC):
    def __init__(self, builder: RRBuilder, N=192, M=320, num_slices=10, max_support_slices=-1, **kwargs):
        super().__init__(N, M, num_slices, max_support_slices, **kwargs)
        self.slice_size = M//num_slices
        self.group_id = 0
        self.N = N
        self.M = M

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        # y的通道剪枝，为了和后面的lrp输出对应，因此分割成num_slices个compactor
        self.y_compactors = nn.ModuleList(builder.GroupCompactor(self.slice_size, self.group_id + i) for i in range(self.num_slices))
        # 保证lrp的group id和y_compactors的一样
        lrp_group_id_start = self.group_id
        self.group_id += self.num_slices

        self.h_a = nn.Sequential(
            builder.Conv2dRR(M, M, stride=1, padding=1),
            nn.ReLU(),
            builder.Conv2dRR(M, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            builder.Conv2dRR(256, N, kernel_size=5, stride=2, padding=2),
        )

        self.h_mean_s = nn.Sequential(
            builder.ConvTranspose2dRR(N, N, kernel_size=5, stride=2, output_padding=1, padding=2),
            nn.ReLU(),
            builder.ConvTranspose2dRR(N, 256, kernel_size=5, stride=2, output_padding=1, padding=2),
            nn.ReLU(),
            builder.Conv2dRR(256, M, stride=1, padding=1),
        )

        self.h_scale_s = nn.Sequential(
            builder.ConvTranspose2dRR(N, N, kernel_size=5, stride=2, output_padding=1, padding=2),
            nn.ReLU(),
            builder.ConvTranspose2dRR(N, 256, kernel_size=5, stride=2, output_padding=1, padding=2),
            nn.ReLU(),
            # group id和h_mean_s一样
            builder.Conv2dRR(256, M, stride=1, padding=1),
        )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                builder.Conv2dRR(M + self.slice_size*i, 224 + self.slice_size*i*2//3, stride=1, padding=1),
                nn.ReLU(),
                builder.Conv2dRR(224 + self.slice_size*i*2//3, 128 + self.slice_size*i*1//3, stride=1, padding=1),
                nn.ReLU(),
                builder.Conv2dGroupRR(128 + self.slice_size*i*1//3, self.slice_size, stride=1, padding=1, group_id=self.group_id + i),
            ) for i in range(self.num_slices)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                builder.Conv2dRR(M + self.slice_size*i, 224 + self.slice_size*i*2//3, stride=1, padding=1),
                nn.ReLU(),
                builder.Conv2dRR(224 + self.slice_size*i*2//3, 128 + self.slice_size*i*1//3, stride=1, padding=1),
                nn.ReLU(),
                # group id和cc_mean_trainsforms相同
                builder.Conv2dGroupRR(128 + self.slice_size*i*1//3, self.slice_size, stride=1, padding=1, group_id=self.group_id + i),
            ) for i in range(self.num_slices)
        )
        self.group_id += self.num_slices

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                builder.Conv2dRR(M + self.slice_size*(i+1), 224 + self.slice_size*i*2//3, stride=1, padding=1),
                nn.ReLU(),
                builder.Conv2dRR(224 + self.slice_size*i*2//3, 128 + self.slice_size*i*1//3, stride=1, padding=1),
                nn.ReLU(),
                # group id和y+compactors相同
                builder.Conv2dGroupRR(128 + self.slice_size*i*1//3, self.slice_size, stride=1, padding=1, group_id=lrp_group_id_start + i),
            ) for i in range(self.num_slices)
        )

        self.compactors = \
            [(self.y_compactors[i][0], self.lrp_transforms[i][4][1]) for i in range(self.num_slices)] + \
            [(self.cc_mean_transforms[i][4][1], self.cc_scale_transforms[i][4][1]) for i in range(self.num_slices)] + \
            [(layer[1],) for layer in filter(lambda x: isinstance(x, nn.Sequential) and isinstance(x[1], CompactorLayer), self.h_a)] + \
            [(layer[1],) for layer in filter(lambda x: isinstance(x, nn.Sequential) and isinstance(x[1], CompactorLayer), self.h_mean_s)] + \
            [(layer[1],) for layer in filter(lambda x: isinstance(x, nn.Sequential) and isinstance(x[1], CompactorLayer), self.h_scale_s)] + \
            [(self.cc_mean_transforms[i][0][1],) for i in range(self.num_slices)] + \
            [(self.cc_mean_transforms[i][2][1],) for i in range(self.num_slices)] + \
            [(self.cc_scale_transforms[i][0][1],) for i in range(self.num_slices)] + \
            [(self.cc_scale_transforms[i][2][1],) for i in range(self.num_slices)] + \
            [(self.lrp_transforms[i][0][1],) for i in range(self.num_slices)] + \
            [(self.lrp_transforms[i][2][1],) for i in range(self.num_slices)]
        
        self.suc_table = CC_tables.gene_table(self.num_slices)

    def forward(self, x):
        ori_y = self.g_a(x)
        y = torch.zeros_like(ori_y)
        for i in range(self.num_slices):
            y[:, i*self.slice_size:(i+1)*self.slice_size] = self.y_compactors[i](ori_y[:, i*self.slice_size:(i+1)*self.slice_size])
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = self.split_slices(y)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            # mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            # scale = scale[:, :, :y_shape[0], :y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
    
    def resrep_masking(self, ori_deps, args):
        ori_flops = self.cal_cc_flops(ori_deps)
        scores = cal_compactor_scores(self.compactors)
        cur_deps = self.cal_mask_deps()
        sorted_keys = sorted(scores, key=scores.get)
        cur_flops = self.cal_cc_flops(cur_deps)

        if cur_flops <= args.flops_target * ori_flops:
            return
        i = 0
        for key in sorted_keys:
            if i == args.num_per_mask:
                break
            if self.compactors[key[0]][0].mask[key[1]] == 0: # already masked, skip
                continue
            i += 1
            if self.compactors[key[0]][0].get_num_mask_ones() <= args.least_remain_channel: # no more channel in this layer should be masked
                continue
            for compactor in self.compactors[key[0]]:
                compactor.set_mask_to_zero(key[1]) # mask, or to say prune

    def cal_deps(self, thr=1e-5):
        return {
            'y': max(1, sum([get_remain(layer[0], thr) for layer in self.y_compactors])),
            'h_a': [get_remain(layer[1], thr) for layer in filter(lambda x: isinstance(x, nn.Sequential) and isinstance(x[1], CompactorLayer), self.h_a)],
            'h_mean_s': [get_remain(layer[1], thr) for layer in filter(lambda x: isinstance(x, nn.Sequential) and isinstance(x[1], CompactorLayer), self.h_mean_s)],
            'h_scale_s': [get_remain(layer[1], thr) for layer in filter(lambda x: isinstance(x, nn.Sequential) and isinstance(x[1], CompactorLayer), self.h_scale_s)],
            'cc_mean_transforms': [[get_remain(layer[1], thr) for layer in filter(lambda x: isinstance(x, nn.Sequential) and isinstance(x[1], CompactorLayer), cc_mean_transform)] for cc_mean_transform in self.cc_mean_transforms],
            'cc_scale_transforms': [[get_remain(layer[1], thr) for layer in filter(lambda x: isinstance(x, nn.Sequential) and isinstance(x[1], CompactorLayer), cc_scale_transform)] for cc_scale_transform in self.cc_scale_transforms],
            'lrp_transforms': [[get_remain(layer[1], thr) for layer in filter(lambda x: isinstance(x, nn.Sequential) and isinstance(x[1], CompactorLayer), lrp_transform)] for lrp_transform in self.lrp_transforms],
        }
    
    def cal_mask_deps(self):
        return {
            'y': max(1, sum([compactor[0].get_num_mask_ones() for compactor in self.y_compactors])),
            'h_a': [layer[1].get_num_mask_ones() for layer in filter(lambda x: isinstance(x, nn.Sequential) and isinstance(x[1], CompactorLayer), self.h_a)],
            'h_mean_s': [layer[1].get_num_mask_ones() for layer in filter(lambda x: isinstance(x, nn.Sequential) and isinstance(x[1], CompactorLayer), self.h_mean_s)],
            'h_scale_s': [layer[1].get_num_mask_ones() for layer in filter(lambda x: isinstance(x, nn.Sequential) and isinstance(x[1], CompactorLayer), self.h_scale_s)],
            'cc_mean_transforms': [[layer[1].get_num_mask_ones() for layer in filter(lambda x: isinstance(x, nn.Sequential) and isinstance(x[1], CompactorLayer), cc_mean_transform)] for cc_mean_transform in self.cc_mean_transforms],
            'cc_scale_transforms': [[layer[1].get_num_mask_ones() for layer in filter(lambda x: isinstance(x, nn.Sequential) and isinstance(x[1], CompactorLayer), cc_scale_transform)] for cc_scale_transform in self.cc_scale_transforms],
            'lrp_transforms': [[layer[1].get_num_mask_ones() for layer in filter(lambda x: isinstance(x, nn.Sequential) and isinstance(x[1], CompactorLayer), lrp_transform)] for lrp_transform in self.lrp_transforms],
        }

    def cal_cc_flops(self, deps=None):
        if deps is None:
            deps = self.cal_deps()

        flops = cal_conv_flops(deps['y'], deps['h_a'][0], 16, 16, 3)
        flops += cal_conv_flops(deps['h_a'][0], deps['h_a'][1], 8, 8, 5)
        flops += cal_conv_flops(deps['h_a'][1], deps['h_a'][2], 4, 4, 5)
        flops += cal_conv_flops(deps['h_a'][2], deps['h_mean_s'][0], 4, 4, 5)
        flops += cal_conv_flops(deps['h_mean_s'][0], deps['h_mean_s'][1], 4, 4, 5)
        flops += cal_conv_flops(deps['h_mean_s'][1], deps['h_mean_s'][2], 8, 8, 5)
        flops += cal_conv_flops(deps['h_a'][2], deps['h_scale_s'][0], 4, 4, 5)
        flops += cal_conv_flops(deps['h_scale_s'][0], deps['h_scale_s'][1], 4, 4, 5)
        flops += cal_conv_flops(deps['h_scale_s'][1], deps['h_scale_s'][2], 8, 8, 5)

        former_ch_sum = 0
        for i in range(1, self.num_slices):
            flops += cal_conv_flops(deps['h_mean_s'][2] + former_ch_sum, deps['cc_mean_transforms'][i][0], 16, 16, 3)
            flops += cal_conv_flops(deps['cc_mean_transforms'][i][0], deps['cc_mean_transforms'][i][1], 16, 16, 3)
            flops += cal_conv_flops(deps['cc_mean_transforms'][i][1], deps['cc_mean_transforms'][i][2], 16, 16, 3)

            flops += cal_conv_flops(deps['h_scale_s'][2] + former_ch_sum, deps['cc_scale_transforms'][i][0], 16, 16, 3)
            flops += cal_conv_flops(deps['cc_scale_transforms'][i][0], deps['cc_scale_transforms'][i][1], 16, 16, 3)
            flops += cal_conv_flops(deps['cc_scale_transforms'][i][1], deps['cc_scale_transforms'][i][2], 16, 16, 3)

            flops += cal_conv_flops(deps['h_mean_s'][2] + former_ch_sum + deps['cc_mean_transforms'][i][2], deps['lrp_transforms'][i][0], 16, 16, 3)
            flops += cal_conv_flops(deps['lrp_transforms'][i][0], deps['lrp_transforms'][i][1], 16, 16, 3)
            flops += cal_conv_flops(deps['lrp_transforms'][i][1], deps['lrp_transforms'][i][2], 16, 16, 3)
            former_ch_sum += deps['cc_mean_transforms'][i][2]

        return flops

    def load_pretrained(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        cur_state_dict = self.state_dict()
        for k in state_dict:
            if k not in cur_state_dict:
                cur_k = k.rsplit('.', 1)[0]+'.conv.'+k.rsplit('.', 1)[1]
                assert cur_k in cur_state_dict
                cur_state_dict[cur_k] = state_dict[k]
            else:
                cur_state_dict[k] = state_dict[k]
        super().load_state_dict(cur_state_dict)