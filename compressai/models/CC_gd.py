import math
import torch
import torch.nn as nn
from collections import OrderedDict

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from .utils import conv, deconv, update_registered_buffers
from compressai.ops import ste_round
from compressai.layers import conv3x3, subpel_conv3x3, Win_noShift_Attention
from .base import CompressionModel


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class CC_GD(CompressionModel):
    """Channel-wise Context model"""

    def __init__(self, N=192, M=320, sparse_lambda=0.5, **kwargs):
        super().__init__(**kwargs)
        self.num_slices = 10
        self.max_support_slices = 5
        self.sparse_lambda = sparse_lambda

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
            GateDecorator(320),
            nn.ReLU(),
            conv(320, 256, stride=2),
            GateDecorator(256),
            nn.ReLU(),
            conv(256, 192, stride=2),
            GateDecorator(192),
        )

        self.h_mean_s = nn.Sequential(
            deconv(192, 192, stride=2),
            GateDecorator(192),
            nn.ReLU(),
            deconv(192, 256, stride=2),
            GateDecorator(256),
            nn.ReLU(),
            conv3x3(256, 320),
            GateDecorator(320),
        )

        self.h_scale_s = nn.Sequential(
            deconv(192, 192, stride=2),
            GateDecorator(192),
            nn.ReLU(),
            deconv(192, 256, stride=2),
            GateDecorator(256),
            nn.ReLU(),
            conv3x3(256, 320),
            GateDecorator(320),
        )
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv3x3(320 + 32*min(i, 5), 224),
                GateDecorator(224),
                nn.ReLU(),
                conv3x3(224, 128),
                GateDecorator(128),
                nn.ReLU(),
                conv3x3(128, 32),
            ) for i in range(10)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv3x3(320 + 32*min(i, 5), 224),
                GateDecorator(224),
                nn.ReLU(),
                conv3x3(224, 128),
                GateDecorator(128),
                nn.ReLU(),
                conv3x3(128, 32),
            ) for i in range(10)
            )
        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv3x3(320 + 32*min(i+1, 6), 224),
                GateDecorator(224),
                nn.ReLU(),
                conv3x3(224, 128),
                GateDecorator(128),
                nn.ReLU(),
                conv3x3(128, 32),
            ) for i in range(10)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

        self.gds = [
            self.h_a[1], self.h_a[4], self.h_a[7],
            self.h_mean_s[1], self.h_mean_s[4], self.h_mean_s[7],
            self.h_scale_s[1], self.h_scale_s[4], self.h_scale_s[7],
        ]
        self.gds += [self.cc_mean_transforms[i][1] for i in range(10)]
        self.gds += [self.cc_mean_transforms[i][4] for i in range(10)]
        self.gds += [self.cc_scale_transforms[i][1] for i in range(10)]
        self.gds += [self.cc_scale_transforms[i][4] for i in range(10)]
        self.gds += [self.lrp_transforms[i][1] for i in range(10)]
        self.gds += [self.lrp_transforms[i][4] for i in range(10)]

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

        y_slices = y.chunk(self.num_slices, 1)
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

    def load_state_dict(self, state_dict, ckpt=False):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        if ckpt:
            super().load_state_dict(state_dict)
        else:
            cur_state_dict = self.state_dict()
            for k, v in self.KEY_TABLE.items():
                cur_state_dict[v] = state_dict[k]
            super().load_state_dict(cur_state_dict)

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
    
    def param_scale(self):
        # TODO
        return 1.

    class PruneHelper:
        def __init__(self, type, weight, bias, suc_weight, gate, is_conv=True, suc_is_conv=True, start_ch=0, end_ch=-1) -> None:
            self.type = type
            self.weight = weight
            self.bias = bias
            self.suc_weight = suc_weight
            self.gate = gate
            self.is_conv = is_conv
            self.suc_is_conv = suc_is_conv
            self.start_ch = start_ch
            self.end_ch = end_ch

    KEY_TABLE = {
        'h_a.0.weight': 'h_a.0.weight',
        'h_a.0.bias': 'h_a.0.bias',
        'h_a.2.weight': 'h_a.3.weight',
        'h_a.2.bias': 'h_a.3.bias',
        'h_a.4.weight': 'h_a.6.weight',
        'h_a.4.bias': 'h_a.6.bias',
        'h_mean_s.0.weight': 'h_mean_s.0.weight',
        'h_mean_s.0.bias': 'h_mean_s.0.bias',
        'h_mean_s.2.weight': 'h_mean_s.3.weight',
        'h_mean_s.2.bias': 'h_mean_s.3.bias',
        'h_mean_s.4.weight': 'h_mean_s.6.weight',
        'h_mean_s.4.bias': 'h_mean_s.6.bias',
        'h_scale_s.0.weight': 'h_scale_s.0.weight',
        'h_scale_s.0.bias': 'h_scale_s.0.bias',
        'h_scale_s.2.weight': 'h_scale_s.3.weight',
        'h_scale_s.2.bias': 'h_scale_s.3.bias',
        'h_scale_s.4.weight': 'h_scale_s.6.weight',
        'h_scale_s.4.bias': 'h_scale_s.6.bias',
        'cc_mean_transforms.0.0.weight': 'cc_mean_transforms.0.0.weight',
        'cc_mean_transforms.0.0.bias': 'cc_mean_transforms.0.0.bias',
        'cc_mean_transforms.0.2.weight': 'cc_mean_transforms.0.3.weight',
        'cc_mean_transforms.0.2.bias': 'cc_mean_transforms.0.3.bias',
        'cc_mean_transforms.0.4.weight': 'cc_mean_transforms.0.6.weight',
        'cc_mean_transforms.0.4.bias': 'cc_mean_transforms.0.6.bias',
        'cc_mean_transforms.1.0.weight': 'cc_mean_transforms.1.0.weight',
        'cc_mean_transforms.1.0.bias': 'cc_mean_transforms.1.0.bias',
        'cc_mean_transforms.1.2.weight': 'cc_mean_transforms.1.3.weight',
        'cc_mean_transforms.1.2.bias': 'cc_mean_transforms.1.3.bias',
        'cc_mean_transforms.1.4.weight': 'cc_mean_transforms.1.6.weight',
        'cc_mean_transforms.1.4.bias': 'cc_mean_transforms.1.6.bias',
        'cc_mean_transforms.2.0.weight': 'cc_mean_transforms.2.0.weight',
        'cc_mean_transforms.2.0.bias': 'cc_mean_transforms.2.0.bias',
        'cc_mean_transforms.2.2.weight': 'cc_mean_transforms.2.3.weight',
        'cc_mean_transforms.2.2.bias': 'cc_mean_transforms.2.3.bias',
        'cc_mean_transforms.2.4.weight': 'cc_mean_transforms.2.6.weight',
        'cc_mean_transforms.2.4.bias': 'cc_mean_transforms.2.6.bias',
        'cc_mean_transforms.3.0.weight': 'cc_mean_transforms.3.0.weight',
        'cc_mean_transforms.3.0.bias': 'cc_mean_transforms.3.0.bias',
        'cc_mean_transforms.3.2.weight': 'cc_mean_transforms.3.3.weight',
        'cc_mean_transforms.3.2.bias': 'cc_mean_transforms.3.3.bias',
        'cc_mean_transforms.3.4.weight': 'cc_mean_transforms.3.6.weight',
        'cc_mean_transforms.3.4.bias': 'cc_mean_transforms.3.6.bias',
        'cc_mean_transforms.4.0.weight': 'cc_mean_transforms.4.0.weight',
        'cc_mean_transforms.4.0.bias': 'cc_mean_transforms.4.0.bias',
        'cc_mean_transforms.4.2.weight': 'cc_mean_transforms.4.3.weight',
        'cc_mean_transforms.4.2.bias': 'cc_mean_transforms.4.3.bias',
        'cc_mean_transforms.4.4.weight': 'cc_mean_transforms.4.6.weight',
        'cc_mean_transforms.4.4.bias': 'cc_mean_transforms.4.6.bias',
        'cc_mean_transforms.5.0.weight': 'cc_mean_transforms.5.0.weight',
        'cc_mean_transforms.5.0.bias': 'cc_mean_transforms.5.0.bias',
        'cc_mean_transforms.5.2.weight': 'cc_mean_transforms.5.3.weight',
        'cc_mean_transforms.5.2.bias': 'cc_mean_transforms.5.3.bias',
        'cc_mean_transforms.5.4.weight': 'cc_mean_transforms.5.6.weight',
        'cc_mean_transforms.5.4.bias': 'cc_mean_transforms.5.6.bias',
        'cc_mean_transforms.6.0.weight': 'cc_mean_transforms.6.0.weight',
        'cc_mean_transforms.6.0.bias': 'cc_mean_transforms.6.0.bias',
        'cc_mean_transforms.6.2.weight': 'cc_mean_transforms.6.3.weight',
        'cc_mean_transforms.6.2.bias': 'cc_mean_transforms.6.3.bias',
        'cc_mean_transforms.6.4.weight': 'cc_mean_transforms.6.6.weight',
        'cc_mean_transforms.6.4.bias': 'cc_mean_transforms.6.6.bias',
        'cc_mean_transforms.7.0.weight': 'cc_mean_transforms.7.0.weight',
        'cc_mean_transforms.7.0.bias': 'cc_mean_transforms.7.0.bias',
        'cc_mean_transforms.7.2.weight': 'cc_mean_transforms.7.3.weight',
        'cc_mean_transforms.7.2.bias': 'cc_mean_transforms.7.3.bias',
        'cc_mean_transforms.7.4.weight': 'cc_mean_transforms.7.6.weight',
        'cc_mean_transforms.7.4.bias': 'cc_mean_transforms.7.6.bias',
        'cc_mean_transforms.8.0.weight': 'cc_mean_transforms.8.0.weight',
        'cc_mean_transforms.8.0.bias': 'cc_mean_transforms.8.0.bias',
        'cc_mean_transforms.8.2.weight': 'cc_mean_transforms.8.3.weight',
        'cc_mean_transforms.8.2.bias': 'cc_mean_transforms.8.3.bias',
        'cc_mean_transforms.8.4.weight': 'cc_mean_transforms.8.6.weight',
        'cc_mean_transforms.8.4.bias': 'cc_mean_transforms.8.6.bias',
        'cc_mean_transforms.9.0.weight': 'cc_mean_transforms.9.0.weight',
        'cc_mean_transforms.9.0.bias': 'cc_mean_transforms.9.0.bias',
        'cc_mean_transforms.9.2.weight': 'cc_mean_transforms.9.3.weight',
        'cc_mean_transforms.9.2.bias': 'cc_mean_transforms.9.3.bias',
        'cc_mean_transforms.9.4.weight': 'cc_mean_transforms.9.6.weight',
        'cc_mean_transforms.9.4.bias': 'cc_mean_transforms.9.6.bias',
        'cc_scale_transforms.0.0.weight': 'cc_scale_transforms.0.0.weight',
        'cc_scale_transforms.0.0.bias': 'cc_scale_transforms.0.0.bias',
        'cc_scale_transforms.0.2.weight': 'cc_scale_transforms.0.3.weight',
        'cc_scale_transforms.0.2.bias': 'cc_scale_transforms.0.3.bias',
        'cc_scale_transforms.0.4.weight': 'cc_scale_transforms.0.6.weight',
        'cc_scale_transforms.0.4.bias': 'cc_scale_transforms.0.6.bias',
        'cc_scale_transforms.1.0.weight': 'cc_scale_transforms.1.0.weight',
        'cc_scale_transforms.1.0.bias': 'cc_scale_transforms.1.0.bias',
        'cc_scale_transforms.1.2.weight': 'cc_scale_transforms.1.3.weight',
        'cc_scale_transforms.1.2.bias': 'cc_scale_transforms.1.3.bias',
        'cc_scale_transforms.1.4.weight': 'cc_scale_transforms.1.6.weight',
        'cc_scale_transforms.1.4.bias': 'cc_scale_transforms.1.6.bias',
        'cc_scale_transforms.2.0.weight': 'cc_scale_transforms.2.0.weight',
        'cc_scale_transforms.2.0.bias': 'cc_scale_transforms.2.0.bias',
        'cc_scale_transforms.2.2.weight': 'cc_scale_transforms.2.3.weight',
        'cc_scale_transforms.2.2.bias': 'cc_scale_transforms.2.3.bias',
        'cc_scale_transforms.2.4.weight': 'cc_scale_transforms.2.6.weight',
        'cc_scale_transforms.2.4.bias': 'cc_scale_transforms.2.6.bias',
        'cc_scale_transforms.3.0.weight': 'cc_scale_transforms.3.0.weight',
        'cc_scale_transforms.3.0.bias': 'cc_scale_transforms.3.0.bias',
        'cc_scale_transforms.3.2.weight': 'cc_scale_transforms.3.3.weight',
        'cc_scale_transforms.3.2.bias': 'cc_scale_transforms.3.3.bias',
        'cc_scale_transforms.3.4.weight': 'cc_scale_transforms.3.6.weight',
        'cc_scale_transforms.3.4.bias': 'cc_scale_transforms.3.6.bias',
        'cc_scale_transforms.4.0.weight': 'cc_scale_transforms.4.0.weight',
        'cc_scale_transforms.4.0.bias': 'cc_scale_transforms.4.0.bias',
        'cc_scale_transforms.4.2.weight': 'cc_scale_transforms.4.3.weight',
        'cc_scale_transforms.4.2.bias': 'cc_scale_transforms.4.3.bias',
        'cc_scale_transforms.4.4.weight': 'cc_scale_transforms.4.6.weight',
        'cc_scale_transforms.4.4.bias': 'cc_scale_transforms.4.6.bias',
        'cc_scale_transforms.5.0.weight': 'cc_scale_transforms.5.0.weight',
        'cc_scale_transforms.5.0.bias': 'cc_scale_transforms.5.0.bias',
        'cc_scale_transforms.5.2.weight': 'cc_scale_transforms.5.3.weight',
        'cc_scale_transforms.5.2.bias': 'cc_scale_transforms.5.3.bias',
        'cc_scale_transforms.5.4.weight': 'cc_scale_transforms.5.6.weight',
        'cc_scale_transforms.5.4.bias': 'cc_scale_transforms.5.6.bias',
        'cc_scale_transforms.6.0.weight': 'cc_scale_transforms.6.0.weight',
        'cc_scale_transforms.6.0.bias': 'cc_scale_transforms.6.0.bias',
        'cc_scale_transforms.6.2.weight': 'cc_scale_transforms.6.3.weight',
        'cc_scale_transforms.6.2.bias': 'cc_scale_transforms.6.3.bias',
        'cc_scale_transforms.6.4.weight': 'cc_scale_transforms.6.6.weight',
        'cc_scale_transforms.6.4.bias': 'cc_scale_transforms.6.6.bias',
        'cc_scale_transforms.7.0.weight': 'cc_scale_transforms.7.0.weight',
        'cc_scale_transforms.7.0.bias': 'cc_scale_transforms.7.0.bias',
        'cc_scale_transforms.7.2.weight': 'cc_scale_transforms.7.3.weight',
        'cc_scale_transforms.7.2.bias': 'cc_scale_transforms.7.3.bias',
        'cc_scale_transforms.7.4.weight': 'cc_scale_transforms.7.6.weight',
        'cc_scale_transforms.7.4.bias': 'cc_scale_transforms.7.6.bias',
        'cc_scale_transforms.8.0.weight': 'cc_scale_transforms.8.0.weight',
        'cc_scale_transforms.8.0.bias': 'cc_scale_transforms.8.0.bias',
        'cc_scale_transforms.8.2.weight': 'cc_scale_transforms.8.3.weight',
        'cc_scale_transforms.8.2.bias': 'cc_scale_transforms.8.3.bias',
        'cc_scale_transforms.8.4.weight': 'cc_scale_transforms.8.6.weight',
        'cc_scale_transforms.8.4.bias': 'cc_scale_transforms.8.6.bias',
        'cc_scale_transforms.9.0.weight': 'cc_scale_transforms.9.0.weight',
        'cc_scale_transforms.9.0.bias': 'cc_scale_transforms.9.0.bias',
        'cc_scale_transforms.9.2.weight': 'cc_scale_transforms.9.3.weight',
        'cc_scale_transforms.9.2.bias': 'cc_scale_transforms.9.3.bias',
        'cc_scale_transforms.9.4.weight': 'cc_scale_transforms.9.6.weight',
        'cc_scale_transforms.9.4.bias': 'cc_scale_transforms.9.6.bias',
        'lrp_transforms.0.0.weight': 'lrp_transforms.0.0.weight',
        'lrp_transforms.0.0.bias': 'lrp_transforms.0.0.bias',
        'lrp_transforms.0.2.weight': 'lrp_transforms.0.3.weight',
        'lrp_transforms.0.2.bias': 'lrp_transforms.0.3.bias',
        'lrp_transforms.0.4.weight': 'lrp_transforms.0.6.weight',
        'lrp_transforms.0.4.bias': 'lrp_transforms.0.6.bias',
        'lrp_transforms.1.0.weight': 'lrp_transforms.1.0.weight',
        'lrp_transforms.1.0.bias': 'lrp_transforms.1.0.bias',
        'lrp_transforms.1.2.weight': 'lrp_transforms.1.3.weight',
        'lrp_transforms.1.2.bias': 'lrp_transforms.1.3.bias',
        'lrp_transforms.1.4.weight': 'lrp_transforms.1.6.weight',
        'lrp_transforms.1.4.bias': 'lrp_transforms.1.6.bias',
        'lrp_transforms.2.0.weight': 'lrp_transforms.2.0.weight',
        'lrp_transforms.2.0.bias': 'lrp_transforms.2.0.bias',
        'lrp_transforms.2.2.weight': 'lrp_transforms.2.3.weight',
        'lrp_transforms.2.2.bias': 'lrp_transforms.2.3.bias',
        'lrp_transforms.2.4.weight': 'lrp_transforms.2.6.weight',
        'lrp_transforms.2.4.bias': 'lrp_transforms.2.6.bias',
        'lrp_transforms.3.0.weight': 'lrp_transforms.3.0.weight',
        'lrp_transforms.3.0.bias': 'lrp_transforms.3.0.bias',
        'lrp_transforms.3.2.weight': 'lrp_transforms.3.3.weight',
        'lrp_transforms.3.2.bias': 'lrp_transforms.3.3.bias',
        'lrp_transforms.3.4.weight': 'lrp_transforms.3.6.weight',
        'lrp_transforms.3.4.bias': 'lrp_transforms.3.6.bias',
        'lrp_transforms.4.0.weight': 'lrp_transforms.4.0.weight',
        'lrp_transforms.4.0.bias': 'lrp_transforms.4.0.bias',
        'lrp_transforms.4.2.weight': 'lrp_transforms.4.3.weight',
        'lrp_transforms.4.2.bias': 'lrp_transforms.4.3.bias',
        'lrp_transforms.4.4.weight': 'lrp_transforms.4.6.weight',
        'lrp_transforms.4.4.bias': 'lrp_transforms.4.6.bias',
        'lrp_transforms.5.0.weight': 'lrp_transforms.5.0.weight',
        'lrp_transforms.5.0.bias': 'lrp_transforms.5.0.bias',
        'lrp_transforms.5.2.weight': 'lrp_transforms.5.3.weight',
        'lrp_transforms.5.2.bias': 'lrp_transforms.5.3.bias',
        'lrp_transforms.5.4.weight': 'lrp_transforms.5.6.weight',
        'lrp_transforms.5.4.bias': 'lrp_transforms.5.6.bias',
        'lrp_transforms.6.0.weight': 'lrp_transforms.6.0.weight',
        'lrp_transforms.6.0.bias': 'lrp_transforms.6.0.bias',
        'lrp_transforms.6.2.weight': 'lrp_transforms.6.3.weight',
        'lrp_transforms.6.2.bias': 'lrp_transforms.6.3.bias',
        'lrp_transforms.6.4.weight': 'lrp_transforms.6.6.weight',
        'lrp_transforms.6.4.bias': 'lrp_transforms.6.6.bias',
        'lrp_transforms.7.0.weight': 'lrp_transforms.7.0.weight',
        'lrp_transforms.7.0.bias': 'lrp_transforms.7.0.bias',
        'lrp_transforms.7.2.weight': 'lrp_transforms.7.3.weight',
        'lrp_transforms.7.2.bias': 'lrp_transforms.7.3.bias',
        'lrp_transforms.7.4.weight': 'lrp_transforms.7.6.weight',
        'lrp_transforms.7.4.bias': 'lrp_transforms.7.6.bias',
        'lrp_transforms.8.0.weight': 'lrp_transforms.8.0.weight',
        'lrp_transforms.8.0.bias': 'lrp_transforms.8.0.bias',
        'lrp_transforms.8.2.weight': 'lrp_transforms.8.3.weight',
        'lrp_transforms.8.2.bias': 'lrp_transforms.8.3.bias',
        'lrp_transforms.8.4.weight': 'lrp_transforms.8.6.weight',
        'lrp_transforms.8.4.bias': 'lrp_transforms.8.6.bias',
        'lrp_transforms.9.0.weight': 'lrp_transforms.9.0.weight',
        'lrp_transforms.9.0.bias': 'lrp_transforms.9.0.bias',
        'lrp_transforms.9.2.weight': 'lrp_transforms.9.3.weight',
        'lrp_transforms.9.2.bias': 'lrp_transforms.9.3.bias',
        'lrp_transforms.9.4.weight': 'lrp_transforms.9.6.weight',
        'lrp_transforms.9.4.bias': 'lrp_transforms.9.6.bias',
    }

    mask_weight_pairs = OrderedDict([
        ('h_a.1.mask', PruneHelper('normal', 'h_a.0.weight', 'h_a.0.bias', ['h_a.3.weight'], 'h_a.1.gate')),
        ('h_a.4.mask', PruneHelper('normal', 'h_a.3.weight', 'h_a.3.bias', ['h_a.6.weight'], 'h_a.4.gate')),
        ('h_a.7.mask', PruneHelper('bottleneck', 'h_a.6.weight', 'h_a.6.bias', ['h_mean_s.0.weight', 'h_scale_s.0.weight'], 'h_a.7.gate', suc_is_conv=False)),
        ('h_mean_s.1.mask', PruneHelper('normal', 'h_mean_s.0.weight', 'h_mean_s.0.bias', ['h_mean_s.3.weight'], 'h_mean_s.1.gate', is_conv=False, suc_is_conv=False)),
        ('h_mean_s.4.mask', PruneHelper('normal', 'h_mean_s.3.weight', 'h_mean_s.3.bias', ['h_mean_s.6.weight'], 'h_mean_s.4.gate', is_conv=False)),
        ('h_mean_s.7.mask', PruneHelper('normal', 'h_mean_s.6.weight', 'h_mean_s.6.bias', [
            'cc_mean_transforms.0.0.weight',
            'cc_mean_transforms.1.0.weight',
            'cc_mean_transforms.2.0.weight',
            'cc_mean_transforms.3.0.weight',
            'cc_mean_transforms.4.0.weight',
            'cc_mean_transforms.5.0.weight',
            'cc_mean_transforms.6.0.weight',
            'cc_mean_transforms.7.0.weight',
            'cc_mean_transforms.8.0.weight',
            'cc_mean_transforms.9.0.weight',
            'lrp_transforms.0.0.weight',
            'lrp_transforms.1.0.weight',
            'lrp_transforms.2.0.weight',
            'lrp_transforms.3.0.weight',
            'lrp_transforms.4.0.weight',
            'lrp_transforms.5.0.weight',
            'lrp_transforms.6.0.weight',
            'lrp_transforms.7.0.weight',
            'lrp_transforms.8.0.weight',
            'lrp_transforms.9.0.weight',
            ], 'h_mean_s.7.gate', end_ch=320)),
        ('h_scale_s.1.mask', PruneHelper('normal', 'h_scale_s.0.weight', 'h_scale_s.0.bias', ['h_scale_s.3.weight'], 'h_scale_s.1.gate', is_conv=False, suc_is_conv=False)),
        ('h_scale_s.4.mask', PruneHelper('normal', 'h_scale_s.3.weight', 'h_scale_s.3.bias', ['h_scale_s.6.weight'], 'h_scale_s.4.gate', is_conv=False)),
        ('h_scale_s.7.mask', PruneHelper('normal', 'h_scale_s.6.weight', 'h_scale_s.6.bias', [
            'cc_scale_transforms.0.0.weight',
            'cc_scale_transforms.1.0.weight',
            'cc_scale_transforms.2.0.weight',
            'cc_scale_transforms.3.0.weight',
            'cc_scale_transforms.4.0.weight',
            'cc_scale_transforms.5.0.weight',
            'cc_scale_transforms.6.0.weight',
            'cc_scale_transforms.7.0.weight',
            'cc_scale_transforms.8.0.weight',
            'cc_scale_transforms.9.0.weight',
            ], 'h_scale_s.7.gate', end_ch=320)),
        ('cc_mean_transforms.0.1.mask', PruneHelper('normal', 'cc_mean_transforms.0.0.weight', 'cc_mean_transforms.0.0.bias', ['cc_mean_transforms.0.3.weight'], 'cc_mean_transforms.0.1.gate')),
        ('cc_mean_transforms.0.4.mask', PruneHelper('normal', 'cc_mean_transforms.0.3.weight', 'cc_mean_transforms.0.3.bias', ['cc_mean_transforms.0.6.weight'], 'cc_mean_transforms.0.4.gate')),
        ('cc_mean_transforms.1.1.mask', PruneHelper('normal', 'cc_mean_transforms.1.0.weight', 'cc_mean_transforms.1.0.bias', ['cc_mean_transforms.1.3.weight'], 'cc_mean_transforms.1.1.gate')),
        ('cc_mean_transforms.1.4.mask', PruneHelper('normal', 'cc_mean_transforms.1.3.weight', 'cc_mean_transforms.1.3.bias', ['cc_mean_transforms.1.6.weight'], 'cc_mean_transforms.1.4.gate')),
        ('cc_mean_transforms.2.1.mask', PruneHelper('normal', 'cc_mean_transforms.2.0.weight', 'cc_mean_transforms.2.0.bias', ['cc_mean_transforms.2.3.weight'], 'cc_mean_transforms.2.1.gate')),
        ('cc_mean_transforms.2.4.mask', PruneHelper('normal', 'cc_mean_transforms.2.3.weight', 'cc_mean_transforms.2.3.bias', ['cc_mean_transforms.2.6.weight'], 'cc_mean_transforms.2.4.gate')),
        ('cc_mean_transforms.3.1.mask', PruneHelper('normal', 'cc_mean_transforms.3.0.weight', 'cc_mean_transforms.3.0.bias', ['cc_mean_transforms.3.3.weight'], 'cc_mean_transforms.3.1.gate')),
        ('cc_mean_transforms.3.4.mask', PruneHelper('normal', 'cc_mean_transforms.3.3.weight', 'cc_mean_transforms.3.3.bias', ['cc_mean_transforms.3.6.weight'], 'cc_mean_transforms.3.4.gate')),
        ('cc_mean_transforms.4.1.mask', PruneHelper('normal', 'cc_mean_transforms.4.0.weight', 'cc_mean_transforms.4.0.bias', ['cc_mean_transforms.4.3.weight'], 'cc_mean_transforms.4.1.gate')),
        ('cc_mean_transforms.4.4.mask', PruneHelper('normal', 'cc_mean_transforms.4.3.weight', 'cc_mean_transforms.4.3.bias', ['cc_mean_transforms.4.6.weight'], 'cc_mean_transforms.4.4.gate')),
        ('cc_mean_transforms.5.1.mask', PruneHelper('normal', 'cc_mean_transforms.5.0.weight', 'cc_mean_transforms.5.0.bias', ['cc_mean_transforms.5.3.weight'], 'cc_mean_transforms.5.1.gate')),
        ('cc_mean_transforms.5.4.mask', PruneHelper('normal', 'cc_mean_transforms.5.3.weight', 'cc_mean_transforms.5.3.bias', ['cc_mean_transforms.5.6.weight'], 'cc_mean_transforms.5.4.gate')),
        ('cc_mean_transforms.6.1.mask', PruneHelper('normal', 'cc_mean_transforms.6.0.weight', 'cc_mean_transforms.6.0.bias', ['cc_mean_transforms.6.3.weight'], 'cc_mean_transforms.6.1.gate')),
        ('cc_mean_transforms.6.4.mask', PruneHelper('normal', 'cc_mean_transforms.6.3.weight', 'cc_mean_transforms.6.3.bias', ['cc_mean_transforms.6.6.weight'], 'cc_mean_transforms.6.4.gate')),
        ('cc_mean_transforms.7.1.mask', PruneHelper('normal', 'cc_mean_transforms.7.0.weight', 'cc_mean_transforms.7.0.bias', ['cc_mean_transforms.7.3.weight'], 'cc_mean_transforms.7.1.gate')),
        ('cc_mean_transforms.7.4.mask', PruneHelper('normal', 'cc_mean_transforms.7.3.weight', 'cc_mean_transforms.7.3.bias', ['cc_mean_transforms.7.6.weight'], 'cc_mean_transforms.7.4.gate')),
        ('cc_mean_transforms.8.1.mask', PruneHelper('normal', 'cc_mean_transforms.8.0.weight', 'cc_mean_transforms.8.0.bias', ['cc_mean_transforms.8.3.weight'], 'cc_mean_transforms.8.1.gate')),
        ('cc_mean_transforms.8.4.mask', PruneHelper('normal', 'cc_mean_transforms.8.3.weight', 'cc_mean_transforms.8.3.bias', ['cc_mean_transforms.8.6.weight'], 'cc_mean_transforms.8.4.gate')),
        ('cc_mean_transforms.9.1.mask', PruneHelper('normal', 'cc_mean_transforms.9.0.weight', 'cc_mean_transforms.9.0.bias', ['cc_mean_transforms.9.3.weight'], 'cc_mean_transforms.9.1.gate')),
        ('cc_mean_transforms.9.4.mask', PruneHelper('normal', 'cc_mean_transforms.9.3.weight', 'cc_mean_transforms.9.3.bias', ['cc_mean_transforms.9.6.weight'], 'cc_mean_transforms.9.4.gate')),
        ('cc_scale_transforms.0.1.mask', PruneHelper('normal', 'cc_scale_transforms.0.0.weight', 'cc_scale_transforms.0.0.bias', ['cc_scale_transforms.0.3.weight'], 'cc_scale_transforms.0.1.gate')),
        ('cc_scale_transforms.0.4.mask', PruneHelper('normal', 'cc_scale_transforms.0.3.weight', 'cc_scale_transforms.0.3.bias', ['cc_scale_transforms.0.6.weight'], 'cc_scale_transforms.0.4.gate')),
        ('cc_scale_transforms.1.1.mask', PruneHelper('normal', 'cc_scale_transforms.1.0.weight', 'cc_scale_transforms.1.0.bias', ['cc_scale_transforms.1.3.weight'], 'cc_scale_transforms.1.1.gate')),
        ('cc_scale_transforms.1.4.mask', PruneHelper('normal', 'cc_scale_transforms.1.3.weight', 'cc_scale_transforms.1.3.bias', ['cc_scale_transforms.1.6.weight'], 'cc_scale_transforms.1.4.gate')),
        ('cc_scale_transforms.2.1.mask', PruneHelper('normal', 'cc_scale_transforms.2.0.weight', 'cc_scale_transforms.2.0.bias', ['cc_scale_transforms.2.3.weight'], 'cc_scale_transforms.2.1.gate')),
        ('cc_scale_transforms.2.4.mask', PruneHelper('normal', 'cc_scale_transforms.2.3.weight', 'cc_scale_transforms.2.3.bias', ['cc_scale_transforms.2.6.weight'], 'cc_scale_transforms.2.4.gate')),
        ('cc_scale_transforms.3.1.mask', PruneHelper('normal', 'cc_scale_transforms.3.0.weight', 'cc_scale_transforms.3.0.bias', ['cc_scale_transforms.3.3.weight'], 'cc_scale_transforms.3.1.gate')),
        ('cc_scale_transforms.3.4.mask', PruneHelper('normal', 'cc_scale_transforms.3.3.weight', 'cc_scale_transforms.3.3.bias', ['cc_scale_transforms.3.6.weight'], 'cc_scale_transforms.3.4.gate')),
        ('cc_scale_transforms.4.1.mask', PruneHelper('normal', 'cc_scale_transforms.4.0.weight', 'cc_scale_transforms.4.0.bias', ['cc_scale_transforms.4.3.weight'], 'cc_scale_transforms.4.1.gate')),
        ('cc_scale_transforms.4.4.mask', PruneHelper('normal', 'cc_scale_transforms.4.3.weight', 'cc_scale_transforms.4.3.bias', ['cc_scale_transforms.4.6.weight'], 'cc_scale_transforms.4.4.gate')),
        ('cc_scale_transforms.5.1.mask', PruneHelper('normal', 'cc_scale_transforms.5.0.weight', 'cc_scale_transforms.5.0.bias', ['cc_scale_transforms.5.3.weight'], 'cc_scale_transforms.5.1.gate')),
        ('cc_scale_transforms.5.4.mask', PruneHelper('normal', 'cc_scale_transforms.5.3.weight', 'cc_scale_transforms.5.3.bias', ['cc_scale_transforms.5.6.weight'], 'cc_scale_transforms.5.4.gate')),
        ('cc_scale_transforms.6.1.mask', PruneHelper('normal', 'cc_scale_transforms.6.0.weight', 'cc_scale_transforms.6.0.bias', ['cc_scale_transforms.6.3.weight'], 'cc_scale_transforms.6.1.gate')),
        ('cc_scale_transforms.6.4.mask', PruneHelper('normal', 'cc_scale_transforms.6.3.weight', 'cc_scale_transforms.6.3.bias', ['cc_scale_transforms.6.6.weight'], 'cc_scale_transforms.6.4.gate')),
        ('cc_scale_transforms.7.1.mask', PruneHelper('normal', 'cc_scale_transforms.7.0.weight', 'cc_scale_transforms.7.0.bias', ['cc_scale_transforms.7.3.weight'], 'cc_scale_transforms.7.1.gate')),
        ('cc_scale_transforms.7.4.mask', PruneHelper('normal', 'cc_scale_transforms.7.3.weight', 'cc_scale_transforms.7.3.bias', ['cc_scale_transforms.7.6.weight'], 'cc_scale_transforms.7.4.gate')),
        ('cc_scale_transforms.8.1.mask', PruneHelper('normal', 'cc_scale_transforms.8.0.weight', 'cc_scale_transforms.8.0.bias', ['cc_scale_transforms.8.3.weight'], 'cc_scale_transforms.8.1.gate')),
        ('cc_scale_transforms.8.4.mask', PruneHelper('normal', 'cc_scale_transforms.8.3.weight', 'cc_scale_transforms.8.3.bias', ['cc_scale_transforms.8.6.weight'], 'cc_scale_transforms.8.4.gate')),
        ('cc_scale_transforms.9.1.mask', PruneHelper('normal', 'cc_scale_transforms.9.0.weight', 'cc_scale_transforms.9.0.bias', ['cc_scale_transforms.9.3.weight'], 'cc_scale_transforms.9.1.gate')),
        ('cc_scale_transforms.9.4.mask', PruneHelper('normal', 'cc_scale_transforms.9.3.weight', 'cc_scale_transforms.9.3.bias', ['cc_scale_transforms.9.6.weight'], 'cc_scale_transforms.9.4.gate')),
        ('lrp_transforms.0.1.mask', PruneHelper('normal', 'lrp_transforms.0.0.weight', 'lrp_transforms.0.0.bias', ['lrp_transforms.0.3.weight'], 'lrp_transforms.0.1.gate')),
        ('lrp_transforms.0.4.mask', PruneHelper('normal', 'lrp_transforms.0.3.weight', 'lrp_transforms.0.3.bias', ['lrp_transforms.0.6.weight'], 'lrp_transforms.0.4.gate')),
        ('lrp_transforms.1.1.mask', PruneHelper('normal', 'lrp_transforms.1.0.weight', 'lrp_transforms.1.0.bias', ['lrp_transforms.1.3.weight'], 'lrp_transforms.1.1.gate')),
        ('lrp_transforms.1.4.mask', PruneHelper('normal', 'lrp_transforms.1.3.weight', 'lrp_transforms.1.3.bias', ['lrp_transforms.1.6.weight'], 'lrp_transforms.1.4.gate')),
        ('lrp_transforms.2.1.mask', PruneHelper('normal', 'lrp_transforms.2.0.weight', 'lrp_transforms.2.0.bias', ['lrp_transforms.2.3.weight'], 'lrp_transforms.2.1.gate')),
        ('lrp_transforms.2.4.mask', PruneHelper('normal', 'lrp_transforms.2.3.weight', 'lrp_transforms.2.3.bias', ['lrp_transforms.2.6.weight'], 'lrp_transforms.2.4.gate')),
        ('lrp_transforms.3.1.mask', PruneHelper('normal', 'lrp_transforms.3.0.weight', 'lrp_transforms.3.0.bias', ['lrp_transforms.3.3.weight'], 'lrp_transforms.3.1.gate')),
        ('lrp_transforms.3.4.mask', PruneHelper('normal', 'lrp_transforms.3.3.weight', 'lrp_transforms.3.3.bias', ['lrp_transforms.3.6.weight'], 'lrp_transforms.3.4.gate')),
        ('lrp_transforms.4.1.mask', PruneHelper('normal', 'lrp_transforms.4.0.weight', 'lrp_transforms.4.0.bias', ['lrp_transforms.4.3.weight'], 'lrp_transforms.4.1.gate')),
        ('lrp_transforms.4.4.mask', PruneHelper('normal', 'lrp_transforms.4.3.weight', 'lrp_transforms.4.3.bias', ['lrp_transforms.4.6.weight'], 'lrp_transforms.4.4.gate')),
        ('lrp_transforms.5.1.mask', PruneHelper('normal', 'lrp_transforms.5.0.weight', 'lrp_transforms.5.0.bias', ['lrp_transforms.5.3.weight'], 'lrp_transforms.5.1.gate')),
        ('lrp_transforms.5.4.mask', PruneHelper('normal', 'lrp_transforms.5.3.weight', 'lrp_transforms.5.3.bias', ['lrp_transforms.5.6.weight'], 'lrp_transforms.5.4.gate')),
        ('lrp_transforms.6.1.mask', PruneHelper('normal', 'lrp_transforms.6.0.weight', 'lrp_transforms.6.0.bias', ['lrp_transforms.6.3.weight'], 'lrp_transforms.6.1.gate')),
        ('lrp_transforms.6.4.mask', PruneHelper('normal', 'lrp_transforms.6.3.weight', 'lrp_transforms.6.3.bias', ['lrp_transforms.6.6.weight'], 'lrp_transforms.6.4.gate')),
        ('lrp_transforms.7.1.mask', PruneHelper('normal', 'lrp_transforms.7.0.weight', 'lrp_transforms.7.0.bias', ['lrp_transforms.7.3.weight'], 'lrp_transforms.7.1.gate')),
        ('lrp_transforms.7.4.mask', PruneHelper('normal', 'lrp_transforms.7.3.weight', 'lrp_transforms.7.3.bias', ['lrp_transforms.7.6.weight'], 'lrp_transforms.7.4.gate')),
        ('lrp_transforms.8.1.mask', PruneHelper('normal', 'lrp_transforms.8.0.weight', 'lrp_transforms.8.0.bias', ['lrp_transforms.8.3.weight'], 'lrp_transforms.8.1.gate')),
        ('lrp_transforms.8.4.mask', PruneHelper('normal', 'lrp_transforms.8.3.weight', 'lrp_transforms.8.3.bias', ['lrp_transforms.8.6.weight'], 'lrp_transforms.8.4.gate')),
        ('lrp_transforms.9.1.mask', PruneHelper('normal', 'lrp_transforms.9.0.weight', 'lrp_transforms.9.0.bias', ['lrp_transforms.9.3.weight'], 'lrp_transforms.9.1.gate')),
        ('lrp_transforms.9.4.mask', PruneHelper('normal', 'lrp_transforms.9.3.weight', 'lrp_transforms.9.3.bias', ['lrp_transforms.9.6.weight'], 'lrp_transforms.9.4.gate')),
        ])

    to_be_pop = [
        'h_a.1.gate', 'h_a.1.score',
        'h_a.4.gate', 'h_a.4.score',
        'h_a.7.gate', 'h_a.7.score',
        'h_mean_s.1.gate', 'h_mean_s.1.score',
        'h_mean_s.4.gate', 'h_mean_s.4.score',
        'h_mean_s.7.gate', 'h_mean_s.7.score',
        'h_scale_s.1.gate', 'h_scale_s.1.score',
        'h_scale_s.4.gate', 'h_scale_s.4.score',
        'h_scale_s.7.gate', 'h_scale_s.7.score',
        'cc_mean_transforms.0.1.gate', 'cc_mean_transforms.0.1.score',
        'cc_mean_transforms.0.4.gate', 'cc_mean_transforms.0.4.score',
        'cc_mean_transforms.1.1.gate', 'cc_mean_transforms.1.1.score',
        'cc_mean_transforms.1.4.gate', 'cc_mean_transforms.1.4.score',
        'cc_mean_transforms.2.1.gate', 'cc_mean_transforms.2.1.score',
        'cc_mean_transforms.2.4.gate', 'cc_mean_transforms.2.4.score',
        'cc_mean_transforms.3.1.gate', 'cc_mean_transforms.3.1.score',
        'cc_mean_transforms.3.4.gate', 'cc_mean_transforms.3.4.score',
        'cc_mean_transforms.4.1.gate', 'cc_mean_transforms.4.1.score',
        'cc_mean_transforms.4.4.gate', 'cc_mean_transforms.4.4.score',
        'cc_mean_transforms.5.1.gate', 'cc_mean_transforms.5.1.score',
        'cc_mean_transforms.5.4.gate', 'cc_mean_transforms.5.4.score',
        'cc_mean_transforms.6.1.gate', 'cc_mean_transforms.6.1.score',
        'cc_mean_transforms.6.4.gate', 'cc_mean_transforms.6.4.score',
        'cc_mean_transforms.7.1.gate', 'cc_mean_transforms.7.1.score',
        'cc_mean_transforms.7.4.gate', 'cc_mean_transforms.7.4.score',
        'cc_mean_transforms.8.1.gate', 'cc_mean_transforms.8.1.score',
        'cc_mean_transforms.8.4.gate', 'cc_mean_transforms.8.4.score',
        'cc_mean_transforms.9.1.gate', 'cc_mean_transforms.9.1.score',
        'cc_mean_transforms.9.4.gate', 'cc_mean_transforms.9.4.score',
        'cc_scale_transforms.0.1.gate', 'cc_scale_transforms.0.1.score',
        'cc_scale_transforms.0.4.gate', 'cc_scale_transforms.0.4.score',
        'cc_scale_transforms.1.1.gate', 'cc_scale_transforms.1.1.score',
        'cc_scale_transforms.1.4.gate', 'cc_scale_transforms.1.4.score',
        'cc_scale_transforms.2.1.gate', 'cc_scale_transforms.2.1.score',
        'cc_scale_transforms.2.4.gate', 'cc_scale_transforms.2.4.score',
        'cc_scale_transforms.3.1.gate', 'cc_scale_transforms.3.1.score',
        'cc_scale_transforms.3.4.gate', 'cc_scale_transforms.3.4.score',
        'cc_scale_transforms.4.1.gate', 'cc_scale_transforms.4.1.score',
        'cc_scale_transforms.4.4.gate', 'cc_scale_transforms.4.4.score',
        'cc_scale_transforms.5.1.gate', 'cc_scale_transforms.5.1.score',
        'cc_scale_transforms.5.4.gate', 'cc_scale_transforms.5.4.score',
        'cc_scale_transforms.6.1.gate', 'cc_scale_transforms.6.1.score',
        'cc_scale_transforms.6.4.gate', 'cc_scale_transforms.6.4.score',
        'cc_scale_transforms.7.1.gate', 'cc_scale_transforms.7.1.score',
        'cc_scale_transforms.7.4.gate', 'cc_scale_transforms.7.4.score',
        'cc_scale_transforms.8.1.gate', 'cc_scale_transforms.8.1.score',
        'cc_scale_transforms.8.4.gate', 'cc_scale_transforms.8.4.score',
        'cc_scale_transforms.9.1.gate', 'cc_scale_transforms.9.1.score',
        'cc_scale_transforms.9.4.gate', 'cc_scale_transforms.9.4.score',
        'lrp_transforms.0.1.gate', 'lrp_transforms.0.1.score',
        'lrp_transforms.0.4.gate', 'lrp_transforms.0.4.score',
        'lrp_transforms.1.1.gate', 'lrp_transforms.1.1.score',
        'lrp_transforms.1.4.gate', 'lrp_transforms.1.4.score',
        'lrp_transforms.2.1.gate', 'lrp_transforms.2.1.score',
        'lrp_transforms.2.4.gate', 'lrp_transforms.2.4.score',
        'lrp_transforms.3.1.gate', 'lrp_transforms.3.1.score',
        'lrp_transforms.3.4.gate', 'lrp_transforms.3.4.score',
        'lrp_transforms.4.1.gate', 'lrp_transforms.4.1.score',
        'lrp_transforms.4.4.gate', 'lrp_transforms.4.4.score',
        'lrp_transforms.5.1.gate', 'lrp_transforms.5.1.score',
        'lrp_transforms.5.4.gate', 'lrp_transforms.5.4.score',
        'lrp_transforms.6.1.gate', 'lrp_transforms.6.1.score',
        'lrp_transforms.6.4.gate', 'lrp_transforms.6.4.score',
        'lrp_transforms.7.1.gate', 'lrp_transforms.7.1.score',
        'lrp_transforms.7.4.gate', 'lrp_transforms.7.4.score',
        'lrp_transforms.8.1.gate', 'lrp_transforms.8.1.score',
        'lrp_transforms.8.4.gate', 'lrp_transforms.8.4.score',
        'lrp_transforms.9.1.gate', 'lrp_transforms.9.1.score',
        'lrp_transforms.9.4.gate', 'lrp_transforms.9.4.score',
        ]


class GateDecorator(nn.Module):
    def __init__(self, channel_size, minimal=0.04):
        super(GateDecorator, self).__init__()
        self.channel_size = channel_size
        self.minimal = int(minimal*channel_size)
        self.gate = nn.Parameter(torch.ones(1, self.channel_size, 1, 1), requires_grad=True)

        self.register_buffer('score', torch.zeros(1, self.channel_size, 1, 1))
        self.register_buffer('mask', torch.ones(1, self.channel_size, 1, 1))
        # self.mask[:, 0] -= 1.1
    
    def cal_score(self):
        self.score += (self.gate.grad * self.gate).abs()
    
    def get_score(self):
        return self.score.view(-1)

    def reset_score(self):
        self.score.zero_()

    def forward(self, x):
        return x * self.gate * self.mask
