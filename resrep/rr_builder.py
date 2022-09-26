import torch
import numpy as np
import torch.nn.init as init
import torch.nn as nn
from compressai.layers import GDN

class RRBuilder:

    def __init__(self):
        self.cur_group_id = -1

    def Conv2dRR(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
               padding_mode='zeros'):
        assert type(kernel_size) is int
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        se = nn.Sequential()
        conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True,
                                padding_mode=padding_mode)
        se.add_module('conv', conv_layer)
        se.add_module('compactor', CompactorLayer(num_features=out_channels))
        print('use compactor on conv with kernel size {}'.format(kernel_size))
        return se

    def Conv2dGroupRR(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
               padding_mode='zeros', group_id=None):
        if group_id is None:
            self.cur_group_id += 1
            group_id = self.cur_group_id
        assert type(kernel_size) is int
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        se = nn.Sequential()
        conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True,
                                padding_mode=padding_mode)
        se.add_module('conv', conv_layer)
        se.add_module('compactor', CompactorLayer(num_features=out_channels, group_id=group_id))
        print('use compactor on conv with kernel size {}'.format(kernel_size))
        return se
    
    def GroupCompactor(self, channels, group_id):
        return nn.Sequential(CompactorLayer(channels, group_id))

    def ConvTranspose2dRR(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
               output_padding=0, padding_mode='zeros'):
        assert type(kernel_size) is int
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        se = nn.Sequential()
        conv_layer = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True,
                                output_padding=output_padding, padding_mode=padding_mode)
        se.add_module('conv', conv_layer)
        se.add_module('compactor', CompactorLayer(num_features=out_channels))
        print('use compactor on deconv with kernel size {}'.format(kernel_size))
        return se


class CompactorLayer(nn.Module):
    def __init__(self, num_features, group_id=-1):
        super(CompactorLayer, self).__init__()
        self.group_id = group_id
        self.pwc = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=1,
                          stride=1, padding=0, bias=False)
        identity_mat = np.eye(num_features, dtype=np.float32)

        # debug
        # identity_mat[0,0] = 0.
        # if num_features == 32:
        #     identity_mat[0,0] = 0.


        self.pwc.weight.data.copy_(torch.from_numpy(identity_mat).reshape(num_features, num_features, 1, 1))
        self.register_buffer('mask', torch.ones(num_features))
        init.ones_(self.mask)
        self.num_features = num_features

    def forward(self, inputs):
        return self.pwc(inputs)

    def set_mask_to_zero(self, zero_idx):
        zero_indices = list(np.where(self.mask.data.cpu() < 0.5)[0])
        zero_indices.append(zero_idx)
        new_mask_value = np.ones(self.num_features, dtype=np.float32)
        new_mask_value[np.array(zero_indices)] = 0.0
        self.mask.data = torch.from_numpy(new_mask_value).cuda().type(torch.cuda.FloatTensor)

    def get_num_mask_ones(self):
        mask_value = self.mask.cpu().numpy()
        return np.sum(mask_value == 1)

    def get_num_mask_zeros(self):
        mask_value = self.mask.cpu().numpy()
        return np.sum(mask_value == 0)

    def get_remain_ratio(self):
        return self.get_num_mask_ones() / self.num_features

    def get_pwc_kernel_detach(self):
        return self.pwc.weight.detach()
    
    def get_pwc_kernel(self):
        return self.pwc.weight

    def get_metric_vector(self):
        metric_vector = torch.sqrt(torch.sum(self.get_pwc_kernel_detach() ** 2, dim=(1, 2, 3))).cpu().numpy()
        return metric_vector