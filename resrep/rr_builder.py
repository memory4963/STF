import torch
import numpy as np
import torch.nn.init as init
import torch.nn as nn
from compressai.layers import GDN

class RRBuilder:

    def __init__(self, group_fisher=False):
        self.cur_group_id = -1
        self.group_fisher = group_fisher

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
        se.add_module('compactor', CompactorLayer(num_features=out_channels, group_fisher=self.group_fisher))
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
        se.add_module('compactor', CompactorLayer(num_features=out_channels, group_fisher=self.group_fisher))
        return se
    
    def SingleCompactor(self, channels, excluded):
        return nn.Sequential(CompactorLayer(channels, excluded=excluded, group_fisher=self.group_fisher))

    def SingleEnhancedCompactor(self, in_channels, out_channels, start_channel, excluded):
        return nn.Sequential(EnhancedCompactorLayer(in_channels, out_channels, start_channel, excluded=excluded, group_fisher=self.group_fisher))

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
        se.add_module('compactor', CompactorLayer(num_features=out_channels, group_fisher=self.group_fisher))
        return se

# x=0
class CompactorLayer(nn.Module):
    def __init__(self, num_features, excluded=False, group_fisher=False):
        super(CompactorLayer, self).__init__()
        self.pwc = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=1,
                          stride=1, padding=0, bias=False)
        identity_mat = np.eye(num_features, dtype=np.float32)

        # debug
        # identity_mat[0,0] = 0.
        # global x
        # identity_mat[x%num_features, x%num_features] = 0.
        # x += 1

        self.pwc.weight.data.copy_(torch.from_numpy(identity_mat).reshape(num_features, num_features, 1, 1))
        self.register_buffer('mask', torch.ones(num_features))
        init.ones_(self.mask)
        self.num_features = num_features
        self.excluded = excluded
        self.group_fisher = group_fisher
        if group_fisher:
            self.init_fisher()
            self.saved_output = 0
            self.register_forward_hook(self.save_input_forward_hook)
            self.register_backward_hook(self.compute_fisher_backward_hook)

    def forward(self, inputs):
        return self.pwc(inputs)

    def set_mask_to_zero(self, zero_idx):
        zero_indices = list(np.where(self.mask.data.cpu() < 0.5)[0])
        zero_indices.append(zero_idx)
        new_mask_value = np.ones(self.num_features, dtype=np.float32)
        new_mask_value[np.array(zero_indices)] = 0.0
        self.mask.data = torch.from_numpy(new_mask_value).cuda().type(torch.cuda.FloatTensor)

    def mask_weight_grad(self):
        weight = self.get_pwc_kernel()
        weight.grad.data = torch.einsum('i,ijkl->ijkl', self.mask, weight.grad.data)

    def add_lasso_penalty(self, lasso_strength):
        weight = self.get_pwc_kernel()
        lasso_grad = weight.data * ((weight.data ** 2).sum(dim=(1, 2, 3), keepdim=True) ** (-0.5))
        weight.grad.data.add_(lasso_strength, lasso_grad)

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

    def get_fisher(self):
        return self.fisher.cpu().numpy()

    def save_input_forward_hook(self, module, inputs, outputs):
        if outputs.requires_grad:
            self.saved_output = outputs

    def compute_fisher_backward_hook(self, module, grad_input, grad_output):
        self.fisher += (self.saved_output * grad_output[0]).sum([0, -1, -2])

    def init_fisher(self):
        if hasattr(self, 'fisher'):
            self.fisher.zero_()
        else:
            self.register_buffer('fisher', torch.zeros(self.num_features))

class EnhancedCompactorLayer(CompactorLayer):
    def __init__(self, in_channels, out_channels, start_channel, excluded=False, group_fisher=False):
        super().__init__(out_channels, excluded, group_fisher)

        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=1, padding=0, bias=False)
        identity_mat = torch.zeros([out_channels, in_channels], dtype=torch.float32)
        idxes = torch.Tensor([[start_channel+i] for i in range(out_channels)]).long()
        identity_mat.scatter_(1, idxes, 1.)

        # debug
        # identity_mat[0,idxes[0][0]] = 0.

        self.pwc.weight.data.copy_(identity_mat.reshape(out_channels, in_channels, 1, 1))

        self.in_channels = in_channels
        self.out_channels = out_channels # equal to self.num_features
        self.start_channel = start_channel
