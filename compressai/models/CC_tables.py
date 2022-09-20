import torch


def gene_table(num_slices=10):
    cc_table = {
        'h_a.0.compactor': {'weight': 'h_a.0.conv.weight', 'bias': 'h_a.0.conv.bias', 'suc_weight': ['h_a.2.conv.weight'], 'conv': True, 'suc_conv': [True], 'type': 'split'},
        'h_a.2.compactor': {'weight': 'h_a.2.conv.weight', 'bias': 'h_a.2.conv.bias', 'suc_weight': ['h_a.4.conv.weight'], 'conv': True, 'suc_conv': [True], 'type': 'normal'},
        'h_a.4.compactor': {'weight': 'h_a.4.conv.weight', 'bias': 'h_a.4.conv.bias', 'suc_weight': ['h_mean_s.0.conv.weight', 'h_scale_s.0.conv.weight'], 'conv': True, 'suc_conv': [False, False], 'type': 'bottleneck'},
        'h_mean_s.0.compactor': {'weight': 'h_mean_s.0.conv.weight', 'bias': 'h_mean_s.0.conv.bias', 'suc_weight': ['h_mean_s.2.conv.weight'], 'conv': False, 'suc_conv': [False], 'type': 'normal'},
        'h_mean_s.2.compactor': {'weight': 'h_mean_s.2.conv.weight', 'bias': 'h_mean_s.2.conv.bias', 'suc_weight': ['h_mean_s.4.conv.weight'], 'conv': False, 'suc_conv': [True], 'type': 'normal'},
        'h_mean_s.4.compactor': {'weight': 'h_mean_s.4.conv.weight', 'bias': 'h_mean_s.4.conv.bias', 'suc_weight':
            ['cc_mean_transforms.{}.0.conv.weight'.format(i) for i in range(num_slices)] + ['lrp_transforms.{}.0.conv.weight'.format(i) for i in range(num_slices)]
        , 'conv': True, 'suc_conv': [True]*(2*num_slices), 'type': 'suc_partial', 'ch': [(0, 320)]*(2*num_slices)},
        'h_scale_s.0.compactor': {'weight': 'h_scale_s.0.conv.weight', 'bias': 'h_scale_s.0.conv.bias', 'suc_weight': ['h_scale_s.2.conv.weight'], 'conv': False, 'suc_conv': [False], 'type': 'normal'},
        'h_scale_s.2.compactor': {'weight': 'h_scale_s.2.conv.weight', 'bias': 'h_scale_s.2.conv.bias', 'suc_weight': ['h_scale_s.4.conv.weight'], 'conv': False, 'suc_conv': [True], 'type': 'normal'},
        'h_scale_s.4.compactor': {'weight': 'h_scale_s.4.conv.weight', 'bias': 'h_scale_s.4.conv.bias', 'suc_weight':
            ['cc_scale_transforms.{}.0.conv.weight'.format(i) for i in range(num_slices)]
        , 'conv': True, 'suc_conv': [True]*num_slices, 'type': 'suc_partial', 'ch': [(0, 320)]*num_slices}
        }
    for i in range(num_slices):
        cc_table['cc_mean_transforms.{}.0.compactor'.format(i)] = {
            'weight': 'cc_mean_transforms.{}.0.conv.weight'.format(i), 'bias': 'cc_mean_transforms.{}.0.conv.bias'.format(i), 'suc_weight': ['cc_mean_transforms.{}.2.conv.weight'.format(i)], 'conv': True, 'suc_conv': [True], 'type': 'split'
        }
        cc_table['cc_mean_transforms.{}.2.compactor'.format(i)] = {
            'weight': 'cc_mean_transforms.{}.2.conv.weight'.format(i), 'bias': 'cc_mean_transforms.{}.2.conv.bias'.format(i), 'suc_weight': ['cc_mean_transforms.{}.4.conv.weight'.format(i)], 'conv': True, 'suc_conv': [True], 'type': 'normal'
        }
        cc_table['cc_mean_transforms.{}.4.compactor'.format(i)] = {
            'weight': 'cc_mean_transforms.{}.4.conv.weight'.format(i), 'bias': 'cc_mean_transforms.{}.4.conv.bias'.format(i), 'suc_weight':
                ['cc_mean_transforms.{}.0.conv.weight'.format(j) for j in range(i+1, num_slices)] + ['lrp_transforms.{}.0.conv.weight'.format(j) for j in range(i, num_slices)]
            , 'conv': True, 'suc_conv': [True]*(2*(num_slices-i-1)+1), 'type': 'suc_partial', 'ch': [(320+i*320//num_slices, 320+(i+1)*320//num_slices)]*(num_slices-i-1) + [(320+i*320//num_slices, 320+(i+1)*320//num_slices)]*(num_slices-i)
        }
    for i in range(num_slices):
        cc_table['cc_scale_transforms.{}.0.compactor'.format(i)] = {
            'weight': 'cc_scale_transforms.{}.0.conv.weight'.format(i), 'bias': 'cc_scale_transforms.{}.0.conv.bias'.format(i), 'suc_weight': ['cc_scale_transforms.{}.2.conv.weight'.format(i)], 'conv': True, 'suc_conv': [True], 'type': 'split'
        }
        cc_table['cc_scale_transforms.{}.2.compactor'.format(i)] = {
            'weight': 'cc_scale_transforms.{}.2.conv.weight'.format(i), 'bias': 'cc_scale_transforms.{}.2.conv.bias'.format(i), 'suc_weight': ['cc_scale_transforms.{}.4.conv.weight'.format(i)], 'conv': True, 'suc_conv': [True], 'type': 'normal'
        }
        cc_table['cc_scale_transforms.{}.4.compactor'.format(i)] = {
            'weight': 'cc_scale_transforms.{}.4.conv.weight'.format(i), 'bias': 'cc_scale_transforms.{}.4.conv.bias'.format(i), 'suc_weight':
                ['cc_scale_transforms.{}.0.conv.weight'.format(j) for j in range(i+1, num_slices)] + ['lrp_transforms.{}.0.conv.weight'.format(j) for j in range(i+1, num_slices)]
            , 'conv': True, 'suc_conv': [True]*(2*(num_slices-i-1)), 'type': 'suc_partial', 'ch': [(320+i*320//num_slices, 320+(i+1)*320//num_slices)]*(2*(num_slices-i-1))
        }
    for i in range(num_slices):
        cc_table['lrp_transforms.{}.0.compactor'.format(i)] = {
            'weight': 'lrp_transforms.{}.0.conv.weight'.format(i), 'bias': 'lrp_transforms.{}.0.conv.bias'.format(i), 'suc_weight': ['lrp_transforms.{}.2.conv.weight'.format(i)], 'conv': True, 'suc_conv': [True], 'type': 'split'
        }
        cc_table['lrp_transforms.{}.2.compactor'.format(i)] = {
            'weight': 'lrp_transforms.{}.2.conv.weight'.format(i), 'bias': 'lrp_transforms.{}.2.conv.bias'.format(i), 'suc_weight': ['lrp_transforms.{}.4.conv.weight'.format(i)], 'conv': True, 'suc_conv': [True], 'type': 'normal'
        }
        cc_table['lrp_transforms.{}.4.compactor'.format(i)] = {
            'weight': 'lrp_transforms.{}.4.conv.weight'.format(i), 'bias': 'lrp_transforms.{}.4.conv.bias'.format(i), 'suc_weight':
                ['lrp_transforms.{}.0.conv.weight'.format(j) for j in range(i+1, num_slices)] + ['g_s.0.weight']
            , 'conv': True, 'suc_conv': [True]*(num_slices-i-1) + [False], 'type': 'suc_partial', 'ch': [(320+i*320//num_slices, 320+(i+1)*320//num_slices)]*(num_slices-i-1) + [(i*320//num_slices, (i+1)*320//num_slices)]
        }
    for i in range(num_slices):
        cc_table['y_compactors.{}.0'.format(i)] = {'weight': 'g_a.6.weight', 'bias': 'g_a.6.bias', 'suc_weight': ['h_a.0.conv.weight', 'g_s.0.weight'], 'conv': True, 'suc_conv': [True, False], 'type': 'partial', 'ch': [(i*320//num_slices, (i+1)*320//num_slices)]*3} # 3是weight + suc_weight一起
    return cc_table


def pre_split_before_pruning(state_dict, num_slices):
    # 只有g_a不太一样，分割第一维
    weight = state_dict.pop('g_a.6.weight')
    bias = state_dict.pop('g_a.6.bias')
    for i in range(num_slices):
        state_dict['g_a.6.weight_{}_{}'.format(i*320//num_slices, (i+1)*320//num_slices)] = weight[i*320//num_slices:(i+1)*320//num_slices]
        state_dict['g_a.6.bias_{}_{}'.format(i*320//num_slices, (i+1)*320//num_slices)] = bias[i*320//num_slices:(i+1)*320//num_slices]

    weight = state_dict.pop('h_a.0.conv.weight')
    for i in range(num_slices):
        state_dict['h_a.0.conv.weight_{}_{}'.format(i*320//num_slices, (i+1)*320//num_slices)] = weight[:, i*320//num_slices:(i+1)*320//num_slices]

    # g_s.0是deconv
    weight = state_dict.pop('g_s.0.weight')
    for i in range(num_slices):
        state_dict['g_s.0.weight_{}_{}'.format(i*320//num_slices, (i+1)*320//num_slices)] = weight[i*320//num_slices:(i+1)*320//num_slices]

    for i in range(num_slices):
        weight = state_dict.pop('cc_mean_transforms.{}.0.conv.weight'.format(i))
        state_dict['cc_mean_transforms.{}.0.conv.weight_0_320'.format(i)] = weight[:, :320]
        for j in range(i):
            state_dict['cc_mean_transforms.{}.0.conv.weight_{}_{}'.format(i, 320+j*320//num_slices, 320+(j+1)*320//num_slices)] = weight[:, 320+j*320//num_slices:320+(j+1)*320//num_slices]

    for i in range(num_slices):
        weight = state_dict.pop('cc_scale_transforms.{}.0.conv.weight'.format(i))
        state_dict['cc_scale_transforms.{}.0.conv.weight_0_320'.format(i)] = weight[:, :320]
        for j in range(i):
            state_dict['cc_scale_transforms.{}.0.conv.weight_{}_{}'.format(i, 320+j*320//num_slices, 320+(j+1)*320//num_slices)] = weight[:, 320+j*320//num_slices:320+(j+1)*320//num_slices]

    for i in range(num_slices):
        weight = state_dict.pop('lrp_transforms.{}.0.conv.weight'.format(i))
        state_dict['lrp_transforms.{}.0.conv.weight_0_320'.format(i)] = weight[:, :320]
        for j in range(i):
            state_dict['lrp_transforms.{}.0.conv.weight_{}_{}'.format(i, 320+j*320//num_slices, 320+(j+1)*320//num_slices)] = weight[:, 320+j*320//num_slices:320+(j+1)*320//num_slices]
        state_dict['lrp_transforms.{}.0.conv.weight_{}_{}'.format(i, 320+i*320//num_slices, 320+(i+1)*320//num_slices)] = weight[:, 320+i*320//num_slices:320+(i+1)*320//num_slices]

    return state_dict


def post_combine_after_pruning(state_dict, num_slices):
    # 只有g_a不太一样，分割第一维
    weights = []
    biases = []
    for i in range(num_slices):
        weights.append(state_dict.pop('g_a.6.weight_{}_{}'.format(i*320//num_slices, (i+1)*320//num_slices)))
        biases.append(state_dict.pop('g_a.6.bias_{}_{}'.format(i*320//num_slices, (i+1)*320//num_slices)))
    state_dict['g_a.6.weight'] = torch.cat(weights, dim=0)
    state_dict['g_a.6.bias'] = torch.cat(biases, dim=0)

    weights = []
    for i in range(num_slices):
        weights.append(state_dict.pop('h_a.0.conv.weight_{}_{}'.format(i*320//num_slices, (i+1)*320//num_slices)))
    state_dict['h_a.0.conv.weight'] = torch.cat(weights, dim=1)

    weights = []
    for i in range(num_slices):
        weights.append(state_dict.pop('g_s.0.weight_{}_{}'.format(i*320//num_slices, (i+1)*320//num_slices)))
    state_dict['g_s.0.weight'] = torch.cat(weights, dim=0)

    for i in range(num_slices):
        weights = []
        weights.append(state_dict.pop('cc_mean_transforms.{}.0.conv.weight_0_320'.format(i)))
        for j in range(i):
            weights.append(state_dict.pop('cc_mean_transforms.{}.0.conv.weight_{}_{}'.format(i, 320+j*320//num_slices, 320+(j+1)*320//num_slices)))
        state_dict['cc_mean_transforms.{}.0.conv.weight'.format(i)] = torch.cat(weights, dim=1)

    for i in range(num_slices):
        weights = []
        weights.append(state_dict.pop('cc_scale_transforms.{}.0.conv.weight_0_320'.format(i)))
        for j in range(i):
            weights.append(state_dict.pop('cc_scale_transforms.{}.0.conv.weight_{}_{}'.format(i, 320+j*320//num_slices, 320+(j+1)*320//num_slices)))
        state_dict['cc_scale_transforms.{}.0.conv.weight'.format(i)] = torch.cat(weights, dim=1)

    for i in range(num_slices):
        weights = []
        weights.append(state_dict.pop('lrp_transforms.{}.0.conv.weight_0_320'.format(i)))
        for j in range(i):
            weights.append(state_dict.pop('lrp_transforms.{}.0.conv.weight_{}_{}'.format(i, 320+j*320//num_slices, 320+(j+1)*320//num_slices)))
        weights.append(state_dict.pop('lrp_transforms.{}.0.conv.weight_{}_{}'.format(i, 320+i*320//num_slices, 320+(i+1)*320//num_slices)))
        state_dict['lrp_transforms.{}.0.conv.weight'.format(i)] = torch.cat(weights, dim=1)
    
    return state_dict