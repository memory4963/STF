import numpy as np
from compressai.models import CC_tables
from resrep.rr_builder import CompactorLayer, RRBuilder
import torch
import torch.nn.functional as F


def get_remain(vec: CompactorLayer, thr, min_channel=1):
    return max(min_channel, np.sum(vec.get_metric_vector(cal_deps=True) >= thr))


def get_group_remain(compactors, thr):
    remaining = np.zeros(compactors[0].num_features)
    for compactor in compactors:
        if not compactor.excluded:
            remaining = np.logical_or(remaining, compactor.get_metric_vector(cal_deps=True) >= thr)
    return np.sum(remaining)


def cal_conv_flops(in_ch, out_ch, h, w, ks):
    # transposed conv的h和w是输入feature map
    # 普通conv的h和w是输出feature map
    return in_ch*out_ch*h*w*ks*ks*2


def cal_compactor_scores(compactors, norm_scores=None, score_mode="resrep"):
    scores = {}
    # p = {}
    for i, group in enumerate(compactors):
        cnt = 0
        metric_vector = np.zeros_like(group[0].get_metric_vector())
        for compactor in group:
            if not compactor.excluded:
                metric_vector += compactor.get_metric_vector()
                cnt += 1
        # ^2 and abs() are the same for scoring
        metric_vector = abs(metric_vector)
        if norm_scores is None:
            metric_vector /= cnt
        else:
            metric_vector /= norm_scores[i] / 1e7
        for j, v in enumerate(metric_vector):
            scores[(i, j)] = v
            # p[i*1000+j] = float(v)
    # import json
    # json.dump(p, open('scores7.json', 'w'))
    return scores


def already_masked(compactor: CompactorLayer, k):
    return compactor.mask[k] == 0


def cal_cc_flops(deps):
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
    for i in range(1, len(deps['cc_mean_transforms'])):
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


def cc_model_prune(model, ori_deps, thresh, enhanced_resrep, without_y=False, min_channel=1):
    table = model.suc_table
    num_slices = model.num_slices
    already_pruned_suc = set()

    pruned_deps = model.cal_deps(thr=thresh, min_channel=min_channel)
    pruned_flops = cal_cc_flops(pruned_deps)
    ori_flops = cal_cc_flops(ori_deps)
    print('pruned deps: ')
    print(pruned_deps)
    print('keep portion: ', pruned_flops/ori_flops)

    cur_state_dict = model.state_dict()
    save_dict = {}
    for k, v in cur_state_dict.items():
        v = v.detach().cpu()
        save_dict[k] = v
    save_dict = CC_tables.pre_split_before_pruning(save_dict, num_slices, enhanced_resrep, without_y)

    for k, v in table.items():
        if k in model.group_compactor_names:
            # 保证同组的compactor最终的输出维度是一样的
            compactor = (save_dict[k+'.pwc.weight'], list(map(lambda x: save_dict[x+'.pwc.weight'], model.group_compactor_names[k])))
        else:
            compactor = save_dict[k+'.pwc.weight']

        if v['type'] == 'split':
            for k1 in save_dict:
                if v['weight'] not in k1:
                    continue
                weight = save_dict[k1]
                pruned_weight, pruned_bias, survived_ids = fold_conv(weight, None, thresh, compactor, not v['conv'], min_channel)
                save_dict[k1] = pruned_weight
            save_dict[v['bias']] = save_dict[v['bias']][survived_ids.long()]
        else:
            if v['type'] == 'partial':
                weight_name = v['weight'] + '_{}_{}'.format(*v['ch'][0])
                bias_name = v['bias'] + '_{}_{}'.format(*v['ch'][0])
            else:
                weight_name = v['weight']
                bias_name = v['bias']
            weight = save_dict[weight_name]
            bias = save_dict[bias_name]
            pruned_weight, pruned_bias, survived_ids = fold_conv(weight, bias, thresh, compactor, not v['conv'], min_channel)
            save_dict[weight_name] = pruned_weight
            save_dict[bias_name] = pruned_bias

        # change bottleneck
        if v['type'] == 'bottleneck':
            for bk, bv in save_dict.items():
                if 'entropy_bottleneck.' in bk and bv.shape[0] == model.N:
                    save_dict[bk] = bv[survived_ids]

        if pruned_weight.shape == weight.shape:
            # not pruned, no need to change suc_weight
            continue
        
        for i, suc_weight_name in enumerate(v['suc_weight']):
            if v['type'] == 'suc_partial':
                suc_weight_name += '_{}_{}'.format(*v['ch'][i])
            if v['type'] == 'partial':
                suc_weight_name += '_{}_{}'.format(*v['ch'][i+1])
            if suc_weight_name in already_pruned_suc:
                continue
            already_pruned_suc.add(suc_weight_name)
            suc_weight = save_dict[suc_weight_name]
            if v['suc_conv'][i]:
                suc_weight = suc_weight[:, survived_ids.long()]
            else:
                suc_weight = suc_weight[survived_ids.long()]
            save_dict[suc_weight_name] = suc_weight
    save_dict = CC_tables.post_combine_after_pruning(save_dict, num_slices, without_y)

    already_proc = set()
    for k, v in table.items():
        save_dict.pop(k+'.mask')
        save_dict.pop(k+'.pwc.weight')
        if v['weight'] in already_proc:
            continue
        already_proc.add(v['weight'])
        save_dict[v['weight'].replace('.conv.', '.')] = save_dict.pop(v['weight'])
        save_dict[v['bias'].replace('.conv.', '.')] = save_dict.pop(v['bias'])

    save_dict = {k: v for k, v in save_dict.items() if '.fisher' not in k and 'accum_grad' not in k}
    save_dict = {k.replace('module.', '') : v for k, v in save_dict.items()}
    final_dict = {
        'state_dict': save_dict,
        'deps': pruned_deps,
        'keep_portion': pruned_flops/ori_flops
    }
    return final_dict


def cc_model_prune_once(model, thresh, enhanced_resrep, without_y=False, min_channel=1):
    table = model.suc_table
    num_slices = model.num_slices
    already_pruned_suc = set()

    cur_state_dict = model.state_dict()
    save_dict = {}
    for k, v in cur_state_dict.items():
        v = v.detach().cpu()
        save_dict[k] = v
    save_dict = CC_tables.pre_split_before_pruning(save_dict, num_slices, enhanced_resrep, without_y)

    for k, v in table.items():
        if k in model.group_compactor_names:
            # 保证同组的compactor最终的输出维度是一样的
            compactor = (save_dict[k+'.pwc.weight'], list(map(lambda x: save_dict[x+'.pwc.weight'], model.group_compactor_names[k])))
            mask = (save_dict[k+'.pwc.mask'], list(map(lambda x: save_dict[x+'.pwc.mask'], model.group_compactor_names[k])))
        else:
            compactor = save_dict[k+'.pwc.weight']
            mask = save_dict[k+'.pwc.mask']

        if v['type'] == 'split':
            for k1 in save_dict:
                if v['weight'] not in k1:
                    continue
                weight = save_dict[k1]
                pruned_weight, pruned_bias, survived_ids = fold_conv_mask(weight, None, thresh, compactor, mask, not v['conv'], min_channel)
                save_dict[k1] = pruned_weight
            save_dict[v['bias']] = save_dict[v['bias']][survived_ids.long()]
        else:
            if v['type'] == 'partial':
                weight_name = v['weight'] + '_{}_{}'.format(*v['ch'][0])
                bias_name = v['bias'] + '_{}_{}'.format(*v['ch'][0])
            else:
                weight_name = v['weight']
                bias_name = v['bias']
            weight = save_dict[weight_name]
            bias = save_dict[bias_name]
            pruned_weight, pruned_bias, survived_ids = fold_conv_mask(weight, bias, thresh, compactor, mask, not v['conv'], min_channel)
            save_dict[weight_name] = pruned_weight
            save_dict[bias_name] = pruned_bias

        # change bottleneck
        if v['type'] == 'bottleneck':
            for bk, bv in save_dict.items():
                if 'entropy_bottleneck.' in bk and bv.shape[0] == model.N:
                    save_dict[bk] = bv[survived_ids]

        if pruned_weight.shape == weight.shape:
            # not pruned, no need to change suc_weight
            continue
        
        for i, suc_weight_name in enumerate(v['suc_weight']):
            if v['type'] == 'suc_partial':
                suc_weight_name += '_{}_{}'.format(*v['ch'][i])
            if v['type'] == 'partial':
                suc_weight_name += '_{}_{}'.format(*v['ch'][i+1])
            if suc_weight_name in already_pruned_suc:
                continue
            already_pruned_suc.add(suc_weight_name)
            suc_weight = save_dict[suc_weight_name]
            if v['suc_conv'][i]:
                suc_weight = suc_weight[:, survived_ids.long()]
            else:
                suc_weight = suc_weight[survived_ids.long()]
            save_dict[suc_weight_name] = suc_weight
    save_dict = CC_tables.post_combine_after_pruning(save_dict, num_slices, without_y)

    already_proc = set()
    for k, v in table.items():
        save_dict.pop(k+'.mask')
        save_dict.pop(k+'.pwc.weight')
        if v['weight'] in already_proc:
            continue
        already_proc.add(v['weight'])
        save_dict[v['weight'].replace('.conv.', '.')] = save_dict.pop(v['weight'])
        save_dict[v['bias'].replace('.conv.', '.')] = save_dict.pop(v['bias'])

    save_dict = {k: v for k, v in save_dict.items() if '.fisher' not in k and 'accum_grad' not in k}
    save_dict = {k.replace('module.', '') : v for k, v in save_dict.items()}
    return { 'state_dict': save_dict }


def fold_conv(fused_k, fused_b, thresh, compactor_mat, deconv=False, min_channel=1):
    # 只要有一个compactor中的这个channel超过阈值就保留
    if isinstance(compactor_mat, tuple):
        metric_vec = torch.zeros_like(torch.sqrt(torch.sum(compactor_mat[1][0] ** 2, axis=(1, 2, 3))) > thresh)
        for __compactor_mat in compactor_mat[1]:
            metric_vec = torch.logical_or(metric_vec, torch.sqrt(torch.sum(__compactor_mat ** 2, axis=(1, 2, 3))) > thresh)
        filter_ids_higher_thresh = torch.where(metric_vec)[0]
        compactor_mat = compactor_mat[0]
    else:
        metric_vec = torch.sqrt(torch.sum(compactor_mat ** 2, axis=(1, 2, 3)))
        filter_ids_higher_thresh = torch.where(metric_vec > thresh)[0]

    if len(filter_ids_higher_thresh) < min_channel:
        sortd_ids = torch.argsort(metric_vec)
        filter_ids_higher_thresh = sortd_ids[-min_channel:]

    if len(filter_ids_higher_thresh) < len(metric_vec):
        compactor_mat = compactor_mat.index_select(0, filter_ids_higher_thresh)

    if deconv:
        kernel = F.conv2d(fused_k, compactor_mat, padding=(0, 0))
    else:
        kernel = F.conv2d(fused_k.permute(1, 0, 2, 3), compactor_mat,
                        padding=(0, 0)).permute(1, 0, 2, 3)
    Dprime = compactor_mat.shape[0]

    if fused_b is not None:
        bias = torch.zeros(Dprime)
        for i in range(Dprime):
            bias[i] = fused_b.dot(compactor_mat[i,:,0,0])
    else:
        bias = None

    return kernel, bias, filter_ids_higher_thresh

def fold_conv_mask(fused_k, fused_b, thresh, compactor_mat, mask, deconv=False, min_channel=1):
    # 只要有一个compactor中的这个channel超过阈值就保留
    if isinstance(compactor_mat, tuple):
        metric_vec = torch.zeros_like(torch.sqrt(torch.sum(compactor_mat[1][0] ** 2, axis=(1, 2, 3))) > thresh)
        for __compactor_mat in compactor_mat[1]:
            metric_vec = torch.logical_or(metric_vec, torch.sqrt(torch.sum(__compactor_mat ** 2, axis=(1, 2, 3))) > thresh)
        filter_ids_higher_thresh = torch.where(metric_vec)[0]
        compactor_mat = compactor_mat[0]
    else:
        metric_vec = torch.sqrt(torch.sum(compactor_mat ** 2, axis=(1, 2, 3)))
        filter_ids_higher_thresh = torch.where(metric_vec > thresh)[0]

    if isinstance(compactor_mat, tuple):
        final_mask = torch.zeros_like(mask[1][0])
        for __compactor_mat in compactor_mat[1]:
            final_mask = torch.logical_or(final_mask, torch.sqrt(torch.sum(__compactor_mat ** 2, axis=(1, 2, 3))) > thresh)
        filter_ids_higher_thresh = torch.where(final_mask)[0]
        compactor_mat = compactor_mat[0]
    else:
        final_mask = torch.sqrt(torch.sum(compactor_mat ** 2, axis=(1, 2, 3)))
        filter_ids_higher_thresh = torch.where(final_mask > thresh)[0]

    if len(filter_ids_higher_thresh) < min_channel:
        sortd_ids = torch.argsort(metric_vec)
        filter_ids_higher_thresh = sortd_ids[-min_channel:]

    if len(filter_ids_higher_thresh) < len(metric_vec):
        compactor_mat = compactor_mat.index_select(0, filter_ids_higher_thresh)

    if deconv:
        kernel = F.conv2d(fused_k, compactor_mat, padding=(0, 0))
    else:
        kernel = F.conv2d(fused_k.permute(1, 0, 2, 3), compactor_mat,
                        padding=(0, 0)).permute(1, 0, 2, 3)
    Dprime = compactor_mat.shape[0]

    if fused_b is not None:
        bias = torch.zeros(Dprime)
        for i in range(Dprime):
            bias[i] = fused_b.dot(compactor_mat[i,:,0,0])
    else:
        bias = None

    return kernel, bias, filter_ids_higher_thresh