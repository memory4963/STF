import numpy as np
from resrep.rr_builder import CompactorLayer, RRBuilder

def get_remain(vec: CompactorLayer, thr):
    return np.sum(vec.get_metric_vector() >= thr)

def cal_conv_flops(in_ch, out_ch, h, w, ks):
    # transposed conv的h和w是输入feature map
    # 普通conv的h和w是输出feature map
    return in_ch*out_ch*h*w*ks*ks*2

def cal_compactor_scores(compactors):
    scores = {}
    for i, group in enumerate(compactors):
        metric_vector = group[0].get_metric_vector()
        for compactor in group[1:]:
            metric_vector += compactor.get_metric_vector()
        metric_vector /= len(group)
        for j, v in enumerate(metric_vector):
            scores[(i, j)] = v
    return scores

def already_masked(compactor: CompactorLayer, k):
    return compactor.mask[k] == 0