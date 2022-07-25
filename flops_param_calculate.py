from compressai.zoo import models
from compressai.zoo import load_state_dict
import torch
from thop import profile, clever_format
import time
import sys

# state_dict = load_state_dict(torch.load('output/rr/gmm_lamda0.0067_lasso1e-9_target0.3_372000_lessCH_update.pth.tar'))
# model = architectures['cheng2020-jinming-rr'].from_state_dict(state_dict['state_dict'], state_dict['deps'], infer=True)
# state_dict = load_state_dict(torch.load('output/finetune_rr/hyper_lambda0.0018_lasso1e-9_target0.3/372000/checkpoint_best_loss_update.pth.tar'))

# state_dict = load_state_dict(torch.load('output/hit_hyper_lambda0.0018_gamma0.01/pruned_model.pth'))
# model = architectures['bmshj2018-hyperprior-rr'].from_state_dict(state_dict['state_dict'], state_dict['deps'], infer=True)

# state_dict = load_state_dict(torch.load('output/finetune_rr/main_hyper_lambda0.0018_lasso8e-10_target0.3/556000/checkpoint_best_loss_update.pth.tar'))
# state_dict = load_state_dict(torch.load('output/gd_hyper_0.0018_hyper_target0.5/pruned_model_update.pth.tar'))
# model = architectures['bmshj2018-hyperprior-rr'].from_state_dict(state_dict['state_dict'], state_dict['deps'], infer=True)

# state_dict = load_state_dict(torch.load('output/gd_hyper_whole_0.0018_t0.5/pruned_model_update.pth.tar'))
# model = architectures['bmshj2018-hyperprior-whole-rr'].from_state_dict(state_dict['state_dict'], state_dict['deps'], infer=True)
# state_dict = load_state_dict(torch.load('output/finetune_gd/gd_hyper_0.0018_t0.8/checkpoint_best_loss_update.pth.tar'))
# model = architectures['bmshj2018-hyperprior-main-rr'].from_state_dict(state_dict['state_dict'], state_dict['deps'], infer=True)


# state_dict = load_state_dict(torch.load('output/finetune_rr/cm_hyper_lambda0.0130_lasso1e-9_target0.3/556000/checkpoint_best_loss_update.pth.tar'))
# model = architectures['cheng2020-jinming-rr-cm'].from_state_dict(state_dict['state_dict'], state_dict['deps'], infer=True)

# model = models['bmshj2018-hyperprior'](quality=6)
# model = models['bmshj2018-hyperprior-effi'](quality=4)
# model = models['mbt2018-effi'](quality=1)
# small = [96, 128, 160, 192, 96, 128]
medium = [128, 192, 256, 320, 192, 192]
# large = [160, 256, 352, 448, 192, 256]
model = models['tbc'](channels=medium, num_slices=1)
# model = models['cheng2020-jinming-effi'](quality=3)

ipt = torch.rand([1,3,256,256])
# ipt = torch.rand([1,3,256,256]).cuda()
# ipt = torch.rand([1,192,16,16])
flops, params = profile(model, (ipt,))
flops, params = clever_format([flops, params], "%.3f")
print(flops, params)

# print((5.969-float(params[:-1]))/4.969)

# from torchsummary import summary
# summary(model.cuda(), input_size=(3,256,256))

# from ptflops import get_model_complexity_info
# ops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=True, verbose=True)

# model.eval()
# with torch.no_grad():
#     o = model(ipt)
# print(1)

# input_names = ['image']
# output_names = ['output']
# torch.onnx.export(model, ipt, "test.onnx", verbose=True,
#     input_names=input_names, output_names=output_names, opset_version=11)