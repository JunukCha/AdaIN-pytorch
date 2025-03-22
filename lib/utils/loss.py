import torch.nn as nn
from lib.utils.adain import calc_mean_std

mse_loss = nn.MSELoss()

def calc_content_loss(output, target):
    assert (output.size() == target.size())
    return mse_loss(output, target)

def calc_style_loss(output, target):
    assert (output.size() == target.size())
    output_mean, output_std = calc_mean_std(output)
    target_mean, target_std = calc_mean_std(target)
    return mse_loss(output_mean, target_mean) + \
           mse_loss(output_std, target_std)