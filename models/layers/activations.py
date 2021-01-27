# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2020/11/16

import torch
import torch.nn as nn
import torch.nn.functional as F


def hswish(x, inplace=True):
    out = x * F.relu6(x + 3, inplace=inplace) / 6
    return out


def hsigmoid(x, slope=0.2, offset=0.5):
    """
    pytorch中也自带有该函数，但是参数和paddle的有差别，公式为 x / 6 + 0.5， paddle里的是 x / 5 + 0.5
    :param x: 输入
    :param slope:
    :param offset:
    :return:
    """
    out = x * slope + offset
    torch.clamp_max_(out, 1)
    torch.clamp_min_(out, 0)
    return out


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hsigmoid(x, self.inplace)


class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hswish(x, self.inplace)


activations = {
    'relu': nn.ReLU,
    'hard_swish': HSwish,
    'hsigmoid': HSigmoid
}

f_activations = {
    'relu': F.relu,
    'hard_swish': hswish,
    'hsigmoid': hsigmoid
}
