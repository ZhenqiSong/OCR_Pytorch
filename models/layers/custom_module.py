# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2020/11/16

from typing import Union
import torch.nn as nn
import torch.nn.functional as F
from .activations import activations, f_activations


class Conv2dBNLayer(nn.Module):
    """
    自定义模块，依次执行Conv2d, Bn2d, 激活函数
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, tuple], act: Union[str, None] = '',
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', if_act=True,
                 eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, inplace=True):
        super(Conv2dBNLayer, self).__init__()
        self.is_act = if_act

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias,
                              padding_mode=padding_mode)

        self.bn = nn.BatchNorm2d(num_features=out_channels,
                                 eps=eps,
                                 momentum=momentum,
                                 affine=affine,
                                 track_running_stats=track_running_stats)

        if act is not None:
            self.act = activations[act](inplace=inplace)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.act(x) if self.is_act else x


class SeModule(nn.Module):
    """
    Squeeze-and-Excite
    """
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_size, in_size//reduction, 1, 1, 0)
        self.conv2 = nn.Conv2d(in_size//reduction, in_size, 1, 1, 0)

    def forward(self, inputs):
        x = self.avg_pool(inputs)
        x = F.relu(self.conv1(x))
        x = f_activations['hsigmoid'](self.conv2(x))
        return inputs * x