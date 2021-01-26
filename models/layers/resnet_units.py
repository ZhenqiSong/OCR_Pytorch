# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-26

import torch.nn as nn
from torch.nn import functional as F
from . import Conv2dBNLayer


class BottleneckBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False):
        super(BottleneckBlock, self).__init__()

        # 1*1卷积
        self.conv0 = Conv2dBNLayer(in_channels=in_channels,
                                   out_channels=out_channels,
                                   bias=False,
                                   kernel_size=1,
                                   stride=1,
                                   act='relu')
        # 常规卷积
        self.conv1 = Conv2dBNLayer(in_channels=out_channels,
                                   out_channels=out_channels,
                                   bias=False,
                                   kernel_size=3,
                                   padding=1,
                                   stride=stride,
                                   act='relu')

        self.conv2 = Conv2dBNLayer(in_channels=out_channels,
                                   out_channels=out_channels*4,
                                   if_act=False,
                                   kernel_size=1,
                                   stride=1)

        if not shortcut:
            self.is_vd_mode = not if_first and stride[0] != 1
            if self.is_vd_mode:
                self.pool2d_avg = nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0, ceil_mode=True)
            self.short = Conv2dBNLayer(in_channels=in_channels,
                                       out_channels=out_channels*4,
                                       kernel_size=1,
                                       stride=1 if self.is_vd_mode else stride,
                                       bias=False,
                                       if_act=False,
                                       )
        self.shortcut = shortcut
    
    def forward(self, inputs):
        y = self.conv0(inputs)

        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            if self.is_vd_mode:
                inputs = self.pool2d_avg(inputs)
            short = self.short(inputs)

        y = short + conv2
        y = F.relu(y)

        return y


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 name=None):
        super(BasicBlock, self).__init__()
        self.stride=stride

        self.conv0 = Conv2dBNLayer(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   stride=stride,
                                   act='relu',
                                   bias=False,
                                   padding=1)

        self.conv1 = Conv2dBNLayer(in_channels=out_channels,
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   if_act=False,
                                   padding=1,
                                   bias=False)

        if not shortcut:
            self.is_vd_mode = not if_first and stride[0] != 1
            if self.is_vd_mode:
                self.pool_2d = nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0, ceil_mode=True)
            self.short = Conv2dBNLayer(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=1,
                                       stride=1 if self.is_vd_mode else stride,
                                       bias=False,
                                       if_act=False)
        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            if self.is_vd_mode:
                inputs = self.pool_2d(inputs)
            short = self.short(inputs)
        y = conv1 + short
        y = F.relu(y)
        return y