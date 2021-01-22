# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-20


import torch.nn as nn
from .custom_module import Conv2dBNLayer, SeModule


class ResidualBlock(nn.Module):
    """
    MobileV3子结构, 对应论文图3，图4
    """
    def __init__(self, in_filters, middle_filters, out_filters, stride, filter_size, act=None, use_se=False):
        super(ResidualBlock, self).__init__()

        self.in_filters = in_filters
        self.out_filters = out_filters
        self.use_se = use_se
        self.short_cut = self.in_filters == self.out_filters and stride == 1

        # expansion to a much higher-dimesional dimensionl space
        self.expand_conv = Conv2dBNLayer(in_filters, middle_filters, 1, act,
                                         stride=1,
                                         padding=0,
                                         bias=False)

        # d-wise
        self.bottleneck_conv = Conv2dBNLayer(middle_filters, middle_filters, filter_size, act,
                                             stride=stride,
                                             padding=int((filter_size - 1) // 2),
                                             bias=False,
                                             groups=middle_filters)

        self.mid_se = SeModule(middle_filters) if use_se else None

        # projection to the output
        self.linear_conv = Conv2dBNLayer(middle_filters, out_filters, 1, None,
                                         if_act=False,
                                         stride=1,
                                         padding=0,
                                         bias=False)

    def forward(self, x):
        out = self.expand_conv(x)
        out = self.bottleneck_conv(out)

        if self.use_se:
            out = self.mid_se(out)

        out = self.linear_conv(out)

        if self.short_cut:
            return out + x
        else:
            return out
