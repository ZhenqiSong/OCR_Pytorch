# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-20
from abc import ABC

import torch
import torch.nn as nn
from ..layers import Conv2dBNLayer, ResidualBlock
from . import register_rec_backbone, register_det_backbone


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@register_rec_backbone('MobileNetV3')
class RecMobileNetV3(nn.Module):
    def __init__(self, in_channels: int = 3,
                 model_name: str = 'small',
                 scale: float = 0.5,
                 large_stride=None,
                 small_stride=None,
                 **kwargs):
        super(RecMobileNetV3, self).__init__()

        if small_stride is None:
            small_stride = [2, 2, 2, 2]
        if large_stride is None:
            large_stride = [1, 2, 2, 2]

        assert isinstance(large_stride, list), "large_stride type must " \
                                               "be list but got {}".format(type(large_stride))
        assert isinstance(small_stride, list), "small_stride type must " \
                                               "be list but got {}".format(type(small_stride))
        assert len(large_stride) == 4, "large_stride length must be " \
                                       "4 but got {}".format(len(large_stride))
        assert len(small_stride) == 4, "small_stride length must be " \
                                       "4 but got {}".format(len(small_stride))

        if model_name == "large":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', large_stride[0]],
                [3, 64, 24, False, 'relu', (large_stride[1], 1)],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', (large_stride[2], 1)],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hard_swish', 1],
                [3, 200, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 480, 112, True, 'hard_swish', 1],
                [3, 672, 112, True, 'hard_swish', 1],
                [5, 672, 160, True, 'hard_swish', (large_stride[3], 1)],
                [5, 960, 160, True, 'hard_swish', 1],
                [5, 960, 160, True, 'hard_swish', 1],
            ]
            cls_ch_squeeze = 960
        elif model_name == "small":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', (small_stride[0], 1)],
                [3, 72, 24, False, 'relu', (small_stride[1], 1)],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hard_swish', (small_stride[2], 1)],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 120, 48, True, 'hard_swish', 1],
                [5, 144, 48, True, 'hard_swish', 1],
                [5, 288, 96, True, 'hard_swish', (small_stride[3], 1)],
                [5, 576, 96, True, 'hard_swish', 1],
                [5, 576, 96, True, 'hard_swish', 1],
            ]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert scale in supported_scale, \
            "supported scales are {} but input scale is {}".format(supported_scale, scale)

        inplanes = 16
        self.conv1 = Conv2dBNLayer(in_channels=in_channels,
                                   out_channels=make_divisible(inplanes * scale),
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   groups=1,
                                   bias=False,
                                   if_act=True,
                                   act='hard_swish')

        self.blocks = nn.Sequential()
        inplanes = make_divisible(inplanes * scale)
        for i, (k, exp, c, se, nl, s) in enumerate(cfg):
            self.blocks.add_module(
                name=f'{i}',
                module=ResidualBlock(in_filters=inplanes,
                                     middle_filters=make_divisible(scale * exp),
                                     out_filters=make_divisible(scale * c),
                                     filter_size=k,
                                     stride=s,
                                     use_se=se,
                                     act=nl))
            inplanes = make_divisible(scale * c)

        self.conv2 = Conv2dBNLayer(in_channels=inplanes,
                                   out_channels=make_divisible(scale*cls_ch_squeeze),
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   groups=1,
                                   if_act=True,
                                   bias=False,
                                   act='hard_swish')

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = make_divisible(scale*cls_ch_squeeze)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x


@register_det_backbone("MobileNetV3")
class DetMobileV3(nn.Module):
    def __init__(self, in_channels=3, model_name='large', scale=0.5, disable_se=False, **kwargs):
        super(DetMobileV3, self).__init__()

        self.disable_se = disable_se
        if model_name == "large":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', 1],
                [3, 64, 24, False, 'relu', 2],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', 2],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hard_swish', 2],
                [3, 200, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 480, 112, True, 'hard_swish', 1],
                [3, 672, 112, True, 'hard_swish', 1],
                [5, 672, 160, True, 'hard_swish', 2],
                [5, 960, 160, True, 'hard_swish', 1],
                [5, 960, 160, True, 'hard_swish', 1],
            ]
            cls_ch_squeeze = 960
        elif model_name == "small":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', 2],
                [3, 72, 24, False, 'relu', 2],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hard_swish', 2],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 120, 48, True, 'hard_swish', 1],
                [5, 144, 48, True, 'hard_swish', 1],
                [5, 288, 96, True, 'hard_swish', 2],
                [5, 576, 96, True, 'hard_swish', 1],
                [5, 576, 96, True, 'hard_swish', 1],
            ]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert scale in supported_scale, \
            "supported scale are {} but input scale is {}".format(supported_scale, scale)
        inplanes = 16

        self.conv = Conv2dBNLayer(in_channels=in_channels,
                                  out_channels=make_divisible(inplanes*scale),
                                  kernel_size=3,
                                  stride=2,
                                  padding=1,
                                  bias=False,
                                  act='hard_swish')
        self.stages = []
        self.out_channels = []
        block_list = []
        self.blocks = nn.Sequential()
        inplanes = make_divisible(inplanes * scale)
        for i, (k, exp, c, se, nl, s) in enumerate(cfg):
            se = se and not self.disable_se

            if s == 2 and i > 2:
                self.out_channels.append(inplanes)
                self.stages.append(nn.Sequential(*block_list))
                block_list = []

            block_list.append(
                ResidualBlock(in_filters=inplanes,
                              middle_filters=make_divisible(scale * exp),
                              out_filters=make_divisible(scale * c),
                              filter_size=k,
                              stride=s,
                              use_se=se,
                              act=nl))
            inplanes = make_divisible(scale * c)

        block_list.append(
            Conv2dBNLayer(in_channels=inplanes,
                          out_channels=make_divisible(scale * cls_ch_squeeze),
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          groups=1,
                          if_act=True,
                          bias=False,
                          act='hard_swish'))
        self.stages.append(nn.Sequential(*block_list))
        self.out_channels.append(make_divisible(scale * cls_ch_squeeze))
        for i, stage in enumerate(self.stages):
            self.add_module(name=f'stage{i}', module=stage)

    def forward(self, x):
        x = self.conv(x)
        out_list = []
        for stage in self.stages:
            x = stage(x)
            out_list.append(x)
        return out_list
