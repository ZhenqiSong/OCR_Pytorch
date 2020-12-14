# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2020/11/16

from models.layers import Conv2dBNLayer

from typing import List
import torch.nn as nn
import torchsummary as summary


class SeModule(nn.Module):
    """
    Squeeze-and-Excite
    """
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dBNLayer(in_size, in_size//reduction, 1, 'relu', stride=1, padding=0, bias=False),
            Conv2dBNLayer(in_size//reduction, in_size, 1, 'hsigmoid', stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        return x * self.se(x)


class ResidualBlock(nn.Module):
    """
    MobileV3子结构, 对应论文图3，图4
    """
    def __init__(self, in_filters, middle_filters, out_filters, stride, filter_size, act=None, use_se=False):
        super(ResidualBlock, self).__init__()

        self.stride = 1
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.use_se = use_se

        # expansion to a much higher-dimesional dimensionl space
        self.conv0 = Conv2dBNLayer(in_filters, middle_filters, 1, act, stride=1, padding=0, bias=False)

        # d-wise
        self.conv1 = Conv2dBNLayer(middle_filters, middle_filters, filter_size, act,
                                   stride=stride,
                                   padding=int((filter_size - 1) // 2),
                                   bias=False,
                                   groups=middle_filters)

        self.se = SeModule(middle_filters) if use_se else None

        # projection to the output
        self.conv2 = Conv2dBNLayer(middle_filters, out_filters, 1, act, is_act=False, stride=1, padding=0, bias=False)

    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1(out)

        if self.use_se:
            out = self.se(out)

        out = self.conv2(out)

        if self.in_filters == self.out_filters and self.stride == 1:
            return out + x
        else:
            return out


class BaseMobileNetV3(nn.Module):
    def __init__(self, cfg: List, scale=1.0, cls_ch_squeeze=576, cls_ch_expand=1280, is_gray=True):
        """
        MobileV3
        :param cfg: 网络配置，分为small和large
        :param scale: 滤波器的不同缩放尺寸
        :param cls_ch_squeeze: 压缩通道数
        :param cls_ch_expand: 扩展通道数
        :param is_gray: 输入是否为灰度图，默认为灰度
        """
        super(BaseMobileNetV3, self).__init__()

        self.layers = cfg
        self.cls_ch_squeeze = cls_ch_squeeze
        self.cls_ch_expand = cls_ch_expand
        self.scale = scale
        self.inplanes = 16
        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert scale in supported_scale, "supported scales are {} but input scale is {}".format(supported_scale,
                                                                                                self.scale)

        self.layer1 = Conv2dBNLayer(in_channels=1 if is_gray else 3,
                                    out_channels=self.make_divisible(self.inplanes * self.scale),
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    act='hswish',
                                    bias=False)

        in_filters = self.make_divisible(self.inplanes * scale)
        self.inplanes = self.make_divisible(self.inplanes * scale)
        self.blocks = nn.Sequential()

        for i, layer_cfg in enumerate(self.layers):
            self.blocks.add_module(f'block{i}',
                                   ResidualBlock(in_filters=in_filters,
                                                 middle_filters=self.make_divisible(self.scale * layer_cfg[1]),
                                                 out_filters=self.make_divisible(self.scale * layer_cfg[2]),
                                                 act=layer_cfg[4],
                                                 stride=layer_cfg[5],
                                                 filter_size=layer_cfg[0],
                                                 use_se=layer_cfg[3]))
            in_filters = self.make_divisible(scale * layer_cfg[2])

    def forward(self, x):
        x = self.layer1(x)
        x = self.blocks(x)
        return x

    def make_divisible(self, v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v


def rec_mobile_v3_large(*args, **kwargs):
    """
    mobile V3 large 版

    :param args:
    :param kwargs:
        stride : 用于指定池化层
        is_gray: 输入是否为灰度图
    :return:
    """
    large_stride = kwargs.get("stride", [2, 2, 2, 2])
    is_gray = kwargs.get("is_gray", True)
    scale = kwargs.get('scale', 1)

    assert isinstance(large_stride, list), "large_stride type must " \
                                           "be list but got {}".format(type(large_stride))
    assert len(large_stride) == 4, "large_stride length must be " \
                                   "4 but got {}".format(len(large_stride))
    layers_config = [
        # k, exp, c,  se,     nl,  s,
        [3, 16, 16, False, 'relu', 1],
        [3, 64, 24, False, 'relu', large_stride[0]],
        [3, 72, 24, False, 'relu', 1],
        [5, 72, 40, True, 'relu', large_stride[1]],
        [5, 120, 40, True, 'relu', 1],
        [5, 120, 40, True, 'relu', 1],
        [3, 240, 80, False, 'hswish', large_stride[2]],
        [3, 200, 80, False, 'hswish', 1],
        [3, 184, 80, False, 'hswish', 1],
        [3, 184, 80, False, 'hswish', 1],
        [3, 480, 112, True, 'hswish', 1],
        [3, 672, 112, True, 'hswish', 1],
        [5, 672, 160, True, 'hswish', large_stride[3]],
        [5, 960, 160, True, 'hswish', 1],
        [5, 960, 160, True, 'hswish', 1],
    ]

    return BaseMobileNetV3(cfg=layers_config, scale=scale, is_gray=is_gray)


if __name__ == '__main__':
    model = rec_mobile_v3_large()
    print(model)
    import torch, torchsummary

    x = torch.rand((1, 1, 224, 224))
    torchsummary.summary(model, (1, 224, 224), device='cpu')
