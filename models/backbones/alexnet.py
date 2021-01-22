# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2020/12/3

import torch
import torch.nn as nn

__all__ = ['AlexNet']


class AlexNet(nn.Module):
    def __init__(self, class_nums):
        super(AlexNet, self).__init__()
