# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-21

import torch.nn as nn
import torch.nn.functional as F
from . import register_head


@register_head('CTCHead')
class CTCHead(nn.Module):
    def __init__(self, in_channels, out_channels, fc_decay=0.0004, **kwargs):
        super(CTCHead, self).__init__()
        # weight_attr, bias_attr = get_para_bias_attr(
        #     l2_decay=fc_decay, k=in_channels, name='ctc_fc')
        self.fc = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels

    def forward(self, x, labels=None):
        predicts = self.fc(x)
        if not self.training:
            predicts = F.softmax(predicts, dim=2)
        return predicts
