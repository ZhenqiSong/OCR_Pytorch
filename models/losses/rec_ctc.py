# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-02-01

import torch
from . import register_loss
import torch.nn as nn


@register_loss('CTCLoss')
class CTCLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')

    def forward(self, predicts: torch.Tensor, batch):
        predicts = predicts.permute((1, 0, 2))
        N, B, _ = predicts.shape
        preds_lengths = torch.tensor([N]*B, dtype=torch.int64)
        labels = batch[1].astype("int32")
        label_length = batch[2].astype('int64')
        loss = self.loss_func(predicts, labels, preds_lengths, label_length)
        loss = loss.mean()
        return {"loss": loss}