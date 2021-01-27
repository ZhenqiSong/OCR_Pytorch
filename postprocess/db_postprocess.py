# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-27


import numpy as np
import torch
import cv2 as cv

from . import register_post


@register_post('DBPostProcess')
class DBPostProcess(object):

    def __init__(self,
                 thresh=0.3,
                 box_thresh=0.7,
                 max_candidates=1000,
                 unclip_ratio=2.0,
                 use_dilation=False,
                 **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.dilation_kernel = None if not use_dilation else np.array([[1, 1], [1, 1]])

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict['maps']
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().detach().numpy()
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv.dilate(np.array(segmentation[batch_index]).astype(np.uint8),
                                 self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            boxes, scores = self.boxes