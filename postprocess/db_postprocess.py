# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-27


import numpy as np
import torch
import cv2 as cv
from shapely.geometry import Polygon
import pyclipper

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
            # 膨胀操作
            if self.dilation_kernel is not None:
                mask = cv.dilate(np.array(segmentation[batch_index]).astype(np.uint8),
                                 self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask,
                                                   src_w, src_h)
            boxes_batch.append({'points': boxes})
        return boxes_batch

    def boxes_from_bitmap(self, pred, bit_map, dst_width, dst_height):
        """
        从位图中解析除文本框
        :param pred:
        :param bit_map:
        :param dst_width:
        :param dst_height:
        :return:
        """
        height, width = bit_map.shape

        outs = cv.findContours((bit_map*255).astype(np.uint8),
                               cv.RETR_LIST,
                               cv.CHAIN_APPROX_SIMPLE)

        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, size = self.get_mini_box(contour)
            if size < self.min_size:
                continue

            points = np.array(points)
            score = self.get_box_score(pred, points.reshape(-1, 2))
            if score < self.box_thresh:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, size = self.get_mini_box(box)
            if size < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dst_width), 0, dst_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dst_height), 0, dst_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores

    def get_mini_box(self, contour):
        """
        根据轮廓获取最小外接矩形，并根据 从左上开始的，顺时针顺序返回四个坐标点
        :param contour: 轮廓
        :return:
        """
        bounding_boxes = cv.minAreaRect(contour)
        # 首先根据四个点的x坐标排序
        points = sorted(list(cv.boxPoints(bounding_boxes)), key=lambda x: x[0])

        # 根据y坐标调整顺序
        index_1, index_2, index_3, index_4 = 0, 2, 3, 1
        if points[1][1] < points[0][1]:
            index_1, index_4 = 1, 0

        if points[3][1] < points[2][1]:
            index_2, index_3 = 3, 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_boxes[1])

    def get_box_score(self, bitmap, _box):
        """
        获取box的得分，根据框的位置生成掩码，取得结果位图上对应区域的值求平均
        :param bitmap: 结果位图
        :param _box: 当前的box
        :return:
        """
        h, w = bitmap.shape[:2]
        box = _box.copy()

        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w-1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w-1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h-1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h-1)

        mask = np.zeros((ymax-ymin+1, xmax-xmin+1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv.mean(bitmap[ymin:ymax+1, xmin:xmax + 1], mask)[0]

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded