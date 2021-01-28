
import os

import numpy as np
from typing import Union
import math
import cv2 as cv
import sys


class KeepKeys(object):
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list


class DecodeImage(object):
    def __init__(self, mode='BGR', channel_first=False, **kwargs):
        self.img_mode = mode
        self.channel_first = channel_first

    def __call__(self, data):
        """
        Args: data['image'] -> Union[str, np.array]
        """
        img = data['image']
        if isinstance(img, str):
            if not os.path.exists:
                raise FileNotFoundError("不存在的文件：{} ".format(img))
            img = cv.imread(img)

        if img is None:
            return None

        # 使用opencv读取的默认就是BGR，因此默认mode应该为None，不用做任何转化
        if self.img_mode == "GRAY":
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        elif self.img_mode == "RGB":
            assert img.shape[2] == 3, '无效图像: image[%s]' % (img.shape)
            img = img[:, :, ::-1]

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        data['image'] = img
        return data


class RecResizeImg(object):
    def __init__(self, image_shape, infer_mode=False, character_type='ch', **kwargs):
        self.image_shape = image_shape
        self.infer_mode = infer_mode
        self.character_type = character_type

    def __call__(self, data):
        img = data['image']
        if self.infer_mode and self.character_type == 'ch':
            norm_img = resize_norm_img_chinese(img, self.image_shape)
        else:
            norm_img = resize_norm_img(img, self.image_shape)
        data['image'] = norm_img
        return data


def resize_norm_img_chinese(img, image_shape):
    imgC, imgH, imgW = image_shape
    max_wh_ratio = 0
    h, w = img.shape[:2]
    ratio = w * 1.0 / h
    max_wh_ratio = max(max_wh_ratio, ratio)
    imgW = int(32*max_wh_ratio)
    if math.ceil(imgH * ratio) > imgH:
        resize_w = imgW
    else:
        resize_w = int(math.ceil(imgH * ratio))
    resized_image = cv.resize(img, (resize_w, imgH))
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resize_w] = resized_image
    return padding_im


def resize_norm_img(img, image_shape):
    imgC, imgH, imgW = image_shape
    h = img.shape[0]
    w = img.shape[1]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


class DetResizeForTest():
    """
    用于文本检测时，对测试数据进行图像缩放
    """
    def __init__(self, **kwargs):
        self.resize_type = 0
        if kwargs.get('image_shape', None) is not None:
            self.image_shape = kwargs['image_shape']
            self.resize_type = 1
        elif kwargs.get('limit_side_len', None) is not None:
            self.limit_side_len = kwargs['limit_side_len']
            self.limit_type = kwargs.get('limit_type', 'min')
        elif kwargs.get('resize_long', None) is not None:
            self.resize_type = 2
            self.resize_long = kwargs.get('resize_long', 960)
        else:
            self.limit_side_len = 736
            self.limit_type = 'min'

    def __call__(self, data):
        img = data['image']
        src_h, src_w = img.shape[:2]

        if self.resize_type == 0:
            img, [ratio_h, ratio_w] = self.resize_image_type0(img)
        elif self.resize_type == 2:
            img, [ratio_h, ratio_w] = self.resize_image_type2(img)
        else:
            img, [ratio_h, ratio_w] = self.resize_image_type1(img)
        data['image'] = img
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def resize_image_type0(self, img):
        """
        将图像缩放为32的倍数
        :param img:
        :return:
        """
        limit_side_len = self.limit_side_len
        h, w = img.shape[:2]

        ratio = 1.
        if self.limit_type == 'max' and max(h, w) > limit_side_len:
            ratio = float(limit_side_len) / max(h, w)
        elif min(h, w) < limit_side_len:
            ratio = float(limit_side_len) / min(h, w)

        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = int(round(resize_h / 32) * 32)
        resize_w = int(round(resize_w / 32) * 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            img = cv.resize(img, (int(resize_w), int(resize_h)))
        except Exception:
            print(img.shape, resize_w, resize_h)
            sys.exit(0)

        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return img, [ratio_h, ratio_w]

    def resize_image_type1(self, img):
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = img.shape[:2]
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv.resize(img, (int(resize_w), int(resize_h)))
        return img, [ratio_h, ratio_w]

    def resize_image_type2(self, img):
        h, w, _ = img.shape

        resize_w, resize_h = w, h

        ratio = float(self.resize_long) / max(w, h)
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_h + max_stride - 1) // max_stride * max_stride

        img = cv.resize(img, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return img, (ratio_h, ratio_w)


class NormalizeImage(object):
    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)

        self.scale = np.float(scale if scale is not None else 1.0 / 255)
        mean = mean if mean is not None else [0.485, 0.465, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        img = data['image']

        data['image'] = (img.astype('float32') * self.scale - self.mean) / self.std
        return data


class ToCHWImage(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        data['image'] = data['image'].transpose((2, 0, 1))
        return data