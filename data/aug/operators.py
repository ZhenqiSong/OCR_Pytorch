
import os

import numpy as np
from typing import Union
import math
import cv2 as cv


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