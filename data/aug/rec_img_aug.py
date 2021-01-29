# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-29

import math
import cv2 as cv
import numpy as np
import random
from .warp_mls import WarpMLS


class RecAug(object):
    def __init__(self, use_tia=True, aug_prob=0.4, **kwargs):
        self.use_tia = use_tia
        self.prob = aug_prob

    def __call__(self, data):
        img = data['image']
        img = self.warp(img, 10)
        data['image'] = img
        return data

    def warp(self, img, ang):
        h, w = img.shape[:2]
        self.make(w, h, ang)
        new_img = img

        if self.distort:
            if random.random() <= self.prob and h >= 20 and w >= 20:
                new_img = tia_distort(new_img, random.randint(3, 6))

        if self.stretch:
            if random.random() <= self.prob and h >= 20 and w >= 20:
                new_img = tia_stretch(new_img, random.randint(3, 6))

        if self.perspective:
            if random.random() <= self.prob:
                new_img = tia_perspective(new_img)

        if self.crop:
            if random.random() <= self.prob and h >= 20 and w >= 20:
                new_img = get_crop(new_img)

        if self.blur:
            new_h, new_w = new_img.shape[:2]
            if random.random() <= self.prob and new_h > 10 and new_w > 10:
                new_img = cv.GaussianBlur(new_img, (5, 5), 1)

        if self.color:
            if random.random() <= self.prob:
                new_img = cvt_color(new_img)

        if self.jitter:
            new_img = jitter(new_img)

        if self.noise:
            if random.random() <= self.prob:
                new_img = add_noise(new_img)

        if self.reverse:
            if random.random() <= self.prob:
                new_img = 255 - new_img

        return new_img


    def make(self, w, h, ang):
        self.anglex = random.random() * 5 * do()
        self.angley = random.random() * 5 * do()
        self.anglez = -1 * random.random() * int(ang) * do()
        self.fov = 42
        self.r = 0
        self.shearx = 0
        self.sheary = 0
        self.borderMode = cv.BORDER_REPLICATE

        self.perspective = self.use_tia
        self.stretch = self.use_tia
        self.distort = self.use_tia

        self.crop = True
        self.affine = False
        self.reverse = True
        self.noise = True
        self.jitter = True
        self.blur = True
        self.color = True


def do():
    return 1 if random.random() > 0.5000001 else -1


def tia_distort(img, segment=4):
    h, w = img.shape[:2]

    cut = w // segment
    thresh = cut // 3

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([w, 0])
    src_pts.append([w, h])
    src_pts.append([0, h])

    dst_pts.append([np.random.randint(thresh), np.random.randint(thresh)])
    dst_pts.append([w - np.random.randint(thresh), np.random.randint(thresh)])
    dst_pts.append([w - np.random.randint(thresh), h - np.random.randint(thresh)])
    dst_pts.append([np.random.randint(thresh), h - np.random.randint(thresh)])

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        src_pts.append([cut_idx * cut, 0])
        src_pts.append([cut_idx * cut, h])
        dst_pts.append([cut_idx * cut + np.random.randint(thresh) - half_thresh,
                        np.random.randint(thresh) - half_thresh])
        dst_pts.append([cut_idx * cut + np.random.randint(thresh) - half_thresh,
                        h + np.random.randint(thresh) - half_thresh])
    trans = WarpMLS(img, src_pts, dst_pts, w, h)
    dst = trans.generate()

    return dst


def tia_stretch(src, segment=4):
    img_h, img_w = src.shape[:2]

    cut = img_w // segment
    thresh = cut * 4 // 5

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([0, 0])
    dst_pts.append([img_w, 0])
    dst_pts.append([img_w, img_h])
    dst_pts.append([0, img_h])

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        move = np.random.randint(thresh) - half_thresh
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([cut * cut_idx + move, 0])
        dst_pts.append([cut * cut_idx + move, img_h])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()
    return dst


def tia_perspective(src):
    img_h, img_w = src.shape[:2]

    thresh = img_h // 2

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([0, np.random.randint(thresh)])
    dst_pts.append([img_w, np.random.randint(thresh)])
    dst_pts.append([img_w, img_h - np.random.randint(thresh)])
    dst_pts.append([0, img_h - np.random.randint(thresh)])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst


def get_crop(image):
    h, w = image.shape[:2]
    top_min, top_max = 1, 8
    top_crop = min(int(random.randint(top_min, top_max)), 1)
    crop_img = image.copy()
    ratio = random.randint(0, 1)
    if ratio:
        crop_img = crop_img[top_crop:h, :, :]
    else:
        crop_img = crop_img[0: h-top_crop, :, :]
    return crop_img


def blur(image):
    return cv.GaussianBlur(image, (5, 5), 1)


def cvt_color(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    delta = 0.001 * random.random() * do()
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + delta)
    new_img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return new_img


def jitter(img):
    """抖动"""
    w, h = img.shape[:2]
    if w > 10 and h > 10:
        thres = min(w, h)
        s = int(random.random() * thres * 0.01)
        src_img = img.copy()
        for i in range(s):
            img[i:, i:, :] = src_img[:w-i, :h-i, :]

    return img


def add_noise(image, mean=0, var=0.1):
    noise = np.random.normal(mean, var**0.5, image.shape)
    out = image + 0.5 * noise
    out = np.clip(out, 0, 255)
    out = np.uint8(out)
    return out
