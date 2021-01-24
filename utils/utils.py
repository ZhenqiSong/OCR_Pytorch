# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-20

import os
import sys
import yaml
import logging
import functools

logger_initialized = set()


def get_img_list(img_file):
    img_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise FileNotFoundError("file path: {} is not exist".format(img_file))

    if os.path.isfile(img_file):
        img_lists.append(img_file)
    elif os.path.isdir(img_file):
        for file_name in os.listdir(img_file):
            file_path = os.path.join(img_file, file_name)
            if os.path.isfile(file_path):
                img_lists.append(file_name)
    if len(img_lists) == 0:
        raise Exception('not find any img file in {}'.format(img_file))
    return img_lists


def get_config(file):
    """
    读取yaml配置文件，获取网络配置
    :param file: 配置文件，只支持yaml/yml格式
    :return: 配置 dict
    """
    _, ext = os.path.splitext(file)
    assert ext in ['.yaml', '.yml'], "只支持yaml/yml格式的文件"
    config = yaml.load(open(file, 'rb'), Loader=yaml.Loader)
    return config


@functools.lru_cache()
def get_logger(name: str = 'root', file: str = None, level=logging.INFO) -> logging.Logger:
    """
    初始化日志logger，配置日志的设置
    :param name: 日志名称
    :param file: 保存本地的日志文件
    :param level: 日志显示的等级
    :return: 使用的Logger对象
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger

    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # 设置日志的显示格式
    formatter = logging.Formatter('[%(asctime)s] %(name)s %(levelname)s: %(message)s',
                                  datefmt="%Y/%m/%d %H:%M:%S")

    # 设置日志流句柄
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 设置日志文件
    if file is not None:
        log_file_folder = os.path.split(file)[0]
        os.makedirs(log_file_folder, exist_ok=True)
        file_handle = logging.FileHandler(file, 'a')
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)

    logger.setLevel(level)
    return logger
