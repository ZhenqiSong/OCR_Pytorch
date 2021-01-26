# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-26

from argparse import ArgumentParser
import torch

from utils import get_logger, get_config
from models.architectures import build_model


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--cfg', '-c', help='Config file', required=True)

    return parser.parse_args()


def main():
    global_config = config['Global']

    device = torch.device('cpu')
    if global_config['use_gpu'] and torch.cuda.is_available():
        device = torch.device('cuda')
    logger.info('使用设备：{}'.format(device))

    logger.info('模型信息：{}'.format(config['Architecture']))
    model = build_model(config['Architecture'])

    for k, v in model.state_dict().items():
        print(k, v.shape)


if __name__ == '__main__':
    args = get_args()
    config = get_config(args.cfg)
    logger = get_logger()
    logger.info("当前配置文件为：{}".format(args.cfg))
    main()
