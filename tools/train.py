# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2020/11/10

import argparse
import yaml
import logging
import random
import torch
import numpy as np

from utils import get_logger, get_config
from data import build_dataloader


def set_seed(seed, gpu=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu:
        torch.cuda.manual_seed_all(seed)


def main():
    logger = get_logger()

    global_config = config['Global']
    use_gpu = global_config['use_gpu']

    device = torch.device('cpu')
    n_gpus = 1
    if use_gpu:
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            device = torch.device('cuda')
        else:
            logger.warning("未发现可用于计算的GPU设备")

    config['Train']['loader']['batch_size'] = config['Train']['loader']['batch_size_per_card'] * n_gpus
    config['Eval']['loader']['batch_size'] = config['Eval']['loader']['batch_size_per_card'] * n_gpus
    train_dataloader = build_dataloader(config, device, logger, 'Train')

    # logger.warning(
    #     "Process rank: %s, device: %s, distributed training: %s, 16-bits training: %s",
    #     config['Common']['local_rank'],
    #     device,
    #     bool(config['Common']['local_rank'] != -1),
    #     config['Common']['fp16'],
    # )

    # set_seed(config['Common']['seed'], config['Common']['gpu'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help='The path of config file!')
    args = parser.parse_args()

    config = get_config(args.config)

    main()
