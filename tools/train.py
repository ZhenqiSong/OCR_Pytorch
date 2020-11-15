# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2020/11/10

import argparse
import yaml
import logging
import random
import torch
import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed, gpu=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu:
        torch.cuda.manual_seed_all(seed)


def main(config):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if config['Common']['local_rank'] in [-1, 0] else logging.WARN,
    )

    device = torch.device('cuda') if torch.cuda.is_available() and config['Common']['gpu'] \
        else torch.device('cpu')
    logger.warning(
        "Process rank: %s, device: %s, distributed training: %s, 16-bits training: %s",
        config['Common']['local_rank'],
        device,
        bool(config['Common']['local_rank'] != -1),
        config['Common']['fp16'],
    )

    set_seed(config['Common']['seed'], config['Common']['gpu'])

    # 同步所有进程
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help='The path of config file!')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_ = yaml.load(f, Loader=yaml.FullLoader)

    main(config_)
