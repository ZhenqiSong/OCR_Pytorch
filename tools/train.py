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
from postprocess import build_post_process
from models.architectures import build_model
from models.losses import build_loss
from models.optimizer import build_optimizer


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

    set_seed(global_config['seed'], global_config['use_gpu'])

    config['Train']['loader']['batch_size'] = config['Train']['loader']['batch_size_per_card'] * n_gpus
    config['Eval']['loader']['batch_size'] = config['Eval']['loader']['batch_size_per_card'] * n_gpus
    logger.info("加载训练集：{}".format(config['Train']['loader']))
    # train_dataloader = build_dataloader(config, device, logger, 'Train')
    logger.info("加载验证集：{}".format(config['Eval']['loader']))
    # valid_dataloader = build_dataloader(config, device, logger, 'Eval') if config['Eval'] else None

    post_process_class = build_post_process(config['PostProcess'], global_config)

    # 构建模型
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        config['Architecture']["Head"]['out_channels'] = char_num
    logger.info("创建模型：{}".format(config['Architecture']))
    model = build_model(config['Architecture'])

    logger.info('创建损失函数：{}'.format(config['Loss']['name']))
    loss_class = build_loss(config['Loss'])

    optimizer, lr_scheduler = build_optimizer(
        config['Optimizer'],
        epochs=config['Global']['epoch_num'],
        step_each_epoch=1024,
        parameters=model.parameters()
    )

    # logger.warning(
    #     "Process rank: %s, device: %s, distributed training: %s, 16-bits training: %s",
    #     config['Common']['local_rank'],
    #     device,
    #     bool(config['Common']['local_rank'] != -1),
    #     config['Common']['fp16'],
    # )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help='The path of config file!')
    args = parser.parse_args()

    config = get_config(args.config)

    main()
