# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2020/11/10

import argparse
import yaml
import logging
import random
import torch
import numpy as np
import json

import torch.distributed as dist
from utils import get_logger, get_config
from utils.train_utils import is_main_process
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

def train():
    pass

def main():
    logger = get_logger()

    global_config = config['Global']

    # 初始化设备
    use_gpu = global_config['use_gpu']
    if global_config['local_rank'] == -1 or not use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        global_config.update({'n_gpu': torch.cuda.device_count() if use_gpu else 1})
    else:
        torch.cuda.set_device(global_config['local_rank'])
        device = torch.device('cuda', global_config['local_rank'])
        dist.init_process_group(backend='nccl')
        global_config.update({'n_gpu': 1})
    global_config.update({'device': device})
    logger.warning(f"\n\tProcess Rank：{global_config['local_rank']} \n"
                   f"\tDevice: {device}\n"
                   f"\tGpus: {global_config['n_gpu']}\n"
                   f"\tDistributed: {bool(global_config['local_rank'] != -1)}\n"
                   f"\t16-bits training: {global_config['fp16']}")

    rank_id = global_config['local_rank']
    set_seed(global_config['seed'], use_gpu)

    # 阻塞子进程，下面的操作仅主进程进行
    if not is_main_process(rank_id):
        dist.barrier()

    post_process = build_post_process(config['PostProcess'], global_config)

    # 构建模型
    arch_config = config.pop('Architecture')
    if hasattr(post_process, 'character'):
        char_num = len(getattr(post_process, 'character'))
        arch_config["Head"]['out_channels'] = char_num
    logger.info(f"\nModel Info:"
                f"\n{json.dumps(arch_config, indent=4)}")
    model = build_model(arch_config)
    state_dict = torch.load(global_config['pretrained_model'])
    model.load_state_dict(state_dict)

    # 加载训练数据
    if global_config['local_rank'] == 0:
        dist.barrier()
    logger.info(f"\nLoad train Data:"
                f"\n{json.dumps(config['Train'], indent=4)}")
    train_dataloader = build_dataloader(config, logger, 'Train')

    logger.info(f"\nLoad Eval Data:"
                f"\n{json.dumps(config['Eval'], indent=4)}")
    eval_dataloader = build_dataloader(config, logger, 'Eval')
    if global_config['local_rank'] == 0:
        dist.barrier()

    model.to(device)


    # loss_class = build_loss(config['Loss'])
    #
    # optimizer, lr_scheduler = build_optimizer(
    #     config['Optimizer'],
    #     epochs=config['Global']['epoch_num'],
    #     step_each_epoch=1024,
    #     parameters=model.parameters()
    # )

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
