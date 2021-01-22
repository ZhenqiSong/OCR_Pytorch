# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-20

from argparse import ArgumentParser

import torch

from utils import get_logger, get_config
from models.architectures import build_model
from postprocess import build_post_process


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--in_path', '-i', help='input file or dir')
    parser.add_argument('--cfg', '-c', help='Config file')

    return parser.parse_args()


def main():
    global_config = config['Global']
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)
    if hasattr(post_process_class, 'character'):
        config['Architecture']["Head"]['out_channels'] = len(
            getattr(post_process_class, 'character'))
    model = build_model(config['Architecture'])

    state_dict = torch.load(global_config['pretrained_model'])
    model.load_state_dict(state_dict)

    inputs = torch.randn(2, 3, 32, 320)
    model(inputs)


if __name__ == '__main__':
    args = get_args()
    config = get_config(args.cfg)
    logger = get_logger()
    main()