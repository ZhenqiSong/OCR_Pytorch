# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-20

from argparse import ArgumentParser
import paddle
from utils import get_logger, get_config
from models.architectures import build_model
import torch


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--paddle', '-p', help='Paddle Model path')
    parser.add_argument('--cfg', '-c', help='Config File')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(args.cfg)
    config = get_config(args.cfg)
    config['Architecture']["Head"]['out_channels'] = 6625
    net = build_model(config['Architecture'])
    # static_dict = torch.load('./test.pth')
    paddle_dict = paddle.load(args.paddle)
    # net.load_state_dict(static_dict)
    net.load_paddle_state_dict(paddle_dict)
    torch.save(net.state_dict(), 'mobilev3_crnn_ctc.pth')