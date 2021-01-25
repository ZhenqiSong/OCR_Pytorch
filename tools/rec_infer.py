# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-20

from argparse import ArgumentParser

import torch

from utils import get_logger, get_config, get_img_list
from models.architectures import build_model
from postprocess import build_post_process
from data import create_transformers


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--cfg', '-c', help='Config file', required=True)

    return parser.parse_args()


def main():
    global_config = config['Global']
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    if hasattr(post_process_class, 'character'):
        config['Architecture']["Head"]['out_channels'] = len(getattr(post_process_class, 'character'))
    logger.info('构建模型，字典包含{}个字'.format(config['Architecture']["Head"]['out_channels']))
    model = build_model(config['Architecture'])

    logger.info('加载预训练模型 {}...'.format(global_config['pretrained_model']))
    state_dict = torch.load(global_config['pretrained_model'])
    model.load_state_dict(state_dict)

    ops = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name in ['RecResizeImg']:
            op[op_name]['infer_mode'] = True
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image']
        ops.append(op)
    global_config['infer_mode'] = True
    transforms = create_transformers(ops, global_config)

    model.eval()
    for file in get_img_list(config['Global']['infer_img']):
        logger.info('输入图像：{}'.format(file))
        data = {'image': file}
        batch = transforms(data)

        images = torch.from_numpy(batch[0]).unsqueeze(0)
        preds = model(images)
        post_result = post_process_class(preds)
        logger.info("result: {}".format(post_result))


if __name__ == '__main__':
    args = get_args()
    config = get_config(args.cfg)
    logger = get_logger()
    logger.info("当前配置文件为：{}".format(args.cfg))
    main()
