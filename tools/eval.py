# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-02-01

import argparse
from utils import get_config, get_logger
from data import build_dataloader
from postprocess import build_post_process
from models.architectures import build_model
from models.metric import build_metric
import torch
from utils import train_utils


def main():
    global_config = config['Global']

    use_gpu = global_config['use_gpu']
    n_gpus = 1
    device = torch.device('cpu')
    if use_gpu:
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            device = torch.device('cuda')
        else:
            logger.warning("未发现可用于计算的GPU设备")

    # 创建数据集
    config['Eval']['loader'].update(
        {'batch_size': config['Eval']['loader']['batch_size_per_card'] * n_gpus})
    dataloader = build_dataloader(config, device, logger, 'Eval')
    batch_size = config['Eval']['loader']['batch_size']
    logger.info(f'测试数据共 {len(dataloader)}个batch, 每个batch包含{batch_size}个样本')

    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)
    if hasattr(post_process_class, 'character'):
        config['Architecture']["Head"]['out_channels'] = len(
            getattr(post_process_class, 'character'))
    model = build_model(config['Architecture'])

    # 加载预训练模型
    state_dict = torch.load(global_config['pretrained_model'],
                            map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.to(device)

    eval_class = build_metric(config['Metric'])
    metric = train_utils.eval(model, dataloader, post_process_class, eval_class, device)
    logger.info('metric eval ***************')
    for k, v in metric.items():
        logger.info('{}:{}'.format(k, v))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help='The path of config file!')
    args = parser.parse_args()

    config = get_config(args.config)
    logger = get_logger()
    main()
