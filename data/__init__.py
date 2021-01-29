
import copy
from .aug import create_transformers
from .rec_dataloader import RecDataset
from torch.utils.data import DataLoader, DistributedSampler, BatchSampler

__all__ = ['create_transformers']


def build_dataloader(config, device, logger, mode):
    """创建dataloader， 用于数据加载"""
    config = copy.deepcopy(config)

    support_dict = ['RecDataset', 'DetDataset']
    module_name = config[mode]['dataset']['name']

    if module_name not in support_dict:
        logger.error('{} not in support dict {}'.format(module_name, support_dict))
        exit(-1)

    assert mode in ['Train', 'Eval', 'Test'], "Mode should be Train, Eval or Test"
    loader_config = config[mode]['loader']
    dataset = eval(module_name)(config, mode, logger)

    dataloader = DataLoader(dataset,
                            batch_size=loader_config['batch_size'],
                            shuffle=loader_config['shuffle'],
                            num_workers=loader_config['num_workers'],
                            drop_last=loader_config['drop_last'])
    for i in dataloader:
        pass
    return dataloader

