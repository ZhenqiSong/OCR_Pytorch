from torchvision import transforms
from .operators import *
from .rec_img_aug import RecAug
from .label_trans import CTCLabelEncode


def create_transformers(op_param_list: list, global_config=None):
    """
    根据配置文件创建图像预处理操作
    """
    assert isinstance(op_param_list, list), "operator config should be a list"
    ops = []
    for operator in op_param_list:
        assert isinstance(operator, dict) and len(operator) == 1, "config file format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return transforms.Compose(ops)

