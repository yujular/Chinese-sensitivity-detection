import os
import random
from collections import OrderedDict

import numpy as np
import torch


def initRandom(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样


def load_torch_model(model, model_path, device, strict=True):
    """Load state dict to models.

       Args:
            :param model: models to be loaded
            :param model_path: state dict file path
            :param device: device
            :param strict: whether to strictly load the models

       Returns:
           loaded models

    """
    pretrained_model_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, value in pretrained_model_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # remove `module.` in nn.DataParallel
        else:
            name = k
        new_state_dict[name] = value
    model.load_state_dict(new_state_dict, strict=strict)
    model.to(device)
    return model


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(model_dict, path, filename):
    """Save models to file.

       Args:
            :param model_dict: models state dict
            :param path: path to save models
            :param filename: models file name

    """
    # os.makedirs(path, exist_ok=True)
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model_dict, os.path.join(path, filename))
