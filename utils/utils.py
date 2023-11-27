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
    """Load state dict to model.

       Args:
            :param model: model to be loaded
            :param model_path: state dict file path
            :param device: device
            :param strict: whether to strictly load the model

       Returns:
           loaded model

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
