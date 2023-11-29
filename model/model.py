import os

import torch

from model.bert import BertBaseModel
from model.transfer import TransferNet
from utils import load_torch_model


class Model:
    MODEL_MAP = {
        'bert-base-linear': BertBaseModel
    }

    def __init__(self, args):
        self.args = args

    def get_model(self):
        if self.args.model['model_type'] not in self.MODEL_MAP:
            raise ValueError('model_type not supported')
        model = self.MODEL_MAP[self.args.model['model_type']](self.args)
        model.to(self.args.train['device'])
        if self.args.train['device'] == 'cuda' and self.args.train['n_gpu'] > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, find_unused_parameters=True)
        return model


def get_trained_model(args, transfer=False):
    model = Model(args).get_model()

    if transfer:
        model = TransferNet(args, model, transfer_loss=args.train['transfer_loss']).to(args.train['device'])
        model_path = os.path.join(args.train['model_out_path'],
                                  'transfer',
                                  'class-' + str(args.dataset['class_num']),
                                  args.model['model_name'] + 'model.bin')
    else:
        model_path = os.path.join(args.train['model_out_path'],
                                  'class-' + str(args.dataset['class_num']),
                                  args.model['model_name'] + 'model.bin')
    print("Loading model from {}...".format(model_path))
    model = load_torch_model(model=model, model_path=model_path, device=args.train['device'], strict=True)
    return model
