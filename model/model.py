import torch

from model.bert import BertBaseModel


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
