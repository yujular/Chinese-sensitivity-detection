import os

import torch
from torch import nn

from models.bert import BertGenericModel
from models.cnnmodel import CNNAdapter
from models.transfer import TransferNet
from utils import load_torch_model


class Model:
    def __init__(self, args):
        self.args = args

    def get_generic_model(self):
        if self.args.hugging_face:
            bert_path_name = self.args.model
        else:
            bert_path_name = os.path.join(self.args.model_path, self.args.model)

        if self.args.bottleneck == 'cnn':
            bottleneck_layer = CNNAdapter(max_length=self.args.max_length,
                                          hidden_size=self.args.hidden_size,
                                          dropout_rate=self.args.dropout_rate)
            using_cls = False
        else:
            using_cls = True
            bottleneck_layer = nn.Dropout(self.args.dropout_rate)
            bottleneck_layer.num_features = self.args.hidden_size

        model = BertGenericModel(bert_path_name=bert_path_name,
                                 bottleneck=bottleneck_layer,
                                 class_num=self.args.class_num,
                                 using_cls=using_cls,
                                 freezing=self.args.freeze)

        return model

    def get_model(self, load_model):
        if self.args.transfer:
            # 加载单模型
            model = self.get_generic_model()
            if self.args.pretrained:
                # 加载由COLD预训练过的模型
                model_path = os.path.join(self.args.output_dir,
                                          self.args.model_out_path,
                                          'base',
                                          'COLD-class-' + str(self.args.class_num),
                                          self.args.model + '-models.bin')
                model = load_torch_model(model=model, model_path=model_path, device=self.args.device)
            # 加载迁移模型
            model = TransferNet(base_net=model,
                                class_num=self.args.class_num,
                                transfer_loss=self.args.transfer_loss,
                                use_bottleneck=self.args.use_bottleneck,
                                bottleneck_width=self.args.bottleneck_width,
                                max_iter=self.args.max_iter)
        else:
            # 加载数据集模型
            model = self.get_generic_model()

        if load_model:
            if self.args.transfer:
                model_path = os.path.join(self.args.output_dir,
                                          self.args.model_out_path,
                                          'transfer',
                                          self.args.source_data + '-' + self.args.transfer_loss
                                          + '-class-' + str(self.args.class_num),
                                          self.args.model + '-models.bin')
            else:
                model_path = os.path.join(self.args.output_dir,
                                          self.args.model_out_path,
                                          'base',
                                          self.args.train_data + '-class-' + str(self.args.class_num),
                                          self.args.model + '-models.bin')
            print('Loading model from {}...'.format(model_path))
            model = load_torch_model(model=model, model_path=model_path, device=self.args.device)

        # 并行计算
        if self.args.device == 'cuda' and self.args.n_gpu > 1:
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            model = torch.nn.parallel.DistributedDataParallel(
                model, find_unused_parameters=True)

        return model.to(self.args.device)

# def get_trained_model(args, transfer=False, pretrained=False):
#     """
#     :param args:
#     :param transfer: if True, load transfer models
#     :param pretrained: if True, load pretrain models
#     :return:
#     """
#     model = Model(args).get_model()
#
#     if transfer:
#         model = TransferNet(args, model, transfer_loss=args.train['transfer_loss']).to(args.train['device'])
#         model_name = args.model['model_name'] + 'models.bin'
#         model_path = os.path.join(args.train['model_out_path'],
#                                   'transfer',
#                                   args.train['transfer_dataset'] + '-' + args.train['transfer_loss'],
#                                   'class-' + str(args.dataset['class_num']),
#                                   model_name)
#     else:
#         model_path = os.path.join(args.train['model_out_path'],
#                                   'class-' + str(args.dataset['class_num']),
#                                   args.model['model_name'] + 'models.bin')
#     if pretrained:
#         print("Loading models from {}...".format(model_path))
#         model = load_torch_model(model=model, model_path=model_path, device=args.train['device'], strict=True)
#     return model
