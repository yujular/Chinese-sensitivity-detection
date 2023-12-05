import torch
from torch import nn

from loss import TransferLoss


class TransferNet(nn.Module):
    def __init__(self, args, base_net,
                 transfer_loss='mmd', use_bottleneck=True,
                 bottleneck_width=256, max_iter=1000):
        super(TransferNet, self).__init__()

        self.args = args
        self.class_num = self.args.dataset['class_num']
        self.base_net = base_net
        self.transfer_loss = transfer_loss
        self.use_bottleneck = use_bottleneck
        self.bottleneck_width = bottleneck_width
        self.max_iter = max_iter

        # 瓶颈层配置
        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(self.base_net.feature_num(), self.bottleneck_width),
                nn.ReLU()
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            feature_dim = self.bottleneck_width
        else:
            feature_dim = self.base_net.output_num()

        self.classifier_layer = nn.Linear(feature_dim, self.class_num)

        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": self.class_num
        }
        self.criterion = torch.nn.CrossEntropyLoss()
        self.adapt_loss = TransferLoss(**transfer_loss_args)

    def forward(self, source, target, source_label, target_label=None):
        source_id = source['input_ids']
        source_mask = source['attention_mask']
        target_id = target['input_ids']
        target_mask = target['attention_mask']

        source = self.base_net.get_cls_feature(source_id, source_mask)
        target = self.base_net.get_cls_feature(target_id, target_mask)

        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)

        source_clf = self.classifier_layer(source)
        source_clf_loss = self.criterion(source_clf, source_label)
        target_clf = self.classifier_layer(target)
        if target_label is not None:
            target_clf_loss = self.criterion(target_clf, target_label)

        kwargs = {}
        if self.transfer_loss == "lmmd":
            # 目标域无标签时采用预测标签
            kwargs['source_label'] = source_label
            if target_label is not None:
                kwargs['target_logits'] = target_label
            else:
                kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
                kwargs['pseudo'] = True
        # elif self.transfer_loss == "daan":
        #     source_clf = self.classifier_layer(source)
        #     kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
        #     target_clf = self.classifier_layer(target)
        #     kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        # elif self.transfer_loss == 'bnm':
        #     tar_clf = self.classifier_layer(target)
        #     target = nn.Softmax(dim=1)(tar_clf)
        transfer_loss = self.adapt_loss(source, target, **kwargs)
        if target_label is not None:
            return source_clf_loss, target_clf_loss, transfer_loss
        else:
            return source_clf_loss, transfer_loss

    def predict(self, x):
        input_ids = x['input_ids']
        attention_mask = x['attention_mask']

        features = self.base_net.get_cls_feature(input_ids, attention_mask)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        clf = self.classifier_layer(features)
        return clf
