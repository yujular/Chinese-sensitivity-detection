import os

import torch.nn.functional as F
from torch import nn
from transformers import BertModel, XLMRobertaModel

from models.cnnmodel import CNNAdapter


class BertBaseModel(nn.Module):
    """BERT with simple linear models."""

    def __init__(self, args):
        """Initialize the models with args dict.
            Args:
            args: python dict must contain the attributes below:
                args.models['model_root_path']: string, pretrained models root path
                    e.g. 'models'
                args.models['model_name']: string, pretrained models name
                    e.g. 'bert-base-chinese'
                args.models['hugging_face']: bool, whether to use hugging face to load models
                args.models['hidden_size']: The same as BERT models, usually 768
                args.dataset['num_classes']: int, e.g. 2
                args.models['dropout']: bool, whether to use dropout
                args.models['dropout_rate']: float between 0 and 1
        """
        super(BertBaseModel, self).__init__()

        self.args = args
        self.class_num = self.args.dataset['class_num']

        if self.args.model['hugging_face']:
            # 根据模型名称直接加载，需要访问hugging face
            self.bert = BertModel.from_pretrained(args.model['model_name'])
        else:
            # 根据路径加载本地模型
            self.path = os.path.join(self.args.model['model_root_path'], self.args.model['model_name'])
            self.bert = BertModel.from_pretrained(self.path)

        # 是否解冻预训练BERT
        if self.args.model['freezing']:
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            for param in self.bert.parameters():
                param.requires_grad = True

        self.fc = nn.Linear(self.args.model['hidden_size'], self.class_num)

        if self.args.model['dropout']:
            self.dropout = nn.Dropout(self.args.model['dropout_rate'])

    def forward(self, input_ids, attention_mask):
        """Forward inputs and get logits, for single sentence classification, token_type_id=0.

        Args:
            input_ids: (batch_size, max_seq_len)
            attention_mask: (batch_size, max_seq_len)

        Returns:
            outs: (batch_size, num_classes)
        """
        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=False)
        # bert_output[0]: last_hidden_state(batch_size, sequence_length, hidden_size)
        # bert_output[1]: pooled_output: (batch_size, hidden_size)
        pooled_output = bert_output[1]

        # dropout
        if self.args.model['dropout']:
            pooled_output = self.dropout(pooled_output)
        # linear
        logits = self.fc(pooled_output)
        # softmax
        outs = nn.functional.softmax(logits, dim=-1)
        return outs

    def get_cls_feature(self, input_ids, attention_mask):
        """Get cls feature for visualization.

        Args:
            input_ids: (batch_size, max_seq_len)
            attention_mask: (batch_size, max_seq_len)
        """
        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=False)
        pooled_output = bert_output[1]
        # dropout
        if self.args.model['dropout']:
            pooled_output = self.dropout(pooled_output)

        return pooled_output

    def feature_num(self):
        return self.args.model['hidden_size']


class BertCNNModel(nn.Module):
    """BERT with CNN."""

    def __init__(self, args):
        super(BertCNNModel, self).__init__()

        self.args = args
        self.class_num = self.args.dataset['class_num']

        # 加载BERT主干网络
        if self.args.model['hugging_face']:
            # 根据模型名称直接加载，需要访问hugging face
            self.bert = BertModel.from_pretrained(args.model['model_name'])
        else:
            # 根据路径加载本地模型
            self.path = os.path.join(self.args.model['model_root_path'], self.args.model['model_name'])
            self.bert = BertModel.from_pretrained(self.path)

        # 是否解冻预训练BERT
        if self.args.model['freezing']:
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            for param in self.bert.parameters():
                param.requires_grad = True

        # CNN
        self.cnn_adapter = CNNAdapter(args)
        self.fc = nn.Linear(self.cnn_adapter.feature_num(), self.class_num)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=False)
        # bert_output[0]: last_hidden_state(batch_size, sequence_length, hidden_size)
        bert_out = bert_output[0]
        cnn_out = self.cnn_adapter(bert_out)

        # linear
        logits = self.fc(cnn_out)
        # softmax
        outs = F.softmax(logits, dim=-1)
        return outs

    def feature_num(self):
        return self.cnn_adapter.feature_num()

    def get_cls_feature(self, input_ids, attention_mask):
        """Get cls feature for visualization.

        Args:
            input_ids: (batch_size, max_seq_len)
            attention_mask: (batch_size, max_seq_len)
        """
        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=False)
        bert_out = bert_output[0]
        cnn_out = self.cnn_adapter(bert_out)
        return cnn_out


class BertGenericModel(nn.Module):
    def __init__(self, bert_path_name, bottleneck=None, class_num=2, using_cls=False, freezing=False):
        super(BertGenericModel, self).__init__()

        self.bert_path_name = bert_path_name

        self.class_num = class_num
        self.freezing = freezing
        self.using_cls = using_cls

        # 加载BERT网络
        if 'bert-base-chinese' in bert_path_name:
            self.bert = BertModel.from_pretrained(bert_path_name)
        elif 'xlm-roberta' in bert_path_name:
            self.bert = XLMRobertaModel.from_pretrained(bert_path_name)
        # 是否解冻预训练BERT
        if self.freezing:
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            for param in self.bert.parameters():
                param.requires_grad = True

        self.cls_model = bottleneck
        self.fc = nn.Linear(self.cls_model.num_features, self.class_num)

    def forward(self, inputs):
        cls_out = self.get_cls_feature(inputs)
        logits = self.fc(cls_out)
        outs = F.softmax(logits, dim=-1)
        return outs

    def feature_num(self):
        if self.cls_model is not None:
            return self.cls_model.num_features
        else:
            return self.bert.config.hidden_size

    def get_cls_feature(self, inputs):
        """Get cls feature for visualization.

        Args:
            inputs: (batch_size, max_seq_len)
        """
        bert_output = self.bert(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            encoder_hidden_states=False)
        if self.using_cls:
            # (N, hidden_size): (N,768)
            bert_out = bert_output[1]
        else:
            # (N, sequence_length, hidden_size): (N, 128, 768)
            bert_out = bert_output[0]

        cls_out = self.cls_model(bert_out)
        return cls_out


if __name__ == '__main__':
    print(1)
