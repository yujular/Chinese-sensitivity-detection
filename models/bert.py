import os
import torch.nn.functional as F
from torch import nn
from transformers import BertModel

from dataset import get_dataloader
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


def test_BERTCNN():
    import config
    args = config.load_args('config/config.yml')

    data_loader = get_dataloader('COLD', args, ['train'])

    iter_source = iter(data_loader['train'])
    data_source_id, data_source_mask, label_source = next(iter_source).values()
    data_source_id, data_source_mask, label_source = (data_source_id.to(args.train['device']),
                                                      data_source_mask.to(args.train['device']),
                                                      label_source.to(args.train['device']))

    model = BertCNNModel(args).to(args.train['device'])
    out = model(data_source_id, data_source_mask)
    print(out)


if __name__ == '__main__':
    test_BERTCNN()
    print(2)
