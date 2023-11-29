import os

from torch import nn
from transformers import BertModel


class BertBaseModel(nn.Module):
    """BERT with simple linear model."""

    def __init__(self, args):
        """Initialize the model with args dict.
            Args:
            args: python dict must contain the attributes below:
                args.model['model_root_path']: string, pretrained model root path
                    e.g. 'model'
                args.model['model_name']: string, pretrained model name
                    e.g. 'bert-base-chinese'
                args.model['hugging_face']: bool, whether to use hugging face to load model
                args.model['hidden_size']: The same as BERT model, usually 768
                args.dataset['num_classes']: int, e.g. 2
                args.model['dropout']: bool, whether to use dropout
                args.model['dropout_rate']: float between 0 and 1
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
