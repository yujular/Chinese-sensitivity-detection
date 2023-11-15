import os.path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


# 定义自定义数据集类
class COLDataset(Dataset):
    def __init__(self, args, max_length=128):
        self.args = args

        self.root_path = args.dataset['dataset_root_path']
        self.dataset_name = args.dataset['dataset_name']
        self.path = os.path.join(self.root_path, self.dataset_name)
        self.model = args.model['model_name']
        self.max_length = max_length

        # 加载tokenizer, 自动添加CLS, SEP
        self.vocab = os.path.join(self.args.model['model_root_path'], self.args.model['model_name'], 'vocab.txt')
        if self.model == 'bert-base-chinese':
            self.tokenizer = BertTokenizerFast.from_pretrained(self.model, add_special_tokens=True, do_lower_case=True,
                                                               do_basic_tokenize=True)
        else:
            # from vocab.txt
            self.tokenizer = BertTokenizerFast(vocab_file=self.vocab, do_lower_case=True, do_basic_tokenize=True)

        # 加载数据, train/dev:32157, test: 5323, total: 37480
        self.data = {'topic': [], 'label': [], 'text': [], 'length': 0}
        self.load_data()

    def load_data(self, train: bool = True):
        if train:
            self._load_data(os.path.join(self.path, 'train.csv'))
            self._load_data(os.path.join(self.path, 'dev.csv'))
        else:
            self._load_data(os.path.join(self.path, 'test.csv'))

    def _load_data(self, filename):
        dataframe = pd.read_csv(filename)
        self.data['topic'].extend(dataframe['topic'].tolist())
        self.data['label'].extend(dataframe['label'].tolist())
        self.data['text'].extend(dataframe['TEXT'].tolist())
        self.data['length'] += len(dataframe['TEXT'].tolist())
        print('load data from {} success'.format(filename) + ', length: {}'.format(self.data['length']))

    def __len__(self):
        return self.data['length']

    def __getitem__(self, index):
        if self.args.dataset['class_num'] == 2:
            label = self.data['label'][index]
        else:
            # topic映射类别
            label = self.data['topic'][index]

        text = self.data['text'][index]

        # 使用 tokenizer 对文本进行标记化和编码
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 返回输入文本张量和对应的标签
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def test_data():
    """ test CLODataset """
    import config
    args = config.load_args('../config/config.yml')

    data = COLDataset(args)
    print(data.__getitem__(0))


if __name__ == '__main__':
    test_data()
