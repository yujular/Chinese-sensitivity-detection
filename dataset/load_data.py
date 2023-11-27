import json
import os.path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


# 定义自定义数据集类
class COLDataset(Dataset):
    TOPIC2LABEL = {
        'race': 0,
        'region': 1,
        'gender': 2
    }

    def __init__(self, args, datatype, max_length=128):
        """
           COLDataset
           Args:
                args(`dict`):
                    config.yml配置参数
                datatype(`str`):
                    train, dev, test, train/dev, all
                max_length(`int`):
                    token最大长度

            Returns:
                each record: a dict of {input_ids, attention_mask, label}
                input_ids(`torch.tensor`):
                    token id
                attention_mask(`torch.tensor`):
                    mask
                labels(`torch.tensor`):
                    标签
        """

        self.args = args

        self.root_path = args.dataset['dataset_root_path']
        self.dataset_name = args.dataset['dataset_name']
        self.path = os.path.join(self.root_path, self.dataset_name)
        self.model = args.model['model_name']
        self.max_length = max_length
        self.datatype = datatype

        # 加载tokenizer, 自动添加CLS, SEP
        self.vocab = os.path.join(self.args.model['model_root_path'], self.args.model['model_name'], 'vocab.txt')
        if self.model == 'bert-base-chinese':
            model_path = os.path.join(self.args.model['model_root_path'], self.args.model['model_name'])
            self.tokenizer = BertTokenizerFast.from_pretrained(model_path, add_special_tokens=True, do_lower_case=True,
                                                               do_basic_tokenize=True)
        else:
            # from vocab.txt
            self.tokenizer = BertTokenizerFast(vocab_file=self.vocab, do_lower_case=True, do_basic_tokenize=True)

        # 加载数据, train/dev:32157, test: 5323, total: 37480
        self.data = {'topic': [], 'label': [], 'text': [], 'length': 0, 'fine-grained-label': []}
        self.load_data(datatype)

    def load_data(self, datatype):
        if datatype == 'all':
            self._load_data(os.path.join(self.path, 'train.csv'))
            self._load_data(os.path.join(self.path, 'dev.csv'))
            self._load_data(os.path.join(self.path, 'test.csv'))
        elif datatype == 'train/dev':
            self._load_data(os.path.join(self.path, 'train.csv'))
            self._load_data(os.path.join(self.path, 'dev.csv'))
        else:
            self._load_data(filename=os.path.join(self.path, datatype + '.csv'),
                            test=(datatype == 'test'))

    def _load_data(self, filename, test=False):
        dataframe = pd.read_csv(filename)
        self.data['topic'].extend(dataframe['topic'].tolist())
        self.data['label'].extend(dataframe['label'].tolist())
        self.data['text'].extend(dataframe['TEXT'].tolist())
        self.data['length'] += len(dataframe['TEXT'].tolist())
        if test:
            self.data['fine-grained-label'].extend(dataframe['fine-grained-label'].tolist())
        print('load data from {} success'.format(filename) + ', length: {}'.format(self.data['length']))

    def __len__(self):
        return self.data['length']

    def __getitem__(self, index):
        if not self.args.dataset['multi_class']:
            label = self.data['label'][index]
        else:
            # topic映射类别
            topic_id = self.TOPIC2LABEL[self.data['topic'][index]]
            label = self.data['label'][index] * 3 + topic_id

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
        if self.datatype == 'test':
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': torch.tensor(label, dtype=torch.long),
                'fine-grained-label': torch.tensor(self.data['fine-grained-label'][index], dtype=torch.long)
            }
        else:
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': torch.tensor(label, dtype=torch.long)
            }


def test_COLD_data():
    """ test CLODataset """
    import config
    args = config.load_args('config/config.yml')

    data = COLDataset(args, 'dev')
    print(data.__getitem__(0))


class KOLDataset(Dataset):
    TRAIN_LEN = 36386
    DEV_LEN = 2021
    TEST_LEN = 2022

    def __init__(self, args, datatype, max_length=128):
        """
           KOLDataset, total: 40429, train: 36386, dev: 2021, test: 2022
           Args:
                args(`dict`):
                    config.yml配置参数
                datatype(`str`):
                    train, dev, test, train/dev, all
                max_length(`int`):
                    token最大长度

            Returns:
                each record: a dict of {input_ids, attention_mask, label}
                input_ids(`torch.tensor`):
                    token id
                attention_mask(`torch.tensor`):
                    mask
                labels(`torch.tensor`):
                    标签
        """

        self.args = args

        self.root_path = args.dataset['dataset_root_path']
        self.dataset_name = args.dataset['dataset_name']
        self.path = os.path.join(self.root_path, self.dataset_name)
        self.model = args.model['model_name']
        self.max_length = max_length

        # 加载tokenizer, 自动添加CLS, SEP
        self.vocab = os.path.join(self.args.model['model_root_path'], self.args.model['model_name'], 'vocab.txt')
        if self.model == 'bert-base-chinese':
            model_path = os.path.join(self.args.model['model_root_path'], self.args.model['model_name'])
            self.tokenizer = BertTokenizerFast.from_pretrained(model_path, add_special_tokens=True, do_lower_case=True,
                                                               do_basic_tokenize=True)
        else:
            # from vocab.txt
            self.tokenizer = BertTokenizerFast(vocab_file=self.vocab, do_lower_case=True, do_basic_tokenize=True)

        # 加载数据
        self.data = []
        self.load_json_data(datatype)

    def load_json_data(self, datatype):
        # 从Json文件加载数据
        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if datatype == 'all':
            self.data = data
        elif datatype == 'train/dev':
            self.data = data[:self.TRAIN_LEN + self.DEV_LEN]
        elif datatype == 'train':
            self.data = data[:self.TRAIN_LEN]
        elif datatype == 'dev':
            self.data = data[self.TRAIN_LEN:self.TRAIN_LEN + self.DEV_LEN]
        elif datatype == 'test':
            self.data = data[self.TRAIN_LEN + self.DEV_LEN:]
        else:
            raise ValueError('datatype must be one of [all, train/dev, train, dev, test]')
        print('load data from {} success'.format(self.path) +
              ', length: {}'.format(len(self.data)) +
              ', datatype: {}'.format(datatype))

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, index):
    # text = self.data[index]['text']
    #
    # # 使用 tokenizer 对文本进行标记化和编码
    # inputs = self.tokenizer(
    #     text,
    #     padding='max_length',
    #     truncation=True,
    #     max_length=self.max_length,
    #     return_tensors='pt'
    # )
    #
    # # 返回输入文本张量和对应的标签
    # return {
    #     'input_ids': inputs['input_ids'].squeeze(),
    #     'attention_mask': inputs['attention_mask'].squeeze(),
    #     'labels': torch.tensor(label, dtype=torch.long)
    # }


if __name__ == '__main__':
    test_COLD_data()
