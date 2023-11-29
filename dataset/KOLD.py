import json
import os

from torch.utils.data import Dataset
from transformers import BertTokenizerFast


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
