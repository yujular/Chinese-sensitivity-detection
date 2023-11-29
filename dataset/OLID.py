import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


class OLIDataset(Dataset):
    LABEL_MAP = {
        'NOT': 0,
        'OFF': 1
    }

    def __init__(self, args, datatype, max_length=128):
        """
           OLIDataset, total: 14,100, train: 13,240, test: 860
           Args:
                args(`dict`):
                    config.yml配置参数
                datatype(`str`):
                    train, test, train/dev, all
                max_length(`int`):
                    token最大长度

            Returns:
                each record: a dict of {input_ids, attention_mask, label}
                input_ids(`torch.tensor`):
                    token id
                attention_mask(`torch.tensor`):
                    mask
                labels(`torch.tensor`):
                    label
        """
        self.args = args
        self.datatype = datatype
        self.max_length = max_length
        self.model = args.model['model_name']

        # 加载tokenizer, 自动添加CLS, SEP
        self.vocab = os.path.join(self.args.model['model_root_path'], self.args.model['model_name'], 'vocab.txt')
        if self.model == 'bert-base-chinese':
            model_path = os.path.join(self.args.model['model_root_path'], self.args.model['model_name'])
            self.tokenizer = BertTokenizerFast.from_pretrained(model_path, add_special_tokens=True, do_lower_case=True,
                                                               do_basic_tokenize=True)
        else:
            # from vocab.txt
            self.tokenizer = BertTokenizerFast(vocab_file=self.vocab, do_lower_case=True, do_basic_tokenize=True)

        self.data = self.load_data()

    def load_data(self):
        path = os.path.join(self.args.dataset['dataset_root_path'],
                            self.args.dataset['OLID_name'])
        if self.datatype == 'train':
            filename = 'olid-training-v1.0.tsv'
        # else:
        # filename = ['testset-levela.tsv', 'testset-levelb.tsv', 'testset-levelc.tsv']
        # TODO
        dataframe = pd.read_csv(os.path.join(path, filename), sep='\t')
        data = {'id': dataframe['id'].tolist(), 'tweet': dataframe['tweet'].tolist(),
                'subtask_a': dataframe['subtask_a'].tolist(), 'subtask_b': dataframe['subtask_b'].tolist(),
                'subtask_c': dataframe['subtask_c'].tolist(), 'length': len(dataframe['id'].tolist())}
        print('load data from {} success'.format(filename) + ', length: {}'.format(data['length']))
        return data

    def __len__(self):
        return self.data['length']

    def __getitem__(self, index):
        label = self.LABEL_MAP[self.data['subtask_a'][index]]

        inputs = self.tokenizer(self.data['tweet'][index],
                                max_length=self.max_length,
                                padding='max_length',
                                truncation=True,
                                return_tensors='pt')
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def test_OLID():
    import config
    args = config.load_args('config/config.yml')

    dataset = OLIDataset(args, 'train')
    print(dataset[0])


if __name__ == '__main__':
    test_OLID()
