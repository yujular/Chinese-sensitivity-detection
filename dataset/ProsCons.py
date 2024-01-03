import json
import os

import torch

from dataset.OLD import OLDBase


class ProsCons(OLDBase):
    DATA_FOLDER = 'ProsCons'

    def __init__(self, root_path, datatype, model_name_or_path, class_num=2, max_length=128):
        # total: 32667, train: 26134, dev: 3267, test: 3266
        super().__init__(root_path, datatype, model_name_or_path, class_num, max_length)

    def load_data(self, datatype):
        # messages: proposal': [], 'post': []
        # labels: 'stance': [], 'offense': [], 'sarcasm': [], 'sentiment': []

        with open(os.path.join(self.path, self.DATA_FOLDER), 'r', encoding='utf-8') as f:
            data = json.load(f)

        if datatype == 'all':
            self.data = data
        elif datatype == 'train/dev':
            self.data = data[:26134 + 3267]
        elif datatype == 'train':
            self.data = data[:26134]
        elif datatype == 'dev':
            self.data = data[26134:26134 + 3267]
        elif datatype == 'test':
            self.data = data[26134 + 3267:]
        else:
            raise ValueError('datatype must be one of [all, train/dev, train, dev, test]')
        print('load data from {} success'.format(self.path) +
              ', length: {}'.format(len(self.data)) +
              ', datatype: {}'.format(datatype))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        post = self.data[index]['post']
        label = self.data[index]['offense']

        inputs = self.tokenizer(
            post,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length)

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
