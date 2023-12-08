import os

import pandas as pd
import torch

from dataset.OLD import OLDBase


class COLDataset(OLDBase):
    TOPIC2LABEL = {
        'race': 0,
        'region': 1,
        'gender': 2
    }
    DATA_FOLDER = 'COLDataset'

    def __init__(self, root_path, datatype, model_name_or_path, class_num=2, max_length=128):
        """
           COLDataset
           Args:
                root_path(`str`): 数据集根目录
                datatype(`str`): 数据集类型
                model_name_or_path(`str`): 模型名称或路径, 用于初始化tokenizer
                class_num(`int`): 类别数量
                max_length(`int`): 最大长度

            Returns:
                each record: a dict of {input_ids, attention_mask, label}
                input_ids(`torch.tensor`):
                    token id
                attention_mask(`torch.tensor`):
                    mask
                labels(`torch.tensor`):
                    标签
        """
        # train/dev:32157, test: 5323, total: 37480
        super().__init__(root_path, datatype, model_name_or_path, class_num, max_length)

    def load_data(self, datatype):
        self.data = {'topic': [], 'label': [], 'text': [], 'length': 0, 'fine-grained-label': []}
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
        if self.class_num == 2:
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

    def __get_max_length__(self):
        # 遍历所有文本，获取最大语句长度和对应index
        max_length = 0
        index = 0
        for i, text in enumerate(self.data['text']):
            if len(text) > max_length:
                max_length = len(text)
                index = i

        return max_length, index

    def get_truncation_num(self):
        num = 0
        for text in self.data['text']:
            if len(text) > self.max_length:
                num += 1
        return num


def test_COLD_data():
    """ test CLODataset """
    cold_dataset = COLDataset('data', 'train',
                              'models/bert-base-chinese',
                              2, 128)
    print(len(cold_dataset))
    print(cold_dataset.__getitem__(0))


if __name__ == '__main__':
    test_COLD_data()
