import json
import os.path

import torch

from dataset.OLD import OLDBase


class KOLDataset(OLDBase):
    TRAIN_LEN = 36386
    DEV_LEN = 2021
    TEST_LEN = 2022
    DATA_FOLDER = 'KOLD'
    FILE_NAME = 'kold_v1.json'

    def __init__(self, root_path, datatype, model_name_or_path, class_num=2, max_length=128):
        """
           KOLDataset, total: 40429, train: 36386, dev: 2021, test: 2022
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
        super(KOLDataset, self).__init__(root_path, datatype, model_name_or_path, class_num, max_length)

    def load_data(self, datatype):
        # 从Json文件加载数据
        with open(os.path.join(self.path, self.FILE_NAME), 'r', encoding='utf-8') as f:
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

    def __getitem__(self, index):
        comment = self.data[index]['comment']
        # label=1 when OFF =TRUE; label=0 when OFF =FALSE
        off_label = 1 if self.data[index]['OFF'] else 0

        # 使用 tokenizer 对文本进行标记化和编码
        inputs = self.tokenizer(
            comment,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 返回输入文本张量和对应的标签
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(off_label, dtype=torch.long)
        }


def test_KOLDataset():
    kold_dataset = KOLDataset(root_path='data',
                              datatype='all',
                              model_name_or_path='bert-base-chinese',
                              class_num=2,
                              max_length=128)
    print(len(kold_dataset))
    print(kold_dataset.__getitem__(0))


if __name__ == '__main__':
    test_KOLDataset()
