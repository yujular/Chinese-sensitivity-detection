import os

import pandas as pd
import torch

from dataset.OLD import OLDBase


class OLIDataset(OLDBase):
    LABEL_MAP = {
        'NOT': 0,
        'OFF': 1
    }
    DATA_FOLDER = 'OLID'

    def __init__(self, root_path, datatype, model_name_or_path, class_num=2, max_length=128):
        """
           OLIDataset, total: 14,100, train: 13,240, test: 860
           Args:
                root_path(`str`): root path of dataset
                datatype(`str`): train, test, dev
                model_name_or_path(`str`): model name or path
                class_num(`int`): number of classes
                max_length(`int`): max length of input sequence

            Returns:
                each record: a dict of {input_ids, attention_mask, label}
                input_ids(`torch.tensor`):
                    token id
                attention_mask(`torch.tensor`):
                    mask
                labels(`torch.tensor`):
                    label
        """
        super().__init__(root_path, datatype, model_name_or_path, class_num, max_length)

    def load_data(self, datatype):
        if self.datatype == 'train' or self.datatype == 'dev':
            filename = 'olid-training-v1.0.tsv'
        else:
            filename = 'olid-training-v1.0.tsv'
            # filename = ['testset-levela.tsv', 'testset-levelb.tsv', 'testset-levelc.tsv']
            # TODO
        dataframe = pd.read_csv(os.path.join(self.path, filename), sep='\t')
        # total:13420, train: 12000, dev: 1420
        if self.datatype == 'tran':
            dataframe = dataframe[:12000]
        elif self.datatype == 'dev':
            dataframe = dataframe[12000:]

        self.data = {'id': dataframe['id'].tolist(), 'tweet': dataframe['tweet'].tolist(),
                     'subtask_a': dataframe['subtask_a'].tolist(), 'subtask_b': dataframe['subtask_b'].tolist(),
                     'subtask_c': dataframe['subtask_c'].tolist(), 'length': len(dataframe['id'].tolist())}
        print('load data from {} success'.format(filename) + ', length: {}'.format(self.data['length']))

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
    olid_dataset = OLIDataset(root_path='data',
                              datatype='train',
                              model_name_or_path='bert-base-chinese',
                              class_num=2,
                              max_length=128)
    print(olid_dataset.__getitem__(0))
    print(len(olid_dataset))


if __name__ == '__main__':
    test_OLID()
