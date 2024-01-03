import copy

import torch

from dataset import COLDataset


class COLDMLMDataset(COLDataset):
    def __init__(self, root_path, datatype, model_name_or_path, class_num, max_length, mlm_config):
        super(COLDMLMDataset, self).__init__(root_path, datatype, model_name_or_path, class_num, max_length)
        self.data_length = super(COLDMLMDataset, self).__len__()
        self.mlm_config = mlm_config
        self.ori_data = copy.deepcopy(self.data)

    def __getitem__(self, index):
        # item = super(COLDMLMDataset, self).__getitem__(index)
        batch_item = self.data['text'][:self.mlm_config.batch_size]
        features = self.tokenizer(batch_item,
                                  max_length=128,
                                  padding=True,
                                  truncation=True,
                                  return_tensors='pt')
        inputs, labels = self.mask_token(features['input_ids'])
        self.data['text'] = self.data['text'][self.mlm_config.batch_size:]
        if not len(self):
            self.data = copy.deepcopy(self.ori_data)
        return {
            'input_ids': inputs,
            'attention_mask': features['attention_mask'],
            'labels': labels
        }

    def __len__(self):
        return len(self.data['text']) // self.mlm_config.batch_size

    def mask_token(self, inputs):
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_config.mlm_probability)
        if self.mlm_config.special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                # self.tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = self.mlm_config.special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(
            torch.full(labels.shape, self.mlm_config.prob_replace_mask)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        current_prob = self.mlm_config.prob_replace_rand / (1 - self.mlm_config.prob_replace_mask)
        indices_random = torch.bernoulli(
            torch.full(labels.shape, current_prob)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
