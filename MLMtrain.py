import os.path

import torch
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from transformers import AutoModelForMaskedLM, AdamW, get_linear_schedule_with_warmup

from MLM import get_MLMConfig, COLDMLMDataset


def train(model, train_loader, config):
    data_len = len(train_loader)

    assert config.device.startswith('cuda') or config.device == 'cpu', ValueError("Invalid device.")
    device = torch.device(config.device)

    model.to(device)

    # optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=config.learning_rate, weight_decay=config.weight_decay)

    # scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=config.warmup_steps,
                                                num_training_steps=data_len * config.epochs)

    for epoch in trange(config.epochs, desc="Epoch"):
        training_loss = 0
        print("Epoch: {}".format(epoch + 1))
        model.train()

        for step, batch in enumerate(tqdm(train_loader, desc="Step")):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(input_ids=batch['input_ids'].squeeze(0),
                         attention_mask=batch['attention_mask'].squeeze(0),
                         labels=batch['labels'].squeeze(0)).loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            training_loss += loss.item()
        print("Training loss: {}".format(training_loss / data_len))


if __name__ == '__main__':
    mlm_config = get_MLMConfig()
    cold_dataset = COLDMLMDataset(root_path='data',
                                  datatype='train',
                                  model_name_or_path=mlm_config.from_path,
                                  class_num=2,
                                  max_length=128,
                                  mlm_config=mlm_config)
    cold_loader = DataLoader(cold_dataset, shuffle=True)

    mlm_model = AutoModelForMaskedLM.from_pretrained(mlm_config.from_path)
    train(mlm_model, cold_loader, mlm_config)

    if mlm_config.save_path is not None:
        os.makedirs(mlm_config.save_path, exist_ok=True)
    torch.save(mlm_model.state_dict(), os.path.join(mlm_config.save_path, 'MLM_epoch{}.bin'.format(mlm_config.epochs)))
