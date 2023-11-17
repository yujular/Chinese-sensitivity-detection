import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import load_args
from dataset import COLDataset
from model import BertBaseModel
from utils import Trainer


def initRandom(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样


MODEL_MAP = {
    'bert-base-linear': BertBaseModel
}

if __name__ == '__main__':
    # 传入config.yml文件的路径作为参数
    args = load_args("config/config.yml")

    # 设置随机种子
    initRandom(args.train['seed'])

    # 创建数据集
    train_dataset = COLDataset(args, datatype='train')
    dev_dataset = COLDataset(args, datatype='dev')
    test_dataset = COLDataset(args, datatype='test')

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.train['batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.train['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.train['batch_size'], shuffle=False)

    # 加载模型
    # model = MODEL_MAP[args.model['model_type']](args)
    model = BertBaseModel(args)
    model.to(args.train['device'])
    if args.train['device'] == 'cuda' and args.train['n_gpu'] > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True)

    # 训练模型
    train_dataloader = {'train': train_loader, 'dev': dev_loader}
    trainer = Trainer(args, model, train_dataloader)
    best_model_state_dict = trainer.train()
    # 保存模型
    torch.save(best_model_state_dict,
               os.path.join(args.train['model_save_path'], args.model['model_name'] + 'model.bin'))
