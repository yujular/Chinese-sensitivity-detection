import os
import torch

from config import load_args
from dataset import get_dataloader
from models import BertBaseModel, Model
from utils import Trainer, initRandom, save_model

MODEL_MAP = {
    'bert-base-linear': BertBaseModel
}

if __name__ == '__main__':
    # 传入config.yml文件的路径作为参数
    args = load_args("config/config.yml")
    print(args)

    # 设置随机种子
    initRandom(args.train['seed'])

    print("Loading data...")
    # 创建数据集
    data_loader = get_dataloader(args.dataset['dataset_name'], args, ['train', 'dev'])

    # 加载模型
    model = Model(args).get_model()
    model.to(args.train['device'])
    if args.train['device'] == 'cuda' and args.train['n_gpu'] > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True)

    # 训练模型

    trainer = Trainer(args, model, data_loader)
    best_model_state_dict, best_epoch = trainer.train()
    # 保存模型
    print("best epoch: {}".format(best_epoch))
    model_path = os.path.join(args.train['model_out_path'],
                              'class-' + str(args.dataset['class_num']))
    model_name = args.model['model_name'] + 'models.bin'
    save_model(model_dict=best_model_state_dict, path=model_path, filename=model_name)
