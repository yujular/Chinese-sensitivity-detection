import os
import torch
from torch.utils.data import DataLoader

from config import load_args
from dataset import load_dataset
from models import TransferNet, get_trained_model
from utils import initRandom, TransferTrainer, save_model

if __name__ == '__main__':
    # 传入config.yml文件的路径作为参数
    args = load_args("config/config.yml")
    print(args)

    # 设置随机种子
    initRandom(args.train['seed'])

    # 加载数据
    print("Loading data...")

    COLD_train = load_dataset('COLD', args, datatype='train')
    COLD_dev = load_dataset('COLD', args, datatype='dev')
    transfer_train = load_dataset(args.train['transfer_dataset'], args, datatype='train')

    COLD_train_loader = DataLoader(COLD_train, batch_size=args.train['batch_size'], shuffle=True)
    COLD_dev_loader = DataLoader(COLD_dev, batch_size=args.train['batch_size'], shuffle=False)
    transfer_loader = DataLoader(transfer_train, batch_size=args.train['batch_size'], shuffle=True)

    # 加载基础模型
    print("Loading models...")
    model = get_trained_model(args, transfer=False, pretrained=True)
    # 迁移模型
    model.to(args.train['device'])
    trans_model = TransferNet(args, model, transfer_loss=args.train['transfer_loss']).to(args.train['device'])
    # 是否采用平行计算
    if args.train['device'] == 'cuda' and args.train['n_gpu'] > 1:
        trans_model = torch.nn.parallel.DistributedDataParallel(
            trans_model, find_unused_parameters=True)

    # 迁移训练
    print("Transfer training...")
    target_dataloaders = {'train': COLD_train_loader, 'dev': COLD_dev_loader}
    trainer = TransferTrainer(args, trans_model, transfer_loader, target_dataloaders)

    best_model_state_dict, best_epoch = trainer.train()

    # 保存模型

    print("best epoch: {}".format(best_epoch))
    model_save_path = os.path.join(args.train['model_out_path'],
                                   'transfer',
                                   args.train['transfer_dataset'] + '-' + args.train['transfer_loss'],
                                   'class-' + str(args.dataset['class_num']))
    model_save_filename = args.model['model_name'] + 'models.bin'
    save_model(model_dict=best_model_state_dict, path=model_save_path, filename=model_save_filename)
