from config import load_args
from dataset import COLDataset

if __name__ == '__main__':
    # 传入config.yml文件的路径作为参数
    args = load_args("config/config.yml")
    # 创建数据集
    train_dataset = COLDataset(args, datatype='train')
    dev_dataset = COLDataset(args, datatype='dev')
    test_dataset = COLDataset(args, datatype='test')
