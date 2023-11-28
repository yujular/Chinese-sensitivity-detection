import os

from torch.utils.data import DataLoader

from config import load_args
from dataset import COLDataset
from model import Model
from utils import get_prediction, evaluate_subcategory
from utils import load_torch_model, initRandom

if __name__ == '__main__':
    # 传入config.yml文件的路径作为参数
    args = load_args("config/config.yml")
    print(args)

    # 设置随机种子
    initRandom(args.train['seed'])

    print("Loading test data...")
    # 创建数据集
    dev_dataset = COLDataset(args, datatype='dev')
    test_dataset = COLDataset(args, datatype='test')

    # 创建数据加载器
    dev_loader = DataLoader(dev_dataset, batch_size=args.train['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.train['batch_size'], shuffle=False)

    # 加载模型
    model = Model(args).get_model()

    model_path = os.path.join(args.train['model_out_path'],
                              'class-' + str(args.dataset['class_num']),
                              args.model['model_name'] + 'model.bin')
    print("Loading model from {}...".format(model_path))
    model = load_torch_model(model=model, model_path=model_path, device=args.train['device'], strict=True)

    # 开发集预测
    # ans, label = get_prediction(model, dev_loader, args.train['device'], test=False)
    # acc, f1 = calculate_accuracy_f1(label, ans, class_num=args.dataset['class_num'])
    # print("Dev acc: {}, f1: {}".format(acc, f1))

    # 测试集预测
    ans, label, fine_grained_label = get_prediction(model, test_loader, args.train['device'], test=True)
    test_evaluate = evaluate_subcategory(label, ans, fine_grained_label, class_num=args.dataset['class_num'],
                                         average='macro')
    print("Test acc: {}, f1: {}, acc_I: {}, f1_I: {}, acc_G: {}, f1_G: {}, acc_anti: {}, f1_anti: {}, acc_non_offen: "
          "{}, f1_non_offen: {}".format(*test_evaluate)
          )
