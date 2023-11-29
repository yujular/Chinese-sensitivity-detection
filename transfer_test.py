from config import load_args
from dataset import get_dataloader
from model import get_trained_model
from utils import initRandom, calculate_accuracy_f1, get_prediction, evaluate_subcategory

if __name__ == '__main__':
    # 传入config.yml文件的路径作为参数
    args = load_args("config/config.yml")
    print(args)

    # 设置随机种子
    initRandom(args.train['seed'])

    # 加载数据
    print("Loading test data...")
    dataloader = get_dataloader(args.dataset['dataset_name'], args, datatypes=['dev', 'test'])

    # 加载模型
    model = get_trained_model(args, transfer=True)

    # 开发集预测
    ans, label = get_prediction(model, dataloader['dev'], args.train['device'], test=False, transfer=True)
    acc, f1 = calculate_accuracy_f1(label, ans, class_num=args.dataset['class_num'])
    print("Dev acc: {}, f1: {}".format(acc, f1))

    # 测试集预测
    ans, label, fine_grained_label = get_prediction(model, dataloader['test'], args.train['device'],
                                                    test=True, transfer=True)
    test_evaluate = evaluate_subcategory(label, ans, fine_grained_label, class_num=args.dataset['class_num'],
                                         average='macro')
    print("Test acc: {}, f1: {}, acc_I: {}, f1_I: {}, acc_G: {}, f1_G: {}, acc_anti: {}, f1_anti: {}, acc_non_offen: "
          "{}, f1_non_offen: {}".format(*test_evaluate))
