import os

from config import get_parser
from train import Trainer
from utils import initRandom, calculate_average_std

if __name__ == '__main__':
    # 获取参数
    args = get_parser()
    print(args)
    # 设置随机种子
    if args.random_num == 1:
        initRandom(args.seed)

    test_result = []
    # 训练模型
    for i in range(args.random_num):
        print('--------------------random_num: {}----------------'.format(i))
        seed = int.from_bytes(os.urandom(4), byteorder='little')
        initRandom(seed)

        trainer = Trainer(args)

        if args.train:
            best_epoch = trainer.train()
            print("Best epoch: {}".format(best_epoch))
        if args.test:
            test_result.append(trainer.test())

    mean_result = calculate_average_std(test_result)
    print(mean_result)

    print('--------------------end----------------')
    # os.system("/usr/bin/shutdown")
