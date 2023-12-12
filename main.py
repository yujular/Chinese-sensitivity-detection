from config import get_parser
from train import Trainer
from utils import initRandom

if __name__ == '__main__':
    # 获取参数
    args = get_parser()
    print(args)
    # 设置随机种子
    initRandom(args.seed)

    # 训练模型
    trainer = Trainer(args)

    best_epoch = trainer.train()
    print("Best epoch: {}".format(best_epoch))
    trainer.test()

    # os.system("/usr/bin/shutdown")
