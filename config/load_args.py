import configargparse
import torch.cuda
import yaml

from utils import str2bool


def load_args(config_file):
    # 从config.yml文件中加载参数
    with open(config_file, 'r', encoding='utf-8') as config_stream:
        config = yaml.safe_load(config_stream)

    if torch.cuda.is_available():
        config['train']['device'] = 'cuda'
        config['train']['n_gpu'] = torch.cuda.device_count()
    else:
        config['train']['device'] = 'cpu'
        config['train']['n_gpu'] = 0

    # 创建一个ArgumentParser对象
    parser = argparse.ArgumentParser(description='Args base on config.yml')

    # 遍历配置文件中的参数并添加到解析器中
    for key, value in config.items():
        # 如果value是列表，则将其转换为字符串
        if isinstance(value, list):
            value = ','.join(value)
        parser.add_argument(f'--{key}', default=value)

    # 解析命令行参数并返回
    return parser.parse_args()


def add_input_args(parser):
    # global config
    parser.add_argument('--seed', type=int, default=3407, help='Random seed.')
    parser.add_argument('--num_workers', type=int, default=0)

    # model config
    parser.add_argument('--transfer', type=str2bool, default=False, help='Whether to use transfer learning.')
    parser.add_argument('--pretrained', type=str2bool, default=False, help='Whether to use pretrained model.')
    parser.add_argument('--model_path', type=str, default='models', help='Model root path.')
    parser.add_argument('--model', '-m', type=str, default='bert-base-chinese', help='backbone network')
    parser.add_argument('--cls', '-c', type=str, default='linear', help='classifier')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--hugging_face', type=str2bool, default=False,
                        help='whether to use hugging face to load models')

    # data config
    parser.add_argument('--data_dir', type=str, default='data', help='Data root path.')
    parser.add_argument('--train_data', type=str, default='COLD', help='Training dataset name.')
    parser.add_argument('--dev_data', type=str, default='COLD', help='Dev dataset name.')
    parser.add_argument('--test_data', type=str, default='COLD', help='Test dataset name.')

    parser.add_argument('--source_data', type=str, default='KOLD', help='Source dataset name.')
    parser.add_argument('--target_data', type=str, default='COLD', help='Target dataset name.')
    parser.add_argument('--class_num', type=int, default=2, help='Number of classes.')
    parser.add_argument('--max_length', type=int, default=128, help='Max length of input sequence.')

    # train config
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of batch size to train.')
    parser.add_argument('--freeze', type=str2bool, default=False, help='Freeze the encoder weights.')
    parser.add_argument('--clip_grad', type=str2bool, default=False, help='Clip the gradient norm.')
    parser.add_argument('--max_grad_norm', type=float, default=1.0e-3, help='Max gradient norm.')

    # optimizer config
    parser.add_argument('--optimizer', '-o', type=str, default='AdamW', help='Optimizer.')
    parser.add_argument('--lr', type=float, default=5.0e-5, help='Initial learning rate.')

    # scheduler config
    parser.add_argument('--scheduler', '-s', type=str, default='warmuplinear', help='Scheduler.')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warm up steps.')


def get_parser():
    parser = configargparse.ArgumentParser(
        description="Transfer learning for OLD config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # get parser from config file
    # priority: command line > config file > default
    parser.add_argument("--config", is_config_file=True, help="config file path")
    add_input_args(parser)

    parser_args = parser.parse_args()
    # device setting
    setattr(parser_args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
    setattr(parser_args, 'n_gpu', torch.cuda.device_count())
    return parser_args


if __name__ == "__main__":
    # 传入config.yml文件的路径作为参数
    args = get_parser()

    # 使用args中的参数进行你的操作
    print(args)
    print(args.transfer)
