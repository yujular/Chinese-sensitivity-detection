import os

import configargparse
import torch.cuda

from utils.utils import str2bool


def add_input_args(parser):
    # global config
    parser.add_argument('--seed', type=int, default=3407, help='Random seed.')
    parser.add_argument('--num_workers', type=int, default=0)

    # model config
    parser.add_argument('--transfer', type=str2bool, default=False, help='Whether to use transfer learning.')
    parser.add_argument('--pretrained', type=str2bool, default=False,
                        help='Whether to use pretrained model for transfer.')
    parser.add_argument('--model_path', type=str, default='models', help='Model root path.')
    parser.add_argument('--model', '-m', type=str, default='bert-base-chinese', help='backbone network')
    parser.add_argument('--hidden_size', type=int, default=768, help='hidden size of backbone network')
    parser.add_argument('--bottleneck', '-b', type=str, default='cnn', help='bottleneck network')
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
    parser.add_argument('--save_epoch_model', type=str2bool, default=True, help='Save model per epoch.')

    # optimizer config
    parser.add_argument('--optimizer', '-o', type=str, default='AdamW', help='Optimizer.')
    parser.add_argument('--lr', type=float, default=5.0e-5, help='Initial learning rate.')

    # scheduler config
    parser.add_argument('--scheduler', '-s', type=str, default='warmuplinear', help='Scheduler.')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warm up steps.')
    parser.add_argument('--num_warmup_steps', type=int, default=500, help='Number of warm up steps.')

    # transfer config
    parser.add_argument('--transfer_loss', type=str, default='mmd', help='Transfer loss.')
    parser.add_argument('--use_bottleneck', type=str2bool, default=True, help='Use bottleneck(Transfer Net).')
    parser.add_argument('--bottleneck_width', type=int, default=256, help='Bottleneck width.')
    parser.add_argument('--max_iter', type=int, default=1000, help='Max iteration.')
    parser.add_argument('--transfer_loss_weight', type=float, default=0.5, help='Transfer loss weight.')
    parser.add_argument('--source_cls_weight', type=float, default=1.0, help='Source classification loss weight.')
    parser.add_argument('--pseudo', type=str2bool, default=False, help='Use pseudo label.')

    # outputs config
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output root path.')
    parser.add_argument('--model_out_path', type=str, default='model', help='Model output path.')
    parser.add_argument('--log_path', type=str, default='logger', help='Log output path.')


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

    if parser_args.device == 'cuda' and parser_args.n_gpu > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        setattr(parser_args, 'device', f'cuda:{local_rank}')

    return parser_args


if __name__ == "__main__":
    # 传入config.yml文件的路径作为参数
    args = get_parser()

    # 使用args中的参数进行你的操作
    print(args)
    print(args.transfer)
