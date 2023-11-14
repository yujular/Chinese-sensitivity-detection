import argparse

import yaml


def load_args(config_file):
    # 从config.yml文件中加载参数
    with open(config_file, 'r', encoding='utf-8') as config_stream:
        config = yaml.safe_load(config_stream)

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


if __name__ == "__main__":
    # 传入config.yml文件的路径作为参数
    args = load_args("config.yml")

    # 使用args中的参数进行你的操作
    print(args)
