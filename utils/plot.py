import pandas as pd
from matplotlib import pyplot as plt


def plot_loss_from_csv(csv_file, save_path=None):
    # 读取CSV文件
    data = pd.read_csv(csv_file)

    # 第一列通常是序号（epoch或step）
    index_column = data.columns[0]

    # 使用seaborn风格绘图
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 循环绘制每一列与第一列的曲线
    for column in data.columns[1:]:
        plt.plot(data[index_column], data[column], label=column)

    # 设置图表标题和轴标签
    ax1.set_xlabel(index_column)
    ax1.set_title('Training Metrics Over ' + index_column)

    # 添加图例
    ax1.legend(loc='upper left')

    # 保存或显示图像
    if save_path:
        plt.show()
        plt.savefig(save_path)
    else:
        plt.show()


def test_plot():
    plot_loss_from_csv('C:/Users/yujular/Desktop/实验结果/BERT_二分类_LMMD/bert-base-chinese-epoch.csv')
    plot_loss_from_csv('C:/Users/yujular/Desktop/实验结果/BERT_二分类_LMMD/bert-base-chinese-step.csv')


if __name__ == '__main__':
    test_plot()
