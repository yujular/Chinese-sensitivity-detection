import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from slugify import slugify


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


def plot_confusion_matrix(true_labels, predicted_labels, title='Confusion Matrix',
                          labels=None, normalize=False, save_path=None):
    """
    Function to plot a confusion matrix.
    :param true_labels: List of true labels
    :param predicted_labels: List of predicted labels
    :param title:  Optional; Title of the plot
    :param labels: Optional; List of label names
    :param normalize: Optional; If True, shows the proportions instead of numbers
    :param save_path: Optional; Path to save the plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)

    # Normalize the confusion matrix if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plotting
    plt.figure(figsize=(10, 8))
    if labels is None:
        labels = range(len(cm))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Save the plot
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = slugify(title) + '.png'
        plt.savefig(os.path.join(save_path, file_name))

    # Display the plot
    plt.show()


def test_plot():
    plot_loss_from_csv('C:/Users/yujular/Desktop/实验结果/BERT_二分类_LMMD/bert-base-chinese-epoch.csv')
    plot_loss_from_csv('C:/Users/yujular/Desktop/实验结果/BERT_二分类_LMMD/bert-base-chinese-step.csv')


def test_plot_confusion_matrix():
    true_labels = [0, 1, 0, 0, 1, 1, 0, 1, 1]
    predicted_labels = [0, 0, 0, 0, 1, 1, 0, 0, 0]
    plot_confusion_matrix(true_labels, predicted_labels, labels=[0, 1], normalize=False)


if __name__ == '__main__':
    str = 'Dev Confusion Matrix'
    print(slugify(str))

    test_plot_confusion_matrix()
