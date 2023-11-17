from typing import List

import torch
from sklearn import metrics
from tqdm import tqdm


def get_prediction(model, data_loader, device, class_num) -> tuple[List[str], List[str]]:
    """Get model prediction on data_loader in device.

    Args:
        :param model: model to be evaluate
        :param data_loader: data loader
        :param device: device to evaluate
        :param class_num: number of classes
    """
    LABELS = [str(i) for i in range(class_num)]

    model.eval()
    outputs = torch.tensor([], dtype=torch.float).to(device)
    labels = torch.tensor([], dtype=torch.long).to(device)
    for batch in tqdm(data_loader, desc='Evaluation', ncols=80):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, label = batch
        with torch.no_grad():
            outs = model(input_ids, input_mask)
        outputs = torch.cat((outputs, outs), dim=0)
        labels = torch.cat((labels, label), dim=0)

    # 根据输出分类
    answer_list = []
    for i in range(0, len(outputs), len(LABELS)):
        logits = outputs[i:i + len(LABELS)]
        answer = int(torch.argmax(logits))
        answer_list.append(LABELS[answer])
    return answer_list, labels.numpy().tolist()


def calculate_accuracy_f1(
        labels: List[str], predicts: List[str], class_num=2, average='macro') -> tuple:
    """Calculate accuracy and f1 score.

    Args:
        :param class_num: number of classes
        :param predicts: model prediction
        :param labels: ground truth
        :param average: average method, macro or micro

    Returns:
        accuracy, f1 score

    """
    LABELS = [str(i) for i in range(class_num)]
    return metrics.accuracy_score(labels, predicts), \
        metrics.f1_score(
            labels, predicts,
            labels=LABELS, average=average)
