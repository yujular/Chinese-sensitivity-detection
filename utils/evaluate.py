from typing import List

import torch
from sklearn import metrics
from tqdm import tqdm


# def get_prediction(models, data_loader, device, class_num) -> tuple[List[str], List[str]]:
def get_prediction(model, data_loader, device, test=False, transfer=False):
    """Get models prediction on data_loader in device.

    Args:
        :param test: if test, extra return fine-grained-label
        :param model: models to be evaluate
        :param data_loader: data loader
        :param device: device to evaluate
    """

    model.eval()
    outputs = torch.tensor([], dtype=torch.float).to(device)
    labels = torch.tensor([], dtype=torch.long).to(device)

    fine_grained_labels = torch.tensor([], dtype=torch.long).to(device)

    for batch in tqdm(data_loader, desc='Evaluation', ncols=80):
        batch = tuple(t.to(device) for t in batch.values())
        if test:
            input_ids, input_mask, label, fine_grained_label = batch
            fine_grained_labels = torch.cat((fine_grained_labels, fine_grained_label),
                                            dim=0)
        else:
            input_ids, input_mask, label = batch
        with torch.no_grad():
            if transfer:
                outs = model.predict({'input_ids': input_ids, 'attention_mask': input_mask})
            else:
                outs = model(input_ids, input_mask)
        outputs = torch.cat((outputs, outs), dim=0)
        labels = torch.cat((labels, label), dim=0)

    # 根据输出分类

    answer = outputs.argmax(dim=1).tolist()
    if test:
        return answer, labels.to('cpu').numpy().tolist(), fine_grained_labels.to('cpu').numpy().tolist()
    else:
        return answer, labels.to('cpu').numpy().tolist()


def get_trans_prediction(model, data_loader, device, test=False):
    model.eval()
    outputs = torch.tensor([], dtype=torch.float).to(device)
    labels = torch.tensor([], dtype=torch.long).to(device)

    # COLD, test
    fine_grained_labels = torch.tensor([], dtype=torch.long).to(device)

    for batch in tqdm(data_loader, desc='Evaluation', ncols=80):
        batch = tuple(t.to(device) for t in batch.values())
        if test:
            input_ids, input_mask, label, fine_grained_label = batch
            fine_grained_labels = torch.cat((fine_grained_labels, fine_grained_label),
                                            dim=0)
        else:
            input_ids, input_mask, label = batch
        with torch.no_grad():
            inputs = {'input_ids': input_ids, 'attention_mask': input_mask}
            outs = model.predict(inputs)
        outputs = torch.cat((outputs, outs), dim=0)
        labels = torch.cat((labels, label), dim=0)
    answer = outputs.argmax(dim=1).tolist()
    if test:
        return answer, labels.to('cpu').numpy().tolist(), fine_grained_labels.to('cpu').numpy().tolist()
    else:
        return answer, labels.to('cpu').numpy().tolist()


def calculate_accuracy_f1(
        labels: List[str], predicts: List[str], class_num=2, average='binary') -> tuple:
    """Calculate accuracy and f1 score.

    Args:
        :param class_num: number of classes
        :param predicts: models prediction
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


def evaluate_subcategory(
        labels: List[str], predicts: List[str], fine_grained_labels: List[str], class_num=2, average='macro') -> tuple:
    """Calculate accuracy and f1 score.

    Args:
        :param fine_grained_labels: label of fine-grained
        :param class_num: number of classes
        :param predicts: models prediction
        :param labels: ground truth
        :param average: average method, macro or micro

    Returns:
        accuracy, f1 score

    """
    f1, acc = calculate_accuracy_f1(labels, predicts, class_num, average)
    att_I_predicts, att_I_labels, att_G_predicts, att_G_labels = [], [], [], []
    anti_predicts, anti_labels, non_offen_predicts, non_offen_labels = [], [], [], []
    for i in range(len(fine_grained_labels)):
        if fine_grained_labels[i] == 1:
            att_I_predicts.append(predicts[i])
            att_I_labels.append(labels[i])
        elif fine_grained_labels[i] == 2:
            att_G_predicts.append(predicts[i])
            att_G_labels.append(labels[i])
        elif fine_grained_labels[i] == 3:
            anti_predicts.append(predicts[i])
            anti_labels.append(labels[i])
        elif fine_grained_labels[i] == 0:
            non_offen_predicts.append(predicts[i])
            non_offen_labels.append(labels[i])
    acc_I, f1_I = calculate_accuracy_f1(att_I_labels, att_I_predicts, class_num, average)
    acc_G, f1_G = calculate_accuracy_f1(att_G_labels, att_G_predicts, class_num, average)
    acc_anti, f1_anti = calculate_accuracy_f1(anti_labels, anti_predicts, class_num, average)
    acc_non_offen, f1_non_offen = calculate_accuracy_f1(non_offen_labels, non_offen_predicts, class_num, average)
    return acc, f1, acc_I, f1_I, acc_G, f1_G, acc_anti, f1_anti, acc_non_offen, f1_non_offen

# def test_evaluate():
#     args = load_args("config/config.yml")
#     print(args)
#     models = BertBaseModel(args)
#     dev_dataset = COLDataset(args, datatype='dev')
#     dev_loader = DataLoader(dev_dataset, batch_size=args.train['batch_size'], shuffle=False)
#
#     outputs = torch.tensor([], dtype=torch.float)
#     labels = torch.tensor([], dtype=torch.long)
#
#     for i in range(5):
#         input_ids, input_mask, label = dev_loader.__iter__().__next__().values()
#         with torch.no_grad():
#             outs = models(input_ids, input_mask)
#         outputs = torch.cat((outputs, outs), dim=0)
#         labels = torch.cat((labels, label), dim=0)
#
#     answer = outputs.argmax(dim=1).tolist()
#     labels = labels.numpy().tolist()
#
#     acc, f = calculate_accuracy_f1(labels, answer, class_num=2)
#
#
# if __name__ == '__main__':
#     test_evaluate()
