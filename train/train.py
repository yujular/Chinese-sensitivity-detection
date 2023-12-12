import gc
import os

import torch
from torch import nn
from torch.optim import Adam, AdamW
from tqdm import trange
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, \
    get_constant_schedule

from dataset import get_dataloaders
from models import Model
from utils import get_prediction, calculate_accuracy_f1, get_trans_prediction, evaluate_subcategory
from utils import step_log, get_csv_logger, epoch_log


class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = Model(args).get_model(False)
        self.data_loader = get_dataloaders(args)

        if self.args.transfer:
            len_source_loader = len(self.data_loader['source_train'])
            len_target_loader = len(self.data_loader['target_train'])
            n_batch = min(len_source_loader, len_target_loader)
            if self.args.transfer_loss == 'lmmd':
                n_batch = n_batch - 1
            self.args.n_batch = n_batch
        else:
            self.args.n_batch = len(self.data_loader['train'])
        self.args.num_train_steps = self.args.epochs * self.args.n_batch

        self.optimizer = self._get_optimizer(self.args.optimizer)
        self.scheduler = self._get_scheduler(self.args.scheduler)
        self.criterion = nn.CrossEntropyLoss()
        self.step_logger, self.epoch_logger = self._init_logger()

    def _get_optimizer(self, optimizer_name):
        """Get optimizer for different models.
               Args:
                   optimizer_name: str, optimizer name, 'adam' or 'adamw'
               Returns:
                   optimizer
        """
        if optimizer_name.casefold() == 'adamw':
            # bias, gamma, beta 参数不参与训练, 其余权重0.01
            # no_decay = ['bias', 'gamma', 'beta']
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_parameters = [
                {'params': [p for n, p in self.model.named_parameters()
                            if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in self.model.named_parameters()
                            if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}]
            optimizer = AdamW(optimizer_parameters,
                              lr=self.args.lr,
                              betas=(0.9, 0.999),
                              weight_decay=1e-8)
        else:
            optimizer = Adam(self.model.named_parameters(), lr=self.args.lr)
        return optimizer

    def _get_scheduler(self, scheduler_name):
        """Get scheduler for different models.
               Args:
                   scheduler_name: str, scheduler name, 'warmuplinear' or 'warmupconstant' or 'constant'
               Returns:
                   scheduler
        """
        if scheduler_name.casefold() == 'warmuplinear':
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.num_warmup_steps,
                num_training_steps=self.args.num_train_steps)
        elif scheduler_name.casefold() == 'warmupconstant':
            scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.num_warmup_steps)
        elif scheduler_name.casefold() == 'constant':
            scheduler = get_constant_schedule(
                self.optimizer)
        else:
            scheduler = None

        return scheduler

    def _init_logger(self):
        """Initialize logger.
        """
        log_path = os.path.join(self.args.output_dir,
                                self.args.log_path)
        step_path_name = os.path.join(log_path,
                                      self.args.model + '-' + self.args.bottleneck + '-step.csv')
        epoch_path_name = os.path.join(log_path,
                                       self.args.model + '-' + self.args.bottleneck + '-epoch.csv')

        if not os.path.exists(log_path):
            os.makedirs(log_path)

        step_logger = get_csv_logger(step_path_name, title='step,loss')
        epoch_logger = get_csv_logger(epoch_path_name, title='epoch,train_acc,train_f1,dev_acc,dev_f1')

        return step_logger, epoch_logger

    def _evaluate(self, dataset_type):
        """Evaluate models and get acc and f1 score.
        Args:
            dataset_type: str, 'train' or 'dev'

        Returns:
            accuracy and f1 score
        """
        if self.args.transfer:
            if dataset_type == 'train':
                if self.args.pseudo:
                    domain = 'source'
                else:
                    domain = 'target'
                dataset_type = domain + '_train'
            predictions, labels = get_trans_prediction(model=self.model, data_loader=self.data_loader[dataset_type],
                                                       device=self.args.device)
        else:
            predictions, labels = get_prediction(model=self.model, data_loader=self.data_loader[dataset_type],
                                                 device=self.args.device)
        accuracy, f1 = calculate_accuracy_f1(labels, predictions, class_num=self.args.class_num)
        return accuracy, f1

    def _save_model(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), os.path.join(path, filename))

    def train(self):
        """Train models on train set and evaluate on train and valid set.

        Returns:
            state dict of the best models with the highest valid f1 score
        """

        best_dev_f1, global_step = 0, 0
        best_epoch = 0

        tqdm_epoch = trange(self.args.epochs, desc='Epoch', ncols=120)
        for epoch in tqdm_epoch:
            self.model.train()

            iter_train = {}
            if self.args.transfer:
                iter_train['source'] = iter(self.data_loader['source_train'])
                iter_train['target'] = iter(self.data_loader['target_train'])
            else:
                iter_train['train'] = iter(self.data_loader['train'])

            tqdm_train = trange(self.args.n_batch, ncols=100)
            for _ in tqdm_train:
                transfer_loss = None
                if self.args.transfer:
                    # 迁移损失
                    data_source_id, data_source_mask, label_source = next(iter_train['source']).values()
                    data_target_id, data_target_mask, label_target = next(iter_train['target']).values()
                    data_source_id, data_source_mask, label_source = (data_source_id.to(self.args.device),
                                                                      data_source_mask.to(self.args.device),
                                                                      label_source.to(self.args.device))
                    data_target_id, data_target_mask, label_target = (data_target_id.to(self.args.device),
                                                                      data_target_mask.to(self.args.device),
                                                                      label_target.to(self.args.device))
                    # 拼接BERT模型输入
                    data_source = {'input_ids': data_source_id, 'attention_mask': data_source_mask}
                    data_target = {'input_ids': data_target_id, 'attention_mask': data_target_mask}
                    if self.args.pseudo:
                        source_clf_loss, transfer_loss = self.model(data_source, data_target, label_source)
                        loss = source_clf_loss + self.args.transfer_loss_weight * transfer_loss
                    else:
                        source_clf_loss, target_clf_loss, transfer_loss = self.model(data_source, data_target,
                                                                                     label_source, label_target)
                        loss = (source_clf_loss * self.args.source_cls_weight + target_clf_loss
                                + self.args.transfer_loss_weight * transfer_loss)
                else:
                    # 非迁移损失
                    input_ids, input_mask, label = next(iter_train['train']).values()
                    input_ids, input_mask, label = (input_ids.to(self.args.device),
                                                    input_mask.to(self.args.device),
                                                    label.to(self.args.device))
                    # 模型输出
                    model_inputs = {'input_ids': input_ids, 'attention_mask': input_mask}
                    model_out = self.model(model_inputs)
                    # 计算损失
                    loss = self.criterion(model_out, label)

                self.optimizer.zero_grad()
                loss.backward()

                if self.args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm)

                self.optimizer.step()
                self.scheduler.step()
                global_step += 1
                if self.args.transfer:

                    tqdm_train.set_description('Train loss: {:.6f}, transfer loss: {:.6f}'.format(
                        loss.item(), transfer_loss.item()), refresh=False)
                else:
                    tqdm_train.set_description('Train loss: {:.6f}'.format(loss.item()), refresh=False)

                step_log(global_step, loss, self.step_logger)

            # 每个epoch结束后在train和dev上评估
            train_accuracy, train_f1 = self._evaluate('train')
            dev_accuracy, dev_f1 = self._evaluate('dev')
            epoch_log(epoch + 1, (train_accuracy, train_f1, dev_accuracy, dev_f1), self.epoch_logger)
            tqdm_epoch.set_description(
                'Epoch: {:d}, train_acc: {:.6f}, train_f1: {:.6f}, '
                'valid_acc: {:.6f}, valid_f1: {:.6f}, '.format(
                    epoch, train_accuracy, train_f1, dev_accuracy, dev_f1))

            # 保存模型
            if self.args.transfer:
                model_path = os.path.join(self.args.output_dir,
                                          self.args.model_out_path,
                                          'transfer',
                                          self.args.source_data + '-' + self.args.transfer_loss
                                          + '-class-' + str(self.args.class_num))
            else:
                model_path = os.path.join(self.args.output_dir,
                                          self.args.model_out_path,
                                          'base',
                                          self.args.train_data + '-class-' + str(self.args.class_num))
            if self.args.save_epoch_model:
                model_name = self.args.model + '-' + str(epoch + 1) + '.bin'
                self._save_model(path=model_path, filename=model_name)

            if dev_f1 > best_dev_f1:
                model_name = self.args.model + '-models.bin'
                self._save_model(path=model_path, filename=model_name)
                best_dev_f1 = dev_f1
                best_epoch = epoch + 1
                print('Save model in {}'.format(os.path.join(model_path, model_name)))
            # 清除内存
            if self.args.device == 'cuda':
                gc.collect()
                torch.cuda.empty_cache()

        return best_epoch

    def test(self):
        # 加载保存模型
        self.model = Model(self.args).get_model(True)
        # 测试模型
        # 开发集预测
        ans, label = get_prediction(self.model, self.data_loader['dev'], self.args.device,
                                    test=False, transfer=self.args.transfer)
        acc, f1 = calculate_accuracy_f1(label, ans, class_num=self.args.class_num,
                                        average='macro' if self.args.class_num > 2 else None)
        print("Dev acc: {}, f1: {}".format(acc, f1))
        # 测试集预测
        ans, label, fine_grained_label = get_prediction(self.model, self.data_loader['test'],
                                                        self.args.device, test=True,
                                                        transfer=self.args.transfer)
        test_evaluate = evaluate_subcategory(label, ans, fine_grained_label,
                                             class_num=self.args.class_num,
                                             average='macro' if self.args.class_num > 2 else None)
        print(
            "Test acc: {}, f1: {},\n acc_I: {}, f1_I: {},\n acc_G: {}, f1_G: {},\n acc_anti: {}, f1_anti: {},"
            "\n acc_non_offen:{}, f1_non_offen: {}\n".format(*test_evaluate)
        )
        print('--------------------end----------------')
