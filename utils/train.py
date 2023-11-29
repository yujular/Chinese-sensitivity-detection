import gc
import os
from copy import deepcopy

import torch
from torch import nn
from torch.optim import Adam, AdamW
from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, \
    get_constant_schedule

from utils import save_model
from utils.evaluate import get_prediction, calculate_accuracy_f1, get_trans_prediction
from utils.logger import step_log, get_csv_logger, epoch_log
from utils.utils import AverageMeter


class Trainer:
    def __init__(self, args, model, data_loader):
        """Initialize trainer with args.
        Initialize optimizer, scheduler, criterion.

        Args:
            args:
                args.train['num_epoch']: number of epochs
                args.train['batch_size']: batch size
                args.train['lr']: learning rate
                args.train['num_warmup_steps']: number of warmup steps
                args.train['scheduler']: scheduler name, 'warmuplinear' or 'warmupconstant'
            model: model to be evaluated
            data_loader: dict of torch.utils.data.DataLoader, including 'train' and 'dev'
        """
        self.model = model
        self.args = args
        self.data_loader = data_loader

        self.args.train['num_train_steps'] = args.train['num_epoch'] * len(data_loader['train'])
        self.optimizer = self._get_optimizer(self.args.train['optimizer'])
        self.scheduler = self._get_scheduler(self.args.train['scheduler'])
        self.criterion = nn.CrossEntropyLoss()
        self.step_logger, self.epoch_logger = self._init_logger()

    def _init_logger(self):
        """Initialize logger.
        """
        step_path_name = os.path.join(self.args.train['log_path'], self.args.model['model_name'] + '-step.csv')
        epoch_path_name = os.path.join(self.args.train['log_path'], self.args.model['model_name'] + '-epoch.csv')

        step_logger = get_csv_logger(step_path_name, title='step,loss')
        epoch_logger = get_csv_logger(epoch_path_name, title='epoch,train_acc,train_f1,dev_acc,dev_f1')

        return step_logger, epoch_logger

    def _get_optimizer(self, optimizer_name):
        """Get optimizer for different models.
               Args:
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
                              lr=self.args.train['lr'],
                              betas=(0.9, 0.999),
                              weight_decay=1e-8)
        else:
            optimizer = Adam(self.model.named_parameters(), lr=self.args.train['lr'])
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
                num_warmup_steps=self.args.train['num_warmup_steps'],
                num_training_steps=self.args.train['num_train_steps'])
        elif scheduler_name.casefold() == 'warmupconstant':
            scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.train['num_warmup_steps'])
        elif scheduler_name.casefold() == 'constant':
            scheduler = get_constant_schedule(
                self.optimizer)
        else:
            scheduler = None

        return scheduler

    def save_model(self, path, filename):
        """Save model to filename.
        Args:
            :param filename: str, filename to save model
            :param path: str, path to save model
        """
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, filename)
        torch.save(self.model.state_dict(), filename)

    def _evaluate(self, dataset_type):
        """Evaluate model and get acc and f1 score.
        Args:
            dataset_type: str, 'train' or 'dev'

        Returns:
            accuracy and f1 score
        """
        predictions, labels = get_prediction(model=self.model, data_loader=self.data_loader[dataset_type],
                                             device=self.args.train['device'])
        accuracy, f1 = calculate_accuracy_f1(labels, predictions, class_num=self.args.dataset['class_num'])
        return accuracy, f1

    def train(self):
        """Train model on train set and evaluate on train and valid set.

        Returns:
            state dict of the best model with the highest valid f1 score
        """

        best_model_state_dict, best_dev_f1, global_step = None, 0, 0
        best_epoch = 0

        tqdm_epoch = trange(self.args.train['num_epoch'], desc='Epoch', ncols=120)
        for epoch in tqdm_epoch:
            self.model.train()

            tqdm_train = tqdm(self.data_loader['train'], ncols=80)
            for step, batch in enumerate(tqdm_train):
                # 获取输入
                batch = tuple(t.to(self.args.train['device']) for t in batch.values())
                input_ids, input_mask, label = batch
                # 模型输出
                model_out = self.model(input_ids, input_mask)
                # 计算损失
                loss = self.criterion(model_out, label)
                if self.args.train['gradient_accumulation_steps'] > 1:
                    loss = loss / self.args.train['gradient_accumulation_steps']

                self.optimizer.zero_grad()
                loss.backward()

                if (step + 1) % self.args.train['gradient_accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.train['max_grad_norm'])
                    self.optimizer.step()
                    self.scheduler.step()
                    global_step += 1
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
            self.save_model(os.path.join(
                self.args.train['model_out_path'],
                'class-' + str(self.args.dataset['class_num'])),
                self.args.model['model_name'] + '-' + str(epoch + 1) + '.bin')

            if dev_f1 > best_dev_f1:
                best_model_state_dict = deepcopy(self.model.state_dict())
                best_dev_f1 = dev_f1
                best_epoch = epoch + 1
            # 清除内存
            if self.args.train['device'] == 'cuda':
                gc.collect()
                torch.cuda.empty_cache()

        return best_model_state_dict, best_epoch


class TransferTrainer(Trainer):
    def __init__(self, args, model, data_loader_source, data_loader_target):
        super().__init__(args, model, data_loader_target)
        self.data_loader_source = data_loader_source

    def train(self):
        len_source_loader = len(self.data_loader_source)
        len_target_loader = len(self.data_loader['train'])
        n_batch = min(len_source_loader, len_target_loader)

        best_model_state_dict, best_dev_f1, global_step = None, 0, 0
        best_epoch = 0

        tqdm_epoch = trange(self.args.train['num_epoch'], desc='Epoch', ncols=120)
        for epoch in tqdm_epoch:
            self.model.train()

            train_loss_clf = AverageMeter()
            train_loss_transfer = AverageMeter()
            train_loss_total = AverageMeter()

            # model.epoch_based_processing(n_batch) # daan
            iter_source, iter_target = iter(self.data_loader_source), iter(self.data_loader['train'])

            tqdm_train = tqdm(range(n_batch), ncols=80)
            for step in tqdm_train:
                data_source_id, data_source_mask, label_source = next(iter_source).values()
                data_target_id, data_target_mask, label_target = next(iter_target).values()
                data_source_id, data_source_mask, label_source = (data_source_id.to(self.args.train['device']),
                                                                  data_source_mask.to(self.args.train['device']),
                                                                  label_source.to(self.args.train['device']))
                data_target_id, data_target_mask, label_target = (data_target_id.to(self.args.train['device']),
                                                                  data_target_mask.to(self.args.train['device']),
                                                                  label_target.to(self.args.train['device']))
                # 拼接BERT模型输入
                data_source = {'input_ids': data_source_id, 'attention_mask': data_source_mask}
                data_target = {'input_ids': data_target_id, 'attention_mask': data_target_mask}

                source_clf_loss, target_clf_loss, transfer_loss = self.model(data_source, data_target,
                                                                             label_source, label_target)
                loss = source_clf_loss + target_clf_loss + self.args.train['transfer_loss_weight'] * transfer_loss

                if self.args.train['gradient_accumulation_steps'] > 1:
                    loss = loss / self.args.train['gradient_accumulation_steps']

                self.optimizer.zero_grad()
                loss.backward()

                if (step + 1) % self.args.train['gradient_accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.train['max_grad_norm'])
                    self.optimizer.step()
                    self.scheduler.step()

                    train_loss_clf.update(target_clf_loss.item())
                    train_loss_transfer.update(transfer_loss.item())
                    train_loss_total.update(loss.item())

                    global_step += 1
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
            save_path = os.path.join(self.args.train['model_out_path'],
                                     'transfer',
                                     'class-' + str(self.args.dataset['class_num']))
            model_name = self.args.model['model_name'] + '-' + str(epoch + 1) + '.bin'
            save_model(model_dict=self.model.state_dict(), path=save_path, filename=model_name)

            if dev_f1 > best_dev_f1:
                best_model_state_dict = deepcopy(self.model.state_dict())
                best_dev_f1 = dev_f1
                best_epoch = epoch + 1
            # 清除内存
            if self.args.train['device'] == 'cuda':
                gc.collect()
                torch.cuda.empty_cache()

        return best_model_state_dict, best_epoch

    def _evaluate(self, dataset_type):
        """Evaluate model and get acc and f1 score in target domain.
        Args:
            dataset_type: str, 'train' or 'dev'
        """
        predictions, labels = get_trans_prediction(model=self.model, data_loader=self.data_loader[dataset_type],
                                                   device=self.args.train['device'])
        accuracy, f1 = calculate_accuracy_f1(labels, predictions, class_num=self.args.dataset['class_num'])
        return accuracy, f1
