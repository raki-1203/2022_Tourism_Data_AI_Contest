import os
import shutil

import numpy as np
import wandb
import albumentations as A
import torch
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer
from albumentations.pytorch.transforms import ToTensorV2

from utils.custom_dataset import MultiModelDataset, NLPDataset, ImageDataset
from utils.custom_model import MultiModalModel, NLPModel, ImageModel
from utils.loss import RDropLoss
from utils.optimizer import MADGRAD


class Trainer:

    def __init__(self, args, logger, df, splits=None):
        self.args = args
        self.logger = logger

        if args.method != 'nlp':
            self.train_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ImageCompression(quality_lower=99, quality_upper=100),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
                A.Resize(args.img_size, args.img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0,
                            always_apply=False, p=1.0),
                ToTensorV2(),
            ])
            self.valid_transform = A.Compose([
                A.ImageCompression(quality_lower=99, quality_upper=100),
                A.Resize(args.img_size, args.img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0,
                            always_apply=False, p=1.0),
                ToTensorV2(),
            ])
        elif args.method != 'image':
            self.tokenizer = AutoTokenizer.from_pretrained(args.text_model_name_or_path)

        # load dataset setting
        self.make_datasets(splits, df, is_train=args.is_train)

        if args.method == 'multimodal':
            self.model = MultiModalModel(self.args)
        elif args.method == 'nlp':
            self.model = NLPModel(self.args)
        elif args.method == 'image':
            self.model = ImageModel(self.args)
        else:
            raise NotImplementedError('args.method 를 잘 선택 해주세요.')
        self.model.to(args.device)

        self.supervised_loss = nn.CrossEntropyLoss()
        self.rdrop_loss = RDropLoss()

        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

        self.train_loss = AverageMeter()
        self.train_acc = AverageMeter()

        self.best_valid_f1_score = 0
        self.best_model_folder = None

    def make_datasets(self, splits, df, is_train):
        if is_train:
            train_idx, valid_idx = splits
            train_df = df.iloc[train_idx]
            valid_df = df.iloc[valid_idx]

            if self.args.method == 'multimodal':
                train_dataset = MultiModelDataset(self.args, train_df, self.tokenizer, self.train_transform)
                valid_dataset = MultiModelDataset(self.args, valid_df, self.tokenizer, self.valid_transform)
            elif self.args.method == 'nlp':
                train_dataset = NLPDataset(self.args, train_df, self.tokenizer)
                valid_dataset = NLPDataset(self.args, valid_df, self.tokenizer)
            elif self.args.method == 'image':
                train_dataset = ImageDataset(self.args, train_df, self.train_transform)
                valid_dataset = ImageDataset(self.args, valid_df, self.valid_transform)
            else:
                raise NotImplementedError('args.method 를 잘 선택 해주세요.')
            self.train_loader = train_dataset.loader
            self.valid_loader = valid_dataset.loader

            self.step_per_epoch = len(self.train_loader)
        else:
            if self.args.method == 'multimodal':
                test_dataset = MultiModelDataset(self.args, df, self.tokenizer, self.valid_transform, is_test=True)
            elif self.args.method == 'nlp':
                test_dataset = NLPDataset(self.args, df, self.tokenizer, is_test=True)
            elif self.args.method == 'image':
                test_dataset = ImageDataset(self.args, df, self.valid_transform, is_test=True)
            else:
                raise NotImplementedError('args.method 를 잘 선택 해주세요.')
            self.test_loader = test_dataset.loader

    def train_epoch(self, epoch):
        self.train_loss.reset()
        self.train_acc.reset()

        self.model.train()

        self.optimizer.zero_grad()

        train_iterator = tqdm(self.train_loader, desc='Train Iteration')
        for step, batch in enumerate(train_iterator):
            batch = self.batch_to_device(batch)

            total_step = epoch * self.step_per_epoch + step

            with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                logits = self.model(batch)

                loss = self.supervised_loss(logits, batch['label'])
                if 'rdrop' in self.args.method:
                    logits_2 = self.model(batch)

                    ce_loss = (self.supervised_loss(logits, batch['label']) +
                               self.supervised_loss(logits_2, batch['label'])) * 0.5
                    kl_loss = self.rdrop_loss(logits, logits_2)
                    loss = ce_loss + kl_loss * self.args.rdrop_coef

                self.scaler.scale(loss).backward()

                preds = torch.argmax(logits, dim=-1)
                acc = accuracy_score(batch['label'].cpu(), preds.cpu())

                if (step + 1) % self.args.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                self.train_loss.update(loss.item(), self.args.train_batch_size)
                self.train_acc.update(acc, self.args.train_batch_size)

                if total_step != 0 and total_step % self.args.eval_steps == 0:
                    valid_acc, valid_f1_score, valid_loss = self.validate()

                    self.scheduler.step(valid_acc)

                    self.model.train()
                    if self.args.wandb:
                        wandb.log({
                            'train/loss': self.train_loss.avg,
                            'train/acc': self.train_acc.avg,
                            'eval/loss': valid_loss,
                            'eval/acc': valid_acc,
                            'eval/f1_score': valid_f1_score,
                        })

                    self.logger.info(
                        f'STEP {total_step} | eval loss: {valid_loss:.4f} | eval acc: {valid_acc:.4f} | eval f1_score: {valid_f1_score:.4f} | train loss: {self.train_loss.avg:.4f} | train acc: {self.train_acc.avg:.4f}'
                    )

                    if valid_f1_score > self.best_valid_f1_score:
                        self.logger.info(f'BEST_BEFORE : {self.best_valid_f1_score:.4f}, NOW : {valid_f1_score:.4f}')
                        self.logger.info('Saving Model...')
                        self.best_valid_f1_score = valid_f1_score
                        self.save_model(total_step)

    def validate(self):
        self.model.eval()

        valid_iterator = tqdm(self.valid_loader, desc="Valid Iteration")

        valid_acc = AverageMeter()
        valid_loss = AverageMeter()

        preds_list = []
        label_list = []
        with torch.no_grad():
            for step, batch in enumerate(valid_iterator):
                batch = self.batch_to_device(batch)

                logits = self.model(batch)

                loss = self.supervised_loss(logits, batch['label'])

                preds = torch.argmax(logits, dim=-1)
                preds_list.append(preds.detach().cpu().numpy())
                label_list.append(batch['label'].detach().cpu().numpy())
                acc = accuracy_score(batch['label'].cpu(), preds.cpu())

                valid_loss.update(loss.item(), self.args.valid_batch_size)
                valid_acc.update(acc, self.args.valid_batch_size)

        preds_list = np.hstack(preds_list)
        label_list = np.hstack(label_list)
        valid_f1_score = f1_score(label_list, preds_list, average='weighted')

        return valid_acc.avg, valid_f1_score, valid_loss.avg

    def _get_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        if self.args.optimizer == 'AdamW':
            optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        elif self.args.optimizer == 'MADGRAD':
            optimizer = MADGRAD(optimizer_grouped_parameters, lr=self.args.lr)
        else:
            raise NotImplementedError('args.optimizer 를 잘 선택 해주세요.')

        return optimizer

    def _get_scheduler(self):
        scheduler = ReduceLROnPlateau(self.optimizer, 'max', patience=self.args.patience, factor=0.9)
        return scheduler

    def save_model(self, step):
        if self.best_model_folder:
            shutil.rmtree(self.best_model_folder)

        if not os.path.exists(self.args.output_path):
            os.makedirs(self.args.output_path, exist_ok=True)

        file_name = f'FOLD{self.args.fold}_STEP_{step}_{self.args.method}_LR{self.args.lr}_WD{self.args.weight_decay}_IMG_SIZE{self.args.img_size}_RDROP{self.args.rdrop_coef}_F1{self.best_valid_f1_score:.4f}'
        output_path = os.path.join(self.args.output_path, file_name)

        os.makedirs(output_path, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(output_path, 'model_state_dict.pt'))

        self.logger.info(f'Model Saved at {output_path}')
        self.best_model_folder = output_path

        if self.args.wandb:
            wandb.log({'eval/best_f1_score': self.best_valid_f1_score})

    def batch_to_device(self, batch):
        batch = {k: v.to(self.args.device) for k, v in batch.items()}
        return batch

    def predict(self):
        model_state_dict = torch.load(os.path.join(self.args.saved_model_path, 'model_state_dict.pt'),
                                      map_location=self.args.device)
        self.model.load_state_dict(model_state_dict)

        test_iterator = tqdm(self.test_loader, desc='Test Iteration')

        preds_list = []
        probs_list = []
        with torch.no_grad():
            for step, batch in enumerate(test_iterator):
                batch = self.batch_to_device(batch)

                logits = self.model(batch)
                probs_list.append(logits.detach().cpu().numpy())
                preds = torch.argmax(logits, dim=-1)
                preds_list.append(preds.detach().cpu().numpy())

        preds_list = np.hstack(preds_list)
        probs_list = np.vstack(probs_list)

        return preds_list, probs_list


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
