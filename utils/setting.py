import logging
import random

import numpy as np
import pandas as pd
import torch
import transformers

from argparse import ArgumentParser


class Arguments:

    def __init__(self):
        self.parser = ArgumentParser()

    def add_type_of_processing(self):
        self.add_argument('--use_amp', action='store_true')
        self.add_argument('--is_train', action='store_true')
        self.add_argument('--cv', action='store_true')
        self.add_argument('--device', type=str,
                          default=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
        self.add_argument('--wandb', action='store_true')

    def add_hyper_parameters(self):
        self.add_argument('--method', type=str, default='multimodal', choices=('multimodal', 'nlp'))
        self.add_argument('--optimizer', type=str, default='AdamW', choices=('AdamW', 'MADGRAD'))
        self.add_argument('--text_model_name_or_path', type=str, default='klue/roberta-base')
        self.add_argument('--image_model_name_or_path', type=str, default='efficientnet_b0')
        self.add_argument('--max_seq_len', type=int, default=256)
        self.add_argument('--train_batch_size', type=int, default=8)
        self.add_argument('--valid_batch_size', type=int, default=128)
        self.add_argument('--epochs', type=int, default=3)
        self.add_argument('--accumulation_steps', type=int, default=1)
        self.add_argument('--eval_steps', type=int, default=100)
        self.add_argument('--seed', type=int, default=42)
        self.add_argument('--lr', type=float, default=1e-5)
        self.add_argument('--weight_decay', type=float, default=0.1)
        self.add_argument('--warmup_ratio', type=float, default=0.05)
        self.add_argument('--rdrop_coef', type=float, default=0.0)
        self.add_argument('--patience', type=int, default=5)
        self.add_argument('--num_labels', type=int, default=128)
        self.add_argument('--img_size', type=int, default=128)

    def add_data_parameters(self):
        self.add_argument('--text_path_to_train_data', type=str, default='./data/train.csv')
        self.add_argument('--text_path_to_test_data', type=str, default='./data/test.csv')
        self.add_argument('--image_path_to_train_data', type=str, default='./data/image/train')
        self.add_argument('--image_path_to_test_data', type=str, default='./data/image/test')
        self.add_argument('--output_path', type=str, default='./saved_model')
        self.add_argument('--predict_path', type=str, default='./predict')

    def print_args(self, args):
        for idx, (key, value) in enumerate(args.__dict__.items()):
            if idx == 0:
                print('argparse{\n', '\t', key, ':', value)
            elif idx == len(args.__dict__) - 1:
                print('\t', key, ':', value, '\n')
            else:
                print('\t', key, ':', value)

    def add_argument(self, *args, **kw_args):
        return self.parser.add_argument(*args, **kw_args)

    def parse(self):
        args = self.parser.parse_args()
        if args.device == '0':
            args.device = torch.device('cuda:0')
        if args.device == '1':
            args.device = torch.device('cuda:1')

        df = pd.read_csv(args.text_path_to_train_data)
        label_to_idx = {cat: i for i, cat in enumerate(df['cat3'].unique())}
        idx_to_label = {v: k for k, v in label_to_idx.items()}

        self.print_args(args)

        args.label_to_idx = label_to_idx
        args.idx_to_label = idx_to_label

        return args


class Setting:

    def set_logger(self):
        _logger = logging.getLogger()
        formatter = logging.Formatter(
            '[%(levelname)s] %(asctime)s [ %(message)s ] | file::%(filename)s | line::%(lineno)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        _logger.addHandler(stream_handler)
        _logger.setLevel(logging.INFO)

        transformers.logging.set_verbosity_error()

        return _logger

    def set_seed(self, args):
        seed = args.seed

        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def run(self):
        parser = Arguments()
        parser.add_type_of_processing()
        parser.add_hyper_parameters()
        parser.add_data_parameters()

        args = parser.parse()
        logger = self.set_logger()
        self.set_seed(args)

        return args, logger
