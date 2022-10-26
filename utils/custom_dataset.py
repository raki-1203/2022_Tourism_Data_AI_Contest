import cv2
import torch

from torch.utils.data import Dataset, DataLoader


class MultiModelDataset(Dataset):

    def __init__(self, args, df, tokenizer, transforms, is_test=False):
        self.df = df
        self.transforms = transforms
        self.is_test = is_test
        self.collate_fn = CollateMultiModal(args, tokenizer, is_test)
        self.loader = DataLoader(dataset=self,
                                 batch_size=args.train_batch_size if not is_test else args.valid_batch_size,
                                 shuffle=True if not is_test else False,
                                 sampler=None,
                                 collate_fn=self.collate_fn)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # text
        text = self.df['overview'].iloc[idx]

        # Image
        img_path = self.df['img_path'].iloc[idx]
        image = cv2.imread(img_path)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        # Label
        if self.is_test:
            return text, image
        else:
            label = self.df['label'].iloc[idx]
            return text, image, label


class CollateMultiModal:

    def __init__(self, args, tokenizer, is_test):
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        self.is_test = is_test

    def __call__(self, batches):
        b_input_ids = []
        b_input_attention_mask = []
        b_input_images = []
        if not self.is_test:
            b_labels = []

        for b in batches:
            text = b[0]
            text_ids = self.tokenizer.encode(text)

            # truncate
            SPECIAL_TOKENS_NUM = 2  # <CLS>text_ids<EOS>
            limit = self.max_seq_len - SPECIAL_TOKENS_NUM
            if len(text_ids) > limit:
                text_ids = text_ids[:limit]

            # ids, mask
            input_ids = [self.tokenizer.bos_token_id] + text_ids + [self.tokenizer.eos_token_id]
            input_attention_mask = [1] * len(input_ids)

            # padding, max_padding 을 해야만 여러 batch 를 inference 했을 때 같은 결과값이 나옴
            if len(text_ids) < self.max_seq_len:
                pad_num = self.max_seq_len - len(input_ids)
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_num
                input_attention_mask = input_attention_mask + [0] * pad_num

                assert len(input_ids) == self.max_seq_len
                assert len(input_attention_mask) == self.max_seq_len

            b_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            b_input_attention_mask.append(torch.tensor(input_attention_mask, dtype=torch.long))
            b_input_images.append(b[1])
            if not self.is_test:
                b_labels.append(b[2])

        t_input_ids = torch.stack(b_input_ids)  # List[Tensor] -> Tensor List
        t_input_attention_mask = torch.stack(b_input_attention_mask)  # List[Tensor] -> Tensor List
        t_input_images = torch.stack(b_input_images)  # List[Tensor] -> Tensor List
        if self.is_test:
            return t_input_ids, t_input_attention_mask, t_input_images
        else:
            t_labels = torch.tensor(b_labels)  # List -> Tensor
            return t_input_ids, t_input_attention_mask, t_input_images, t_labels


class NLPDataset(Dataset):

    def __init__(self, args, df, tokenizer, is_test=False):
        self.df = df
        self.is_test = is_test
        self.collate_fn = CollateNLP(args, tokenizer, is_test)
        self.loader = DataLoader(dataset=self,
                                 batch_size=args.train_batch_size if not is_test else args.valid_batch_size,
                                 shuffle=True if not is_test else False,
                                 sampler=None,
                                 collate_fn=self.collate_fn)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # text
        text = self.df['overview'].iloc[idx]

        # Label
        if self.is_test:
            return text
        else:
            label = self.df['label'].iloc[idx]
            return text, label


class CollateNLP:

    def __init__(self, args, tokenizer, is_test):
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        self.is_test = is_test

    def __call__(self, batches):
        b_input_ids = []
        b_input_attention_mask = []
        if not self.is_test:
            b_labels = []

        for b in batches:
            if self.is_test:
                text = b
            else:
                text = b[0]
            text_ids = self.tokenizer.encode(text)

            # truncate
            SPECIAL_TOKENS_NUM = 2  # <CLS>text_ids<EOS>
            limit = self.max_seq_len - SPECIAL_TOKENS_NUM
            if len(text_ids) > limit:
                text_ids = text_ids[:limit]

            # ids, mask
            input_ids = text_ids
            input_attention_mask = [1] * len(input_ids)

            # padding, max_padding 을 해야만 여러 batch 를 inference 했을 때 같은 결과값이 나옴
            if len(text_ids) < self.max_seq_len:
                pad_num = self.max_seq_len - len(input_ids)
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_num
                input_attention_mask = input_attention_mask + [0] * pad_num

                assert len(input_ids) == self.max_seq_len
                assert len(input_attention_mask) == self.max_seq_len

            b_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            b_input_attention_mask.append(torch.tensor(input_attention_mask, dtype=torch.long))
            if not self.is_test:
                b_labels.append(b[1])

        t_input_ids = torch.stack(b_input_ids)  # List[Tensor] -> Tensor List
        t_input_attention_mask = torch.stack(b_input_attention_mask)  # List[Tensor] -> Tensor List
        if self.is_test:
            return t_input_ids, t_input_attention_mask
        else:
            t_labels = torch.tensor(b_labels)  # List -> Tensor
            return t_input_ids, t_input_attention_mask, t_labels
