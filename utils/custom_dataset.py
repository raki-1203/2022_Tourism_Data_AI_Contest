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
            return {'text': text, 'image': image}
        else:
            label = self.df['label'].iloc[idx]
            return {'text': text, 'image': image, 'label': label}


class CollateMultiModal:

    def __init__(self, args, tokenizer, is_test):
        self.args = args
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        self.is_test = is_test

    def __call__(self, batches):
        b_input_ids = []
        b_input_attention_mask = []
        b_input_token_type_ids = []
        b_input_images = []
        if not self.is_test:
            b_labels = []

        for b in batches:
            text = b['text']
            tokenized_result = self.tokenizer(text,
                                              padding='max_length',
                                              truncation=True,
                                              max_length=self.args.max_seq_len)
            input_ids = tokenized_result['input_ids']
            input_attention_mask = tokenized_result['attention_mask']
            input_token_type_ids = tokenized_result['token_type_ids']

            b_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            b_input_attention_mask.append(torch.tensor(input_attention_mask, dtype=torch.long))
            b_input_token_type_ids.append(torch.tensor(input_token_type_ids, dtype=torch.long))

            b_input_images.append(b['image'])

            if not self.is_test:
                b_labels.append(b['label'])

        t_input_ids = torch.stack(b_input_ids)  # List[Tensor] -> Tensor List
        t_input_attention_mask = torch.stack(b_input_attention_mask)  # List[Tensor] -> Tensor List
        t_input_token_type_ids = torch.stack(b_input_token_type_ids)  # List[Tensor] -> Tensor List
        t_input_images = torch.stack(b_input_images)  # List[Tensor] -> Tensor List
        if self.is_test:
            return {'input_ids': t_input_ids, 'attention_mask': t_input_attention_mask,
                    'token_type_ids': t_input_token_type_ids, 'image': t_input_images}
        else:
            t_labels = torch.tensor(b_labels)  # List -> Tensor
            return {'input_ids': t_input_ids, 'attention_mask': t_input_attention_mask,
                    'token_type_ids': t_input_token_type_ids, 'image': t_input_images, 'label': t_labels}


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
            return {'text': text}
        else:
            label = self.df['label'].iloc[idx]
            return {'text': text, 'label': label}


class CollateNLP:

    def __init__(self, args, tokenizer, is_test):
        self.args = args
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        self.is_test = is_test

    def __call__(self, batches):
        b_input_ids = []
        b_input_attention_mask = []
        b_input_token_type_ids = []
        if not self.is_test:
            b_labels = []

        for b in batches:
            text = b['text']
            tokenized_result = self.tokenizer(text,
                                              padding='max_length',
                                              truncation=True,
                                              max_length=self.args.max_seq_len)
            input_ids = tokenized_result['input_ids']
            input_attention_mask = tokenized_result['attention_mask']
            input_token_type_ids = tokenized_result['token_type_ids']

            b_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            b_input_attention_mask.append(torch.tensor(input_attention_mask, dtype=torch.long))
            b_input_token_type_ids.append(torch.tensor(input_token_type_ids, dtype=torch.long))
            if not self.is_test:
                b_labels.append(b['label'])

        t_input_ids = torch.stack(b_input_ids)  # List[Tensor] -> Tensor List
        t_input_attention_mask = torch.stack(b_input_attention_mask)  # List[Tensor] -> Tensor List
        t_input_token_type_ids = torch.stack(b_input_token_type_ids)  # List[Tensor] -> Tensor List
        if self.is_test:
            return {'input_ids': t_input_ids, 'attention_mask': t_input_attention_mask,
                    'token_type_ids': t_input_token_type_ids}
        else:
            t_labels = torch.tensor(b_labels)  # List -> Tensor
            return {'input_ids': t_input_ids, 'attention_mask': t_input_attention_mask,
                    'token_type_ids': t_input_token_type_ids, 'label': t_labels}


class NLPCatDataset(Dataset):

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
            return {'text': text}
        else:
            cat1 = self.df['cat1'].iloc[idx]
            cat2 = self.df['cat2'].iloc[idx]
            cat3 = self.df['cat3'].iloc[idx]
            return {'text': text, 'cat1': cat1, 'cat2': cat2, 'cat3': cat3}


class CollateNLPCat:

    def __init__(self, args, tokenizer, is_test):
        self.args = args
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        self.is_test = is_test

    def __call__(self, batches):
        b_input_ids = []
        b_input_attention_mask = []
        b_input_token_type_ids = []
        if not self.is_test:
            b_cat1 = []
            b_cat2 = []
            b_cat3 = []

        for b in batches:
            text = b['text']
            tokenized_result = self.tokenizer(text,
                                              padding='max_length',
                                              truncation=True,
                                              max_length=self.args.max_seq_len)
            input_ids = tokenized_result['input_ids']
            input_attention_mask = tokenized_result['attention_mask']
            input_token_type_ids = tokenized_result['token_type_ids']

            b_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            b_input_attention_mask.append(torch.tensor(input_attention_mask, dtype=torch.long))
            b_input_token_type_ids.append(torch.tensor(input_token_type_ids, dtype=torch.long))
            if not self.is_test:
                b_cat1.append(b['cat1'])
                b_cat2.append(b['cat2'])
                b_cat3.append(b['cat3'])

        t_input_ids = torch.stack(b_input_ids)  # List[Tensor] -> Tensor List
        t_input_attention_mask = torch.stack(b_input_attention_mask)  # List[Tensor] -> Tensor List
        t_input_token_type_ids = torch.stack(b_input_token_type_ids)  # List[Tensor] -> Tensor List
        if self.is_test:
            return {'input_ids': t_input_ids, 'attention_mask': t_input_attention_mask,
                    'token_type_ids': t_input_token_type_ids}
        else:
            t_cat1 = torch.tensor(b_cat1)  # List -> Tensor
            t_cat2 = torch.tensor(b_cat2)  # Lis[t -> Tensor
            t_cat3 = torch.tensor(b_cat3)  # List -> Tensor
            return {'input_ids': t_input_ids, 'attention_mask': t_input_attention_mask,
                    'token_type_ids': t_input_token_type_ids, 'cat1': t_cat1, 'cat2': t_cat2, 'cat3': t_cat3}


class ImageDataset(Dataset):

    def __init__(self, args, df, transforms, is_test=False):
        self.df = df
        self.transforms = transforms
        self.is_test = is_test
        self.collate_fn = CollateImage(is_test)
        self.loader = DataLoader(dataset=self,
                                 batch_size=args.train_batch_size if not is_test else args.valid_batch_size,
                                 shuffle=True if not is_test else False,
                                 sampler=None,
                                 collate_fn=self.collate_fn)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Image
        img_path = self.df['img_path'].iloc[idx]
        image = cv2.imread(img_path)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        # Label
        if self.is_test:
            return {'image': image}
        else:
            label = self.df['cat1'].iloc[idx]
            return {'image': image, 'label': label}


class CollateImage:

    def __init__(self, is_test):
        self.is_test = is_test

    def __call__(self, batches):
        b_input_images = []
        if not self.is_test:
            b_labels = []

        for b in batches:
            b_input_images.append(b['image'])
            if not self.is_test:
                b_labels.append(b['label'])

        t_input_images = torch.stack(b_input_images)  # List[Tensor] -> Tensor List
        if self.is_test:
            return {'image': t_input_images}
        else:
            t_labels = torch.tensor(b_labels)  # List -> Tensor
            return {'image': t_input_images, 'label': t_labels}
