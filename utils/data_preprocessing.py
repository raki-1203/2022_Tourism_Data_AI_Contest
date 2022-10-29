import os
import re

import pandas as pd


def text_preprocess(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r'#', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣ<>()\s\.\?!》《≪≫\'<>〈〉:‘’%,『』「」＜＞・\"-“”∧]", "", text)
    return text


def get_df(args):
    if args.is_train:
        df = pd.read_csv(args.text_path_to_train_data)
        df['overview'] = df['overview'].apply(lambda x: text_preprocess(x))
        df['img_path'] = df['id'].apply(lambda x: os.path.join(args.image_path_to_train_data, x + '.jpg'))
        df['cat1'] = df['cat1'].map(args.cat1_to_idx)
        df['cat2'] = df['cat2'].map(args.cat2_to_idx)
        df['label'] = df['cat3'].map(args.cat3_to_idx)
    else:
        df = pd.read_csv(args.text_path_to_test_data)
        df['overview'] = df['overview'].apply(lambda x: text_preprocess(x))
        df['img_path'] = df['id'].apply(lambda x: os.path.join(args.image_path_to_test_data, x + '.jpg'))
    return df
