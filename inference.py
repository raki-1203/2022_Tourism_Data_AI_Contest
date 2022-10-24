import os

import numpy as np
import pandas as pd

from glob import glob

from utils.setting import Setting
from utils.trainer import Trainer


def main():
    args, logger = Setting().run()

    df = pd.read_csv(args.text_path_to_test_data)
    df['img_path'] = df['id'].apply(lambda x: os.path.join(args.image_path_to_test_data, x + '.jpg'))

    if args.cv:
        model_list = glob(os.path.join(args.output_path, '*'))
        model_list = sorted(model_list)
    else:
        model_list = glob(os.path.join(args.output_path, '*'))

    output_probs = np.zeros((df.shape[0], args.num_labels))
    for i, model_name in enumerate(model_list, start=1):
        args.saved_model_path = model_name
        logger.info(f'{i} 번째 predict 진행 중!')
        trainer = Trainer(args, logger, df)
        preds_list, probs_list = trainer.predict()

        output_probs += probs_list

    pred_answer = np.argmax(output_probs, axis=-1).tolist()

    output_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/sample_submission.csv'))
    output_df['cat3'] = pred_answer
    output_df['cat3'] = output_df['cat3'].map(args.idx_to_label)

    if not os.path.exists(args.predict_path):
        os.makedirs(args.predict_path, exist_ok=True)

    file_save_path = os.path.join(args.predict_path, f'submission_{args.output_path.split("/")[-1]}')
    output_df.to_csv(f'{file_save_path}.csv', index=False)
    logger.info(f'File Save at {file_save_path}')


if __name__ == '__main__':
    main()
