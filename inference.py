import os

import numpy as np
import pandas as pd

from glob import glob

from utils.data_preprocessing import get_df
from utils.setting import Setting
from utils.trainer import Trainer


def main():
    args, logger = Setting().run()

    df = get_df(args)

    if args.output_path_list:
        model_list = []
        for output_path in args.output_path_list:
            model_list += glob(os.path.join(output_path, '*'))
    else:
        model_list = glob(os.path.join(args.output_path, '*'))
    model_list = sorted(model_list)

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
    output_df['cat3'] = output_df['cat3'].map(args.idx_to_cat3)

    if not os.path.exists(args.predict_path):
        os.makedirs(args.predict_path, exist_ok=True)

    file_save_path = os.path.join(args.predict_path, f'submission_{args.output_path.split("/")[-1]}')
    output_df.to_csv(f'{file_save_path}.csv', index=False)
    logger.info(f'File Save at {file_save_path}')


if __name__ == '__main__':
    main()
