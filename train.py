import os

import numpy as np
import pandas as pd
import wandb

from sklearn.model_selection import StratifiedKFold

from utils.setting import Setting
from utils.trainer import Trainer

if __name__ == '__main__':

    args, logger = Setting().run()

    df = pd.read_csv(args.text_path_to_train_data)
    df['img_path'] = df['id'].apply(lambda x: os.path.join(args.image_path_to_train_data, x + '.jpg'))
    df['label'] = df['cat3'].map(args.label_to_idx)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    fold_valid_f1_score_list = []
    for fold, splits in enumerate(skf.split(df, df['label']), start=1):
        args.fold = fold

        # cv=True 인 경우 5폴드를 모두 실행
        if not args.cv:
            if fold > 1:
                break

        if args.wandb:
            name = f'{args.method}_LR{args.lr}_WD{args.weight_decay}_IMG_SIZE{args.img_size}_RDROP{args.rdrop_coef}_FOLD{args.fold}'
            wandb.init(project='2022 관광데이터 AI 경진대회',
                       name=name,
                       config=vars(args),
                       reinit=True)

        logger.info(f'>> Cross Validation {fold} Starts!')
        trainer = Trainer(args, logger, df, splits)

        for epoch in range(args.epochs):
            logger.info(f'Start Training Epoch {epoch}')
            trainer.train_epoch(epoch)
            logger.info(f'Finish Training Epoch {epoch}')

        fold_valid_f1_score_list.append(trainer.best_valid_f1_score)
        wandb.join()

    logger.info('Training Finished')

    logger.info(f'cv_f1_score_list: {fold_valid_f1_score_list}')
    logger.info(f'cv_mean_f1_score: {np.mean(fold_valid_f1_score_list)}')
