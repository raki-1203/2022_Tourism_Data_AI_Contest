import numpy as np
import wandb

from sklearn.model_selection import StratifiedKFold

from utils.data_preprocessing import get_df
from utils.setting import Setting
from utils.trainer import Trainer

if __name__ == '__main__':

    args, logger = Setting().run()

    df = get_df(args)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    fold_valid_f1_score_list = []
    for fold, splits in enumerate(skf.split(df, df['cat3']), start=1):
        args.fold = fold

        # cv=True 인 경우 5폴드를 모두 실행
        if not args.cv:
            if fold > 1:
                break

        if args.wandb:
            name = f'{args.method}_{args.output_path.split("/")[-1]}_FOLD{args.fold}'
            wandb_config = {k: v for k, v in vars(args).items() if 'idx' not in k}
            wandb.init(project='2022 관광데이터 AI 경진대회',
                       name=name,
                       config=wandb_config,
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
