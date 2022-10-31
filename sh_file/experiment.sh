# multimodal CrossValidation Train
#python train.py --is_train --use_amp --device 1 --epochs 15 --accumulation_steps 8 --cv --wandb

# multimodal Image size 324 Train
#python train.py --is_train --use_amp --device 1 --epochs 30 --accumulation_steps 8 --wandb --img_size 324 --output_path ./saved_model/baseline_img324

# nlp Train
#python train.py --is_train --use_amp --device 1 --epochs 15 --accumulation_steps 32 --wandb --output_path ./saved_model/nlp_only --method nlp --text_model_name_or_path klue/roberta-large --max_seq_len 256 --train_batch_size 4

# nlp MADGRAD Train
#python train.py --is_train --use_amp --device 1 --epochs 15 --accumulation_steps 32 --wandb --output_path ./saved_model/nlp_only_MADGRAD --method nlp --text_model_name_or_path klue/roberta-large --max_seq_len 256 --train_batch_size 4 --optimizer MADGRAD

# nlp MADGRAD CrossValidation Patience Train
#python train.py --is_train --use_amp --device 1 --epochs 10 --accumulation_steps 32 --cv --wandb --output_path ./saved_model/nlp_only_MADGRAD_cv --method nlp --text_model_name_or_path klue/roberta-large --max_seq_len 256 --train_batch_size 4 --optimizer MADGRAD --patience 10
#python inference.py --device 1 --output_path ./saved_model/nlp_only_MADGRAD_cv --predict_path ./predict/nlp_only_MADGRAD_cv --method nlp --text_model_name_or_path klue/roberta-large

# nlp base model CrossValidation Train
#python train.py --is_train --use_amp --device 1 --epochs 10 --accumulation_steps 1 --cv --wandb --output_path ./saved_model/nlp_base_cv --method nlp --text_model_name_or_path klue/roberta-base --max_seq_len 256 --train_batch_size 32
#python inference.py --device 1 --output_path ./saved_model/nlp_base_cv --predict_path ./predict/nlp_base_cv --method nlp --text_model_name_or_path klue/roberta-base

# nlp preprocessed text klue/roberta-base model CrossValidation Train
#python train.py --is_train --use_amp --device 1 --epochs 20 --accumulation_steps 1 --cv --wandb --output_path ./saved_model/nlp_preprocessed_text_base_cv --method nlp --text_model_name_or_path klue/roberta-base --max_seq_len 256 --train_batch_size 32
#python inference.py --device 1 --output_path ./saved_model/nlp_preprocessed_text_base_cv --predict_path ./predict/nlp_preprocessed_text_base_cv --method nlp --text_model_name_or_path klue/roberta-base

# nlp preprocessed text klue/roberta-base model LabelSmoothing CrossValidation Train
#python train.py --is_train --use_amp --device 1 --epochs 20 --accumulation_steps 1 --cv --wandb --output_path ./saved_model/nlp_preprocessed_text_label_smoothing_cv --method nlp --text_model_name_or_path klue/roberta-base --max_seq_len 256 --train_batch_size 32 --loss LabelSmoothing
#python inference.py --device 1 --output_path ./saved_model/nlp_preprocessed_text_label_smoothing_cv --predict_path ./predict/nlp_preprocessed_text_label_smoothing_cv --method nlp --text_model_name_or_path klue/roberta-base

# Image cat1 Train -> 성능 너무 안나옴
#python train.py --is_train --use_amp --device 1 --epoch 20 --accumulation_steps 1 --wandb --output_path ./saved_model/image_cat1 --method image --train_batch_size 32 --optimizer MADGRAD --image_model_name_or_path nfnet_f3 --lr 1e-6

# nlp preprocessed text klue/roberta-base model cat1, cat2, cat3 CrossValidation Train
#python train.py --is_train --use_amp --device 1 --epochs 20 --accumulation_steps 1 --cv --wandb --output_path ./saved_model/nlp_preprocessed_text_base_cat1_cat2_cat3_cv --method nlp --max_seq_len 256 --train_batch_size 32
#python inference.py --device 1 --output_path ./saved_model/nlp_preprocessed_text_base_cat1_cat2_cat3_cv --predict_path ./predict/nlp_preprocessed_text_base_cat1_cat2_cat3_cv --method nlp

# nlp preprocessed text klue/roberta-large model cat1, cat2, cat3 CrossValidation Train
#python train.py --is_train --use_amp --device 1 --epochs 20 --accumulation_steps 8 --cv --wandb --output_path ./saved_model/nlp_preprocessed_text_large_cat1_cat2_cat3_cv --method nlp --max_seq_len 256 --train_batch_size 4 --text_model_name_or_path klue/roberta-large
#python inference.py --device 1 --output_path ./saved_model/nlp_preprocessed_text_large_cat1_cat2_cat3_cv --predict_path ./predict/nlp_preprocessed_text_large_cat1_cat2_cat3_cv --method nlp

# Final Ensemble
python inference.py --device 1 --output_path_list ./saved_model/nlp_preprocessed_text_large_cat1_cat2_cat3_cv ./saved_model/nlp_preprocessed_text_large_seed43_cv --predict_path ./predict/final_ensemble --method nlp
