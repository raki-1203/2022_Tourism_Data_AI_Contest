# multimodal CrossValidation Train
#python train.py --is_train --use_amp --device 1 --epochs 15 --accumulation_steps 8 --cv --wandb

# multimodal Image size 324 Train
#python train.py --is_train --use_amp --device 1 --epochs 30 --accumulation_steps 8 --wandb --img_size 324 --output_path ./saved_model/baseline_img324

# nlp Train
#python train.py --is_train --use_amp --device 1 --epochs 15 --accumulation_steps 32 --wandb --output_path ./saved_model/nlp_only --method nlp --text_model_name_or_path klue/roberta-large --max_seq_len 256 --train_batch_size 4

# nlp MADGRAD Train
#python train.py --is_train --use_amp --device 1 --epochs 15 --accumulation_steps 32 --wandb --output_path ./saved_model/nlp_only_MADGRAD --method nlp --text_model_name_or_path klue/roberta-large --max_seq_len 256 --train_batch_size 4 --optimizer MADGRAD

# nlp MADGRAD CrossValidation Patience Train
python train.py --is_train --use_amp --device 1 --epochs 10 --accumulation_steps 32 --cv --wandb --output_path ./saved_model/nlp_only_MADGRAD_cv --method nlp --text_model_name_or_path klue/roberta-large --max_seq_len 256 --train_batch_size 4 --optimizer MADGRAD --patience 50
python inference.py --device 1 --output_path ./saved_model/nlp_only_MADGRAD_cv --predict_path ./predict/nlp_only_MADGRAD_cv --method nlp --text_model_name_or_path klue/roberta-large
