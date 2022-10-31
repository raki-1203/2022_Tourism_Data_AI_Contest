<div align="center">
  <h1>2022 관광데이터 AI 경진대회</h1>
</div>

![image](https://user-images.githubusercontent.com/52475378/197491406-3b7b5fd4-656c-4d5a-9446-aee2bbf6b7bc.png)

## :fire: Getting Started
### Dependencies
```
albumentations==1.3.0
numpy==1.23.4
opencv-python==4.6.0.66
pandas==1.5.1
Pillow==9.2.0
scikit-learn==1.1.2
timm==0.6.11
tqdm==4.64.1
transformers==4.23.1
wandb==0.13.4

# cuda 버전에 맞게 알아서 설치하셔야 합니다. (https://pytorch.org/get-started/previous-versions/)
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### Install Requirements
- `pip install -r requirements.txt`

### Contents
```
.
├── data
│   ├── image
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv
├── predict
├── saved_model
├── sh_file
│   └── experiment.sh
├── utils
│   ├── custom_dataset.py
│   ├── custom_model.py
│   ├── data_preprocessing.py
│   ├── loss.py
│   ├── optimizer.py
│   ├── __pycache__
│   ├── setting.py
│   └── trainer.py
├── train.py
├── inference.py
├── requirements.txt
└── EDA.ipynb
```

## :mag: Overview
### Background
> 최신 핫플레이스를 소개하는 디지털 여행정보서비스가 다양한 채널을 통해 개발되고 있으며, 관광지점 정보(POI:Point Of Interests)데이터의 가치가 더욱 높아지고 있습니다. <br>
한국관광공사가 제공하는 국문관광정보의 생산을 인공지능의 힘으로 자동화 한다면, 더 적은 공공의 예산으로 더 많은 POI 데이터가 만들어질 수 있습니다. <br>
관광지에 관한 텍스트와 이미지 데이터를 이용하여 카테고리를 자동분류할수 있는 최적의 알고리즘을 만들어 주세요

### Subject
> 국문관광정보(POI)의 텍스트와 이미지데이터를 이용한 카테고리 자동분류 AI솔루션 개발

### Problem definition
> 데이터로 주어지는 2만 3천여개의 POI를 사용하여 관광지 카테고리 (대>중>소) 중 '소' 카테고리를 예측

### Organizer
- 주최: 한국관광공사, 씨엘컴퍼니
- 주관: 데이콘

### Development environment
- GPU rtx-2080ti 
- Ubuntu 18.04.3
- PyCharm | Python 3.9.13

### Evaluation
- 평가 산식 : Weighted F1 Score

### Training
```
python train.py
--use_amp
--is_train
--cv
--device 0
--wandb
--method nlp
--text_model_name_or_path klue/roberta-large
--image_model_name_or_path efficientnet_b0
--train_batch_size 4
--epochs 15
--accumulation_steps 8
--seed 42
--output_path ./saved_model
```

### Inference
```
python inference.py
--device 1
--output_path ./saved_model 
--predict_path ./predict/ 
--method nlp
```
