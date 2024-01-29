# EyeMoSt
* This repository provides the code for our paper 【MICCAI 2023 Early Accept】"Reliable Multimodality Eye Disease Screening via Mixture of Student's t Distributions"
* Current official implementation of [EyeMoSt](https://arxiv.org/abs/2303.09790)
* All of codes are released [EyeMoSt+](https://github.com/Cocofeat/EyeMoSt/tree/main/MedIA%E2%80%9924).

## Datasets
* [GAMMA dataset](https://gamma.grand-challenge.org/)

## Code Usage
### 1. Prepare dataset
* Download the datasets and change the dataset path:
* [GAMMA dataset basepath and datapath](https://github.com/Cocofeat/EyeMoSt/blob/fb471c67beafe70dfb4d67f896d3220ec0a48df3/MedIA%E2%80%9924/train3_trans.py#L431)

### 2. Pretrained models
* Download pretrained models and put them in ./pretrain/
* Fundus (2D): [Res2Net](https://github.com/LeiJiangJNU/Res2Net)
* OCT (3D):  [Med3d](https://github.com/cshwhale/Med3D)

### 3. Train & Test
Run the script ```main_train2.sh python train.py``` to test our model (change ``` model_name ```& ```mode```)
