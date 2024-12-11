# 【EyeMoSt & EyeMoSt+】 
* This repository provides the code for our paper 【MICCAI 2023 Early Accept】"Reliable Multimodality Eye Disease Screening via Mixture of Student's t Distributions" and 【Medical Image Analysis submission 2024】"Confidence-aware multi-modality learning for eye disease screening"
* Current official implementation of [EyeMoSt](https://arxiv.org/abs/2303.09790)
* All codes are released in the version of [EyeMoSt+](https://github.com/Cocofeat/EyeMoSt/tree/main/MedIA%E2%80%9924).

## Requirment
- Pytorch 1.3.0
- Python 3
- sklearn
- numpy
- scipy
- ...

## Datasets
* [GAMMA dataset](https://gamma.grand-challenge.org/)
* [OLIVES dataset](https://doi.org/10.5281/zenodo.7105232)

## Code Usage
### 1. Prepare dataset
* Download the datasets and change the dataset path:
* [OLIVES dataset path](https://github.com/Cocofeat/EyeMoSt/blob/fb471c67beafe70dfb4d67f896d3220ec0a48df3/MedIA%E2%80%9924/train3_trans.py#L409)
* [GAMMA dataset basepath and datapath](https://github.com/Cocofeat/EyeMoSt/blob/fb471c67beafe70dfb4d67f896d3220ec0a48df3/MedIA%E2%80%9924/train3_trans.py#L431)

### 2. Pretrained models
* Download pretrained models and put them in ./pretrain/

#### 2.1 CNN-based
* Fundus (2D): [Res2Net](https://github.com/LeiJiangJNU/Res2Net)
* OCT (3D):  [Med3d](https://github.com/cshwhale/Med3D)
#### 2.2 Transformer-based
* Fundus (2D): [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
* OCT (3D): [UNETR](https://github.com/Project-MONAI/research-contributions/tree/main/UNETR)

### 3. Train
#### 3.1 Train Baseline
Run the script [main_train2.sh](https://github.com/Cocofeat/EyeMoSt/blob/main/MedIA%E2%80%9924/main_train2.sh)```main_train2.sh python baseline_train3_trans.py``` to train the baselines (change ``` model_name ```& ```mode```), models will be saved in folder ```results```
#### 3.2 Train Our Model
Run the script [main_train2.sh](https://github.com/Cocofeat/EyeMoSt/blob/main/MedIA%E2%80%9924/main_train2.sh) ```main_train2.sh python train3_trans.py``` to train our model (change ``` model_name ```), models will be saved in folder ```results```
### 4. Test
#### 4.1 Test Baseline
Run the script [main_train2.sh](https://github.com/Cocofeat/EyeMoSt/blob/main/MedIA%E2%80%9924/main_train2.sh) ```main_train2.sh python baseline_train3_trans.py``` to test our model  (change ``` model_name ```& ```mode```)
#### 4.2 Test Our Model
Run the script [main_train2.sh](https://github.com/Cocofeat/EyeMoSt/blob/main/MedIA%E2%80%9924/main_train2.sh) ```main_train2.sh python train3_trans.py``` to test our model (change ``` model_name ```& ```mode```)

## Citation
If you find uMedGround helps your research, please cite our paper:
```
@InProceedings{uMedGround_Zou_2024,
author="Zou, Ke
and Lin, Tian
and Yuan, Xuedong
and Chen, Haoyu
and Shen, Xiaojing
and Wang, Meng
and Fu, Huazhu",
title="Reliable Multimodality Eye Disease Screening via Mixture of Student's t Distributions",
journal={arXiv preprint arXiv:2404.06798},
year={2024}
}
```
