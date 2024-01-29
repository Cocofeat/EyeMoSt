# EyeMoSt & EyeMoSt+
* This repository provides the code for our paper 【MICCAI 2023 Early Accept】"Reliable Multimodality Eye Disease Screening via Mixture of Student's t Distributions" and 【Medical Image Analysis submission 2024】"Confidence-aware multi-modality learning for eye disease screening"
* Current official implementation of [EyeMoSt](https://arxiv.org/abs/2303.09790)
* All codes are released in the version of [EyeMoSt+](https://github.com/Cocofeat/EyeMoSt/tree/main/MedIA%E2%80%9924).

## Datasets
* [GAMMA dataset](https://gamma.grand-challenge.org/)\
* [OLIVES dataset](https://doi.org/10.5281/zenodo.7105232)

## Code Usage
### 1. Prepare dataset
* Download the datasets and change the dataset path:\
* [OLIVES dataset path](https://github.com/Cocofeat/EyeMoSt/blob/fb471c67beafe70dfb4d67f896d3220ec0a48df3/MedIA%E2%80%9924/train3_trans.py#L409)\
* [GAMMA dataset basepath and datapath](https://github.com/Cocofeat/EyeMoSt/blob/fb471c67beafe70dfb4d67f896d3220ec0a48df3/MedIA%E2%80%9924/train3_trans.py#L431)

### 2. Pretrained models
* Download pretrained models and put them in ./pretrain/\

#### 2.1 CNN-based\
* Fundus (2D): [Res2Net](https://github.com/LeiJiangJNU/Res2Net)\
* OCT (3D):  [Med3d](https://github.com/cshwhale/Med3D)\
#### 2.2 Transformer-based\
* Fundus (2D): [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)\
* OCT (3D): [UNETR](https://github.com/Project-MONAI/research-contributions/tree/main/UNETR)

### 3. Train
#### 3.1 Train Baseline

#### 3.2 Train Our Model

### 4. Test
#### 4.1 Train Baseline

#### 4.2 Train Our Model

## Citation
If you find EyeMoSt helps your research, please cite our paper:
```
@InProceedings{EyeMoSt_Zou_2023,
author="Zou, Ke
and Lin, Tian
and Yuan, Xuedong
and Chen, Haoyu
and Shen, Xiaojing
and Wang, Meng
and Fu, Huazhu",
title="Reliable Multimodality Eye Disease Screening via Mixture of Student's t Distributions",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="596--606",
}
```
