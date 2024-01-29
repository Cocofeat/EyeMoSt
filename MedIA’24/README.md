# Confidence-aware multi-modality learning for eye disease screening
## 1. Requirment
- Pytorch 1.3.0
- Python 3
- sklearn
- numpy
- scipy
## 2. Prepare dataset
* Download the datasets and change the dataset path:
* [OLIVES dataset path](https://github.com/Cocofeat/EyeMoSt/blob/fb471c67beafe70dfb4d67f896d3220ec0a48df3/MedIA%E2%80%9924/train3_trans.py#L409)
* [GAMMA dataset basepath and datapath](https://github.com/Cocofeat/EyeMoSt/blob/fb471c67beafe70dfb4d67f896d3220ec0a48df3/MedIA%E2%80%9924/train3_trans.py#L431)

## 2. Pretrained models
* Download pretrained models and put them in ./pretrain/

### 2.1 CNN-based
* Fundus (2D): [Res2Net](https://github.com/LeiJiangJNU/Res2Net)
* OCT (3D):  [Med3d](https://github.com/cshwhale/Med3D)
### 2.2 Transformer-based
* Fundus (2D): [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
* OCT (3D): [UNETR](https://github.com/Project-MONAI/research-contributions/tree/main/UNETR)

## 3. Train
### 3.1 Train Baseline
Run the script ```main_train2.sh python baseline_train3_trans.py``` to train the baselines (change ``` model_name ```& ```mode```), models will be saved in folder ```results```
### 3.2 Train Our Model
Run the script ```main_train2.sh python train3_trans.py``` to train our model (change ``` model_name ```), models will be saved in folder ```results```

## 4. Test
### 4.1 Test Baseline
Run the script ```main_train2.sh python baseline_train3_trans.py``` to test our model  (change ``` model_name ```& ```mode```)
### 4.2 Test Our Model
Run the script ```main_train2.sh python train3_trans.py``` to test our model (change ``` model_name ```& ```mode```)
