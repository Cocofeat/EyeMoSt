# 1.train&test GAMMA
# 1.1 train GAMMA
# model name: EyeMost_Plus_transformer,EyeMost_Plus,EyeMost,EyeMost_prior;TMC
# EyeMost_Plus means EyeMost (CNN)
# EyeMost_Plus_transformer means EyeMost (Transformer)

# model_base: cnn/Transformer
# dataset: MGamma/OLIVES/MMOCTF
# condition: noise/normal
# EyeMost_Plus means EyeMost (CNN)
# EyeMost_Plus_transformer means EyeMost (Transformer)


# 1.1 train GAMMA
# CUDA_VISIBLE_DEVICES=1 python train3_trans.py \
#                          --folder "folder0"\
#                          --mode "train&test"\
#                          --model_name "EyeMost_Plus_transformer"\
#                          --model_base "transformer"\
#                          --dataset "MGamma"\
#                          --condition "normal"
# 1.2 test GAMMA
# CUDA_VISIBLE_DEVICES=1 python train3_trans.py \
#                          --folder "folder2"\
#                          --mode "test"\
#                          --model_base "transformer"\
#                          --model_name "EyeMost_Plus_transformer"\
#                          --dataset "MGamma"\
#                          --condition "noise"

# 1.3 train baseline Base_transformer/Res2Net2D/ResNet3D/Multi_ResNet(B-CNN)/Multi_EF_ResNet(B-EF)/Multi_CBAM_ResNet(M2LC)/Multi_dropout_ResNet(MCDO)
CUDA_VISIBLE_DEVICES=0 python baseline_train3_trans.py \
                         --folder "folder0"\
                         --mode "train&test"\
                         --model_name "Base_transformer"\
                         --model_base "transformer"\
                         --dataset "MGamma"\
                         --condition "normal"

# # 1.4 test baseline Base_transformer/BIF
# CUDA_VISIBLE_DEVICES=0 python baseline_train3_trans.py \
#                          --folder "folder0"\
#                          --mode "test"\
#                          --model_name "Base_transformer"\
#                          --model_base "transformer"\
#                          --dataset "MGamma"\
#                          --condition "noise"