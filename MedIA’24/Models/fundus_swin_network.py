# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
#from opts import opts
#opt = opts().parse()
import torch

def build_model():
    #model_type = config.MODEL.TYPE

    img_size = 384 #224
    patch_size = 4
    in_chans = 3
    num_classes = 3
    embed_dim = 128
    depths = [2,2,18,2]
    num_heads = [4, 8, 16, 32]
    window_size = 12 #7
    mlp_ratio = 4
    qkv_bias = True
    qk_scale = None
    drop_rate = 0.0
    drop_path_rate = 0.5
    ape = False
    patch_norm = True
    use_checkpoint = False
    #import pdb;pdb.set_trace()
    if True:#opt.model_type == 'swin':
        model = SwinTransformer(img_size= img_size,#config.DATA.IMG_SIZE,
                                patch_size= patch_size, #config.MODEL.SWIN.PATCH_SIZE,
                                in_chans= in_chans,#config.MODEL.SWIN.IN_CHANS,
                                num_classes=num_classes, #config.MODEL.NUM_CLASSES,
                                embed_dim=embed_dim,#config.MODEL.SWIN.EMBED_DIM,
                                depths=depths,#config.MODEL.SWIN.DEPTHS,
                                num_heads=num_heads,#config.MODEL.SWIN.NUM_HEADS,
                                window_size=window_size,#config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=mlp_ratio,#config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=qkv_bias,#config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=qk_scale,#config.MODEL.SWIN.QK_SCALE,
                                drop_rate=drop_rate,#config.MODEL.DROP_RATE,
                                drop_path_rate=drop_path_rate,#config.MODEL.DROP_PATH_RATE,
                                ape=ape,#config.MODEL.SWIN.APE,
                                patch_norm=patch_norm,#config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=False)#config.TRAIN.USE_CHECKPOINT)
    #else:
    #    raise NotImplementedError(f"Unkown model: {model_type}")
    #import pdb;pdb.set_trace()

    #snapshot_name = '../data/trained_models/swin_base_patch4_window7_224.pth'
    # snapshot_name = './pretrain/swin_base_patch4_window12_384_22k.pth'
    # snapshot_name = '/data/zou_ke/projects/2021-gamma-main/src/pretrain/swin_base_patch4_window12_384.pth'
    snapshot_name = '/data/zou_ke/projects/2021-gamma-main/src/pretrain/swin_base_patch4_window12_384.pth'
    pre_state_dict = torch.load(snapshot_name)
    print("load model OK.")
    pre_state_dict = pre_state_dict['model']
    cnt = 0
    state_dict = model.state_dict()
    for key_old in pre_state_dict.keys():
        key = key_old
        if key not in state_dict:
            continue
        value = pre_state_dict[key_old]
        if not isinstance(value, torch.FloatTensor):
            value = value.data
        state_dict[key] = value
        cnt += 1
    print('Load para num:', cnt)
    model.load_state_dict(state_dict)

    return model

