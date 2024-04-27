###
 # @Author: annzstbl@tianhaoli1996@gmail.com
 # @Date: 2024-04-16 17:47:31
 # @LastEditors: annzstbl@tianhaoli1996@gmail.com
 # @LastEditTime: 2024-04-23 14:30:08
 # @FilePath: /mmdetection/users/train_tool.sh
 # @Description: 
 # 
 # Copyright (c) 2024 by ${annzstbl}, All Rights Reserved. 
### 


# 检查
python tools/misc/print_config.py /PATH/TO/CONFIG

# rtmdet
# todo 待确认
CUDA_VISIBLE_DEVICES=3 python tools/train.py configs/rtmdet/rtmdet_l_8xb32-300e_coco_hod3kSA.py

# ddq
## train
CUDA_VISIBLE_DEVICES=3 python tools/train.py configs/ddq/ddq-detr-4scale_r50_8xb2-12e_coco_hod3ksa.py
## test
CUDA_VISIBLE_DEVICES=2 python tools/test.py configs/ddq/ddq-detr-4scale_r50_8xb2-12e_coco_hod3ksa.py work_dirs/ddq-detr-4scale_r50_8xb2-12e_coco_hod3ksa/epoch_12.pth --show-dir workdirs/ddq-detr-4scale_r50_8xb2-12e_coco_hod3ksa/vis




# Difussion Det
## train
CUDA_VISIBLE_DEVICES=3 python tools/train.py projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-1xb4_7k2_hod3ksa_resnetpre.py
## test
CUDA_VISIBLE_DEVICES=3 python tools/test.py projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-1xb4_7k2_hod3ksa.pycheckpoints/diffdet_coco_res50.pth
## check
CUDA_VISIBLE_DEVICES=3 python tools/misc/print_config.py projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-1xb4_7k2_hod3ksa.py
## 转换pth
## python projects/DiffusionDet/model_converters/diffusiondet_resnet_to_mmdet.py ${DiffusionDet ckpt path} ${MMDetectron ckpt path}
python projects/DiffusionDet/model_converters/diffusiondet_resnet_to_mmdet.py checkpoints/diffdet_coco_res50.pth checkpoints/mmdet_diffdet_coco_res50.pth