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
CUDA_VISIBLE_DEVICES=7 python tools/train.py configs/rtmdet/rtmdet_l_8xb32-300e_coco_hod3kSA.py

CUDA_VISIBLE_DEVICES=6 python tools/train.py configs/ddq/ddq-detr-4scale_r50_8xb2-12e_coco_hod3ksa.py
