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
## 可视化
# python tools/analysis_tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--eval-interval ${EVALUATION_INTERVAL}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
# 绘制lr
python tools/analysis_tools/analyze_logs.py plot_curve \
work_dirs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-1xb4_10k8_hod3k16b_selfmeantsd_resnetpre_firstrandom/20240428_140831/vis_data/20240428_140831.json \
work_dirs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-1xb4_10k8_hod3ksa_selfmeanstd_resnetpre/20240428_120651/vis_data/20240428_120651.json \
--title loss_diffdet_hod3k16b_vs_sa \
--keys loss \
--legend hod3k16b_selfmeantsd_resnetpre_firstrandom hod3ksa_selfmeanstd_resnetpre \
--out work_dirs/vis_plot/loss_diffdet_1.pdf

# 绘制mAP
python tools/analysis_tools/analyze_logs.py plot_curve \
work_dirs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-1xb4_10k8_hod3k16b_selfmeantsd_resnetpre_firstrandom/20240428_140831/vis_data/20240428_140831.json \
work_dirs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-1xb4_10k8_hod3ksa_selfmeanstd_resnetpre/20240428_120651/vis_data/20240428_120651.json \
--title mAP_diffdet_hod3k16b_vs_sa \
--keys bbox_mAP \
--legend hod3k16b_selfmeantsd_resnetpre_firstrandom hod3ksa_selfmeanstd_resnetpre \
--out work_dirs/vis_plot/mAP_diffdet_1.pdf

# train and train
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-1xb4_14k4_hod3k16b_selfmeantsd_resnetpre_unfreeze_copy.py \
&& CUDA_VISIBLE_DEVICES=0 python tools/train.py \
projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-1xb4_14k4_hod3k16b_selfmeantsd_resnetpre_unfreeze_copymean.py \
&& CUDA_VISIBLE_DEVICES=0 python tools/train.py \
projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-1xb4_14k4_hod3k16b_selfmeantsd_resnetpre_unfreeze_random3-zero.py \
&& CUDA_VISIBLE_DEVICES=2 python tools/train.py \
projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-1xb4_14k4_hod3k16b_selfmeantsd_resnetpre_unfreeze_random3.py \
&& CUDA_VISIBLE_DEVICES=2 python tools/train.py \
projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-1xb4_14k4_hod3k16b_selfmeantsd_resnetpre_unfreeze_unfreeze.py \