'''
Author: annzstbl@tianhaoli1996@gmail.com
Date: 2024-04-16 17:44:39
LastEditors: annzstbl@tianhaoli1996@gmail.com
LastEditTime: 2024-04-16 17:45:03
FilePath: /mmdetection/users/inference_tool.py
Description: 

Copyright (c) 2024 by ${annzstbl}, All Rights Reserved. 
'''
from mmdet.apis import DetInferencer

# Choose to use a config
model_name = 'rtmdet_tiny_8xb32-300e_coco'
# Setup a checkpoint file to load
checkpoint = './checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

# Set the device to be used for evaluation
device = 'cuda:7'

# Initialize the DetInferencer
inferencer = DetInferencer(model_name, checkpoint, device)

# Use the detector to do inference
img = './demo/demo.jpg'
result = inferencer(img, out_dir='./output')

# Show the structure of result dict
from rich.pretty import pprint
pprint(result, max_length=4)

# Show the output image
from PIL import Image
Image.open('./output/vis/demo.jpg')