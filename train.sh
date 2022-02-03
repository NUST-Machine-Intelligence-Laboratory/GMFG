#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3
export PYTHONWARNINGS="ignore"

# resnet18, resnet50
export NET='resnet18'
export path='cub_new'
export data='/home/zcy/data/fg-web-data/web-bird'
export data_meta='/home/zcy/farm_zcy/FG/meta-weight-net/meta_data/bird'
export N_CLASSES=200
export lr=0.01
export w_decay=1e-5


python train.py --net ${NET} --n_classes ${N_CLASSES} --path ${path} --data_base ${data} --data_meta ${data_meta} --lr ${lr} --w_decay ${w_decay} --epochs 100 --batch_size 24 --drop_rate 0.35 --relabel_rate 0.05  --tk 5