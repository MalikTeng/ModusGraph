#!/bin/bash

##### train #####
# datasets=("cap" "sct")
datasets=("sct")

for dataset in "${datasets[@]}"; do
    python train.py \
    --mode online \
    --subdiv_level 2 \
    --max_epochs 150 \
    --delay_epochs 100 \
    --val_interval 10 \
    --save_on "$dataset" \
    --template_dir /home/yd21/Documents/ModusGraph/template/template_mesh-myo.obj \
    --ct_json_dir ./dataset/dataset_task20_f0.json \
    --mr_json_dir ./dataset/dataset_task11_f0.json
done