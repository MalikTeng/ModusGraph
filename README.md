# Introduction
This repository contains scripts and tools that run code for the paper titled ‘[ModusGraph: Automated 3D and 4D Mesh Model Reconstruction from cine CMR with Improved Accuracy and Efficiency](https://link.springer.com/chapter/10.1007/978-3-031-43990-2_17)’. The sole purpose of this repository is to provide a reference for the paper. Please note that the code is written in Python 3.9.16 and Pytorch 1.12.1, and is not optimized for efficiency or guaranteed to be bug-free.
 
![Alt text](figure/Fig-1.png)

To gain a better understanding of the idea behind the network design and parameter settings, we invite you to watch the following brief presentation using your MICCAI 2023 access.

[MICCAI Virtual Presentation](https://miccai2023.conflux.events/app/schedule/session/3294/2693)

## Connect and Contact with the Author
[LinkedIn](https://www.linkedin.com/in/malik-teng-86085149/)

# Installation
The code is tested on Ubuntu 18.04.6 LTS. To install the code, first clone the repository:

    $ git clone https://github.com/MalikTeng/ModusGraph
    
    Then install the conda environment:
    $ conda env create -f environment.yml
    
    Then activate the environment:
    $ conda activate modusgraph

# Usage
## Data
The data used in the paper is not included in this repository. The data is available upon request, and details provided in the paper. The data is organized in the following structure:
```
data
├── imagesTr
│   ├── 1.nii.gz
│   ├── 2.nii.gz
│   ├── ...
│   └── 100.nii.gz
├── labelsTr
│   ├── 1.nii.gz
│   ├── 2.nii.gz
│   ├── ...
│   └── 100.nii.gz
```
Template meshes were needed for runing the code. The template meshes are included for reference. The template meshes are organized in the following structure:
```
template
├── control_mesh-lv.obj
├── control_mesh-myo.obj
└── control_mesh-rv.obj
```

## Preprocessing
Data preprocessing is a must so that images, segmentations, and template meshes are in the same space. While ways to conduct such a preprocessing are not included in this script, the result can be varified by the script 'test_XXXX.py'. 

## Training
Detail structure of ModusGraph can be found in the paper.

The training process contains two stages:
1. Training: train the whole pipeline, including both ct and mr modality-handles.
2. Fine-tune: fine tuning the R-StGCN module with mr data.

Parameters of the network is customizable but not recommended.

We recommand using Weights and Bias for monitoring the training process. Following snapshot shows the training process of the two training stages.
![Alt Text](figure/wandb_screenshot.png)

## Command line Running
Running the whole training process is straightforward, just use commandline tool. See details of adjustable parameters in the script.

    $ python train.py \

    --save_on cap                               # choose from 'cap' or 'sct' \
    --template_dir  /your/path/to/template.obj  # choose from template folder according to the save_on \
    --ct_data_dir   /your/path/to/ct_data       # path to your Dataset020_SCOTHEART_COMBINE \
    --mr_data_dir   /your/path/to/mr_data       # path to your Dataset017_CAP_COMBINE \
    --ckpt_dir      /your/path/to/checkpoint \
    --out_dir       /your/path/to/output \

    --max_epochs 500 \
    --delay_epochs 250 \
    --val_interval 50 \

    --crop_window_size 128 128 128 \
    --batch_size 24 \
    --point_limit 53_500 \

    --cache_rate 1.0