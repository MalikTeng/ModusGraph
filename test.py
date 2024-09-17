import os
import time
from glob import glob
import argparse
import torch
import wandb

from utils.tools import draw_eval_score
wandb.login()

from run import *
from utils import *

import warnings
warnings.filterwarnings('ignore')


torch.multiprocessing.set_sharing_strategy('file_system')

def config():
    """
        This function is for parsing commandline arguments.
    """
    parser = argparse.ArgumentParser()
    # mode parameters
    parser.add_argument("--mode", type=str, default="disabled", help="choose the mode for wandb, can be 'disabled', 'offline', 'online'")
    parser.add_argument("--save_on", type=str, default="cap", help="the dataset for testing, can be 'cap', or 'sct'")
    parser.add_argument("--template_dir", type=str,
                        default="/home/yd21/Documents/ModusGraph/template/template_mesh-myo.obj",
                        help="the path to your template meshes")
    
    # training parameters
    parser.add_argument("--max_epochs", type=int, default=6, help="the maximum number of epochs for training")
    parser.add_argument("--delay_epochs", type=int, default=3, help="the number of epochs for pre-training")
    parser.add_argument("--val_interval", type=int, default=1, help="the interval of validation")
    parser.add_argument("--num_classes", type=int, default=2, help="the number of segmentation classes of foreground plus background")
    parser.add_argument("--cache_rate", type=float, default=1.0, help="the cache rate for training, see MONAI document for more details")
    parser.add_argument("--lr", type=float, default=1e-2, help="the learning rate for training")
    parser.add_argument("--crop_window_size", type=int, nargs='+', default=[128, 128, 128], help="the size of the crop window for training")
    parser.add_argument("--lambda_", type=float, nargs='+', default=[1.0, 1.0, 10.0, 1.0, 10.0], help="the coefficients of segmentation, chamfer distance, edge, face, and laplacian loss")

    # data parameters
    parser.add_argument("--ct_json_dir", type=str,
                        default="./dataset/dataset_task20_f0.json", 
                        help="the path to the json file with named list of CTA train/valid/test sets")
    parser.add_argument("--ct_data_dir", type=str, 
                        default="/mnt/data/Experiment/nnUNet/nnUNet_raw/Dataset020_SCOTHEART", 
                        help="the path to your processed images, must be in nifti format")
    parser.add_argument("--mr_json_dir", type=str,
                        default="./dataset/dataset_task11_f0.json", 
                        help="the path to the json file with named list of CMR train/valid/test sets")
    parser.add_argument("--mr_data_dir", type=str, 
                        default="/mnt/data/Experiment/nnUNet/nnUNet_raw/Dataset011_CAP_SAX", 
                        help="the path to your processed images")
    parser.add_argument("--ckpt_dir", type=str, 
                        default="/mnt/data/Experiment/ModusGraph/logs", 
                        help="the path to your checkpoint directory, for holding trained models and logs")
    parser.add_argument("--out_dir", type=str, 
                        default="/mnt/data/Experiment/ModusGraph/outs", 
                        help="the path to your output directory, for saving checkpoints and outputs")
     
    # path to the pretrained Voxel Processing Module
    parser.add_argument("--pre_trained_ct_module_dir", type=str, default=None, help="the path to the pretrained Voxel Processing Module")
    parser.add_argument("--pre_trained_mr_module_dir", type=str, default=None, help="the path to the pretrained Voxel Processing Module")

    # structure parameters for modality handel and ResNet decoder
    parser.add_argument("--init_filters", type=int, default=8, help="the number of initial filters for the modality handel")
    parser.add_argument("--in_channels", type=int, default=1, help="the number of input channels for the modality handel")
    parser.add_argument("--num_init_blocks", type=int, nargs='+', default=(1, 2, 2, 4), help="the number of residual blocks for the modality handel")
    parser.add_argument("--num_up_blocks", type=int, nargs='+', default=(1, 1, 1), help="the number of up blocks for the ResNet decoder")

    # structure parameters for R-StGCN
    parser.add_argument("--subdiv_level", type=int, default=2, help="the number of subdivision levels for R-StGCN")
    parser.add_argument("--hidden_features", type=int, default=32, help="the number of hidden features for R-StGCN")

    # run_id for wandb, will create automatically if not specified for training
    parser.add_argument("--run_id", type=str, default=None, help="the run name for wandb and local machine")

    # the best epoch for testing
    parser.add_argument("--best_epoch", type=int, default=None, help="the best epoch for testing")

    args = parser.parse_args()

    return args


def test(super_params):
    wandb.init(mode=super_params.mode)
    pipeline = TrainPipeline(
        super_params=super_params,
        seed=42, num_workers=0,
        is_training=False
    )
    pipeline.test(super_params.save_on)


if __name__ == '__main__':
    super_params = config()

    # checkpoints info
    super_params.save_on = "sct"
    ckpt_date = "2024-07-28-1609"   # ct
    super_params.best_epoch = 141

    # super_params.save_on = "cap"
    # ckpt_date = "2024-07-20-0039"   # mr
    # super_params.best_epoch = 121

    # target dataset info
    target = "mmwhs"
    super_params.ct_json_dir = "./dataset/dataset_task22_f0.json"
    super_params.ct_data_dir = "/mnt/data/Experiment/nnUNet/nnUNet_raw/Dataset022_MMWHS_CT"
    super_params.mr_json_dir = "./dataset/dataset_task11_f0.json"
    super_params.mr_data_dir = "/mnt/data/Experiment/nnUNet/nnUNet_raw/Dataset011_CAP_SAX"

    # output info    
    super_params.run_id = f"{super_params.save_on}--myo--f0--{ckpt_date}"
    super_params.ckpt_dir = f"/mnt/data/Experiment/ModusGraph/logs/stationary/{super_params.run_id}/ckpt_weights"
    super_params.template_dir = "/home/yd21/Documents/ModusGraph/template/template_mesh-myo.obj"
    super_params.out_dir = f"/mnt/data/Experiment/TMI_2024/{target}/ModusGraph/myo/f0/"
    

    test(super_params)
