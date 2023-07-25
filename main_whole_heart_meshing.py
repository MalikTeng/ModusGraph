import os
import time
from glob import glob
import argparse
import torch
import wandb

from utils.tools import draw_eval_score
wandb.login()

from run_whole_heart_meshing import *
from utils import *

import warnings
warnings.filterwarnings('ignore')


# TODO: add weight&bias integration to watch the back propagation in the graph

torch.multiprocessing.set_sharing_strategy('file_system')

def config():
    """
        This function is for parsing commandline arguments.
    """
    parser = argparse.ArgumentParser()
    # mode parameters
    parser.add_argument("--mode", type=str, default="train", help="the mode of the script, can be 'train' or 'test'")
    # data parameters
    parser.add_argument("--image_dir", type=str, 
                        default="/mnt/DATA/Experiment/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task504_SCOTHEART/imagesTr", 
                        help="the path to your processed images, recommanded using NIfTI format")
    parser.add_argument("--label_dir", type=str, 
                        default="/mnt/DATA/Experiment/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task504_SCOTHEART/labelsTr", 
                        help="the path to your processed segmentations, recommanded using NIfTI format")
    parser.add_argument("--template_dir", type=str,
                        default="/home/yd21/Documents/ModusGraph/template/scotheart/",
                        help="the path to your template meshes")
    parser.add_argument("--log_dir", type=str, 
                        default="/mnt/DATA/Experiment/ModusGraph/logs", 
                        help="the path to your log directory, for tensorboard")
    parser.add_argument("--out_dir", type=str, 
                        default="/mnt/DATA/Experiment/ModusGraph/outs", 
                        help="the path to your output directory, for saving checkpoints and outputs")
    # path to the pretrained Voxel Processing Module
    parser.add_argument("--pre_trained_module_dir", type=str, default=None, help="the path to the pretrained Voxel Processing Module")
    # training parameters
    parser.add_argument("--num_classes", type=int, default=7, help="the number of segmentation classes of foreground plus background")
    parser.add_argument("--batch_size", type=int, default=4, help="the batch size for training")
    parser.add_argument("--cache_rate", type=float, default=1.0, help="the cache rate for training, see MONAI document for more details")
    parser.add_argument("--max_epochs", type=int, default=300, help="the maximum number of epochs for training")
    parser.add_argument("--delay_epochs", type=int, default=150, help="the number of epochs for pre-training")
    parser.add_argument("--val_interval", type=int, default=25, help="the interval of validation")

    parser.add_argument("--lr", type=float, default=1e-3, help="the learning rate for training")
    parser.add_argument("--crop_window_size", type=int, nargs='+', default=[128, 128, 128], help="the size of the crop window for training")
    parser.add_argument("--point_limit", type=int, default=53_500, help="the number limits of points for each mesh, it's a number will be defined in the pre-processing")
    parser.add_argument("--lambda_", type=float, nargs='+', default=[0.1, 1.0, 0.1, 0.1, 0.1], help="the coefficients of segmentation, chamfer distance, edge, face, and laplacian loss")
    # structure parameters for modality handel and ResNet decoder
    parser.add_argument("--init_filters", type=int, default=8, help="the number of initial filters for the modality handel")
    parser.add_argument("--in_channels", type=int, default=1, help="the number of input channels for the modality handel")
    parser.add_argument("--num_init_blocks", type=int, nargs='+', default=(1, 2, 2, 4), help="the number of residual blocks for the modality handel")
    parser.add_argument("--num_up_blocks", type=int, nargs='+', default=(1, 1, 1), help="the number of up blocks for the ResNet decoder")
    # structure parameters for R-StGCN
    parser.add_argument("--hidden_features", type=int, default=32, help="the number of hidden features for R-StGCN")
    # run_id for wandb, will create automatically if not specified for training
    parser.add_argument("--run_id", type=str, default=None, help="the run name for wandb and local machine")
    # the best epoch for testing
    parser.add_argument("--best_epoch", type=int, default=None, help="the best epoch for testing")

    args = parser.parse_args()

    return args

def pairing_check(image_paths, label_paths):
    """
        This function is for checking if images and labels are paired. Not necessary for training on your own data.
    """
    for image_path, label_path in zip(image_paths, label_paths):
        case_id = os.path.split(label_path)[-1].replace('.nii.gz', '')
        assert case_id in image_path, Exception(f'image and label do not match')

def train(**kwargs):
    # load images and labels (here used 400 cases from SCOTHEART)
    image_paths = sorted(glob(f"{super_params.image_dir}/*.nii.gz"))[:400]
    label_paths = sorted(glob(f"{super_params.label_dir}/*.nii.gz"))[:400]
    # check if images and labels are paired
    pairing_check(image_paths, label_paths)
    # initialize the training pipeline
    if super_params.pre_trained_module_dir is None:
        if kwargs.get("pre_trained_module_dir") is not None:
            super_params.pre_trained_module_dir = kwargs.get("pre_trained_module_dir")
        else:
            super_params.pre_trained_module_dir = None
    run_id = f"{time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))}"
    super_params.run_id = run_id
    wandb.init(project="ModusGraph_whole_heart_meshing", name=run_id, config=super_params, mode="online")
    pipeline = TrainPipeline(
        super_params=super_params,
        seed=2048, num_workers=0,
        data_bundle={
        'image_paths': image_paths, 'label_paths': label_paths,
        'keys': ('ct_image', 'ct_label'), 'dataset': 'scotheart'
        },
    )
    # train the network
    for epoch in range(super_params.max_epochs):
        # 1. train the Voxel Processing Module
        if super_params.pre_trained_module_dir is None and epoch < super_params.delay_epochs:
            pipeline.pre_train_iter(epoch)
            continue
        # 2. train the Graph Processing Module
        pipeline.train_iter(epoch)
        if (epoch - super_params.delay_epochs) % super_params.val_interval == 0:
            pipeline.valid(epoch)
    wandb.finish()

def test(**kwargs):
    image_paths = sorted(glob(f"{super_params.image_dir}/*.nii.gz"))[-40:]
    label_paths = sorted(glob(f"{super_params.label_dir}/*.nii.gz"))[-40:]
    pairing_check(image_paths, label_paths)
    if super_params.run_id is None:
        if kwargs.get("run_id") is not None:
            super_params.run_id = kwargs.get("run_id")
        else:
            raise Exception(f"run_id is not specified")
    if super_params.best_epoch is None:
        if kwargs.get("best_epoch") is not None:
            super_params.best_epoch = kwargs.get("best_epoch")
        else:
            raise Exception(f"best_epoch is not specified")
    wandb.init(mode="disabled")
    pipeline = TrainPipeline(
        super_params=super_params,
        seed=2048, num_workers=0,
        data_bundle={
        'image_paths': image_paths, 'label_paths': label_paths,
        'keys': ('ct_image', 'ct_label'), 'dataset': 'scotheart'
        },
        is_training=False,
    )
    pipeline.test()


if __name__ == '__main__':
    super_params = config()

    if super_params.mode == "train":
        # Start training
        train(pre_trained_module_dir=None)  # default control of pre-training
    else:
        # Start test
        test(run_id="2023-07-24-1840", best_epoch=180)  # default control of run_id and best_epoch
