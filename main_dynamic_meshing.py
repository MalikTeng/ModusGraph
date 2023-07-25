import os
import time

import pandas as pd
from tqdm import tqdm
from glob import glob
from itertools import chain

from monai.config import print_config

import torch


torch.multiprocessing.set_sharing_strategy('file_system')

def pairing_check(image_paths, label_paths, ):
    for image_path, label_path in zip(image_paths, label_paths):
        case_id = os.path.split(label_path)[-1].replace('.nii.gz', '')
        assert case_id in image_path, Exception(f'image and label do not match')

def train(sup_params):
    ct_image_paths = sorted(
        glob(
            "/mnt/DATA/Experiment/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task508_SCOTCAP/imagesTr/*.nii.gz"
        )
    )
    ct_label_paths = sorted(
        glob(
            "/mnt/DATA/Experiment/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task508_SCOTCAP/labelsTr/*.nii.gz"
        )
    )
    pairing_check(ct_image_paths, ct_label_paths)
    ct_bundle = {
        'image_paths': ct_image_paths, 'label_paths': ct_label_paths,
        'keys': ('ct_image', 'ct_label'), 'dataset': 'scotheart'
    }

    pipeline = TrainPipeline(
        sup_params,
        log_dir, out_dir,
        seed=2048, num_workers=16,
        bundle=ct_bundle,
    )
    pipeline.train()

def fine_tune(sup_params):
    split_list = pd.read_csv(
        "/home/yd21/Documents/ModusGraph/utils/CAP_SPLIT.csv",
        header=0, index_col=False
    )
    train_cases = split_list['subject'][(split_list['split'] == 'train').values ^ (split_list['split'] == 'val').values].tolist()
    train_cases = set([i.split('-')[0] for i in train_cases])
    # train_cases = [
    #     'CHD0035401', 'CHD0035701',
    #     'CHD0036701', 'CHD0036702',
    #     'CHD1055302', 'CHD1355901', 
    #     'CHD1368501', 'CHD1631001', 
    #     'CHD1725301', 'CHD2285901',
    #     'CHD2884501', 'CHD3537602',
    #     'CHD3568301', 'CHD3977501',
    #     'CHD4698201', 'CHD4918401',
    #     'CHD4935201', 'CHD5085102',
    #     'CHD5287101', 'CHD5295901',
    #     'CHD5413102', 'CHD5779802_1',
    #     'CHD5779802_2', 'CHD6072601',
    #     'CHD6581101', 'CHD7371501',
    #     'CHD7552902', 'CHD7566601',
    # ]

    image_paths = sorted([
        glob(
            f"/mnt/DATA/Experiment/ModusGraph/Data_CAP_ModusGraph-new/images/{case}_0000.nii.gz"
        )[0]
        for case in train_cases
    ])
    label_paths = sorted([
        glob(
            f"/mnt/DATA/Experiment/ModusGraph/Data_CAP_ModusGraph-new/labels/{case}.nii.gz"
        )[0]
        for case in train_cases
    ])

    pairing_check(image_paths, label_paths)
    bundle = {
        'image_paths': image_paths, 'label_paths': label_paths,
        'keys': ('image', 'label'), 'dataset': 'cap'
    }

    pipeline = TrainPipeline(
        sup_params,
        log_dir, out_dir,
        seed=2048, num_workers=16,
        bundle=bundle,
        is_training=False, is_finetune=True,
        experiment_ID='2023-03-16-1112-ModusGraph'
    )
    pipeline.gru_train(19)

def test(sup_params):
    split_list = pd.read_csv(
        "/home/yd21/Documents/ModusGraph/utils/CAP_SPLIT.csv",
        header=0, index_col=False
    )
    train_cases = split_list['subject'][(split_list['split'] == 'test').values].tolist()
    train_cases = set([i.split('-')[0] for i in train_cases])

    image_paths = sorted([
        glob(
            f"/mnt/DATA/Experiment/ModusGraph/Data_CAP_ModusGraph-new/images/{case}_0000.nii.gz"
        )[0]
        for case in train_cases
    ])
    label_paths = sorted([
        glob(
            f"/mnt/DATA/Experiment/ModusGraph/Data_CAP_ModusGraph-new/labels/{case}.nii.gz"
        )[0]
        for case in train_cases
    ])

    pairing_check(image_paths, label_paths)
    bundle = {
        'image_paths': image_paths, 'label_paths': label_paths,
        'keys': ('image', 'label'), 'dataset': 'cap'
    }

    pipeline = TrainPipeline(
        sup_params,
        log_dir, out_dir,
        seed=2048, num_workers=16,
        bundle=bundle,
        is_training=False, is_finetune=False,
        experiment_ID='2023-03-16-1123-ModusGraph'
    )
    pipeline.test(99)


if __name__ == '__main__':
    from run_solomesh import *

    import seaborn as sns
    import matplotlib.pyplot as plt

    import warnings
    warnings.filterwarnings('ignore')

    log_dir = "/home/yd21/Documents/ModusGraph/output/logs"
    out_dir = "/home/yd21/Documents/ModusGraph/output/outs"

    sup_params = SuperParams(
        max_epochs=100, delay_epochs=-1, val_interval=20,
        crop_window_size=(128, 128, 128),
        cache_rate=1.0,
        
        batch_size=1,  # batch_size sets to 1 for now
        point_limit=10_000,   # point_limit sets the num_vertices of the initial mesh
        num_classes=5
    )

    # # Start training
    # train(sup_params)

    # Start fine-tuning
    fine_tune(sup_params)

    # # Start testing
    # test(sup_params)


