import os

import torch
import numpy as np

from monai.transforms import (
    MapTransform,
    Activationsd,
    AsDiscreted,
    CropForegroundd,
    Compose,
    CopyItemsd,
    Invertd,
    LoadImaged,
    LabelToMaskd,
    ScaleIntensityd,
    Orientationd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGibbsNoised,
    RandKSpaceSpikeNoised,
    RandRotated,
    RandScaleCropd,
    RandAdjustContrastd,
    ResizeWithPadOrCropd,
    Rotate90d,
    Spacingd,
    Zoomd,
    RandZoomd,
    SaveImaged,
    EnsureChannelFirstd,
    EnsureTyped, Flipd, Resized, SpatialPadd, ToMetaTensord,
)

from data.components import *

from scipy.ndimage import shift

__all__ = ["pre_transform"]


def pre_transform(
        keys: tuple,
        crop_window_size: list,
        section: str,
):
    """
    Conducting pre-transformation that comprises multichannel conversion,
    resampling in regard of space distance, reorientation, foreground cropping,
    normalization and data augmentation.
    
    :params keys: designated items (at most two) for pre-transformation, image and label
    :params crop_window_size: image and label will be cropped to match the size of network input
    :params section: identifier of either training, validation or testing set
    """
    # data loading
    transforms = [
        LoadImaged(
            keys, ensure_channel_first=True,
            image_only=True,
            dtype=(np.float32, np.int32)
        ),
        # ToMetaTensord(keys)
        Spacingd(keys, pixdim=(2.0, 2.0, -1), mode=("bilinear", "nearest")),
        Orientationd(keys, axcodes="RAS"),
        # Spacingd(keys, pixdim=(-1, 2.0, 2.0), mode=("bilinear", "nearest")), # Particularly, ACDC
        Spacingd(keys, pixdim=(2.0, -1, -1), mode=("bilinear", "nearest")),
    ]

    # process images and labels
    transforms.extend(
        [
            # foreground cropping and down-sampling
            CropForegroundd(
                keys,
                source_key=keys[1],
                mode="minimum",
                margin=1
            ),
            CopyItemsd(
                keys[1],
                names=f"{keys[1]}_downsample"
            ),
            Resized(
                keys,
                spatial_size=max(crop_window_size),
                size_mode="longest", 
                mode=("trilinear", "nearest-exact")
            ),
            SpatialPadd(
                keys,
                spatial_size=crop_window_size,
                mode="minimum",
            ),
            
            # generate downsampled label maps
            Spacingd(
                f"{keys[1]}_downsample",
                pixdim=[8.0, 8.0, 8.0],
                mode="nearest",
                padding_mode="zeros"
            ),
            Resized(
                f"{keys[1]}_downsample",
                spatial_size=max(crop_window_size) // 8,
                size_mode="longest", mode="nearest-exact"
            ),
            SpatialPadd(
                f"{keys[1]}_downsample",
                spatial_size=max(crop_window_size) // 8,
                mode="minimum",
            ),
        ]
    )

    # final touch with fixed transforms
    if section == "training":
        # data-augmentation
        transforms.extend([
            RandScaleIntensityd(
                keys=keys[0], factors=0.4, prob=0.1
            ),
            RandShiftIntensityd(
                keys=keys[0], offsets=0.4, prob=0.1
            ),
            RandAdjustContrastd(
                keys=keys[0],
                gamma=(2.5, 4.5)
            ),
        ])
    transforms.append(ScaleIntensityd(keys[0], minv=0, maxv=1))

    return Compose(transforms)
