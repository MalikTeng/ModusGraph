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
        dataset: str,
        crop_window_size: list,
        section: str,
        point_limit: int,
        one_or_multi: str,
        template_dir: str,
):
    """
    Conducting pre-transformation that comprises multichannel conversion,
    resampling in regard of space distance, reorientation, foreground cropping,
    normalization and data augmentation.
    
    :params keys: designated items (at most two) for pre-transformation, image and label
    :params dataset: ID of dataset that is used
    :params crop_window_size: image and label will be cropped to match the size of network input
    :params section: identifier of either training, validation or testing set
    """
    # data loading
    transforms = [
        LoadImaged(
            keys, 
            reader="NibabelReader",
            ensure_channel_first=True,
            dtype=(np.float32, np.int32)
        ),
        ToMetaTensord(keys)
    ]

    # label indices correction
    if one_or_multi == "multi":
        # static CT images and multi-classes labels shall be processed for the whole heart meshing task, and only data from SCOT-HEART is available
        if dataset.lower() == "scotheart":
            transforms.append(ConvertToMultiChannelBasedOnSCOTHEARTClassesd(keys[1]))
        else:
            raise ValueError(f"the function to process labels from {dataset} is not defined for {one_or_multi} task.")
    else:
        # both static CT images and labels and dynamic MR images and labels shall be processed for the dynamic meshing task.
        # the first channel is number of frames, and labels are not in one-hot format
        if dataset.lower() == "scotheart":
            transforms.append(Flipd(keys, spatial_axis=0))
            pass
        elif dataset.lower() == "cap":
            transforms.append(SelectFramesd(keys[1]))
        else:
            raise ValueError(f"the function to process labels from {dataset} is not defined for {one_or_multi} task.")
 
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
            
            # generate meshes from voxel label maps
            GenerateMeshesd(
                keys,
                one_or_multi=one_or_multi,
                dataset=dataset,
                size=max(crop_window_size),
                point_limit=point_limit,
                template_dir=template_dir,
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
            MergeMultiChanneld([keys[1], f"{keys[1]}_downsample"]) if dataset.lower() == "scotheart" else Identityd(),
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


# def mask_ground_truth(batch_data, keys):
#     """
#     Source the true slice position for each label
#     :param: batch_data: a batch of MR data comprises array, meta_dict and foreground coordinates
#     :param: crop_window_size: the size of network input

#     :return:    mask of slices with manual labels (1 for labels and 0 for others).
#     """
#     label = batch_data[keys[1]].get_array(np.ndarray)
#     B, _, H, W, D = np.round(label.shape).astype(int)
#     start_idx = [np.min(np.nonzero(i)[-1]) for i in np.argmax(label, axis=1)]
#     dz = [
#         torch.round(batch_data[keys[1]].meta["pixdim"][i, 3]).int().item()
#         for i in range(B)
#     ]

#     masks = np.zeros((B, 1, H, W, D))
#     for i in range(B):
#         masks[i, ..., np.arange(start_idx[i], D, dz[i])] = 1
#     masks = masks.astype(bool)

#     return torch.from_numpy(masks)
