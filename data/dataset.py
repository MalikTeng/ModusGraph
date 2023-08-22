import os
import sys
from glob import glob
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Union, SupportsIndex

import numpy as np

from monai.config.type_definitions import PathLike
from monai.data import (
    CacheDataset,
    partition_dataset,
    select_cross_validation_folds,
)
from monai.data.utils import list_data_collate
from monai.transforms import LoadImaged, Randomizable, MapTransform, Transform
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type

from torch.utils.data import Dataset
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData

from pytorch3d.structures import Meshes, Pointclouds

import torch

__all__ = ["MultimodalDataset", "collate_batched_meshes"]


def collate_batched_meshes(batch: list) -> dict:
    """
    Merge a mini-batch of meshes following PyG's procedure and save it
    as a single dictionary. This function can be used with a Dataset
    object to create a torch.utils.data.Dataloader which directly
    returns a mini-batch of torch_geometric.data.Data.

    :param batch: List of dictionaries containing information about objects in the dataset.
    :return collated_dict: Dictionary of collated lists. If batch contains torch_geometric.data.Data, a collated mini-batch is returned.
    """
    if batch is None or len(batch) == 0:
        return None
    collated_dict = {}
    for k in batch[0].keys():
        collated_dict[k] = [d[k] for d in batch]

    for k in collated_dict.keys():
        if "template" in k:
            if "ct" in k:
                # pack all parts (six for ct data) per sample as a Meshes object
                collated_dict[k] = {
                    "meshes": Meshes(
                        verts=[mesh_dict["meshes"].verts_packed() for mesh_dict in collated_dict[k]],
                        faces=[mesh_dict["meshes"].faces_packed() for mesh_dict in collated_dict[k]]
                        ),
                    "faces_labels": torch.concat([
                        mesh_dict["faces_labels"] for mesh_dict in collated_dict[k]
                        ]),
                    }
            elif "mr" in k:
                assert len(collated_dict[k]) == 1, "WARNING: works only for batch size of 1 and dynamic meshing task"
                collated_dict[k] = collated_dict[k][0]

        elif "point_clouds" in k:
            if "ct" in k:
                collated_dict[k] = {
                    "point_clouds": Pointclouds(
                        points=[pc_dict["point_clouds"].points_packed() for pc_dict in collated_dict[k]],
                        normals=[pc_dict["point_clouds"].normals_packed() for pc_dict in collated_dict[k]]
                        ),
                    "points_labels": torch.concat([
                        pc_dict["points_labels"] for pc_dict in collated_dict[k]
                        ]),
                }
            elif "mr" in k:
                assert len(collated_dict[k]) == 1, "WARNING: works only for batch size of 1 and dynamic meshing task"
                collated_dict[k] = collated_dict[k][0]

        else:
            collated_dict[k] = list_data_collate(collated_dict[k])
            if "mr" in k:
                assert len(collated_dict[k]) == 1, "WARNING: works only for batch size of 1"
                collated_dict[k] = collated_dict[k].permute(1, 0, 2, 3, 4)
    
    return collated_dict


def load_multimodal_datalist(
        image_paths: list,
        label_paths: list,
        keys: tuple
) -> list:
    """
    :params image_paths: sorted list of image file paths
    :params label_paths: sorted list of label file paths
    :params keys: IDs of image and label in a sequence, e.g., ('image', 'label')

    :returns sorted list of dictionary -- {'label': label_path, 'image': image_path}
    """
    assert len(image_paths) == len(label_paths)
    expected_data = [
        {
            keys[0]: image_path,
            keys[1]: label_path,
        } for image_path, label_path in zip(image_paths, label_paths)
    ]

    return expected_data


class MultimodalDataset(Randomizable, CacheDataset):
    """
        :params image_paths: sorted list of image file absolute paths
        :params label_paths: sorted list of label file absolute paths, paired with image file
        :params keys: IDs of image and label in a sequence, e.g., ('image', 'label')
        :params section: expected data section, can be: `training`, `validation` or `test`.
        :params transform: transforms to execute operations on input data.
                for further usage, use `AddChanneld` or `AsChannelFirstd` to convert the shape to [C, H, W, D].
        :params seed: random seed to randomly shuffle the datalist before splitting into training and validation, default is 0.
                note to set same seed for `training` and `validation` sections.
        :params val_frac: percentage of validation fraction in the whole dataset, default is 0.2.
        :params subset_frac: percentage of a subset to the whole dataset, default is 1.0.
        :params cache_num: number of items to be cached. Default is `sys.maxsize`.
                will take the minimum of (cache_num, data_length x cache_rate, data_length).
        :params cache_rate: percentage of cached data in total, default is 1.0 (cache all).
                will take the minimum of (cache_num, data_length x cache_rate, data_length).
        :params copy_cache: choose to copy it or not the data with deep_copy() in GPU, default is True.
        :params num_workers: the number of worker threads to use.
                if 0 a single thread will be used. Default is 0.
    """
    def __init__(
            self,
            image_paths: Union[list, SupportsIndex],
            label_paths: Union[list, SupportsIndex],
            keys: tuple,
            section: str,
            transform: Union[Sequence[Callable], Callable] = (),
            seed: int = 0,
            val_frac: float = 0.2,
            cache_num: int = sys.maxsize,
            cache_rate: float = 1.0,
            num_workers: int = 0,
    ):
        self.section = section
        self.val_frac = val_frac
        self.set_random_state(seed=seed)

        if len(image_paths) == 0 or len(label_paths) == 0:
            raise RuntimeError(f"Cannot find image and label paths.")
        self.indices: np.ndarray = np.array([])
        data = self._generate_data_list(image_paths, label_paths, keys)

        CacheDataset.__init__(
            self, data, transform, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers,
        )

    def get_indices(self) -> np.ndarray:
        """
            Get the indices of datalist used in this dataset.
        """
        return self.indices

    def _generate_data_list(self, image_paths, label_paths, keys) -> list:
        datalist = load_multimodal_datalist(image_paths, label_paths, keys)
        return self._split_data_list(datalist)

    def _split_data_list(self, datalist: list) -> list:
        self.indices = np.arange(0, len(datalist))
        val_length = int(len(self.indices) * self.val_frac)
        if self.section == "test":
            return [datalist[i] for i in self.indices]
        elif self.section == "training":
            self.indices = self.indices[val_length:]
            return [datalist[i] for i in self.indices]
        else:
            self.indices = self.indices[:val_length]
            return [datalist[i] for i in self.indices]

