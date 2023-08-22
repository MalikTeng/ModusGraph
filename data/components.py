from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import os
from glob import glob
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pytorch3d.structures import Meshes, Pointclouds
from torch_geometric.data import Data, Batch
from monai.transforms import MapTransform
from monai.data import MetaTensor
from scipy.ndimage import shift, measurements
from skimage.measure import marching_cubes
from trimesh import Trimesh, PointCloud
from trimesh.voxel.ops import matrix_to_marching_cubes
from trimesh import load

__all__ = [
    "Identityd",
    "SelectFramesd",
    "ConvertToMultiChannelBasedOnSCOTHEARTClassesd",
    "MergeMultiChanneld",
    "GenerateMeshesd"
]


class Identityd(MapTransform):
    """
        Identity transform
    """

    def __init__(self, keys: tuple = None, allow_missing_keys: bool = True):
        MapTransform.__init__(self, keys, allow_missing_keys)

    def __call__(self, data):
        return data

class SelectFramesd(MapTransform):
    """
        Select the ed and es frames as the first and last frames in the sequence
    """
    def __init__(self, keys: tuple, allow_missing_keys: bool = False):
        MapTransform.__init__(self, keys, allow_missing_keys)
        
    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            result = d[key].as_tensor()
            result = torch.stack([result[0], result[-1]])
            d[key] = MetaTensor(
                result, meta=d[key].meta,
                applied_operations=[
                    *d[key].applied_operations,
                    {
                        "class": "SelectFramesd",
                        "orig_shape": d[key].as_tensor().shape
                    }
                ]
            )
        return d

class ConvertToMultiChannelBasedOnSCOTHEARTClassesd(MapTransform):
    """
        Convert labels to multi channels based on ScotHeart classes:
        Background -> 0, # excluded
        LV-cavity -> 1, LV-myo -> 2, 
        RV-cavity -> 3, RV-myo -> 4
        LA -> 5, RA -> 6
    """

    def __init__(self, keys: tuple, allow_missing_keys: bool = False):
        MapTransform.__init__(self, keys, allow_missing_keys)

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            result = d[key].as_tensor()
            result = torch.vstack(
                [torch.where(result == k, v, 0) for k, v in zip(range(1, 7), [1] * 6)],
            )
            d[key] = MetaTensor(
                result, meta=d[key].meta,
                applied_operations=[
                    *d[key].applied_operations,
                    {
                        "class": "ConvertToMultiChannelBasedOnSCOTHEARTClassesd",
                        "orig_shape": d[key].as_tensor().shape
                    }
                ]
            )
        return d


class MergeMultiChanneld(MapTransform):
    """
        Merge multi-channel label maps into one
    """

    def __init__(self, keys: tuple, allow_missing_keys: bool = False):
        MapTransform.__init__(self, keys, allow_missing_keys)

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            result = d[key].as_tensor()
            result = result * torch.arange(1, result.shape[0] + 1).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            result = torch.sum(result, dim=0, keepdim=True)
            d[key] = MetaTensor(
                result, meta=d[key].meta,
                applied_operations=[
                    *d[key].applied_operations,
                    {
                        "class": "MergeMultiChanneld",
                        "orig_shape": d[key].as_tensor().shape
                    }
                ]
            )
        return d


def normalize_vertices(vertices: Tensor, size: Union[Tensor, int]):
    """
    Resize the values of coordinates of vertices as in the range of -1 to 1
    (Normalized Device Coordinates)
    :param vertices:    set of vertices" coordinates values
    :param shape:       the shape of label maps
    :return:
    """
    return 2 * (vertices / (size - 1) - 0.5)


def convert_trimesh(meshes: list) -> dict:
    """
    convert trimesh.Trimesh object to torch_geometric.data.Data object
    :param meshes:  list of trimesh.Trimesh or trimesh.PointCloud objects
    :param y:       ndarray of labels in integer for each and every vertex
    """
    if isinstance(meshes[0], PointCloud):
        meshes = Pointclouds(
            points=[torch.FloatTensor(mesh.vertices.copy()) for mesh in meshes], 
            normals=[torch.FloatTensor(mesh.metadata["vertex_normals"].copy()) for mesh in meshes]
            )
        num_verts_submeshes = meshes.num_points_per_cloud()
        cum_num_verts_submeshes = num_verts_submeshes.cumsum(0)
        verts_labels = torch.zeros((1, len(num_verts_submeshes), cum_num_verts_submeshes[-1]), dtype=torch.long)
        for i, (cum_num_verts, num_verts) in enumerate(zip(cum_num_verts_submeshes, num_verts_submeshes)):
            verts_labels[0, i, cum_num_verts-num_verts:cum_num_verts] = torch.arange(cum_num_verts-num_verts, cum_num_verts) + 1    # avoid verts missing when using nonzero method

        return {"point_clouds": meshes, "points_labels": verts_labels}
    else:
        meshes = Meshes(
            verts=[torch.FloatTensor(mesh.vertices.copy()) for mesh in meshes],
            faces=[torch.LongTensor(mesh.faces.copy()) for mesh in meshes],
        )
        num_faces_submeshes = meshes.num_faces_per_mesh()
        cum_num_faces_submeshes = num_faces_submeshes.cumsum(0)
        faces_labels = torch.zeros((1, len(num_faces_submeshes), cum_num_faces_submeshes[-1]), dtype=torch.long)
        for i, (cum_num_faces, num_faces) in enumerate(zip(cum_num_faces_submeshes, num_faces_submeshes)):
            faces_labels[0, i, cum_num_faces-num_faces:cum_num_faces] = torch.arange(cum_num_faces-num_faces, cum_num_faces) + 1    ## avoid verts missing when using nonzero method

        return {"meshes": meshes, "faces_labels": faces_labels}


class GenerateMeshesd(MapTransform):
    """
        Generate coordinates of mesh vertices from voxel label maps,
        should be insert right before the end of Compose
    """
    def __init__(
            self,
            keys: Tuple[str],
            one_or_multi: str,
            dataset: str,
            point_limit: int = 3000,
            size: int = 128,
            template_dir: str = None,
            allow_missing_labels: bool = True,
    ):
        MapTransform.__init__(self, keys, allow_missing_labels)
        self.one_or_multi = one_or_multi
        self.point_limit = point_limit
        self.dataset = dataset
        self.size = size
        self.template_dir = template_dir

        self.case_id = None

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            # generate mesh from voxel by Marching Cubes
            if "label" in key:
                voxels = d[key].as_tensor()
                if self.one_or_multi.lower() == "multi":
                    d[f"{key}_point_clouds"] = self._generate_multi_point_clouds(voxels, self.size)
                else:
                    d[f"{key}_point_clouds"] = self._generate_solo_point_clouds(voxels, self.size)
            # do nothing to the image, but load the template mesh
            elif "image" in key:
                voxels = d[key].as_tensor()
                if self.one_or_multi.lower() == "multi":
                    d[f"{key[:2]}_template_meshes"] = self._load_template_multi_meshes()
                else:
                    d[f"{key[:2]}_template_meshes"] = self._load_template_solo_mesh(voxels)

        return d

    def _subset(self, meshes: list) -> list:
        """
        Select a number of vertices from each mesh, to cut down memory usage
        :param  meshes: un-sampled mesh with labels of every vertex, List[Trimesh]
        :return vertex indices of sub-meshes, List[np.ndarray]
        """
        mesh_verts = np.array([mesh.vertices.shape[0] for mesh in meshes])
        if len(np.unique(np.diff(mesh_verts, axis=0))) < 3:
            # sanity check for created solo meshes
            num_verts = mesh_verts[0]
            assert num_verts > self.point_limit, "The number of vertices is less than the point limit, cannot do oversampling."
            point_limit = np.array([11_612 for _ in range(len(mesh_verts))], dtype=np.int32)
            assert self.point_limit == point_limit[0], f"The super_params: point_limit should be {point_limit[0]}."

        else:
            # sanity check for created multi meshes
            num_verts = mesh_verts.sum()
            assert num_verts > self.point_limit, "The number of vertices is less than the point limit, cannot do oversampling."
            point_limit = np.array(
                [8_000, 20_000, 10_000, 6_000, 6_000, 3_500],   # the number 53_500 comes from here
                dtype=np.int32
            )
            assert self.point_limit == sum(point_limit), f"The super_params: point_limit should be {sum(point_limit)}."

        point_limit = [
            np.random.choice(
                np.arange(mesh.vertices.shape[0]), face_limit_per_mesh, replace=False
            ) for mesh, face_limit_per_mesh in zip(meshes, point_limit)
        ]

        return point_limit

    def apply_marching_cubes(self, voxels: Tensor) -> list:
        entries = voxels.shape[0]
        meshes = []
        for entry in range(entries):
            mesh = matrix_to_marching_cubes(voxels[entry])
            meshes.append(mesh)

        return meshes

    def apply_norm(self, meshes: list, size: int) -> list:
        for i in range(len(meshes)):
            meshes[i].vertices = normalize_vertices(meshes[i].vertices, size)

        return meshes

    def apply_sampling(self, meshes: list, verts_limit: list) -> list:
        point_clouds = []
        for i, mesh in enumerate(meshes):
            point_cloud = PointCloud(
                vertices=mesh.vertices[verts_limit[i]],
                metadata={"vertex_normals": mesh.vertex_normals[verts_limit[i]]}
            )
            point_clouds.append(point_cloud)

        return point_clouds
    
    def _generate_multi_point_clouds(
            self, voxels: Tensor, size: Union[Tensor, int],
        ) -> dict:
        """
        Generate surface mesh as trimesh.Trimesh from voxel using Marching Cubes
        for MMWHS with seven labels
        :param voxels:  arrays of voxel label maps
        :param size:    the largest size of label maps
        """
        # create mesh from voxel
        meshes = self.apply_marching_cubes(voxels.numpy())

        # normalise the coordinates
        meshes = self.apply_norm(meshes, size)

        # sampling meshes for sub-meshes
        point_clouds = self.apply_sampling(meshes, self._subset(meshes))

        # convert meshes
        point_clouds_dict = convert_trimesh(point_clouds)

        return point_clouds_dict

    def _load_template_multi_meshes(self, ) -> dict:
        """
            get the template multi-parts mesh generated from a patient in Scotheart
        """
        # load template mesh for each part of the heart
        assert self.template_dir is not None, "Please specify the template directory."
        meshes = [
            load(f"{self.template_dir}/{i}") 
            for i in sorted(os.listdir(self.template_dir))  # sorted by name, which is the order of parts
        ]

        # convert trimesh to Meshes object
        meshes_dict = convert_trimesh(meshes)
        
        return meshes_dict

    def _generate_solo_point_clouds(
            self, voxels, size,
        ):
        """
            generate point clouds from voxel using Marching Cubes
            and labels for each and every vertex
            :param voxels:  arrays of voxel label maps
            :param size:    the largest size of label maps
        """
        voxels = voxels.cpu().numpy()

        # select voxel valued by 2 or 4 as left and right ventricular myocardium
        voxel_myo = np.logical_or(voxels == 2, voxels == 4)
        # duplicate the first frame as place holder
        voxel_myo[:-1] = voxel_myo[:1]
        meshes = self.apply_marching_cubes(voxel_myo)

        # normalise the coordinates
        meshes = self.apply_norm(meshes, size)

        # sampling meshes for sub-meshes
        point_clouds = self.apply_sampling(meshes, self._subset(meshes))

        # convert meshes
        point_clouds_dict = convert_trimesh(point_clouds)

        return point_clouds_dict

        # def intersect_meshes_vertices(mesh_vertices_a, mesh_vertices_b):
        #     mesh_vertices_a = {
        #         "{:.1f}-{:.1f}-{:.1f}".format(*coord): idx
        #         for idx, coord in enumerate(mesh_vertices_a)
        #     }
        #     mesh_vertices_b = {
        #         "{:.1f}-{:.1f}-{:.1f}".format(*coord): idx
        #         for idx, coord in enumerate(mesh_vertices_b)
        #     }
        #     indices_a = np.zeros(len(mesh_vertices_a), dtype=np.int64)
        #     indices_b = np.zeros(len(mesh_vertices_b), dtype=np.int64)
        #     for coord in mesh_vertices_b.keys():
        #         if coord in mesh_vertices_a:
        #             indices_a.put(mesh_vertices_a[coord], 1)
        #             indices_b.put(mesh_vertices_b[coord], 1)
        #     indices_c = np.nonzero(indices_a.copy())[0]
        #     indices_a = np.nonzero(1 - indices_a)[0]
        #     indices_b = np.nonzero(1 - indices_b)[0]

        #     return indices_a, indices_c, indices_b

        # def intersect_meshes_faces(mesh, exclude_vertices=None, include_vertices=None):
        #     if include_vertices is not None:
        #         exclude_vertices = np.delete(np.arange(mesh.vertices.shape[0]), include_vertices)
        #     elif exclude_vertices is not None:
        #         pass
        #     else:
        #         raise Exception("No valid index of vertices were received.")
        #     exclude_faces = mesh.vertex_faces[exclude_vertices]
        #     exclude_faces, c = np.unique(exclude_faces.flatten(), return_counts=True)
        #     exclude_faces = exclude_faces[c == 3]
        #     include_faces = np.delete(np.arange(mesh.faces.shape[0]), exclude_faces)

        #     return [include_faces, exclude_faces]

        # def annotate_vertices(mesh, parts):
        #     mesh_points = {"{:.1f}-{:.1f}-{:.1f}".format(*coord): idx for idx, coord in enumerate(mesh.vertices)}
        #     labels_all = np.full(len(mesh_points), 2, dtype=np.int64)
        #     for l, part in enumerate(parts):
        #         vertices = {"{:.1f}-{:.1f}-{:.1f}".format(*coord): idx for idx, coord in enumerate(part.vertices)}
        #         indices = []
        #         for coord in vertices.keys():
        #             if coord in mesh_points:
        #                 indices.append(mesh_points[coord])
        #         labels_all.put(indices, l)

        #     return labels_all

        # # create mesh from voxel
        # verts_myo, faces_myo, *_ = marching_cubes(
        #     np.sum(voxels[[2, -1]], axis=0), level=0.5, spacing=spacing
        # )
        # mesh_myo = Trimesh(vertices=verts_myo, faces=faces_myo)
        # verts_lv, faces_lv, *_ = marching_cubes(
        #     voxels[1], level=0.5, spacing=spacing
        # )
        # mesh_lv = Trimesh(vertices=verts_lv, faces=faces_lv)
        # verts_lv_myo, faces_lv_myo, *_ = marching_cubes(
        #     voxels[2], level=0.5, spacing=spacing
        # )
        # mesh_lv_myo = Trimesh(vertices=verts_lv_myo, faces=faces_lv_myo)
        # verts_rv, faces_rv, *_ = marching_cubes(
        #     voxels[3], level=0.5, spacing=spacing
        # )
        # mesh_rv = Trimesh(vertices=verts_rv, faces=faces_rv)

        # # create segment from intersections of surface meshes
        # y_myo = np.zeros(mesh_myo.vertices.shape[0], dtype=np.int64)
        # _, vertices_lv, _ = intersect_meshes_vertices(mesh_myo.vertices, mesh_lv.vertices)
        # y_myo.put(vertices_lv, 1)
        # vertices_freewall, vertices_sptm, _ = intersect_meshes_vertices(
        #     mesh_rv.vertices, mesh_lv_myo.vertices
        # )
        # _, vertices_freewall, _ = intersect_meshes_vertices(
        #     mesh_myo.vertices, mesh_rv.vertices[vertices_freewall]
        # )
        # y_myo.put(vertices_freewall, 2)
        # _, vertices_sptm, _ = intersect_meshes_vertices(
        #     mesh_myo.vertices, mesh_rv.vertices[vertices_sptm]
        # )
        # y_myo.put(vertices_sptm, 3)

        # # normalise the coordinates
        # mesh_myo.vertices = normalize_vertices(mesh_myo.vertices, size)

        # # sampling meshes for sub-meshes
        # verts_limit = self._subset([mesh_myo])
        # mesh_myo = Trimesh(
        #     vertices=mesh_myo.vertices[verts_limit[0]],
        #     vertex_normals=mesh_myo.vertex_normals[verts_limit[0]]
        # )
        # y_myo = y_myo[verts_limit[0]]

        # # convert meshes
        # mesh_dict = convert_trimesh(
        #     [mesh_myo],
        #     torch.tensor(y_myo, dtype=torch.long),
        #     point_cloud=True
        # )

        # return mesh_dict

    def _load_template_solo_mesh(self, voxels):
        """
            load the template myocardium mesh generated from Biobank data
            :param voxels:  arrays of voxel label maps
        """
        assert self.template_dir is not None, "Please specify the template directory."
        assert len(os.listdir(self.template_dir)) == 1, "There should be only one template mesh."
        # load template mesh as list of meshes for all frames
        meshes = [load(glob(f"{self.template_dir}/*.obj")[0]) for _ in range(voxels.shape[0])]

        # apply rescaling in the x and y direction to mimic the contraction of the heart"s chambers
        matrices = [
            np.identity(4) * [1-0.02*i,1-0.02*i,1,1] for i in range(voxels.shape[0])
        ]
        meshes = [mesh.apply_transform(matrix) for matrix, mesh in zip(matrices, meshes)]

        # convert trimesh to Meshes object
        meshes_dict = convert_trimesh(meshes)

        return meshes_dict