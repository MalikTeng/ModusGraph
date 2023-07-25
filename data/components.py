from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from torch import Tensor
import os

import torch
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
    'Identityd',
    'ConvertToMultiChannelBasedOnACDCClassesd',
    'AddRVMyoToACDCClassesd',
    'ConvertToMultiChannelBasedOnMMWHSClassesd',
    'ConvertToMultiChannelBasedOnSCOTHEARTClassesd',
    'MergeMultiChanneld',
    'CopySliceToSpacingd', 'LabelShuffled', 'SliceShiftd',
    'GenerateMeshesd'
]


class Identityd(MapTransform):
    """
        Identity transform
    """

    def __init__(self, keys, allow_missing_keys: bool = False):
        MapTransform.__init__(self, keys, allow_missing_keys)

    def __call__(self, data):
        return data


class ConvertToMultiChannelBasedOnACDCClassesd(MapTransform):
    """
        Convert labels to multi channels based on ACDC classes:
        Background: 0, LV-cavity is 3, LV-myo is 2, and RV-cavity is 1
    """

    def __init__(self, keys, allow_missing_keys: bool = False):
        MapTransform.__init__(self, keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = d[key].get_array(np.ndarray)
            result = np.vstack(
                [np.where(result == k, v, 0) for k, v in zip([0, 3, 2, 1], [0, 1, 1, 1])],
            )
            d[key] = MetaTensor(
                torch.from_numpy(result), array=d[key].array, meta=d[key].meta,
                applied_operations=[
                    *d[key].applied_operations,
                    {
                        'class': 'ConvertToMultiChannelBasedOnACDCClassesd',
                        'orig_shape': d[key].as_tensor().shape
                    }
                ]
            )
        return d


class AddRVMyoToACDCClassesd(MapTransform):
    """
        Add Rv-myo to ACDC label maps by inflating the RV cavity
    """

    def __init__(self, keys, allow_missing_keys: bool = False):
        MapTransform.__init__(self, keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = d[key].as_tensor()
            rv_myo = F.interpolate(
                result[-1:].float(), scale_factor=1.20, mode='nearest-exact'
            ).squeeze(0)
            center_rv = torch.tensor(
                measurements.center_of_mass(result[-1].numpy())
            )
            center_rv_myo = torch.tensor(
                measurements.center_of_mass(rv_myo.numpy())
            )
            shift = [i.item() for i in torch.round(center_rv - center_rv_myo).int()]
            rv_myo = torch.roll(rv_myo, shift, dims=(0, 1, 2))
            rv_myo = rv_myo[:result.shape[-3], :result.shape[-2], :result.shape[-1]]
            result = torch.cat([result, rv_myo.unsqueeze(0)], dim=0)    #[LV, LV-myo, RV, RV-myo]

            d[key] = MetaTensor(
                result.long(), array=d[key].array, meta=d[key].meta,
                applied_operations=[
                    *d[key].applied_operations,
                    {
                        'class': 'ConvertToMultiChannelBasedOnACDCClassesd',
                        'orig_shape': d[key].as_tensor().shape
                    }
                ]
            )

        return d
    
    
class ConvertToMultiChannelBasedOnMMWHSClassesd(MapTransform):
    """
        Convert labels to multi channels based on MMWHS classes:
        Background: 0,
        LV: 500, LV-myo: 205, RV: 600,
        LA: 420, RA: 550, AV: 820, PV: 850
    """

    def __init__(self, keys, allow_missing_keys: bool = False):
        MapTransform.__init__(self, keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = d[key].get_array(np.ndarray)
            result = np.concatenate(
                [
                    np.where(result == i, 1, 0)
                    for i in [0, 500, 205, 600, 420, 550, 820, 850]
                ],
                axis=0, dtype=np.float32
            )
            d[key] = MetaTensor(
                torch.from_numpy(result), meta=d[key].meta,
                applied_operations=[
                    *d[key].applied_operations,
                    {
                        'class': 'ConvertToMultiChannelBasedOnMMWHSClassesd',
                        'orig_shape': d[key].as_tensor().shape
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


class CopySliceToSpacingd(MapTransform):
    """
        Duplicate slices in the through-plane direction to make the spacing a unit long
    """

    def __init__(
            self, keys, allow_missing_keys: bool = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            array = d[key].as_tensor().float()
            H, W, D = array.shape[-3:]
            dx, dy, dz = np.round(d[key].meta['pixdim'][1:4]).astype(int)
            temp_array = F.interpolate(
                array.unsqueeze(0), (dx * H, dy * W, dz * D),
                mode='nearest-exact', antialias=False
            )
            d[key].meta['pixdim'] = 1.0
            d[key] = MetaTensor(
                temp_array.squeeze(0), meta=d[key].meta,
                applied_operations=[
                    *d[key].applied_operations,
                    {
                        'class': 'CopySliceToSpacing',
                        'orig_size': array.shape
                    }
                ]
            )

        return d


class LabelShuffled(MapTransform):
    """
        Randomly shuffle labels for masks
    """

    def __init__(self, keys, allow_missing_keys: bool = False):
        MapTransform.__init__(self, keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            array = d[key]
            channels = np.arange(1, 4)
            np.random.shuffle(channels)
            d[key] = array[[0, *channels]]

        return d


class SliceShiftd(MapTransform):
    """
        Dropping slices by the factor of 2,
        translating selected SAX slices by a random distance on a random direction (X-axis or Y-axis),
        and smoothing surface by interpolation.
    """

    def __init__(
            self,
            keys,
            factor: int = 4,
            translate_range: tuple = (1, 3),
            allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            translate_range: the range of a randomly chosen distance for the translation.
            allow_missing_keys: don't raise exception if key is missing.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        np.random.seed(505)
        self.factor = factor
        self.translate_range = translate_range

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            array = d[key]
            H, W, D = array.shape[-3:]

            # drop slices by a factor of self.factor
            array = np.argmax(array, axis=0)
            array = np.pad(array, (0, (array.shape[-1] - 1) % self.factor), mode='constant')
            array = (array[..., ::self.factor]).astype(int)

            # translating selected slices by a distance
            for idx in np.arange(0, array.shape[-1], 4 * self.factor):
                offset = np.random.randint(self.translate_range[0], self.translate_range[1], 2)
                orientation = np.random.randint(-1, 1, 2)
                translate_values = (orientation * offset).tolist()
                array[..., idx] = shift(array[..., idx], translate_values)

            # smoothing surface
            array = torch.nn.functional.interpolate(
                torch.from_numpy(array[None, None].astype(np.float32)),
                size=(H, W, D),
                mode='nearest',
            ).squeeze().cpu().numpy()
            d[key] = np.asarray(
                [np.where(array == i, 1, 0) for i in range(4)]
            )

        return d


def normalize_vertices(vertices: Tensor, size: Union[Tensor, int]):
    """
    Resize the values of coordinates of vertices as in the range of -1 to 1
    (Normalized Device Coordinates)
    :param vertices:    set of vertices' coordinates values
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
            faces=[torch.IntTensor(mesh.faces.copy()) for mesh in meshes],
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
                    d[f"{key}_point_clouds"] = self._generate_target_multimesh(voxels, self.size)
                else:
                    spacing = (d[key].pixdim).float().numpy()
                    d[f"{key}_point_clouds"] = self._generate_target_solomesh(
                        voxels, spacing, self.size,
                        load_dir=d[key].meta['filename_or_obj']
                    )
            # do nothing to the image, but load the template mesh
            elif "image" in key:
                voxels = d[key].as_tensor()
                if self.one_or_multi.lower() == "multi":
                    d[f"{key[:2]}_template_meshes"] = self._load_template_multimesh()
                else:
                    d[f"{key[:2]}_template_meshes"] = self._load_template_solomesh(self.size)

        return d

    def _subset(self, meshes: list) -> list:
        """
        Select a number of vertices from each mesh, to cut down memory usage
        :param  meshes: un-sampled mesh with labels of every vertex, List[Trimesh]
        :return vertex indices of sub-meshes, List[np.ndarray]
        """
        num_verts = sum([mesh.vertices.shape[0] for mesh in meshes])
        assert num_verts > self.point_limit, "The number of vertices is less than the point limit, cannot do oversampling."
        
        point_limit = np.asarray(
            [8_000, 20_000, 10_000, 6_000, 6_000, 3_500],   # the number 53_500 comes from here
            dtype=np.int64
        )
        assert self.point_limit == sum(point_limit), f"The super_params: point_limit should be {sum(point_limit)}."
        point_limit = [
            np.random.choice(
                np.arange(mesh.vertices.shape[0]), face_limit_per_mesh, replace=False
            ) for mesh, face_limit_per_mesh in zip(meshes, point_limit)
        ]

        return point_limit

    def _generate_target_multimesh(
            self, voxels: Tensor, size: Union[Tensor, int],
        ) -> dict:
        """
        Generate surface mesh as trimesh.Trimesh from voxel using Marching Cubes
        for MMWHS with seven labels
        :param voxels:  arrays of voxel label maps
        :param size:    the largest size of label maps
        """
        def apply_marching_cubes(voxels: Tensor) -> list:
            assert voxels.shape[0] > 1, "check your labels, which should be multi-class"
            classes = voxels.shape[0]
            meshes = []
            for c in range(classes):
                # mesh = matrix_to_marching_cubes(voxels[c].T)  # (z, y, x) -> (x, y, z)
                mesh = matrix_to_marching_cubes(voxels[c])
                meshes.append(mesh)

            return meshes

        def apply_norm(meshes: list) -> list:
            for i in range(len(meshes)):
                meshes[i].vertices = normalize_vertices(meshes[i].vertices, size)

            return meshes

        def apply_sampling(meshes: list, verts_limit: list) -> list:
            point_clouds = []
            for i, mesh in enumerate(meshes):
                point_cloud = PointCloud(
                    vertices=mesh.vertices[verts_limit[i]],
                    metadata={"vertex_normals": mesh.vertex_normals[verts_limit[i]]}
                )
                point_clouds.append(point_cloud)

            return point_clouds

        # create mesh from voxel
        meshes = apply_marching_cubes(voxels.numpy())
        # normalise the coordinates
        meshes = apply_norm(meshes)
        # sampling meshes for sub-meshes
        point_clouds = apply_sampling(meshes, self._subset(meshes))
        # convert meshes
        point_clouds_dict = convert_trimesh(point_clouds)

        return point_clouds_dict

    def _load_template_multimesh(self, ) -> dict:
        """
        get the template multi-part mesh generated for a patient in Scotheart
        :return:
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

    def _generate_target_solomesh(
            self, voxels, spacing, size, **kwargs
        ):
        """
            Generate surface mesh as trimesh.Trimesh from voxel using Marching Cubes
            and labels for each and every vertex
        :param voxels:  arrays of voxel label maps
        :param spacing: pixdim in all direction
        :param size:    the largest size of label maps
        """
        def intersect_meshes_vertices(mesh_vertices_a, mesh_vertices_b):
            mesh_vertices_a = {
                '{:.1f}-{:.1f}-{:.1f}'.format(*coord): idx
                for idx, coord in enumerate(mesh_vertices_a)
            }
            mesh_vertices_b = {
                '{:.1f}-{:.1f}-{:.1f}'.format(*coord): idx
                for idx, coord in enumerate(mesh_vertices_b)
            }
            indices_a = np.zeros(len(mesh_vertices_a), dtype=np.int64)
            indices_b = np.zeros(len(mesh_vertices_b), dtype=np.int64)
            for coord in mesh_vertices_b.keys():
                if coord in mesh_vertices_a:
                    indices_a.put(mesh_vertices_a[coord], 1)
                    indices_b.put(mesh_vertices_b[coord], 1)
            indices_c = np.nonzero(indices_a.copy())[0]
            indices_a = np.nonzero(1 - indices_a)[0]
            indices_b = np.nonzero(1 - indices_b)[0]

            return indices_a, indices_c, indices_b

        def intersect_meshes_faces(mesh, exclude_vertices=None, include_vertices=None):
            if include_vertices is not None:
                exclude_vertices = np.delete(np.arange(mesh.vertices.shape[0]), include_vertices)
            elif exclude_vertices is not None:
                pass
            else:
                raise Exception('No valid index of vertices were received.')
            exclude_faces = mesh.vertex_faces[exclude_vertices]
            exclude_faces, c = np.unique(exclude_faces.flatten(), return_counts=True)
            exclude_faces = exclude_faces[c == 3]
            include_faces = np.delete(np.arange(mesh.faces.shape[0]), exclude_faces)

            return [include_faces, exclude_faces]

        def annotate_vertices(mesh, parts):
            mesh_points = {'{:.1f}-{:.1f}-{:.1f}'.format(*coord): idx for idx, coord in enumerate(mesh.vertices)}
            labels_all = np.full(len(mesh_points), 2, dtype=np.int64)
            for l, part in enumerate(parts):
                vertices = {'{:.1f}-{:.1f}-{:.1f}'.format(*coord): idx for idx, coord in enumerate(part.vertices)}
                indices = []
                for coord in vertices.keys():
                    if coord in mesh_points:
                        indices.append(mesh_points[coord])
                labels_all.put(indices, l)

            return labels_all

        # index slice the voxels by spacing
        voxels = voxels.cpu().numpy()

        if self.dataset == 'scotheart':
            # flip voxels against YZ plane (works for SCOTHEART)
            voxels = np.flip(voxels, axis=1)

        # create mesh from voxel
        verts_myo, faces_myo, *_ = marching_cubes(
            np.sum(voxels[[2, -1]], axis=0), level=0.5, spacing=spacing
        )
        mesh_myo = Trimesh(vertices=verts_myo, faces=faces_myo)
        verts_lv, faces_lv, *_ = marching_cubes(
            voxels[1], level=0.5, spacing=spacing
        )
        mesh_lv = Trimesh(vertices=verts_lv, faces=faces_lv)
        verts_lv_myo, faces_lv_myo, *_ = marching_cubes(
            voxels[2], level=0.5, spacing=spacing
        )
        mesh_lv_myo = Trimesh(vertices=verts_lv_myo, faces=faces_lv_myo)
        verts_rv, faces_rv, *_ = marching_cubes(
            voxels[3], level=0.5, spacing=spacing
        )
        mesh_rv = Trimesh(vertices=verts_rv, faces=faces_rv)

        # annotate vertices in each mesh
        y_myo = np.zeros(mesh_myo.vertices.shape[0], dtype=np.int64)
        _, vertices_lv, _ = intersect_meshes_vertices(mesh_myo.vertices, mesh_lv.vertices)
        y_myo.put(vertices_lv, 1)
        vertices_freewall, vertices_sptm, _ = intersect_meshes_vertices(
            mesh_rv.vertices, mesh_lv_myo.vertices
        )
        _, vertices_freewall, _ = intersect_meshes_vertices(
            mesh_myo.vertices, mesh_rv.vertices[vertices_freewall]
        )
        y_myo.put(vertices_freewall, 2)
        _, vertices_sptm, _ = intersect_meshes_vertices(
            mesh_myo.vertices, mesh_rv.vertices[vertices_sptm]
        )
        y_myo.put(vertices_sptm, 3)

        # normalise the coordinates
        mesh_myo.vertices = normalize_vertices(mesh_myo.vertices, size)

        # sampling meshes for sub-meshes
        verts_limit = self._subset([mesh_myo])
        mesh_myo = Trimesh(
            vertices=mesh_myo.vertices[verts_limit[0]],
            vertex_normals=mesh_myo.vertex_normals[verts_limit[0]]
        )
        y_myo = y_myo[verts_limit[0]]

        # convert meshes
        mesh_dict = convert_trimesh(
            [mesh_myo],
            torch.tensor(y_myo, dtype=torch.long),
            point_cloud=True
        )

        return mesh_dict

    def _load_template_solomesh(self, size):
        """
            generate a template bi-ventricles mesh from Biobank
            and labels for each and every vertex
        """
        # get list of vertices, labels and faces
        mesh = load(
            "/home/yd21/Documents/ModusGraph/template/cap/initial_mesh.obj"
        )
        y_ = color_mapping(mesh.visual.vertex_colors)

        # convert trimesh to Meshes object
        mesh_dict = convert_trimesh(
            [mesh],
            torch.tensor(y_, dtype=torch.long),
            point_cloud=False
        )

        return mesh_dict

        # vertices = pd.read_csv(
        #     '/home/yd21/Documents/ModusGraph/template/biobank/'
        #     'VERTICES.CSV',
        #     index_col=None,
        #     header=None
        # )
        # vertices = np.asarray(vertices.values, dtype=np.float)
        # vertices = normalize_vertices(vertices, size)
        # faces = pd.read_csv(
        #     '/home/yd21/Documents/ModusGraph/template/biobank/'
        #     'FACES.CSV',
        #     index_col=None,
        #     header=None,
        #     names=['vert1', 'vert2', 'vert3', 'label']
        # )
        # y = np.asarray(faces.values[:, -1], dtype=str)
        # faces = np.asarray(faces.values[:, :3] - 1, dtype=np.int64)

        # # create the mesh
        # mesh = Trimesh(vertices=vertices, faces=faces)
        # # exclude valves' faces and vertices
        # faces_valve = np.in1d(y, ['MV', 'AV', 'PV', 'TV', 'SPTM_OUTLINE'])
        # mesh = mesh.submesh([~faces_valve])[0]

        # # convert face label to integer and annotate vertex accordingly
        # y = y[~faces_valve]
        # y_ = np.zeros(mesh.vertices.shape[0], dtype=np.int64)
        # for k, v in zip(np.unique(y), np.arange(4)):
        #     select_verts = np.unique(mesh.faces[y == k])
        #     y_.put(select_verts, v)

        # # rotate mesh to a standard orientation
        # center_coord = mesh.center_mass
        # mesh.vertices = mesh.vertices - center_coord[None]
        # R0 = rotation_matrix(np.pi / 4, (1, 0, 0))
        # R1 = rotation_matrix(np.pi / 4, (0, 1, 0))
        # R2 = rotation_matrix(np.pi / 4, (0, 0, 1))
        # mesh.apply_transform(concatenate_matrices(R2, R1, R0))
        

def color_mapping(labels):
    color_map = {
        5: 0,
        4: 1,
        3: 2,
        8: 3,
    }
    colors = np.zeros(
        labels.shape[0], dtype=np.int32
    )
    for c, n in color_map.items():
        colors[np.sum(np.cumsum(labels//255, axis=1), axis=1) == c] = n

    return colors
