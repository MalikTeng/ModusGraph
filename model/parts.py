from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from monai.config.type_definitions import NdarrayOrTensor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.segresnet_block import (
    # ResBlock,
    get_conv_layer,
    get_upsample_layer
)
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils.enums import TransformBackends
from monai.utils import (
    UpsampleMode,
    ensure_tuple,
    LossReduction,
    BlendMode,
    PytorchPadMode,
    fall_back_tuple,
    look_up_option
)
from monai.transforms import (
    Compose,
    Affine,
    EnsureType,
    AsDiscrete,
    Activations,
    SpatialPadd,
    Affined,
    Invertd,
)
from monai.data import decollate_batch

from einops.einops import rearrange

from scipy.ndimage import center_of_mass

# from pytorch_metric_learning import losses, miners

from pytorch3d.structures import Meshes


__all__ = [
    # "RotationMatrix", "AffineMatrix",
    # "ResNetEncoder", "ResNetDecoder",
    # "VanillaEncoder", "VanillaDecoder",
    "ResNetDecoder", "ResBlock",
    "ConvLayer", 
    "SubdivideMeshes"
    ]


# class RotationMatrix(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.matrix = nn.Linear(3, 3)
#         self.registration_loss = nn.L1Loss(reduction="sum")

#     @staticmethod
#     def _find_anchor_points(x):  # shape: {B, C, H, W, D}
#         matrix = torch.empty((3, 3), device=x.device, dtype=x.dtype)
#         rv = torch.mean(x[:, 3], dim=0)
#         matrix[0, :3] = torch.as_tensor(center_of_mass(rv.detach().cpu().numpy()))
#         lv = torch.mean(x[:, 1], dim=0)
#         matrix[1, :3] = torch.as_tensor(center_of_mass(lv.detach().cpu().numpy()))
#         apex_s = (torch.max(torch.nonzero(lv)[:, 2])).cpu()
#         matrix[2, :3] = torch.as_tensor(
#             [*center_of_mass(lv[..., apex_s].detach().cpu().numpy()), apex_s]
#         )
#         return matrix  # matrix: {[x1,y1,z1,1], [x2,y2,z2,1], [x3,y3,z3,1]}

#     def _get_registration_loss(self, x_ct, x_mr):
#         matrix_ct = self._find_anchor_points(x_ct)  # matrix_ct.shape: {3, 3}
#         matrix_ct = matrix_ct / torch.norm(matrix_ct)
#         matrix_mr = self._find_anchor_points(x_mr)  # matrix_mr.shape: {3, 3}
#         matrix_mr = matrix_mr / torch.norm(matrix_mr)

#         trans_ct = self.trans_matrix(matrix_ct)  # trans_mr.shape: {3, 3}
#         loss = self.registration_loss(trans_ct, matrix_mr)

#         return loss

#     def _affine_transforms(self, img_size=None, is_training=True, keys=None):
#         affine = self.trans_matrix.weight.data
#         offset = self.trans_matrix.bias.data
#         affine = torch.vstack([affine, offset])
#         self.affine = torch.vstack(
#             [
#                 affine.T,
#                 torch.as_tensor([0, 0, 0, 1], device=affine.device, dtype=affine.dtype)
#             ]
#         )
#         if is_training:
#             return Affine(
#                 padding_mode="zeros",
#                 affine=self.affine,
#                 mode="nearest",
#                 image_only=True,
#                 device=affine.device
#             )
#         else:
#             if img_size is None:
#                 raise ValueError("While inferencing the size of input image must be identified.")
#             return Compose(
#                 [
#                     SpatialPadd(keys=keys, spatial_size=[2 * i for i in img_size]),
#                     Affined(
#                         keys=keys,
#                         padding_mode="zeros",
#                         affine=self.affine,
#                         mode="nearest",
#                     )
#                 ]
#             )

#     @staticmethod
#     def _reverse_affine_transforms(affine_transform, keys, orig_keys):
#         return Invertd(
#             keys=keys,
#             transform=affine_transform,
#             orig_keys=orig_keys,
#         )

#     def forward(self, x_ct, x_mr):
#         loss = self._get_registration_loss(x_ct, x_mr)
#         return loss

# class AffineMatrix(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.trans_matrix = nn.Linear(3, 3)
#         self.registration_loss = nn.L1Loss(reduction="sum")

#     @staticmethod
#     def _find_anchor_points(x):  # shape: {B, C, H, W, D}
#         matrix = torch.empty((3, 3), device=x.device, dtype=x.dtype)
#         rv = torch.mean(x[:, 3], dim=0)
#         matrix[0, :3] = torch.as_tensor(center_of_mass(rv.detach().cpu().numpy()))
#         lv = torch.mean(x[:, 1], dim=0)
#         matrix[1, :3] = torch.as_tensor(center_of_mass(lv.detach().cpu().numpy()))
#         apex_s = (torch.max(torch.nonzero(lv)[:, 2])).cpu()
#         matrix[2, :3] = torch.as_tensor(
#             [*center_of_mass(lv[..., apex_s].detach().cpu().numpy()), apex_s]
#         )
#         return matrix  # matrix: {[x1,y1,z1,1], [x2,y2,z2,1], [x3,y3,z3,1]}

#     def _get_registration_loss(self, x_ct, x_mr):
#         matrix_ct = self._find_anchor_points(x_ct)  # matrix_ct.shape: {3, 3}
#         matrix_mr = self._find_anchor_points(x_mr)  # matrix_mr.shape: {3, 3}
#         trans_ct = self.trans_matrix(matrix_ct)  # trans_mr.shape: {3, 3}
#         loss = self.registration_loss(trans_ct, matrix_mr)
#         return loss

#     def _affine_transforms(self, img_size=None, is_training=True, keys=None):
#         affine = self.trans_matrix.weight.data
#         offset = self.trans_matrix.bias.data
#         affine = torch.vstack([affine, offset])
#         self.affine = torch.vstack(
#             [
#                 affine.T,
#                 torch.as_tensor([0, 0, 0, 1], device=affine.device, dtype=affine.dtype)
#             ]
#         )
#         if is_training:
#             return Affine(
#                 padding_mode="zeros",
#                 affine=self.affine,
#                 mode="nearest",
#                 image_only=True,
#                 device=affine.device
#             )
#         else:
#             if img_size is None:
#                 raise ValueError("While inferencing the size of input image must be identified.")
#             return Compose(
#                 [
#                     SpatialPadd(keys=keys, spatial_size=[2 * i for i in img_size]),
#                     Affined(
#                         keys=keys,
#                         padding_mode="zeros",
#                         affine=self.affine,
#                         mode="nearest",
#                     )
#                 ]
#             )

#     @staticmethod
#     def _reverse_affine_transforms(affine_transform, keys, orig_keys):
#         return Invertd(
#             keys=keys,
#             transform=affine_transform,
#             orig_keys=orig_keys,
#         )

#     def forward(self, x_ct, x_mr):
#         loss = self._get_registration_loss(x_ct, x_mr)
#         return loss

# class ResNetEncoder(nn.Module):
#     """
#         ResNet encoder for mapping the input to a latent space
#         where a Fourier transformation decoder will apply to it.
#         The encoder supports 2D or 3D inputs

#         Args:
#             spatial_dims: spatial dimension of the input data. Defaults to 3.
#             init_filters: number of output channels for initial convolution layer. Defaults to 8.
#             in_channels: number of input channels for the network. Defaults to 1.
#             dropout_prob: probability of an element to be zero-ed. Defaults to ``None``.
#             act: activation type and arguments. Defaults to ``RELU``.
#             norm: feature normalization type and arguments. Defaults to ``GROUP``.
#             blocks_down: number of down sample blocks in each layer. Defaults to ``[1,2,2,4]``.
#             upsample_mode: [``"deconv"``, ``"nontrainable"``, ``"pixelshuffle"``]
#     """

#     def __init__(
#             self,
#             spatial_dims: int = 3,
#             init_filters: int = 8,
#             in_channels: int = 1,
#             dropout_prob: Optional[float] = None,
#             act: Union[Tuple, str] = ("RELU", {"inplace": True}),
#             norm: Union[Tuple, str] = "instance",
#             blocks_down: tuple = (1, 2, 2, 4)
#     ):
#         super(ResNetEncoder, self).__init__()

#         if spatial_dims not in (2, 3):
#             raise AssertionError("spatial_dims can only be 2 or 3")

#         self.spatial_dims = spatial_dims
#         self.init_filters = init_filters
#         self.in_channels = in_channels
#         self.dropout_prob = dropout_prob
#         self.blocks_down = blocks_down
#         self.norm = norm
#         self.act = get_act_layer(act)

#         self.conv_init = get_conv_layer(spatial_dims, in_channels, init_filters)
#         self.down_layers = self._make_down_layers()

#         if dropout_prob is not None:
#             self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

#     def _make_down_layers(self):
#         blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
#         down_layers = nn.ModuleList()
#         for i in range(len(blocks_down)):
#             layer_in_channels = filters * 2 ** i
#             pre_conv = (
#                 get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
#                 if i > 0
#                 else nn.Identity()
#             )
#             down_layer = nn.Sequential(
#                 pre_conv, *[ResBlock(spatial_dims, layer_in_channels, norm=norm) for _ in range(blocks_down[i])]
#             )
#             down_layers.append(down_layer)
#         return down_layers

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.conv_init(x)
#         if self.dropout_prob is not None:
#             x = self.dropout(x)

#         down_x = dict()
#         for i, down in enumerate(self.down_layers):
#             x = down(x)
#             down_x[len(self.blocks_down) - i - 1] = x

#         return down_x

# class VanillaEncoder(nn.Module):
#     def __init__(
#             self,
#             init_filters: int = 8,
#             in_channels: int = 1,
#             act: str = 'relu',
#             norm: str = 'batchnorm',
#             num_blocks: int = 3,
#     ):
#         super(VanillaEncoder, self).__init__()


#         self.init_filters = init_filters
#         self.in_channels = in_channels

#         self.act = act
#         self.norm = norm

#         self.init_block = nn.Conv3d(
#             in_channels, init_filters, 3, padding=1
#         )
#         self.blocks_down = self._make_block_layers(num_blocks)

#     def _make_block_layers(self, num_blocks):
#         act = nn.ReLU(inplace=True) if self.act == 'relu' else nn.PReLU()
#         blocks = nn.ModuleList()
#         for i in range(num_blocks):
#             in_channels = self.init_filters * 2 ** i
#             norm = \
#                 nn.BatchNorm3d(in_channels) if self.norm == 'batchnorm' else nn.InstanceNorm3d(in_channels)
#             conv_layer = \
#                 nn.Conv3d(
#                     in_channels // 2, in_channels, 3, 2, padding=1,
#                     bias=True if self.norm == 'batchnorm' else False
#                 ) if i > 0 else nn.Identity()
#             blocks.append(
#                 nn.Sequential(
#                     conv_layer, norm, act
#                 )
#             )

#         return blocks

#     def forward(self, x):
#         x = self.init_block(x)

#         down_x = []
#         for block_down in self.blocks_down:
#             x = block_down(x)
#             down_x.append(x)

#         return x, down_x

# class VanillaDecoder(nn.Module):
#     def __init__(
#             self,
#             init_filters: int = 8,
#             out_channels: int = 1,
#             act: str = 'relu',
#             norm: str = 'instance',
#             use_conv_final: bool = True,
#             num_blocks: int = 4,

#     ):
#         super(VanillaDecoder, self).__init__()

#         self.init_filters = init_filters

#         self.act = act
#         self.norm = norm

#         self.num_blocks = num_blocks
#         self.up_blocks = self._make_block_layers(num_blocks)
#         # self.extra_blocks_res, self.extra_blocks_up = self._make_extra_blocks(num_blocks)
#         if use_conv_final:
#             self.final_layer = nn.Sequential(
#                 nn.Conv3d(
#                     init_filters, out_channels, 1,
#                     bias=True if norm == 'batchnorm' else False
#                 ),
#                 nn.BatchNorm3d(out_channels) if self.norm == 'batchnorm'
#                 else nn.InstanceNorm3d(out_channels),
#                 # nn.Tanh()
#                 nn.Sigmoid()
#             )
#         else:
#             self.final_layer = None

#     def _make_block_layers(self, num_blocks):
#         act = nn.ReLU(inplace=True) if self.act == 'relu' else nn.PReLU()
#         blocks = nn.ModuleList()
#         for i in range(num_blocks):
#             in_channels = self.init_filters * 2 ** (num_blocks - i)
#             norm = \
#                 nn.BatchNorm3d(in_channels) if self.norm == 'batchnorm' else nn.InstanceNorm3d(in_channels)
#             conv_layer = nn.Conv3d(
#                 2 * in_channels if i > 0 else in_channels,
#                 in_channels // 2,
#                 1,
#                 bias=True if self.norm == 'batchnorm' else False
#             )
#             sample_layer = nn.Upsample(
#                 size=None, scale_factor=2, mode='trilinear', align_corners=True
#             )
#             blocks.append(
#                 nn.Sequential(
#                     conv_layer, norm, act, sample_layer,
#                 )
#             )

#         return blocks

#     def _make_extra_blocks(self, num_blocks):
#         act = nn.ReLU(inplace=True) if self.act == 'relu' else nn.PReLU()
#         res_blocks, up_blocks = nn.ModuleList(), nn.ModuleList()
#         for i in range(num_blocks):
#             in_channels = self.init_filters
#             norm = \
#                 nn.BatchNorm3d(in_channels) if self.norm == 'batchnorm' \
#                     else nn.InstanceNorm3d(in_channels)
#             conv_layer = nn.Conv3d(
#                 in_channels, in_channels, 3, padding=1,
#                 bias=True if self.norm == 'batchnorm' else False
#             )
#             sample_layer = nn.Upsample(
#                 size=None, scale_factor=2, mode='trilinear', align_corners=True
#             )
#             res_blocks.append(
#                 nn.Sequential(
#                     conv_layer, norm, act
#                 )
#             )
#             up_blocks.append(sample_layer)

#             return res_blocks, up_blocks

#     def forward(self, x, down_x):
#         x = self.up_blocks[0](x)
#         for i, block_up in enumerate(self.up_blocks[1:]):
#             x = torch.cat([x, down_x[i + 1]], dim=1)
#             x = block_up(x)

#         # for extra_block_res, extra_block_up \
#         #         in zip(self.extra_blocks_res, self.extra_blocks_up):
#         #     x = extra_block_up(extra_block_res(x) + x)

#         if self.final_layer:
#             x = self.final_layer(x)

#         return x


class ResNetDecoder(nn.Module):
    def __init__(
        self,
        init_filters: int = 8,
        out_channels: int = 1,
        norm: str = 'instance',
        act: str = 'relu',
        use_conv_final: bool = True,
        num_layers_blocks: tuple = (1, 1, 1),
        ):
        super(ResNetDecoder, self).__init__()

        self.init_filters = init_filters
        self.act = act
        self.norm = norm
        self.up_blocks, self.res_blocks = self._make_block_layers(num_layers_blocks)

        assert use_conv_final == True, 'use_conv_final must be True'
        self.final_layer = ConvLayer(
            init_filters, out_channels, 1, 1, 0,
            norm, 'sigmoid'
        )

    def _make_block_layers(self, num_layers_blocks):
        up_blocks, res_blocks = nn.ModuleList(), nn.ModuleList()
        for i in range(len(num_layers_blocks), 0, -1):
            in_channels = self.init_filters * 2 ** i
            out_channels = in_channels // 2
            conv_layer = ConvLayer(
                in_channels, out_channels, 3, 1, 1,
                self.norm, self.act
            )
            res_blocks.append(ResBlock(
                in_channels, in_channels, 
                self.norm, self.act, 
                num_layers=num_layers_blocks[i - 1]
                ))
            sample_layer = nn.Upsample(
                size=None, scale_factor=2, mode='trilinear', align_corners=True
                )
            up_blocks.append(nn.Sequential(conv_layer, sample_layer))

        return up_blocks, res_blocks

    def forward(self, x):
        up_x = []
        for i, (up_block, res_block) in enumerate(zip(self.up_blocks, self.res_blocks)):
            x, _ = res_block(x)
            x = up_block(x)
            up_x.append(x)
        x = self.final_layer(x)

        return x, up_x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm, act, 
        num_layers,
        ):
        super(ResBlock, self).__init__()
        assert out_channels is not None

        self.blocks = []
        for i in range(num_layers):
            if i == 0:
                conv_layer = ConvLayer(
                    in_channels, out_channels, 3, 1, 1,
                    norm, act
                )
            else:
                conv_layer = ConvLayer(
                    out_channels, out_channels, 3, 1, 1,
                    norm, act
                )
            self.blocks.append(conv_layer)

        self.blocks = nn.Sequential(*self.blocks)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x_ = self.blocks(x)
        # x = torch.cat([x, x_], dim=1)
        x = x + x_

        return x, x_


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels, out_channels,
        kernel_size, stride, padding,
        norm, act, 
    ):
        super(ConvLayer, self).__init__()

        self.conv_layer = nn.Conv3d(
            in_channels, out_channels, 
            bias=True if norm == 'batchnorm' else False,
            kernel_size=kernel_size, stride=stride, padding=padding
        )

        if norm == 'batchnorm':
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm3d(out_channels)
        else:
            raise NotImplementedError
        
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'prelu':
            self.act = nn.PReLU()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.norm(x)
        x = self.act(x)

        return x


class SubdivideMeshes(nn.Module):
    """
    Subdivide a triangle mesh by adding a new vertex at the center of each edge
    and dividing each face into four new faces. Vectors of vertex
    attributes can also be subdivided by averaging the values of the attributes
    at the two vertices which form each edge. This implementation
    preserves face orientation - if the vertices of a face are all ordered
    counter-clockwise, then the faces in the subdivided meshes will also have
    their vertices ordered counter-clockwise.

    If meshes is provided as an input, the initializer performs the relatively
    expensive computation of determining the new face indices. This one-time
    computation can be reused for all meshes with the same face topology
    but different vertex positions.
    """

    def __init__(self, meshes=None, faces_labels=None) -> None:
        """
        Args:
            meshes: Meshes object or None. If a meshes object is provided,
                the first mesh is used to compute the new faces of the
                subdivided topology which can be reused for meshes with
                the same input topology.
        """
        super(SubdivideMeshes, self).__init__()

        self._N = -1
        if meshes is not None:
            # This computation is on indices, so gradients do not need to be
            # tracked.
            mesh = meshes[0]
            assert faces_labels is not None, "faces_labels must be provided if meshes is provided"
            faces_labels = list(
                faces_index.nonzero().flatten() for faces_index in faces_labels.squeeze()
                )
            with torch.no_grad():
                subdivided_faces, self.faces_labels = self.subdivide_faces(mesh, faces_labels)
                if subdivided_faces.shape[1] != 3:
                    raise ValueError("faces can only have three vertices")
                self.register_buffer("_subdivided_faces", subdivided_faces)

    def subdivide_faces(self, meshes, faces_labels):
        r"""
        Args:
            meshes: a Meshes object.
            faces_labels: list of (1, F_n) shape face labels for each submesh.

        Returns:
            subdivided_faces_packed: (4*sum(F_n), 3) shape LongTensor of
            original and new faces.

        Refer to pytorch3d.structures.meshes.py for more details on packed
        representations of faces.

        Each face is split into 4 faces e.g. Input face
        ::
                   v0
                   /\
                  /  \
                 /    \
             e1 /      \ e0
               /        \
              /          \
             /            \
            /______________\
          v2       e2       v1

          faces_packed = [[0, 1, 2]]
          faces_packed_to_edges_packed = [[2, 1, 0]]

        `faces_packed_to_edges_packed` is used to represent all the new
        vertex indices corresponding to the mid-points of edges in the mesh.
        The actual vertex coordinates will be computed in the forward function.
        To get the indices of the new vertices, offset
        `faces_packed_to_edges_packed` by the total number of vertices.
        ::
            faces_packed_to_edges_packed = [[2, 1, 0]] + 3 = [[5, 4, 3]]

        e.g. subdivided face
        ::
                   v0
                   /\
                  /  \
                 / f0 \
             v4 /______\ v3
               /\      /\
              /  \ f3 /  \
             / f2 \  / f1 \
            /______\/______\
           v2       v5       v1

           f0 = [0, 3, 4]
           f1 = [1, 5, 3]
           f2 = [2, 4, 5]
           f3 = [5, 4, 3]

        """
        meshes = meshes.submeshes([faces_labels])
        subdivided_faces_packed = []
        num_labels_submeshes = []
        for mesh in meshes:
            verts_packed = mesh.verts_packed()
            with torch.no_grad():
                faces_packed = mesh.faces_packed()
                faces_packed_to_edges_packed = (
                    mesh.faces_packed_to_edges_packed() + verts_packed.shape[0]
                    )

                f0 = torch.stack(
                    [
                        faces_packed[:, 0],
                        faces_packed_to_edges_packed[:, 2],
                        faces_packed_to_edges_packed[:, 1],
                    ],
                    dim=1,
                    )
                f1 = torch.stack(
                    [
                        faces_packed[:, 1],
                        faces_packed_to_edges_packed[:, 0],
                        faces_packed_to_edges_packed[:, 2],
                    ],
                    dim=1,
                    )
                f2 = torch.stack(
                    [
                        faces_packed[:, 2],
                        faces_packed_to_edges_packed[:, 1],
                        faces_packed_to_edges_packed[:, 0],
                    ],
                    dim=1,
                    )
                f3 = faces_packed_to_edges_packed
                subdivided_faces_packed.append(torch.cat(
                    [f0, f1, f2, f3], dim=0
                    ))  # (4*sum(F_n), 3)
                num_labels_submeshes.append(f0.shape[0] + f1.shape[0] + f2.shape[0] + f3.shape[0])
                
        subdivided_faces_packed = torch.concat(subdivided_faces_packed, dim=0)

        num_labels_submeshes = torch.as_tensor(num_labels_submeshes)
        cum_num_labels_submeshes = num_labels_submeshes.cumsum(0)
        subdivided_faces_labels = torch.zeros((1, len(num_labels_submeshes), cum_num_labels_submeshes[-1]), dtype=torch.long)
        for i, (cum_num_labels, num_labels) in enumerate(zip(cum_num_labels_submeshes, num_labels_submeshes)):
            subdivided_faces_labels[0, i, cum_num_labels - num_labels:cum_num_labels] = torch.arange(cum_num_labels-num_labels, cum_num_labels)

        return subdivided_faces_packed, subdivided_faces_labels

    def forward(self, meshes, feats=None):
        """
        Subdivide a batch of meshes by adding a new vertex on each edge, and
        dividing each face into four new faces. New meshes contains two types
        of vertices:
        1) Vertices that appear in the input meshes.
           Data for these vertices are copied from the input meshes.
        2) New vertices at the midpoint of each edge.
           Data for these vertices is the average of the data for the two
           vertices that make up the edge.

        Args:
            meshes: Meshes object representing a batch of meshes.
            feats: Per-vertex features to be subdivided along with the verts.
                Should be parallel to the packed vert representation of the
                input meshes; so it should have shape (V, D) where V is the
                total number of verts in the input meshes. Default: None.

        Returns:
            2-element tuple containing

            - **new_meshes**: Meshes object of a batch of subdivided meshes.
            - **new_feats**: (optional) Tensor of subdivided feats, parallel to the
              (packed) vertices of the subdivided meshes. Only returned
              if feats is not None.

        """
        self._N = len(meshes)

        return self.subdivide_homogeneous(meshes, feats)

    def subdivide_homogeneous(self, meshes, feats=None):
        """
        Subdivide verts (and optionally features) of a batch of meshes
        where each mesh has the same topology of faces. The subdivided faces
        are precomputed in the initializer.

        Args:
            meshes: Meshes object representing a batch of meshes.
            feats: Per-vertex features to be subdivided along with the verts.

        Returns:
            2-element tuple containing

            - **new_meshes**: Meshes object of a batch of subdivided meshes.
            - **new_feats**: (optional) Tensor of subdivided feats, parallel to the
              (packed) vertices of the subdivided meshes. Only returned
              if feats is not None.
        """
        verts = meshes.verts_padded()  # (N, V, D)
        edges = meshes[0].edges_packed()

        # The set of faces is the same across the different meshes.
        new_faces = self._subdivided_faces.view(1, -1, 3).expand(self._N, -1, -1)
        new_faces_labels = self.faces_labels.expand(self._N, -1, -1)

        # Add one new vertex at the midpoint of each edge by taking the average
        # of the vertices that form each edge.
        new_verts = verts[:, edges].mean(dim=2)
        new_verts = torch.cat([verts, new_verts], dim=1)  # (sum(V_n)+sum(E_n), 3)
        new_feats = None

        # Calculate features for new vertices.
        if feats is not None:
            if feats.dim() == 2:
                # feats is in packed format, transform it from packed to
                # padded, i.e. (N*V, D) to (N, V, D).
                feats = feats.view(verts.size(0), verts.size(1), feats.size(1))
            if feats.dim() != 3:
                raise ValueError("features need to be of shape (N, V, D) or (N*V, D)")

            # Take average of the features at the vertices that form each edge.
            new_feats = feats[:, edges].mean(dim=2)
            new_feats = torch.cat([feats, new_feats], dim=1)  # (sum(V_n)+sum(E_n), 3)

        new_meshes = Meshes(verts=new_verts, faces=new_faces)

        return new_meshes, new_faces_labels, new_feats