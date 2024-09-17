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
    "ResNetDecoder", "ResBlock",
    "ConvLayer", 
    # "SubdivideMeshes"
    ]


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
        self.init_layer = ConvLayer(
            out_channels, init_filters * 2 ** len(num_layers_blocks), 3, 1, 1,
            norm, act
        )
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
        x = self.init_layer(x)
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

    def __init__(self, meshes=None) -> None:
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
            with torch.no_grad():
                subdivided_faces = self.subdivide_faces(mesh)
                if subdivided_faces.shape[1] != 3:
                    raise ValueError("faces can only have three vertices")
                self.register_buffer("_subdivided_faces", subdivided_faces)

    def subdivide_faces(self, mesh):
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
            new_faces_packed = torch.cat([f0, f1, f2, f3], dim=0)
                
        return new_faces_packed

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
        new_faces = self._subdivided_faces.expand(self._N, -1, -1)

        # Add one new vertex at the midpoint of each edge by taking the average
        # of the vertices that form each edge.
        new_verts = verts[:, edges].mean(dim=2)
        new_verts = torch.cat([verts, new_verts], dim=1)  # (sum(V_n)+sum(E_n), 3)

        new_meshes = Meshes(verts=new_verts, faces=new_faces)

        return new_meshes