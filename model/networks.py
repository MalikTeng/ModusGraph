import os, sys
# if debugging this script, change working dircetory to the root of the project
os.chdir(os.getcwd())
sys.path.append(os.getcwd())
from skimage.measure import subdivide_polygon
from torch_geometric.typing import Dict
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torch.utils.cpp_extension import load
from torch_geometric.nn import Sequential, GCNConv, ChebConv
from torch_geometric.utils import degree
from torch_geometric.utils.to_dense_adj import to_dense_adj
from torch_geometric_temporal.nn.recurrent import GConvGRU, DyGrEncoder
from pytorch3d.structures import Meshes
from einops.einops import rearrange, repeat
from einops.layers.torch import Rearrange
# import FastGeodis

from model.parts import *


__all__ = ["VoxelProcessingModule", "ModalityHandle", "RStGCN"]


class VoxelProcessingModule(nn.Module):
    def __init__(
        self,
        modality_handle: nn.Module,
        use_conv_final: bool = True,
        num_up_blocks: tuple = (1, 1, 1),
    ):
        super(VoxelProcessingModule, self).__init__()

        self.modality_handle = modality_handle
        assert len(num_up_blocks) == len(modality_handle.num_init_blocks) - 1, "number of ResNet layers must be one layer less than the number of modality handle"
        self.resnet_decoder = ResNetDecoder(
            modality_handle.init_filters, modality_handle.out_channels,
            modality_handle.norm, modality_handle.act, 
            use_conv_final, num_up_blocks
        )

    def forward(self, x):
        x, x_seg = self.modality_handle(x)
        x, _ = self.resnet_decoder(x)

        return x, x_seg

# code for the paper
# class VoxelProcessingModule(nn.Module):
#     def __init__(
#         self,
#         modality_handle: nn.Module,
#         use_conv_final: bool = True,
#         num_up_blocks: tuple = (1, 1, 1),
#         ):
#         super(VoxelProcessingModule, self).__init__()

#         self.modality_handle = modality_handle
#         self.decoder = ResNetDecoder(
#             modality_handle.init_filters, modality_handle.out_channels,
#             modality_handle.act, modality_handle.norm,
#             use_conv_final, num_up_blocks
#         )

#     def forward(self, x):
#         x, x_mid = self.modality_handle(x)
#         x, _ = self.decoder(x)

#         return x, x_mid

class ModalityHandle(nn.Module):
    def __init__(
        self,
        init_filters: int,
        in_channels: int,
        out_channels: int,
        act: str = 'prelu',
        norm: str = 'batchnorm',
        num_init_blocks: tuple = (1, 2, 2, 4),
    ):
        super(ModalityHandle, self).__init__()

        self.init_filters = init_filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.norm = norm
        self.num_init_blocks = num_init_blocks

        self.seg_blocks = self._make_seg_blocks(num_init_blocks)
        
    def _make_seg_blocks(self, num_init_blocks: tuple) -> nn.ModuleList:
        act = nn.ReLU(inplace=True) if self.act == 'relu' else nn.PReLU()
        down_blocks  = nn.ModuleList()
        out_channels = 0
        for i in range(len(num_init_blocks)):
            if i == 0:
                in_channels = self.in_channels
                out_channels = self.init_filters
                conv_layer = nn.Identity()
                res_block = ResBlock(
                    in_channels, out_channels,
                    self.norm, self.act,
                    num_layers=num_init_blocks[i]
                    )
            else:
                in_channels = self.init_filters * 2 ** (i - 1)
                out_channels = in_channels * 2
                conv_layer = ConvLayer(
                    in_channels, out_channels, 3, 2, 1,
                    self.norm, self.act,
                )
                res_block = ResBlock(
                    out_channels, out_channels,
                    self.norm, self.act,
                    num_layers=num_init_blocks[i]
                    )
            down_blocks.append(nn.Sequential(conv_layer, res_block))
        down_blocks.append(ConvLayer(
            out_channels, self.out_channels, 1, 1, 0,
            self.norm, 'sigmoid'
        ))
            
        return down_blocks

    def forward(self, x):
        for block in self.seg_blocks[:-1]:
            x, _ = block(x)
        x_seg = self.seg_blocks[-1](x)

        return x, x_seg

# code for the paper
# class ModalityHandle(nn.Module):
#     def __init__(
#         self,
#         init_filters: int = 8,
#         in_channels: int = 4,
#         out_channels: int = 1,
#         act: str = 'relu',
#         norm: str = 'batchnorm',
#         num_init_blocks: tuple = (1, 2, 2, 4),
#         ):
#         super(ModalityHandle, self).__init__()

#         self.init_filters = init_filters
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.act = act
#         self.norm = norm
#         self.num_layers_blocks = num_init_blocks

#         self.seg_blocks = nn.ModuleList([
#             ConvLayer(
#                 'relu', norm,
#                 in_channels, out_channels * 4
#                 ),
#             ResBlock(
#                 out_channels * 4, None,
#                 act, norm, num_layers=4
#                 ),
#             ConvLayer(
#                 # 'tanh', norm,
#                 'prelu', norm,
#                 out_channels * 4, out_channels
#                 )
#             ])
        
#         self.blocks = self._make_block_layers(len(num_init_blocks))

#     def _make_block_layers(self, num_blocks):
#         blocks = nn.ModuleList()
#         previous_in_channels = 0
#         for i in range(num_blocks):
#             in_channels = self.init_filters * 2 ** i
#             blocks.append(
#                 ResBlock(
#                     previous_in_channels + in_channels // 2 if i > 0 else self.out_channels, in_channels,
#                     act=self.act, norm=self.norm,
#                     num_layers=self.num_layers_blocks[i]
#                     )
#                 )
#             previous_in_channels = \
#                 previous_in_channels + in_channels // 2 if i > 0 \
#                     else self.out_channels

#         return blocks

#     def forward(self, x):
#         x = self.seg_blocks[0](x)
#         for block in self.seg_blocks[1:-1]:
#             x, _ = block(x)
#         x = self.seg_blocks[-1](x)

#         x_ = one_hot(
#             torch.argmax(x, dim=1, keepdim=True),
#             num_classes=self.out_channels,
#             dtype=x.dtype
#             )
#         for block in self.blocks[:-1]:
#             x_, _ = block(x_)
#         _, x_ = self.blocks[-1](x_)

#         return x_, x


class GraphAAGCN:
    r"""
    Defining the Graph for the Two-Stream Adaptive Graph Convolutional Network.
    It's composed of the normalized inward-links, outward-links and
    self-links between the nodes as originally defined in the
    `authors repo  <https://github.com/lshiwjx/2s-AGCN/blob/master/graph/tools.py>`
    resulting in the shape of (3, num_nodes, num_nodes).
    Args:
        edge_index (Tensor array): Edge indices
        num_nodes (int): Number of nodes
    Return types:
            * **A** (PyTorch Float Tensor) - Three layer normalized adjacency matrix
    """
    def __init__(self, edge_index: list, num_nodes: int):
        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.A = self.get_spatial_graph(self.num_nodes)

    def get_spatial_graph(self, num_nodes):
        self_mat = torch.eye(num_nodes)
        inward_mat = torch.squeeze(to_dense_adj(self.edge_index))
        inward_mat_norm = F.normalize(inward_mat, dim=0, p=1)
        outward_mat = inward_mat.transpose(0, 1)
        outward_mat_norm = F.normalize(outward_mat, dim=0, p=1)
        adj_mat = torch.stack((self_mat, inward_mat_norm, outward_mat_norm))
        return adj_mat


class UnitTCN(nn.Module):
    r"""
    Temporal Convolutional Block applied to nodes in the Two-Stream Adaptive Graph
    Convolutional Network as originally implemented in the
    `Github Repo <https://github.com/lshiwjx/2s-AGCN>`. For implementational details
    see https://arxiv.org/abs/1805.07694
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        kernel_size (int): Convolutional kernel size. (default: :obj:`9`)
        stride (int): Temporal Convolutional kernel stride. (default: :obj:`1`)
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1
    ):
        super(UnitTCN, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self._conv_init(self.conv)
        self._bn_init(self.bn, 1)

    def _bn_init(self, bn, scale):
        nn.init.constant_(bn.weight, scale)
        nn.init.constant_(bn.bias, 0)

    def _conv_init(self, conv):
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
        nn.init.constant_(conv.bias, 0)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class UnitGCN(nn.Module):
    r"""
    Graph Convolutional Block applied to nodes in the Two-Stream Adaptive Graph Convolutional
    Network as originally implemented in the `Github Repo <https://github.com/lshiwjx/2s-AGCN>`.
    For implementational details see https://arxiv.org/abs/1805.07694.
    Temporal attention, spatial attention and channel-wise attention will be applied.
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        A (Tensor array): Adaptive Graph.
        coff_embedding (int, optional): Coefficient Embeddings. (default: :int:`4`)
        num_subset (int, optional): Subsets for adaptive graphs, see
        :math:`\mathbf{A}, \mathbf{B}, \mathbf{C}` in https://arxiv.org/abs/1805.07694
        for details. (default: :int:`3`)
        adaptive (bool, optional): Apply Adaptive Graph Convolutions. (default: :obj:`True`)
        attention (bool, optional): Apply Attention. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.FloatTensor,
        coff_embedding: int = 4,
        num_subset: int = 3,
        adaptive: bool = True,
        attention: bool = True,
    ):
        super(UnitGCN, self).__init__()
        self.inter_c = out_channels // coff_embedding
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = num_subset
        self.A = A
        self.num_jpts = A.shape[-1]
        self.attention = attention
        self.adaptive = adaptive

        self.conv_d = nn.ModuleList()

        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if self.adaptive:
            self._init_adaptive_layers()
        else:
            self.A = Variable(self.A, requires_grad=False)

        if self.attention:
            self._init_attention_layers()

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        self._init_conv_bn()

    def _bn_init(self, bn, scale):
        nn.init.constant_(bn.weight, scale)
        nn.init.constant_(bn.bias, 0)

    def _conv_init(self, conv):
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
        nn.init.constant_(conv.bias, 0)

    def _conv_branch_init(self, conv, branches):
        weight = conv.weight
        n = weight.size(0)
        k1 = weight.size(1)
        k2 = weight.size(2)
        nn.init.normal_(weight, 0, math.sqrt(2.0 / (n * k1 * k2 * branches)))
        nn.init.constant_(conv.bias, 0)

    def _init_conv_bn(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                self._bn_init(m, 1)
        self._bn_init(self.bn, 1e-6)

        for i in range(self.num_subset):
            self._conv_branch_init(self.conv_d[i], self.num_subset)

    def _init_attention_layers(self):
        # temporal attention
        self.conv_ta = nn.Conv1d(self.out_c, 1, 9, padding=4)
        nn.init.constant_(self.conv_ta.weight, 0)
        nn.init.constant_(self.conv_ta.bias, 0)

        # s attention
        ker_jpt = self.num_jpts - 1 if not self.num_jpts % 2 else self.num_jpts
        pad = (ker_jpt - 1) // 2
        self.conv_sa = nn.Conv1d(self.out_c, 1, ker_jpt, padding=pad)
        nn.init.xavier_normal_(self.conv_sa.weight)
        nn.init.constant_(self.conv_sa.bias, 0)

        # channel attention
        rr = 2
        self.fc1c = nn.Linear(self.out_c, self.out_c // rr)
        self.fc2c = nn.Linear(self.out_c // rr, self.out_c)
        nn.init.kaiming_normal_(self.fc1c.weight)
        nn.init.constant_(self.fc1c.bias, 0)
        nn.init.constant_(self.fc2c.weight, 0)
        nn.init.constant_(self.fc2c.bias, 0)

    def _init_adaptive_layers(self):
        self.PA = nn.Parameter(self.A)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(self.in_c, self.inter_c, 1))
            self.conv_b.append(nn.Conv2d(self.in_c, self.inter_c, 1))

    def _attentive_forward(self, y):
        # spatial attention
        se = y.mean(-2)  # N C V
        se1 = self.sigmoid(self.conv_sa(se))
        y = y * se1.unsqueeze(-2) + y

        # temporal attention
        se = y.mean(-1)
        se1 = self.sigmoid(self.conv_ta(se))
        y = y * se1.unsqueeze(-1) + y

        # channel attention
        se = y.mean(-1).mean(-1)
        se1 = self.relu(self.fc1c(se))
        se2 = self.sigmoid(self.fc2c(se1))
        y = y * se2.unsqueeze(-1).unsqueeze(-1) + y

        return y

    def _adaptive_forward(self, x, y):
        N, C, T, V = x.size()

        A = self.PA
        for i in range(self.num_subset):
            A1 = (
                self.conv_a[i](x)
                .permute(0, 3, 1, 2)
                .contiguous()
                .view(N, V, self.inter_c * T)
            )
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A[i] + A1 * self.alpha
            A2 = rearrange(x, 'n c t v -> n (c t) v')
            # A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        return y

    def _non_adaptive_forward(self, x, y):
        N, C, T, V = x.size()
        for i in range(self.num_subset):
            A1 = self.A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z
        return y

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            y = self._adaptive_forward(x, y)
        else:
            y = self._non_adaptive_forward(x, y)
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        if self.attention:
            y = self._attentive_forward(y)
        return y


class RStGCN(nn.Module):
    """Recurrent Spatial-temporal Graph Convolution Network.

    For details of Adaptive Graph Convolution Network (Graph AAGCN), see this paper: `"Two-Stream Adaptive Graph Convolutional Networks for
    Skeleton-Based Action Recognition." <https://arxiv.org/abs/1805.07694>`_.
    This implementation is based on the authors Github Repo https://github.com/lshiwjx/2s-AGCN.
    It's used by the author for classifying actions from sequences of 3D body joint coordinates.

    :params in_channels (int): Number of input features.
    :params out_channels (int): Number of output features.
    :params edge_index (PyTorch LongTensor): Graph edge indices.
    :params num_nodes (int): Number of nodes in the network.
    :params stride (int, optional): Time strides during temporal convolution. (default: :obj:`1`)
    :params residual (bool, optional): Applying residual connection. (default: :obj:`True`)
    :params adaptive (bool, optional): Adaptive node connection weights. (default: :obj:`True`)
    :params attention (bool, optional): Applying spatial-temporal-channel-attention. (default: :obj:`True`)
    """

    def __init__(
        self,
        hidden_features: int,
        num_blocks: int,
        sdm_out_channel: int,
        template_mesh_dict: dict,
        stride: int = 1,
        residual: bool = True,
        adaptive: bool = True,
        attention: bool = True,
        task_code: str = "whole_heart_meshing"
        ):
        super(RStGCN, self).__init__()
        self.task_code = task_code
        template_mesh, faces_labels = template_mesh_dict["meshes"], template_mesh_dict["faces_labels"]
        self.graph = GraphAAGCN(
            template_mesh.edges_packed().T, template_mesh._V
        )
        self.A = self.graph.A

        self.agcn_stack = nn.ModuleList()
        self.cgcn_stack = nn.ModuleList()
        in_channels = 3
        for i in range(num_blocks):
            out_channels = hidden_features * 2 ** i
            gcn1 = UnitGCN(
                in_channels, out_channels, self.A, adaptive=adaptive, attention=attention
            )
            gcn2 = self._graph_conv_layers(out_channels, out_channels)
            self.agcn_stack.append(gcn1)
            self.cgcn_stack.append(gcn2)
            in_channels = out_channels
        self.graph_fc = nn.Sequential(
            nn.Linear(out_channels, hidden_features),
            nn.PReLU(),
            nn.Linear(hidden_features, 3),
            nn.Tanh()
        )
        
        self.sdm_fc = nn.Sequential(
            nn.Linear(sdm_out_channel, hidden_features),
            nn.PReLU(),
            nn.Linear(hidden_features, 3),  # number of classes -> number of vertex dimension
            nn.Tanh()
        )
        self.subdvider = SubdivideMeshes(template_mesh, faces_labels)

        self.tcn1 = UnitTCN(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.attention = attention

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = UnitTCN(
                in_channels, out_channels, kernel_size=1, stride=stride
            )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _graph_conv_layers(self, in_channels, out_channels):
        return Sequential(
            'x, edge_index', [
                (ChebConv(in_channels, out_channels, 3), 'x, edge_index -> x'),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                (ChebConv(out_channels, out_channels, 3), 'x, edge_index -> x'),
                nn.ReLU(inplace=True),
                ]
            )

    def _sdm_sample(self, v_pos, x_batch):
        # x_batch = x_batch[:, 1:].float() # remove background
        # for b in range(x_batch.shape[0]):
        #     x_batch[b] = FastGeodis.signed_generalised_geodesic3d(
        #         x_batch[b].unsqueeze(0), x_batch[b].unsqueeze(0),
        #         spacing=[1, 1, 1], v=1, lamb=1.0, iter=4
        #         )
        x_batch = x_batch[:, 1:] # remove background

        v_grid = rearrange(v_pos, "b v p -> b v () () p")
        x_batch = F.grid_sample(
            x_batch, v_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
            )
        x_batch = rearrange(x_batch, 'b c v () () -> b v c')
        v_pos = v_pos + self.sdm_fc(x_batch)

        return v_pos

    def forward(self, mesh_batch: Meshes, seg_batch: Tensor, subdivision: bool = False):
        # Signed Distance Map
        v_batch = mesh_batch.verts_padded()
        v_batch_ = self._sdm_sample(v_batch, seg_batch)

        # R-StGCN
        for agcn, cgcn in zip(self.agcn_stack, self.cgcn_stack):
            if self.task_code.lower().strip(' ') == "whole_heart_meshing":
                v_batch_ = rearrange(v_batch_, "b p f -> b f () p")
                v_batch_ = agcn(v_batch_)
                v_batch_ = rearrange(v_batch_, "b f () p -> (b p) f")
                v_batch_ = cgcn(v_batch_, mesh_batch.edges_packed().T)
                v_batch_ = rearrange(v_batch_, "(b p) f -> b p f", b=seg_batch.shape[0])
            else:
                v_batch_ = rearrange(v_batch_, 'l p f -> () f l p')
                v_batch_ = agcn(v_batch_)
                v_batch_ = rearrange(v_batch_, '() f l p -> (l p) f')
                v_batch_ = cgcn(v_batch_, mesh_batch.edges_packed().T)
                v_batch_ = rearrange(v_batch_, "(l p) f -> l p f", l=seg_batch.shape[0])
 
        # Deform meshes
        v_batch_ = self.graph_fc(v_batch_)
        v_batch_ = rearrange(v_batch_, "b p f -> (b p) f")

        if subdivision:
            # Surface Subdivision
            # WARNING: will change the number of vertices, edges, and faces
            mesh_batch = mesh_batch.offset_verts(v_batch_)
            mesh_batch, faces_labels, _ = self.subdvider(mesh_batch)

            return mesh_batch, faces_labels
        else:
            mesh_batch = mesh_batch.offset_verts(v_batch_)

            return mesh_batch


# The following is test only code

class Args:
    def __init__(self):
        self.keys = ("ct_image", "ct_label")
        self.dataset = "scotheart"
        self.crop_window_size = [128, 128, 128]
        self.section = "test"
        self.point_limit = 5_806
        self.one_or_multi = "solo"
        self.template_dir = "/home/yd21/Documents/ModusGraph/template/cap/"

if __name__ == "__main__":
    from data.transform import pre_transform
    from pytorch3d.structures import Pointclouds
    args = Args()
    test_transform = pre_transform(
        keys=args.keys, dataset=args.dataset, crop_window_size=args.crop_window_size,
        section=args.section, point_limit=args.point_limit, 
        one_or_multi=args.one_or_multi, template_dir=args.template_dir
    )
    voxel_module = VoxelProcessingModule(
        ModalityHandle(
            init_filters=8,
            in_channels=1,
            out_channels=2, # myocardium and background
            num_init_blocks=(1, 2, 2, 4),
            ),
        num_up_blocks=(1, 1, 1),
        ).to("cuda")
        
    # print structure of modality handle
    print(voxel_module.modality_handle)
    # print structure of ResNet decoder
    print(voxel_module.resnet_decoder)
    
    source_root_dir = "/mnt/data/Experiment/nnUNet/nnUNet_raw/nnUNet_raw_data/Task504_SCOTHEART/"
    patient_list = [i.strip(".nii.gz") for i in os.listdir(f"{source_root_dir}/labelsTr")][:1]
    for patient_id in patient_list:
        image_dir = os.path.join(
            source_root_dir, "imagesTr", f"{patient_id}_0000.nii.gz"
        )
        label_dir = os.path.join(
            source_root_dir, "labelsTr", f"{patient_id}.nii.gz"
        )
        output_data = test_transform({
            "ct_image": image_dir,
            "ct_label": label_dir
        })
        image = output_data["ct_image"].as_tensor().unsqueeze(1).to("cuda")
        template_mesh = output_data["ct_template_meshes"]["meshes"].to("cuda")

        output_data["ct_template_meshes"]["meshes"] = output_data["ct_template_meshes"]["meshes"][0]
        output_data["ct_template_meshes"]["faces_labels"] = output_data["ct_template_meshes"]["faces_labels"][:, :1]
        graph_module = RStGCN(
            hidden_features=32, num_blocks=3,
            sdm_out_channel=1,  # the number of classes (out_channel) excludeing background
            template_mesh_dict=output_data["ct_template_meshes"],
            attention=False,
            task_code="dynamic_meshing"
        ).to("cuda")
        # print structure of graph module
        print(graph_module)

        pred_seg, pred_seg_downsample = voxel_module(image)
        print(f"Voxel Module Output Shape:\t{pred_seg.shape} | {pred_seg_downsample.shape}")
        template_mesh = template_mesh.to("cuda")
        pred_meshes_batch = graph_module(template_mesh, pred_seg, subdivision=False)
        print(f"Graph Module Output Shape:\t{pred_meshes_batch.verts_padded().shape} | {pred_meshes_batch.faces_padded().shape}")
        # pred_meshes_batch, new_faces_labels = graph_module(template_mesh, pred_seg, subdivision=False)
        # print(f"Graph Module Output Shape:\t{pred_meshes_batch.verts_padded().shape} | {new_faces_labels.shape}")
