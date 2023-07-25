import os
import time
from collections import OrderedDict
import numpy as np
from itertools import chain
import torch
import torch.nn.functional as F
from trimesh.voxel.ops import matrix_to_marching_cubes
from pytorch3d.io import save_obj
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_normal_consistency,
)
from monai.data import DataLoader
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, 
    EnsureType, 
    AsDiscrete, 
    KeepLargestConnectedComponent
)
from monai.utils import set_determinism
from sklearn.utils import shuffle
import wandb

from data.dataset import *
from data.transform import *
from model.networks import *
from utils.loss import *
from utils.tools import *
from utils.rasterize.rasterize import Rasterize


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TrainPipeline:
    def __init__(
            self,
            super_params,
            data_bundle,
            seed, num_workers,
            is_training=True,
            **kwargs
        ):
        """
        :param super_params: parameters for setting up dataset, network structure, training, etc.
        :param data_bundle: bundle of image_paths, label_paths, keys, dataset for loading data
        :param seed: random seed to shuffle data during augmentation
        :param num_workers: tourch.utils.data.DataLoader num_workers
        :param is_training: switcher for training (True, default) or testing (False)
        :param kwargs: other parameters required during testing
        """
        self.super_params = super_params
        self.data_bundle = data_bundle
        self.seed = seed
        self.num_workers = num_workers
        self.is_training = is_training
        set_determinism(seed=self.seed)

        self.log_dir = os.path.join(super_params.log_dir, "whole_heart_meshing", super_params.run_id)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if is_training:
            self.train_loss = OrderedDict(
                {k: np.asarray([]) for k in ['total', 'seg', 'chamfer', 'edge', 'face', 'laplacian']}
            )
            self.eval_seg_score = OrderedDict(
                {k: np.asarray([]) for k in ['lv', 'lv-myo', 'rv', 'la', 'ra', 'av']}
            )
            self.eval_msh_score = self.eval_seg_score.copy()
            self.best_eval_score = 0
        else:
            self.out_dir = os.path.join(super_params.out_dir, "whole_heart_meshing", super_params.run_id)
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)

        train_trans, val_trans = self._prepare_transform()
        self.post_transform = Compose(
            [
                AsDiscrete(argmax=True),
                KeepLargestConnectedComponent(
                    applied_labels=list(range(1, self.super_params.num_classes)), # foreground classes
                    independent=True
                ),
                EnsureType()
            ]
        )

        train_ds, val_ds = self._prepare_dataset(train_trans, val_trans)
        if is_training:
            self.train_loader = DataLoader(
                train_ds, batch_size=self.super_params.batch_size,
                shuffle=True, num_workers=self.num_workers,
                collate_fn=collate_batched_meshes
                )
        self.val_loader = DataLoader(
            val_ds, batch_size=1,
            shuffle=False, num_workers=self.num_workers,
            collate_fn=collate_batched_meshes
            )

        self._prepare_models()

    def _prepare_transform(self, ):
        if not self.is_training:
            val_transform = pre_transform(
                self.data_bundle['keys'], self.data_bundle['dataset'], 
                self.super_params.crop_window_size, 'test',
                point_limit=self.super_params.point_limit, 
                one_or_multi='multi',    # TODO: pass this parameter from outside of class for choosing whole heart or dynamic meshing
                template_dir=self.super_params.template_dir
                )
            return None, val_transform
        else:
            train_transform = pre_transform(
                self.data_bundle['keys'], self.data_bundle['dataset'], 
                self.super_params.crop_window_size, 'training',
                point_limit=self.super_params.point_limit, 
                one_or_multi='multi',    # TODO: pass this parameter from outside of class for choosing whole heart or dynamic meshing
                template_dir=self.super_params.template_dir
                )
            val_transform = pre_transform(
                self.data_bundle['keys'], self.data_bundle['dataset'], 
                self.super_params.crop_window_size, 'validation',
                point_limit=self.super_params.point_limit, 
                one_or_multi='multi',    # TODO: pass this parameter from outside of class for choosing whole heart or dynamic meshing
                template_dir=self.super_params.template_dir
                )
            return train_transform, val_transform

    def _prepare_dataset(self, train_transform, val_transform):
        if not self.is_training:
            val_ds = MultimodalDataset(
                self.data_bundle['image_paths'], self.data_bundle['label_paths'], self.data_bundle['keys'],
                section='test',
                transform=val_transform,
                seed=self.seed,
                cache_rate=self.super_params.cache_rate,
                num_workers=self.num_workers,
                )
            return None, val_ds
        else:
            image_paths, label_paths = shuffle(
                self.data_bundle['image_paths'], self.data_bundle['label_paths'], random_state=self.seed
                )
            train_ds = MultimodalDataset(
                image_paths, label_paths, self.data_bundle['keys'],
                section='training',
                transform=train_transform,
                seed=self.seed,
                cache_rate=self.super_params.cache_rate,
                num_workers=self.num_workers,
                )
            val_ds = MultimodalDataset(
                image_paths, label_paths, self.data_bundle['keys'],
                section='validation',
                transform=val_transform,
                seed=self.seed,
                cache_rate=self.super_params.cache_rate,
                num_workers=self.num_workers,
                )
            return train_ds, val_ds

    def _prepare_models(self, ):
        self.voxel_module = VoxelProcessingModule(
            ModalityHandle(
                init_filters=self.super_params.init_filters,
                in_channels=self.super_params.in_channels,
                out_channels=self.super_params.num_classes,
                num_init_blocks=self.super_params.num_init_blocks,
                ),
            num_up_blocks=self.super_params.num_up_blocks,
            ).to(DEVICE)
        if self.super_params.pre_trained_module_dir is not None:
            self.voxel_module.load_state_dict(torch.load(self.super_params.pre_trained_module_dir))
        else:
            print("WARN: loading no pre-trained voxel module")
            self.pretrain_optimizer = torch.optim.SGD(
                self.voxel_module.parameters(), 
                lr=self.super_params.lr * 10
                )
            self.pretrain_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.pretrain_optimizer)
            self.pretrain_scaler = torch.cuda.amp.GradScaler()

        self.graph_module = RStGCN(
            hidden_features=32, num_blocks=3,
            sdm_out_channel=self.super_params.num_classes - 1,  # exclude background
            template_mesh=next(iter(self.val_loader))["ct_template_meshes"],
            attention=False
            ).to(DEVICE)

        self.rasterizer = Rasterize(self.super_params.crop_window_size) # tool for rasterizing mesh

        self.seg_loss = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            squared_pred=False,
            ).to(DEVICE)
        self.downsample_seg_loss = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            squared_pred=False,
            ).to(DEVICE)

        self.optimizer = torch.optim.Adam(
            chain(self.voxel_module.parameters(), self.graph_module.parameters()), 
            lr=self.super_params.lr
            )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.scaler = torch.cuda.amp.GradScaler()

        torch.backends.cudnn.enabled = torch.backends.cudnn.is_available()
        torch.backends.cudnn.benchmark = torch.backends.cudnn.is_available()

    def pre_train_iter(self, epoch):
        self.voxel_module.train()

        step = 0
        train_loss_epoch = torch.tensor(0)
        for step, train_data in enumerate(self.train_loader):
            images_batch, labels_batch, labels_downsample_batch = (
                train_data["ct_image"].as_tensor().to(DEVICE),
                train_data["ct_label"].as_tensor().to(DEVICE),
                train_data["ct_label_downsample"].as_tensor().to(DEVICE)
                )

            self.pretrain_optimizer.zero_grad()
            with torch.autocast(device_type=DEVICE):
                preds_seg, preds_downsample_seg = self.voxel_module(images_batch)
                loss_seg = self.seg_loss(preds_seg, labels_batch) + \
                    self.downsample_seg_loss(preds_downsample_seg, labels_downsample_batch)

            self.pretrain_scaler.scale(loss_seg).backward()
            self.pretrain_scaler.step(self.pretrain_optimizer)
            self.pretrain_scaler.update()

            train_loss_epoch = loss_seg.detach().cpu() + train_loss_epoch

        train_loss_epoch = train_loss_epoch / (step + 1)
        wandb.log({"pre_train_loss": train_loss_epoch}, step=epoch + 1)
        self.pretrain_lr_scheduler.step(train_loss_epoch)

    def train_iter(self, epoch):
        self.voxel_module.train()
        self.graph_module.train()

        step = 0
        self.train_loss_epoch = {
            tag: torch.tensor(0) for tag in ['total', 'seg', 'chamfer', 'edge', 'face', 'laplacian']
            }
        for step, train_data in enumerate(self.train_loader):
            images_batch, labels_batch, labels_downsample_batch = (
                train_data["ct_image"].as_tensor().to(DEVICE),
                train_data["ct_label"].as_tensor().to(DEVICE),
                train_data["ct_label_downsample"].as_tensor().to(DEVICE)
                )
            template_meshes_batch, target_point_clouds_batch = (
                train_data["ct_template_meshes"]["meshes"].to(DEVICE), 
                train_data["ct_label_point_clouds"]["point_clouds"].to(DEVICE)
                )

            self.optimizer.zero_grad()
            with torch.autocast(device_type=DEVICE):
                preds_seg, preds_downsample_seg = self.voxel_module(images_batch)
                loss_seg = self.seg_loss(preds_seg, labels_batch) + \
                    self.downsample_seg_loss(preds_downsample_seg, labels_downsample_batch)
                
                pred_meshes_batch = self.graph_module(
                    template_meshes_batch, preds_seg
                    )
                loss_chamfer, *_ = chamfer_distance(
                    pred_meshes_batch.verts_padded(), target_point_clouds_batch.points_padded(),
                    )
                loss_edges = mesh_edge_loss(pred_meshes_batch)
                loss_faces = mesh_normal_consistency(pred_meshes_batch)
                loss_laplacian = mesh_laplacian_smoothing(pred_meshes_batch)

                # segmentation, chamfer distance, edge, face, and laplacian loss
                total_loss = self.super_params.lambda_[0] * loss_seg + \
                    self.super_params.lambda_[1] * loss_chamfer + \
                        self.super_params.lambda_[2] * loss_edges + \
                            self.super_params.lambda_[3] * loss_faces + \
                                self.super_params.lambda_[4] * loss_laplacian
                
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k, v in zip(
                ['total', 'seg', 'chamfer', 'edge', 'face', 'laplacian'],
                [total_loss, loss_seg, loss_chamfer, loss_edges, loss_faces, loss_laplacian]
                ):
                self.train_loss_epoch[k] = v.detach().cpu() + self.train_loss_epoch.get(k)

        for k, v in self.train_loss_epoch.items():
            self.train_loss_epoch[k] = v / (step + 1)
            self.train_loss[k] = np.append(self.train_loss[k], self.train_loss_epoch[k])
        wandb.log(
            {"train_loss": self.train_loss_epoch["total"]}, 
            step=epoch + 1
        )
        self.lr_scheduler.step(self.train_loss_epoch["total"])

    def valid(self, epoch):
        self.voxel_module.eval()
        self.graph_module.eval()

        self.seg_metric_batch_decoder = DiceMetric(
            include_background=False,
            reduction="mean_batch", 
            num_classes=self.super_params.num_classes
            )
        self.msh_metric_batch_decoder = DiceMetric(
            include_background=False,
            reduction="mean_batch", 
            num_classes=self.super_params.num_classes
            )

        cached_data = dict()
        choice_case = np.random.choice(len(self.val_loader), 1)[0]
        with torch.no_grad():
            for step, val_data in enumerate(self.val_loader):
                images_batch, labels_batch = (
                    val_data[f"ct_image"].as_tensor().to(DEVICE),
                    val_data[f"ct_label"].as_tensor().to(DEVICE),
                    )
                template_meshes, faces_labels = (
                    val_data["ct_template_meshes"]["meshes"].to(DEVICE),
                    val_data["ct_template_meshes"]["faces_labels"].to(DEVICE)
                    )

                preds_seg, _ = self.voxel_module(images_batch)
                pred_meshes = self.graph_module(
                    template_meshes, preds_seg
                    )
                preds_seg = self.post_transform(preds_seg.squeeze()).unsqueeze(0)
                self.seg_metric_batch_decoder(preds_seg, labels_batch)

                pred_meshes = pred_meshes.submeshes([list(
                    faces_index.nonzero().flatten() - 1 for faces_index in faces_labels.squeeze()
                    )])
                pred_voxels_batch = []
                for pred_mesh in pred_meshes:
                    pred_voxels = self.rasterizer(
                        pred_mesh.verts_padded(), pred_mesh.faces_padded()
                    )
                    pred_voxels_batch.append(pred_voxels)
                pred_voxels_batch = torch.cat([torch.zeros_like(pred_voxels_batch[0]), *pred_voxels_batch], dim=1)
                pred_voxels_batch = self.post_transform(pred_voxels_batch.squeeze()).unsqueeze(0)
                self.msh_metric_batch_decoder(pred_voxels_batch, labels_batch)

                if step == choice_case:
                    cached_data = {
                        "template_meshes": template_meshes.cpu(),
                        "images_batch": images_batch.cpu(),
                        "labels_batch": labels_batch.cpu(),
                        "labels_point_clouds": val_data["ct_label_point_clouds"]["point_clouds"],
                        "preds_seg": preds_seg.cpu(),
                        "pred_meshes": pred_meshes.cpu(),
                    }

        # log dice score
        for v, k in enumerate(['lv', 'lv-myo', 'rv', 'la', 'ra', 'av']):
            self.eval_seg_score[k] = np.append(self.eval_seg_score[k], self.seg_metric_batch_decoder.aggregate()[v].cpu())
            self.eval_msh_score[k] = np.append(self.eval_msh_score[k], self.msh_metric_batch_decoder.aggregate()[v].cpu())
        draw_train_loss(self.train_loss, self.super_params)
        draw_eval_score(self.eval_seg_score, self.super_params, task="seg")
        draw_eval_score(self.eval_msh_score, self.super_params, task="msh")
        wandb.log({
            "train_categorised_loss": wandb.Table(
            columns=[f"train_loss \u2193", f"eval_seg_score \u2191", f"eval_msh_score \u2191"],
            data=[[
                wandb.Image(f"{self.super_params.log_dir}/whole_heart_meshing/{self.super_params.run_id}/train_loss.png"),
                wandb.Image(f"{self.super_params.log_dir}/whole_heart_meshing/{self.super_params.run_id}/eval_seg_loss.png"),
                wandb.Image(f"{self.super_params.log_dir}/whole_heart_meshing/{self.super_params.run_id}/eval_msh_loss.png"),
                ]]
            )},
            step=epoch + 1
            )
        eval_score_epoch = self.msh_metric_batch_decoder.aggregate().mean()
        wandb.log({"eval_score": eval_score_epoch}, step=epoch + 1)

        # save model
        ckpt_weight_path = os.path.join(self.log_dir, "ckpt_weights")
        if not os.path.exists(ckpt_weight_path):
            os.makedirs(ckpt_weight_path)
        torch.save(
            self.voxel_module.state_dict(),
            os.path.join(self.log_dir, 'ckpt_weights', f'{epoch + 1}_VoxelProcess.pth')
        )
        torch.save(
            self.graph_module.state_dict(),
            os.path.join(self.log_dir, 'ckpt_weights', f'{epoch + 1}_RStGCN.pth')
        )

        # save visualization when the eval score is the best
        # WARNING: this part is very time-consuming, please comment it if you don't need it
        if eval_score_epoch > self.best_eval_score:
            self.best_eval_score = eval_score_epoch
            wandb.log(
                {
                    # "template_mesh": wandb.Plotly(draw_plotly(
                    #     images=cached_data["images_batch"], 
                    #     template_meshes=cached_data["template_meshes"]
                    #     )), 
                    # "image vs seg_label": wandb.Plotly(draw_plotly(
                    #     images=cached_data["images_batch"], 
                    #     labels=cached_data["labels_batch"]
                    #     )),
                    # "seg_label vs seg_pred": wandb.Plotly(draw_plotly(
                    #     labels=cached_data["labels_batch"], 
                    #     pred_seg=cached_data["preds_seg"]
                    #     )),
                    "point_cloud_label vs mesh_pred": wandb.Plotly(draw_plotly(
                        point_clouds=cached_data["labels_point_clouds"], 
                        pred_meshes=cached_data["pred_meshes"]
                        ))
                },
                step=epoch + 1
            )
        
    def test(self, ):
        # load_model
        self.voxel_module.load_state_dict(
            torch.load(os.path.join(f"{self.log_dir}/ckpt_weights", f"{self.super_params.best_epoch}_VoxelProcess.pth"))
        )
        self.graph_module.load_state_dict(
            torch.load(os.path.join(f"{self.log_dir}/ckpt_weights", f"{self.super_params.best_epoch}_RStGCN.pth"))
        )
        self.voxel_module.eval()
        self.graph_module.eval()

        # testing
        with torch.no_grad():
            for val_data in self.val_loader:
                images_batch, labels_batch = (
                    val_data[f"ct_image"].as_tensor().to(DEVICE),
                    val_data[f"ct_label"].as_tensor().to(DEVICE),
                    )
                template_meshes, faces_labels = (
                    val_data["ct_template_meshes"]["meshes"].to(DEVICE),
                    val_data["ct_template_meshes"]["faces_labels"].to(DEVICE)
                    )

                preds_seg, _ = self.voxel_module(images_batch)
                pred_meshes = self.graph_module(
                    template_meshes, preds_seg
                    )
                preds_seg = self.post_transform(preds_seg.squeeze()).unsqueeze(0)
                preds_seg = F.one_hot(
                    preds_seg.squeeze(1).long(), num_classes=self.super_params.num_classes
                ).permute(0, 4, 1, 2, 3).squeeze()[1:]  # exclude background
                pred_meshes = pred_meshes.submeshes([list(
                    faces_index.nonzero().flatten() - 1 for faces_index in faces_labels.squeeze()
                    )])
                label_names = ["lv", "lv_myo", "rv", "rv_myo", "la", "ra"]
                for i in range(len(label_names)):
                    seg2mesh = matrix_to_marching_cubes(preds_seg[i].cpu())
                    seg2mesh.export(f"{self.out_dir}/{label_names[i]}_pred_VoxelProcess.obj")
                    save_obj(
                        f"{self.out_dir}/{label_names[i]}_pred_RStGCN.obj", 
                        pred_meshes[i].verts_packed(), pred_meshes[i].faces_packed()
                    )

