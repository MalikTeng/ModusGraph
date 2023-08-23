import os
from collections import OrderedDict
import numpy as np
from itertools import chain
import torch
import torch.nn.functional as F
from trimesh.voxel.ops import matrix_to_marching_cubes
from pytorch3d.io import save_obj
from pytorch3d.structures import Pointclouds
from pytorch3d.ops.marching_cubes import marching_cubes
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

        self.log_dir = os.path.join(super_params.log_dir, "dynamic_meshing", super_params.run_id)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if is_training:
            self.train_loss = OrderedDict(
                {k: np.asarray([]) for k in ["total", "seg", "chamfer", "edge", "face", "laplacian"]}
            )
            self.eval_seg_score = OrderedDict(
                {k: np.asarray([]) for k in ["myo"]}
            )
            self.eval_msh_score = self.eval_seg_score.copy()
            self.best_eval_score = 0
        else:
            self.out_dir = os.path.join(super_params.out_dir, "dynamic_meshing", super_params.run_id)
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)

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

        if is_training:
            ct_train_ds, _ = self._prepare_dataset(
                *self._prepare_transform(("ct_image", "ct_label"), "scotheart"), 
                data_bundle["ct_image_paths"], data_bundle["ct_label_paths"], 
                ("ct_image", "ct_label")
            )
        mr_train_ds, mr_val_ds = self._prepare_dataset(
            *self._prepare_transform(("mr_image", "mr_label"), "cap"), 
            data_bundle["mr_image_paths"], data_bundle["mr_label_paths"], 
            ("mr_image", "mr_label")
        )
        if is_training:
            self.ct_train_loader = DataLoader(
                ct_train_ds, batch_size=1,
                shuffle=True, num_workers=self.num_workers,
                collate_fn=collate_batched_meshes
                )
            self.mr_train_loader = DataLoader(
                mr_train_ds, batch_size=1,
                shuffle=True, num_workers=self.num_workers,
                collate_fn=collate_batched_meshes
                )
        self.val_loader = DataLoader(
            mr_val_ds, batch_size=1,
            shuffle=False, num_workers=self.num_workers,
            collate_fn=collate_batched_meshes
            )

        self._prepare_models()

    def _prepare_transform(self, keys, dataset):
        if not self.is_training:
            val_transform = pre_transform(
                keys, dataset, 
                self.super_params.crop_window_size, "test",
                point_limit=self.super_params.point_limit, 
                one_or_multi="solo",
                template_dir=self.super_params.template_dir
                )
            return None, val_transform
        else:
            train_transform = pre_transform(
                keys, dataset, 
                self.super_params.crop_window_size, "training",
                point_limit=self.super_params.point_limit, 
                one_or_multi="solo",
                template_dir=self.super_params.template_dir
                )
            val_transform = pre_transform(
                keys, dataset, 
                self.super_params.crop_window_size, "validation",
                point_limit=self.super_params.point_limit, 
                one_or_multi="solo",
                template_dir=self.super_params.template_dir
                )
            return train_transform, val_transform

    def _prepare_dataset(self, train_transform, val_transform, image_paths, label_paths, keys):
        if not self.is_training:
            val_ds = MultimodalDataset(
                image_paths, label_paths, keys,
                section="test",
                transform=val_transform,
                seed=self.seed,
                cache_rate=self.super_params.cache_rate,
                num_workers=self.num_workers,
                )
            return None, val_ds
        else:
            image_paths, label_paths = shuffle(
                image_paths, label_paths, random_state=self.seed
                )
            train_ds = MultimodalDataset(
                image_paths, label_paths, keys,
                section="training",
                transform=train_transform,
                seed=self.seed,
                cache_rate=self.super_params.cache_rate,
                num_workers=self.num_workers,
                )
            val_ds = MultimodalDataset(
                image_paths, label_paths, keys,
                section="validation",
                transform=val_transform,
                seed=self.seed,
                cache_rate=self.super_params.cache_rate,
                num_workers=self.num_workers,
                )
            return train_ds, val_ds

    def _prepare_models(self, ):
        self.mr_handle = ModalityHandle(
            init_filters=self.super_params.init_filters,
            in_channels=self.super_params.in_channels,
            out_channels=self.super_params.num_classes,
            num_init_blocks=self.super_params.num_init_blocks,
        ).to(DEVICE)
        self.voxel_module = VoxelProcessingModule(
            ModalityHandle(
                init_filters=self.super_params.init_filters,
                in_channels=self.super_params.in_channels,
                out_channels=self.super_params.num_classes,
                num_init_blocks=self.super_params.num_init_blocks,
            ),
            num_up_blocks=self.super_params.num_up_blocks,
            ).to(DEVICE)
        # take the mesh at the first frame as template
        template_mesh = next(iter(self.val_loader))["mr_template_meshes"].copy()
        template_mesh["meshes"] = template_mesh.get("meshes")[0]
        template_mesh["faces_labels"] = template_mesh["faces_labels"][:, :1]
        self.graph_module = RStGCN(
            hidden_features=32, num_blocks=3,
            sdm_out_channel=self.super_params.num_classes - 1,  # exclude background
            template_mesh_dict=template_mesh,
            attention=False,
            task_code="whole_heart_meshing"
            ).to(DEVICE)
        if self.super_params.pre_trained_ct_module_dir is not None and self.super_params.pre_trained_mr_module_dir is not None:
            self.voxel_module.load_state_dict(torch.load(self.super_params.pre_trained_ct_module_dir))
            self.mr_handle.load_state_dict(torch.load(self.super_params.pre_trained_mr_module_dir))
        else:
            print("WARN: loading no pre-trained voxel module")

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
        self.mesh_loss = TotalLosses(
            lambda_chamfer=self.super_params.lambda_[1],
            lambda_edge=self.super_params.lambda_[2],
            lambda_face=self.super_params.lambda_[3],
            lambda_laplacian=self.super_params.lambda_[4],
            ).to(DEVICE)

        self.optimizer = torch.optim.Adam(
            chain(self.voxel_module.parameters(), self.graph_module.parameters()), 
            lr=self.super_params.lr
            )
        self.scaler = torch.cuda.amp.GradScaler()
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

        torch.backends.cudnn.enabled = torch.backends.cudnn.is_available()
        torch.backends.cudnn.benchmark = torch.backends.cudnn.is_available()

    def seudo_extractor(self, preds_seg, target_point_clouds):
        try:
            verts, *_ = marching_cubes(
                preds_seg.squeeze(1).permute(0, 3, 2, 1), 
                isolevel=0.5,
                return_local_coords=True,
            )
            seudo_point_clouds = Pointclouds(points=verts.float())
            seudo_point_clouds = seudo_point_clouds.subsample(2 * self.super_params.point_limit)
        except AttributeError:
            seudo_point_clouds = target_point_clouds

        return seudo_point_clouds

    def train_iter(self, epoch, phase):
        if phase == "pretrain":
            modality = "ct"
            data_loader = self.ct_train_loader
        elif phase == "train":
            modality = "mr"
            data_loader = self.mr_train_loader
            if epoch == self.super_params.delay_epochs:
                for name, param in self.voxel_module.modality_handle.named_parameters():
                    param.data = self.mr_handle.state_dict()[name].data
                self.graph_module.task_code = "dynamic_meshing"
        
        self.voxel_module.train()
        self.graph_module.train()

        step = 0
        self.train_loss_epoch = {
            tag: torch.tensor(0) for tag in ["total", "seg", "chamfer", "edge", "face", "laplacian"]
            }
        for step, train_data in enumerate(data_loader):
            images_batch, labels_batch, labels_downsample_batch = (
                train_data[f"{modality}_image"].as_tensor().to(DEVICE),
                train_data[f"{modality}_label"].as_tensor(),
                train_data[f"{modality}_label_downsample"].as_tensor()
                )
            # convert multi-classes labels to binary classes
            labels_batch = torch.logical_or(labels_batch == 2, labels_batch == 4).to(DEVICE)
            labels_downsample_batch = torch.logical_or(
                labels_downsample_batch == 2, labels_downsample_batch == 4
            ).to(DEVICE)
            template_meshes_batch, target_point_clouds = (
                train_data[f"{modality}_template_meshes"]["meshes"].to(DEVICE),
                train_data[f"{modality}_label_point_clouds"]["point_clouds"].to(DEVICE)
                )

            self.optimizer.zero_grad()
            with torch.autocast(device_type=DEVICE):
                preds_seg, preds_downsample_seg = self.voxel_module(images_batch)
                # calculate the loss only at the first (ED) and last (ES) frame
                if phase == "pretrain":
                    loss_seg = self.seg_loss(preds_seg, labels_batch) + \
                        self.downsample_seg_loss(preds_downsample_seg, labels_downsample_batch)
                else:
                    preds_downsample_seg = preds_downsample_seg[[0, -1]]
                    loss_seg = self.downsample_seg_loss(preds_downsample_seg, labels_downsample_batch)
                
                pred_meshes_batch = self.graph_module(
                    template_meshes_batch, preds_seg
                    )
                
                preds_seg_ = torch.stack([self.post_transform(p) for p in preds_seg])  # get myocardium prediction
                seudo_point_clouds = self.seudo_extractor(preds_seg_, target_point_clouds)    # get seudo point clouds
                total_loss, log_losses = self.mesh_loss(pred_meshes_batch, seudo_point_clouds)
                total_loss += self.super_params.lambda_[0] * loss_seg
                log_losses["seg"] = self.super_params.lambda_[0] * loss_seg
                log_losses["total"] = total_loss

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k, v in log_losses.items():
                self.train_loss_epoch[k] = v.detach().cpu() + self.train_loss_epoch.get(k)

        for k, v in self.train_loss_epoch.items():
            self.train_loss_epoch[k] = v / (step + 1)
            self.train_loss[k] = np.append(self.train_loss[k], self.train_loss_epoch[k])
        wandb.log(
            {f"{phase}_loss": self.train_loss_epoch["total"]},
            step=epoch + 1
        )
        self.lr_scheduler.step(self.train_loss_epoch["total"])

    def valid(self, epoch):
        self.voxel_module.eval()
        self.graph_module.eval()

        seg_metric_batch_decoder = DiceMetric(
            include_background=True,
            reduction="mean_batch", 
            )
        msh_metric_batch_decoder = DiceMetric(
            include_background=True,
            reduction="mean_batch", 
            )

        cached_data = dict()
        choice_case = np.random.choice(len(self.val_loader), 1)[0]
        with torch.no_grad():
            for step, val_data in enumerate(self.val_loader):
                images_batch, labels_batch = (
                    val_data["mr_image"].as_tensor().to(DEVICE),
                    val_data["mr_label"].as_tensor().to(DEVICE),
                    )
                # convert multi-classes labels to binary classes
                labels_batch = torch.logical_or(labels_batch == 2, labels_batch == 4)
                template_meshes = val_data["mr_template_meshes"]["meshes"].to(DEVICE)

                preds_seg, _ = self.voxel_module(images_batch)
                pred_meshes = self.graph_module(
                    template_meshes, preds_seg
                    )
                preds_seg_ = torch.stack([self.post_transform(p) for p in preds_seg[[0, -1]]])
                # TODO: evaluate the error only at slice position before interpolation
                seg_metric_batch_decoder(preds_seg_, labels_batch)

                pred_meshes = pred_meshes[[0, -1]]
                pred_voxels_batch = []
                for pred_mesh in pred_meshes:
                    pred_voxels = self.rasterizer(
                        pred_mesh.verts_padded(), pred_mesh.faces_padded()
                    )
                    pred_voxels_batch.append(pred_voxels)
                pred_voxels_batch = torch.cat(pred_voxels_batch, dim=0)
                # TODO: evaluate the error between seudo mesh and predicted mesh
                msh_metric_batch_decoder(pred_voxels_batch, labels_batch)

                if step == choice_case:
                    cached_data = {
                        # "template_meshes": template_meshes.cpu(),
                        # "images_batch": images_batch.cpu(),
                        # "labels_batch": labels_batch.cpu(),
                        "labels_point_clouds": val_data["mr_label_point_clouds"]["point_clouds"],
                        # "preds_seg": preds_seg.cpu(),
                        "pred_meshes": pred_meshes.cpu(),
                    }

        # log dice score
        self.eval_seg_score["myo"] = np.array([seg_metric_batch_decoder.aggregate().cpu()])
        self.eval_msh_score["myo"] = np.array([msh_metric_batch_decoder.aggregate().cpu()])
        draw_train_loss(self.train_loss, self.super_params, task_code="dynamic_meshing")
        draw_eval_score(self.eval_seg_score, self.super_params, task_code="dynamic_meshing", module="seg")
        draw_eval_score(self.eval_msh_score, self.super_params, task_code="dynamic_meshing", module="msh")
        wandb.log({
            "train_categorised_loss": wandb.Table(
            columns=[f"train_loss \u2193", f"eval_seg_score \u2191", f"eval_msh_score \u2191"],
            data=[[
                wandb.Image(f"{self.super_params.log_dir}/dynamic_meshing/{self.super_params.run_id}/train_loss.png"),
                wandb.Image(f"{self.super_params.log_dir}/dynamic_meshing/{self.super_params.run_id}/eval_seg_loss.png"),
                wandb.Image(f"{self.super_params.log_dir}/dynamic_meshing/{self.super_params.run_id}/eval_msh_loss.png"),
                ]]
            )},
            step=epoch + 1
            )
        eval_score_epoch = msh_metric_batch_decoder.aggregate().mean()
        wandb.log({"eval_score": eval_score_epoch}, step=epoch + 1)

        # save model
        ckpt_weight_path = os.path.join(self.log_dir, "ckpt_weights")
        if not os.path.exists(ckpt_weight_path):
            os.makedirs(ckpt_weight_path)
        torch.save(
            self.voxel_module.state_dict(),
            os.path.join(self.log_dir, "ckpt_weights", f"{epoch + 1}_VoxelProcess.pth")
        )
        torch.save(
            self.mr_handle.state_dict(),
            os.path.join(self.log_dir, "ckpt_weights", f"{epoch + 1}_MRHandle.pth")
        )
        torch.save(
            self.graph_module.state_dict(),
            os.path.join(self.log_dir, "ckpt_weights", f"{epoch + 1}_RStGCN.pth")
        )

        # save visualization when the eval score is the best
        # WARNING: this part is very time-consuming, please comment it if you don"t need it
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
                        point_clouds=cached_data["labels_point_clouds"][0], #ED frame for visualisation
                        pred_meshes=cached_data["pred_meshes"][0]   #ED frame for visualisation
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
                images_batch = val_data[f"mr_image"].as_tensor().to(DEVICE)
                template_meshes = val_data["mr_template_meshes"]["meshes"].to(DEVICE)

                preds_seg, _ = self.voxel_module(images_batch)
                pred_meshes = self.graph_module(
                    template_meshes, preds_seg
                    )
                preds_seg_ = torch.stack([self.post_transform(p) for p in preds_seg[[0, -1]]]).squeeze(1)
                pred_meshes = pred_meshes[[0, -1]]
                for i, frame in enumerate(["ed", "es"]):
                    seg2mesh = matrix_to_marching_cubes(preds_seg_[i].cpu())
                    seg2mesh.export(f"{self.out_dir}/{frame}_pred_VoxelProcess.obj")
                    save_obj(
                        f"{self.out_dir}/{frame}_pred_RStGCN.obj", 
                        pred_meshes[i].verts_packed(), pred_meshes[i].faces_packed()
                    )

