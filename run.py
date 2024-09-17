import os, sys, json
from collections import OrderedDict
import numpy as np
from itertools import chain
import torch
import torch.nn.functional as F
from trimesh.voxel.ops import matrix_to_marching_cubes
from pytorch3d.io import save_obj, save_ply, load_objs_as_meshes
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
import wandb

import plotly.io as pio
import plotly.graph_objects as go

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
            seed, num_workers,
            is_training=True,
        ):
        """
        :param 
            super_params: parameters for setting up dataset, network structure, training, etc.
            seed: random seed to shuffle data during augmentation.
            num_workers: tourch.utils.data.DataLoader num_workers.
            is_training: switcher for training (True, default) or testing (False).
        """
            
        self.super_params = super_params
        self.seed = seed
        self.num_workers = num_workers
        self.is_training = is_training
        set_determinism(seed=self.seed)

        if is_training:
            self.ckpt_dir = os.path.join(super_params.ckpt_dir, "stationary", super_params.run_id)
            os.makedirs(self.ckpt_dir, exist_ok=True)
            self.train_loss = OrderedDict(
                {k: np.asarray([]) for k in ["total", "seg", "chamfer", "edge", "face", "laplacian"]}
            )
            self.fine_tune_loss = OrderedDict(
                {k: np.asarray([]) for k in ["total", "chamfer", "edge", "face", "laplacian"]}
            )
            self.eval_seg_score = OrderedDict(
                {k: np.asarray([]) for k in ["myo"]}
            )
            self.eval_msh_score = self.eval_seg_score.copy()
            self.best_eval_score = 0
        else:
            self.ckpt_dir = super_params.ckpt_dir
            self.out_dir = super_params.out_dir
            os.makedirs(self.out_dir, exist_ok=True)

        self.post_transform = Compose([
            AsDiscrete(argmax=True),
            KeepLargestConnectedComponent(is_onehot=False),
            EnsureType(data_type="tensor", dtype=torch.float32, device=DEVICE)
            ])

        if not is_training and super_params.save_on != 'cap':
            pass
        else:
            with open(super_params.mr_json_dir, "r") as f:
                mr_train_transform, mr_valid_transform = self._prepare_transform(["mr_image", "mr_label"])
                mr_train_ds, mr_valid_ds, mr_test_ds = self._prepare_dataset(
                    json.load(f), "mr", mr_train_transform, mr_valid_transform
                )
                self.mr_train_loader, self.mr_valid_loader, self.mr_test_loader = self._prepare_dataloader(
                    mr_train_ds, mr_valid_ds, mr_test_ds
                )

        if not is_training and super_params.save_on != 'sct':
            pass
        else:
            with open(super_params.ct_json_dir, "r") as f:
                ct_train_transform, ct_valid_transform = self._prepare_transform(["ct_image", "ct_label"])
                ct_train_ds, ct_valid_ds, ct_test_ds = self._prepare_dataset(
                    json.load(f), "ct", ct_train_transform, ct_valid_transform
                )
                self.ct_train_loader, self.ct_valid_loader, self.ct_test_loader = self._prepare_dataloader(
                    ct_train_ds, ct_valid_ds, ct_test_ds
                )

        self._prepare_models()

    def _prepare_transform(self, keys):
        if not self.is_training:
            val_transform = pre_transform(
                keys, self.super_params.crop_window_size, "test",
                )
            return None, val_transform
        else:
            train_transform = pre_transform(
                keys, self.super_params.crop_window_size, "training",
                )
            val_transform = pre_transform(
                keys, self.super_params.crop_window_size, "validation",
                )
            return train_transform, val_transform

    def _remap_abs_path(self, data_list, modal):
        if modal == "mr":
            return [{
                "mr_image": os.path.join(self.super_params.mr_data_dir, "imagesTr", os.path.basename(d["image"])),
                "mr_label": os.path.join(self.super_params.mr_data_dir, "labelsTr", os.path.basename(d["label"])),
            } for d in data_list]
        elif modal == "ct":
            return [{
                "ct_image": os.path.join(self.super_params.ct_data_dir, "imagesTr", os.path.basename(d["image"])),
                "ct_label": os.path.join(self.super_params.ct_data_dir, "labelsTr", os.path.basename(d["label"])),
            } for d in data_list]
        
    def _prepare_dataset(self, data_json, modal, train_transform, valid_transform):
        train_data = self._remap_abs_path(data_json["train_fold0"], modal)
        valid_data = self._remap_abs_path(data_json["validation_fold0"], modal)
        test_data = self._remap_abs_path(data_json["test"], modal)

        train_ds = Dataset(
            train_data, train_transform, self.seed, sys.maxsize,
            self.super_params.cache_rate, self.num_workers
            )
        valid_ds = Dataset(
            valid_data, valid_transform, self.seed, sys.maxsize,
            self.super_params.cache_rate, self.num_workers
            )
        test_ds = Dataset(
            test_data, valid_transform, self.seed, sys.maxsize,
            self.super_params.cache_rate, self.num_workers
            )
        
        return train_ds, valid_ds, test_ds

    def _prepare_dataloader(self, train_ds, valid_ds, test_ds):
        if train_ds.__len__() == 0:
            train_loader = None
        else:
            train_loader = DataLoader(
                train_ds, batch_size=1,
                shuffle=True, num_workers=self.num_workers,
                )
        if valid_ds.__len__() == 0:
            val_loader = None
        else:
            val_loader = DataLoader(
                valid_ds, batch_size=1,
                shuffle=False, num_workers=self.num_workers,
                )
        test_loader = DataLoader(
            test_ds, batch_size=1,
            shuffle=False, num_workers=self.num_workers,
            )
        
        return train_loader, val_loader, test_loader

    def _prepare_models(self, ):
        self.mr_handle = ModalityHandle(
            init_filters=self.super_params.init_filters,
            in_channels=self.super_params.in_channels,
            out_channels=self.super_params.num_classes,
            num_init_blocks=self.super_params.num_init_blocks,
        ).to(DEVICE)
        self.ct_handle = ModalityHandle(
            init_filters=self.super_params.init_filters,
            in_channels=self.super_params.in_channels,
            out_channels=self.super_params.num_classes,
            num_init_blocks=self.super_params.num_init_blocks,
        ).to(DEVICE)
        self.resnet = ResNetDecoder(
            init_filters=self.super_params.init_filters,
            out_channels=self.super_params.num_classes,
            norm=self.mr_handle.norm, act=self.mr_handle.act,
            num_layers_blocks=self.super_params.num_up_blocks,
            ).to(DEVICE)
        
        # take the mesh at the first frame as template
        self.template_mesh = load_objs_as_meshes([self.super_params.template_dir], load_textures=False)
        # select the label from segmentation based on the template_dir
        if "myo" in self.super_params.template_dir:
            self.seg_indices = "myo"
        elif "lv" in self.super_params.template_dir:
            self.seg_indices = 1
        elif "rv" in self.super_params.template_dir:
            self.seg_indices = 3

        self.graph_module = RStGCN(
            hidden_features=32, num_blocks=3,
            sdm_out_channel=self.super_params.num_classes - 1,  # exclude background
            template_mesh=self.template_mesh,
            attention=False,
            task_code="stationary"
            ).to(DEVICE)
        if self.super_params.pre_trained_ct_module_dir is not None and self.super_params.pre_trained_mr_module_dir is not None:
            self.ct_handle.load_state_dict(torch.load(self.super_params.pre_trained_ct_module_dir))
            self.mr_handle.load_state_dict(torch.load(self.super_params.pre_trained_mr_module_dir))
        else:
            print("WARN: loading no pre-trained voxel module")

        self.rasterizer = Rasterize(self.super_params.crop_window_size) # tool for rasterizing mesh

        self.seg_loss = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            ).to(DEVICE)
        self.downsample_seg_loss = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            ).to(DEVICE)
        self.mesh_loss = TotalLosses(
            lambda_=self.super_params.lambda_[1:], time_series=False
            ).to(DEVICE)

        self.optimizer_train = torch.optim.Adam(
            chain(self.mr_handle.parameters(), self.ct_handle.parameters(), self.resnet.parameters(),
                  self.graph_module.parameters()), 
            lr=self.super_params.lr
            )
        self.lr_scheduler_train = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_train)
        self.optimizer_fine_tune = torch.optim.Adam(
            self.graph_module.parameters(), 
            lr=self.super_params.lr
            )
        self.lr_scheduler_fine_tune = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_fine_tune)
        self.scaler = torch.cuda.amp.GradScaler()

        torch.backends.cudnn.enabled = torch.backends.cudnn.is_available()
        torch.backends.cudnn.benchmark = torch.backends.cudnn.is_available()

    def seudo_extractor(self, seg):
        verts, *_ = marching_cubes(
            seg.squeeze(1).permute(0, 3, 1, 2), 
            isolevel=0.5,
            return_local_coords=True,
        )
        # verts =[vert[:, [1, 0, 2]] for vert in verts]
        pc_suedo = Pointclouds(points=verts)

        return pc_suedo
    
    def train_iter(self, epoch, phase):
        if phase == "train":
            self.mr_handle.train()
            self.ct_handle.train()
            self.resnet.train()
            self.graph_module.train()

            train_loss = {tag: 0.0 for tag in ["total", "seg", "chamfer", "edge", "face", "laplacian"]}
            for step, (data_ct, data_mr) in enumerate(zip(self.ct_train_loader, self.mr_train_loader)):
                image_mr, label_mr, label_ds_mr = (
                    data_mr["mr_image"].as_tensor().to(DEVICE),
                    data_mr["mr_label"].as_tensor().to(DEVICE),
                    data_mr["mr_label_downsample"].as_tensor().to(DEVICE)
                    )
                image_ct, label_ct, label_ds_ct = (
                    data_ct["ct_image"].as_tensor().to(DEVICE),
                    data_ct["ct_label"].as_tensor().to(DEVICE),
                    data_ct["ct_label_downsample"].as_tensor().to(DEVICE)
                    )
                
                # select the class label from segmentation
                label_ct = torch.logical_or(label_ct == 2, label_ct == 4) if isinstance(self.seg_indices, str) else (label_ct == self.seg_indices)
                label_mr = torch.logical_or(label_mr == 2, label_mr == 4) if isinstance(self.seg_indices, str) else (label_mr == self.seg_indices)
                label_ds_ct = torch.logical_or(label_ds_ct == 2, label_ds_ct == 4) if isinstance(self.seg_indices, str) else (label_ds_ct == self.seg_indices)
                label_ds_mr = torch.logical_or(label_ds_mr == 2, label_ds_mr == 4) if isinstance(self.seg_indices, str) else (label_ds_mr == self.seg_indices)

                self.optimizer_train.zero_grad()
                # forward the Voxel Processing Module
                _, pred_ds_ct = self.ct_handle(image_ct)
                pred_ct, _ = self.resnet(pred_ds_ct)
                _, pred_ds_mr = self.mr_handle(image_mr)
                pred_mr, _ = self.resnet(pred_ds_mr)
                loss_seg = self.seg_loss(
                    torch.cat([pred_ct, pred_mr], dim=0), torch.cat([label_ct, label_mr], dim=0).long()
                    ) +\
                        self.downsample_seg_loss(
                            torch.cat([pred_ds_ct, pred_ds_mr], dim=0), torch.cat([label_ds_ct, label_ds_mr], dim=0).long()
                            )
                
                # forward the R-StGCN
                pc_seudo = self.seudo_extractor(label_ct.float())
                template_meshes = self.template_mesh.extend(len(image_ct)).to(DEVICE)
                pred_meshes = self.graph_module(template_meshes, label_ct.float(), subdiv_level=self.super_params.subdiv_level)
                loss_mesh, log_losses = self.mesh_loss(pred_meshes, pc_seudo)
                loss = self.super_params.lambda_[0] * loss_seg + loss_mesh

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer_train)
                self.scaler.update()

                train_loss["total"] += loss.item()
                train_loss["seg"] += loss_seg.item()
                for k, v in log_losses.items():
                    train_loss[k] += v.item()

            for k, v in train_loss.items():
                train_loss[k] = v / (step + 1)
                self.train_loss[k] = np.append(self.train_loss[k], train_loss[k])
            wandb.log(
                {f"{phase}_loss": train_loss["total"]},
                step=epoch + 1
            )
            self.lr_scheduler_train.step(train_loss["total"])

        elif phase == "fine_tune":
            self.mr_handle.eval()
            self.resnet.eval()
            self.graph_module.train()

            fine_tune_loss = {tag: 0.0 for tag in ["total", "chamfer", "edge", "face", "laplacian"]}

            for step, data_mr in enumerate(self.mr_train_loader):
                image_mr = data_mr["mr_image"].as_tensor().to(DEVICE)

                self.optimizer_fine_tune.zero_grad()
                pred_mr, _ = self.resnet(self.mr_handle(image_mr)[1])
        
                pred_mr_ = torch.stack([self.post_transform(p) for p in pred_mr.detach()])
                if len(pred_mr_.unique()) == 1:
                    continue
                pc_seudo = self.seudo_extractor(pred_mr_)   # get seudo point clouds
                template_meshes = self.template_mesh.extend(len(image_mr)).to(DEVICE)
                pred_meshes = self.graph_module(template_meshes, pred_mr_.detach(), subdiv_level=self.super_params.subdiv_level)
                loss, log_losses = self.mesh_loss(pred_meshes, pc_seudo)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer_fine_tune)
                self.scaler.update()

                fine_tune_loss["total"] += loss.item()
                for k, v in log_losses.items():
                    fine_tune_loss[k] += v.item()

            for k, v in fine_tune_loss.items():
                fine_tune_loss[k] = v / (step + 1)
                self.fine_tune_loss[k] = np.append(self.fine_tune_loss[k], fine_tune_loss[k])
            wandb.log(
                {f"{phase}_loss": fine_tune_loss["total"]},
                step=epoch + 1
            )
            self.lr_scheduler_fine_tune.step(fine_tune_loss["total"])

    def valid(self, epoch, save_on):
        self.resnet.eval()
        self.graph_module.eval()

        if save_on == "cap":
            modal = "mr"
            val_loader = self.mr_valid_loader
            self.mr_handle.eval()
            handle = self.mr_handle
        elif save_on == "sct":
            modal = "ct"
            val_loader = self.ct_valid_loader
            self.ct_handle.eval()
            handle = self.ct_handle

        seg_metric_batch_decoder = DiceMetric(
            include_background=False, reduction="mean_batch",
            )
        msh_metric_batch_decoder = DiceMetric(reduction="mean_batch")

        cached_data = dict()
        choice_case = np.random.choice(len(val_loader), 1)[0]
        with torch.no_grad():
            for step, val_data in enumerate(val_loader):
                image, label = (
                    val_data[f"{modal}_image"].as_tensor().to(DEVICE),
                    val_data[f"{modal}_label"].as_tensor().to(DEVICE),
                    )
                # select the class label from segmentation
                label = torch.logical_or(label == 2, label == 4) if isinstance(self.seg_indices, str) else (label == self.seg_indices)

                pred, _ = self.resnet(handle(image)[1])
                pred_ = torch.stack([self.post_transform(p) for p in pred])
                seg_metric_batch_decoder(pred_, label.long())

                template_meshes = self.template_mesh.extend(len(image)).to(DEVICE)
                pred_meshes = self.graph_module(template_meshes, pred_, subdiv_level=self.super_params.subdiv_level)

                pred_voxeld = []
                for pred_mesh in pred_meshes:
                    pred_voxel = self.rasterizer(
                        pred_mesh.verts_padded(), pred_mesh.faces_padded()
                    )
                    pred_voxeld.append(pred_voxel)
                pred_voxeld = torch.cat(pred_voxeld, dim=0)
                msh_metric_batch_decoder(pred_voxeld, label.long())

                if step == choice_case:
                    cached_data = {
                        "labels_batch": label.cpu(),
                        "pred_meshes": pred_meshes.cpu(),
                    }

        # log dice score
        self.eval_seg_score["myo"] = np.append(self.eval_seg_score["myo"], seg_metric_batch_decoder.aggregate().cpu())
        self.eval_msh_score["myo"] = np.append(self.eval_msh_score["myo"], msh_metric_batch_decoder.aggregate().cpu())
        draw_train_loss(self.train_loss, self.super_params, task_code="stationary", phase="train")
        draw_train_loss(self.fine_tune_loss, self.super_params, task_code="stationary", phase="fine_tune")
        draw_eval_score(self.eval_seg_score, self.super_params, task_code="stationary", module="seg")
        draw_eval_score(self.eval_msh_score, self.super_params, task_code="stationary", module="msh")
        wandb.log({
            "train_categorised_loss": wandb.Table(
            columns=[f"train_loss \u2193", f"fine_tune_loss \u2191", f"eval_seg_score \u2191", f"eval_msh_score \u2191"],
            data=[[
                wandb.Image(f"{self.super_params.ckpt_dir}/stationary/{self.super_params.run_id}/train_loss.png"),
                wandb.Image(f"{self.super_params.ckpt_dir}/stationary/{self.super_params.run_id}/fine_tune_loss.png"),
                wandb.Image(f"{self.super_params.ckpt_dir}/stationary/{self.super_params.run_id}/eval_seg_loss.png"),
                wandb.Image(f"{self.super_params.ckpt_dir}/stationary/{self.super_params.run_id}/eval_msh_loss.png"),
                ]]
            )},
            step=epoch + 1
            )
        eval_score_epoch = msh_metric_batch_decoder.aggregate().mean()
        wandb.log({"eval_score": eval_score_epoch}, step=epoch + 1)

        # save model
        ckpt_weight_path = os.path.join(self.ckpt_dir, "ckpt_weights")
        os.makedirs(ckpt_weight_path, exist_ok=True)
        torch.save(
            self.mr_handle.state_dict(),
            os.path.join(self.ckpt_dir, "ckpt_weights", f"{epoch + 1}_MRHandle.pth")
            )
        torch.save(
            self.ct_handle.state_dict(),
            os.path.join(self.ckpt_dir, "ckpt_weights", f"{epoch + 1}_CTHandle.pth")
            )
        torch.save(
            self.resnet.state_dict(),
            os.path.join(self.ckpt_dir, "ckpt_weights", f"{epoch + 1}_ResNet.pth")
            )
        torch.save(
            self.graph_module.state_dict(),
            os.path.join(self.ckpt_dir, "ckpt_weights", f"{epoch + 1}_RStGCN.pth")
            )

        # save visualization when the eval score is the best
        if eval_score_epoch > self.best_eval_score:
            self.best_eval_score = eval_score_epoch
            wandb.log({
                "seg_label vs mesh_pred": wandb.Plotly(draw_plotly(
                    labels=cached_data["labels_batch"],
                    pred_meshes=cached_data["pred_meshes"]
                    ))
                }, step=epoch + 1)
         
    @torch.no_grad()
    def test(self, save_on):
        # load_model
        self.resnet.load_state_dict(
            torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_ResNet.pth")))
        self.graph_module.load_state_dict(
            torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_RStGCN.pth")))
        self.resnet.eval()
        self.graph_module.eval()
        if save_on in "cap":
            modal = "mr"
            val_loader = self.mr_test_loader
            self.mr_handle.load_state_dict(
                torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_MRHandle.pth")))
            self.mr_handle.eval()
            handle = self.mr_handle
        elif save_on in "sct":
            modal = "ct"
            val_loader = self.ct_test_loader
            self.ct_handle.load_state_dict(
                torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_CTHandle.pth")))
            self.ct_handle.eval()
            handle = self.ct_handle
        else:
            raise ValueError("save_on should be either 'cap' or 'sct'")

        # testing
        for i, val_data in enumerate(val_loader):
            id = os.path.basename(val_loader.dataset.data[i][f"{modal}_label"]).replace(".nii.gz", '')
            image = val_data[f"{modal}_image"].as_tensor().to(DEVICE)

            pred, _ = self.resnet(handle(image)[1])
            pred_ = torch.stack([self.post_transform(p) for p in pred])

            template_meshes = self.template_mesh.extend(len(image)).to(DEVICE)
            pred_meshes = self.graph_module(template_meshes, pred_, subdiv_level=self.super_params.subdiv_level)

            save_obj(
                f"{self.out_dir}/{id}.obj", 
                pred_meshes.verts_packed(), pred_meshes.faces_packed()
            )
