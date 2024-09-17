from typing import Union
from matplotlib import axis
from monai.utils.misc import str2list
import torch
import numpy as np
import pandas as pd
from argparse import Namespace
from torch.nn.functional import one_hot
from torch import Tensor
from pytorch3d.structures import Meshes, Pointclouds
from trimesh.voxel.ops import matrix_to_marching_cubes
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go


__all__ = ["draw_plotly", "draw_train_loss", "draw_eval_score"]

def draw_plotly(
    template_meshes: Union[Meshes, None] = None,
    images: Union[Tensor, None] = None, labels: Union[Tensor, None] = None, 
    point_clouds: Union[Pointclouds, None] = None,
    pred_seg: Union[Tensor, None] = None, pred_meshes: Union[Meshes, None] = None 
    ):
    fig = make_subplots(rows=1, cols=1)
    if images is not None:
        # Add images as cross sections, perpendicular to each other.
        assert images.shape[0] == 1, "Only support one image at a time."
        images = images.squeeze()   # remove batch and channel dimension
        X, Y, Z = images.shape
        _, _, z_grid = np.mgrid[0:X, 0:Y, 0:Z]
        fig.add_traces([
            go.Surface(
                z=Z // 2 * np.ones((X, Y)), surfacecolor=images[..., Z // 2].T, 
                colorscale="Gray", cmin=0, cmax=1,
                showscale=False
                ),
            go.Surface(
                z=z_grid[0], x=X // 2 * np.ones((Y, Z)), surfacecolor=images[X // 2], 
                colorscale="Gray", cmin=0, cmax=1,
                showscale=False
                ),
        ])

    if labels is not None:
        # Add labels as mesh, first convert each label into a mesh using marching cubes, then add to the figure.
        assert labels.shape[0] == 1, "Only support one label at a time."
        num_classes = torch.unique(labels).shape[0]
        if num_classes == 2:
            mesh = matrix_to_marching_cubes(labels.squeeze().cpu().numpy())
            y, x, z = mesh.vertices.T
            I, J, K = mesh.faces.T
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=I, j=J, k=K,
                color="pink",
                opacity=0.25,
                name="seg_label"
            ))
        else:
            raise ValueError("Only support binary segmentation for now.")
    
    if pred_meshes is not None:
        # Add pred_meshes as mesh, which is pytorch3d.Meshes object. first rescale to the same size as the label mesh, and shift to the center of labels, then add to the figure.
        pred_meshes.scale_verts_(labels.shape[-1] / 2)
        pred_meshes.offset_verts_(torch.tensor([labels.shape[-1] / 2] * 3, dtype=torch.float32, device=pred_meshes.device))
        for pred_mesh in pred_meshes:
            x, y, z = pred_mesh.verts_packed().T
            I, J, K = pred_mesh.faces_packed().T
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=I, j=J, k=K,
                color="blue",
                opacity=0.1,
                name="meshes_pred"
            ))
    
    # Add layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=4, range=[0, 127],),
            yaxis=dict(nticks=4, range=[0, 127],),
            zaxis=dict(nticks=4, range=[0, 127],),
            aspectratio=dict(x=1, y=1, z=1),
            camera_eye=dict(x=1.2, y=1.2, z=1.2)
        ),
        margin=dict(r=0, l=0, b=0, t=0)
    )

    return fig

def draw_train_loss(train_loss: dict, super_params: Namespace, task_code: str, phase: str):
    sns.set_theme(style="whitegrid")
    _, ax = plt.subplots(figsize=(10, 8))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    df = pd.DataFrame(train_loss)
    if phase == "fine_tune":
        df.set_index(super_params.delay_epochs + df.index + 1, inplace=True)
        lambda_ = super_params.lambda_[1:]
    else:
        df.set_index(df.index + 1, inplace=True)
        lambda_ = super_params.lambda_
    for i, coeff in enumerate(lambda_, start=1):
        df.iloc[:, i] = df.iloc[:, i - 1] - coeff * df.iloc[:, i]

    if len(df) > 0:
        colors = sns.color_palette("hls", len(df.columns.values))
        for i in range(len(df.columns.values) - 1):
            ax = sns.lineplot(
                x=df.index.values, y=df.iloc[:, i], 
                ax=ax, color=colors[i], label=df.columns[i+1]
            )
            curve = ax.lines[i]
            x_i = curve.get_xydata()[:, 0]
            y_i = curve.get_xydata()[:, 1]
            ax.fill_between(x_i, y_i, color=colors[i], alpha=0.6)
        plt.legend()

    plt.savefig(f"{super_params.ckpt_dir}/{task_code}/{super_params.run_id}/{phase}_loss.png")

def draw_eval_score(eval_score: dict, super_params: Namespace, task_code: str, module: str):
    df = pd.DataFrame(eval_score)
    df["Epoch"] = super_params.delay_epochs + (df.index + 1) * super_params.val_interval
    df_melted = df.melt(id_vars="Epoch", var_name="Label", value_name="Score")
    mean_scores = df.drop("Epoch", axis=1).mean(axis=1)
    mean_scores.name = 'Average Score'
    sns.set_theme(style="whitegrid")
    _, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(x=df_melted["Epoch"], y=df_melted["Score"], ax=ax, color="skyblue", showfliers=False, width=0.2)
    sns.lineplot(x=mean_scores.index.values, y=mean_scores, ax=ax, color="green", label="Average")
    LOW = df.drop("Epoch", axis=1).idxmin(axis=1)
    HIGH = df.drop("Epoch", axis=1).idxmax(axis=1)
    for epoch, (l, h) in enumerate(zip(LOW, HIGH)):
        ax.text(epoch, df.loc[epoch, l], f'{l}', horizontalalignment="center", color="black", weight="semibold")
        ax.text(epoch, df.loc[epoch, h], f'{h}', horizontalalignment="center", color="black", weight="semibold")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.savefig(f"{super_params.ckpt_dir}/{task_code}/{super_params.run_id}/eval_{module}_loss.png")