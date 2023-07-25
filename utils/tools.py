from typing import Union
from matplotlib import axis
from monai.utils.misc import str2list
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
        labels = one_hot(
            labels.long().squeeze(1), num_classes=7
        ).permute(0, 4, 1, 2, 3).squeeze().cpu().numpy()
        label_names = ["lv", "lv_myo", "rv", "rv_myo", "la", "ra"]
        for i, label in enumerate(labels[1:]):
            mesh = matrix_to_marching_cubes(label)
            x, y, z = mesh.vertices.T
            I, J, K = mesh.faces.T
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=I, j=J, k=K,
                color="pink",
                opacity=0.25,
                name=f"{label_names[i]}_seg_label"
            ))
    
    if pred_seg is not None:
        # Add pred_seg as mesh, first convert each label into a mesh using marching cubes, then add to the figure.
        pred_seg = one_hot(
            pred_seg.long().squeeze(1), num_classes=7
        ).permute(0, 4, 1, 2, 3).squeeze().cpu().numpy()
        label_names = ["lv", "lv_myo", "rv", "rv_myo", "la", "ra"]
        for i, label in enumerate(pred_seg[1:]):
            mesh = matrix_to_marching_cubes(label)
            x, y, z = mesh.vertices.T
            I, J, K = mesh.faces.T
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=I, j=J, k=K,
                color="blue",
                opacity=0.25,
                name=f"{label_names[i]}_seg_pred"
            ))

    if point_clouds is not None:
        # Add point_clouds as scatter, which is pytorch3d.Pointclouds object. Give different colors to different labels, except blue.
        point_clouds = point_clouds.subsample(5_000)
        point_clouds = point_clouds.update_padded(
            64 * (point_clouds.points_padded() + 1)
        )
        x, y, z = point_clouds.points_list()[0].T
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(
                size=1,
                color="pink",
                opacity=1.0
            ),
            name="point_clouds_label"
        ))
    
    if template_meshes is not None:
        # Add template mesh, which is pytorch3d.Meshes object. first rescale to the same size as the label mesh, and shift to the center of labels, then add to the figure.
        for template_mesh in template_meshes:
            template_mesh = template_mesh.update_padded(
                64 * (template_mesh.verts_padded() + 1)
            )
            x, y, z = template_mesh.verts_list()[0].T
            I, J, K = template_mesh.faces_list()[0].T
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=I, j=J, k=K,
                color="cyan",
                opacity=0.1,
                name="template_meshes"
            ))
    
    if pred_meshes is not None:
        # Add pred_meshes as mesh, which is pytorch3d.Meshes object. first rescale to the same size as the label mesh, and shift to the center of labels, then add to the figure.
        pred_meshes = pred_meshes.update_padded(
            64 * (pred_meshes.verts_padded() + 1)
        )
        x, y, z = pred_meshes.verts_packed().T
        I, J, K = pred_meshes.faces_packed().T
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

def draw_train_loss(train_loss: dict, super_params: Namespace):
    df = pd.DataFrame(train_loss)
    df.set_index(super_params.delay_epochs + df.index + 1, inplace=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    colors = sns.color_palette("hls", len(df.columns.values))
    for i, coeff in enumerate(super_params.lambda_, start=1):
        df.iloc[:, i] = df.iloc[:, i - 1] - coeff * df.iloc[:, i]
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
    plt.savefig(f"{super_params.log_dir}/whole_heart_meshing/{super_params.run_id}/train_loss.png")

def draw_eval_score(eval_score: dict, super_params: Namespace, task: str):
    df = pd.DataFrame(eval_score)
    df["Epoch"] = super_params.delay_epochs + (df.index + 1) * super_params.val_interval
    df_melted = df.melt(id_vars="Epoch", var_name="Label", value_name="Score")
    mean_scores = df.drop("Epoch", axis=1).mean(axis=1)
    mean_scores.name = 'Average Score'
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(x=df_melted["Epoch"], y=df_melted["Score"], ax=ax, color="skyblue", showfliers=False, width=0.2)
    sns.lineplot(x=mean_scores.index.values, y=mean_scores, ax=ax, color="green", label="Average")
    LOW = df.drop("Epoch", axis=1).idxmin(axis=1)
    HIGH = df.drop("Epoch", axis=1).idxmax(axis=1)
    for epoch, (l, h) in enumerate(zip(LOW, HIGH)):
        ax.text(epoch, df.loc[epoch, l], f'{l}', horizontalalignment="center", color="black", weight="semibold")
        ax.text(epoch, df.loc[epoch, h], f'{h}', horizontalalignment="center", color="black", weight="semibold")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.savefig(f"{super_params.log_dir}/whole_heart_meshing/{super_params.run_id}/eval_{task}_loss.png")