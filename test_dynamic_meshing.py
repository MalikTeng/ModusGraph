import os
from tqdm import tqdm
import numpy as np
from trimesh.voxel.ops import matrix_to_marching_cubes
import plotly.graph_objects as go
import plotly.io as pio

from data.transform import *


def draw_plots(images, labels, point_clouds, template_meshes, patient_id, dataset):
    # illustrate images and labels as volume slices, and meshes as meshes using plotly
    # Define frames
    # images, labels = images.T, labels.T    # cardiac relative coordinate in: (z, y, x) voxel coordinate -> (x, y, z) world coordinate (optional)
    X, Y, Z = images.shape

    # Add images as cross sections, perpendicular to each other.
    _, _, z_grid = np.mgrid[0:X, 0:Y, 0:Z]
    fig = go.Figure(data=[
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

    # Add label as mesh, first convert myo label into a mesh using marching cubes, then add to the figure.
    label = np.logical_or(labels == 2, labels == 4)
    mesh = matrix_to_marching_cubes(label)
    x, y, z = mesh.vertices.T
    I, J, K = mesh.faces.T
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=I, j=J, k=K,
        color="red",
        opacity=0.25,
        name="voxel_label"
    ))

    # Add point_clouds as scatter, which is pytorch3d.Pointclouds object. Give different colors to different labels, except blue.
    point_cloud = point_clouds.update_padded(
        64 * (point_clouds.points_padded() + 1)
    )
    x, y, z = point_cloud.points_list()[0].T
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(
            size=1,
            color="yellow",
            opacity=0.8
        ),
        name="point_clouds"
    ))
        
    # Add template mesh, which is pytorch3d.Meshes object. first rescale to the same size as the label mesh, and shift to the center of labels, then add to the figure.
    template_mesh = template_meshes.update_padded(
        64 * (template_meshes.verts_padded() + 1)
    )
    x, y, z = template_mesh.verts_list()[0].T
    I, J, K = template_mesh.faces_list()[0].T
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=I, j=J, k=K,
        color="cyan",
        opacity=0.1,
        name=f"template"
    ))
    
    # Add layout with legend and 3 axis
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x"),
            yaxis=dict(title="y"),
            zaxis=dict(title="z"),
            aspectratio=dict(x=1, y=1, z=1),
            camera_eye=dict(x=1.5, y=1.5, z=1.5)
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    pio.write_html(fig, file=f"test_dynamic_{dataset}-{patient_id}.html")


class Args:
    def __init__(self):
        self.keys = ("mr_image", "mr_label")
        self.dataset = "cap"
        self.crop_window_size = [128, 128, 128]
        self.section = "test"
        self.point_limit = 11_612
        self.one_or_multi = "solo"
        self.template_dir = "/home/yd21/Documents/ModusGraph/template/cap/"


if __name__ == '__main__':
    args = Args()
    
    # TEST CAP
    # transformation setting
    test_transform = pre_transform(
        keys=args.keys, dataset=args.dataset, crop_window_size=args.crop_window_size,
        section=args.section, point_limit=args.point_limit, 
        one_or_multi=args.one_or_multi, template_dir=args.template_dir
    )
    # directory setting
    source_root_dir = "/mnt/data/Experiment/nnUNet/nnUNet_raw/nnUNet_raw_data/Task507_CAP_ModusGraph/"
 
    patient_list = [i.strip(".nii.gz") for i in os.listdir(f"{source_root_dir}/labelsTs")][:1]
    for patient_id in tqdm(patient_list):
        image_dir = f"{source_root_dir}/imagesTs/{patient_id}_{0:04d}.nii.gz"
        label_dir = f"{source_root_dir}/labelsTs/{patient_id}.nii.gz"

        output_data = test_transform({
            "mr_image": image_dir,
            "mr_label": label_dir
        })

        # pick out the ed frame (the 1st frame) of all frames
        images = output_data["mr_image"][0].get_array()
        labels = output_data["mr_label"][0].get_array()
        point_clouds = output_data["mr_label_point_clouds"]["point_clouds"][0]
        template_meshes = output_data["mr_template_meshes"]["meshes"][0]

        # draw plots for checking overlayed images, labels, point_clouds and template_meshes
        draw_plots(images, labels, point_clouds, template_meshes, patient_id, args.dataset)

    # TEST SCOTHEART
    # update arguments of transformation
    test_transform = pre_transform(
        keys=("ct_image", "ct_label"), dataset="scotheart", crop_window_size=args.crop_window_size,
        section=args.section, point_limit=args.point_limit, 
        one_or_multi=args.one_or_multi, template_dir=args.template_dir
    )
    # directory setting
    source_root_dir = "/mnt/data/Experiment/nnUNet/nnUNet_raw/nnUNet_raw_data/Task508_SCOTCAP/"

    patient_list = [i.strip(".nii.gz") for i in os.listdir(f"{source_root_dir}/labelsTr")][:1]
    for patient_id in tqdm(patient_list):
        image_dir = f"{source_root_dir}/imagesTr/{patient_id}_{0:04d}.nii.gz"
        label_dir = f"{source_root_dir}/labelsTr/{patient_id}.nii.gz"

        output_data = test_transform({
            "ct_image": image_dir,
            "ct_label": label_dir
        })

        # pick out the ed frame (the 1st frame) of all frames
        images = output_data["ct_image"][0].get_array()
        labels = output_data["ct_label"][0].get_array()
        point_clouds = output_data["ct_label_point_clouds"]["point_clouds"][0]
        template_meshes = output_data["ct_template_meshes"]["meshes"][0]

        # draw plots for checking overlayed images, labels, point_clouds and template_meshes
        draw_plots(images, labels, point_clouds, template_meshes, patient_id, "scotheart")
