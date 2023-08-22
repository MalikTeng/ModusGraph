import os
from tqdm import tqdm
import numpy as np
from trimesh.voxel.ops import matrix_to_marching_cubes
import plotly.graph_objects as go
import plotly.io as pio

from data.transform import *


class Args:
    def __init__(self):
        self.keys = ("ct_image", "ct_label")
        self.dataset = "scotheart"
        self.crop_window_size = [128, 128, 128]
        self.section = "test"
        self.point_limit = 53_500
        self.one_or_multi = "multi"
        self.template_dir = "/home/yd21/Documents/ModusGraph/template/scotheart/"


if __name__ == '__main__':
    args = Args()
    test_transform = pre_transform(
        keys=args.keys, dataset=args.dataset, crop_window_size=args.crop_window_size,
        section=args.section, point_limit=args.point_limit, 
        one_or_multi=args.one_or_multi, template_dir=args.template_dir
    )
    
    # directory setting
    source_root_dir = "/mnt/data/Experiment/nnUNet/nnUNet_raw/nnUNet_raw_data/Task504_SCOTHEART/"

    patient_list = [i.strip(".nii.gz") for i in os.listdir(os.path.join(source_root_dir, "labelsTr"))][:1]
    for patient_id in tqdm(patient_list):
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

        images = output_data["ct_image"][0].get_array()
        labels = output_data["ct_label"][0].get_array()
        point_clouds = output_data["ct_label_point_clouds"]["point_clouds"]
        template_meshes = output_data["ct_template_meshes"]["meshes"]

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

        # Add labels as mesh, first convert each label into a mesh using marching cubes, then add to the figure.
        label_names = ["lv", "lv_myo", "rv", "rv_myo", "la", "ra"]
        labels = [np.where(labels == i, 1, 0) for i in range(1, 7)]
        for i, label in enumerate(labels):
            mesh = matrix_to_marching_cubes(label)
            x, y, z = mesh.vertices.T
            I, J, K = mesh.faces.T
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=I, j=J, k=K,
                color="blue",
                opacity=0.25,
                name=f"{label_names[i]}_seg"
            ))

        # Add point_clouds as scatter, which is pytorch3d.Pointclouds object. Give different colors to different labels, except blue.
        colours = ["red", "green", "yellow", "orange", "purple", "brown"]
        for i, point_cloud in enumerate(point_clouds):
            point_cloud = point_cloud.subsample(5_000)
            point_cloud = point_cloud.update_padded(
                64 * (point_cloud.points_padded() + 1)
            )
            x, y, z = point_cloud.points_list()[0].T
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                marker=dict(
                    size=1,
                    color=colours[i],
                    opacity=0.8
                ),
                name=f"{label_names[i]}_mesh"
            ))
            
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

        pio.write_html(fig, file=f"test_whm_scotheart-{patient_id}.html")

        # # process meshes to create template meshes
        # # combine meshes into one mesh
        # mesh = meshes[0]
        # for i in range(1, len(meshes)):
        #     mesh = mesh + meshes[i]
        # # find the center of mesh's bounding box
        # mesh_center = mesh.bounding_box.centroid
        # # find the longest length of side of mesh's bounding box
        # mesh_length = max(mesh.bounding_box.extents)
        # # apply translation and resize to every mesh, and save for creating template using Trimesh.export
        # for i, mesh in enumerate(meshes):
        #     mesh.apply_translation(-mesh_center)
        #     mesh.apply_scale(2 / mesh_length)
        #     mesh.export(f"/home/yd21/Documents/ModusGraph/tools/create_template/source_scotheart/init_mesh-{label_names[i]}.obj")