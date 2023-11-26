import numpy as np
import open3d as o3d


def visualize_point_cloud(point_cloud, bbox=None):
    point_cloud.colors = o3d.utility.Vector3dVector(
        np.ones((
            np.asarray(point_cloud.points).shape[0], 3)
        ))

    vis = o3d.visualization.Visualizer()    
    vis.create_window()
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
    vis.add_geometry(point_cloud)

    if bbox is not None:     
        for box in bbox:
            vis.add_geometry(box)
            vis.update_renderer()

    vis.run()
    vis.destroy_window()
