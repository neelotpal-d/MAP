"""Boundary samples and the build-axis band shared by training and preview."""
import numpy as np


def surface_samples(vertices, faces, up_axis, base_band_mm):
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-12
    centers = triangles.mean(axis=1)[valid]
    normals = cross[valid] / double_area[valid, None]
    area = double_area[valid] / 2
    axis = 'xyz'.index(up_axis)
    base = (centers[:, axis] < vertices[:, axis].min() + base_band_mm) if base_band_mm > 0 else np.zeros(len(centers), dtype=bool)
    return valid, centers, normals, area, base
