"""
Utilities for exporting ordered toolpaths with surface normals.

The current normal evaluation uses a face-centroid KD-tree to shortlist nearby
triangles before running exact closest-point tests.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from heat_method_core import FloatArray
from heat_method_core import IntArray

try:
    import igl  # type: ignore
except ImportError:  # pragma: no cover - legacy fallback remains available
    igl = None


def compute_vertex_normals(V: FloatArray, F: IntArray, *, tolerance: float = 1e-12) -> FloatArray:
    """Compute area-weighted vertex normals for a triangle mesh."""
    vertices = np.asarray(V, dtype=np.float64)
    faces = np.asarray(F, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("V must have shape (n, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("F must have shape (m, 3).")

    normals = np.zeros_like(vertices)
    tri_vertices = vertices[faces]
    face_normals = np.cross(
        tri_vertices[:, 1] - tri_vertices[:, 0],
        tri_vertices[:, 2] - tri_vertices[:, 0],
    )
    for local_idx in range(3):
        np.add.at(normals, faces[:, local_idx], face_normals)

    norms = np.linalg.norm(normals, axis=1)
    # Area has squared-length units. An absolute area cutoff erased normals
    # on valid small facets near an isosurface/mesh vertex.
    normals = np.divide(normals,norms[:,None],out=np.zeros_like(normals),where=norms[:,None]>0)
    return normals


def _closest_point_on_triangle(
    point: FloatArray,
    triangle: FloatArray,
    *,
    tolerance: float = 1e-12,
) -> tuple[FloatArray, FloatArray]:
    """
    Return the closest point on a triangle and its barycentric coordinates.

    Based on the region tests from "Real-Time Collision Detection".
    """
    a, b, c = triangle
    ab = b - a
    ac = c - a
    ap = point - a

    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, np.array([1.0, 0.0, 0.0], dtype=np.float64)

    bp = point - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, np.array([0.0, 1.0, 0.0], dtype=np.float64)

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / max(d1 - d3, tolerance)
        closest = a + v * ab
        return closest, np.array([1.0 - v, v, 0.0], dtype=np.float64)

    cp = point - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c, np.array([0.0, 0.0, 1.0], dtype=np.float64)

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / max(d2 - d6, tolerance)
        closest = a + w * ac
        return closest, np.array([1.0 - w, 0.0, w], dtype=np.float64)

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        edge = c - b
        w = (d4 - d3) / max((d4 - d3) + (d5 - d6), tolerance)
        closest = b + w * edge
        return closest, np.array([0.0, 1.0 - w, w], dtype=np.float64)

    denom = max(va + vb + vc, tolerance)
    v = vb / denom
    w = vc / denom
    u = 1.0 - v - w
    closest = u * a + v * b + w * c
    return closest, np.array([u, v, w], dtype=np.float64)


def compute_path_normals(
    V: FloatArray,
    F: IntArray,
    path_points: FloatArray,
    *,
    tolerance: float = 1e-12,
    candidate_triangles: int = 8,
) -> FloatArray:
    """Compute interpolated normals at the exact closest mesh triangles."""
    vertices = np.asarray(V, dtype=np.float64)
    faces = np.asarray(F, dtype=np.int64)
    path = np.asarray(path_points, dtype=np.float64)

    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError("path_points must have shape (k, 3).")
    if candidate_triangles <= 0:
        raise ValueError("candidate_triangles must be positive.")

    tri=vertices[faces];ab0=tri[:,1]-tri[:,0];ac0=tri[:,2]-tri[:,0]
    area2=np.linalg.norm(np.cross(ab0,ac0),axis=1)
    scale2=np.maximum(np.sum(ab0*ab0,axis=1),np.sum(ac0*ac0,axis=1))
    nondegenerate=area2>32*np.finfo(float).eps*scale2
    faces=faces[nondegenerate]
    if not len(faces):raise ValueError('Path normals require a nondegenerate surface triangle.')

    vertex_normals = compute_vertex_normals(vertices, faces, tolerance=tolerance)
    tri_vertices = vertices[faces]
    tri_vertex_normals = vertex_normals[faces]

    if igl is not None and path.shape[0] > 0:
        # libigl performs the closest-triangle search for the whole path in
        # native code. The returned face makes edge points valid as well: one
        # barycentric coordinate is zero for a point on that face's edge.
        _, face_indices, closest_points = igl.point_mesh_squared_distance(path, vertices, faces)
        face_indices = np.asarray(face_indices, dtype=np.int64).reshape(-1)
        closest = np.asarray(closest_points, dtype=np.float64)
        selected_triangles = tri_vertices[face_indices]
        a = selected_triangles[:, 0]
        ab = selected_triangles[:, 1] - a
        ac = selected_triangles[:, 2] - a
        ap = closest - a
        dot_ab_ab = np.einsum("ij,ij->i", ab, ab)
        dot_ab_ac = np.einsum("ij,ij->i", ab, ac)
        dot_ac_ac = np.einsum("ij,ij->i", ac, ac)
        dot_ap_ab = np.einsum("ij,ij->i", ap, ab)
        dot_ap_ac = np.einsum("ij,ij->i", ap, ac)
        denominator = dot_ab_ab * dot_ac_ac - dot_ab_ac * dot_ab_ac
        valid_denominator=np.abs(denominator)>32*np.finfo(float).eps*np.maximum(dot_ab_ab*dot_ac_ac,np.finfo(float).tiny)
        safe_denominator = np.where(valid_denominator, denominator, 1.0)
        beta = (dot_ac_ac * dot_ap_ab - dot_ab_ac * dot_ap_ac) / safe_denominator
        gamma = (dot_ab_ab * dot_ap_ac - dot_ab_ac * dot_ap_ab) / safe_denominator
        barycentric = np.column_stack((1.0 - beta - gamma, beta, gamma))
        barycentric[~valid_denominator] = np.array((1.0, 0.0, 0.0))

        selected_normals = tri_vertex_normals[face_indices]
        path_normals = np.einsum("ij,ijk->ik", barycentric, selected_normals)
        normal_lengths = np.linalg.norm(path_normals, axis=1)
        valid = normal_lengths > tolerance
        if not np.all(valid):
            # Degenerate normal interpolation is rare; use the selected face
            # normal rather than failing an otherwise valid toolpath.
            face_normals = np.cross(ab, ac)
            face_lengths = np.linalg.norm(face_normals, axis=1)
            valid_faces = face_lengths > 0
            face_normals[valid_faces] /= face_lengths[valid_faces, None]
            path_normals[~valid] = face_normals[~valid]
            normal_lengths = np.linalg.norm(path_normals, axis=1)
            valid = normal_lengths > tolerance
        if not np.all(valid):
            raise RuntimeError("Failed to compute a normal for one or more path points.")
        return path_normals / normal_lengths[:, None]

    # Compatibility fallback for environments without libigl.
    tri_centers = tri_vertices.mean(axis=1)
    center_tree = cKDTree(tri_centers)

    k = min(candidate_triangles, tri_vertices.shape[0])

    path_normals = np.zeros_like(path)
    for idx, point in enumerate(path):
        best_distance = np.inf
        best_normal = None
        _, candidate_indices = center_tree.query(point, k=k)
        candidate_indices = np.atleast_1d(candidate_indices).astype(np.int64)

        for tri_idx in candidate_indices:
            triangle = tri_vertices[tri_idx]
            triangle_normals = tri_vertex_normals[tri_idx]
            closest, bary = _closest_point_on_triangle(point, triangle, tolerance=tolerance)
            distance = float(np.linalg.norm(point - closest))
            if distance < best_distance:
                interpolated = (
                    bary[0] * triangle_normals[0]
                    + bary[1] * triangle_normals[1]
                    + bary[2] * triangle_normals[2]
                )
                normal_norm = np.linalg.norm(interpolated)
                if normal_norm > tolerance:
                    best_distance = distance
                    best_normal = interpolated / normal_norm

        if best_normal is None:
            raise RuntimeError("Failed to compute a normal for one or more path points.")
        path_normals[idx] = best_normal

    return path_normals


def compute_strand_normals(V, F, strands, *, tolerance=1e-12, candidate_triangles=8):
    """Amortize the same exact normal query over independent path strands.

    Mesh preparation and closest-triangle policy are unchanged. The returned
    groups retain their original order, including empty strands; no path segment
    is inserted between groups.
    """
    paths = [np.asarray(p, dtype=np.float64) for p in strands]
    if any(p.ndim != 2 or p.shape[1] != 3 for p in paths):
        raise ValueError('Each strand must have shape (k, 3).')
    if candidate_triangles <= 0:
        raise ValueError('candidate_triangles must be positive.')
    lengths = [len(p) for p in paths]
    if not any(lengths):
        return [np.empty((0, 3), dtype=np.float64) for p in paths]
    normals = compute_path_normals(V, F, np.vstack(paths), tolerance=tolerance,
        candidate_triangles=candidate_triangles)
    return np.split(normals, np.cumsum(lengths)[:-1])


def path_with_normals(
    V: FloatArray,
    F: IntArray,
    path_points: FloatArray,
    *,
    tolerance: float = 1e-12,
    candidate_triangles: int = 8,
) -> FloatArray:
    """Return rows of x y z nx ny nz for an ordered path."""
    points = np.asarray(path_points, dtype=np.float64)
    normals = compute_path_normals(
        V,
        F,
        points,
        tolerance=tolerance,
        candidate_triangles=candidate_triangles,
    )
    return np.hstack((points, normals))


def save_path_with_normals(
    path: str,
    V: FloatArray,
    F: IntArray,
    path_points: FloatArray,
    *,
    tolerance: float = 1e-12,
    candidate_triangles: int = 8,
) -> None:
    """Save an ordered path with normals as rows: x y z nx ny nz."""
    output = path_with_normals(
        V,
        F,
        path_points,
        tolerance=tolerance,
        candidate_triangles=candidate_triangles,
    )
    np.savetxt(path, output, fmt="%.10f")


def save_numeric_array(path, values, *, return_serialized: bool = False,write_file: bool=True):
    """Persist the usual decimal text and optionally return its exact parsed values.

    Parsing the same text in memory preserves the existing file-rounding boundary
    without reopening the saved artifact. Default callers keep the original writer.
    """
    if not write_file and not return_serialized:
        raise ValueError('Memory-only decimal serialization requires return_serialized=True.')
    if not return_serialized:
        np.savetxt(path, values, fmt="%.10f")
        return None
    stream = StringIO()
    np.savetxt(stream, values, fmt="%.10f")
    text = stream.getvalue()
    if write_file:Path(path).write_text(text, encoding="utf-8")
    if not values.size:
        return np.empty(values.shape, dtype=np.float64)
    return np.loadtxt(StringIO(text), ndmin=values.ndim)


def save_oriented_path(
    path: str,
    path_points: FloatArray,
    path_normals: FloatArray,
    *,
    tolerance: float = 1e-12,
    return_serialized: bool = False,
    write_file: bool=True,
) -> FloatArray | None:
    """Save an already-oriented path as rows: x y z nx ny nz."""
    points = np.asarray(path_points, dtype=np.float64)
    normals = np.asarray(path_normals, dtype=np.float64)
    if points.shape != normals.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("path_points and path_normals must both have shape (k, 3).")

    normal_norms = np.linalg.norm(normals, axis=1)
    safe_norms = np.maximum(normal_norms, tolerance)
    output = np.hstack((points, normals / safe_norms[:, None]))
    return save_numeric_array(path, output, return_serialized=return_serialized,write_file=write_file)
