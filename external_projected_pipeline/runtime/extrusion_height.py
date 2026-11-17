"""
Utilities for estimating extrusion support height from ordered path files.

The current implementation focuses on 3-axis printing with a fixed machine
axis. For each path point, it looks for the highest supporting point on recent
previous meshes along that axis and returns the resulting layer-height gap.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree

from heat_method_core import FloatArray
from heat_method_core import IntArray
from mesh_distance_workflow import load_obj_mesh

try:
    import igl  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    igl = None


def axis_to_index(axis: str) -> int:
    axis_map = {"x": 0, "y": 1, "z": 2}
    try:
        return axis_map[axis.lower()]
    except KeyError as exc:
        raise ValueError("axis must be one of: x, y, z") from exc


def projected_dimensions(axis_index: int) -> tuple[int, int]:
    dims = [0, 1, 2]
    dims.remove(axis_index)
    return dims[0], dims[1]


def apply_axis_lift(points: FloatArray, *, axis: str, amount: float) -> FloatArray:
    vertices = np.asarray(points, dtype=np.float64).copy()
    if abs(float(amount)) <= 0.0:
        return vertices
    axis_index = axis_to_index(axis)
    vertices[:, axis_index] += float(amount)
    return vertices


@dataclass(frozen=True, slots=True)
class LayerKey:
    main: int
    sub: int

    @property
    def stem(self) -> str:
        return f"{self.main}_{self.sub}"


@dataclass(slots=True)
class ReferenceMesh3Axis:
    key: LayerKey
    path: Path
    axis_index: int
    vertices: FloatArray
    faces: IntArray
    vertex_projected: FloatArray
    vertex_axis_values: FloatArray
    face_projected: FloatArray
    face_axis_values: FloatArray
    face_normals: FloatArray
    vertex_tree: cKDTree
    vertex_incident_faces: tuple[np.ndarray, ...]
    vertex_face_offsets: IntArray | None = None
    vertex_face_indices: IntArray | None = None

    def __post_init__(self):
        # Compact incidence permits batched point/triangle queries without
        # building Python lists and deduplicating faces for every waypoint.
        counts = np.fromiter((len(faces) for faces in self.vertex_incident_faces), dtype=np.int64)
        self.vertex_face_offsets = np.r_[0, np.cumsum(counts)]
        self.vertex_face_indices = np.concatenate(self.vertex_incident_faces)


@dataclass(slots=True)
class ReferenceMesh5Axis:
    key: LayerKey
    path: Path
    vertices: FloatArray
    faces: IntArray


@dataclass(slots=True)
class FiveAxisAlignmentDiagnostics:
    support_points: FloatArray
    alignment: FloatArray
    deviation_degrees: FloatArray
    major_deviation_mask: np.ndarray
    major_deviation_count: int


def load_path_rows(path_file: Path) -> FloatArray:
    if path_file.stat().st_size == 0:
        return np.empty((0, 6), dtype=np.float64)
    data = np.loadtxt(path_file, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError(f"Path file must contain at least x y z nx ny nz columns: {path_file}")
    if not np.isfinite(data[:,:6]).all():raise ValueError(f'Path coordinates/normals must be finite: {path_file}')
    return np.asarray(data[:, 0:6], dtype=np.float64)


def save_path_with_heights(path_file: Path, path_rows: FloatArray, heights: FloatArray) -> None:
    rows = np.asarray(path_rows, dtype=np.float64)
    height_values = np.asarray(heights, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] < 6:
        raise ValueError("path_rows must have shape (n, 6) or wider.")
    if height_values.ndim != 1 or height_values.shape[0] != rows.shape[0]:
        raise ValueError("heights must be a 1D array matching the number of path rows.")
    if not np.isfinite(rows[:,:6]).all() or not np.isfinite(height_values).all() or np.any(height_values<0):
        raise ValueError('Path coordinates/normals and extrusion heights must be finite, with nonnegative heights.')
    output = np.hstack((rows[:, 0:6], height_values[:, None]))
    np.savetxt(path_file, output, fmt="%.10f")


def build_vertex_incident_faces(num_vertices: int, faces: IntArray) -> tuple[np.ndarray, ...]:
    adjacency: list[list[int]] = [[] for _ in range(num_vertices)]
    for face_idx, face in enumerate(np.asarray(faces, dtype=np.int64)):
        adjacency[int(face[0])].append(face_idx)
        adjacency[int(face[1])].append(face_idx)
        adjacency[int(face[2])].append(face_idx)
    return tuple(np.asarray(items, dtype=np.int64) for items in adjacency)


def load_reference_mesh_3axis(
    mesh_path: Path,
    key: LayerKey,
    *,
    axis: str = "y",
    coordinate_lift: float = 0.0,
    coordinate_lift_axis: str | None = None,
) -> ReferenceMesh3Axis:
    V, F = load_obj_mesh(str(mesh_path))
    lift_axis = axis if coordinate_lift_axis is None else str(coordinate_lift_axis)
    if abs(float(coordinate_lift)) > 0.0:
        V = apply_axis_lift(V, axis=lift_axis, amount=float(coordinate_lift))
    return prepare_reference_mesh_3axis(V, F, key, path=mesh_path, axis=axis)


def prepare_reference_mesh_3axis(V: FloatArray, F: IntArray, key: LayerKey, *,
                               path: Path, axis: str = "y") -> ReferenceMesh3Axis:
    """Prepare already-loaded (and already-lifted) geometry without reparsing it."""
    axis_index = axis_to_index(axis)
    dim0, dim1 = projected_dimensions(axis_index)
    faces = np.asarray(F, dtype=np.int64)
    face_vertices = np.asarray(V[faces], dtype=np.float64)
    raw_face_normals = np.cross(
        face_vertices[:, 1, :] - face_vertices[:, 0, :],
        face_vertices[:, 2, :] - face_vertices[:, 0, :],
    )
    face_normal_norms = np.linalg.norm(raw_face_normals, axis=1, keepdims=True)
    safe_face_normal_norms = np.where(face_normal_norms > 1e-12, face_normal_norms, 1.0)
    face_normals = raw_face_normals / safe_face_normal_norms
    vertex_projected = np.asarray(V[:, [dim0, dim1]], dtype=np.float64)
    vertex_axis_values = np.asarray(V[:, axis_index], dtype=np.float64)
    return ReferenceMesh3Axis(
        key=key,
        path=path,
        axis_index=axis_index,
        vertices=np.asarray(V, dtype=np.float64),
        faces=faces,
        vertex_projected=vertex_projected,
        vertex_axis_values=vertex_axis_values,
        face_projected=np.asarray(vertex_projected[faces], dtype=np.float64),
        face_axis_values=np.asarray(vertex_axis_values[faces], dtype=np.float64),
        face_normals=np.asarray(face_normals, dtype=np.float64),
        vertex_tree=cKDTree(vertex_projected),
        vertex_incident_faces=build_vertex_incident_faces(V.shape[0], F),
    )


def _interpolate_axis_value_on_triangle(
    query_projected: FloatArray,
    triangle_projected: FloatArray,
    triangle_axis_values: FloatArray,
    *,
    tolerance: float = 1e-9,
) -> float | None:
    a = triangle_projected[0]
    b = triangle_projected[1]
    c = triangle_projected[2]
    v0 = b - a
    v1 = c - a
    v2 = query_projected - a

    denom = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(denom) <= tolerance:
        return None

    inv_denom = 1.0 / denom
    beta = (v2[0] * v1[1] - v1[0] * v2[1]) * inv_denom
    gamma = (v0[0] * v2[1] - v2[0] * v0[1]) * inv_denom
    alpha = 1.0 - beta - gamma

    if (
        alpha < -tolerance
        or beta < -tolerance
        or gamma < -tolerance
        or alpha > 1.0 + tolerance
        or beta > 1.0 + tolerance
        or gamma > 1.0 + tolerance
    ):
        return None

    bary = np.array([alpha, beta, gamma], dtype=np.float64)
    return float(np.dot(bary, triangle_axis_values))


def _query_reference_mesh_support_height(
    point: FloatArray,
    reference_mesh: ReferenceMesh3Axis,
    *,
    k_nearest_vertices: int,
    tolerance: float = 1e-9,
) -> float | None:
    axis_index = reference_mesh.axis_index
    dim0, dim1 = projected_dimensions(axis_index)
    projected_point = np.asarray(point[[dim0, dim1]], dtype=np.float64)
    point_axis_value = float(point[axis_index])

    k = min(max(k_nearest_vertices, 1), reference_mesh.vertices.shape[0])
    _, nearest_vertex_indices = reference_mesh.vertex_tree.query(projected_point, k=k)
    nearest_vertex_indices = np.atleast_1d(nearest_vertex_indices).astype(np.int64)

    candidate_faces: set[int] = set()
    for vertex_idx in nearest_vertex_indices:
        candidate_faces.update(reference_mesh.vertex_incident_faces[int(vertex_idx)].tolist())

    best_support: float | None = None
    for face_idx in candidate_faces:
        face = reference_mesh.faces[face_idx]
        triangle_projected = reference_mesh.vertex_projected[face]
        triangle_axis_values = reference_mesh.vertex_axis_values[face]
        support_value = _interpolate_axis_value_on_triangle(
            projected_point,
            triangle_projected,
            triangle_axis_values,
            tolerance=tolerance,
        )
        if support_value is None:
            continue
        if support_value > point_axis_value + tolerance:
            continue
        if best_support is None or support_value > best_support:
            best_support = support_value

    return best_support


def _query_reference_mesh_support_height_and_normal(
    point: FloatArray,
    reference_mesh: ReferenceMesh3Axis,
    *,
    k_nearest_vertices: int,
    tolerance: float = 1e-9,
) -> tuple[float | None, np.ndarray | None]:
    axis_index = reference_mesh.axis_index
    dim0, dim1 = projected_dimensions(axis_index)
    projected_point = np.asarray(point[[dim0, dim1]], dtype=np.float64)
    point_axis_value = float(point[axis_index])

    k = min(max(k_nearest_vertices, 1), reference_mesh.vertices.shape[0])
    _, nearest_vertex_indices = reference_mesh.vertex_tree.query(projected_point, k=k)
    nearest_vertex_indices = np.atleast_1d(nearest_vertex_indices).astype(np.int64)

    candidate_faces: set[int] = set()
    for vertex_idx in nearest_vertex_indices:
        candidate_faces.update(reference_mesh.vertex_incident_faces[int(vertex_idx)].tolist())

    best_support: float | None = None
    best_normal: np.ndarray | None = None
    for face_idx in candidate_faces:
        face = reference_mesh.faces[face_idx]
        triangle_projected = reference_mesh.vertex_projected[face]
        triangle_axis_values = reference_mesh.vertex_axis_values[face]
        support_value = _interpolate_axis_value_on_triangle(
            projected_point,
            triangle_projected,
            triangle_axis_values,
            tolerance=tolerance,
        )
        if support_value is None:
            continue
        if support_value > point_axis_value + tolerance:
            continue
        if best_support is None or support_value > best_support:
            best_support = float(support_value)
            best_normal = np.asarray(reference_mesh.face_normals[face_idx], dtype=np.float64).copy()

    return best_support, best_normal


def _query_reference_mesh_support_heights(
    points: FloatArray,
    reference_mesh: ReferenceMesh3Axis,
    *,
    k_nearest_vertices: int,
    tolerance: float = 1e-9,
    tree_query_workers: int = 1,
) -> FloatArray:
    axis_index = reference_mesh.axis_index
    dim0, dim1 = projected_dimensions(axis_index)
    projected_points = np.asarray(points[:, [dim0, dim1]], dtype=np.float64)
    point_axis_values = np.asarray(points[:, axis_index], dtype=np.float64)

    k = min(max(k_nearest_vertices, 1), reference_mesh.vertices.shape[0])
    _, nearest_vertex_indices = reference_mesh.vertex_tree.query(
        projected_points,
        k=k,
        workers=tree_query_workers,
    )
    nearest_vertex_indices = np.asarray(nearest_vertex_indices, dtype=np.int64)
    if nearest_vertex_indices.ndim == 1:
        nearest_vertex_indices = nearest_vertex_indices[:, None]

    supports = np.full(projected_points.shape[0], np.nan, dtype=np.float64)
    face_projected = np.asarray(reference_mesh.face_projected, dtype=np.float64)
    face_axis_values = np.asarray(reference_mesh.face_axis_values, dtype=np.float64)

    offsets = reference_mesh.vertex_face_offsets
    incidence = reference_mesh.vertex_face_indices
    for start in range(0, len(projected_points), 512):
        stop = min(start + 512, len(projected_points))
        vertices = nearest_vertex_indices[start:stop].reshape(-1)
        counts = offsets[vertices + 1] - offsets[vertices]
        total = int(counts.sum())
        if not total:
            continue
        # Duplicate incident faces are harmless for a maximum reduction. Keep
        # the original nearest-vertex candidate policy, including unused vertices.
        rows = np.repeat(np.arange(stop - start), counts.reshape(stop - start, -1).sum(axis=1))
        local = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        candidate_faces = incidence[np.repeat(offsets[vertices], counts) + local]

        triangles_projected = face_projected[candidate_faces]
        triangles_axis_values = face_axis_values[candidate_faces]

        a = triangles_projected[:, 0, :]
        b = triangles_projected[:, 1, :]
        c = triangles_projected[:, 2, :]
        v0 = b - a
        v1 = c - a
        v2 = projected_points[start:stop][rows] - a

        denom = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        valid = np.abs(denom) > tolerance
        if not np.any(valid):
            continue

        inv_denom = np.zeros_like(denom)
        inv_denom[valid] = 1.0 / denom[valid]
        beta = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) * inv_denom
        gamma = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) * inv_denom
        alpha = 1.0 - beta - gamma

        valid &= alpha >= -tolerance
        valid &= beta >= -tolerance
        valid &= gamma >= -tolerance
        valid &= alpha <= 1.0 + tolerance
        valid &= beta <= 1.0 + tolerance
        valid &= gamma <= 1.0 + tolerance
        if not np.any(valid):
            continue

        bary = np.stack((alpha, beta, gamma), axis=1)
        support_values = np.einsum("ij,ij->i", bary, triangles_axis_values)
        valid &= support_values <= point_axis_values[start:stop][rows] + tolerance
        if not np.any(valid):
            continue

        best = np.full(stop - start, -np.inf, dtype=np.float64)
        np.maximum.at(best, rows[valid], support_values[valid])
        found = best > -np.inf
        supports[start:stop][found] = best[found]

    return supports


def load_reference_mesh_5axis(
    mesh_path: Path,
    key: LayerKey,
    *,
    coordinate_lift: float = 0.0,
    coordinate_lift_axis: str = "y",
) -> ReferenceMesh5Axis:
    V, F = load_obj_mesh(str(mesh_path))
    if abs(float(coordinate_lift)) > 0.0:
        V = apply_axis_lift(V, axis=coordinate_lift_axis, amount=float(coordinate_lift))
    return ReferenceMesh5Axis(
        key=key,
        path=mesh_path,
        vertices=np.asarray(V, dtype=np.float64),
        faces=np.asarray(F, dtype=np.int64),
    )


def compute_5axis_alignment_diagnostics(
    path_points: FloatArray,
    path_normals: FloatArray,
    support_points: FloatArray,
    *,
    major_alignment_threshold: float = 0.5,
    tolerance: float = 1e-12,
) -> FiveAxisAlignmentDiagnostics:
    points = np.asarray(path_points, dtype=np.float64)
    normals = np.asarray(path_normals, dtype=np.float64)
    supports = np.asarray(support_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("path_points must have shape (n, 3).")
    if normals.shape != points.shape:
        raise ValueError("path_normals must have the same shape as path_points.")
    if supports.shape != points.shape:
        raise ValueError("support_points must have the same shape as path_points.")
    if not (-1.0 <= major_alignment_threshold <= 1.0):
        raise ValueError("major_alignment_threshold must lie in [-1, 1].")

    offset = supports - points
    offset_norm = np.linalg.norm(offset, axis=1)
    normal_norm = np.linalg.norm(normals, axis=1)
    valid = (offset_norm > tolerance) & (normal_norm > tolerance)

    alignment = np.full(points.shape[0], np.nan, dtype=np.float64)
    if np.any(valid):
        offset_dir = offset[valid] / offset_norm[valid][:, None]
        minus_normal_dir = -normals[valid] / normal_norm[valid][:, None]
        alignment[valid] = np.einsum("ij,ij->i", offset_dir, minus_normal_dir)
        alignment[valid] = np.clip(alignment[valid], -1.0, 1.0)

    deviation_degrees = np.full(points.shape[0], np.nan, dtype=np.float64)
    valid_alignment = ~np.isnan(alignment)
    if np.any(valid_alignment):
        deviation_degrees[valid_alignment] = np.degrees(np.arccos(alignment[valid_alignment]))

    major_deviation_mask = valid_alignment & (alignment < major_alignment_threshold)
    return FiveAxisAlignmentDiagnostics(
        support_points=supports,
        alignment=alignment,
        deviation_degrees=deviation_degrees,
        major_deviation_mask=major_deviation_mask,
        major_deviation_count=int(np.count_nonzero(major_deviation_mask)),
    )


def compute_point_heights_3axis(
    path_points: FloatArray,
    reference_meshes: Iterable[ReferenceMesh3Axis],
    *,
    axis: str = "y",
    platform_value: float = 0.0,
    k_nearest_vertices: int = 12,
    max_height: float | None = None,
    tolerance: float = 1e-9,
    num_workers: int = 1,
    tree_query_workers: int = 1,
    return_platform_mask: bool = False,
) -> FloatArray | tuple[FloatArray, np.ndarray]:
    points = np.asarray(path_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("path_points must have shape (n, 3).")
    if k_nearest_vertices <= 0:
        raise ValueError("k_nearest_vertices must be positive.")
    if max_height is not None and max_height < 0.0:
        raise ValueError("max_height must be nonnegative when provided.")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive.")
    if tree_query_workers == 0 or tree_query_workers < -1:
        raise ValueError("tree_query_workers must be positive or -1.")

    axis_index = axis_to_index(axis)
    references = tuple(reference_meshes)

    def compute_chunk(chunk_points: FloatArray) -> tuple[FloatArray, np.ndarray]:
        chunk_points = np.asarray(chunk_points, dtype=np.float64)
        point_axis_values = np.asarray(chunk_points[:, axis_index], dtype=np.float64)
        # TODO: Outer-edge points can currently miss the intended lower-layer
        # support and fall back to the platform plane instead. This is most
        # likely near projected boundaries where the nearest-vertex candidate
        # set does not include the true supporting triangle. Revisit this when
        # improving edge robustness for the 3-axis height query.
        best_support = np.full(chunk_points.shape[0], np.nan, dtype=np.float64)
        for reference in references:
            support_values = _query_reference_mesh_support_heights(
                chunk_points,
                reference,
                k_nearest_vertices=k_nearest_vertices,
                tolerance=tolerance,
                tree_query_workers=tree_query_workers,
            )
            valid_mask = ~np.isnan(support_values)
            if not np.any(valid_mask):
                continue
            update_mask = valid_mask & (np.isnan(best_support) | (support_values > best_support))
            if np.any(update_mask):
                best_support[update_mask] = support_values[update_mask]

        platform_mask = np.isnan(best_support)
        best_support = np.where(platform_mask, platform_value, best_support)
        heights = np.maximum(0.0, point_axis_values - best_support)
        if max_height is not None:
            heights = np.asarray(heights, dtype=np.float64).copy()
            heights[platform_mask] = np.minimum(heights[platform_mask], float(max_height))
        return heights, platform_mask

    if num_workers == 1 or points.shape[0] <= 1:
        heights, platform_mask = compute_chunk(points)
        if return_platform_mask:
            return heights, platform_mask
        return heights

    chunks = np.array_split(points, min(num_workers, points.shape[0]))
    height_outputs: list[np.ndarray] = []
    mask_outputs: list[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(compute_chunk, chunk)
            for chunk in chunks
            if chunk.shape[0] > 0
        ]
        for future in futures:
            heights, platform_mask = future.result()
            height_outputs.append(heights)
            mask_outputs.append(platform_mask)
    heights = np.concatenate(height_outputs, axis=0) if height_outputs else np.empty((0,), dtype=np.float64)
    platform_mask = np.concatenate(mask_outputs, axis=0) if mask_outputs else np.empty((0,), dtype=bool)
    if return_platform_mask:
        return heights, platform_mask
    return heights


def _estimate_path_tangents(points: FloatArray, *, tolerance: float = 1e-12) -> FloatArray:
    pts = np.asarray(points, dtype=np.float64)
    tangents = np.zeros_like(pts, dtype=np.float64)
    if pts.shape[0] <= 1:
        return tangents
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    if pts.shape[0] > 2:
        tangents[1:-1] = pts[2:] - pts[:-2]
    tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    valid = tangent_norms[:, 0] > tolerance
    tangents[valid] = tangents[valid] / tangent_norms[valid]
    return tangents


def compute_cross_sectional_height_scale_3axis(
    path_points: FloatArray,
    reference_meshes: Iterable[ReferenceMesh3Axis],
    *,
    axis: str = "y",
    k_nearest_vertices: int = 12,
    min_scale: float = 0.0,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(path_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("path_points must have shape (n, 3).")
    if not (0.0 <= min_scale <= 1.0):
        raise ValueError("min_scale must lie in [0, 1].")

    tangents = _estimate_path_tangents(points, tolerance=tolerance)
    tool_axis = np.zeros((3,), dtype=np.float64)
    tool_axis[axis_to_index(axis)] = 1.0

    scales = np.ones((points.shape[0],), dtype=np.float64)
    platform_mask = np.ones((points.shape[0],), dtype=bool)
    references = tuple(reference_meshes)

    for point_idx, point in enumerate(points):
        tangent = tangents[point_idx]
        tangent_norm = float(np.linalg.norm(tangent))
        best_support: float | None = None
        best_normal: np.ndarray | None = None
        for reference in references:
            support_value, support_normal = _query_reference_mesh_support_height_and_normal(
                point,
                reference,
                k_nearest_vertices=k_nearest_vertices,
                tolerance=tolerance,
            )
            if support_value is None:
                continue
            if best_support is None or support_value > best_support:
                best_support = support_value
                best_normal = support_normal

        if best_support is None or best_normal is None:
            continue

        platform_mask[point_idx] = False
        normal = np.asarray(best_normal, dtype=np.float64)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= tolerance:
            continue
        normal = normal / normal_norm
        if tangent_norm > tolerance:
            tangent = tangent / tangent_norm
            normal_perp = normal - float(np.dot(normal, tangent)) * tangent
        else:
            normal_perp = normal.copy()
        normal_perp_norm = float(np.linalg.norm(normal_perp))
        if normal_perp_norm <= tolerance:
            correction = abs(float(np.dot(normal, tool_axis)))
        else:
            normal_perp = normal_perp / normal_perp_norm
            correction = abs(float(np.dot(normal_perp, tool_axis)))
        scales[point_idx] = float(np.clip(correction, min_scale, 1.0))

    return scales, platform_mask


def compute_point_heights_5axis(
    path_points: FloatArray,
    path_normals: FloatArray,
    reference_meshes: Iterable[ReferenceMesh5Axis],
    *,
    platform_axis: str = "y",
    platform_value: float = 0.0,
    major_alignment_threshold: float = 0.5,
    return_diagnostics: bool = False,
    tolerance: float = 1e-12,
) -> FloatArray | tuple[FloatArray, FiveAxisAlignmentDiagnostics]:
    # Planned first-pass 5-axis assumption:
    # find the nearest point on the previous-layer mesh surfaces in full 3D,
    # define height as the Euclidean distance to that support point, and
    # temporarily assume that this nearest support lies approximately along
    # -normal. This function now includes an optional deviation check.
    if igl is None:
        raise ImportError(
            "5-axis height estimation requires the optional `igl` package "
            "for point_mesh_squared_distance queries."
        )

    points = np.asarray(path_points, dtype=np.float64)
    normals = np.asarray(path_normals, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("path_points must have shape (n, 3).")
    if normals.ndim != 2 or normals.shape != points.shape:
        raise ValueError("path_normals must have the same shape as path_points.")
    platform_axis_index = axis_to_index(platform_axis)

    references = tuple(reference_meshes)
    if not references:
        support_points = np.asarray(points, dtype=np.float64).copy()
        support_points[:, platform_axis_index] = platform_value
        heights = np.linalg.norm(points - support_points, axis=1)
        empty_heights = np.empty((points.shape[0],), dtype=np.float64)
        if return_diagnostics:
            diagnostics = compute_5axis_alignment_diagnostics(
                points,
                normals,
                support_points,
                major_alignment_threshold=major_alignment_threshold,
                tolerance=tolerance,
            )
            return heights, diagnostics
        return heights

    best_squared = np.full(points.shape[0], np.inf, dtype=np.float64)
    best_points = np.full_like(points, np.nan, dtype=np.float64)

    for reference in references:
        squared_distance, _, closest_points = igl.point_mesh_squared_distance(
            points,
            reference.vertices,
            reference.faces,
        )
        squared_distance = np.asarray(squared_distance, dtype=np.float64).reshape(-1)
        closest_points = np.asarray(closest_points, dtype=np.float64)
        if squared_distance.shape[0] != points.shape[0]:
            raise RuntimeError("Unexpected libigl result shape for point-mesh distance query.")
        if closest_points.shape != points.shape:
            raise RuntimeError("Unexpected libigl closest-point shape for point-mesh distance query.")
        update_mask = squared_distance < best_squared
        if np.any(update_mask):
            best_squared[update_mask] = squared_distance[update_mask]
            best_points[update_mask] = closest_points[update_mask]

    inf_mask = np.isinf(best_squared)
    safe_squared = np.maximum(best_squared, tolerance)
    heights = np.sqrt(safe_squared).astype(np.float64, copy=False)
    heights[inf_mask] = np.nan

    diagnostics = compute_5axis_alignment_diagnostics(
        points,
        normals,
        best_points,
        major_alignment_threshold=major_alignment_threshold,
        tolerance=tolerance,
    )
    if return_diagnostics:
        return heights, diagnostics
    return heights
