"""
Experimental stripe-field solver based on vector heat transport and scalar reconstruction.

Pipeline:
- build a plane/mesh intersection source curve
- choose a preferred world direction (x/y/z)
- project that direction to the source-vertex tangent planes
- transport those tangent vectors over the surface with MeshVectorHeatSolver
- enforce sign coherence against the projected preferred direction
- reconstruct a scalar field with zero Dirichlet constraints on source-support vertices
- extract iso-contours from the reconstructed scalar
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from contour_extraction import ContourLevel
from contour_extraction import _interpolate_edge_point
from contour_extraction import extract_isocontour_polylines
from contour_extraction import stitch_segments_into_polylines
from heat_method_core import FloatArray
from mesh_distance_workflow import load_obj_mesh
from mesh_distance_workflow import to_pyvista_polydata
from test_potpourri3d_signed_heat_plane import AXIS_TO_INDEX
from test_potpourri3d_signed_heat_plane import CurvePoint
from test_potpourri3d_signed_heat_plane import build_polyline_polydata
from test_potpourri3d_signed_heat_plane import intersect_mesh_with_axis_plane
from test_potpourri3d_signed_heat_plane import save_curves_obj
from test_potpourri3d_signed_heat_plane import stitch_segments_into_curves

try:
    import potpourri3d as pp3d  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pp3d = None

try:
    import pyvista as pv  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pv = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experimental stripe-field solver using MeshVectorHeatSolver and constrained scalar reconstruction.",
    )
    parser.add_argument("mesh_path", type=Path, help="Input triangle mesh.")
    parser.add_argument("output_dir", type=Path, help="Directory to write debug outputs.")
    parser.add_argument("--plane-axis", choices=("x", "y", "z"), default="x", help="Axis normal for the source plane.")
    parser.add_argument("--plane-value", type=float, default=None, help="Optional plane value override. Default uses midpoint extent.")
    parser.add_argument(
        "--constraint-mode",
        choices=("constrained", "unconstrained"),
        default="constrained",
        help="Scalar reconstruction mode. 'constrained' uses source Dirichlet values; 'unconstrained' solves the same Poisson system with only nullspace regularization and recenters to zero mean.",
    )
    parser.add_argument("--spacing", type=float, required=True, help="Contour spacing.")
    parser.add_argument("--max-level-fraction", type=float, default=1.0, help="Optional fraction of scalar range used to trim contour extraction.")
    parser.add_argument("--source-vertex-mode", choices=("intersected_faces",), default="intersected_faces", help="How to select source-support vertices from the source curve.")
    parser.add_argument("--preview", action="store_true", help="Preview the reconstructed scalar and contours with PyVista.")
    parser.add_argument(
        "--preview-field",
        choices=("scalar", "transported_alignment", "transported_magnitude", "preferred_alignment"),
        default="scalar",
        help="Vertex field used to color the mesh in preview mode.",
    )
    parser.add_argument("--show-mesh-edges", action="store_true", help="Show triangle edges in preview mode.")
    parser.add_argument("--show-vectors", action="store_true", help="Overlay source and propagated vectors in preview mode.")
    parser.add_argument("--show-constraints", action="store_true", help="Overlay constrained nodes colored by their Dirichlet values in preview mode.")
    parser.add_argument("--vector-scale", type=float, default=1.0, help="Display scale for preview vector arrows.")
    parser.add_argument("--tolerance", type=float, default=1e-8, help="Geometric tolerance.")
    return parser


def compute_midpoint_plane_value(V: FloatArray, axis_index: int) -> float:
    coords = np.asarray(V[:, axis_index], dtype=np.float64)
    return float(0.5 * (np.min(coords) + np.max(coords)))


def choose_plane_value(V: FloatArray, axis_index: int, explicit_value: float | None) -> float:
    return float(explicit_value) if explicit_value is not None else compute_midpoint_plane_value(V, axis_index)


def choose_world_direction(direction_name: str) -> FloatArray:
    axis_index = AXIS_TO_INDEX[direction_name]
    direction = np.zeros(3, dtype=np.float64)
    direction[axis_index] = 1.0
    return direction


def build_source_support_vertices(
    F: np.ndarray,
    segments: list[tuple[CurvePoint, CurvePoint]],
) -> np.ndarray:
    faces = np.asarray(F, dtype=np.int64)
    source_vertices: set[int] = set()
    for start, end in segments:
        face_index = int(start.face_index)
        if face_index != int(end.face_index):
            raise ValueError("Expected face-local source segments.")
        for vertex_index in faces[face_index]:
            source_vertices.add(int(vertex_index))
    if not source_vertices:
        raise RuntimeError("No source-support vertices were found from the source curve.")
    return np.asarray(sorted(source_vertices), dtype=np.int64)


def build_source_support_faces(
    segments: list[tuple[CurvePoint, CurvePoint]],
) -> np.ndarray:
    face_indices = sorted({int(start.face_index) for start, _ in segments})
    if not face_indices:
        raise RuntimeError("No source-support faces were found from the source curve.")
    return np.asarray(face_indices, dtype=np.int64)


def build_signed_plane_constraints(
    V: FloatArray,
    source_support_vertices: np.ndarray,
    *,
    plane_axis_index: int,
    plane_value: float,
) -> tuple[np.ndarray, FloatArray]:
    vertices = np.asarray(V, dtype=np.float64)
    support_vertices = np.asarray(source_support_vertices, dtype=np.int64)
    signed_plane_distance = vertices[support_vertices, plane_axis_index] - float(plane_value)
    return support_vertices, np.asarray(signed_plane_distance, dtype=np.float64)


def project_world_direction_to_tangent(
    direction_3d: FloatArray,
    frame_x: FloatArray,
    frame_y: FloatArray,
    frame_n: FloatArray,
    *,
    tolerance: float,
) -> tuple[FloatArray, FloatArray]:
    direction = np.asarray(direction_3d, dtype=np.float64)
    tangent_projection = direction[None, :] - np.einsum("ij,j->i", frame_n, direction)[:, None] * frame_n
    norms = np.linalg.norm(tangent_projection, axis=1)
    valid = norms > tolerance
    tangent_projection_unit = np.zeros_like(tangent_projection)
    tangent_projection_unit[valid] = tangent_projection[valid] / norms[valid, None]

    tangent_coords = np.zeros((frame_x.shape[0], 2), dtype=np.float64)
    tangent_coords[:, 0] = np.einsum("ij,ij->i", tangent_projection_unit, frame_x)
    tangent_coords[:, 1] = np.einsum("ij,ij->i", tangent_projection_unit, frame_y)
    return tangent_projection_unit, tangent_coords


def transport_preferred_source_vectors(
    V: FloatArray,
    F: np.ndarray,
    source_vertices: np.ndarray,
    preferred_direction_name: str,
    *,
    tolerance: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    if pp3d is None:
        raise ImportError("This script requires the optional `potpourri3d` package.")

    solver = pp3d.MeshVectorHeatSolver(V, F)
    frame_x, frame_y, frame_n = [np.asarray(arr, dtype=np.float64) for arr in solver.get_tangent_frames()]
    preferred_direction = choose_world_direction(preferred_direction_name)
    preferred_tangent_3d, preferred_tangent_2d = project_world_direction_to_tangent(
        preferred_direction,
        frame_x,
        frame_y,
        frame_n,
        tolerance=tolerance,
    )

    source_vectors_2d = preferred_tangent_2d[source_vertices]
    transported_2d = np.asarray(
        solver.transport_tangent_vectors(source_vertices.tolist(), source_vectors_2d),
        dtype=np.float64,
    )
    transported_3d = transported_2d[:, [0]] * frame_x + transported_2d[:, [1]] * frame_y
    return frame_x, frame_y, frame_n, preferred_tangent_3d, preferred_tangent_2d, transported_3d


def orient_and_normalize_transport_field(
    transported_3d: FloatArray,
    preferred_tangent_3d: FloatArray,
    *,
    tolerance: float,
) -> FloatArray:
    vectors = np.asarray(transported_3d, dtype=np.float64).copy()
    preferred = np.asarray(preferred_tangent_3d, dtype=np.float64)
    preferred_norm = np.linalg.norm(preferred, axis=1)
    valid_preferred = preferred_norm > tolerance

    alignment = np.einsum("ij,ij->i", vectors, preferred)
    flip_mask = valid_preferred & (alignment < 0.0)
    vectors[flip_mask] *= -1.0

    norms = np.linalg.norm(vectors, axis=1)
    weak_mask = norms <= tolerance
    vectors[weak_mask & valid_preferred] = preferred[weak_mask & valid_preferred]
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > tolerance
    unit_vectors = np.zeros_like(vectors)
    unit_vectors[valid] = vectors[valid] / norms[valid, None]
    return unit_vectors


def rotate_tangent_field_90(
    tangent_field: FloatArray,
    frame_n: FloatArray,
) -> FloatArray:
    return np.cross(np.asarray(frame_n, dtype=np.float64), np.asarray(tangent_field, dtype=np.float64))


def compute_face_gradients_from_vertex_scalar(
    F: np.ndarray,
    grad_bary: FloatArray,
    scalar_values: FloatArray,
) -> FloatArray:
    faces = np.asarray(F, dtype=np.int64)
    values = np.asarray(scalar_values, dtype=np.float64)
    return np.einsum("fvc,fv->fc", grad_bary, values[faces])


def build_face_gradient_operator(
    F: np.ndarray,
    grad_bary: FloatArray,
    *,
    n_vertices: int,
) -> scipy.sparse.csr_matrix:
    faces = np.asarray(F, dtype=np.int64)
    gradients = np.asarray(grad_bary, dtype=np.float64)
    n_faces = faces.shape[0]

    row_ids: list[np.ndarray] = []
    col_ids: list[np.ndarray] = []
    data: list[np.ndarray] = []
    base_rows = np.arange(n_faces, dtype=np.int64)
    for local_vertex in range(3):
        vertex_ids = faces[:, local_vertex]
        for component in range(3):
            row_ids.append(3 * base_rows + component)
            col_ids.append(vertex_ids)
            data.append(gradients[:, local_vertex, component])

    return scipy.sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(row_ids), np.concatenate(col_ids))),
        shape=(3 * n_faces, n_vertices),
    )


def align_constraint_values_with_source_direction(
    V: FloatArray,
    F: np.ndarray,
    constrained_vertices: np.ndarray,
    constrained_values: FloatArray,
    source_support_faces: np.ndarray,
    source_direction_field: FloatArray,
    *,
    tolerance: float,
) -> tuple[FloatArray, bool, float]:
    vertices = np.asarray(V, dtype=np.float64)
    faces = np.asarray(F, dtype=np.int64)
    constrained_vertex_ids = np.asarray(constrained_vertices, dtype=np.int64)
    constrained_vertex_values = np.asarray(constrained_values, dtype=np.float64)
    source_faces = np.asarray(source_support_faces, dtype=np.int64)
    source_field = np.asarray(source_direction_field, dtype=np.float64)

    scalar_values = np.zeros(vertices.shape[0], dtype=np.float64)
    scalar_values[constrained_vertex_ids] = constrained_vertex_values

    grad_bary, _ = compute_face_basis_gradients(vertices, faces, tolerance=tolerance)
    source_face_gradients = compute_face_gradients_from_vertex_scalar(faces, grad_bary, scalar_values)[source_faces]
    source_face_directions = np.mean(source_field[faces[source_faces]], axis=1)

    grad_norm = np.linalg.norm(source_face_gradients, axis=1)
    dir_norm = np.linalg.norm(source_face_directions, axis=1)
    valid = (grad_norm > tolerance) & (dir_norm > tolerance)
    if not np.any(valid):
        return constrained_vertex_values, False, 0.0

    grad_unit = source_face_gradients[valid] / grad_norm[valid, None]
    dir_unit = source_face_directions[valid] / dir_norm[valid, None]
    mean_alignment = float(np.mean(np.einsum("ij,ij->i", grad_unit, dir_unit)))
    if mean_alignment < 0.0:
        return -constrained_vertex_values, True, -mean_alignment
    return constrained_vertex_values, False, mean_alignment


def compute_face_areas_and_unit_normals(
    V: FloatArray,
    F: np.ndarray,
) -> tuple[FloatArray, FloatArray]:
    vertices = np.asarray(V, dtype=np.float64)
    faces = np.asarray(F, dtype=np.int64)
    tri_points = vertices[faces]
    cross_vectors = np.cross(
        tri_points[:, 1] - tri_points[:, 0],
        tri_points[:, 2] - tri_points[:, 0],
    )
    doubled_areas = np.linalg.norm(cross_vectors, axis=1)
    areas = 0.5 * doubled_areas
    unit_normals = np.zeros_like(cross_vectors)
    valid = doubled_areas > 0.0
    unit_normals[valid] = cross_vectors[valid] / doubled_areas[valid, None]
    return areas, unit_normals


def compute_face_basis_gradients(
    V: FloatArray,
    F: np.ndarray,
    *,
    tolerance: float,
) -> tuple[FloatArray, FloatArray]:
    vertices = np.asarray(V, dtype=np.float64)
    faces = np.asarray(F, dtype=np.int64)
    tri_points = vertices[faces]
    areas, unit_normals = compute_face_areas_and_unit_normals(vertices, faces)
    twice_areas = 2.0 * np.maximum(areas, tolerance)

    grad_bary = np.zeros((faces.shape[0], 3, 3), dtype=np.float64)
    grad_bary[:, 0, :] = np.cross(unit_normals, tri_points[:, 2] - tri_points[:, 1]) / twice_areas[:, None]
    grad_bary[:, 1, :] = np.cross(unit_normals, tri_points[:, 0] - tri_points[:, 2]) / twice_areas[:, None]
    grad_bary[:, 2, :] = np.cross(unit_normals, tri_points[:, 1] - tri_points[:, 0]) / twice_areas[:, None]
    return grad_bary, areas


def solve_scalar_from_vertex_gradient_field(
    V: FloatArray,
    F: np.ndarray,
    gradient_field: FloatArray,
    constrained_vertices: np.ndarray,
    constrained_values: FloatArray,
    *,
    constraint_mode: str,
    tolerance: float,
) -> FloatArray:
    vertices = np.asarray(V, dtype=np.float64)
    faces = np.asarray(F, dtype=np.int64)
    field = np.asarray(gradient_field, dtype=np.float64)
    n_vertices = vertices.shape[0]
    face_field = np.mean(field[faces], axis=1)
    grad_bary, face_areas = compute_face_basis_gradients(vertices, faces, tolerance=tolerance)
    gradient_operator = build_face_gradient_operator(faces, grad_bary, n_vertices=n_vertices)
    area_weights = np.repeat(np.sqrt(np.maximum(face_areas, tolerance)), 3)
    weighted_gradient_operator = scipy.sparse.diags(area_weights) @ gradient_operator
    weighted_target = area_weights[:, None] * face_field.reshape(-1, 1)
    weighted_target_flat = weighted_target.reshape(-1)
    system_matrix = weighted_gradient_operator.T @ weighted_gradient_operator
    rhs = weighted_gradient_operator.T @ weighted_target_flat

    normalized_constraint_mode = constraint_mode.strip().lower()
    regularization = 1e-8
    if normalized_constraint_mode == "unconstrained":
        full_system = system_matrix + regularization * scipy.sparse.eye(system_matrix.shape[0], format="csr")
        scalar = np.asarray(
            scipy.sparse.linalg.spsolve(full_system, rhs),
            dtype=np.float64,
        ).reshape(-1)
        scalar -= float(np.mean(scalar))
        return scalar
    if normalized_constraint_mode != "constrained":
        raise ValueError("constraint_mode must be either 'constrained' or 'unconstrained'.")

    fixed_vertex_indices = np.asarray(constrained_vertices, dtype=np.int64)
    fixed_values = np.asarray(constrained_values, dtype=np.float64)
    source_mask = np.zeros(n_vertices, dtype=bool)
    source_mask[fixed_vertex_indices] = True
    free_vertices = np.flatnonzero(~source_mask)
    fixed_vertices = np.flatnonzero(source_mask)
    if free_vertices.size == 0:
        scalar = np.zeros(n_vertices, dtype=np.float64)
        scalar[fixed_vertex_indices] = fixed_values
        return scalar

    free_free = system_matrix[free_vertices][:, free_vertices].tocsr()
    fixed_fixed = system_matrix[free_vertices][:, fixed_vertex_indices].tocsr()
    rhs_free = np.asarray(rhs[free_vertices]).reshape(-1) - fixed_fixed @ fixed_values

    free_free = free_free + regularization * scipy.sparse.eye(free_free.shape[0], format="csr")
    scalar_free = scipy.sparse.linalg.spsolve(free_free, rhs_free)

    scalar = np.zeros(n_vertices, dtype=np.float64)
    scalar[free_vertices] = np.asarray(scalar_free, dtype=np.float64).reshape(-1)
    scalar[fixed_vertex_indices] = fixed_values
    return scalar


def build_scalar_levels(
    values: FloatArray,
    spacing: float,
    *,
    max_level_fraction: float | None,
    tolerance: float,
) -> FloatArray:
    scalar_values = np.asarray(values, dtype=np.float64)
    data_min = float(np.min(scalar_values))
    data_max = float(np.max(scalar_values))
    if data_max - data_min <= spacing + tolerance:
        return np.array([], dtype=np.float64)

    lower = data_min + 0.5 * spacing
    upper = data_max - 0.5 * spacing
    if max_level_fraction is not None:
        center = 0.5 * (data_min + data_max)
        half_extent = 0.5 * (data_max - data_min) * max_level_fraction
        lower = max(lower, center - half_extent)
        upper = min(upper, center + half_extent)

    if upper < lower - tolerance:
        return np.array([], dtype=np.float64)

    levels: list[float] = []
    level = lower
    while level <= upper + tolerance:
        if abs(level) > tolerance:
            levels.append(float(level))
        level += spacing
    return np.asarray(levels, dtype=np.float64)


def _dedupe_points(points: list[FloatArray], tolerance: float) -> list[FloatArray]:
    deduped: list[FloatArray] = []
    for point in points:
        if not any(np.linalg.norm(np.asarray(point, dtype=np.float64) - np.asarray(existing, dtype=np.float64)) <= tolerance for existing in deduped):
            deduped.append(np.asarray(point, dtype=np.float64))
    return deduped


def extract_iso_segments_from_cached_candidates(
    tri_points: FloatArray,
    tri_values: FloatArray,
    candidate_face_indices: np.ndarray,
    level: float,
    *,
    tolerance: float,
) -> list[tuple[FloatArray, FloatArray]]:
    segments: list[tuple[FloatArray, FloatArray]] = []
    edge_pairs = ((0, 1), (1, 2), (2, 0))
    for face_index in np.asarray(candidate_face_indices, dtype=np.int64):
        face_points = tri_points[face_index]
        face_values = tri_values[face_index]
        intersections: list[FloatArray] = []
        for i0, i1 in edge_pairs:
            point = _interpolate_edge_point(
                face_points[i0],
                face_points[i1],
                float(face_values[i0]),
                float(face_values[i1]),
                level,
                tolerance,
            )
            if point is not None:
                intersections.append(np.asarray(point, dtype=np.float64))

        intersections = _dedupe_points(intersections, tolerance)
        if len(intersections) == 2:
            segments.append((intersections[0], intersections[1]))
        elif len(intersections) > 2:
            max_pair: tuple[FloatArray, FloatArray] | None = None
            max_distance = -1.0
            for idx, first in enumerate(intersections):
                for second in intersections[idx + 1 :]:
                    distance = float(np.linalg.norm(first - second))
                    if distance > max_distance:
                        max_distance = distance
                        max_pair = (first, second)
            if max_pair is not None and max_distance > tolerance:
                segments.append(max_pair)
    return segments


def build_contours_from_scalar_cached(
    V: FloatArray,
    F: np.ndarray,
    scalar_field: FloatArray,
    spacing: float,
    *,
    max_level_fraction: float | None,
    tolerance: float,
) -> list[ContourLevel]:
    levels = build_scalar_levels(
        scalar_field,
        spacing,
        max_level_fraction=max_level_fraction,
        tolerance=tolerance,
    )
    vertices = np.asarray(V, dtype=np.float64)
    faces = np.asarray(F, dtype=np.int64)
    values = np.asarray(scalar_field, dtype=np.float64)
    tri_points = vertices[faces]
    tri_values = values[faces]
    tri_min = np.min(tri_values, axis=1)
    tri_max = np.max(tri_values, axis=1)

    contours: list[ContourLevel] = []
    for level in levels:
        candidate_mask = (tri_min <= float(level) + tolerance) & (tri_max >= float(level) - tolerance)
        candidate_face_indices = np.flatnonzero(candidate_mask)
        segments = extract_iso_segments_from_cached_candidates(
            tri_points,
            tri_values,
            candidate_face_indices,
            float(level),
            tolerance=tolerance,
        )
        polylines = stitch_segments_into_polylines(segments, tolerance=tolerance)
        contours.append(ContourLevel(level=float(level), polylines=[np.asarray(polyline, dtype=np.float64) for polyline in polylines]))
    return contours


def build_contours_from_scalar(
    V: FloatArray,
    F: np.ndarray,
    scalar_field: FloatArray,
    spacing: float,
    *,
    max_level_fraction: float | None,
    tolerance: float,
) -> list[ContourLevel]:
    levels = build_scalar_levels(
        scalar_field,
        spacing,
        max_level_fraction=max_level_fraction,
        tolerance=tolerance,
    )
    contours: list[ContourLevel] = []
    for level in levels:
        polylines = extract_isocontour_polylines(
            V,
            F,
            scalar_field,
            float(level),
            tolerance=tolerance,
        )
        contours.append(ContourLevel(level=float(level), polylines=polylines))
    return contours


def save_contour_levels_txt(output_dir: Path, levels: list[ContourLevel]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for level_idx, level in enumerate(levels):
        for poly_idx, polyline in enumerate(level.polylines):
            out_path = output_dir / f"level_{level_idx:04d}_poly_{poly_idx:04d}.txt"
            np.savetxt(out_path, np.asarray(polyline, dtype=np.float64), fmt="%.10f")


def build_fallback_source_curves(
    segments: list[tuple[CurvePoint, CurvePoint]],
) -> list[list[CurvePoint]]:
    return [[start, end] for start, end in segments]


def generate_vector_heat_stripe_field(
    V: FloatArray,
    F: np.ndarray,
    *,
    plane_axis: str,
    plane_value: float | None = None,
    spacing: float,
    max_level_fraction: float | None = 1.0,
    constraint_mode: str = "constrained",
    source_vertex_mode: str = "intersected_faces",
    tolerance: float = 1e-8,
) -> dict[str, object]:
    plane_axis_index = AXIS_TO_INDEX[plane_axis]
    effective_plane_value = choose_plane_value(V, plane_axis_index, plane_value)

    segments = intersect_mesh_with_axis_plane(
        V,
        F,
        plane_axis_index,
        effective_plane_value,
        tolerance=float(tolerance),
    )
    if not segments:
        raise RuntimeError("Plane/mesh intersection produced no segments.")

    source_curve_mode = "stitched"
    try:
        source_curves = stitch_segments_into_curves(segments, tolerance=float(tolerance))
    except ValueError as exc:
        if "branching" not in str(exc).lower():
            raise
        source_curves = build_fallback_source_curves(segments)
        source_curve_mode = "segment_fallback"
        print("Warning: plane intersection graph is branching; using per-segment source curves for preview/output.")
    if not source_curves:
        raise RuntimeError("Failed to derive source curves from plane-intersection segments.")

    if source_vertex_mode != "intersected_faces":
        raise ValueError("Only source-vertex mode 'intersected_faces' is implemented.")
    source_vertices = build_source_support_vertices(F, segments)
    source_support_faces = build_source_support_faces(segments)
    preferred_direction_name = plane_axis
    constrained_vertices, constrained_values = build_signed_plane_constraints(
        V,
        source_vertices,
        plane_axis_index=plane_axis_index,
        plane_value=effective_plane_value,
    )

    frame_x, frame_y, frame_n, preferred_tangent_3d, preferred_tangent_2d, transported_3d = transport_preferred_source_vectors(
        V,
        F,
        source_vertices,
        preferred_direction_name,
        tolerance=float(tolerance),
    )
    coherent_field = orient_and_normalize_transport_field(
        transported_3d,
        preferred_tangent_3d,
        tolerance=float(tolerance),
    )
    reconstruction_field = np.asarray(coherent_field, dtype=np.float64)
    reconstruction_norm = np.linalg.norm(np.asarray(reconstruction_field, dtype=np.float64), axis=1)
    valid_reconstruction = reconstruction_norm > float(tolerance)
    if np.any(valid_reconstruction):
        reconstruction_field = np.asarray(reconstruction_field, dtype=np.float64).copy()
        reconstruction_field[valid_reconstruction] /= reconstruction_norm[valid_reconstruction, None]
    constrained_values, constraints_flipped, constraint_alignment = align_constraint_values_with_source_direction(
        V,
        F,
        constrained_vertices,
        constrained_values,
        source_support_faces,
        reconstruction_field,
        tolerance=float(tolerance),
    )
    transported_magnitude = np.linalg.norm(np.asarray(transported_3d, dtype=np.float64), axis=1)
    transported_alignment = np.einsum(
        "ij,ij->i",
        np.asarray(coherent_field, dtype=np.float64),
        np.asarray(preferred_tangent_3d, dtype=np.float64),
    )
    preferred_alignment = np.linalg.norm(np.asarray(preferred_tangent_3d, dtype=np.float64), axis=1)
    scalar_field = solve_scalar_from_vertex_gradient_field(
        V,
        F,
        reconstruction_field,
        constrained_vertices,
        constrained_values,
        constraint_mode=constraint_mode,
        tolerance=float(tolerance),
    )
    contours = build_contours_from_scalar_cached(
        V,
        F,
        scalar_field,
        float(spacing),
        max_level_fraction=float(max_level_fraction) if max_level_fraction is not None else None,
        tolerance=float(tolerance),
    )

    return {
        "plane_axis": plane_axis,
        "plane_value": float(effective_plane_value),
        "preferred_direction": preferred_direction_name,
        "field_role": "stripe_normal",
        "constraint_mode": constraint_mode,
        "constraints_flipped": bool(constraints_flipped),
        "constraint_alignment": float(constraint_alignment),
        "source_curve_mode": source_curve_mode,
        "source_curves": source_curves,
        "source_segments": segments,
        "source_vertices": source_vertices,
        "source_support_faces": source_support_faces,
        "constrained_vertices": constrained_vertices,
        "constrained_values": constrained_values,
        "preferred_tangent_field": preferred_tangent_3d,
        "transported_raw_field": transported_3d,
        "transported_field": coherent_field,
        "reconstruction_field": reconstruction_field,
        "transported_magnitude": transported_magnitude,
        "transported_alignment": transported_alignment,
        "preferred_alignment": preferred_alignment,
        "scalar_field": scalar_field,
        "contours": contours,
        "frame_x": frame_x,
        "frame_y": frame_y,
        "frame_n": frame_n,
        "preferred_tangent_2d": preferred_tangent_2d,
    }


def preview_result(
    V: FloatArray,
    F: np.ndarray,
    field_name: str,
    field_values: FloatArray,
    source_curves: list[list[CurvePoint]],
    contours: list[ContourLevel],
    source_vertices: np.ndarray,
    source_vectors: FloatArray,
    propagated_vectors: FloatArray,
    constrained_vertices: np.ndarray,
    constrained_values: FloatArray,
    *,
    show_mesh_edges: bool,
    show_vectors: bool,
    show_constraints: bool,
    vector_scale: float,
    tolerance: float,
) -> None:
    if pv is None:
        raise ImportError("PyVista is required for preview mode.")

    mesh = to_pyvista_polydata(V, F)
    mesh[field_name] = np.asarray(field_values, dtype=np.float64)
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars=field_name,
        cmap="viridis",
        show_edges=show_mesh_edges,
        scalar_bar_args={"title": field_name},
    )

    for curve in source_curves:
        curve_points = np.asarray([point.xyz for point in curve], dtype=np.float64)
        curve_poly = build_polyline_polydata(
            curve_points,
            close_loop=bool(np.linalg.norm(curve_points[0] - curve_points[-1]) <= tolerance),
            tolerance=tolerance,
        )
        plotter.add_mesh(curve_poly, color="cyan", line_width=5.0)

    for level in contours:
        for polyline in level.polylines:
            points = np.asarray(polyline, dtype=np.float64)
            if points.shape[0] < 2:
                continue
            poly = build_polyline_polydata(
                points,
                close_loop=bool(np.linalg.norm(points[0] - points[-1]) <= tolerance),
                tolerance=tolerance,
            )
            plotter.add_mesh(poly, color="white", line_width=3.0)

    if show_vectors:
        all_points = np.asarray(V, dtype=np.float64)
        propagated = np.asarray(propagated_vectors, dtype=np.float64)
        propagated_norm = np.linalg.norm(propagated, axis=1)
        valid_propagated = propagated_norm > tolerance
        if np.any(valid_propagated):
            propagated_poly = pv.PolyData(all_points[valid_propagated])
            propagated_poly["vectors"] = propagated[valid_propagated]
            propagated_glyphs = propagated_poly.glyph(orient="vectors", scale=False, factor=vector_scale)
            plotter.add_mesh(propagated_glyphs, color="yellow")

        source_vertex_indices = np.asarray(source_vertices, dtype=np.int64)
        source_vec = np.asarray(source_vectors, dtype=np.float64)
        source_norm = np.linalg.norm(source_vec, axis=1)
        valid_source = source_norm > tolerance
        if source_vertex_indices.size > 0 and np.any(valid_source):
            source_points = all_points[source_vertex_indices[valid_source]]
            source_poly = pv.PolyData(source_points)
            source_poly["vectors"] = source_vec[valid_source]
            source_glyphs = source_poly.glyph(orient="vectors", scale=False, factor=1.5 * vector_scale)
            plotter.add_mesh(source_glyphs, color="magenta")

    if show_constraints:
        constraint_indices = np.asarray(constrained_vertices, dtype=np.int64)
        constraint_vals = np.asarray(constrained_values, dtype=np.float64)
        if constraint_indices.size > 0:
            constraint_points = all_points[constraint_indices]
            constraint_poly = pv.PolyData(constraint_points)
            constraint_poly["constraint_value"] = constraint_vals
            plotter.add_mesh(
                constraint_poly,
                scalars="constraint_value",
                cmap="coolwarm",
                point_size=12.0,
                render_points_as_spheres=True,
                scalar_bar_args={"title": "constraint_value"},
            )

    plotter.show()


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    V, F = load_obj_mesh(str(args.mesh_path))
    result = generate_vector_heat_stripe_field(
        V,
        F,
        plane_axis=args.plane_axis,
        plane_value=args.plane_value,
        spacing=float(args.spacing),
        max_level_fraction=float(args.max_level_fraction) if args.max_level_fraction is not None else None,
        constraint_mode=args.constraint_mode,
        source_vertex_mode=args.source_vertex_mode,
        tolerance=float(args.tolerance),
    )
    plane_value = float(result["plane_value"])
    source_curves = result["source_curves"]
    source_curve_mode = str(result["source_curve_mode"])
    segments = result["source_segments"]
    source_vertices = np.asarray(result["source_vertices"], dtype=np.int64)
    constrained_vertices = np.asarray(result["constrained_vertices"], dtype=np.int64)
    constrained_values = np.asarray(result["constrained_values"], dtype=np.float64)
    preferred_direction_name = str(result["preferred_direction"])
    preferred_tangent_3d = np.asarray(result["preferred_tangent_field"], dtype=np.float64)
    transported_3d = np.asarray(result["transported_raw_field"], dtype=np.float64)
    coherent_field = np.asarray(result["transported_field"], dtype=np.float64)
    reconstruction_field = np.asarray(result["reconstruction_field"], dtype=np.float64)
    transported_magnitude = np.asarray(result["transported_magnitude"], dtype=np.float64)
    transported_alignment = np.asarray(result["transported_alignment"], dtype=np.float64)
    preferred_alignment = np.asarray(result["preferred_alignment"], dtype=np.float64)
    scalar_field = np.asarray(result["scalar_field"], dtype=np.float64)
    contours = result["contours"]
    constraints_flipped = bool(result["constraints_flipped"])
    constraint_alignment = float(result["constraint_alignment"])

    source_curves_path = output_dir / "source_curves.obj"
    save_curves_obj(source_curves_path, source_curves, tolerance=float(args.tolerance))
    np.savetxt(output_dir / "stripe_scalar.txt", np.asarray(scalar_field, dtype=np.float64), fmt="%.10f")
    np.savetxt(output_dir / "transported_field.txt", np.asarray(coherent_field, dtype=np.float64), fmt="%.10f")
    np.savetxt(output_dir / "reconstruction_field.txt", np.asarray(reconstruction_field, dtype=np.float64), fmt="%.10f")
    np.savetxt(output_dir / "source_vertices.txt", source_vertices, fmt="%d")
    np.savetxt(output_dir / "constrained_vertices.txt", constrained_vertices, fmt="%d")
    np.savetxt(output_dir / "constrained_values.txt", np.asarray(constrained_values, dtype=np.float64), fmt="%.10f")
    constrained_debug = np.column_stack(
        (
            np.asarray(constrained_vertices, dtype=np.int64),
            np.asarray(V, dtype=np.float64)[np.asarray(constrained_vertices, dtype=np.int64)],
            np.asarray(constrained_values, dtype=np.float64)[:, None],
        )
    )
    np.savetxt(
        output_dir / "constrained_nodes_debug.txt",
        constrained_debug,
        fmt=["%d", "%.10f", "%.10f", "%.10f", "%.10f"],
    )
    np.savetxt(output_dir / "preferred_tangent_field.txt", np.asarray(preferred_tangent_3d, dtype=np.float64), fmt="%.10f")
    np.savetxt(output_dir / "source_vectors_at_source_vertices.txt", np.asarray(preferred_tangent_3d[source_vertices], dtype=np.float64), fmt="%.10f")
    np.savetxt(output_dir / "source_points_at_source_vertices.txt", np.asarray(V, dtype=np.float64)[source_vertices], fmt="%.10f")
    np.savetxt(output_dir / "transported_raw_field.txt", np.asarray(transported_3d, dtype=np.float64), fmt="%.10f")
    np.savetxt(output_dir / "transported_magnitude.txt", np.asarray(transported_magnitude, dtype=np.float64), fmt="%.10f")
    np.savetxt(output_dir / "transported_alignment.txt", np.asarray(transported_alignment, dtype=np.float64), fmt="%.10f")
    np.savetxt(output_dir / "preferred_alignment.txt", np.asarray(preferred_alignment, dtype=np.float64), fmt="%.10f")
    contour_dir = output_dir / "contours"
    save_contour_levels_txt(contour_dir, contours)

    summary = {
        "mesh_path": str(args.mesh_path),
        "plane_axis": args.plane_axis,
        "plane_value": float(plane_value),
        "preferred_direction": preferred_direction_name,
        "field_role": "stripe_normal",
        "constraint_mode": args.constraint_mode,
        "constraints_flipped": bool(constraints_flipped),
        "constraint_alignment": float(constraint_alignment),
        "spacing": float(args.spacing),
        "source_curve_count": len(source_curves),
        "source_curve_mode": source_curve_mode,
        "source_segment_count": len(segments),
        "source_vertex_count": int(source_vertices.size),
        "constrained_vertex_count": int(constrained_vertices.size),
        "contour_level_count": len(contours),
        "contour_polyline_count": int(sum(len(level.polylines) for level in contours)),
        "scalar_min": float(np.min(scalar_field)),
        "scalar_max": float(np.max(scalar_field)),
        "source_curves_obj": source_curves_path.name,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Source curves: {source_curves_path}")
    print(f"Contours directory: {contour_dir}")
    print(f"Summary: {summary_path}")
    print(f"Source vertex count: {source_vertices.size}")
    print(f"Contour polyline count: {sum(len(level.polylines) for level in contours)}")

    if args.preview:
        if args.preview_field == "scalar":
            field_name = "stripe_scalar"
            field_values = scalar_field
        elif args.preview_field == "transported_alignment":
            field_name = "transported_alignment"
            field_values = transported_alignment
        elif args.preview_field == "transported_magnitude":
            field_name = "transported_magnitude"
            field_values = transported_magnitude
        else:
            field_name = "preferred_alignment"
            field_values = preferred_alignment
        preview_result(
            V,
            F,
            field_name,
            field_values,
            source_curves,
            contours,
            source_vertices,
            preferred_tangent_3d[source_vertices],
            reconstruction_field,
            constrained_vertices,
            constrained_values,
            show_mesh_edges=bool(args.show_mesh_edges),
            show_vectors=bool(args.show_vectors),
            show_constraints=bool(args.show_constraints),
            vector_scale=float(args.vector_scale),
            tolerance=float(args.tolerance),
        )


if __name__ == "__main__":
    main()
