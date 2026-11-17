"""
Standalone prototype for hybrid contours:

- outer shell from boundary-distance contours
- interior fill from signed-heat parallel contours

This script is intentionally separate from the main pipeline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contour_extraction import ContourLevel
from contour_extraction import extract_isocontour_polylines
from contour_extraction import generate_boundary_loops
from contour_extraction import is_closed_polyline
from contour_extraction import reorder_polyline_from_vertex
from contour_extraction import resample_polyline_equal_spacing
from contour_extraction import smooth_polyline_laplacian
from heat_method_core import FloatArray
from mesh_distance_workflow import find_boundary_edges
from mesh_distance_workflow import load_obj_mesh
from mesh_distance_workflow import stitch_boundary_loops
from mesh_distance_workflow import solve_boundary_distance_field
from mesh_distance_workflow import to_pyvista_polydata
from test_potpourri3d_signed_heat_plane import AXIS_TO_INDEX
from test_potpourri3d_signed_heat_plane import CurvePoint
from test_potpourri3d_signed_heat_plane import build_polyline_polydata
from test_potpourri3d_signed_heat_plane import build_solver_curve_payloads
from test_potpourri3d_signed_heat_plane import intersect_mesh_with_axis_plane
from test_potpourri3d_signed_heat_plane import save_curves_obj
from test_potpourri3d_signed_heat_plane import stitch_segments_into_curves
from test_vector_heat_stripe_field import generate_vector_heat_stripe_field

try:
    import potpourri3d as pp3d  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pp3d = None

try:
    import pyvista as pv  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pv = None

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - optional dependency
    cKDTree = None


EDGE_PAIRS = ((0, 1), (1, 2), (2, 0))


@dataclass(slots=True)
class HybridSample:
    xyz: FloatArray
    face_index: int
    barycentric: FloatArray
    boundary_value: float


@dataclass(slots=True)
class HybridSegment:
    start: HybridSample
    end: HybridSample
    parallel_level: float


@dataclass(slots=True)
class HybridPolyline:
    points: FloatArray
    family: str
    level: float
    closed: bool
    effective_width: float | None = None
    point_widths: FloatArray | None = None


@dataclass(slots=True)
class ParallelTriangleCache:
    tri_points: FloatArray
    tri_parallel: FloatArray
    tri_boundary: FloatArray
    tri_parallel_min: FloatArray
    tri_parallel_max: FloatArray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prototype hybrid contours from boundary distance plus signed-heat parallel field.",
    )
    parser.add_argument("mesh_path", type=Path, help="Input triangle mesh.")
    parser.add_argument("output_dir", type=Path, help="Directory to write hybrid debug outputs.")
    parser.add_argument("--spacing", type=float, required=True, help="Contour spacing.")
    parser.add_argument("--resample-spacing", type=float, default=None, help="Optional resampling spacing used before greedy chaining. Defaults to --spacing.")
    parser.add_argument("--sharp-angle-deg", type=float, default=35.0, help="Angle threshold in degrees for preserving sharp points during hybrid resampling.")
    parser.add_argument(
        "--boundary-count",
        type=int,
        required=True,
        help="Number of boundary-family contours, counting the exact boundary as contour 0.",
    )
    parser.add_argument(
        "--axis-mode",
        choices=("fixed", "auto_tangent_alternating"),
        default="fixed",
        help="How to choose the signed-heat plane axis. 'fixed' uses --axis, 'auto_tangent_alternating' drops the dominant normal axis and alternates the remaining two axes by layer main index parity.",
    )
    parser.add_argument(
        "--axis",
        choices=("x", "y", "z"),
        default="x",
        help="Axis normal for the midpoint plane source when --axis-mode=fixed. 'x' means a yz-parallel plane.",
    )
    parser.add_argument(
        "--plane-value",
        type=float,
        default=None,
        help="Optional plane value override. Default uses midpoint extent along the selected axis.",
    )
    parser.add_argument(
        "--boundary-field-backend",
        choices=("native", "potpourri3d_heat", "potpourri_fast_marching_unsigned"),
        default="native",
        help="Backend used for the hybrid boundary-distance field.",
    )
    parser.add_argument(
        "--parallel-field-backend",
        choices=("signed_heat", "fast_marching_unsigned", "vector_heat_stripe"),
        default="signed_heat",
        help="Backend used for the hybrid interior parallel field.",
    )
    parser.add_argument(
        "--level-set-constraint",
        choices=("ZeroSet", "None", "Multiple"),
        default="None",
        help="Level-set constraint passed to MeshSignedHeatSolver. Ignored by fast_marching_unsigned and vector_heat_stripe.",
    )
    parser.add_argument(
        "--soft-level-set-weight",
        type=float,
        default=-1.0,
        help="Soft level-set weight passed to MeshSignedHeatSolver. Ignored by fast_marching_unsigned and vector_heat_stripe.",
    )
    parser.add_argument(
        "--contour-max-level-fraction",
        type=float,
        default=1.0,
        help="Optional fraction of field range used to trim contour extraction.",
    )
    parser.add_argument(
        "--min-polyline-points",
        type=int,
        default=2,
        help="Discard kept parallel contour polylines with fewer than this many points.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview the hybrid result with PyVista.",
    )
    parser.add_argument(
        "--preview-field",
        choices=("parallel", "boundary"),
        default="parallel",
        help="Scalar field used to color the mesh in preview mode.",
    )
    parser.add_argument(
        "--show-mesh-edges",
        action="store_true",
        help="Show triangle edges in preview mode.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
        help="Geometric tolerance.",
    )
    return parser


def save_polylines_obj(path: Path, polylines: list[FloatArray], *, tolerance: float) -> None:
    lines: list[str] = []
    vertex_offset = 1
    for polyline in polylines:
        points = np.asarray(polyline, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
            continue
        for point in points:
            x, y, z = point.tolist()
            lines.append(f"v {x:.10f} {y:.10f} {z:.10f}")
        indices = list(range(vertex_offset, vertex_offset + points.shape[0]))
        if np.linalg.norm(points[0] - points[-1]) <= tolerance:
            indices = indices[:-1]
            if indices:
                indices.append(indices[0])
        if len(indices) >= 2:
            lines.append("l " + " ".join(str(idx) for idx in indices))
        vertex_offset += points.shape[0]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def save_contour_levels_txt(output_dir: Path, levels: list[ContourLevel]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for level_idx, level in enumerate(levels):
        for poly_idx, polyline in enumerate(level.polylines):
            out_path = output_dir / f"level_{level_idx:04d}_poly_{poly_idx:04d}.txt"
            np.savetxt(out_path, np.asarray(polyline, dtype=np.float64), fmt="%.10f")


def build_boundary_curve_segments(
    V: FloatArray,
    F: np.ndarray,
    *,
    tolerance: float,
) -> list[tuple[CurvePoint, CurvePoint]]:
    faces = np.asarray(F, dtype=np.int64)
    boundary_edges = np.asarray(find_boundary_edges(faces), dtype=np.int64)
    if boundary_edges.size == 0:
        return []

    edge_to_face: dict[tuple[int, int], int] = {}
    for face_index, (a, b, c) in enumerate(faces):
        for i0, i1 in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = tuple(sorted((i0, i1)))
            edge_to_face[key] = face_index

    vertices = np.asarray(V, dtype=np.float64)
    segments: list[tuple[CurvePoint, CurvePoint]] = []
    for v0, v1 in boundary_edges:
        i0 = int(v0)
        i1 = int(v1)
        face_index = edge_to_face.get(tuple(sorted((i0, i1))))
        if face_index is None:
            continue
        face = faces[face_index]
        bary0 = np.zeros(3, dtype=np.float64)
        bary1 = np.zeros(3, dtype=np.float64)
        local0 = np.where(face == i0)[0]
        local1 = np.where(face == i1)[0]
        if local0.size != 1 or local1.size != 1:
            continue
        bary0[int(local0[0])] = 1.0
        bary1[int(local1[0])] = 1.0
        segments.append(
            (
                CurvePoint(xyz=np.asarray(vertices[i0], dtype=np.float64), face_index=int(face_index), barycentric=bary0),
                CurvePoint(xyz=np.asarray(vertices[i1], dtype=np.float64), face_index=int(face_index), barycentric=bary1),
            ),
        )
    return segments


def solve_hybrid_boundary_field(
    V: FloatArray,
    F: np.ndarray,
    *,
    backend: str,
    tolerance: float,
) -> tuple[FloatArray, list[FloatArray], list[list[CurvePoint]], list[tuple[CurvePoint, CurvePoint]]]:
    boundary_loops = generate_boundary_loops(V, F)
    normalized_backend = backend.strip().lower()
    if normalized_backend == "native":
        boundary_result = solve_boundary_distance_field(V, F, prefer_libigl=True, backend="native", return_intermediates=False)
        return (
            np.asarray(boundary_result.distance, dtype=np.float64),
            boundary_loops,
            [],
            [],
        )
    if normalized_backend == "potpourri3d_heat":
        boundary_result = solve_boundary_distance_field(V, F, prefer_libigl=True, backend="potpourri3d", return_intermediates=False)
        return (
            np.asarray(boundary_result.distance, dtype=np.float64),
            boundary_loops,
            [],
            [],
        )
    if normalized_backend != "potpourri_fast_marching_unsigned":
        raise ValueError(
            "boundary_field_backend must be one of 'native', 'potpourri3d_heat', or 'potpourri_fast_marching_unsigned'.",
        )
    if pp3d is None:
        raise ImportError("Boundary field backend 'potpourri_fast_marching_unsigned' requires the optional `potpourri3d` package.")

    boundary_segments = build_boundary_curve_segments(V, F, tolerance=tolerance)
    if not boundary_segments:
        raise RuntimeError("Mesh has no boundary segments to use as fast-marching boundary sources.")
    boundary_curves = stitch_segments_into_curves(boundary_segments, tolerance=tolerance)
    if not boundary_curves:
        raise RuntimeError("Failed to stitch boundary segments into source curves.")
    curve_payloads = build_solver_curve_payloads(boundary_curves, boundary_segments, tolerance=tolerance)
    solver = pp3d.MeshFastMarchingDistanceSolver(V, F)
    source_distances = [[0.0] * len(curve) for curve in curve_payloads]
    distances = solver.compute_distance(curve_payloads, source_distances, False)
    return (
        np.asarray(distances, dtype=np.float64),
        boundary_loops,
        boundary_curves,
        boundary_segments,
    )


def _point_key(point: FloatArray, tolerance: float) -> tuple[int, int, int]:
    scaled = np.asarray(point, dtype=np.float64) / tolerance
    return tuple(np.round(scaled).astype(np.int64))


def stitch_segments_into_fragment_polylines(
    segments: list[tuple[FloatArray, FloatArray]],
    *,
    tolerance: float,
) -> list[FloatArray]:
    """Stitch segments into simple fragments, splitting at branch nodes."""
    if not segments:
        return []

    points_by_key: dict[tuple[int, int, int], FloatArray] = {}
    adjacency: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    undirected_edges: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()

    for start, end in segments:
        key0 = _point_key(start, tolerance)
        key1 = _point_key(end, tolerance)
        if key0 == key1:
            continue
        points_by_key.setdefault(key0, np.asarray(start, dtype=np.float64))
        points_by_key.setdefault(key1, np.asarray(end, dtype=np.float64))
        adjacency.setdefault(key0, []).append(key1)
        adjacency.setdefault(key1, []).append(key0)
        edge = (key0, key1) if key0 <= key1 else (key1, key0)
        undirected_edges.add(edge)

    if not undirected_edges:
        return []

    visited_edges: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
    polylines: list[FloatArray] = []

    def canon_edge(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return (a, b) if a <= b else (b, a)

    def trace_path(start_key: tuple[int, int, int], next_key: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        ordered = [start_key, next_key]
        prev_key = start_key
        current_key = next_key
        visited_edges.add(canon_edge(start_key, next_key))

        while True:
            current_neighbors = adjacency.get(current_key, [])
            if len(current_neighbors) != 2:
                break
            candidates = [
                candidate
                for candidate in current_neighbors
                if candidate != prev_key and canon_edge(current_key, candidate) not in visited_edges
            ]
            if not candidates:
                break
            candidate = candidates[0]
            visited_edges.add(canon_edge(current_key, candidate))
            ordered.append(candidate)
            prev_key, current_key = current_key, candidate
        return ordered

    start_nodes = [key for key, neighbors in adjacency.items() if len(neighbors) != 2]
    for start_key in start_nodes:
        for neighbor_key in adjacency.get(start_key, []):
            edge = canon_edge(start_key, neighbor_key)
            if edge in visited_edges:
                continue
            ordered_keys = trace_path(start_key, neighbor_key)
            polylines.append(np.asarray([points_by_key[key] for key in ordered_keys], dtype=np.float64))

    for edge in undirected_edges:
        if edge in visited_edges:
            continue
        key0, key1 = edge
        ordered_keys = trace_path(key0, key1)
        polylines.append(np.asarray([points_by_key[key] for key in ordered_keys], dtype=np.float64))

    return polylines


def compute_midpoint_plane_value(V: FloatArray, axis_index: int) -> float:
    coords = np.asarray(V[:, axis_index], dtype=np.float64)
    return float(0.5 * (np.min(coords) + np.max(coords)))


def infer_layer_axes_from_surface_normals(V: FloatArray, F: np.ndarray) -> tuple[str, tuple[str, str]]:
    vertices = np.asarray(V, dtype=np.float64)
    faces = np.asarray(F, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("F must have shape (m, 3).")
    if faces.shape[0] == 0:
        return "z", ("x", "y")

    tri_points = vertices[faces]
    face_normals = np.cross(
        tri_points[:, 1] - tri_points[:, 0],
        tri_points[:, 2] - tri_points[:, 0],
    )
    normal_strength = np.sum(np.abs(face_normals), axis=0)
    dominant_axis_index = int(np.argmax(normal_strength))
    dominant_axis = "xyz"[dominant_axis_index]
    tangent_axes = tuple(axis for axis in ("x", "y", "z") if axis != dominant_axis)
    return dominant_axis, tangent_axes  # type: ignore[return-value]


def choose_hybrid_parallel_axis(
    V: FloatArray,
    F: np.ndarray,
    *,
    axis_mode: str,
    fixed_axis: str,
    layer_main_index: int,
) -> tuple[str, str | None, tuple[str, str] | None]:
    normalized_mode = axis_mode.strip().lower()
    if normalized_mode == "fixed":
        return fixed_axis, None, None
    if normalized_mode != "auto_tangent_alternating":
        raise ValueError(
            "axis_mode must be either 'fixed' or 'auto_tangent_alternating'.",
        )

    dominant_normal_axis, tangent_axes = infer_layer_axes_from_surface_normals(V, F)
    selected_axis = tangent_axes[layer_main_index % 2]
    return selected_axis, dominant_normal_axis, tangent_axes


def _polyline_arclength(points: FloatArray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def resample_polyline_uniform(
    polyline: FloatArray,
    spacing: float,
    *,
    sharp_angle_deg: float,
    tolerance: float,
) -> FloatArray:
    return resample_polyline_equal_spacing(
        np.asarray(polyline, dtype=np.float64),
        spacing,
        angle_threshold_degrees=sharp_angle_deg,
        tolerance=tolerance,
    )


def reverse_open_polyline(points: FloatArray) -> FloatArray:
    pts = np.asarray(points, dtype=np.float64)
    return pts[::-1].copy()


def reverse_closed_opened_polyline(points: FloatArray) -> FloatArray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return pts.copy()
    return np.vstack((pts[:1], pts[:0:-1]))


def choose_closed_loop_orientation(points: FloatArray) -> FloatArray:
    pts = np.asarray(points, dtype=np.float64)
    reversed_pts = reverse_closed_opened_polyline(pts)
    if pts.shape[0] < 3:
        return pts
    if np.linalg.norm(pts[1] - pts[-1]) <= np.linalg.norm(reversed_pts[1] - reversed_pts[-1]):
        return pts
    return reversed_pts


def choose_open_orientation(points: FloatArray, current_point: FloatArray) -> FloatArray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return pts.copy()
    if np.linalg.norm(pts[0] - current_point) <= np.linalg.norm(pts[-1] - current_point):
        return pts
    return reverse_open_polyline(pts)


def choose_orientation_with_reference(points: FloatArray, reference_direction: FloatArray) -> FloatArray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return pts.copy()
    direction = pts[-1] - pts[0]
    if np.linalg.norm(direction) <= 1e-12 or np.linalg.norm(reference_direction) <= 1e-12:
        return pts.copy()
    if float(np.dot(direction, reference_direction)) >= 0.0:
        return pts.copy()
    return reverse_open_polyline(pts)


def first_nonzero_polyline_direction(items: list[HybridPolyline], *, tolerance: float) -> FloatArray | None:
    for item in items:
        points = np.asarray(item.points, dtype=np.float64)
        if item.closed and points.shape[0] >= 2:
            points = points[:-1]
        if points.shape[0] < 2:
            continue
        direction = points[-1] - points[0]
        if np.linalg.norm(direction) > tolerance:
            return direction
    return None


def resample_boundary_family(
    boundary_loops: list[FloatArray],
    boundary_levels: list[ContourLevel],
    spacing: float,
    *,
    nominal_width: float,
    sharp_angle_deg: float,
    tolerance: float,
) -> list[HybridPolyline]:
    items: list[HybridPolyline] = []
    for loop in boundary_loops:
        points = resample_polyline_uniform(loop, spacing, sharp_angle_deg=sharp_angle_deg, tolerance=tolerance)
        items.append(
            HybridPolyline(
                points=points,
                family="boundary",
                level=0.0,
                closed=is_closed_polyline(points, tolerance=tolerance),
                effective_width=float(nominal_width),
            ),
        )
    for level in boundary_levels:
        for polyline in level.polylines:
            points = resample_polyline_uniform(np.asarray(polyline, dtype=np.float64), spacing, sharp_angle_deg=sharp_angle_deg, tolerance=tolerance)
            items.append(
                HybridPolyline(
                    points=points,
                    family="boundary",
                    level=float(level.level),
                    closed=is_closed_polyline(points, tolerance=tolerance),
                    effective_width=float(nominal_width),
                ),
            )
    return items


def filter_boundary_items_by_start_offset(
    items: list[HybridPolyline],
    *,
    spacing: float,
    boundary_start_offset_count: int,
    tolerance: float,
) -> list[HybridPolyline]:
    if boundary_start_offset_count < 0:
        raise ValueError("boundary_start_offset_count must be non-negative.")
    if boundary_start_offset_count == 0:
        return items
    min_level = float(boundary_start_offset_count) * float(spacing)
    return [item for item in items if item.level >= min_level - tolerance]


def resample_parallel_family(
    parallel_polylines: list[FloatArray],
    level_by_polyline: list[float],
    spacing: float,
    *,
    nominal_width: float,
    sharp_angle_deg: float,
    tolerance: float,
) -> list[HybridPolyline]:
    items: list[HybridPolyline] = []
    for polyline, level in zip(parallel_polylines, level_by_polyline):
        points = resample_polyline_uniform(np.asarray(polyline, dtype=np.float64), spacing, sharp_angle_deg=sharp_angle_deg, tolerance=tolerance)
        items.append(
                HybridPolyline(
                    points=points,
                    family="parallel",
                    level=float(level),
                    closed=is_closed_polyline(points, tolerance=tolerance),
                    effective_width=float(nominal_width),
                ),
            )
    return items


def smooth_polylines(
    polylines: list[FloatArray],
    *,
    enabled: bool,
    iterations: int,
    alpha: float,
    tolerance: float,
) -> list[FloatArray]:
    if not enabled:
        return [np.asarray(polyline, dtype=np.float64).copy() for polyline in polylines]
    return [
        smooth_polyline_laplacian(
            np.asarray(polyline, dtype=np.float64),
            iterations=iterations,
            alpha=alpha,
            tolerance=tolerance,
        )
        for polyline in polylines
    ]


def group_boundary_items(items: list[HybridPolyline], *, tolerance: float) -> list[list[HybridPolyline]]:
    ordered = sorted(items, key=lambda item: item.level, reverse=True)
    groups: list[list[HybridPolyline]] = []
    for item in ordered:
        if not groups or abs(groups[-1][0].level - item.level) > tolerance:
            groups.append([item])
        else:
            groups[-1].append(item)
    return groups


def collect_polyline_handoff_targets(
    items: list[HybridPolyline],
    *,
    tolerance: float,
) -> list[FloatArray]:
    targets: list[FloatArray] = []
    for item in items:
        points = np.asarray(item.points, dtype=np.float64)
        if points.shape[0] == 0:
            continue
        targets.append(points[0])
        if points.shape[0] >= 2 and np.linalg.norm(points[0] - points[-1]) > tolerance:
            targets.append(points[-1])
    return targets


def open_closed_loop_with_half_width(
    points: FloatArray,
    start_idx: int,
    *,
    split_width: float,
    tolerance: float,
) -> FloatArray:
    loop = np.asarray(points, dtype=np.float64)
    if loop.shape[0] <= 1 or np.linalg.norm(loop[0] - loop[-1]) > tolerance:
        return loop.copy()

    core = np.asarray(loop[:-1], dtype=np.float64)
    count = core.shape[0]
    if count == 0:
        return loop.copy()
    if count == 1:
        return np.vstack((core[0], core[0]))

    rotated = np.vstack((core[start_idx:], core[:start_idx]))
    edge_vectors = np.roll(rotated, -1, axis=0) - rotated
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    perimeter = float(np.sum(edge_lengths))
    if perimeter <= tolerance:
        return loop.copy()

    split_offset = max(0.0, 0.25 * float(split_width))
    max_offset = max(0.0, 0.5 * perimeter - tolerance)
    offset = min(split_offset, max_offset)
    if offset <= tolerance:
        return reorder_polyline_from_vertex(loop, int(start_idx) % count, tolerance=tolerance)

    def point_at_arc_length(distance: float) -> tuple[np.ndarray, int]:
        distance = min(max(float(distance), 0.0), perimeter)
        traversed = 0.0
        for edge_idx, edge_length in enumerate(edge_lengths):
            next_traversed = traversed + float(edge_length)
            if distance <= next_traversed + tolerance or edge_idx == count - 1:
                if edge_length <= tolerance:
                    return rotated[(edge_idx + 1) % count].copy(), (edge_idx + 1) % count
                t = (distance - traversed) / edge_length
                t = min(1.0, max(0.0, t))
                point = (1.0 - t) * rotated[edge_idx] + t * rotated[(edge_idx + 1) % count]
                return np.asarray(point, dtype=np.float64), (edge_idx + 1) % count
            traversed = next_traversed
        return rotated[0].copy(), 0

    forward_point, _ = point_at_arc_length(offset)
    backward_point, _ = point_at_arc_length(perimeter - offset)
    # On densely sampled corners the end trim can span multiple edges.
    # Stop at the backward trim point, rather than visiting the discarded
    # vertices and drawing a shortcut back across a hole or concavity.
    vertex_distances = np.r_[0.0, np.cumsum(edge_lengths)[:-1]]
    retained = rotated[(vertex_distances > offset + tolerance)
                       & (vertex_distances < perimeter - offset - tolerance)]
    return np.vstack((forward_point, retained, backward_point))


def open_closed_loop_for_item(
    item: HybridPolyline,
    start_idx: int,
    *,
    tolerance: float,
) -> FloatArray:
    split_width = float(item.effective_width if item.effective_width is not None else 0.0)
    return open_closed_loop_with_half_width(
        item.points,
        start_idx,
        split_width=split_width,
        tolerance=tolerance,
    )


def choose_closed_loop_opening_for_handoff(
    item: HybridPolyline,
    *,
    previous_point: FloatArray | None,
    target_points: list[FloatArray],
    tolerance: float,
) -> FloatArray:
    loop = np.asarray(item.points, dtype=np.float64)
    if loop.shape[0] <= 1 or np.linalg.norm(loop[0] - loop[-1]) > tolerance:
        return loop.copy()

    search_points = loop[:-1]
    if search_points.shape[0] == 0:
        return loop.copy()

    target_array = np.asarray(target_points, dtype=np.float64) if target_points else np.empty((0, 3), dtype=np.float64)
    target_tree = None
    if target_array.shape[0] > 0:
        from scipy.spatial import cKDTree

        target_tree = cKDTree(target_array)

    candidate_orientation: list[np.ndarray] = []
    candidate_start_indices: list[np.ndarray] = []
    candidate_start_distances: list[np.ndarray] = []
    candidate_exit_distances: list[np.ndarray] = []
    for orientation_index, oriented_loop in enumerate((loop, loop[::-1].copy())):
        core = np.asarray(oriented_loop[:-1], dtype=np.float64)
        count = core.shape[0]
        edge_vectors = np.roll(core, -1, axis=0) - core
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        cumulative = np.cumsum(edge_lengths)
        perimeter = float(cumulative[-1]) if cumulative.size else 0.0
        if perimeter <= tolerance:
            continue
        offset = min(max(0.0, 0.25 * float(item.effective_width or 0.0)), max(0.0, 0.5 * perimeter - tolerance))
        if offset <= tolerance:
            start_points = core
            exit_points = core
        else:
            starts = np.arange(count, dtype=np.int64)
            start_distances = np.concatenate(([0.0], cumulative[:-1]))

            def interpolate_cyclic(distances: np.ndarray) -> np.ndarray:
                wrapped = np.mod(distances, perimeter)
                edge_indices = np.searchsorted(cumulative, wrapped, side="left")
                edge_indices = np.minimum(edge_indices, count - 1)
                previous = np.where(edge_indices > 0, cumulative[edge_indices - 1], 0.0)
                local = (wrapped - previous) / edge_lengths[edge_indices]
                return (1.0 - local[:, None]) * core[edge_indices] + local[:, None] * core[(edge_indices + 1) % count]

            start_points = interpolate_cyclic(start_distances + offset)
            exit_points = interpolate_cyclic(start_distances - offset)

        start_distance_values = (
            np.linalg.norm(start_points - previous_point[None, :], axis=1)
            if previous_point is not None
            else np.zeros(count, dtype=np.float64)
        )
        exit_distance_values = (
            np.asarray(target_tree.query(exit_points, k=1)[0], dtype=np.float64)
            if target_tree is not None
            else np.zeros(count, dtype=np.float64)
        )
        candidate_orientation.append(np.full(count, orientation_index, dtype=np.int64))
        candidate_start_indices.append(np.arange(count, dtype=np.int64))
        candidate_start_distances.append(start_distance_values)
        candidate_exit_distances.append(exit_distance_values)

    if not candidate_orientation:
        return loop.copy()
    orientation_values = np.concatenate(candidate_orientation)
    start_indices = np.concatenate(candidate_start_indices)
    start_distances = np.concatenate(candidate_start_distances)
    exit_distances = np.concatenate(candidate_exit_distances)
    best_candidate = int(
        np.lexsort(
            (
                start_indices,
                orientation_values,
                start_distances,
                exit_distances,
                start_distances + exit_distances,
            ),
        )[0]
    )
    best_orientation = int(orientation_values[best_candidate])
    best_start_idx = int(start_indices[best_candidate])
    best_loop = loop if best_orientation == 0 else loop[::-1].copy()
    return open_closed_loop_for_item(
        HybridPolyline(
            points=best_loop,
            family=item.family,
            level=item.level,
            closed=item.closed,
            effective_width=item.effective_width,
        ),
        best_start_idx,
        tolerance=tolerance,
    )


def chain_boundary_family(
    items: list[HybridPolyline],
    *,
    start_selection_mode: str = "first",
    handoff_targets: list[FloatArray] | None = None,
    tolerance: float,
) -> tuple[list[FloatArray], list[FloatArray], FloatArray | None]:
    if start_selection_mode == "lowest_z":
        start_selection_mode = "lowest_y"
    if start_selection_mode not in {"first", "lowest_y"}:
        raise ValueError("start_selection_mode must be either 'first' or 'lowest_y'.")
    # Degenerate extraction fragments can contain only a closing point, which
    # leaves no candidate vertices after removing the duplicated endpoint.
    items = [
        item for item in items
        if (len(item.points) >= (4 if item.closed else 2)
            and np.linalg.norm(np.diff(item.points, axis=0), axis=1).sum() > tolerance)
    ]
    groups = group_boundary_items(items, tolerance=tolerance)
    if not groups:
        return [], [], None

    ordered_paths: list[FloatArray] = []
    ordered_widths: list[FloatArray] = []
    current_point: FloatArray | None = None
    for group_index, group in enumerate(groups):
        remaining = group.copy()
        while remaining:
            if current_point is None:
                if start_selection_mode == "lowest_y":
                    best: tuple[float, float, float, int, int] | None = None
                    for idx, item in enumerate(remaining):
                        search_points = item.points[:-1] if item.closed else item.points
                        lex_order = np.lexsort((search_points[:, 0], search_points[:, 2], search_points[:, 1]))
                        start_idx = int(lex_order[0])
                        point = search_points[start_idx]
                        candidate = (float(point[1]), float(point[2]), float(point[0]), idx, start_idx)
                        if best is None or candidate < best:
                            best = candidate
                    assert best is not None
                    _, _, _, selected_idx, selected_start_idx = best
                else:
                    selected_idx = 0
                    selected_start_idx = 0
            else:
                search_parts = [item.points[:-1] if item.closed else item.points for item in remaining]
                part_lengths = np.asarray([points.shape[0] for points in search_parts], dtype=np.int64)
                all_points = np.vstack(search_parts)
                closest_flat_idx = int(np.argmin(np.linalg.norm(all_points - current_point[None, :], axis=1)))
                part_ends = np.cumsum(part_lengths)
                selected_idx = int(np.searchsorted(part_ends, closest_flat_idx, side="right"))
                previous_end = int(part_ends[selected_idx - 1]) if selected_idx > 0 else 0
                selected_start_idx = closest_flat_idx - previous_end

            item = remaining.pop(selected_idx)
            if item.closed:
                is_final_boundary_item = group_index == len(groups) - 1 and not remaining
                if is_final_boundary_item and handoff_targets:
                    opened = choose_closed_loop_opening_for_handoff(
                        item,
                        previous_point=current_point,
                        target_points=handoff_targets,
                        tolerance=tolerance,
                    )
                else:
                    opened = open_closed_loop_for_item(
                        item,
                        selected_start_idx,
                        tolerance=tolerance,
                    )
                    opened = choose_closed_loop_orientation(opened)
            else:
                opened = reorder_polyline_from_vertex(item.points, selected_start_idx, tolerance=tolerance)
            # Opening/deduplicating a collapsed offset ring can leave a pair
            # of almost identical endpoints. It is not deposited material.
            length=float(np.linalg.norm(np.diff(opened,axis=0),axis=1).sum())
            if not np.isfinite(length):raise ValueError('Boundary path has nonfinite length.')
            if len(opened)<2 or length<=max(tolerance,1e-6):continue
            ordered_paths.append(opened)
            ordered_widths.append(width_profile_for_polyline(item, opened))
            current_point = opened[-1]

    return ordered_paths, ordered_widths, current_point


def choose_parallel_candidate(
    remaining: list[HybridPolyline],
    current_point: FloatArray,
    *,
    reference_direction: FloatArray | None = None,
    tolerance: float = 1e-8,
) -> tuple[int, FloatArray]:
    best_idx = -1
    best_points: FloatArray | None = None
    best_score: tuple[float, float, int] | None = None
    for idx, item in enumerate(remaining):
        points = np.asarray(item.points, dtype=np.float64)
        candidates = [points]
        if points.shape[0] >= 2 and np.linalg.norm(points[0] - points[-1]) > tolerance:
            candidates.append(reverse_open_polyline(points))

        for oriented in candidates:
            endpoint_distance = float(np.linalg.norm(oriented[0] - current_point))
            direction_penalty = 0.0
            if reference_direction is not None and oriented.shape[0] >= 2:
                direction = oriented[-1] - oriented[0]
                if np.linalg.norm(direction) > tolerance and float(np.dot(direction, reference_direction)) < 0.0:
                    direction_penalty = 1.0
            score = (endpoint_distance, direction_penalty, idx)
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
                best_points = oriented
    if best_points is None:
        raise RuntimeError("Failed to choose the next parallel polyline.")
    return best_idx, best_points


def chain_parallel_family(
    items: list[HybridPolyline],
    current_point: FloatArray | None,
    *,
    direction_mode: str = "connect_nearest",
    tolerance: float,
) -> tuple[list[FloatArray], list[FloatArray], FloatArray | None]:
    if direction_mode not in {"connect_nearest", "one_direction"}:
        raise ValueError("direction_mode must be either 'connect_nearest' or 'one_direction'.")
    if not items:
        return [], [], current_point

    remaining = items.copy()
    ordered_paths: list[FloatArray] = []
    ordered_widths: list[FloatArray] = []
    reference_direction = first_nonzero_polyline_direction(remaining, tolerance=tolerance) if direction_mode == "one_direction" else None

    if current_point is None:
        start_idx = 0
        start_item = remaining.pop(start_idx)
        if reference_direction is None:
            oriented = start_item.points if np.linalg.norm(start_item.points[0] - start_item.points[-1]) >= tolerance else start_item.points[:-1]
        else:
            oriented = choose_orientation_with_reference(start_item.points, reference_direction)
        ordered_paths.append(oriented)
        ordered_widths.append(width_profile_for_polyline(start_item, oriented))
        current_point = oriented[-1]

    while remaining:
        chosen_idx, chosen_points = choose_parallel_candidate(
            remaining,
            current_point,
            reference_direction=reference_direction,
            tolerance=tolerance,
        )
        chosen_item = remaining.pop(chosen_idx)
        ordered_paths.append(chosen_points)
        ordered_widths.append(width_profile_for_polyline(chosen_item, chosen_points))
        current_point = chosen_points[-1]

    return ordered_paths, ordered_widths, current_point


def concatenate_paths(parts: list[FloatArray]) -> FloatArray:
    if not parts:
        return np.empty((0, 3), dtype=np.float64)
    stacked: list[FloatArray] = []
    for idx, part in enumerate(parts):
        points = np.asarray(part, dtype=np.float64)
        if points.shape[0] == 0:
            continue
        if idx > 0 and np.linalg.norm(stacked[-1][-1] - points[0]) <= 1e-8:
            stacked.append(points[1:])
        else:
            stacked.append(points)
    nonempty = [part for part in stacked if part.shape[0] > 0]
    if not nonempty:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(nonempty)


def concatenate_width_profiles(parts: list[FloatArray], width_parts: list[FloatArray]) -> FloatArray:
    if len(parts) != len(width_parts):
        raise ValueError("parts and width_parts must have the same length.")
    if not parts:
        return np.empty((0,), dtype=np.float64)

    stacked: list[FloatArray] = []
    for idx, (part, widths) in enumerate(zip(parts, width_parts)):
        points = np.asarray(part, dtype=np.float64)
        width_values = np.asarray(widths, dtype=np.float64).reshape(-1)
        if points.shape[0] == 0:
            continue
        if width_values.shape[0] != points.shape[0]:
            raise ValueError("Each width profile must have one value per point.")
        if idx > 0 and stacked and np.linalg.norm(np.asarray(parts[idx - 1], dtype=np.float64)[-1] - points[0]) <= 1e-8:
            stacked.append(width_values[1:])
        else:
            stacked.append(width_values)
    nonempty = [part for part in stacked if part.shape[0] > 0]
    if not nonempty:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(nonempty)


def concatenate_path_tags(parts: list[FloatArray], tags: list[str], *, tolerance: float = 1e-8) -> np.ndarray:
    """Create one role label per concatenated path point."""
    if len(parts) != len(tags):
        raise ValueError("parts and tags must have the same length.")
    values: list[np.ndarray] = []
    previous: np.ndarray | None = None
    for part, tag in zip(parts, tags, strict=False):
        points = np.asarray(part, dtype=np.float64)
        if points.shape[0] == 0:
            continue
        labels = np.full(points.shape[0], str(tag), dtype="U32")
        if previous is not None and np.linalg.norm(previous - points[0]) <= tolerance:
            labels = labels[1:]
        if labels.size:
            values.append(labels)
        previous = points[-1]
    return np.concatenate(values) if values else np.empty((0,), dtype="U32")


def encode_path_tag_blocks(tags: np.ndarray) -> list[tuple[int, int, str]]:
    """Run-length encode point roles as half-open index blocks."""
    values = np.asarray(tags, dtype=str).reshape(-1)
    if values.size == 0:
        return []
    starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    ends = np.r_[starts[1:], values.size]
    return [(int(start), int(end), str(values[start])) for start, end in zip(starts, ends, strict=False)]


def width_profile_for_polyline(item: HybridPolyline, oriented_points: FloatArray) -> FloatArray:
    points = np.asarray(oriented_points, dtype=np.float64)
    if item.point_widths is not None:
        widths = np.asarray(item.point_widths, dtype=np.float64).reshape(-1)
        original = np.asarray(item.points, dtype=np.float64)
        if widths.shape[0] == points.shape[0] == original.shape[0]:
            if np.allclose(points, original, atol=1e-9, rtol=0.0):
                return widths.copy()
            if np.allclose(points, original[::-1], atol=1e-9, rtol=0.0):
                return widths[::-1].copy()
    base_width = float(item.effective_width if item.effective_width is not None else 0.0)
    return np.full(points.shape[0], base_width, dtype=np.float64)


def _compute_signed_level_bounds(
    values: FloatArray,
    spacing: float,
    *,
    max_level_fraction: float | None,
    tolerance: float,
) -> tuple[float, float] | None:
    scalar_values = np.asarray(values, dtype=np.float64)
    data_min = float(np.min(scalar_values))
    data_max = float(np.max(scalar_values))
    if data_max - data_min <= spacing + tolerance:
        return None
    lower = data_min + 0.5 * spacing
    upper = data_max - 0.5 * spacing
    if max_level_fraction is not None:
        center = 0.5 * (data_min + data_max)
        half_extent = 0.5 * (data_max - data_min) * max_level_fraction
        lower = max(lower, center - half_extent)
        upper = min(upper, center + half_extent)
    if upper < lower - tolerance:
        return None
    return lower, upper


def _compute_unsigned_level_bounds(
    values: FloatArray,
    spacing: float,
    *,
    max_level_fraction: float | None,
    tolerance: float,
) -> tuple[float, float] | None:
    scalar_values = np.asarray(values, dtype=np.float64)
    data_min = float(np.min(scalar_values))
    data_max = float(np.max(scalar_values))
    if data_max - data_min <= spacing + tolerance:
        return None
    lower = max(data_min + 0.5 * spacing, 0.5 * spacing)
    upper = data_max - 0.5 * spacing
    if max_level_fraction is not None:
        upper = min(upper, data_max * max_level_fraction)
    if upper < lower - tolerance:
        return None
    return lower, upper


def adjust_terminal_parallel_widths(
    items: list[HybridPolyline],
    *,
    spacing: float,
    backend: str,
    values: FloatArray,
    max_level_fraction: float | None,
    tolerance: float,
) -> tuple[list[HybridPolyline], dict[str, float]]:
    if not items:
        return items, {}

    normalized_backend = backend.strip().lower()
    if normalized_backend in {"signed_heat", "vector_heat_stripe"}:
        bounds = _compute_signed_level_bounds(
            values,
            spacing,
            max_level_fraction=max_level_fraction,
            tolerance=tolerance,
        )
    elif normalized_backend == "fast_marching_unsigned":
        bounds = _compute_unsigned_level_bounds(
            values,
            spacing,
            max_level_fraction=max_level_fraction,
            tolerance=tolerance,
        )
    else:
        return items, {}

    if bounds is None:
        return items, {}

    lower_bound, upper_bound = bounds
    level_values = np.asarray([float(item.level) for item in items], dtype=np.float64)
    max_level = float(np.max(level_values))
    lower_remainder = 0.0
    raw_upper_remainder = max(0.0, upper_bound - max_level)
    upper_remainder = float(np.fmod(raw_upper_remainder, float(spacing))) if raw_upper_remainder > tolerance else 0.0
    if upper_remainder <= tolerance and raw_upper_remainder > tolerance:
        upper_remainder = min(float(spacing), raw_upper_remainder)
    diagnostics: dict[str, float] = {
        "parallel_lower_terminal_remainder": float(lower_remainder),
        "parallel_upper_terminal_remainder_raw": float(raw_upper_remainder),
        "parallel_upper_terminal_remainder": float(upper_remainder),
    }

    for item in items:
        nominal_width = float(spacing)
        item.effective_width = nominal_width
        if abs(item.level - max_level) <= tolerance and upper_remainder > tolerance:
            item.effective_width = max(float(item.effective_width), nominal_width + upper_remainder)

    return items, diagnostics


def compute_parallel_wall_clearance_diagnostics(
    boundary_items: list[HybridPolyline],
    parallel_items: list[HybridPolyline],
    *,
    tolerance: float,
) -> dict[str, float]:
    if not boundary_items or not parallel_items:
        return {}

    boundary_levels = np.asarray([float(item.level) for item in boundary_items], dtype=np.float64)
    if boundary_levels.size == 0:
        return {}
    innermost_level = float(np.max(boundary_levels))

    wall_polylines = [
        np.asarray(item.points, dtype=np.float64)
        for item in boundary_items
        if np.asarray(item.points, dtype=np.float64).shape[0] > 0
        and abs(float(item.level) - innermost_level) <= float(tolerance)
    ]
    if not wall_polylines:
        return {}
    level_values = np.asarray([float(item.level) for item in parallel_items], dtype=np.float64)
    min_level = float(np.min(level_values))
    max_level = float(np.max(level_values))
    diagnostics: dict[str, float] = {}

    def point_to_segments_distance(point: np.ndarray, polyline: np.ndarray) -> float:
        pts = np.asarray(polyline, dtype=np.float64)
        if pts.shape[0] == 1:
            return float(np.linalg.norm(point - pts[0]))
        seg_start = pts[:-1]
        seg_end = pts[1:]
        seg_vec = seg_end - seg_start
        seg_len_sq = np.einsum("ij,ij->i", seg_vec, seg_vec)
        point_vec = point[None, :] - seg_start
        t = np.zeros_like(seg_len_sq)
        valid = seg_len_sq > 1e-12
        if np.any(valid):
            t[valid] = np.einsum("ij,ij->i", point_vec[valid], seg_vec[valid]) / seg_len_sq[valid]
        t = np.clip(t, 0.0, 1.0)
        closest = seg_start + t[:, None] * seg_vec
        return float(np.min(np.linalg.norm(closest - point[None, :], axis=1)))

    def point_to_wall_distance(point: np.ndarray) -> float:
        best = np.inf
        for polyline in wall_polylines:
            best = min(best, point_to_segments_distance(point, polyline))
        return float(best)

    for label, target_level in (("first", min_level), ("last", max_level)):
        family_points = [
            np.asarray(item.points, dtype=np.float64)
            for item in parallel_items
            if np.asarray(item.points, dtype=np.float64).shape[0] > 0 and abs(float(item.level) - target_level) <= 1e-9
        ]
        if not family_points:
            continue
        family_cloud = np.vstack(family_points)
        if family_cloud.shape[0] == 0:
            continue
        distances = np.asarray(
            [point_to_wall_distance(point) for point in family_cloud],
            dtype=np.float64,
        ).reshape(-1)
        diagnostics[f"{label}_parallel_wall_gap_min"] = float(np.min(distances))
        diagnostics[f"{label}_parallel_wall_gap_max"] = float(np.max(distances))
        diagnostics[f"{label}_parallel_wall_gap_mean"] = float(np.mean(distances))
        diagnostics[f"{label}_parallel_wall_gap_std"] = float(np.std(distances))
        diagnostics[f"{label}_parallel_wall_gap_count"] = float(distances.shape[0])

    return diagnostics


def apply_boundary_gap_parallel_widths(
    parallel_items: list[HybridPolyline],
    wall_gap_diagnostics: dict[str, float],
    *,
    nominal_width: float,
    tolerance: float,
) -> tuple[list[HybridPolyline], dict[str, float]]:
    if not parallel_items:
        return parallel_items, {}

    level_values = np.asarray([float(item.level) for item in parallel_items], dtype=np.float64)
    if level_values.size == 0:
        return parallel_items, {}
    min_level = float(np.min(level_values))
    max_level = float(np.max(level_values))

    first_gap = wall_gap_diagnostics.get("first_parallel_wall_gap_mean")
    last_gap = wall_gap_diagnostics.get("last_parallel_wall_gap_mean")
    diagnostics: dict[str, float] = {}

    if first_gap is not None:
        first_width = 0.5 * (float(nominal_width) + float(first_gap))
        diagnostics["first_parallel_effective_width_from_gap"] = float(first_width)
        for item in parallel_items:
            if abs(float(item.level) - min_level) <= tolerance:
                item.effective_width = float(first_width)

    if last_gap is not None:
        last_width = 0.5 * (float(nominal_width) + float(last_gap))
        diagnostics["last_parallel_effective_width_from_gap"] = float(last_width)
        for item in parallel_items:
            if abs(float(item.level) - max_level) <= tolerance:
                item.effective_width = float(last_width)

    return parallel_items, diagnostics


def _point_to_segments_distance(point: np.ndarray, polyline: np.ndarray) -> float:
    pts = np.asarray(polyline, dtype=np.float64)
    if pts.shape[0] <= 0:
        return float("inf")
    if pts.shape[0] == 1:
        return float(np.linalg.norm(point - pts[0]))
    seg_start = pts[:-1]
    seg_end = pts[1:]
    seg_vec = seg_end - seg_start
    seg_len_sq = np.einsum("ij,ij->i", seg_vec, seg_vec)
    point_vec = point[None, :] - seg_start
    t = np.zeros_like(seg_len_sq)
    valid = seg_len_sq > 1e-12
    if np.any(valid):
        t[valid] = np.einsum("ij,ij->i", point_vec[valid], seg_vec[valid]) / seg_len_sq[valid]
    t = np.clip(t, 0.0, 1.0)
    closest = seg_start + t[:, None] * seg_vec
    return float(np.min(np.linalg.norm(closest - point[None, :], axis=1)))


def _mean_cloud_to_polyline_set_distance(points: np.ndarray, polylines: list[np.ndarray]) -> float:
    distances = _cloud_to_polyline_set_distances(points, polylines)
    return float(np.mean(distances)) if distances.size > 0 else 0.0


def _cloud_to_polyline_set_distances(points: np.ndarray, polylines: list[np.ndarray]) -> np.ndarray:
    cloud = np.asarray(points, dtype=np.float64)
    valid_polylines = [np.asarray(polyline, dtype=np.float64) for polyline in polylines if np.asarray(polyline).shape[0] > 0]
    if cloud.shape[0] == 0 or not valid_polylines:
        return np.empty((0,), dtype=np.float64)
    return np.asarray(
        [
            min(_point_to_segments_distance(point, polyline) for polyline in valid_polylines)
            for point in cloud
        ],
        dtype=np.float64,
    )


def apply_top_layer_parallel_width_check(
    parallel_items: list[HybridPolyline],
    *,
    nominal_width: float,
    max_factor: float,
    tolerance: float,
) -> tuple[list[HybridPolyline], dict[str, float]]:
    if not parallel_items:
        return parallel_items, {}

    level_values = np.asarray([float(item.level) for item in parallel_items], dtype=np.float64)
    unique_levels = np.asarray(sorted({float(level) for level in level_values}), dtype=np.float64)
    if unique_levels.size < 3:
        return parallel_items, {}

    family_polylines: dict[float, list[np.ndarray]] = {}
    for level in unique_levels:
        family_polylines[float(level)] = [
            np.asarray(item.points, dtype=np.float64)
            for item in parallel_items
            if abs(float(item.level) - float(level)) <= tolerance and np.asarray(item.points, dtype=np.float64).shape[0] > 0
        ]

    adjusted_widths: list[float] = []
    adjusted_families = 0
    width_cap = float(max_factor) * float(nominal_width)
    for idx in range(1, unique_levels.size - 1):
        level = float(unique_levels[idx])
        current_polylines = family_polylines.get(level, [])
        if not current_polylines:
            continue
        for item in parallel_items:
            if abs(float(item.level) - level) > tolerance:
                continue
            points = np.asarray(item.points, dtype=np.float64)
            prev_distances = _cloud_to_polyline_set_distances(points, family_polylines.get(float(unique_levels[idx - 1]), []))
            next_distances = _cloud_to_polyline_set_distances(points, family_polylines.get(float(unique_levels[idx + 1]), []))
            if prev_distances.shape[0] != points.shape[0] or next_distances.shape[0] != points.shape[0]:
                continue
            point_widths = np.minimum(width_cap, 0.5 * (prev_distances + next_distances))
            point_widths = np.where(point_widths > tolerance, point_widths, float(nominal_width))
            item.point_widths = np.asarray(point_widths, dtype=np.float64)
            item.effective_width = float(np.mean(point_widths))
            adjusted_widths.extend(point_widths.tolist())
            adjusted_families += 1

    diagnostics: dict[str, float] = {
        "top_layer_width_check_family_count": float(adjusted_families),
        "top_layer_width_check_nominal_width": float(nominal_width),
        "top_layer_width_check_max_factor": float(max_factor),
    }
    if adjusted_widths:
        adjusted_arr = np.asarray(adjusted_widths, dtype=np.float64)
        diagnostics["top_layer_width_check_applied_min"] = float(np.min(adjusted_arr))
        diagnostics["top_layer_width_check_applied_max"] = float(np.max(adjusted_arr))
        diagnostics["top_layer_width_check_applied_mean"] = float(np.mean(adjusted_arr))
    return parallel_items, diagnostics


def solve_parallel_field(
    V: FloatArray,
    F: np.ndarray,
    axis_name: str,
    axis_index: int,
    plane_value: float,
    *,
    backend: str,
    level_set_constraint: str,
    soft_level_set_weight: float,
    tolerance: float,
) -> tuple[FloatArray, list[list[CurvePoint]], list[tuple[CurvePoint, CurvePoint]]]:
    normalized_backend = backend.strip().lower()
    if normalized_backend == "vector_heat_stripe":
        stripe_result = generate_vector_heat_stripe_field(
            V,
            F,
            plane_axis=axis_name,
            plane_value=plane_value,
            spacing=1.0,
            max_level_fraction=1.0,
            constraint_mode="constrained",
            source_vertex_mode="intersected_faces",
            tolerance=tolerance,
        )
        return (
            np.asarray(stripe_result["scalar_field"], dtype=np.float64),
            stripe_result["source_curves"],
            stripe_result["source_segments"],
        )

    if pp3d is None:
        raise ImportError("This script requires the optional `potpourri3d` package.")

    segments = intersect_mesh_with_axis_plane(
        V,
        F,
        axis_index,
        plane_value,
        tolerance=tolerance,
    )
    if not segments:
        raise RuntimeError("Plane/mesh intersection produced no segments.")

    curves = stitch_segments_into_curves(segments, tolerance=tolerance)
    if not curves:
        raise RuntimeError("Failed to stitch midpoint plane segments into source curves.")

    curve_payloads = build_solver_curve_payloads(curves, segments, tolerance=tolerance)
    if normalized_backend == "signed_heat":
        solver = pp3d.MeshSignedHeatSolver(V, F)
        signed_flags = [True] * len(curve_payloads)
        distances = solver.compute_distance(
            curve_payloads,
            signed_flags,
            [],
            preserve_source_normals=False,
            level_set_constraint=level_set_constraint,
            soft_level_set_weight=soft_level_set_weight,
        )
    elif normalized_backend == "fast_marching_unsigned":
        solver = pp3d.MeshFastMarchingDistanceSolver(V, F)
        source_distances = [[0.0] * len(curve) for curve in curve_payloads]
        distances = solver.compute_distance(curve_payloads, source_distances, False)
    else:
        raise ValueError(
            "parallel_field_backend must be one of 'signed_heat', 'fast_marching_unsigned', or 'vector_heat_stripe'.",
        )
    return np.asarray(distances, dtype=np.float64), curves, segments


def build_signed_levels(
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


def build_unsigned_levels(
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

    lower = max(data_min + 0.5 * spacing, 0.5 * spacing)
    upper = data_max - 0.5 * spacing
    if max_level_fraction is not None:
        upper = min(upper, data_max * max_level_fraction)
    if upper < lower - tolerance:
        return np.array([], dtype=np.float64)

    levels: list[float] = []
    level = lower
    while level <= upper + tolerance:
        levels.append(float(level))
        level += spacing
    return np.asarray(levels, dtype=np.float64)


def build_parallel_triangle_cache(
    V: FloatArray,
    F: np.ndarray,
    parallel_field: FloatArray,
    boundary_field: FloatArray,
) -> ParallelTriangleCache:
    vertices = np.asarray(V, dtype=np.float64)
    faces = np.asarray(F, dtype=np.int64)
    parallel_values = np.asarray(parallel_field, dtype=np.float64)
    boundary_values = np.asarray(boundary_field, dtype=np.float64)

    tri_points = vertices[faces]
    tri_parallel = parallel_values[faces]
    tri_boundary = boundary_values[faces]
    return ParallelTriangleCache(
        tri_points=tri_points,
        tri_parallel=tri_parallel,
        tri_boundary=tri_boundary,
        tri_parallel_min=np.min(tri_parallel, axis=1),
        tri_parallel_max=np.max(tri_parallel, axis=1),
    )


def build_boundary_contours(
    V: FloatArray,
    F: np.ndarray,
    boundary_field: FloatArray,
    spacing: float,
    boundary_count: int,
    *,
    tolerance: float,
) -> tuple[list[FloatArray], list[ContourLevel]]:
    if boundary_count < 1:
        raise ValueError("boundary_count must be at least 1.")
    boundary_loops = generate_boundary_loops(V, F)
    contour_levels: list[ContourLevel] = []
    for level_idx in range(1, boundary_count):
        level = float(level_idx * spacing)
        polylines = extract_isocontour_polylines(
            V,
            F,
            boundary_field,
            level,
            tolerance=tolerance,
        )
        contour_levels.append(ContourLevel(level=level, polylines=polylines))
    return boundary_loops, contour_levels


def _interpolate_edge_sample(
    tri_points: FloatArray,
    tri_parallel: FloatArray,
    tri_boundary: FloatArray,
    face_index: int,
    edge_start: int,
    edge_end: int,
    level: float,
    *,
    tolerance: float,
) -> HybridSample | None:
    s0 = float(tri_parallel[edge_start])
    s1 = float(tri_parallel[edge_end])

    if abs(s0 - level) <= tolerance and abs(s1 - level) <= tolerance:
        return None
    if abs(s0 - level) <= tolerance:
        bary = np.zeros(3, dtype=np.float64)
        bary[edge_start] = 1.0
        return HybridSample(
            xyz=np.asarray(tri_points[edge_start], dtype=np.float64),
            face_index=face_index,
            barycentric=bary,
            boundary_value=float(tri_boundary[edge_start]),
        )
    if abs(s1 - level) <= tolerance:
        bary = np.zeros(3, dtype=np.float64)
        bary[edge_end] = 1.0
        return HybridSample(
            xyz=np.asarray(tri_points[edge_end], dtype=np.float64),
            face_index=face_index,
            barycentric=bary,
            boundary_value=float(tri_boundary[edge_end]),
        )
    if (s0 - level) * (s1 - level) > 0.0:
        return None

    t = (level - s0) / (s1 - s0)
    if t < -tolerance or t > 1.0 + tolerance:
        return None
    t = min(1.0, max(0.0, t))
    bary = np.zeros(3, dtype=np.float64)
    bary[edge_start] = 1.0 - t
    bary[edge_end] = t
    xyz = (1.0 - t) * tri_points[edge_start] + t * tri_points[edge_end]
    boundary_value = (1.0 - t) * float(tri_boundary[edge_start]) + t * float(tri_boundary[edge_end])
    return HybridSample(
        xyz=np.asarray(xyz, dtype=np.float64),
        face_index=face_index,
        barycentric=bary,
        boundary_value=float(boundary_value),
    )


def _dedupe_hybrid_samples(samples: list[HybridSample], tolerance: float) -> list[HybridSample]:
    deduped: list[HybridSample] = []
    for sample in samples:
        if not any(np.linalg.norm(sample.xyz - existing.xyz) <= tolerance for existing in deduped):
            deduped.append(sample)
    return deduped


def extract_parallel_segments_with_boundary_values(
    cache: ParallelTriangleCache,
    level: float,
    *,
    tolerance: float,
) -> list[HybridSegment]:
    segments: list[HybridSegment] = []
    candidate_mask = (cache.tri_parallel_min <= level + tolerance) & (cache.tri_parallel_max >= level - tolerance)
    candidate_face_indices = np.flatnonzero(candidate_mask)
    for face_index in candidate_face_indices:
        tri_points = cache.tri_points[face_index]
        tri_parallel = cache.tri_parallel[face_index]
        tri_boundary = cache.tri_boundary[face_index]
        intersections: list[HybridSample] = []
        for edge_start, edge_end in EDGE_PAIRS:
            sample = _interpolate_edge_sample(
                tri_points,
                tri_parallel,
                tri_boundary,
                int(face_index),
                edge_start,
                edge_end,
                level,
                tolerance=tolerance,
            )
            if sample is not None:
                intersections.append(sample)

        intersections = _dedupe_hybrid_samples(intersections, tolerance)
        if len(intersections) == 2:
            segments.append(HybridSegment(start=intersections[0], end=intersections[1], parallel_level=float(level)))
        elif len(intersections) > 2:
            max_pair: tuple[HybridSample, HybridSample] | None = None
            max_distance = -1.0
            for idx, first in enumerate(intersections):
                for second in intersections[idx + 1 :]:
                    distance = float(np.linalg.norm(first.xyz - second.xyz))
                    if distance > max_distance:
                        max_distance = distance
                        max_pair = (first, second)
            if max_pair is not None and max_distance > tolerance:
                segments.append(HybridSegment(start=max_pair[0], end=max_pair[1], parallel_level=float(level)))
    return segments


def _interpolate_hybrid_sample(start: HybridSample, end: HybridSample, t: float) -> HybridSample:
    return HybridSample(
        xyz=(1.0 - t) * start.xyz + t * end.xyz,
        face_index=start.face_index,
        barycentric=(1.0 - t) * start.barycentric + t * end.barycentric,
        boundary_value=(1.0 - t) * start.boundary_value + t * end.boundary_value,
    )


def clip_parallel_segments(
    segments: list[HybridSegment],
    boundary_cutoff: float,
    *,
    tolerance: float,
) -> list[tuple[FloatArray, FloatArray]]:
    kept: list[tuple[FloatArray, FloatArray]] = []
    for segment in segments:
        start_inside = segment.start.boundary_value > boundary_cutoff + tolerance
        end_inside = segment.end.boundary_value > boundary_cutoff + tolerance

        if start_inside and end_inside:
            kept.append((segment.start.xyz, segment.end.xyz))
            continue
        if (not start_inside) and (not end_inside):
            continue

        denom = segment.end.boundary_value - segment.start.boundary_value
        if abs(denom) <= tolerance:
            continue
        t = (boundary_cutoff - segment.start.boundary_value) / denom
        t = min(1.0, max(0.0, t))
        crossing = _interpolate_hybrid_sample(segment.start, segment.end, t)
        if start_inside:
            kept.append((segment.start.xyz, crossing.xyz))
        else:
            kept.append((crossing.xyz, segment.end.xyz))
    return kept


def preview_hybrid(
    V: FloatArray,
    F: np.ndarray,
    field_name: str,
    field_values: FloatArray,
    source_curves: list[list[CurvePoint]],
    boundary_loops: list[FloatArray],
    boundary_levels: list[ContourLevel],
    parallel_polylines: list[FloatArray],
    chained_path: FloatArray | None,
    *,
    show_mesh_edges: bool,
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

    for loop in boundary_loops:
        loop_poly = build_polyline_polydata(
            np.asarray(loop, dtype=np.float64),
            close_loop=bool(np.linalg.norm(loop[0] - loop[-1]) <= tolerance),
            tolerance=tolerance,
        )
        plotter.add_mesh(loop_poly, color="red", line_width=5.0)

    for level in boundary_levels:
        for polyline in level.polylines:
            if np.asarray(polyline).shape[0] < 2:
                continue
            poly = build_polyline_polydata(
                np.asarray(polyline, dtype=np.float64),
                close_loop=bool(np.linalg.norm(polyline[0] - polyline[-1]) <= tolerance),
                tolerance=tolerance,
            )
            plotter.add_mesh(poly, color="red", line_width=3.0)

    for polyline in parallel_polylines:
        if np.asarray(polyline).shape[0] < 2:
            continue
        poly = build_polyline_polydata(
            np.asarray(polyline, dtype=np.float64),
            close_loop=bool(np.linalg.norm(polyline[0] - polyline[-1]) <= tolerance),
            tolerance=tolerance,
        )
        plotter.add_mesh(poly, color="white", line_width=3.0)

    if chained_path is not None and np.asarray(chained_path).shape[0] >= 2:
        chained_poly = build_polyline_polydata(np.asarray(chained_path, dtype=np.float64), close_loop=False, tolerance=tolerance)
        plotter.add_mesh(chained_poly, color="yellow", line_width=5.0)

    plotter.show()


def generate_hybrid_toolpath(
    V: FloatArray,
    F: np.ndarray,
    *,
    spacing: float,
    resample_spacing: float | None = None,
    parallel_spacing: float | None = None,
    parallel_resample_spacing: float | None = None,
    boundary_count: int,
    boundary_field_backend: str = "native",
    parallel_field_backend: str = "signed_heat",
    axis_mode: str = "fixed",
    axis: str = "x",
    layer_main_index: int = 0,
    plane_value: float | None = None,
    level_set_constraint: str = "None",
    soft_level_set_weight: float = -1.0,
    contour_max_level_fraction: float | None = 1.0,
    min_polyline_points: int = 2,
    sharp_angle_deg: float = 35.0,
    mild_smooth_all: bool = False,
    mild_smooth_iterations: int = 8,
    mild_smooth_alpha: float = 0.2,
    start_selection_mode: str = "first",
    boundary_start_offset_count: int = 0,
    parallel_direction_mode: str = "connect_nearest",
    disable_parallel: bool = False,
    top_layer_width_check_enabled: bool = False,
    top_layer_width_check_max_factor: float = 2.0,
    is_top_layer: bool = False,
    tolerance: float = 1e-8,
) -> dict[str, object]:
    if start_selection_mode == "lowest_z":
        start_selection_mode = "lowest_y"
    if start_selection_mode not in {"first", "lowest_y"}:
        raise ValueError("start_selection_mode must be either 'first' or 'lowest_y'.")
    if boundary_start_offset_count < 0:
        raise ValueError("boundary_start_offset_count must be non-negative.")
    if parallel_direction_mode not in {"connect_nearest", "one_direction"}:
        raise ValueError("parallel_direction_mode must be either 'connect_nearest' or 'one_direction'.")
    effective_parallel_spacing = float(parallel_spacing) if parallel_spacing is not None else float(spacing)
    effective_resample_spacing = float(resample_spacing) if resample_spacing is not None else float(spacing)
    effective_parallel_resample_spacing = (
        float(parallel_resample_spacing)
        if parallel_resample_spacing is not None
        else effective_resample_spacing
    )
    selected_axis, dominant_normal_axis, tangent_axes = choose_hybrid_parallel_axis(
        V,
        F,
        axis_mode=axis_mode,
        fixed_axis=axis,
        layer_main_index=int(layer_main_index),
    )
    axis_index = AXIS_TO_INDEX[selected_axis]
    effective_plane_value = float(plane_value) if plane_value is not None else compute_midpoint_plane_value(V, axis_index)

    boundary_field, boundary_loops, boundary_source_curves, boundary_source_segments = solve_hybrid_boundary_field(
        V,
        F,
        backend=boundary_field_backend,
        tolerance=float(tolerance),
    )

    normalized_parallel_backend = parallel_field_backend.strip().lower()
    if disable_parallel:
        parallel_field = np.zeros(np.asarray(V, dtype=np.float64).shape[0], dtype=np.float64)
        source_curves = []
        source_segments = []
    else:
        parallel_field, source_curves, source_segments = solve_parallel_field(
            V,
            F,
            selected_axis,
            axis_index,
            effective_plane_value,
            backend=parallel_field_backend,
            level_set_constraint=level_set_constraint,
            soft_level_set_weight=float(soft_level_set_weight),
            tolerance=float(tolerance),
        )

    _, boundary_levels = build_boundary_contours(
        V,
        F,
        boundary_field,
        float(spacing),
        int(boundary_count),
        tolerance=float(tolerance),
    )
    boundary_cutoff = int(boundary_count) * float(spacing)

    if disable_parallel:
        parallel_levels = np.asarray([], dtype=np.float64)
    elif normalized_parallel_backend in {"signed_heat", "vector_heat_stripe"}:
        parallel_levels = build_signed_levels(
            parallel_field,
            effective_parallel_spacing,
            max_level_fraction=contour_max_level_fraction,
            tolerance=float(tolerance),
        )
    elif normalized_parallel_backend == "fast_marching_unsigned":
        parallel_levels = build_unsigned_levels(
            parallel_field,
            effective_parallel_spacing,
            max_level_fraction=contour_max_level_fraction,
            tolerance=float(tolerance),
        )
    else:
        raise ValueError(
            "parallel_field_backend must be one of 'signed_heat', 'fast_marching_unsigned', or 'vector_heat_stripe'.",
        )
    parallel_triangle_cache = None
    if not disable_parallel:
        parallel_triangle_cache = build_parallel_triangle_cache(
            V,
            F,
            parallel_field,
            boundary_field,
        )
    kept_parallel_segments: list[tuple[FloatArray, FloatArray]] = []
    stitched_parallel_polylines: list[FloatArray] = []
    level_by_polyline: list[float] = []
    if not disable_parallel and parallel_triangle_cache is not None:
        for level in parallel_levels:
            raw_segments = extract_parallel_segments_with_boundary_values(
                parallel_triangle_cache,
                float(level),
                tolerance=float(tolerance),
            )
            clipped_segments = clip_parallel_segments(raw_segments, boundary_cutoff, tolerance=float(tolerance))
            kept_parallel_segments.extend(clipped_segments)
            level_polylines = stitch_segments_into_fragment_polylines(clipped_segments, tolerance=float(tolerance))
            for polyline in level_polylines:
                points = np.asarray(polyline, dtype=np.float64)
                if points.shape[0] < int(min_polyline_points):
                    continue
                stitched_parallel_polylines.append(points)
                level_by_polyline.append(float(level))

    stitched_parallel_polylines = [
        polyline
        for polyline in stitched_parallel_polylines
        if np.asarray(polyline).shape[0] >= int(min_polyline_points)
    ]
    stitched_parallel_polylines = smooth_polylines(
        stitched_parallel_polylines,
        enabled=bool(mild_smooth_all),
        iterations=int(mild_smooth_iterations),
        alpha=float(mild_smooth_alpha),
        tolerance=float(tolerance),
    )

    smoothed_boundary_loops = smooth_polylines(
        boundary_loops,
        enabled=bool(mild_smooth_all),
        iterations=int(mild_smooth_iterations),
        alpha=float(mild_smooth_alpha),
        tolerance=float(tolerance),
    )
    smoothed_boundary_levels: list[ContourLevel] = []
    for level in boundary_levels:
        smoothed_boundary_levels.append(
            ContourLevel(
                level=level.level,
                polylines=smooth_polylines(
                    level.polylines,
                    enabled=bool(mild_smooth_all),
                    iterations=int(mild_smooth_iterations),
                    alpha=float(mild_smooth_alpha),
                    tolerance=float(tolerance),
                ),
            ),
        )

    boundary_items = resample_boundary_family(
        smoothed_boundary_loops,
        smoothed_boundary_levels,
        effective_resample_spacing,
        nominal_width=float(spacing),
        sharp_angle_deg=float(sharp_angle_deg),
        tolerance=float(tolerance),
    )
    boundary_items = filter_boundary_items_by_start_offset(
        boundary_items,
        spacing=float(spacing),
        boundary_start_offset_count=int(boundary_start_offset_count),
        tolerance=float(tolerance),
    )
    parallel_items = resample_parallel_family(
        stitched_parallel_polylines,
        level_by_polyline,
        effective_parallel_resample_spacing,
        nominal_width=effective_parallel_spacing,
        sharp_angle_deg=float(sharp_angle_deg),
        tolerance=float(tolerance),
    )
    parallel_items, width_diagnostics = adjust_terminal_parallel_widths(
        parallel_items,
        spacing=effective_parallel_spacing,
        backend=normalized_parallel_backend,
        values=parallel_field,
        max_level_fraction=contour_max_level_fraction,
        tolerance=float(tolerance),
    )
    wall_gap_diagnostics = compute_parallel_wall_clearance_diagnostics(
        boundary_items,
        parallel_items,
        tolerance=float(tolerance),
    )
    parallel_items, boundary_gap_width_diagnostics = apply_boundary_gap_parallel_widths(
        parallel_items,
        wall_gap_diagnostics,
        nominal_width=effective_parallel_spacing,
        tolerance=float(tolerance),
    )
    top_layer_width_diagnostics: dict[str, float] = {}
    if top_layer_width_check_enabled:
        parallel_items, top_layer_width_diagnostics = apply_top_layer_parallel_width_check(
            parallel_items,
            nominal_width=effective_parallel_spacing,
            max_factor=float(top_layer_width_check_max_factor),
            tolerance=float(tolerance),
        )
    parallel_handoff_targets = collect_polyline_handoff_targets(
        parallel_items,
        tolerance=float(tolerance),
    )

    chained_boundary_parts, chained_boundary_widths, chain_endpoint = chain_boundary_family(
        boundary_items,
        start_selection_mode=start_selection_mode,
        handoff_targets=parallel_handoff_targets,
        tolerance=float(tolerance),
    )
    chained_parallel_parts, chained_parallel_widths, chain_endpoint = chain_parallel_family(
        parallel_items,
        chain_endpoint,
        direction_mode=parallel_direction_mode,
        tolerance=float(tolerance),
    )
    all_parts = chained_boundary_parts + chained_parallel_parts
    all_width_parts = chained_boundary_widths + chained_parallel_widths
    chained_hybrid_path = concatenate_paths(all_parts)
    chained_hybrid_widths = concatenate_width_profiles(all_parts, all_width_parts)
    chained_hybrid_tags = concatenate_path_tags(
        all_parts,
        ["top_boundary_loop" if is_top_layer else "boundary_loop"] * len(chained_boundary_parts)
        + ["top_inner_raster" if is_top_layer else "inner_raster"] * len(chained_parallel_parts),
        tolerance=float(tolerance),
    )

    resampled_parallel_polylines = [item.points for item in parallel_items]
    parallel_polyline_levels = [float(item.level) for item in parallel_items]
    resampled_boundary_loops = [item.points for item in boundary_items if abs(item.level) <= float(tolerance)]
    resampled_boundary_levels: list[ContourLevel] = []
    boundary_level_map: dict[float, list[FloatArray]] = {}
    for item in boundary_items:
        if abs(item.level) <= float(tolerance):
            continue
        boundary_level_map.setdefault(float(item.level), []).append(item.points)
    for level in sorted(boundary_level_map):
        resampled_boundary_levels.append(ContourLevel(level=level, polylines=boundary_level_map[level]))

    return {
        "path_points": chained_hybrid_path,
        "path_widths": chained_hybrid_widths,
        "path_tags": chained_hybrid_tags,
        "boundary_field": boundary_field,
        "boundary_field_backend": boundary_field_backend.strip().lower(),
        "parallel_field": parallel_field,
        "parallel_field_backend": normalized_parallel_backend,
        "axis_mode": axis_mode,
        "parallel_axis": selected_axis,
        "dominant_normal_axis": dominant_normal_axis,
        "tangent_axes": list(tangent_axes) if tangent_axes is not None else None,
        "source_curves": source_curves,
        "source_segments": source_segments,
        "boundary_source_curves": boundary_source_curves,
        "boundary_source_segments": boundary_source_segments,
        "boundary_loops": resampled_boundary_loops,
        "boundary_levels": resampled_boundary_levels,
        "parallel_polylines": resampled_parallel_polylines,
        "parallel_polyline_levels": parallel_polyline_levels,
        "parallel_levels": parallel_levels,
        "boundary_cutoff": boundary_cutoff,
        "plane_value": effective_plane_value,
        "parallel_spacing": effective_parallel_spacing,
        "resample_spacing": effective_resample_spacing,
        "parallel_resample_spacing": effective_parallel_resample_spacing,
        "boundary_start_offset_count": int(boundary_start_offset_count),
        "parallel_direction_mode": parallel_direction_mode,
        "parallel_disabled": bool(disable_parallel),
        "sharp_angle_deg": float(sharp_angle_deg),
        "mild_smooth_all": bool(mild_smooth_all),
        "mild_smooth_iterations": int(mild_smooth_iterations),
        "mild_smooth_alpha": float(mild_smooth_alpha),
        "kept_parallel_segments": kept_parallel_segments,
        "chained_boundary_parts": chained_boundary_parts,
        "chained_parallel_parts": chained_parallel_parts,
        "width_diagnostics": width_diagnostics,
        "wall_gap_diagnostics": wall_gap_diagnostics,
        "boundary_gap_width_diagnostics": boundary_gap_width_diagnostics,
        "top_layer_width_diagnostics": top_layer_width_diagnostics,
        "top_layer_width_check_enabled": bool(top_layer_width_check_enabled),
        "is_top_layer": bool(is_top_layer),
        "top_layer_width_check_max_factor": float(top_layer_width_check_max_factor),
    }


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    V, F = load_obj_mesh(str(args.mesh_path))
    result = generate_hybrid_toolpath(
        V,
        F,
        spacing=float(args.spacing),
        resample_spacing=None if args.resample_spacing is None else float(args.resample_spacing),
        boundary_count=int(args.boundary_count),
        boundary_field_backend=args.boundary_field_backend,
        sharp_angle_deg=float(args.sharp_angle_deg),
        parallel_field_backend=args.parallel_field_backend,
        axis_mode=args.axis_mode,
        axis=args.axis,
        layer_main_index=0,
        plane_value=args.plane_value,
        level_set_constraint=args.level_set_constraint,
        soft_level_set_weight=float(args.soft_level_set_weight),
        contour_max_level_fraction=args.contour_max_level_fraction,
        min_polyline_points=int(args.min_polyline_points),
        mild_smooth_all=False,
        mild_smooth_iterations=8,
        mild_smooth_alpha=0.2,
        tolerance=float(args.tolerance),
    )
    boundary_field = result["boundary_field"]
    boundary_field_backend = result["boundary_field_backend"]
    parallel_field = result["parallel_field"]
    source_curves = result["source_curves"]
    source_segments = result["source_segments"]
    resampled_boundary_loops = result["boundary_loops"]
    resampled_boundary_levels = result["boundary_levels"]
    resampled_parallel_polylines = result["parallel_polylines"]
    parallel_levels = result["parallel_levels"]
    boundary_cutoff = result["boundary_cutoff"]
    kept_parallel_segments = result["kept_parallel_segments"]
    parallel_field_backend = result["parallel_field_backend"]
    selected_axis = result["parallel_axis"]
    dominant_normal_axis = result["dominant_normal_axis"]
    tangent_axes = result["tangent_axes"]
    plane_value = result["plane_value"]
    chained_boundary_parts = result["chained_boundary_parts"]
    chained_parallel_parts = result["chained_parallel_parts"]
    chained_hybrid_path = result["path_points"]

    source_curves_path = output_dir / "source_curves.obj"
    save_curves_obj(source_curves_path, source_curves, tolerance=float(args.tolerance))

    boundary_loops_path = output_dir / "boundary_loops.obj"
    save_polylines_obj(boundary_loops_path, resampled_boundary_loops, tolerance=float(args.tolerance))

    boundary_contours_dir = output_dir / "boundary_contours"
    save_contour_levels_txt(boundary_contours_dir, resampled_boundary_levels)

    parallel_contours_path = output_dir / "parallel_hybrid_contours.obj"
    save_polylines_obj(parallel_contours_path, resampled_parallel_polylines, tolerance=float(args.tolerance))

    chained_path_path = output_dir / "hybrid_chained_path.obj"
    save_polylines_obj(chained_path_path, [chained_hybrid_path], tolerance=float(args.tolerance))
    np.savetxt(output_dir / "hybrid_chained_path.txt", chained_hybrid_path, fmt="%.10f")

    np.savetxt(output_dir / "boundary_field.txt", boundary_field, fmt="%.10f")
    np.savetxt(output_dir / "parallel_field.txt", parallel_field, fmt="%.10f")

    summary = {
        "mesh_path": str(args.mesh_path),
        "boundary_field_backend": boundary_field_backend,
        "parallel_field_backend": parallel_field_backend,
        "axis_mode": args.axis_mode,
        "axis": args.axis,
        "selected_parallel_axis": selected_axis,
        "dominant_normal_axis": dominant_normal_axis,
        "tangent_axes": tangent_axes,
        "plane_value": plane_value,
        "spacing": float(args.spacing),
        "resample_spacing": float(result["resample_spacing"]),
        "sharp_angle_deg": float(result["sharp_angle_deg"]),
        "boundary_count": int(args.boundary_count),
        "boundary_cutoff": float(boundary_cutoff),
        "source_curve_count": len(source_curves),
        "source_segment_count": len(source_segments),
        "boundary_loop_count": len(resampled_boundary_loops),
        "boundary_level_count": len(resampled_boundary_levels),
        "boundary_polyline_count": int(sum(len(level.polylines) for level in resampled_boundary_levels)),
        "parallel_level_count": int(parallel_levels.size),
        "kept_parallel_segment_count": len(kept_parallel_segments),
        "kept_parallel_polyline_count": len(resampled_parallel_polylines),
        "chained_boundary_part_count": len(chained_boundary_parts),
        "chained_parallel_part_count": len(chained_parallel_parts),
        "chained_path_point_count": int(chained_hybrid_path.shape[0]),
        "boundary_field_min": float(np.min(boundary_field)),
        "boundary_field_max": float(np.max(boundary_field)),
        "parallel_field_min": float(np.min(parallel_field)),
        "parallel_field_max": float(np.max(parallel_field)),
        "source_curves_obj": source_curves_path.name,
        "boundary_loops_obj": boundary_loops_path.name,
        "parallel_hybrid_contours_obj": parallel_contours_path.name,
        "hybrid_chained_path_obj": chained_path_path.name,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Source curves: {source_curves_path}")
    print(f"Boundary loops: {boundary_loops_path}")
    print(f"Parallel hybrid contours: {parallel_contours_path}")
    print(f"Hybrid chained path: {chained_path_path}")
    print(f"Summary: {summary_path}")
    print(f"Boundary cutoff: {boundary_cutoff:.6f}")
    print(f"Kept parallel polyline count: {len(resampled_parallel_polylines)}")

    if args.preview:
        field_name = "parallel_field" if args.preview_field == "parallel" else "boundary_field"
        field_values = parallel_field if args.preview_field == "parallel" else boundary_field
        preview_hybrid(
            V,
            F,
            field_name,
            field_values,
            source_curves,
            resampled_boundary_loops,
            resampled_boundary_levels,
            resampled_parallel_polylines,
            chained_hybrid_path,
            show_mesh_edges=bool(args.show_mesh_edges),
            tolerance=float(args.tolerance),
        )


if __name__ == "__main__":
    main()
