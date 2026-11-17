"""
Iso-contour extraction utilities for scalar fields defined on triangle meshes.

The initial focus is equally spaced contours from the heat-based boundary
distance field, but the core functions operate on any per-vertex scalar field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from heat_method_core import FloatArray
from heat_method_core import HeatMethodParams
from heat_method_core import IntArray
from mesh_distance_workflow import find_boundary_edges
from mesh_distance_workflow import stitch_boundary_loops
from mesh_distance_workflow import solve_boundary_distance_field
from toolpath_export import compute_vertex_normals


@dataclass(slots=True)
class ContourLevel:
    level: float
    polylines: list[FloatArray]


@dataclass(slots=True)
class OrientedPolyline:
    points: FloatArray
    normals: FloatArray


@dataclass(slots=True)
class OrientedContourLevel:
    level: float
    polylines: list[OrientedPolyline]


def _normalize_vectors(vectors: FloatArray, *, tolerance: float = 1e-12) -> FloatArray:
    array = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1)
    safe_norms = np.maximum(norms, tolerance)
    return array / safe_norms[:, None]


def _level_in_fraction_window(
    level: float,
    max_scalar: float,
    fraction_from: float | None,
    fraction_to: float | None,
) -> bool:
    if max_scalar <= 0.0:
        return False
    lower = 0.0 if fraction_from is None else fraction_from
    upper = 1.0 if fraction_to is None else fraction_to
    if not (0.0 <= lower <= upper <= 1.0):
        raise ValueError("Smoothing level fractions must satisfy 0 <= from <= to <= 1.")
    normalized = level / max_scalar
    return lower <= normalized <= upper


def smooth_polyline_laplacian(
    polyline: FloatArray,
    *,
    iterations: int = 20,
    alpha: float = 0.6,
    tolerance: float = 1e-8,
) -> FloatArray:
    """Apply iterative Laplacian smoothing to a polyline."""
    points = np.asarray(polyline, dtype=np.float64).copy()
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("polyline must have shape (k, 3).")
    if iterations <= 0 or points.shape[0] < 3:
        return points
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in the interval (0, 1].")

    closed = is_closed_polyline(points, tolerance=tolerance)
    if closed:
        working = points[:-1].copy()
        for _ in range(iterations):
            updated = working.copy()
            for idx in range(working.shape[0]):
                prev_pt = working[(idx - 1) % working.shape[0]]
                next_pt = working[(idx + 1) % working.shape[0]]
                updated[idx] = (1.0 - alpha) * working[idx] + 0.5 * alpha * (prev_pt + next_pt)
            working = updated
        return np.vstack((working, working[0]))

    for _ in range(iterations):
        updated = points.copy()
        for idx in range(1, points.shape[0] - 1):
            updated[idx] = (1.0 - alpha) * points[idx] + 0.5 * alpha * (points[idx - 1] + points[idx + 1])
        points = updated
    return points


def smooth_oriented_polyline_laplacian(
    polyline: OrientedPolyline,
    *,
    iterations: int = 20,
    alpha: float = 0.6,
    tolerance: float = 1e-8,
) -> OrientedPolyline:
    """Apply Laplacian smoothing to points while preserving paired normals."""
    smoothed_points = smooth_polyline_laplacian(
        polyline.points,
        iterations=iterations,
        alpha=alpha,
        tolerance=tolerance,
    )
    normals = _normalize_vectors(polyline.normals, tolerance=tolerance)
    if smoothed_points.shape != normals.shape:
        if is_closed_polyline(polyline.points, tolerance=tolerance) and smoothed_points.shape[0] == normals.shape[0] + 1:
            normals = np.vstack((normals[:-1], normals[0]))
        else:
            raise ValueError("Smoothed point count no longer matches normals.")
    return OrientedPolyline(points=smoothed_points, normals=normals)


def smooth_contour_levels_in_fraction_window(
    contour_levels: list[ContourLevel],
    max_scalar: float,
    *,
    fraction_from: float,
    fraction_to: float,
    iterations: int = 20,
    alpha: float = 0.6,
    tolerance: float = 1e-8,
) -> list[ContourLevel]:
    """Smooth contour levels whose scalar values fall within a fraction window."""
    smoothed: list[ContourLevel] = []
    for level in contour_levels:
        if _level_in_fraction_window(level.level, max_scalar, fraction_from, fraction_to):
            polylines = [
                smooth_polyline_laplacian(polyline, iterations=iterations, alpha=alpha, tolerance=tolerance)
                for polyline in level.polylines
            ]
        else:
            polylines = [np.asarray(polyline, dtype=np.float64).copy() for polyline in level.polylines]
        smoothed.append(ContourLevel(level=level.level, polylines=polylines))
    return smoothed


def smooth_oriented_contour_levels_in_fraction_window(
    contour_levels: list[OrientedContourLevel],
    max_scalar: float,
    *,
    fraction_from: float,
    fraction_to: float,
    iterations: int = 20,
    alpha: float = 0.6,
    tolerance: float = 1e-8,
) -> list[OrientedContourLevel]:
    """Smooth oriented contour levels within a scalar-fraction window."""
    smoothed: list[OrientedContourLevel] = []
    for level in contour_levels:
        if _level_in_fraction_window(level.level, max_scalar, fraction_from, fraction_to):
            polylines = [
                smooth_oriented_polyline_laplacian(
                    polyline,
                    iterations=iterations,
                    alpha=alpha,
                    tolerance=tolerance,
                )
                for polyline in level.polylines
            ]
        else:
            polylines = [
                OrientedPolyline(
                    points=np.asarray(polyline.points, dtype=np.float64).copy(),
                    normals=np.asarray(polyline.normals, dtype=np.float64).copy(),
                )
                for polyline in level.polylines
            ]
        smoothed.append(OrientedContourLevel(level=level.level, polylines=polylines))
    return smoothed


def filter_contour_levels_by_point_count(
    contour_levels: list[ContourLevel],
    *,
    min_points: int = 4,
) -> list[ContourLevel]:
    """Remove contour polylines with fewer than the requested number of points."""
    if min_points < 2:
        raise ValueError("min_points must be at least 2.")

    filtered_levels: list[ContourLevel] = []
    for contour_level in contour_levels:
        kept = [polyline for polyline in contour_level.polylines if polyline.shape[0] >= min_points]
        filtered_levels.append(ContourLevel(level=contour_level.level, polylines=kept))
    return filtered_levels


def is_closed_polyline(polyline: FloatArray, *, tolerance: float = 1e-8) -> bool:
    """Return True if the polyline is closed within the given tolerance."""
    points = np.asarray(polyline, dtype=np.float64)
    if points.shape[0] < 3:
        return False
    return bool(np.linalg.norm(points[0] - points[-1]) <= tolerance)


def _polyline_turn_angle(points: FloatArray, index: int, *, closed: bool, tolerance: float) -> float:
    num_points = points.shape[0]
    prev_idx = (index - 1) % num_points if closed else index - 1
    next_idx = (index + 1) % num_points if closed else index + 1

    prev_vec = points[index] - points[prev_idx]
    next_vec = points[next_idx] - points[index]
    prev_norm = np.linalg.norm(prev_vec)
    next_norm = np.linalg.norm(next_vec)
    if prev_norm <= tolerance or next_norm <= tolerance:
        return 0.0

    prev_dir = prev_vec / prev_norm
    next_dir = next_vec / next_norm
    cosine = float(np.clip(np.dot(prev_dir, next_dir), -1.0, 1.0))
    return float(np.arccos(cosine))


def detect_sharp_vertices(
    polyline: FloatArray,
    *,
    angle_threshold_degrees: float = 35.0,
    tolerance: float = 1e-8,
) -> IntArray:
    """Detect vertices that should be preserved during resampling."""
    points = np.asarray(polyline, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("polyline must have shape (k, 3).")
    if points.shape[0] < 2:
        return np.array([], dtype=np.int64)

    closed = is_closed_polyline(points, tolerance=tolerance)
    working = points[:-1] if closed else points
    threshold = np.deg2rad(angle_threshold_degrees)

    sharp_indices: list[int] = []
    for idx in range(working.shape[0]):
        if not closed and (idx == 0 or idx == working.shape[0] - 1):
            sharp_indices.append(idx)
            continue

        # `_polyline_turn_angle` already measures the local bend:
        # 0 for straight continuation, larger values for sharper corners.
        turn_angle = _polyline_turn_angle(working, idx, closed=closed, tolerance=tolerance)
        if turn_angle >= threshold:
            sharp_indices.append(idx)

    if closed and not sharp_indices:
        sharp_indices.append(0)

    return np.asarray(sorted(set(sharp_indices)), dtype=np.int64)


def _polyline_arclength(points: FloatArray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _rotate_closed_polyline(polyline: FloatArray, start_idx: int) -> FloatArray:
    unique_points = np.asarray(polyline[:-1], dtype=np.float64)
    return np.vstack((unique_points[start_idx:], unique_points[:start_idx]))


def _reverse_closed_polyline_from_start(polyline: FloatArray) -> FloatArray:
    points = np.asarray(polyline, dtype=np.float64)
    return np.vstack((points[:1], points[:0:-1]))


def _traverse_open_polyline_from_index(polyline: FloatArray, index: int) -> FloatArray:
    points = np.asarray(polyline, dtype=np.float64)
    forward_first = np.vstack((points[index:], points[: index + 1]))
    backward_first = np.vstack((points[index::-1], points[: index - 1 : -1]))
    if _polyline_arclength(forward_first) <= _polyline_arclength(backward_first):
        return forward_first
    return backward_first


def _resample_open_span(
    points: FloatArray,
    spacing: float,
    *,
    tolerance: float,
) -> FloatArray:
    if points.shape[0] <= 1:
        return np.asarray(points, dtype=np.float64)

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = cumulative[-1]
    if total_length <= tolerance:
        return np.asarray(points[[0, -1]], dtype=np.float64)

    target_distances = np.arange(0.0, total_length, spacing, dtype=np.float64)
    if target_distances.size == 0 or abs(target_distances[-1] - total_length) > tolerance:
        target_distances = np.concatenate((target_distances, [total_length]))

    resampled: list[FloatArray] = []
    seg_idx = 0
    for distance in target_distances:
        while seg_idx < segment_lengths.size - 1 and cumulative[seg_idx + 1] < distance - tolerance:
            seg_idx += 1

        start = points[seg_idx]
        end = points[seg_idx + 1]
        seg_start = cumulative[seg_idx]
        seg_length = max(segment_lengths[seg_idx], tolerance)
        t = (distance - seg_start) / seg_length
        t = min(1.0, max(0.0, t))
        resampled.append((1.0 - t) * start + t * end)

    return np.asarray(resampled, dtype=np.float64)


def _resample_open_span_with_normals(
    points: FloatArray,
    normals: FloatArray,
    spacing: float,
    *,
    tolerance: float,
) -> tuple[FloatArray, FloatArray]:
    if points.shape[0] <= 1:
        return np.asarray(points, dtype=np.float64), _normalize_vectors(normals, tolerance=tolerance)

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = cumulative[-1]
    if total_length <= tolerance:
        span_points = np.asarray(points[[0, -1]], dtype=np.float64)
        span_normals = _normalize_vectors(np.asarray(normals[[0, -1]], dtype=np.float64), tolerance=tolerance)
        return span_points, span_normals
    target_distances = np.arange(0.0, total_length, spacing, dtype=np.float64)
    if target_distances.size == 0 or abs(target_distances[-1] - total_length) > tolerance:
        target_distances = np.concatenate((target_distances, [total_length]))

    resampled_points: list[FloatArray] = []
    resampled_normals: list[FloatArray] = []
    seg_idx = 0
    #print(target_distances)
    for distance in target_distances:
        while seg_idx < segment_lengths.size - 1 and cumulative[seg_idx + 1] < distance - tolerance:
            seg_idx += 1
        #print(distance)
        p0 = points[seg_idx]
        p1 = points[seg_idx + 1]
        n0 = normals[seg_idx]
        n1 = normals[seg_idx + 1]
        seg_start = cumulative[seg_idx]
        seg_length = max(segment_lengths[seg_idx], tolerance)
        t = (distance - seg_start) / seg_length
        t = min(1.0, max(0.0, t))
        resampled_points.append((1.0 - t) * p0 + t * p1)
        normal = (1.0 - t) * n0 + t * n1
        normal_norm = np.linalg.norm(normal)
        if normal_norm <= tolerance:
            normal = n0
            normal_norm = max(np.linalg.norm(normal), tolerance)
        resampled_normals.append(normal / normal_norm)

    return (
        np.asarray(resampled_points, dtype=np.float64),
        np.asarray(resampled_normals, dtype=np.float64),
    )


def resample_polyline_equal_spacing(
    polyline: FloatArray,
    spacing: float,
    *,
    angle_threshold_degrees: float = 35.0,
    tolerance: float = 1e-8,
) -> FloatArray:
    """
    Resample a polyline at approximately equal arc-length spacing while keeping
    detected sharp points fixed.
    """
    points = np.asarray(polyline, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("polyline must have shape (k, 3).")
    if points.shape[0] < 2:
        return points.copy()
    if spacing <= 0.0:
        raise ValueError("spacing must be positive.")

    closed = is_closed_polyline(points, tolerance=tolerance)
    working = points[:-1] if closed else points
    sharp_indices = detect_sharp_vertices(
        points,
        angle_threshold_degrees=angle_threshold_degrees,
        tolerance=tolerance,
    )

    if closed:
        anchor = int(sharp_indices[0])
        rotated = np.vstack((working[anchor:], working[:anchor]))
        rotated = np.vstack((rotated, rotated[0]))
        sharp_indices = detect_sharp_vertices(
            rotated,
            angle_threshold_degrees=angle_threshold_degrees,
            tolerance=tolerance,
        )
        working_points = rotated
    else:
        working_points = working

    if sharp_indices.size == 0:
        sharp_indices = np.array([0, working_points.shape[0] - 1], dtype=np.int64)

    if not closed:
        sharp_indices = np.unique(
            np.concatenate((sharp_indices, [0, working_points.shape[0] - 1])),
        )
    else:
        sharp_indices = np.unique(np.concatenate((sharp_indices, [0])))
        if sharp_indices[-1] != working_points.shape[0] - 1:
            sharp_indices = np.concatenate((sharp_indices, [working_points.shape[0] - 1]))

    resampled_parts: list[FloatArray] = []
    for start_idx, end_idx in zip(sharp_indices[:-1], sharp_indices[1:]):
        span = working_points[start_idx : end_idx + 1]
        resampled_span = _resample_open_span(span, spacing, tolerance=tolerance)
        if resampled_parts:
            resampled_span = resampled_span[1:]
        resampled_parts.append(resampled_span)

    if not resampled_parts:
        return points.copy()

    resampled = np.vstack(resampled_parts)
    if closed:
        if np.linalg.norm(resampled[0] - resampled[-1]) > tolerance:
            resampled = np.vstack((resampled, resampled[0]))
    return resampled


def resample_contour_levels(
    contour_levels: list[ContourLevel],
    spacing: float,
    *,
    angle_threshold_degrees: float = 35.0,
    tolerance: float = 1e-8,
) -> list[ContourLevel]:
    """Resample every contour polyline at approximately equal spacing."""
    print("I am called2")

    resampled_levels: list[ContourLevel] = []
    for contour_level in contour_levels:
        polylines = [
            resample_polyline_equal_spacing(
                polyline,
                spacing,
                angle_threshold_degrees=angle_threshold_degrees,
                tolerance=tolerance,
            )
            for polyline in contour_level.polylines
        ]
        resampled_levels.append(ContourLevel(level=contour_level.level, polylines=polylines))
    return resampled_levels


def resample_oriented_polyline_equal_spacing(
    polyline: OrientedPolyline,
    spacing: float,
    *,
    angle_threshold_degrees: float = 35.0,
    tolerance: float = 1e-8,
) -> OrientedPolyline:
    """Resample an oriented polyline while preserving sharp points and normals."""
    print("I am called3")

    points = np.asarray(polyline.points, dtype=np.float64)
    normals = _normalize_vectors(polyline.normals, tolerance=tolerance)
    if points.shape != normals.shape:
        raise ValueError("Polyline points and normals must have matching shapes.")
    if points.shape[0] < 2:
        return OrientedPolyline(points=points.copy(), normals=normals.copy())

    closed = is_closed_polyline(points, tolerance=tolerance)
    working_points = points[:-1] if closed else points
    working_normals = normals[:-1] if closed else normals
    sharp_indices = detect_sharp_vertices(
        points,
        angle_threshold_degrees=angle_threshold_degrees,
        tolerance=tolerance,
    )

    if closed:
        anchor = int(sharp_indices[0])
        rotated_points = np.vstack((working_points[anchor:], working_points[:anchor]))
        rotated_normals = np.vstack((working_normals[anchor:], working_normals[:anchor]))
        working_points = np.vstack((rotated_points, rotated_points[0]))
        working_normals = np.vstack((rotated_normals, rotated_normals[0]))
        sharp_indices = detect_sharp_vertices(
            working_points,
            angle_threshold_degrees=angle_threshold_degrees,
            tolerance=tolerance,
        )
    #print(sharp_indices)
    if sharp_indices.size == 0:
        sharp_indices = np.array([0, working_points.shape[0] - 1], dtype=np.int64)

    if not closed:
        sharp_indices = np.unique(
            np.concatenate((sharp_indices, [0, working_points.shape[0] - 1])),
        )
    else:
        sharp_indices = np.unique(np.concatenate((sharp_indices, [0])))
        if sharp_indices[-1] != working_points.shape[0] - 1:
            sharp_indices = np.concatenate((sharp_indices, [working_points.shape[0] - 1]))

    point_parts: list[FloatArray] = []
    normal_parts: list[FloatArray] = []
    for start_idx, end_idx in zip(sharp_indices[:-1], sharp_indices[1:]):
        span_points = working_points[start_idx : end_idx + 1]
        span_normals = working_normals[start_idx : end_idx + 1]
        res_points, res_normals = _resample_open_span_with_normals(
            span_points,
            span_normals,
            spacing,
            tolerance=tolerance,
        )
        if point_parts:
            res_points = res_points[1:]
            res_normals = res_normals[1:]
        point_parts.append(res_points)
        normal_parts.append(res_normals)

    if not point_parts:
        return OrientedPolyline(points=points.copy(), normals=normals.copy())

    resampled_points = np.vstack(point_parts)
    resampled_normals = np.vstack(normal_parts)
    if closed and np.linalg.norm(resampled_points[0] - resampled_points[-1]) > tolerance:
        resampled_points = np.vstack((resampled_points, resampled_points[0]))
        resampled_normals = np.vstack((resampled_normals, resampled_normals[0]))

    return OrientedPolyline(points=resampled_points, normals=_normalize_vectors(resampled_normals, tolerance=tolerance))


def resample_oriented_contour_levels(
    contour_levels: list[OrientedContourLevel],
    spacing: float,
    *,
    angle_threshold_degrees: float = 35.0,
    tolerance: float = 1e-8,
) -> list[OrientedContourLevel]:
    """Resample oriented contour levels while preserving normals."""
    return [
        OrientedContourLevel(
            level=level.level,
            polylines=[
                resample_oriented_polyline_equal_spacing(
                    polyline,
                    spacing,
                    angle_threshold_degrees=angle_threshold_degrees,
                    tolerance=tolerance,
                )
                for polyline in level.polylines
            ],
        )
        for level in contour_levels
    ]


def reorder_polyline_from_vertex(
    polyline: FloatArray,
    start_idx: int,
    *,
    tolerance: float = 1e-8,
) -> FloatArray:
    """
    Reorder a polyline so traversal starts at the specified existing vertex.

    Closed loops are rotated. Open polylines are traversed from the entry
    vertex to one endpoint, then continued from the opposite endpoint back to
    the entry vertex.
    """
    points = np.asarray(polyline, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("polyline must have shape (k, 3).")
    if start_idx < 0 or start_idx >= points.shape[0]:
        raise ValueError("start_idx is out of bounds for the polyline.")

    if is_closed_polyline(points, tolerance=tolerance):
        unique_count = points.shape[0] - 1
        if start_idx >= unique_count:
            start_idx = 0
        return _rotate_closed_polyline(points, start_idx)

    return _traverse_open_polyline_from_index(points, start_idx)


def reorder_oriented_polyline_from_vertex(
    polyline: OrientedPolyline,
    start_idx: int,
    *,
    tolerance: float = 1e-8,
) -> OrientedPolyline:
    """Reorder an oriented polyline while preserving normals."""
    points = np.asarray(polyline.points, dtype=np.float64)
    normals = np.asarray(polyline.normals, dtype=np.float64)
    if points.shape != normals.shape:
        raise ValueError("Polyline points and normals must have matching shapes.")

    reordered_points = reorder_polyline_from_vertex(points, start_idx, tolerance=tolerance)
    closed = is_closed_polyline(points, tolerance=tolerance)
    if closed:
        unique_points = points[:-1]
        unique_normals = normals[:-1]
        if start_idx >= unique_points.shape[0]:
            start_idx = 0
        rotated_normals = np.vstack((unique_normals[start_idx:], unique_normals[:start_idx]))
        return OrientedPolyline(points=reordered_points, normals=rotated_normals)

    points_list = [tuple(p) for p in points]
    reordered_normals: list[FloatArray] = []
    used = np.zeros(points.shape[0], dtype=bool)
    for point in reordered_points:
        distances = np.linalg.norm(points - point[None, :], axis=1)
        order = np.argsort(distances)
        matched = None
        for idx in order:
            if not used[idx] and distances[idx] <= tolerance:
                matched = int(idx)
                break
        if matched is None:
            matched = int(order[0])
        used[matched] = True
        reordered_normals.append(normals[matched])
    return OrientedPolyline(points=reordered_points, normals=_normalize_vectors(np.asarray(reordered_normals), tolerance=tolerance))


def boundary_edges_to_polylines(
    V: FloatArray,
    boundary_edges: IntArray,
) -> list[FloatArray]:
    """Convert stitched boundary edge loops/chains into 3D polylines."""
    loops = stitch_boundary_loops(boundary_edges)
    polylines: list[FloatArray] = []
    for loop in loops:
        if loop.size < 2:
            continue
        points = np.asarray(V[np.asarray(loop, dtype=np.int64)], dtype=np.float64)
        if loop[0] == loop[-1]:
            polylines.append(points)
        else:
            polylines.append(points)
    return polylines


def boundary_edges_to_oriented_polylines(
    V: FloatArray,
    boundary_edges: IntArray,
    vertex_normals: FloatArray,
) -> list[OrientedPolyline]:
    """Convert stitched boundary loops/chains into oriented polylines."""
    loops = stitch_boundary_loops(boundary_edges)
    polylines: list[OrientedPolyline] = []
    for loop in loops:
        if loop.size < 2:
            continue
        indices = np.asarray(loop, dtype=np.int64)
        polylines.append(
            OrientedPolyline(
                points=np.asarray(V[indices], dtype=np.float64),
                normals=_normalize_vectors(np.asarray(vertex_normals[indices], dtype=np.float64)),
            ),
        )
    return polylines


def build_connected_contour_polyline(
    contour_levels: list[ContourLevel],
    *,
    start_point: FloatArray | None = None,
    tolerance: float = 1e-8,
) -> FloatArray:
    """
    Greedily chain contour polylines into one connected ordered path.

    The path starts on the outermost contour set. For each next contour, the
    closest existing vertex to the current path endpoint is chosen as the entry
    point. Closed loops are rotated at that vertex. Open polylines are entered
    at that vertex and traversed to one endpoint, then from the opposite
    endpoint back to the entry vertex.
    """
    candidates: list[tuple[float, FloatArray]] = []
    for contour_level in contour_levels:
        for polyline in contour_level.polylines:
            if polyline.shape[0] >= 2:
                candidates.append((contour_level.level, np.asarray(polyline, dtype=np.float64)))

    if not candidates:
        return np.empty((0, 3), dtype=np.float64)

    candidates.sort(key=lambda item: item[0])
    outermost_level = candidates[0][0]
    outermost_indices = [idx for idx, item in enumerate(candidates) if abs(item[0] - outermost_level) <= tolerance]

    if start_point is None:
        outermost_vertices = np.vstack([candidates[idx][1] for idx in outermost_indices])
        lex_order = np.lexsort((outermost_vertices[:, 2], outermost_vertices[:, 1], outermost_vertices[:, 0]))
        current_point = outermost_vertices[lex_order[0]]
    else:
        current_point = np.asarray(start_point, dtype=np.float64)
        if current_point.shape != (3,):
            raise ValueError("start_point must have shape (3,).")

    connected_parts: list[FloatArray] = []
    remaining = candidates.copy()

    while remaining:
        best_poly_idx = -1
        best_vertex_idx = -1
        best_distance = np.inf

        for poly_idx, (_, polyline) in enumerate(remaining):
            if is_closed_polyline(polyline, tolerance=tolerance):
                search_points = polyline[:-1]
            else:
                search_points = polyline

            distances = np.linalg.norm(search_points - current_point[None, :], axis=1)
            vertex_idx = int(np.argmin(distances))
            distance = float(distances[vertex_idx])
            if distance < best_distance:
                best_distance = distance
                best_poly_idx = poly_idx
                best_vertex_idx = vertex_idx

        _, selected_polyline = remaining.pop(best_poly_idx)
        ordered_polyline = reorder_polyline_from_vertex(
            selected_polyline,
            best_vertex_idx,
            tolerance=tolerance,
        )
        connected_parts.append(ordered_polyline)
        current_point = ordered_polyline[-1]

    connected_path = connected_parts[0]
    for part in connected_parts[1:]:
        connected_path = np.vstack((connected_path, part))
    return connected_path


def build_connected_closed_loop_path(
    contour_levels: list[ContourLevel],
    *,
    boundary_polylines: list[FloatArray] | None = None,
    start_point: FloatArray | None = None,
    tolerance: float = 1e-8,
) -> FloatArray:
    """
    Greedily chain predominantly closed contour loops into one connected path.

    Boundary polylines can be provided as the outermost paths. For each next
    closed loop, the nearest existing vertex to the current endpoint is chosen
    as the start, and only the two possible loop orientations are compared.
    Open polylines fall back to the generic reorder logic.
    """
    candidates: list[tuple[float, FloatArray]] = []

    if boundary_polylines is not None:
        for polyline in boundary_polylines:
            if polyline.shape[0] >= 2:
                candidates.append((0.0, np.asarray(polyline, dtype=np.float64)))

    for contour_level in contour_levels:
        for polyline in contour_level.polylines:
            if polyline.shape[0] >= 2:
                candidates.append((contour_level.level, np.asarray(polyline, dtype=np.float64)))

    if not candidates:
        return np.empty((0, 3), dtype=np.float64)

    candidates.sort(key=lambda item: item[0])

    if start_point is None:
        outer_level = candidates[0][0]
        outer_vertices = np.vstack(
            [poly for level, poly in candidates if abs(level - outer_level) <= tolerance],
        )
        lex_order = np.lexsort((outer_vertices[:, 2], outer_vertices[:, 1], outer_vertices[:, 0]))
        current_point = outer_vertices[lex_order[0]]
    else:
        current_point = np.asarray(start_point, dtype=np.float64)
        if current_point.shape != (3,):
            raise ValueError("start_point must have shape (3,).")

    connected_parts: list[FloatArray] = []
    remaining = candidates.copy()

    while remaining:
        best_poly_idx = -1
        best_vertex_idx = -1
        best_distance = np.inf

        for poly_idx, (_, polyline) in enumerate(remaining):
            search_points = polyline[:-1] if is_closed_polyline(polyline, tolerance=tolerance) else polyline
            distances = np.linalg.norm(search_points - current_point[None, :], axis=1)
            vertex_idx = int(np.argmin(distances))
            distance = float(distances[vertex_idx])
            if distance < best_distance:
                best_distance = distance
                best_poly_idx = poly_idx
                best_vertex_idx = vertex_idx

        _, selected_polyline = remaining.pop(best_poly_idx)

        if is_closed_polyline(selected_polyline, tolerance=tolerance):
            rotated = reorder_polyline_from_vertex(
                selected_polyline,
                best_vertex_idx,
                tolerance=tolerance,
            )
            reversed_rotated = _reverse_closed_polyline_from_start(rotated)
            if np.linalg.norm(current_point - reversed_rotated[0]) < np.linalg.norm(current_point - rotated[0]):
                ordered_polyline = reversed_rotated
            else:
                ordered_polyline = rotated
        else:
            ordered_polyline = reorder_polyline_from_vertex(
                selected_polyline,
                best_vertex_idx,
                tolerance=tolerance,
            )

        connected_parts.append(ordered_polyline)
        current_point = ordered_polyline[-1]

    connected_path = connected_parts[0]
    for part in connected_parts[1:]:
        connected_path = np.vstack((connected_path, part))
    return connected_path


def _reverse_closed_oriented_polyline_from_start(polyline: OrientedPolyline) -> OrientedPolyline:
    reversed_points = _reverse_closed_polyline_from_start(polyline.points)
    normals = np.asarray(polyline.normals, dtype=np.float64)
    reversed_normals = np.vstack((normals[:1], normals[:0:-1]))
    return OrientedPolyline(points=reversed_points, normals=_normalize_vectors(reversed_normals))


def filter_oriented_contour_levels_by_point_count(
    contour_levels: list[OrientedContourLevel],
    *,
    min_points: int = 4,
) -> list[OrientedContourLevel]:
    """Remove oriented contour polylines with fewer than the requested number of points."""
    if min_points < 2:
        raise ValueError("min_points must be at least 2.")
    return [
        OrientedContourLevel(
            level=level.level,
            polylines=[poly for poly in level.polylines if poly.points.shape[0] >= min_points],
        )
        for level in contour_levels
    ]


def build_connected_closed_loop_path_with_normals(
    contour_levels: list[OrientedContourLevel],
    *,
    boundary_polylines: list[OrientedPolyline] | None = None,
    start_point: FloatArray | None = None,
    start_selection_mode: str = "lexicographic",
    tolerance: float = 1e-8,
) -> OrientedPolyline:
    """Build a connected path while preserving per-point normals."""
    if start_selection_mode == "lowest_z":
        start_selection_mode = "lowest_y"
    if start_selection_mode not in {"lexicographic", "lowest_y"}:
        raise ValueError("start_selection_mode must be either 'lexicographic' or 'lowest_y'.")
    candidates: list[tuple[float, OrientedPolyline]] = []
    if boundary_polylines is not None:
        for polyline in boundary_polylines:
            if polyline.points.shape[0] >= 2:
                candidates.append((0.0, polyline))
    for contour_level in contour_levels:
        for polyline in contour_level.polylines:
            if polyline.points.shape[0] >= 2:
                candidates.append((contour_level.level, polyline))

    if not candidates:
        return OrientedPolyline(
            points=np.empty((0, 3), dtype=np.float64),
            normals=np.empty((0, 3), dtype=np.float64),
        )

    candidates.sort(key=lambda item: item[0])
    if start_point is None:
        outer_level = candidates[0][0]
        outer_vertices = np.vstack(
            [poly.points for level, poly in candidates if abs(level - outer_level) <= tolerance],
        )
        if start_selection_mode == "lowest_y":
            lex_order = np.lexsort((outer_vertices[:, 0], outer_vertices[:, 2], outer_vertices[:, 1]))
        else:
            lex_order = np.lexsort((outer_vertices[:, 2], outer_vertices[:, 1], outer_vertices[:, 0]))
        current_point = outer_vertices[lex_order[0]]
    else:
        current_point = np.asarray(start_point, dtype=np.float64)
        if current_point.shape != (3,):
            raise ValueError("start_point must have shape (3,).")

    connected_points: list[FloatArray] = []
    connected_normals: list[FloatArray] = []
    remaining = candidates.copy()

    while remaining:
        best_poly_idx = -1
        best_vertex_idx = -1
        best_distance = np.inf
        for poly_idx, (_, polyline) in enumerate(remaining):
            search_points = polyline.points[:-1] if is_closed_polyline(polyline.points, tolerance=tolerance) else polyline.points
            distances = np.linalg.norm(search_points - current_point[None, :], axis=1)
            vertex_idx = int(np.argmin(distances))
            distance = float(distances[vertex_idx])
            if distance < best_distance:
                best_distance = distance
                best_poly_idx = poly_idx
                best_vertex_idx = vertex_idx

        _, selected = remaining.pop(best_poly_idx)
        if is_closed_polyline(selected.points, tolerance=tolerance):
            rotated = reorder_oriented_polyline_from_vertex(selected, best_vertex_idx, tolerance=tolerance)
            reversed_rotated = _reverse_closed_oriented_polyline_from_start(rotated)
            if np.linalg.norm(current_point - reversed_rotated.points[0]) < np.linalg.norm(current_point - rotated.points[0]):
                ordered = reversed_rotated
            else:
                ordered = rotated
        else:
            ordered = reorder_oriented_polyline_from_vertex(selected, best_vertex_idx, tolerance=tolerance)

        connected_points.append(ordered.points)
        connected_normals.append(ordered.normals)
        current_point = ordered.points[-1]

    return OrientedPolyline(
        points=np.vstack(connected_points),
        normals=_normalize_vectors(np.vstack(connected_normals)),
    )


def compute_contour_levels(
    scalar_field: FloatArray,
    spacing: float,
    *,
    min_level: float | None = None,
    max_level: float | None = None,
    include_zero: bool = False,
    epsilon: float = 1e-12,
) -> FloatArray:
    """Compute equally spaced contour levels from a scalar field."""
    values = np.asarray(scalar_field, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("scalar_field must be a 1D array.")
    if spacing <= 0.0:
        raise ValueError("spacing must be positive.")

    lower = float(np.min(values) if min_level is None else min_level)
    upper = float(np.max(values) if max_level is None else max_level)
    if upper < lower:
        raise ValueError("max_level must be greater than or equal to min_level.")

    start = 0.0 if include_zero else spacing
    start = max(start, lower + (0.0 if include_zero else epsilon))
    if upper < start - epsilon:
        return np.array([], dtype=np.float64)

    num_levels = int(np.floor((upper - start) / spacing + epsilon)) + 1
    if num_levels <= 0:
        return np.array([], dtype=np.float64)

    levels = start + spacing * np.arange(num_levels, dtype=np.float64)
    return levels[levels <= upper + epsilon]


def _interpolate_edge_point(
    p0: FloatArray,
    p1: FloatArray,
    s0: float,
    s1: float,
    level: float,
    epsilon: float,
) -> FloatArray | None:
    if abs(s0 - level) <= epsilon and abs(s1 - level) <= epsilon:
        return None
    if abs(s0 - level) <= epsilon:
        return np.asarray(p0, dtype=np.float64)
    if abs(s1 - level) <= epsilon:
        return np.asarray(p1, dtype=np.float64)
    if (s0 - level) * (s1 - level) > 0.0:
        return None

    t = (level - s0) / (s1 - s0)
    if t < -epsilon or t > 1.0 + epsilon:
        return None
    t = min(1.0, max(0.0, t))
    return np.asarray((1.0 - t) * p0 + t * p1, dtype=np.float64)


def _interpolate_edge_point_and_normal(
    p0: FloatArray,
    p1: FloatArray,
    n0: FloatArray,
    n1: FloatArray,
    s0: float,
    s1: float,
    level: float,
    epsilon: float,
) -> tuple[FloatArray, FloatArray] | None:
    point = _interpolate_edge_point(p0, p1, s0, s1, level, epsilon)
    if point is None:
        return None
    if abs(s0 - s1) <= epsilon:
        normal = np.asarray(n0, dtype=np.float64)
    else:
        t = (level - s0) / (s1 - s0)
        t = min(1.0, max(0.0, t))
        normal = (1.0 - t) * np.asarray(n0, dtype=np.float64) + t * np.asarray(n1, dtype=np.float64)
    normal = normal / max(np.linalg.norm(normal), epsilon)
    return np.asarray(point, dtype=np.float64), np.asarray(normal, dtype=np.float64)


def _dedupe_points(points: list[FloatArray], tolerance: float) -> list[FloatArray]:
    deduped: list[FloatArray] = []
    for point in points:
        if not any(np.linalg.norm(point - existing) <= tolerance for existing in deduped):
            deduped.append(point)
    return deduped


def _dedupe_samples(
    samples: list[tuple[FloatArray, FloatArray]],
    tolerance: float,
) -> list[tuple[FloatArray, FloatArray]]:
    deduped: list[tuple[FloatArray, FloatArray]] = []
    for point, normal in samples:
        if not any(np.linalg.norm(point - existing_point) <= tolerance for existing_point, _ in deduped):
            deduped.append((point, normal))
    return deduped


def extract_iso_segments(
    V: FloatArray,
    F: IntArray,
    scalar_field: FloatArray,
    level: float,
    *,
    tolerance: float = 1e-8,
) -> list[tuple[FloatArray, FloatArray]]:
    """
    Extract raw iso-segments for a single contour level.

    Each triangle contributes zero or one segment in the generic case.
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    values = np.asarray(scalar_field, dtype=np.float64)

    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError("V must have shape (n, 3).")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("F must have shape (m, 3).")
    if values.shape != (V.shape[0],):
        raise ValueError("scalar_field must have shape (n_vertices,).")

    edge_pairs = ((0, 1), (1, 2), (2, 0))
    segments: list[tuple[FloatArray, FloatArray]] = []

    for tri in F:
        tri_points = V[tri]
        tri_values = values[tri]
        intersections: list[FloatArray] = []

        for i0, i1 in edge_pairs:
            point = _interpolate_edge_point(
                tri_points[i0],
                tri_points[i1],
                float(tri_values[i0]),
                float(tri_values[i1]),
                level,
                tolerance,
            )
            if point is not None:
                intersections.append(point)

        intersections = _dedupe_points(intersections, tolerance)
        if len(intersections) == 2:
            segments.append((intersections[0], intersections[1]))
        elif len(intersections) > 2:
            # Degenerate cases near vertices/flat edges are simplified by
            # pairing the farthest two intersection points.
            max_pair = None
            max_distance = -1.0
            for idx, first in enumerate(intersections):
                for second in intersections[idx + 1 :]:
                    distance = np.linalg.norm(first - second)
                    if distance > max_distance:
                        max_distance = distance
                        max_pair = (first, second)
            if max_pair is not None and max_distance > tolerance:
                segments.append(max_pair)

    return segments


def extract_iso_segments_with_normals(
    V: FloatArray,
    F: IntArray,
    scalar_field: FloatArray,
    vertex_normals: FloatArray,
    level: float,
    *,
    tolerance: float = 1e-8,
) -> list[tuple[tuple[FloatArray, FloatArray], tuple[FloatArray, FloatArray]]]:
    """Extract raw iso-segments with interpolated normals."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    values = np.asarray(scalar_field, dtype=np.float64)
    normals = _normalize_vectors(np.asarray(vertex_normals, dtype=np.float64))
    if normals.shape != V.shape:
        raise ValueError("vertex_normals must have shape matching V.")

    edge_pairs = ((0, 1), (1, 2), (2, 0))
    segments: list[tuple[tuple[FloatArray, FloatArray], tuple[FloatArray, FloatArray]]] = []
    for tri in F:
        tri_points = V[tri]
        tri_values = values[tri]
        tri_normals = normals[tri]
        intersections: list[tuple[FloatArray, FloatArray]] = []
        for i0, i1 in edge_pairs:
            sample = _interpolate_edge_point_and_normal(
                tri_points[i0],
                tri_points[i1],
                tri_normals[i0],
                tri_normals[i1],
                float(tri_values[i0]),
                float(tri_values[i1]),
                level,
                tolerance,
            )
            if sample is not None:
                intersections.append(sample)

        intersections = _dedupe_samples(intersections, tolerance)
        if len(intersections) == 2:
            segments.append((intersections[0], intersections[1]))
        elif len(intersections) > 2:
            max_pair = None
            max_distance = -1.0
            for idx, (first_point, first_normal) in enumerate(intersections):
                for second_point, second_normal in intersections[idx + 1 :]:
                    distance = np.linalg.norm(first_point - second_point)
                    if distance > max_distance:
                        max_distance = distance
                        max_pair = ((first_point, first_normal), (second_point, second_normal))
            if max_pair is not None and max_distance > tolerance:
                segments.append(max_pair)
    return segments


def _point_key(point: FloatArray, tolerance: float) -> tuple[int, int, int]:
    scaled = np.asarray(point, dtype=np.float64) / tolerance
    return tuple(np.round(scaled).astype(np.int64))


def stitch_segments_into_polylines(
    segments: list[tuple[FloatArray, FloatArray]],
    *,
    tolerance: float = 1e-8,
) -> list[FloatArray]:
    """Stitch raw segments into ordered polylines or loops."""
    if not segments:
        return []

    points_by_key: dict[tuple[int, int, int], FloatArray] = {}
    adjacency: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}

    for start, end in segments:
        key0 = _point_key(start, tolerance)
        key1 = _point_key(end, tolerance)
        points_by_key.setdefault(key0, start)
        points_by_key.setdefault(key1, end)
        adjacency.setdefault(key0, []).append(key1)
        adjacency.setdefault(key1, []).append(key0)

    visited_edges: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
    polylines: list[FloatArray] = []

    def mark_edge(a: tuple[int, int, int], b: tuple[int, int, int]) -> None:
        visited_edges.add((a, b))
        visited_edges.add((b, a))

    start_nodes = [key for key, neighbors in adjacency.items() if len(neighbors) == 1]
    if not start_nodes:
        start_nodes = list(adjacency.keys())

    for start_key in start_nodes:
        for neighbor_key in adjacency.get(start_key, []):
            if (start_key, neighbor_key) in visited_edges:
                continue

            ordered_keys = [start_key, neighbor_key]
            mark_edge(start_key, neighbor_key)
            prev_key = start_key
            current_key = neighbor_key

            while True:
                candidates = [
                    candidate
                    for candidate in adjacency.get(current_key, [])
                    if candidate != prev_key and (current_key, candidate) not in visited_edges
                ]
                if not candidates:
                    break
                next_key = candidates[0]
                ordered_keys.append(next_key)
                mark_edge(current_key, next_key)
                prev_key, current_key = current_key, next_key
                if current_key == start_key:
                    break

            polyline = np.asarray([points_by_key[key] for key in ordered_keys], dtype=np.float64)
            polylines.append(polyline)

    return polylines


def stitch_segments_into_oriented_polylines(
    segments: list[tuple[tuple[FloatArray, FloatArray], tuple[FloatArray, FloatArray]]],
    *,
    tolerance: float = 1e-8,
) -> list[OrientedPolyline]:
    """Stitch raw oriented segments into ordered polylines or loops."""
    if not segments:
        return []

    points_by_key: dict[tuple[int, int, int], FloatArray] = {}
    normals_by_key: dict[tuple[int, int, int], FloatArray] = {}
    adjacency: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}

    for (start_point, start_normal), (end_point, end_normal) in segments:
        key0 = _point_key(start_point, tolerance)
        key1 = _point_key(end_point, tolerance)
        points_by_key.setdefault(key0, start_point)
        points_by_key.setdefault(key1, end_point)
        normals_by_key.setdefault(key0, start_normal)
        normals_by_key.setdefault(key1, end_normal)
        adjacency.setdefault(key0, []).append(key1)
        adjacency.setdefault(key1, []).append(key0)

    visited_edges: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
    polylines: list[OrientedPolyline] = []

    def mark_edge(a: tuple[int, int, int], b: tuple[int, int, int]) -> None:
        visited_edges.add((a, b))
        visited_edges.add((b, a))

    start_nodes = [key for key, neighbors in adjacency.items() if len(neighbors) == 1]
    if not start_nodes:
        start_nodes = list(adjacency.keys())

    for start_key in start_nodes:
        for neighbor_key in adjacency.get(start_key, []):
            if (start_key, neighbor_key) in visited_edges:
                continue

            ordered_keys = [start_key, neighbor_key]
            mark_edge(start_key, neighbor_key)
            prev_key = start_key
            current_key = neighbor_key
            while True:
                candidates = [
                    candidate
                    for candidate in adjacency.get(current_key, [])
                    if candidate != prev_key and (current_key, candidate) not in visited_edges
                ]
                if not candidates:
                    break
                next_key = candidates[0]
                ordered_keys.append(next_key)
                mark_edge(current_key, next_key)
                prev_key, current_key = current_key, next_key
                if current_key == start_key:
                    break

            poly_points = np.asarray([points_by_key[key] for key in ordered_keys], dtype=np.float64)
            poly_normals = _normalize_vectors(
                np.asarray([normals_by_key[key] for key in ordered_keys], dtype=np.float64),
                tolerance=tolerance,
            )
            polylines.append(OrientedPolyline(points=poly_points, normals=poly_normals))

    return polylines


def extract_isocontour_polylines(
    V: FloatArray,
    F: IntArray,
    scalar_field: FloatArray,
    level: float,
    *,
    tolerance: float = 1e-8,
) -> list[FloatArray]:
    """Extract ordered contour polylines for a single iso-level."""
    segments = extract_iso_segments(
        V,
        F,
        scalar_field,
        level,
        tolerance=tolerance,
    )
    return stitch_segments_into_polylines(segments, tolerance=tolerance)


def generate_equally_spaced_contours(
    V: FloatArray,
    F: IntArray,
    scalar_field: FloatArray,
    spacing: float,
    *,
    min_level: float | None = None,
    max_level: float | None = None,
    include_zero: bool = False,
    tolerance: float = 1e-8,
) -> list[ContourLevel]:
    """Generate equally spaced contour polylines for a per-vertex scalar field."""
    levels = compute_contour_levels(
        scalar_field,
        spacing,
        min_level=min_level,
        max_level=max_level,
        include_zero=include_zero,
        epsilon=tolerance,
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


def generate_equally_spaced_oriented_contours(
    V: FloatArray,
    F: IntArray,
    scalar_field: FloatArray,
    vertex_normals: FloatArray,
    spacing: float,
    *,
    min_level: float | None = None,
    max_level: float | None = None,
    include_zero: bool = False,
    tolerance: float = 1e-8,
) -> list[OrientedContourLevel]:
    """Generate equally spaced oriented contour polylines for a scalar field."""
    levels = compute_contour_levels(
        scalar_field,
        spacing,
        min_level=min_level,
        max_level=max_level,
        include_zero=include_zero,
        epsilon=tolerance,
    )
    contours: list[OrientedContourLevel] = []
    for level in levels:
        segments = extract_iso_segments_with_normals(
            V,
            F,
            scalar_field,
            vertex_normals,
            float(level),
            tolerance=tolerance,
        )
        polylines = stitch_segments_into_oriented_polylines(segments, tolerance=tolerance)
        contours.append(OrientedContourLevel(level=float(level), polylines=polylines))
    return contours


def generate_boundary_distance_contours(
    V: FloatArray,
    F: IntArray,
    spacing: float,
    *,
    heat_params: HeatMethodParams | None = None,
    prefer_libigl: bool = True,
    heat_backend: str = "native",
    max_level: float | None = None,
    max_level_fraction: float | None = None,
    tolerance: float = 1e-8,
) -> tuple[FloatArray, list[ContourLevel]]:
    """
    Solve the boundary distance field and extract equally spaced contours from it.
    """
    result = solve_boundary_distance_field(
        V,
        F,
        params=heat_params,
        prefer_libigl=prefer_libigl,
        backend=heat_backend,
        return_intermediates=False,
    )
    scalar_field = result.distance
    effective_max_level = max_level
    if max_level_fraction is not None:
        if not (0.0 < max_level_fraction <= 1.0):
            raise ValueError("max_level_fraction must be in the interval (0, 1].")
        fraction_limit = float(max_level_fraction * np.max(scalar_field))
        effective_max_level = fraction_limit if effective_max_level is None else min(effective_max_level, fraction_limit)
    contours = generate_equally_spaced_contours(
        V,
        F,
        scalar_field,
        spacing,
        min_level=spacing,
        max_level=effective_max_level,
        include_zero=False,
        tolerance=tolerance,
    )
    return scalar_field, contours


def generate_boundary_loops(V: FloatArray, F: IntArray) -> list[FloatArray]:
    """Return exact boundary loops/chains as 3D polylines from mesh topology."""
    boundary_edges = find_boundary_edges(F)
    return boundary_edges_to_polylines(V, boundary_edges)


def generate_boundary_loops_with_normals(V: FloatArray, F: IntArray) -> list[OrientedPolyline]:
    """Return exact boundary loops/chains with interpolated vertex normals."""
    vertex_normals = compute_vertex_normals(V, F)
    boundary_edges = find_boundary_edges(F)
    return boundary_edges_to_oriented_polylines(V, boundary_edges, vertex_normals)


def generate_boundary_distance_contours_with_normals(
    V: FloatArray,
    F: IntArray,
    spacing: float,
    *,
    heat_params: HeatMethodParams | None = None,
    prefer_libigl: bool = True,
    heat_backend: str = "native",
    max_level: float | None = None,
    max_level_fraction: float | None = None,
    tolerance: float = 1e-8,
) -> tuple[FloatArray, list[OrientedContourLevel]]:
    """Solve the boundary distance field and extract oriented contours from it."""
    result = solve_boundary_distance_field(
        V,
        F,
        params=heat_params,
        prefer_libigl=prefer_libigl,
        backend=heat_backend,
        return_intermediates=False,
    )
    scalar_field = result.distance
    vertex_normals = compute_vertex_normals(V, F)
    effective_max_level = max_level
    if max_level_fraction is not None:
        if not (0.0 < max_level_fraction <= 1.0):
            raise ValueError("max_level_fraction must be in the interval (0, 1].")
        fraction_limit = float(max_level_fraction * np.max(scalar_field))
        effective_max_level = fraction_limit if effective_max_level is None else min(effective_max_level, fraction_limit)
    contours = generate_equally_spaced_oriented_contours(
        V,
        F,
        scalar_field,
        vertex_normals,
        spacing,
        min_level=spacing,
        max_level=effective_max_level,
        include_zero=False,
        tolerance=tolerance,
    )
    return scalar_field, contours
