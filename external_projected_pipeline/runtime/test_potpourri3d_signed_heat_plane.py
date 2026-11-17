"""
Standalone test driver for potpourri3d signed-heat distance from plane/mesh
intersection curves.

This script is intentionally separate from the main pipeline. It:
1. loads a triangle mesh
2. intersects it with an axis-aligned plane
3. stitches the resulting intersection segments into ordered curves
4. encodes those curves as face-barycentric source nodes for potpourri3d
5. computes a signed or unsigned distance field with MeshSignedHeatSolver
6. optionally extracts contour polylines from the resulting field
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contour_extraction import ContourLevel
from contour_extraction import generate_equally_spaced_contours
from contour_extraction import extract_isocontour_polylines
from heat_method_core import FloatArray
from mesh_distance_workflow import load_obj_mesh
from mesh_distance_workflow import to_pyvista_polydata

try:
    import potpourri3d as pp3d  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pp3d = None

try:
    import pyvista as pv  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pv = None


AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(slots=True)
class CurvePoint:
    xyz: FloatArray
    face_index: int
    barycentric: FloatArray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test potpourri3d MeshSignedHeatSolver from a mesh/plane intersection curve.",
    )
    parser.add_argument("mesh_path", type=Path, help="Input triangle mesh.")
    parser.add_argument("output_dir", type=Path, help="Directory to write debug outputs.")
    parser.add_argument(
        "--axis",
        choices=("x", "y", "z"),
        default="x",
        help="Axis normal for the cutting plane. 'x' means a yz-parallel plane x=value.",
    )
    parser.add_argument(
        "--value",
        type=float,
        default=0.0,
        help="Plane coordinate value along the selected axis.",
    )
    parser.add_argument(
        "--signed",
        action="store_true",
        help="Compute signed distance to the source curves. Default is unsigned.",
    )
    parser.add_argument(
        "--preserve-source-normals",
        action="store_true",
        help="Forward preserve_source_normals=True to MeshSignedHeatSolver.",
    )
    parser.add_argument(
        "--level-set-constraint",
        choices=("ZeroSet", "None", "Multiple"),
        default="None",
        help="Level-set constraint passed to MeshSignedHeatSolver.",
    )
    parser.add_argument(
        "--soft-level-set-weight",
        type=float,
        default=-1.0,
        help="Soft level-set weight passed to MeshSignedHeatSolver.",
    )
    parser.add_argument(
        "--contour-spacing",
        type=float,
        default=None,
        help="If set, extract contour polylines from the solved field at this spacing.",
    )
    parser.add_argument(
        "--contour-max-level-fraction",
        type=float,
        default=0.95,
        help="Optional fraction of max field value used to trim extracted contours.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
        help="Geometric tolerance for intersection/stitching.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview the solved field on the mesh and overlay source curves/contours using PyVista.",
    )
    parser.add_argument(
        "--show-mesh-edges",
        action="store_true",
        help="Show triangle edges in the PyVista preview.",
    )
    parser.add_argument(
        "--curve-line-width",
        type=float,
        default=5.0,
        help="Line width for source-curve overlay in preview mode.",
    )
    parser.add_argument(
        "--contour-line-width",
        type=float,
        default=3.0,
        help="Line width for contour overlay in preview mode.",
    )
    return parser


def _point_key(point: FloatArray, tolerance: float) -> tuple[int, int, int]:
    scaled = np.asarray(point, dtype=np.float64) / tolerance
    return tuple(np.round(scaled).astype(np.int64))


def _dedupe_curve_points(points: list[CurvePoint], tolerance: float) -> list[CurvePoint]:
    deduped: list[CurvePoint] = []
    for point in points:
        if not any(np.linalg.norm(point.xyz - existing.xyz) <= tolerance for existing in deduped):
            deduped.append(point)
    return deduped


def _intersect_triangle_with_plane(
    tri_points: FloatArray,
    axis_index: int,
    plane_value: float,
    face_index: int,
    *,
    tolerance: float,
) -> list[CurvePoint]:
    values = tri_points[:, axis_index] - plane_value
    edge_pairs = ((0, 1), (1, 2), (2, 0))
    intersections: list[CurvePoint] = []

    for i0, i1 in edge_pairs:
        s0 = float(values[i0])
        s1 = float(values[i1])

        if abs(s0) <= tolerance and abs(s1) <= tolerance:
            # Coplanar edge; ignore for this first test utility.
            continue

        if abs(s0) <= tolerance:
            bary = np.zeros(3, dtype=np.float64)
            bary[i0] = 1.0
            intersections.append(
                CurvePoint(xyz=np.asarray(tri_points[i0], dtype=np.float64), face_index=face_index, barycentric=bary),
            )
            continue

        if abs(s1) <= tolerance:
            bary = np.zeros(3, dtype=np.float64)
            bary[i1] = 1.0
            intersections.append(
                CurvePoint(xyz=np.asarray(tri_points[i1], dtype=np.float64), face_index=face_index, barycentric=bary),
            )
            continue

        if s0 * s1 > 0.0:
            continue

        t = s0 / (s0 - s1)
        t = min(1.0, max(0.0, t))
        xyz = (1.0 - t) * tri_points[i0] + t * tri_points[i1]
        bary = np.zeros(3, dtype=np.float64)
        bary[i0] = 1.0 - t
        bary[i1] = t
        intersections.append(
            CurvePoint(xyz=np.asarray(xyz, dtype=np.float64), face_index=face_index, barycentric=bary),
        )

    intersections = _dedupe_curve_points(intersections, tolerance)
    if len(intersections) <= 2:
        return intersections

    max_pair: tuple[CurvePoint, CurvePoint] | None = None
    max_distance = -1.0
    for idx, first in enumerate(intersections):
        for second in intersections[idx + 1 :]:
            distance = float(np.linalg.norm(first.xyz - second.xyz))
            if distance > max_distance:
                max_distance = distance
                max_pair = (first, second)

    if max_pair is None:
        return []
    return [max_pair[0], max_pair[1]]


def intersect_mesh_with_axis_plane(
    V: FloatArray,
    F: np.ndarray,
    axis_index: int,
    plane_value: float,
    *,
    tolerance: float,
) -> list[tuple[CurvePoint, CurvePoint]]:
    segments: list[tuple[CurvePoint, CurvePoint]] = []
    for face_index, tri in enumerate(np.asarray(F, dtype=np.int64)):
        tri_points = np.asarray(V[tri], dtype=np.float64)
        intersections = _intersect_triangle_with_plane(
            tri_points,
            axis_index,
            plane_value,
            face_index,
            tolerance=tolerance,
        )
        if len(intersections) == 2:
            segments.append((intersections[0], intersections[1]))
    return segments


def stitch_segments_into_curves(
    segments: list[tuple[CurvePoint, CurvePoint]],
    *,
    tolerance: float,
) -> list[list[CurvePoint]]:
    if not segments:
        return []

    point_by_key: dict[tuple[int, int, int], CurvePoint] = {}
    adjacency: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for start, end in segments:
        key0 = _point_key(start.xyz, tolerance)
        key1 = _point_key(end.xyz, tolerance)
        point_by_key.setdefault(key0, start)
        point_by_key.setdefault(key1, end)
        adjacency.setdefault(key0, []).append(key1)
        adjacency.setdefault(key1, []).append(key0)

    visited_edges: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
    curves: list[list[CurvePoint]] = []

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
                if len(candidates) > 1:
                    raise ValueError("Plane intersection graph is branching; cannot form a single ordered curve.")
                next_key = candidates[0]
                ordered_keys.append(next_key)
                mark_edge(current_key, next_key)
                prev_key, current_key = current_key, next_key
                if current_key == start_key:
                    break

            curves.append([point_by_key[key] for key in ordered_keys])

    return curves


def curve_to_potpourri_payload(curve: list[CurvePoint]) -> list[tuple[int, list[float]]]:
    return [
        (
            int(point.face_index),
            [float(point.barycentric[0]), float(point.barycentric[1]), float(point.barycentric[2])],
        )
        for point in curve
    ]


def build_solver_curve_payloads(
    stitched_curves: list[list[CurvePoint]],
    segments: list[tuple[CurvePoint, CurvePoint]],
    *,
    tolerance: float,
) -> list[list[tuple[int, list[float]]]]:
    """
    Build potpourri3d-compatible source curves.

    The signed-heat solver expects each consecutive source segment to lie
    entirely within a single face. Our stitched curves are useful for global
    ordering/preview, but their consecutive points often cross face boundaries.
    Therefore we submit each face-local intersection segment as its own 2-point
    curve, while inheriting orientation from the stitched ordering.
    """
    segment_map: dict[tuple[tuple[int, int, int], tuple[int, int, int]], tuple[CurvePoint, CurvePoint]] = {}
    for start, end in segments:
        key0 = _point_key(start.xyz, tolerance)
        key1 = _point_key(end.xyz, tolerance)
        undirected_key = tuple(sorted((key0, key1)))
        segment_map[undirected_key] = (start, end)

    payloads: list[list[tuple[int, list[float]]]] = []
    for curve in stitched_curves:
        if len(curve) < 2:
            continue
        for start_point, end_point in zip(curve[:-1], curve[1:]):
            key0 = _point_key(start_point.xyz, tolerance)
            key1 = _point_key(end_point.xyz, tolerance)
            undirected_key = tuple(sorted((key0, key1)))
            face_segment = segment_map.get(undirected_key)
            if face_segment is None:
                raise RuntimeError("Failed to map a stitched curve edge back to a face-local segment.")

            seg_start, seg_end = face_segment
            seg_key0 = _point_key(seg_start.xyz, tolerance)
            seg_key1 = _point_key(seg_end.xyz, tolerance)
            if key0 == seg_key0 and key1 == seg_key1:
                ordered_segment = [seg_start, seg_end]
            elif key0 == seg_key1 and key1 == seg_key0:
                ordered_segment = [seg_end, seg_start]
            else:
                raise RuntimeError("Stitched curve ordering does not match the face-local segment endpoints.")
            payloads.append(curve_to_potpourri_payload(ordered_segment))

    return payloads


def save_curves_obj(path: Path, curves: list[list[CurvePoint]], *, tolerance: float) -> None:
    lines: list[str] = []
    vertex_offset = 1
    for curve in curves:
        for point in curve:
            x, y, z = point.xyz.tolist()
            lines.append(f"v {x:.10f} {y:.10f} {z:.10f}")
        indices = list(range(vertex_offset, vertex_offset + len(curve)))
        closed = len(curve) >= 2 and np.linalg.norm(curve[0].xyz - curve[-1].xyz) <= tolerance
        if closed:
            indices = indices[:-1]
            if indices:
                indices.append(indices[0])
        if len(indices) >= 2:
            lines.append("l " + " ".join(str(idx) for idx in indices))
        vertex_offset += len(curve)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def save_contour_polylines_txt(
    output_dir: Path,
    contours: list,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for level_index, contour_level in enumerate(contours):
        for polyline_index, polyline in enumerate(contour_level.polylines):
            out_path = output_dir / f"level_{level_index:04d}_poly_{polyline_index:04d}.txt"
            np.savetxt(out_path, np.asarray(polyline, dtype=np.float64), fmt="%.10f")


def build_test_contours(
    V: FloatArray,
    F: np.ndarray,
    scalar_field: FloatArray,
    spacing: float,
    *,
    signed_field: bool,
    max_level_fraction: float | None,
    tolerance: float,
) -> list[ContourLevel]:
    values = np.asarray(scalar_field, dtype=np.float64)
    data_min = float(np.min(values))
    data_max = float(np.max(values))
    if data_max - data_min <= spacing + tolerance:
        return []

    lower = data_min + 0.5 * spacing
    upper = data_max - 0.5 * spacing
    if max_level_fraction is not None:
        center = 0.5 * (data_min + data_max)
        half_extent = 0.5 * (data_max - data_min) * max_level_fraction
        lower = max(lower, center - half_extent)
        upper = min(upper, center + half_extent)

    if upper < lower - tolerance:
        return []

    if signed_field:
        level_values: list[float] = []
        level = lower
        while level <= upper + tolerance:
            if abs(level) > tolerance:
                level_values.append(float(level))
            level += spacing
        levels = np.asarray(level_values, dtype=np.float64)
    else:
        first_level = max(lower, spacing)
        if upper < first_level - tolerance:
            return []
        level_values = [float(first_level)]
        level = first_level + spacing
        while level <= upper + tolerance:
            level_values.append(float(level))
            level += spacing
        levels = np.asarray(level_values, dtype=np.float64)

    contours: list[ContourLevel] = []
    for level in levels:
        polylines = extract_isocontour_polylines(
            V,
            F,
            values,
            float(level),
            tolerance=tolerance,
        )
        contours.append(ContourLevel(level=float(level), polylines=polylines))

    return contours


def build_polyline_polydata(points: FloatArray, *, close_loop: bool = False, tolerance: float = 1e-8) -> "pv.PolyData":
    if pv is None:
        raise ImportError("PyVista is required for preview mode.")
    poly_points = np.asarray(points, dtype=np.float64)
    if poly_points.ndim != 2 or poly_points.shape[1] != 3 or poly_points.shape[0] < 2:
        raise ValueError("Polyline points must have shape (k, 3) with k >= 2.")

    if close_loop and np.linalg.norm(poly_points[0] - poly_points[-1]) <= tolerance:
        line_indices = np.arange(poly_points.shape[0] - 1, dtype=np.int64)
        poly_points = poly_points[:-1]
    else:
        line_indices = np.arange(poly_points.shape[0], dtype=np.int64)

    line_cells = np.concatenate(([line_indices.size], line_indices)).astype(np.int64)
    poly = pv.PolyData(poly_points)
    poly.lines = line_cells
    return poly


def preview_field_and_curves(
    V: FloatArray,
    F: np.ndarray,
    distances: FloatArray,
    curves: list[list[CurvePoint]],
    contours: list | None,
    *,
    show_mesh_edges: bool,
    curve_line_width: float,
    contour_line_width: float,
    tolerance: float,
) -> None:
    if pv is None:
        raise ImportError("PyVista is required for preview mode.")

    mesh = to_pyvista_polydata(V, F)
    mesh["distance"] = np.asarray(distances, dtype=np.float64)

    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars="distance",
        cmap="viridis",
        show_edges=show_mesh_edges,
        scalar_bar_args={"title": "Signed Heat Field"},
    )

    for curve in curves:
        curve_points = np.asarray([point.xyz for point in curve], dtype=np.float64)
        closed = curve_points.shape[0] >= 2 and np.linalg.norm(curve_points[0] - curve_points[-1]) <= tolerance
        curve_poly = build_polyline_polydata(curve_points, close_loop=closed, tolerance=tolerance)
        plotter.add_mesh(curve_poly, color="red", line_width=curve_line_width)

    if contours is not None:
        for contour_level in contours:
            for polyline in contour_level.polylines:
                poly_points = np.asarray(polyline, dtype=np.float64)
                if poly_points.shape[0] < 2:
                    continue
                contour_poly = build_polyline_polydata(
                    poly_points,
                    close_loop=bool(np.linalg.norm(poly_points[0] - poly_points[-1]) <= tolerance),
                    tolerance=tolerance,
                )
                plotter.add_mesh(contour_poly, color="white", line_width=contour_line_width)

    plotter.show()


def main() -> None:
    args = build_parser().parse_args()
    if pp3d is None:
        raise ImportError(
            "This test script requires the optional `potpourri3d` package to be installed.",
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    V, F = load_obj_mesh(str(args.mesh_path))
    axis_index = AXIS_TO_INDEX[args.axis]
    segments = intersect_mesh_with_axis_plane(
        V,
        F,
        axis_index,
        args.value,
        tolerance=args.tolerance,
    )
    if not segments:
        raise RuntimeError("Plane/mesh intersection produced no segments.")

    curves = stitch_segments_into_curves(segments, tolerance=args.tolerance)
    if not curves:
        raise RuntimeError("Failed to stitch plane-intersection segments into ordered curves.")

    solver = pp3d.MeshSignedHeatSolver(V, F)
    curve_payloads = build_solver_curve_payloads(curves, segments, tolerance=args.tolerance)
    signed_flags = [bool(args.signed)] * len(curve_payloads)
    distances = solver.compute_distance(
        curve_payloads,
        signed_flags,
        [],
        preserve_source_normals=bool(args.preserve_source_normals),
        level_set_constraint=args.level_set_constraint,
        soft_level_set_weight=float(args.soft_level_set_weight),
    )
    distances = np.asarray(distances, dtype=np.float64)

    source_curves_path = output_dir / "source_curves.obj"
    save_curves_obj(source_curves_path, curves, tolerance=args.tolerance)

    distances_path = output_dir / "vertex_distances.txt"
    np.savetxt(distances_path, distances, fmt="%.10f")

    summary = {
        "mesh_path": str(args.mesh_path),
        "axis": args.axis,
        "value": args.value,
        "signed": bool(args.signed),
        "curve_count": len(curves),
        "segment_count": len(segments),
        "vertex_count": int(V.shape[0]),
        "face_count": int(F.shape[0]),
        "distance_min": float(np.min(distances)),
        "distance_max": float(np.max(distances)),
        "source_curves_obj": source_curves_path.name,
        "vertex_distances_txt": distances_path.name,
    }

    contours = None
    if args.contour_spacing is not None:
        contours = build_test_contours(
            V,
            F,
            distances,
            float(args.contour_spacing),
            signed_field=bool(args.signed),
            max_level_fraction=args.contour_max_level_fraction,
            tolerance=args.tolerance,
        )
        contour_dir = output_dir / "contours"
        save_contour_polylines_txt(contour_dir, contours)
        summary["contour_spacing"] = float(args.contour_spacing)
        summary["contour_dir"] = contour_dir.name
        summary["contour_level_count"] = len(contours)
        summary["contour_polyline_count"] = int(sum(len(level.polylines) for level in contours))

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Source curves: {source_curves_path}")
    print(f"Vertex distances: {distances_path}")
    print(f"Summary: {summary_path}")
    print(f"Curve count: {len(curves)}")
    print(f"Distance range: [{float(np.min(distances)):.6f}, {float(np.max(distances)):.6f}]")

    if args.preview:
        preview_field_and_curves(
            V,
            F,
            distances,
            curves,
            contours,
            show_mesh_edges=bool(args.show_mesh_edges),
            curve_line_width=float(args.curve_line_width),
            contour_line_width=float(args.contour_line_width),
            tolerance=float(args.tolerance),
        )


if __name__ == "__main__":
    main()
