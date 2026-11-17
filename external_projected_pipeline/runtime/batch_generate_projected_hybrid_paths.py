"""Fast experimental hybrid planner in the nozzle-perpendicular XZ plane.

The planner assumes each input layer is locally representable as ``y = f(x, z)``.
Boundary rings and raster lines are planned in XZ, then lifted through the
projected mesh triangles. This intentionally differs from the intrinsic
surface-field hybrid planner and avoids heat/Poisson solves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from extrusion_height import apply_axis_lift
from test_hybrid_contours import (
    HybridPolyline,
    chain_boundary_family,
    chain_parallel_family,
    concatenate_paths,
    concatenate_path_tags,
    concatenate_width_profiles,
    encode_path_tag_blocks,
    filter_boundary_items_by_start_offset,
)
from toolpath_export import compute_path_normals, save_oriented_path, save_numeric_array
from connected_patch_planner import plan_connected_patches, planner_mesh

try:
    from shapely import points as shapely_points
    from shapely import polygons as shapely_polygons
    from shapely import affinity
    from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Point, Polygon
    from shapely.ops import nearest_points, unary_union
    from shapely.strtree import STRtree
except ImportError as exc:  # pragma: no cover - environment dependency
    raise ImportError(
        "Projected hybrid planning requires Shapely. Install it with "
        "`python -m pip install shapely` in the selected runtime."
    ) from exc


XZ = np.asarray((0, 2), dtype=np.int64)
# Millimetres: omit only wholly projection-degenerate components no larger
# than the current 0.5 mm waypoint spacing. Valid projected features of any
# size remain eligible; larger edge-on components still require inspection.
DEGENERATE_ARTIFACT_MAX_DIAGONAL_MM = 0.5


def _ring_area(coords: np.ndarray) -> float:
    pts = np.asarray(coords, dtype=np.float64)
    if pts.shape[0] < 3:
        return 0.0
    return 0.5 * float(np.sum(pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1]))


def _boundary_loops(vertices: np.ndarray, faces: np.ndarray) -> list[np.ndarray]:
    """Return closed XZ boundary loops by tracing mesh edges used by one face."""
    edge_counts: dict[tuple[int, int], int] = {}
    for a, b, c in np.asarray(faces, dtype=np.int64):
        for i, j in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            edge = (i, j) if i < j else (j, i)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    adjacency: dict[int, list[int]] = {}
    for (i, j), count in edge_counts.items():
        if count == 1:
            adjacency.setdefault(i, []).append(j)
            adjacency.setdefault(j, []).append(i)
    if not adjacency:
        raise ValueError("Projected hybrid planning needs an open layer mesh with boundary edges.")
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError("Layer boundary graph branches; it cannot be converted to a planar polygon.")

    unused = {tuple(sorted((i, j))) for i, neighbors in adjacency.items() for j in neighbors}
    loops: list[np.ndarray] = []
    while unused:
        first_edge = next(iter(unused))
        start, current = first_edge
        previous = start
        indices = [start]
        while True:
            indices.append(current)
            edge = tuple(sorted((previous, current)))
            unused.discard(edge)
            candidates = [node for node in adjacency[current] if node != previous]
            if not candidates:
                break
            nxt = candidates[0]
            if nxt == start:
                unused.discard(tuple(sorted((current, nxt))))
                indices.append(start)
                break
            previous, current = current, nxt
            if len(indices) > len(adjacency) + 1:
                raise ValueError("Failed to close a projected boundary loop.")
        if len(indices) >= 4 and indices[0] == indices[-1]:
            loops.append(np.asarray(vertices[np.asarray(indices, dtype=np.int64)][:, XZ], dtype=np.float64))
    if not loops:
        raise ValueError("No closed boundary loops were found in the layer mesh.")
    return loops


def _planar_region_from_loops(loops: list[np.ndarray]) -> Polygon | MultiPolygon:
    """Build a polygon-with-holes from nested projected mesh-boundary loops."""
    polygons = [Polygon(loop) for loop in loops]
    if any(not polygon.is_valid or polygon.area <= 1e-12 for polygon in polygons):
        raise ValueError("Projected boundary contains an invalid or degenerate loop.")
    parents: list[int | None] = []
    for index, polygon in enumerate(polygons):
        point = polygon.representative_point()
        containers = [other for other, candidate in enumerate(polygons) if other != index and candidate.contains(point)]
        parents.append(min(containers, key=lambda other: polygons[other].area) if containers else None)
    depths: list[int] = []
    for index in range(len(polygons)):
        depth = 0
        parent = parents[index]
        while parent is not None:
            depth += 1
            parent = parents[parent]
        depths.append(depth)
    shells: list[Polygon] = []
    for index, polygon in enumerate(polygons):
        if depths[index] % 2:
            continue
        holes = [
            np.asarray(polygons[child].exterior.coords, dtype=np.float64)
            for child in range(len(polygons))
            if parents[child] == index and depths[child] == depths[index] + 1
        ]
        shells.append(Polygon(np.asarray(polygon.exterior.coords, dtype=np.float64), holes=holes))
    region = unary_union(shells)
    if region.is_empty or not isinstance(region, (Polygon, MultiPolygon)):
        raise ValueError("Projected boundary did not produce a usable polygonal region.")
    return region


def _planar_region_from_projected_triangles(lifter: "ProjectedSurfaceLifter", *, validate_projection=True) -> Polygon | MultiPolygon:
    """Use the actual projected triangle coverage, not only its boundary loops.

    A projected boundary loop can enclose an area that is uncovered when a
    layer folds or has a non-injective projection. Planning on the union of
    projected faces guarantees every planned 2D point has a candidate face to
    lift back onto.
    """
    if lifter.degenerate_artifact is not None:
        return Polygon()
    from surface_hatch_metric import assert_single_sheet
    # Marching/trimmed layer meshes can contain sub-micron duplicate facets.
    # Treat heights within 0.001 mm as coincident geometry, independently of
    # printer resolution. Larger conflicts still require separate charts.
    if validate_projection:assert_single_sheet(lifter,tolerance=.001)
    region = unary_union(lifter.projected_triangle_polygons)
    if not region.is_valid:
        region = region.buffer(0.0)
    if region.is_empty or not isinstance(region, (Polygon, MultiPolygon)):
        raise ValueError("Projected mesh triangles did not produce a usable polygonal region.")
    return region


def _iter_polygons(geometry: object) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        return [polygon for part in geometry.geoms for polygon in _iter_polygons(part)]
    return []


def _covered_inset(region: Polygon | MultiPolygon, distance: float):
    """Keep an inward offset inside its original projected triangle footprint."""
    if distance < 0.0:
        raise ValueError('Inset distance must be non-negative.')
    if distance == 0.0:
        return region
    # GEOS offsets of narrow/sliver boundaries can leave small polygon islands
    # outside the source. Such islands cannot be lifted onto the source mesh.
    inset = region.buffer(-distance, join_style=1).intersection(region)
    return unary_union(_iter_polygons(inset))


def _iter_lines(geometry: object) -> list[LineString]:
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, MultiLineString):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        result: list[LineString] = []
        for part in geometry.geoms:
            result.extend(_iter_lines(part))
        return result
    return []


def _sample_open(points: np.ndarray, spacing: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return pts.copy()
    lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(np.sum(lengths))
    if total <= 1e-12:
        return pts[:1].copy()
    distances = np.arange(0.0, total, spacing, dtype=np.float64)
    if distances.size == 0 or total - distances[-1] > 1e-9:
        distances = np.append(distances, total)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    # Retain corners: replacing a boundary arc by chords between evenly
    # spaced samples can cross a hole or leave a concave printable region.
    distances=np.unique(np.r_[distances,cumulative])
    edges = np.minimum(np.searchsorted(cumulative, distances, side="right") - 1, lengths.size - 1)
    local = np.divide(distances - cumulative[edges], lengths[edges],
                      out=np.zeros_like(distances), where=lengths[edges] > 1e-12)
    return (1.0 - local[:, None]) * pts[edges] + local[:, None] * pts[edges + 1]


def _sample_closed(points: np.ndarray, spacing: float) -> np.ndarray:
    ring = np.asarray(points, dtype=np.float64)
    if ring.shape[0] < 4:
        return ring.copy()
    sampled = _sample_open(ring, spacing)
    if np.linalg.norm(sampled[0] - sampled[-1]) > 1e-9:
        sampled = np.vstack((sampled, sampled[0]))
    if sampled.shape[0] < 4:
        # A closed ring needs three distinct vertices even if its perimeter is
        # shorter than the configured pitch. Keep valid small islands intact.
        lengths = np.linalg.norm(np.diff(ring, axis=0), axis=1)
        cumulative = np.r_[0.0, np.cumsum(lengths)]
        if cumulative[-1] <= 1e-12:
            return ring[:1].copy()
        distances = np.linspace(0.0, cumulative[-1], 3, endpoint=False)
        edges = np.minimum(np.searchsorted(cumulative, distances, side='right') - 1, len(lengths) - 1)
        fractions = (distances - cumulative[edges]) / lengths[edges]
        sampled = ring[edges] + fractions[:, None] * (ring[edges + 1] - ring[edges])
        sampled = np.vstack((sampled, sampled[0]))
    return sampled


@dataclass(slots=True)
class ProjectedSurfaceLifter:
    vertices: np.ndarray
    faces: np.ndarray
    face_indices: np.ndarray
    projected_triangles: np.ndarray
    projected_triangle_polygons: np.ndarray
    tree: STRtree
    degenerate_artifact: dict | None = None

    @classmethod
    def build(cls, vertices: np.ndarray, faces: np.ndarray) -> "ProjectedSurfaceLifter":
        projected = np.asarray(vertices[:, XZ], dtype=np.float64)
        all_triangles = projected[np.asarray(faces, dtype=np.int64)]
        edge_a=all_triangles[:,1]-all_triangles[:,0]
        edge_b=all_triangles[:,2]-all_triangles[:,0]
        twice_areas=edge_a[:,0]*edge_b[:,1]-edge_a[:,1]*edge_b[:,0]
        valid = np.abs(twice_areas) > 1e-12
        artifact = None
        if not np.any(valid):
            diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
            if not np.isfinite(diagonal) or diagonal > DEGENERATE_ARTIFACT_MAX_DIAGONAL_MM:
                raise ValueError("All projected mesh triangles are degenerate in XZ. "
                                 f"Surface diagonal {diagonal:g} mm exceeds the "
                                 f"{DEGENERATE_ARTIFACT_MAX_DIAGONAL_MM:g} mm degenerate-component limit.")
            triangles_3d = np.asarray(vertices)[faces]
            artifact = dict(reason='small_degenerate_projection', source_faces=len(faces),
                source_vertices=len(vertices), diagonal_mm=diagonal,
                maximum_diagonal_mm=DEGENERATE_ARTIFACT_MAX_DIAGONAL_MM,
                projected_area_mm2=float(np.abs(twice_areas).sum()/2),
                surface_area_mm2=float(np.linalg.norm(np.cross(triangles_3d[:,1]-triangles_3d[:,0],
                    triangles_3d[:,2]-triangles_3d[:,0]),axis=1).sum()/2))
        face_indices = np.flatnonzero(valid)
        triangles = all_triangles[valid]
        triangle_polygons = shapely_polygons(triangles)
        return cls(
            vertices=np.asarray(vertices, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            face_indices=face_indices,
            projected_triangles=triangles,
            projected_triangle_polygons=triangle_polygons,
            tree=STRtree(triangle_polygons),
            degenerate_artifact=artifact,
        )

    def lift(self, points_xz: np.ndarray) -> tuple[np.ndarray, int, int]:
        points = np.asarray(points_xz, dtype=np.float64)
        lifted = np.empty((points.shape[0], 3), dtype=np.float64)
        if points.shape[0] == 0:
            return lifted, 0, 0

        # Shapely 2 bulk-query removes one Python STRtree call per path point.
        candidate_pairs = np.asarray(self.tree.query(shapely_points(points[:, 0], points[:, 1])), dtype=np.int64)
        point_indices = candidate_pairs[0] if candidate_pairs.size else np.empty(0, dtype=np.int64)
        local_indices = candidate_pairs[1] if candidate_pairs.size else np.empty(0, dtype=np.int64)
        accepted = np.zeros(points.shape[0], dtype=bool)
        accepted_y_min = np.full(points.shape[0], np.inf, dtype=np.float64)
        accepted_y_max = np.full(points.shape[0], -np.inf, dtype=np.float64)

        if point_indices.size:
            triangles = self.projected_triangles[local_indices]
            edge_u = triangles[:, 1] - triangles[:, 0]
            edge_v = triangles[:, 2] - triangles[:, 0]
            offsets = points[point_indices] - triangles[:, 0]
            determinants = edge_u[:, 0] * edge_v[:, 1] - edge_u[:, 1] * edge_v[:, 0]
            nondegenerate = np.abs(determinants) > 1e-16
            beta = np.zeros(point_indices.size, dtype=np.float64)
            gamma = np.zeros(point_indices.size, dtype=np.float64)
            beta[nondegenerate] = (
                offsets[nondegenerate, 0] * edge_v[nondegenerate, 1]
                - offsets[nondegenerate, 1] * edge_v[nondegenerate, 0]
            ) / determinants[nondegenerate]
            gamma[nondegenerate] = (
                edge_u[nondegenerate, 0] * offsets[nondegenerate, 1]
                - edge_u[nondegenerate, 1] * offsets[nondegenerate, 0]
            ) / determinants[nondegenerate]
            barycentric = np.column_stack((1.0 - beta - gamma, beta, gamma))
            scores = np.min(barycentric, axis=1)
            valid = nondegenerate & (scores >= -1e-7)
            if np.any(valid):
                valid_points = point_indices[valid]
                valid_local = local_indices[valid]
                valid_scores = scores[valid]
                valid_barycentric = barycentric[valid]
                valid_xyz = np.einsum(
                    "ij,ijk->ik",
                    valid_barycentric,
                    self.vertices[self.faces[self.face_indices[valid_local]]],
                )
                np.minimum.at(accepted_y_min, valid_points, valid_xyz[:, 1])
                np.maximum.at(accepted_y_max, valid_points, valid_xyz[:, 1])

                # Preserve the original rule: largest minimum barycentric weight,
                # with the lowest local triangle index breaking exact ties.
                order = np.lexsort((valid_local, -valid_scores, valid_points))
                first_for_point = np.empty(order.size, dtype=bool)
                first_for_point[0] = True
                first_for_point[1:] = valid_points[order][1:] != valid_points[order][:-1]
                selected = order[first_for_point]
                lifted[valid_points[selected]] = valid_xyz[selected]
                accepted[valid_points[selected]] = True

        ambiguous = int(np.count_nonzero((accepted_y_max - accepted_y_min) > 1e-5))
        snapped = 0
        for point_index in np.flatnonzero(~accepted):
            point_xz = points[point_index]
            candidate_indices = np.asarray(self.tree.query(Point(float(point_xz[0]), float(point_xz[1]))), dtype=np.int64)
            best: tuple[float, int, np.ndarray] | None = None
            for local_index in candidate_indices:
                triangle = self.projected_triangles[int(local_index)]
                matrix = np.column_stack((triangle[1] - triangle[0], triangle[2] - triangle[0]))
                try:
                    beta_gamma = np.linalg.solve(matrix, point_xz - triangle[0])
                except np.linalg.LinAlgError:
                    continue
                barycentric = np.asarray((1.0 - beta_gamma[0] - beta_gamma[1], beta_gamma[0], beta_gamma[1]), dtype=np.float64)
                score = float(np.min(barycentric))
                if score < -1e-7:
                    continue
                face_index = int(self.face_indices[int(local_index)])
                xyz = barycentric @ self.vertices[self.faces[face_index]]
                candidate = (score, face_index, xyz)
                if best is None or candidate[0] > best[0]:
                    best = candidate
            if best is None:
                # Boolean polygon operations can move a boundary point by a few
                # floating-point ulps. Snap only within a negligible XZ distance.
                nearest_local_index = int(self.tree.nearest(Point(float(point_xz[0]), float(point_xz[1]))))
                nearest_geometry = self.tree.geometries[nearest_local_index]
                nearest_projected_point = nearest_points(
                    Point(float(point_xz[0]), float(point_xz[1])),
                    nearest_geometry,
                )[1]
                nearest_xz = np.asarray(nearest_projected_point.coords[0], dtype=np.float64)
                snap_distance = float(np.linalg.norm(nearest_xz - point_xz))
                if snap_distance > 1e-6:
                    raise ValueError(
                        f"Could not lift projected point {point_xz.tolist()} onto a mesh triangle "
                        f"(nearest projected triangle is {snap_distance:.6g} away).",
                    )
                triangle = self.projected_triangles[nearest_local_index]
                matrix = np.column_stack((triangle[1] - triangle[0], triangle[2] - triangle[0]))
                beta_gamma = np.linalg.solve(matrix, nearest_xz - triangle[0])
                barycentric = np.asarray((1.0 - beta_gamma[0] - beta_gamma[1], beta_gamma[0], beta_gamma[1]), dtype=np.float64)
                face_index = int(self.face_indices[nearest_local_index])
                best = (float(np.min(barycentric)), face_index, barycentric @ self.vertices[self.faces[face_index]])
                snapped += 1
            lifted[point_index] = best[2]
        return lifted, ambiguous, snapped


def _subdivide_3d(points: np.ndarray, max_spacing: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return pts.copy()
    delta = pts[1:] - pts[:-1]
    # Match the scalar norm reduction used previously, including edges exactly
    # at the spacing boundary, so subdivision counts and path sidecars agree.
    lengths = np.sqrt(np.einsum('ij,ij->i', delta, delta))
    ratios = lengths / max_spacing
    boundary = np.abs(ratios - np.rint(ratios)) <= 8 * np.finfo(float).eps * np.maximum(1., np.abs(ratios))
    for edge in np.flatnonzero(boundary):
        lengths[edge] = np.linalg.norm(delta[edge])
    counts = np.maximum(1, np.ceil(lengths / max_spacing).astype(np.int64))
    edges = np.repeat(np.arange(len(counts)), counts)
    indices = np.arange(len(edges)) - np.repeat(np.cumsum(counts) - counts, counts) + 1
    fractions = indices / counts[edges]
    return np.vstack((pts[:1], pts[edges] + fractions[:, None] * delta[edges]))


def _ring_items(region: Polygon | MultiPolygon, level: float, spacing: float, lifter: ProjectedSurfaceLifter,
                *, min_points_per_contour: int | None = None, min_resample_spacing: float = 0.0) -> tuple[list[HybridPolyline], int, int]:
    items: list[HybridPolyline] = []
    ambiguous = 0
    snapped = 0
    for polygon in _iter_polygons(region):
        rings = [polygon.exterior, *polygon.interiors]
        for ring in rings:
            ring_spacing = spacing
            if min_points_per_contour is not None:
                ring_spacing = max(min_resample_spacing, min(spacing, ring.length / min_points_per_contour))
            if ring_spacing <= 0.0:
                continue
            sampled = _sample_closed(np.asarray(ring.coords, dtype=np.float64), ring_spacing)
            lifted, count, snap_count = lifter.lift(sampled)
            ambiguous += count
            snapped += snap_count
            # Offset polygons can collapse to nanometre-sized closed rings.
            # Reject them before chaining: they round to one machine point,
            # and must not masquerade as deposited support in later screens.
            if len(lifted) < 4 or np.linalg.norm(np.diff(lifted, axis=0), axis=1).sum() <= 1e-6:
                continue
            items.append(HybridPolyline(points=_subdivide_3d(lifted, ring_spacing), family="boundary", level=level, closed=True, effective_width=spacing))
    return items, ambiguous, snapped


def _raster_items(region: Polygon | MultiPolygon, *, spacing: float, direction: str, lifter: ProjectedSurfaceLifter) -> tuple[list[HybridPolyline], int, int]:
    if direction not in {"x", "z"}:
        raise ValueError("projected_raster_direction must be 'x' or 'z'.")
    min_x, min_z, max_x, max_z = region.bounds
    scan_min, scan_max = (min_z, max_z) if direction == "x" else (min_x, max_x)
    length_min, length_max = (min_x, max_x) if direction == "x" else (min_z, max_z)
    start = np.floor(scan_min / spacing) * spacing
    scan_values = np.arange(start, scan_max + 0.5 * spacing, spacing, dtype=np.float64)
    margin = max(spacing, 1.0)
    items: list[HybridPolyline] = []
    ambiguous = 0
    snapped = 0
    for scan_value in scan_values:
        line = LineString([(length_min - margin, scan_value), (length_max + margin, scan_value)]) if direction == "x" else LineString([(scan_value, length_min - margin), (scan_value, length_max + margin)])
        for segment in _iter_lines(region.intersection(line)):
            coordinates = np.asarray(segment.coords, dtype=np.float64)
            if coordinates.shape[0] < 2 or float(segment.length) <= 1e-9:
                continue
            sampled = _sample_open(coordinates, spacing)
            lifted, count, snap_count = lifter.lift(sampled)
            ambiguous += count
            snapped += snap_count
            items.append(HybridPolyline(points=_subdivide_3d(lifted, spacing), family="parallel", level=float(scan_value), closed=False, effective_width=spacing))
    return items, ambiguous, snapped


def _angled_raster_items(
    region: Polygon | MultiPolygon,
    *,
    spacing: float,
    angle_degrees: float,
    lifter: ProjectedSurfaceLifter,
) -> tuple[list[HybridPolyline], int, int]:
    """Clip an arbitrary-angle XZ raster, then lift its unrotated points."""
    origin = (0.0, 0.0)
    scan_region = affinity.rotate(region, -angle_degrees, origin=origin)
    min_x, min_z, max_x, max_z = scan_region.bounds
    start = np.floor(min_z / spacing) * spacing
    scan_values = np.arange(start, max_z + 0.5 * spacing, spacing, dtype=np.float64)
    margin = max(spacing, 1.0)
    items: list[HybridPolyline] = []
    ambiguous = 0
    snapped = 0
    for scan_value in scan_values:
        line = LineString([(min_x - margin, scan_value), (max_x + margin, scan_value)])
        for segment in _iter_lines(scan_region.intersection(line)):
            if float(segment.length) <= 1e-9:
                continue
            unrotated = np.asarray(
                affinity.rotate(segment, angle_degrees, origin=origin).coords,
                dtype=np.float64,
            )
            sampled = _sample_open(unrotated, spacing)
            lifted, count, snap_count = lifter.lift(sampled)
            ambiguous += count
            snapped += snap_count
            items.append(HybridPolyline(points=_subdivide_3d(lifted, spacing), family="ironing", level=float(scan_value), closed=False, effective_width=spacing))
    return items, ambiguous, snapped


def _lift_ironing_parts_for_tip_clearance(
    parts: list[np.ndarray],
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    nozzle_tip_diameter: float,
    max_lift: float,
) -> tuple[list[np.ndarray], dict[str, float | int]]:
    """Raise only finishing paths so the finite nozzle rim clears cross-path slope."""
    radius = 0.5 * nozzle_tip_diameter
    lifted_parts: list[np.ndarray] = []
    all_lifts: list[np.ndarray] = []
    for part in parts:
        points = np.asarray(part, dtype=np.float64).copy()
        lifts = np.zeros(points.shape[0], dtype=np.float64)
        if points.shape[0] >= 2:
            normals = compute_path_normals(vertices, faces, points)
            for index in range(points.shape[0]):
                tangent = points[min(index + 1, points.shape[0] - 1)] - points[max(index - 1, 0)]
                tangent_length = float(np.linalg.norm(tangent))
                normal_length = float(np.linalg.norm(normals[index]))
                if tangent_length <= 1e-12 or normal_length <= 1e-12:
                    continue
                tangent /= tangent_length
                normal = normals[index] / normal_length
                across_bead = np.cross(tangent, normal)
                across_length = float(np.linalg.norm(across_bead))
                if across_length <= 1e-12:
                    continue
                across_bead /= across_length
                horizontal_length = float(np.hypot(across_bead[0], across_bead[2]))
                lifts[index] = max_lift if horizontal_length <= 1e-12 else min(
                    max_lift,
                    radius * abs(float(across_bead[1])) / horizontal_length,
                )
        points[:, 1] += lifts
        lifted_parts.append(points)
        all_lifts.append(lifts)
    values = np.concatenate(all_lifts) if all_lifts else np.empty(0, dtype=np.float64)
    nonzero = values[values > 0.0]
    return lifted_parts, {
        "point_count": int(values.size),
        "lifted_point_count": int(nonzero.size),
        "mean_lift": float(nonzero.mean()) if nonzero.size else 0.0,
        "max_lift": float(values.max()) if values.size else 0.0,
    }


def _iter_polygon_rings(region: Polygon | MultiPolygon) -> list[LineString]:
    """Return exterior and hole rings that can safely join clipped infill."""
    polygons = [region] if isinstance(region, Polygon) else list(region.geoms)
    rings: list[LineString] = []
    for polygon in polygons:
        rings.append(LineString(polygon.exterior.coords))
        rings.extend(LineString(interior.coords) for interior in polygon.interiors)
    return rings


def _ring_arc(ring: LineString, start_distance: float, end_distance: float) -> np.ndarray:
    """Return the forward arc of a closed ring between the two distances."""
    length = float(ring.length)
    if length <= 1e-12:
        return np.empty((0, 2), dtype=np.float64)
    start = float(start_distance) % length
    end = float(end_distance) % length
    coordinates = np.asarray(ring.coords, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(coordinates, axis=0), axis=1))))
    if end >= start:
        indices = np.flatnonzero((cumulative[1:-1] > start + 1e-10) & (cumulative[1:-1] < end - 1e-10)) + 1
    else:
        # Preserve cyclic traversal across the ring origin. Sorting the union
        # of these two index ranges creates chords through concavities/holes.
        high=np.flatnonzero(cumulative[1:-1] > start + 1e-10)+1
        low=np.flatnonzero(cumulative[1:-1] < end - 1e-10)+1
        indices=np.r_[high,0,low]
    return np.vstack((
        np.asarray(ring.interpolate(start).coords[0], dtype=np.float64),
        coordinates[indices],
        np.asarray(ring.interpolate(end).coords[0], dtype=np.float64),
    ))


def _forward_ring_intervals(start: float, end: float, length: float) -> list[tuple[float, float]]:
    """Split a forward closed-ring arc into non-wrapping intervals."""
    start = float(start) % length
    end = float(end) % length
    if abs((end - start) % length) <= 1e-12:
        return []
    if end > start:
        return [(start, end)]
    return [(start, length), (0.0, end)]


def _ring_intervals_are_free(
    candidate: list[tuple[float, float]],
    consumed: list[tuple[float, float]],
    *,
    tolerance: float = 1e-8,
) -> bool:
    for candidate_start, candidate_end in candidate:
        for used_start, used_end in consumed:
            if min(candidate_end, used_end) - max(candidate_start, used_start) > tolerance:
                return False
    return True


def _link_lies_on_boundary(link: LineString, rings: list[LineString], *, tolerance: float = 1e-8) -> bool:
    return any(ring.buffer(tolerance).covers(link) for ring in rings)


@dataclass(frozen=True)
class _BoundaryConnector:
    points: np.ndarray
    ring_index: int
    intervals: tuple[tuple[float, float], ...]
    length: float


def _shortest_boundary_connector(
    start: np.ndarray,
    end: np.ndarray,
    *,
    rings: list[LineString],
    consumed_intervals: list[list[tuple[float, float]]],
    max_length: float | None,
    endpoint_tolerance: float = 1e-6,
) -> _BoundaryConnector | None:
    """Find the shortest unused same-ring route between two endpoints."""
    start_point = Point(float(start[0]), float(start[1]))
    end_point = Point(float(end[0]), float(end[1]))
    candidates: list[_BoundaryConnector] = []
    for ring_index, ring in enumerate(rings):
        if ring.distance(start_point) > endpoint_tolerance or ring.distance(end_point) > endpoint_tolerance:
            continue
        ring_length = float(ring.length)
        start_distance = float(ring.project(start_point))
        end_distance = float(ring.project(end_point))
        forward_length = (end_distance - start_distance) % ring_length
        reverse_length = ring_length - forward_length
        direction_candidates = (
            (forward_length, start_distance, end_distance, False),
            (reverse_length, end_distance, start_distance, True),
        )
        for arc_length, arc_start, arc_end, reverse in direction_candidates:
            if max_length is not None and arc_length > max_length:
                continue
            intervals = _forward_ring_intervals(arc_start, arc_end, ring_length)
            if not intervals or not _ring_intervals_are_free(intervals, consumed_intervals[ring_index]):
                continue
            arc = _ring_arc(ring, arc_start, arc_end)
            if reverse:
                arc = arc[::-1].copy()
            arc[0], arc[-1] = start, end
            candidates.append(
                _BoundaryConnector(
                    points=arc,
                    ring_index=ring_index,
                    intervals=tuple(intervals),
                    length=float(arc_length),
                )
            )
    return min(candidates, key=lambda candidate: candidate.length) if candidates else None


def _planar_zigzag_infill_items(
    region: Polygon | MultiPolygon,
    *,
    flow_spacing: float,
    density: float,
    angle_degrees: float,
    origin_xz: tuple[float, float],
    overlap: float,
    max_link_length: float,
    max_boundary_link_length: float | None,
    min_segment_length: float,
    resample_spacing: float,
    lifter: ProjectedSurfaceLifter,
) -> tuple[list[HybridPolyline], int, int]:
    """Create clipped planar zig-zag paths with only safe in-region links."""
    if not (0.0 < density <= 1.0):
        raise ValueError("planar_infill density must be in (0, 1].")
    if flow_spacing <= 0.0 or resample_spacing <= 0.0:
        raise ValueError("planar_infill spacing values must be positive.")
    inset_distance = 0.5 * flow_spacing - float(overlap)
    fill_region = (
        _covered_inset(region, inset_distance) if inset_distance >= 0.0
        else region.buffer(-inset_distance, join_style=1)
    )
    if fill_region.is_empty:
        return [], 0, 0
    origin = (float(origin_xz[0]), float(origin_xz[1]))
    scan_region = affinity.rotate(fill_region, -float(angle_degrees), origin=origin)
    min_x, min_y, max_x, max_y = scan_region.bounds
    line_spacing = flow_spacing / density
    first_x = origin[0] + np.floor((min_x - origin[0]) / line_spacing) * line_spacing
    scan_values = np.arange(first_x, max_x + 0.5 * line_spacing, line_spacing, dtype=np.float64)
    margin = max(flow_spacing, 1.0)
    paths_2d: list[np.ndarray] = []
    boundary_rings = _iter_polygon_rings(scan_region)
    consumed_boundary_intervals: list[list[tuple[float, float]]] = [[] for _ in boundary_rings]
    printable_region = scan_region.buffer(1e-8)
    for scan_index, scan_x in enumerate(scan_values):
        scan_line = LineString([(float(scan_x), min_y - margin), (float(scan_x), max_y + margin)])
        segments = [
            np.asarray(segment.coords, dtype=np.float64)
            for segment in _iter_lines(scan_region.intersection(scan_line))
            if float(segment.length) >= min_segment_length
        ]
        segments.sort(key=lambda points: float(points[0, 1]))
        if scan_index % 2:
            segments = [points[::-1].copy() for points in reversed(segments)]
        for segment in segments:
            if paths_2d:
                boundary_link = _shortest_boundary_connector(
                    paths_2d[-1][-1],
                    segment[0],
                    rings=boundary_rings,
                    consumed_intervals=consumed_boundary_intervals,
                    max_length=max_boundary_link_length,
                )
                if boundary_link is not None:
                    consumed_boundary_intervals[boundary_link.ring_index].extend(boundary_link.intervals)
                    paths_2d[-1] = np.vstack((paths_2d[-1], boundary_link.points[1:], segment[1:]))
                    continue
                link = LineString([paths_2d[-1][-1], segment[0]])
                # Clipping moves boundary endpoints by floating-point ulps.
                # The tiny tolerance permits those links without bridging holes.
                if (
                    float(link.length) <= max_link_length
                    and printable_region.covers(link)
                    and not _link_lies_on_boundary(link, boundary_rings)
                ):
                    paths_2d[-1] = np.vstack((paths_2d[-1], segment))
                    continue
            paths_2d.append(segment)

    items: list[HybridPolyline] = []
    ambiguous = 0
    snapped = 0
    for path_2d in paths_2d:
        unrotated = np.asarray(
            affinity.rotate(LineString(path_2d), float(angle_degrees), origin=origin).coords,
            dtype=np.float64,
        )
        sampled = _sample_open(unrotated, resample_spacing)
        lifted, count, snap_count = lifter.lift(sampled)
        ambiguous += count
        snapped += snap_count
        items.append(
            HybridPolyline(
                points=_subdivide_3d(lifted, resample_spacing),
                family="infill",
                level=0.0,
                closed=False,
                effective_width=flow_spacing,
            ),
        )
    return items, ambiguous, snapped


def _path_break_indices(parts: list[np.ndarray], break_before_part_indices: set[int], *, tolerance: float = 1e-8) -> list[int]:
    """Return point indices that begin a non-extruding transition in a concatenated path."""
    point_count = 0
    previous: np.ndarray | None = None
    break_indices: list[int] = []
    for part_index, part in enumerate(parts):
        points = np.asarray(part, dtype=np.float64)
        if points.shape[0] == 0:
            continue
        duplicate_start = previous is not None and np.linalg.norm(previous - points[0]) <= tolerance
        if part_index in break_before_part_indices and point_count > 0 and not duplicate_start:
            break_indices.append(point_count)
        point_count += points.shape[0] - int(duplicate_start)
        previous = points[-1]
    return break_indices


def _apply_projected_top_layer_widths(
    parallel_items: list[HybridPolyline],
    *,
    nominal_width: float,
    max_factor: float,
) -> dict[str, float]:
    """Store local 3D nearest-other-raster spacing as a per-point width."""
    if len(parallel_items) < 2:
        return {}
    from scipy.spatial import cKDTree

    width_cap = float(nominal_width) * float(max_factor)
    widths: list[np.ndarray] = []
    for item_index, item in enumerate(parallel_items):
        other_points = [
            np.asarray(other.points, dtype=np.float64)
            for other_index, other in enumerate(parallel_items)
            if other_index != item_index and np.asarray(other.points, dtype=np.float64).shape[0] > 0
        ]
        points = np.asarray(item.points, dtype=np.float64)
        if points.shape[0] == 0 or not other_points:
            continue
        distances, _ = cKDTree(np.vstack(other_points)).query(points, k=1)
        point_widths = np.asarray(distances, dtype=np.float64)
        point_widths = np.where(point_widths > 1e-8, point_widths, float(nominal_width))
        point_widths = np.minimum(point_widths, width_cap)
        item.point_widths = point_widths
        item.effective_width = float(np.mean(point_widths))
        widths.append(point_widths)
    if not widths:
        return {}
    values = np.concatenate(widths)
    return {
        "top_layer_width_check_family_count": float(len(widths)),
        "top_layer_width_check_nominal_width": float(nominal_width),
        "top_layer_width_check_max_factor": float(max_factor),
        "top_layer_width_check_applied_min": float(np.min(values)),
        "top_layer_width_check_applied_max": float(np.max(values)),
        "top_layer_width_check_applied_mean": float(np.mean(values)),
    }


@plan_connected_patches
def process_projected_hybrid_layer(
    mesh_path: Path,
    output_dir: Path,
    *,
    spacing: float,
    parallel_spacing: float | None,
    resample_spacing: float | None,
    parallel_resample_spacing: float | None,
    boundary_count: int,
    sharp_angle_deg: float,
    boundary_field_backend: str,
    parallel_field_backend: str,
    axis_mode: str,
    axis: str,
    plane_value: float | None,
    level_set_constraint: str,
    soft_level_set_weight: float,
    contour_max_level_fraction: float | None,
    min_polyline_points: int,
    mild_smooth_all: bool,
    mild_smooth_iterations: int,
    mild_smooth_alpha: float,
    start_selection_mode: str = "first",
    boundary_start_offset_count: int = 0,
    parallel_direction_mode: str = "connect_nearest",
    disable_parallel: bool = False,
    top_layer_width_check_enabled: bool = False,
    top_layer_width_check_max_factor: float = 2.0,
    is_top_layer: bool = False,
    model_lift: float = 0.0,
    model_lift_axis: str = "y",
    use_vectorized_extractor: bool = False,
    planar_infill: dict[str, object] | None = None,
    ironing: dict[str, object] | None = None,
    hatch_angle_degrees: float = 0.0,
    tolerance: float = 1e-8,
    _mesh_arrays: tuple[np.ndarray, np.ndarray] | None = None,
    _path_arrays: dict | None = None,
    _write_artifacts: bool=True,
) -> tuple[Path, dict[str, object]]:
    """Drop-in replacement for the surface-field hybrid layer processor."""
    del sharp_angle_deg, boundary_field_backend, parallel_field_backend, plane_value
    del level_set_constraint, soft_level_set_weight, contour_max_level_fraction, min_polyline_points
    del mild_smooth_all, mild_smooth_iterations, mild_smooth_alpha, use_vectorized_extractor, tolerance
    if (not np.isfinite([spacing,model_lift,hatch_angle_degrees]).all()
            or isinstance(boundary_count,bool) or not isinstance(boundary_count,int)
            or spacing <= 0.0 or boundary_count < 0):
        raise ValueError("spacing must be positive and boundary_count must be non-negative.")
    parallel_spacing = float(parallel_spacing if parallel_spacing is not None else spacing)
    path_spacing = float(resample_spacing if resample_spacing is not None else spacing)
    parallel_path_spacing = float(parallel_resample_spacing if parallel_resample_spacing is not None else path_spacing)
    attachment=(planar_infill or {}).get('wall_attachment_overlap')
    if attachment is not None:
        attachment=float(attachment)
        if not np.isfinite(attachment) or not 0<=attachment<=.5*spacing:
            raise ValueError('Model infill_wall_overlap must be between zero and half the wall bead width.')
    half_width_inset=(planar_infill or {}).get('model_half_width_inset',False)
    if not isinstance(half_width_inset,bool):
        raise ValueError('model_half_width_inset must be a boolean.')
    # Inset the material region: outer rings shrink and hole rings expand.
    # A raster-only layer has no wall to compensate or attach to.
    boundary_inset=.5*spacing if half_width_inset and boundary_count else 0.0
    if not np.isfinite([parallel_spacing,path_spacing,parallel_path_spacing]).all() or min(parallel_spacing,path_spacing,parallel_path_spacing)<=0:
        raise ValueError('Parallel and resampling spacings must be finite and positive.')
    vertices, faces = planner_mesh(mesh_path, _mesh_arrays)
    if abs(float(model_lift)) > 0.0:
        vertices = apply_axis_lift(vertices, axis=model_lift_axis, amount=float(model_lift))
    lifter = ProjectedSurfaceLifter.build(vertices, faces)
    region = _planar_region_from_projected_triangles(lifter)

    boundary_items: list[HybridPolyline] = []
    ambiguous_points = 0
    snapped_points = 0
    for level_index in range(boundary_count):
        contour_offset = boundary_inset + float(level_index) * spacing
        offset_region = _covered_inset(region, contour_offset)
        if offset_region.is_empty:
            continue
        items, ambiguous, snapped = _ring_items(offset_region, contour_offset, path_spacing, lifter)
        # Sampling interval is independent of the deposited contour bead width.
        for item in items:
            item.effective_width = float(spacing)
        boundary_items.extend(items)
        ambiguous_points += ambiguous
        snapped_points += snapped
    boundary_items = filter_boundary_items_by_start_offset(
        boundary_items,
        spacing=spacing,
        boundary_start_offset_count=boundary_start_offset_count,
        tolerance=1e-8,
    )

    parallel_items: list[HybridPolyline] = []
    infill_items: list[HybridPolyline] = []
    ironing_items: list[HybridPolyline] = []
    infill_role: str | None = None
    top_layer_width_diagnostics: dict[str, float] = {}
    main_index = int(mesh_path.stem.split("_", maxsplit=1)[0])
    parallel_angle_degrees: float | None = None
    if axis_mode == "fixed":
        direction = str(axis).strip().lower()
        if direction not in {"x", "z"}:
            raise ValueError("Projected hybrid with axis_mode='fixed' requires axis='x' or axis='z'.")
    elif axis_mode == "auto_tangent_alternating":
        direction = "x" if main_index % 2 == 0 else "z"
    elif axis_mode in {"fixed_angle", "alternating_angle"}:
        direction = "angle"
        parallel_angle_degrees = float(hatch_angle_degrees) + (90.0 if axis_mode == "alternating_angle" and main_index % 2 else 0.0)
    else:
        raise ValueError("Projected hybrid axis_mode must be 'fixed', 'fixed_angle', 'alternating_angle', or 'auto_tangent_alternating'.")
    cutoff=float(boundary_count)*spacing + boundary_inset
    if not disable_parallel:
        infill_enabled = bool(planar_infill and planar_infill.get("enabled", False))
        if attachment is not None and boundary_count:
            cutoff=(boundary_count-.5)*spacing if infill_enabled else (boundary_count-1)*spacing+.5*(spacing+parallel_spacing)-attachment
            cutoff+=boundary_inset
        interior = _covered_inset(region,cutoff)
        if not interior.is_empty:
            infill_enabled = bool(planar_infill and planar_infill.get("enabled", False))
            if infill_enabled:
                density = float(planar_infill.get("density", 0.25))
                solid = bool(planar_infill.get("solid", False))
                if solid:
                    density = float(planar_infill.get("solid_density", 1.0))
                base_angle = float(planar_infill.get("angle_degrees", 0.0))
                alternate_angle = planar_infill.get("alternate_angle_degrees")
                angle = (
                    float(alternate_angle)
                    if alternate_angle is not None and main_index % 2
                    else base_angle
                )
                origin_values = planar_infill.get("pattern_origin_xz", (0.0, 0.0))
                if not isinstance(origin_values, (list, tuple)) or len(origin_values) != 2:
                    raise ValueError("planar_infill.pattern_origin_xz must contain [x, z].")
                # A boundary route is only a local serpentine turn. Scale
                # its limit with sparse-infill pitch so density changes do
                # not silently change its connection behavior.
                boundary_link_limit_raw = planar_infill.get("max_boundary_link_length")
                boundary_link_limit = (
                    (parallel_spacing / density) * float(planar_infill.get("boundary_link_pitch_factor", 1.0))
                    + float(planar_infill.get("boundary_link_extra", 0.0))
                    if boundary_link_limit_raw is None
                    else float(boundary_link_limit_raw)
                )
                if boundary_link_limit is not None and boundary_link_limit <= 0.0:
                    raise ValueError("planar_infill.max_boundary_link_length must be positive or null.")
                infill_items, ambiguous, snapped = _planar_zigzag_infill_items(
                    interior,
                    flow_spacing=parallel_spacing,
                    density=density,
                    angle_degrees=angle,
                    origin_xz=(float(origin_values[0]), float(origin_values[1])),
                    overlap=attachment if attachment is not None and boundary_count else float(planar_infill.get("overlap", 0.0)),
                    max_link_length=float(planar_infill.get("max_link_length", 1.5 * parallel_spacing)),
                    max_boundary_link_length=boundary_link_limit,
                    min_segment_length=float(planar_infill.get("min_segment_length", 0.5 * parallel_spacing)),
                    resample_spacing=parallel_path_spacing,
                    lifter=lifter,
                )
                infill_role = "solid_infill" if solid else "sparse_infill"
                ambiguous_points += ambiguous
                snapped_points += snapped
            else:
                if parallel_angle_degrees is None:
                    parallel_items, ambiguous, snapped = _raster_items(
                        interior,
                        spacing=parallel_spacing,
                        direction=direction,
                        lifter=lifter,
                    )
                else:
                    parallel_items, ambiguous, snapped = _angled_raster_items(
                        interior,
                        spacing=parallel_spacing,
                        angle_degrees=parallel_angle_degrees,
                        lifter=lifter,
                    )
                ambiguous_points += ambiguous
                snapped_points += snapped
                parallel_items = [
                    HybridPolyline(points=_subdivide_3d(item.points, parallel_path_spacing), family=item.family, level=item.level, closed=False, effective_width=item.effective_width)
                    for item in parallel_items
                ]
                if top_layer_width_check_enabled:
                    top_layer_width_diagnostics = _apply_projected_top_layer_widths(parallel_items, nominal_width=parallel_spacing, max_factor=float(top_layer_width_check_max_factor))

    if ironing and bool(ironing.get("enabled", False)):
        ironing_spacing = float(ironing.get("spacing", parallel_spacing))
        ironing_resample_spacing = float(ironing.get("resample_spacing", parallel_path_spacing))
        ironing_inset = float(ironing.get("boundary_inset", float(boundary_count) * spacing))
        ironing_flow_width = float(ironing.get("flow_width", 0.0))
        if ironing_spacing <= 0.0 or ironing_resample_spacing <= 0.0 or ironing_inset < 0.0 or ironing_flow_width < 0.0:
            raise ValueError("ironing spacing/resample_spacing must be positive; boundary_inset and flow_width must be non-negative.")
        iron_region = _covered_inset(region, ironing_inset)
        direction_mode = str(ironing.get("direction", "perpendicular")).strip().lower()
        if direction_mode == "perpendicular":
            if parallel_angle_degrees is None:
                iron_direction = "z" if direction == "x" else "x"
                angle_degrees = None
            else:
                iron_direction = None
                angle_degrees = parallel_angle_degrees + 90.0
        elif direction_mode == "angle":
            iron_direction = None
            angle_degrees = float(ironing.get("angle_degrees", 45.0))
        else:
            iron_direction = direction_mode
            angle_degrees = None
        if iron_direction is not None and iron_direction not in {"x", "z"}:
            raise ValueError("ironing.direction must be 'perpendicular', 'x', 'z', or 'angle'.")
        if not iron_region.is_empty:
            if angle_degrees is None:
                assert iron_direction is not None
                ironing_items, ambiguous, snapped = _raster_items(
                    iron_region,
                    spacing=ironing_spacing,
                    direction=iron_direction,
                    lifter=lifter,
                )
            else:
                ironing_items, ambiguous, snapped = _angled_raster_items(
                    iron_region,
                    spacing=ironing_spacing,
                    angle_degrees=angle_degrees,
                    lifter=lifter,
                )
            ambiguous_points += ambiguous
            snapped_points += snapped
            ironing_items = [
                HybridPolyline(
                    points=_subdivide_3d(item.points, ironing_resample_spacing),
                    family="ironing",
                    level=item.level,
                    closed=False,
                    effective_width=ironing_flow_width,
                )
                for item in ironing_items
            ]

    handoff_targets = [item.points[0] for item in parallel_items] + [item.points[-1] for item in parallel_items]
    boundary_parts, boundary_widths, current = chain_boundary_family(
        boundary_items,
        start_selection_mode=start_selection_mode,
        handoff_targets=handoff_targets,
        tolerance=1e-8,
    )
    parallel_parts, parallel_widths, current = chain_parallel_family(parallel_items, current, direction_mode=parallel_direction_mode, tolerance=1e-8)
    # Keep every layer as one ordered point sequence. Disconnected printable
    # components are represented by break indices, not separate path files.
    infill_parts, infill_widths, _ = chain_parallel_family(
        infill_items,
        current,
        direction_mode=parallel_direction_mode,
        tolerance=1e-8,
    )
    ironing_parts, ironing_widths, _ = chain_parallel_family(
        ironing_items,
        None,
        direction_mode="connect_nearest",
        tolerance=1e-8,
    )
    ironing_clearance_stats = None
    clearance_cfg = ironing.get("clearance_lift") if ironing else None
    if clearance_cfg is not None and not isinstance(clearance_cfg, dict):
        raise ValueError("ironing.clearance_lift must be a JSON object when provided.")
    if ironing_parts and isinstance(clearance_cfg, dict) and bool(clearance_cfg.get("enabled", False)):
        nozzle_tip_diameter = float(clearance_cfg.get("nozzle_tip_diameter", 0.4))
        max_clearance_lift = float(clearance_cfg.get("max_lift", 0.08))
        if nozzle_tip_diameter <= 0.0 or max_clearance_lift <= 0.0:
            raise ValueError("ironing.clearance_lift nozzle_tip_diameter and max_lift must be positive when enabled.")
        ironing_parts, ironing_clearance_stats = _lift_ironing_parts_for_tip_clearance(
            ironing_parts,
            vertices,
            faces,
            nozzle_tip_diameter=nozzle_tip_diameter,
            max_lift=max_clearance_lift,
        )
    parts = [*boundary_parts, *parallel_parts, *infill_parts, *ironing_parts]
    widths = concatenate_width_profiles(parts, [*boundary_widths, *parallel_widths, *infill_widths, *ironing_widths])
    path_points = concatenate_paths(parts)
    first_infill_part = len(boundary_parts) + len(parallel_parts)
    first_ironing_part = first_infill_part + len(infill_parts)
    break_before = set(range(first_infill_part, len(parts))) if infill_parts else set(range(first_ironing_part, len(parts)))
    # A greedy connector across disconnected islands or a hole is travel.
    printable = _covered_inset(region,boundary_inset).buffer(1e-8) if boundary_inset else region.buffer(1e-8)
    for index in range(1, len(parts)):
        start, end = parts[index - 1][-1], parts[index][0]
        if np.linalg.norm(start - end) > 1e-8 and not printable.covers(LineString([start[[0, 2]], end[[0, 2]]])):
            break_before.add(index)
    path_break_indices = _path_break_indices(parts, break_before)
    part_roles=(["top_boundary_loop" if is_top_layer else "boundary_loop"] * len(boundary_parts)
        + ["top_inner_raster" if is_top_layer else "inner_raster"] * len(parallel_parts)
        + ([infill_role] * len(infill_parts) if infill_role is not None else [])
        + ["ironing"] * len(ironing_parts))
    path_tags = concatenate_path_tags(parts,part_roles)
    path_normals = compute_path_normals(vertices, faces, path_points) if path_points.size else np.empty((0, 3), dtype=np.float64)
    from early_path_reduction import reduce_generated
    path_points,path_normals,widths,path_tags,path_break_indices,reduction=reduce_generated(
        path_points,path_normals,widths,path_tags,path_break_indices,parts=parts,part_roles=part_roles,
        target_distance=min(path_spacing,parallel_path_spacing),settings=(planar_infill or {}).get('point_reduction'),region=region)
    output_path = output_dir / f"{mesh_path.stem}.txt"
    serialized_rows = save_oriented_path(str(output_path), path_points, path_normals,
                                         return_serialized=_path_arrays is not None,write_file=_write_artifacts)
    width_path = output_dir / f"{mesh_path.stem}.widths.txt"
    tag_path = output_dir / f"{mesh_path.stem}.tags.txt"
    break_path = output_dir / f"{mesh_path.stem}.breaks.txt"
    if widths.size:
        serialized_widths = save_numeric_array(width_path, widths, return_serialized=_path_arrays is not None,write_file=_write_artifacts)
    if path_tags.size and _write_artifacts:
        blocks = encode_path_tag_blocks(path_tags)
        tag_path.write_text(
            "".join(f"{start} {end} {tag}\n" for start, end, tag in blocks),
            encoding="utf-8",
        )
    if path_break_indices and _write_artifacts:
        break_path.write_text("".join(f"{index}\n" for index in path_break_indices), encoding="utf-8")
    elif _write_artifacts and break_path.exists():
        break_path.unlink()

    if _path_arrays is not None:
        _path_arrays.update(rows=serialized_rows,
                            widths=serialized_widths if widths.size else np.empty(0),
                            tags=path_tags, breaks=path_break_indices)

    metric_directions={}
    if parallel_items and not is_top_layer:
        theta=np.deg2rad(parallel_angle_degrees) if parallel_angle_degrees is not None else (0. if direction=='x' else .5*np.pi)
        metric_directions['inner_raster']=[float(np.cos(theta)),float(np.sin(theta))]
    if infill_items and infill_role:
        theta=np.deg2rad(angle)
        metric_directions[infill_role]=[-float(np.sin(theta)),float(np.cos(theta))]
    return output_path, {
        "surface_metric_mesh":str(mesh_path.resolve()),
        "surface_hatch_directions":metric_directions,
        "planner_backend": "xz_projected_2d",
        "planning_axes": "xz",
        "nozzle_axis": "y",
        "boundary_cutoff": cutoff,
        "model_half_width_inset": half_width_inset,
        "outer_contour_center_offset": boundary_inset,
        "infill_wall_overlap":attachment,
        "boundary_field_backend": "planar_polygon_offset",
        "parallel_field_backend": "planar_raster_clip",
        "axis_mode": axis_mode,
        "parallel_axis": (
            f"angle:{parallel_angle_degrees:g}"
            if parallel_items and parallel_angle_degrees is not None
            else direction if parallel_items else None
        ),
        "hatch_angle_degrees": parallel_angle_degrees,
        "dominant_normal_axis": "y",
        "tangent_axes": ["x", "z"],
        "plane_value": None,
        "parallel_spacing": parallel_spacing,
        "resample_spacing": path_spacing,
        "parallel_resample_spacing": parallel_path_spacing,
        "boundary_start_offset_count": int(boundary_start_offset_count),
        "parallel_direction_mode": parallel_direction_mode,
        "parallel_disabled": bool(disable_parallel),
        "top_layer_width_check_enabled": bool(top_layer_width_check_enabled),
        "is_top_layer": bool(is_top_layer),
        "top_layer_width_check_max_factor": float(top_layer_width_check_max_factor),
        "use_vectorized_extractor": False,
        "width_profile_file": width_path.name if widths.size else None,
        "path_tag_file": tag_path.name if path_tags.size else None,
        "path_tag_format": "half_open_index_blocks",
        "path_tag_block_count": len(encode_path_tag_blocks(path_tags)),
        "path_tag_values": sorted({str(tag) for tag in path_tags.tolist()}),
        "width_profile_point_count": int(widths.size),
        "width_profile_min": float(widths.min()) if widths.size else None,
        "width_profile_max": float(widths.max()) if widths.size else None,
        "sharp_angle_deg": None,
        "source_curve_count": 0,
        "source_segment_count": 0,
        "boundary_loop_count": len(_iter_polygons(region)),
        "boundary_level_count": int(boundary_count),
        "parallel_level_count": len(parallel_items),
        "parallel_polyline_count": len(parallel_items),
        "planar_infill_enabled": bool(planar_infill and planar_infill.get("enabled", False)),
        "planar_infill_role": infill_role,
        "planar_infill_polyline_count": len(infill_items),
        "ironing_enabled": bool(ironing and ironing.get("enabled", False)),
        "ironing_polyline_count": len(ironing_items),
        "ironing_flow_width": float(ironing.get("flow_width", 0.0)) if ironing else None,
        "ironing_direction": (f"angle:{angle_degrees:g}" if angle_degrees is not None else iron_direction) if ironing_items else None,
        "ironing_clearance_lift": ironing_clearance_stats,
        "path_break_file": break_path.name if path_break_indices else None,
        "path_break_count": len(path_break_indices),
        "path_break_semantics": "segment_before_index_is_non_printing_travel",
        "chained_boundary_part_count": len(boundary_parts),
        "chained_parallel_part_count": len(parallel_parts),
        "start_selection_mode": start_selection_mode,
        "path_point_count": int(path_points.shape[0]),
        "point_reduction":reduction,
        "projected_ambiguous_lift_point_count": int(ambiguous_points),
        "projected_snapped_lift_point_count": int(snapped_points),
        **({'projected_degenerate_artifact': lifter.degenerate_artifact} if lifter.degenerate_artifact else {}),
        **top_layer_width_diagnostics,
    }
