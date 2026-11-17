"""Visualize uniform surface samples over selected triangle-mesh regions.

Original mesh faces remain the source of topology. The generated samples are
only a test of physical-space coverage before adding this scheme to training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtimeEnv import configure_openmp_runtime

configure_openmp_runtime()

import numpy as np
import pyvista as pv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-path", type=Path, required=True)
    parser.add_argument(
        "--constraint-regions-json",
        type=Path,
        default=None,
        help="Optional exported surface_constraint_regions.json; uses faces whose three vertices share a group.",
    )
    parser.add_argument("--up-direction", nargs=3, type=float, default=(0.0, 1.0, 0.0), metavar=("UX", "UY", "UZ"))
    parser.add_argument("--surface-normal-max-angle", type=float, default=40.0)
    parser.add_argument("--surface-height-min", type=float, default=None)
    parser.add_argument("--surface-height-max", type=float, default=None)
    parser.add_argument("--edge-spacing", type=float, default=2.0, help="Maximum world-space distance between edge samples.")
    parser.add_argument("--interior-sample-area", type=float, default=4.0, help="Target world-space area represented by one interior sample.")
    parser.add_argument(
        "--samples-per-square-unit",
        type=float,
        default=None,
        help="Physical sample density. Overrides --interior-sample-area when supplied.",
    )
    parser.add_argument(
        "--min-samples-per-region",
        type=int,
        default=0,
        help="Minimum samples allocated to each selected constraint region; zero keeps strict area density.",
    )
    parser.add_argument(
        "--max-samples-per-region",
        type=int,
        default=50000,
        help="Maximum samples allocated to each selected constraint region.",
    )
    parser.add_argument("--thin-triangle-aspect-ratio", type=float, default=3.0, help="Use longest-edge strips when longest-edge/altitude exceeds this value.")
    parser.add_argument(
        "--sampling-mode",
        choices=("poisson", "hybrid"),
        default="poisson",
        help="'poisson' uses global area candidates and spacing rejection; 'hybrid' retains the old per-face test.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Number of final Poisson samples. Defaults to selected area / --interior-sample-area.",
    )
    parser.add_argument(
        "--candidate-oversampling",
        type=int,
        default=12,
        help="Area-uniform candidates generated per requested Poisson sample.",
    )
    parser.add_argument(
        "--poisson-radius",
        type=float,
        default=None,
        help="Minimum 3D separation. Defaults to 85%% of the triangular-lattice spacing for the requested density.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducible candidates and rejection order.")
    parser.add_argument("--show-candidates", action="store_true", help="Overlay rejected area-uniform candidates in pale gray.")
    parser.add_argument("--show-source-vertices", action="store_true", help="Overlay original selected mesh vertices in green.")
    parser.add_argument("--show-edge-samples", action="store_true", help="Overlay old per-edge samples in blue when using --sampling-mode hybrid.")
    parser.add_argument("--show-sample-normals", action="store_true", help="Draw normals assigned to final Poisson samples.")
    parser.add_argument("--normal-arrow-scale", type=float, default=None, help="World-space normal-arrow length; defaults to half the Poisson radius.")
    parser.add_argument("--mesh-opacity", type=float, default=0.12)
    parser.add_argument("--point-size", type=float, default=6.0)
    return parser.parse_args()


def triangulated_mesh(mesh_path: Path) -> pv.PolyData:
    mesh = pv.read(mesh_path).triangulate()
    if mesh.n_cells == 0 or mesh.n_points == 0:
        raise ValueError(f"{mesh_path} has no triangular surface cells.")
    return mesh


def faces_of(mesh: pv.PolyData) -> np.ndarray:
    faces = np.asarray(mesh.faces)
    if faces.size % 4 != 0 or np.any(faces.reshape((-1, 4))[:, 0] != 3):
        raise ValueError("Hybrid sampling requires a triangular surface mesh.")
    return faces.reshape((-1, 4))[:, 1:4]


def qualified_face_mask(mesh: pv.PolyData, args: argparse.Namespace) -> np.ndarray:
    mesh.compute_normals(point_normals=False, cell_normals=True, inplace=True)
    faces = faces_of(mesh)
    points = np.asarray(mesh.points, dtype=np.float64)
    normals = np.asarray(mesh.cell_data["Normals"], dtype=np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    up = np.asarray(args.up_direction, dtype=np.float64)
    up /= np.linalg.norm(up)
    selected = normals @ up > np.cos(np.deg2rad(args.surface_normal_max_angle))
    if args.surface_height_min is not None or args.surface_height_max is not None:
        relative_height = np.mean(points[faces], axis=1) @ up - np.min(points @ up)
        if args.surface_height_min is not None:
            selected &= relative_height >= args.surface_height_min
        if args.surface_height_max is not None:
            selected &= relative_height <= args.surface_height_max
    return selected


def selected_faces_from_metadata(mesh: pv.PolyData, metadata_path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    vertex_groups = np.full(mesh.n_points, -1, dtype=np.int64)
    for region in payload.get("regions", []):
        group_id = int(region["group_id"])
        indices = np.asarray(region["vertex_indices"], dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= mesh.n_points):
            raise ValueError(f"{metadata_path} contains vertex indices incompatible with {mesh.n_points} mesh points.")
        vertex_groups[indices] = group_id
    faces = faces_of(mesh)
    face_groups = vertex_groups[faces]
    selected = np.all(face_groups == face_groups[:, :1], axis=1) & (face_groups[:, 0] >= 0)
    return selected, np.where(selected, face_groups[:, 0], -1)


def selected_vertex_normals(mesh: pv.PolyData, selected_faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average only selected incident face normals at each selected vertex."""

    faces = faces_of(mesh)
    face_normals = np.asarray(mesh.cell_data["Normals"], dtype=np.float64)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-12
    sums = np.zeros((mesh.n_points, 3), dtype=np.float64)
    for face, normal in zip(faces[selected_faces], face_normals[selected_faces]):
        sums[face] += normal
    mask = np.linalg.norm(sums, axis=1) > 1e-12
    normals = sums[mask]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return np.asarray(mesh.points)[mask], normals


def hybrid_samples(
    mesh: pv.PolyData,
    selected_faces: np.ndarray,
    edge_spacing: float,
    interior_sample_area: float,
    thin_triangle_aspect_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return edge and aspect-aware interior samples using physical spacing rules."""

    if edge_spacing <= 0.0 or interior_sample_area <= 0.0 or thin_triangle_aspect_ratio <= 1.0:
        raise ValueError("Sampling spacing/area must be positive and aspect ratio must exceed one.")
    faces = faces_of(mesh)
    points = np.asarray(mesh.points, dtype=np.float64)
    face_normals = np.asarray(mesh.cell_data["Normals"], dtype=np.float64)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-12
    selected_indices = np.flatnonzero(selected_faces)
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_index in selected_indices:
        first, second, third = faces[face_index]
        for edge in ((first, second), (second, third), (third, first)):
            edge_to_faces.setdefault(tuple(sorted((int(edge[0]), int(edge[1])))), []).append(int(face_index))

    edge_samples = []
    for (first, second), incident_faces in edge_to_faces.items():
        first_point, second_point = points[first], points[second]
        segment_count = max(1, int(np.ceil(np.linalg.norm(second_point - first_point) / edge_spacing)))
        for fraction in np.arange(1, segment_count, dtype=np.float64) / segment_count:
            edge_samples.append((1.0 - fraction) * first_point + fraction * second_point)

    interior_samples = []
    # An equilateral triangular lattice with spacing h represents roughly
    # sqrt(3)/2 * h^2 area per point.
    target_spacing = np.sqrt(2.0 * interior_sample_area / np.sqrt(3.0))
    for face_index in selected_indices:
        first_point, second_point, third_point = points[faces[face_index]]
        triangle = np.stack((first_point, second_point, third_point))
        edge_pairs = ((0, 1), (1, 2), (2, 0))
        edge_lengths = np.asarray([np.linalg.norm(triangle[end] - triangle[start]) for start, end in edge_pairs])
        longest_edge_index = int(np.argmax(edge_lengths))
        base_start_index, base_end_index = edge_pairs[longest_edge_index]
        apex_index = ({0, 1, 2} - {base_start_index, base_end_index}).pop()
        base_start, base_end, apex = triangle[base_start_index], triangle[base_end_index], triangle[apex_index]
        base_length = edge_lengths[longest_edge_index]
        area = 0.5 * np.linalg.norm(np.cross(base_end - base_start, apex - base_start))
        altitude = 2.0 * area / max(base_length, 1e-12)

        if base_length / max(altitude, 1e-12) <= thin_triangle_aspect_ratio:
            # A regular barycentric lattice gives nearly uniform area coverage
            # for faces whose two in-plane dimensions are comparable.
            target_count = max(1, int(np.ceil(area / interior_sample_area)))
            # n(n + 1)/2 lattice points for n barycentric subdivisions.
            subdivisions = max(1, int(np.ceil((np.sqrt(1.0 + 8.0 * target_count) - 1.0) / 2.0)))
            barycentric_points = []
            for first_index in range(subdivisions):
                for second_index in range(subdivisions - first_index):
                    first_weight = (first_index + 1.0 / 3.0) / subdivisions
                    second_weight = (second_index + 1.0 / 3.0) / subdivisions
                    third_weight = 1.0 - first_weight - second_weight
                    barycentric_points.append(
                        first_weight * first_point + second_weight * second_point + third_weight * third_point
                    )
            interior_samples.append(np.asarray(barycentric_points, dtype=np.float64))
        else:
            # For a long thin face, span it with strips parallel to its longest
            # edge. Each strip is clipped by the triangle's two side edges.
            strip_count = max(1, int(np.ceil(altitude / target_spacing)))
            strip_samples = []
            for strip_index in range(strip_count):
                height_fraction = (strip_index + 0.5) / strip_count
                strip_start = (1.0 - height_fraction) * base_start + height_fraction * apex
                strip_end = (1.0 - height_fraction) * base_end + height_fraction * apex
                strip_length = np.linalg.norm(strip_end - strip_start)
                along_count = max(1, int(np.ceil(strip_length / target_spacing)))
                fractions = (np.arange(along_count, dtype=np.float64) + 0.5) / along_count
                strip_samples.append(
                    (1.0 - fractions[:, None]) * strip_start + fractions[:, None] * strip_end
                )
            interior_samples.append(np.concatenate(strip_samples, axis=0))
    return (
        np.asarray(edge_samples, dtype=np.float64).reshape((-1, 3)),
        np.concatenate(interior_samples, axis=0) if interior_samples else np.empty((0, 3), dtype=np.float64),
    )


def selected_face_areas(mesh: pv.PolyData, selected_faces: np.ndarray) -> np.ndarray:
    """Return physical triangle areas for the selected cells."""

    faces = faces_of(mesh)[selected_faces]
    triangles = np.asarray(mesh.points, dtype=np.float64)[faces]
    return 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )


def area_uniform_candidates(
    mesh: pv.PolyData,
    selected_faces: np.ndarray,
    candidate_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample candidates uniformly in total selected surface area, not per face."""

    if candidate_count <= 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
        )
    selected_indices = np.flatnonzero(selected_faces)
    areas = selected_face_areas(mesh, selected_faces)
    valid = areas > 1e-14
    if not np.any(valid):
        raise ValueError("Selected faces have zero total area.")
    selected_indices, areas = selected_indices[valid], areas[valid]
    face_indices = rng.choice(selected_indices, size=candidate_count, p=areas / areas.sum())
    triangles = np.asarray(mesh.points, dtype=np.float64)[faces_of(mesh)[face_indices]]

    # Square-root barycentrics preserve uniform probability per unit triangle area.
    first_random = np.sqrt(rng.random(candidate_count))
    second_random = rng.random(candidate_count)
    weights = np.column_stack(
        (1.0 - first_random, first_random * (1.0 - second_random), first_random * second_random)
    )
    return np.einsum("ni,nij->nj", weights, triangles), face_indices, weights


def poisson_reject_candidates(candidates: np.ndarray, radius: float) -> np.ndarray:
    """Greedy dart throwing with a spatial hash, using global 3D distances."""

    if radius <= 0.0:
        raise ValueError("--poisson-radius must be positive.")
    radius_squared = radius * radius
    inverse_cell_size = 1.0 / radius
    accepted_indices: list[int] = []
    cells: dict[tuple[int, int, int], list[int]] = {}
    for candidate_index, candidate in enumerate(candidates):
        cell = tuple(np.floor(candidate * inverse_cell_size).astype(np.int64))
        is_far_enough = True
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                for offset_z in (-1, 0, 1):
                    for accepted_index in cells.get(
                        (cell[0] + offset_x, cell[1] + offset_y, cell[2] + offset_z), ()
                    ):
                        delta = candidate - candidates[accepted_indices[accepted_index]]
                        if float(delta @ delta) < radius_squared:
                            is_far_enough = False
                            break
                    if not is_far_enough:
                        break
                if not is_far_enough:
                    break
            if not is_far_enough:
                break
        if is_far_enough:
            cells.setdefault(cell, []).append(len(accepted_indices))
            accepted_indices.append(candidate_index)
    return np.asarray(accepted_indices, dtype=np.int64)


def poisson_surface_samples(
    mesh: pv.PolyData,
    selected_faces: np.ndarray,
    sample_area: float,
    sample_count: int | None,
    oversampling: int,
    radius: float | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Create area-uniform candidates then retain globally separated samples."""

    if sample_area <= 0.0 or oversampling < 1:
        raise ValueError("--interior-sample-area and --candidate-oversampling must be positive.")
    total_area = float(selected_face_areas(mesh, selected_faces).sum())
    requested_count = sample_count or max(1, int(round(total_area / sample_area)))
    lattice_spacing = np.sqrt(2.0 * total_area / (np.sqrt(3.0) * requested_count))
    # Greedy dart throwing does not achieve the ideal triangular packing density.
    # Start conservatively, then relax only the automatic radius if the requested
    # count cannot fit in the candidate pool.
    effective_radius = radius if radius is not None else 0.70 * lattice_spacing
    rng = np.random.default_rng(seed)
    candidates, candidate_faces, candidate_weights = area_uniform_candidates(
        mesh, selected_faces, requested_count * oversampling, rng
    )
    accepted_indices = poisson_reject_candidates(candidates, effective_radius)
    if radius is None:
        for _ in range(4):
            if len(accepted_indices) >= requested_count:
                break
            effective_radius *= 0.90
            accepted_indices = poisson_reject_candidates(candidates, effective_radius)
    # Candidate order is random, so keeping a prefix maintains the Poisson separation property.
    return candidates, candidate_faces, candidate_weights, accepted_indices[:requested_count], effective_radius, requested_count


def sample_normals_from_faces(
    mesh: pv.PolyData,
    selected_faces: np.ndarray,
    parent_faces: np.ndarray,
    barycentric_weights: np.ndarray,
    boundary_tolerance: float = 1e-10,
) -> np.ndarray:
    """Use parent-face normals, averaging only selected incident faces on boundaries."""

    faces = faces_of(mesh)
    face_normals = np.asarray(mesh.cell_data["Normals"], dtype=np.float64)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-12
    selected_indices = np.flatnonzero(selected_faces)
    vertex_to_faces: list[list[int]] = [[] for _ in range(mesh.n_points)]
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_index in selected_indices:
        first, second, third = faces[face_index]
        for vertex in (first, second, third):
            vertex_to_faces[int(vertex)].append(int(face_index))
        for first_vertex, second_vertex in ((first, second), (second, third), (third, first)):
            edge_to_faces.setdefault(tuple(sorted((int(first_vertex), int(second_vertex)))), []).append(int(face_index))

    normals = face_normals[parent_faces].copy()
    for sample_index, (face_index, weights) in enumerate(zip(parent_faces, barycentric_weights)):
        near_zero = np.flatnonzero(weights <= boundary_tolerance)
        if len(near_zero) == 0:
            continue
        vertices = faces[face_index]
        if len(near_zero) == 1:
            edge_vertices = np.delete(vertices, near_zero[0])
            incident = edge_to_faces[tuple(sorted((int(edge_vertices[0]), int(edge_vertices[1]))))]
        else:
            vertex = int(vertices[np.argmax(weights)])
            incident = vertex_to_faces[vertex]
        averaged = face_normals[incident].sum(axis=0)
        normals[sample_index] = averaged / (np.linalg.norm(averaged) + 1e-12)
    return normals


def allocate_region_sample_counts(
    mesh: pv.PolyData,
    selected_faces: np.ndarray,
    face_group_ids: np.ndarray,
    args: argparse.Namespace,
) -> list[tuple[int, np.ndarray, int]]:
    """Allocate each region a physical-area sample budget with optional caps."""

    if args.min_samples_per_region < 0 or args.max_samples_per_region < 1:
        raise ValueError("Per-region sample limits must be non-negative and positive, respectively.")
    if args.min_samples_per_region > args.max_samples_per_region:
        raise ValueError("--min-samples-per-region cannot exceed --max-samples-per-region.")
    sample_area = 1.0 / args.samples_per_square_unit if args.samples_per_square_unit is not None else args.interior_sample_area
    if sample_area <= 0.0:
        raise ValueError("--samples-per-square-unit and --interior-sample-area must be positive.")
    region_ids = np.unique(face_group_ids[selected_faces])
    region_masks = [(region_id, selected_faces & (face_group_ids == region_id)) for region_id in region_ids]
    region_areas = np.asarray([selected_face_areas(mesh, mask).sum() for _, mask in region_masks], dtype=np.float64)
    if args.sample_count is None:
        counts = np.ceil(region_areas / sample_area).astype(np.int64)
    else:
        if args.sample_count < 1:
            raise ValueError("--sample-count must be positive.")
        fractional_counts = region_areas / region_areas.sum() * args.sample_count
        counts = np.floor(fractional_counts).astype(np.int64)
        remaining = args.sample_count - int(counts.sum())
        if remaining:
            counts[np.argsort(fractional_counts - counts)[-remaining:]] += 1
    counts = np.clip(counts, args.min_samples_per_region, args.max_samples_per_region)
    return [(int(region_id), mask, int(count)) for (region_id, mask), count in zip(region_masks, counts)]


def main() -> None:
    args = parse_args()
    mesh_path = args.mesh_path.resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    mesh = triangulated_mesh(mesh_path)
    mesh.compute_normals(point_normals=False, cell_normals=True, inplace=True)
    if args.constraint_regions_json is None:
        selected_faces = qualified_face_mask(mesh, args)
        face_group_ids = np.where(selected_faces, 0, -1)
        source_label = "face-normal qualification"
    else:
        selected_faces, face_group_ids = selected_faces_from_metadata(mesh, args.constraint_regions_json.resolve())
        source_label = "exported constraint metadata"
    if not np.any(selected_faces):
        raise ValueError("No faces were selected. Relax the selection threshold or inspect the metadata.")

    vertex_points, _ = selected_vertex_normals(mesh, selected_faces)
    edge_points = np.empty((0, 3), dtype=np.float64)
    candidate_points = np.empty((0, 3), dtype=np.float64)
    if args.sampling_mode == "poisson":
        sample_area = 1.0 / args.samples_per_square_unit if args.samples_per_square_unit is not None else args.interior_sample_area
        region_results = []
        for region_id, region_mask, region_count in allocate_region_sample_counts(
            mesh, selected_faces, face_group_ids, args
        ):
            region_results.append(
                poisson_surface_samples(
                    mesh,
                    region_mask,
                    sample_area,
                    region_count,
                    args.candidate_oversampling,
                    args.poisson_radius,
                    args.seed + region_id,
                )
            )
        candidate_points = np.concatenate([result[0] for result in region_results], axis=0)
        accepted_points = [result[0][result[3]] for result in region_results]
        accepted_faces = [result[1][result[3]] for result in region_results]
        accepted_weights = [result[2][result[3]] for result in region_results]
        interior_points = np.concatenate(accepted_points, axis=0)
        sample_normals = sample_normals_from_faces(
            mesh,
            selected_faces,
            np.concatenate(accepted_faces, axis=0),
            np.concatenate(accepted_weights, axis=0),
        )
        radii = [result[4] for result in region_results]
        requested_count = sum(result[5] for result in region_results)
        source_label = (
            f"{source_label}, global Poisson rejection "
            f"(regions={len(region_results)}, radius={min(radii):.4g}-{max(radii):.4g}, requested={requested_count})"
        )
    else:
        edge_points, interior_points = hybrid_samples(
            mesh, selected_faces, args.edge_spacing, args.interior_sample_area, args.thin_triangle_aspect_ratio
        )
        source_label = f"{source_label}, legacy face-local hybrid sampling"
        sample_normals = np.empty((0, 3), dtype=np.float64)
    selected_mesh = mesh.extract_cells(np.flatnonzero(selected_faces))
    plotter = pv.Plotter(title="Hybrid surface-sampling test")
    plotter.add_axes()
    plotter.enable_anti_aliasing()
    plotter.add_mesh(mesh, color="lightgray", opacity=args.mesh_opacity, show_edges=True, edge_color="gray")
    plotter.add_mesh(selected_mesh, color="tomato", opacity=0.35, show_edges=True, edge_color="darkred")
    if args.show_source_vertices:
        plotter.add_mesh(pv.PolyData(vertex_points), color="limegreen", point_size=args.point_size, render_points_as_spheres=True, label="Selected vertices")
    if args.show_candidates and candidate_points.size:
        plotter.add_mesh(pv.PolyData(candidate_points), color="lightgray", opacity=0.25, point_size=3.0, render_points_as_spheres=True, label="Area-uniform candidates")
    if args.show_edge_samples and edge_points.size:
        plotter.add_mesh(pv.PolyData(edge_points), color="deepskyblue", point_size=args.point_size, render_points_as_spheres=True, label="Edge samples")
    if interior_points.size:
        plotter.add_mesh(pv.PolyData(interior_points), color="gold", point_size=args.point_size, render_points_as_spheres=True, label="Final surface samples")
    if args.show_sample_normals and sample_normals.size:
        arrow_length = args.normal_arrow_scale if args.normal_arrow_scale is not None else 0.5 * min(radii)
        if arrow_length <= 0.0:
            raise ValueError("--normal-arrow-scale must be positive.")
        plotter.add_arrows(interior_points, sample_normals, mag=arrow_length, color="midnightblue")
    plotter.add_legend(bcolor="white")
    plotter.add_text(
        f"{source_label}\nfaces={int(selected_faces.sum())} vertices={len(vertex_points)} "
        f"final samples={len(interior_points)} candidates={len(candidate_points)}",
        position="upper_left",
        font_size=10,
    )
    plotter.show()


if __name__ == "__main__":
    main()
