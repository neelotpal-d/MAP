"""
Batch-generate support-only expanded OBJ meshes by offsetting boundary vertices
slightly outward along the surface.

This is intended as a temporary workaround for support-height queries near the
outer boundary, where exact-on-boundary points can fail to detect the lower
surface robustly.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from mesh_distance_workflow import build_vertex_adjacency
from mesh_distance_workflow import find_boundary_edges
from mesh_distance_workflow import load_obj_mesh
from mesh_distance_workflow import stitch_boundary_loops
from toolpath_export import compute_vertex_normals


def _normalize(vector: np.ndarray, *, tolerance: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= tolerance:
        return np.zeros_like(vector, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def _project_to_tangent_plane(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return np.asarray(vector, dtype=np.float64) - np.dot(vector, normal) * normal


def _is_closed_loop(loop: np.ndarray) -> bool:
    return loop.shape[0] >= 2 and int(loop[0]) == int(loop[-1])


def _loop_core_vertices(loop: np.ndarray) -> np.ndarray:
    if _is_closed_loop(loop):
        return np.asarray(loop[:-1], dtype=np.int64)
    return np.asarray(loop, dtype=np.int64)


def _boundary_tangent(loop_vertices: np.ndarray, idx: int, V: np.ndarray) -> np.ndarray:
    count = loop_vertices.shape[0]
    closed = count >= 3
    current = V[loop_vertices[idx]]

    if closed:
        prev_vertex = V[loop_vertices[(idx - 1) % count]]
        next_vertex = V[loop_vertices[(idx + 1) % count]]
        return _normalize(next_vertex - prev_vertex)

    if count == 1:
        return np.zeros(3, dtype=np.float64)
    if idx == 0:
        return _normalize(V[loop_vertices[1]] - current)
    if idx == count - 1:
        return _normalize(current - V[loop_vertices[count - 2]])
    prev_vertex = V[loop_vertices[idx - 1]]
    next_vertex = V[loop_vertices[idx + 1]]
    return _normalize(next_vertex - prev_vertex)


def _estimate_outward_direction(
    vertex_idx: int,
    tangent: np.ndarray,
    normal: np.ndarray,
    V: np.ndarray,
    adjacency: dict[int, set[int]],
    boundary_set: set[int],
    mesh_centroid: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> np.ndarray:
    candidate = _normalize(np.cross(tangent, normal), tolerance=tolerance)
    if float(np.linalg.norm(candidate)) <= tolerance:
        return np.zeros(3, dtype=np.float64)

    point = V[vertex_idx]
    interior_neighbors = [nbr for nbr in adjacency.get(int(vertex_idx), set()) if int(nbr) not in boundary_set]

    if interior_neighbors:
        interior_target = V[np.asarray(interior_neighbors, dtype=np.int64)].mean(axis=0) - point
    else:
        interior_target = mesh_centroid - point

    interior_target = _project_to_tangent_plane(interior_target, normal)
    if float(np.linalg.norm(interior_target)) <= tolerance:
        return candidate

    return -candidate if float(np.dot(candidate, interior_target)) > 0.0 else candidate


def _branching_boundary_tangent(
    vertex_idx: int,
    V: np.ndarray,
    boundary_adjacency: dict[int, list[int]],
    *,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Estimate a stable tangent when a non-manifold boundary cannot be walked."""
    neighbors = boundary_adjacency.get(vertex_idx, [])
    if not neighbors:
        return np.zeros(3, dtype=np.float64)
    directions = V[np.asarray(neighbors, dtype=np.int64)] - V[vertex_idx]
    directions = directions[np.linalg.norm(directions, axis=1) > tolerance]
    if directions.size == 0:
        return np.zeros(3, dtype=np.float64)
    # The principal local boundary direction remains usable at a branch; the
    # later interior-neighbor test chooses its outward sign.
    _, _, right_vectors = np.linalg.svd(directions, full_matrices=False)
    return _normalize(right_vectors[0], tolerance=tolerance)


def _expand_branching_boundary_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    boundary_edges: np.ndarray,
    *,
    offset_distance: float,
    tolerance: float,
) -> np.ndarray:
    """Fallback for non-manifold boundary graphs that cannot form simple loops."""
    normals = compute_vertex_normals(vertices, faces, tolerance=tolerance)
    adjacency = build_vertex_adjacency(faces)
    boundary_set = set(np.unique(boundary_edges.reshape(-1)).tolist())
    boundary_adjacency: dict[int, list[int]] = {}
    for first, second in boundary_edges:
        boundary_adjacency.setdefault(int(first), []).append(int(second))
        boundary_adjacency.setdefault(int(second), []).append(int(first))
    mesh_centroid = vertices.mean(axis=0)
    expanded = vertices.copy()
    for vertex_idx in boundary_set:
        tangent = _branching_boundary_tangent(
            int(vertex_idx),
            vertices,
            boundary_adjacency,
            tolerance=tolerance,
        )
        outward = _estimate_outward_direction(
            int(vertex_idx),
            tangent,
            normals[int(vertex_idx)],
            vertices,
            adjacency,
            boundary_set,
            mesh_centroid,
            tolerance=tolerance,
        )
        if float(np.linalg.norm(outward)) > tolerance:
            expanded[int(vertex_idx)] += offset_distance * outward
    return expanded


def expand_boundary_vertices(
    V: np.ndarray,
    F: np.ndarray,
    *,
    offset_distance: float,
    tolerance: float = 1e-12,
) -> np.ndarray:
    if offset_distance < 0.0:
        raise ValueError("offset_distance must be non-negative.")
    if offset_distance == 0.0:
        return np.asarray(V, dtype=np.float64).copy()

    vertices = np.asarray(V, dtype=np.float64)
    faces = np.asarray(F, dtype=np.int64)
    boundary_edges = find_boundary_edges(faces)
    try:
        boundary_loops = stitch_boundary_loops(boundary_edges)
    except ValueError as exc:
        print(
            "Warning: branching boundary detected; using local tangent support-expansion fallback "
            f"({exc})",
        )
        return _expand_branching_boundary_vertices(
            vertices,
            faces,
            boundary_edges,
            offset_distance=offset_distance,
            tolerance=tolerance,
        )
    if not boundary_loops:
        return vertices.copy()

    normals = compute_vertex_normals(vertices, faces, tolerance=tolerance)
    adjacency = build_vertex_adjacency(faces)
    boundary_set = set(np.unique(boundary_edges.reshape(-1)).tolist())
    mesh_centroid = vertices.mean(axis=0)

    direction_sum = np.zeros_like(vertices)
    direction_count = np.zeros((vertices.shape[0],), dtype=np.int32)

    for loop in boundary_loops:
        loop_vertices = _loop_core_vertices(loop)
        if loop_vertices.shape[0] == 0:
            continue
        for idx, vertex_idx in enumerate(loop_vertices):
            tangent = _boundary_tangent(loop_vertices, idx, vertices)
            normal = normals[int(vertex_idx)]
            outward = _estimate_outward_direction(
                int(vertex_idx),
                tangent,
                normal,
                vertices,
                adjacency,
                boundary_set,
                mesh_centroid,
                tolerance=tolerance,
            )
            if float(np.linalg.norm(outward)) <= tolerance:
                continue
            direction_sum[int(vertex_idx)] += outward
            direction_count[int(vertex_idx)] += 1

    expanded = vertices.copy()
    for vertex_idx in boundary_set:
        if direction_count[int(vertex_idx)] <= 0:
            continue
        outward = _normalize(direction_sum[int(vertex_idx)], tolerance=tolerance)
        if float(np.linalg.norm(outward)) <= tolerance:
            continue
        expanded[int(vertex_idx)] += offset_distance * outward

    return expanded


def save_obj_mesh(path: Path, V: np.ndarray, F: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for vertex in np.asarray(V, dtype=np.float64):
            handle.write(f"v {vertex[0]:.10f} {vertex[1]:.10f} {vertex[2]:.10f}\n")
        for face in np.asarray(F, dtype=np.int64):
            handle.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Expand mesh boundaries outward and save the modified OBJ meshes to another folder.",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Folder containing OBJ meshes to expand.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Folder where expanded OBJ meshes will be written.",
    )
    parser.add_argument(
        "--offset",
        type=float,
        required=True,
        help="Boundary expansion distance in mesh units.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional suffix appended to each output stem before .obj. Default keeps the original stem.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing files in the output directory before writing.",
    )
    return parser


def clear_directory(directory: Path) -> None:
    if not directory.exists():
        return
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def iter_obj_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".obj")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    obj_files = iter_obj_files(args.input_dir)
    if not obj_files:
        raise FileNotFoundError(f"No OBJ files found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        clear_directory(args.output_dir)
        print(f"Cleared output directory: {args.output_dir}")

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"OBJ count: {len(obj_files)}")
    print(f"Boundary offset: {args.offset}")

    for obj_path in obj_files:
        V, F = load_obj_mesh(str(obj_path))
        expanded_V = expand_boundary_vertices(V, F, offset_distance=args.offset)
        output_path = args.output_dir / f"{obj_path.stem}{args.suffix}.obj"
        save_obj_mesh(output_path, expanded_V, F)
        print(f"Expanded: {obj_path.name} -> {output_path.name}")


if __name__ == "__main__":
    main()
