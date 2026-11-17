"""
Detect top-like layers by checking whether points are covered by future layers.

This standalone script classifies a layer as top-like when a sufficient
fraction of sampled points are not covered by any of the next N main-index
layers when projected along a fixed print axis.
"""

from __future__ import annotations

from multiprocessing import get_context

import argparse
import colorsys
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from batch_compute_heights import collect_layer_files
from extrusion_height import LayerKey
from extrusion_height import apply_axis_lift
from extrusion_height import axis_to_index
from extrusion_height import ReferenceMesh3Axis
from extrusion_height import load_reference_mesh_3axis
from extrusion_height import projected_dimensions
from mesh_distance_workflow import find_boundary_vertices
from mesh_distance_workflow import load_obj_mesh
from mesh_distance_workflow import to_pyvista_polydata

try:
    import igl  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    igl = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect top-like layers from a folder of layered OBJ meshes.",
    )
    parser.add_argument(
        "mesh_dir",
        type=Path,
        help="Folder containing layer meshes named like <main>_<sub>.obj.",
    )
    parser.add_argument(
        "--axis",
        type=str,
        default="y",
        help="Fixed print axis used for coverage testing. Default: y.",
    )
    parser.add_argument(
        "--lookahead-main-count",
        type=int,
        default=3,
        help="How many future main indices to check for coverage. Default: 3.",
    )
    parser.add_argument(
        "--min-uncovered-fraction",
        type=float,
        default=0.30,
        help="Classify as top-like when uncovered fraction is at least this value. Default: 0.30.",
    )
    parser.add_argument(
        "--k-nearest-vertices",
        type=int,
        default=12,
        help="Number of nearby projected vertices used to gather candidate upper triangles.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("vertices", "face_centroids"),
        default="vertices",
        help="Which mesh samples to test for coverage.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON file where the per-layer classification report will be saved.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Geometric tolerance used for triangle containment checks.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a PyVista preview where detected top-like layers are highlighted.",
    )
    parser.add_argument(
        "--top-color",
        type=str,
        default="tomato",
        help="Preview color for detected top-like layers.",
    )
    parser.add_argument(
        "--other-color",
        type=str,
        default="lightgray",
        help="Preview color for non-top-like layers.",
    )
    parser.add_argument(
        "--top-opacity",
        type=float,
        default=0.9,
        help="Preview opacity for detected top-like layers.",
    )
    parser.add_argument(
        "--other-opacity",
        type=float,
        default=0.18,
        help="Preview opacity for non-top-like layers.",
    )
    parser.add_argument(
        "--layer-workers",
        type=int,
        default=1,
        help="Number of worker processes used to classify layers in parallel. Default: 1.",
    )
    parser.add_argument(
        "--tree-query-workers",
        type=int,
        default=1,
        help="Number of worker threads used internally by SciPy cKDTree.query. Use -1 for all CPU threads.",
    )
    parser.add_argument(
        "--boundary-precheck-enabled",
        action="store_true",
        help="Run a cheap boundary-proximity precheck and skip expensive lookahead coverage for non-candidate layers.",
    )
    parser.add_argument(
        "--boundary-precheck-distance",
        type=float,
        default=1.5,
        help="Projected distance band used by the boundary precheck. Default: 1.5.",
    )
    parser.add_argument(
        "--boundary-precheck-min-fraction",
        type=float,
        default=0.30,
        help="Minimum fraction of sampled points near the boundary band required to run the expensive lookahead test. Default: 0.30.",
    )
    parser.add_argument(
        "--global-boundary-mesh",
        type=Path,
        default=None,
        help="Optional OBJ mesh representing the overall model boundary used for coarse candidate filtering.",
    )
    parser.add_argument(
        "--global-boundary-distance",
        type=float,
        default=1.5,
        help="3D distance band used by the global-boundary coarse precheck. Default: 1.5.",
    )
    parser.add_argument(
        "--global-boundary-min-fraction",
        type=float,
        default=0.30,
        help="Minimum fraction of sampled points near the global boundary required to run the expensive lookahead test. Default: 0.30.",
    )
    parser.add_argument(
        "--normalize-global-boundary-base-to-zero",
        "--normalize-base-to-zero",
        dest="normalize_global_boundary_base_to_zero",
        action="store_true",
        help="Translate only the global-boundary mesh to its minimum print-axis value before classification.",
    )
    return parser


def future_main_indices(keys: list[LayerKey], current_main: int, count: int) -> list[int]:
    unique_mains = sorted({key.main for key in keys if key.main > current_main})
    if count <= 0:
        return []
    return unique_mains[:count]


def sample_points_for_mesh(reference: ReferenceMesh3Axis, *, mode: str) -> np.ndarray:
    vertices = np.asarray(reference.vertices, dtype=np.float64)
    if mode == "vertices":
        return vertices
    if mode == "face_centroids":
        faces = np.asarray(reference.faces, dtype=np.int64)
        if faces.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        return np.mean(vertices[faces], axis=1)
    raise ValueError(f"Unsupported sample mode: {mode}")


def boundary_proximity_fraction(
    points: np.ndarray,
    reference_mesh: ReferenceMesh3Axis,
    *,
    max_distance: float,
    tolerance: float,
) -> tuple[float, int]:
    if points.shape[0] == 0:
        return 0.0, 0
    boundary_vertex_indices = np.asarray(find_boundary_vertices(reference_mesh.faces), dtype=np.int64)
    if boundary_vertex_indices.size == 0:
        return 0.0, 0
    axis_index = reference_mesh.axis_index
    dim0, dim1 = projected_dimensions(axis_index)
    boundary_projected = np.asarray(reference_mesh.vertices[boundary_vertex_indices][:, [dim0, dim1]], dtype=np.float64)
    if boundary_projected.shape[0] == 0:
        return 0.0, 0
    from scipy.spatial import cKDTree
    tree = cKDTree(boundary_projected)
    projected_points = np.asarray(points[:, [dim0, dim1]], dtype=np.float64)
    distances, _ = tree.query(projected_points, k=1)
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    near_mask = distances <= float(max_distance) + float(tolerance)
    near_count = int(np.count_nonzero(near_mask))
    near_fraction = float(near_count / projected_points.shape[0]) if projected_points.shape[0] > 0 else 0.0
    return near_fraction, near_count


def global_boundary_proximity_fraction(
    points: np.ndarray,
    boundary_tree: cKDTree | None,
    *,
    boundary_vertices: np.ndarray | None = None,
    boundary_faces: np.ndarray | None = None,
    max_distance: float,
    tolerance: float,
) -> tuple[float, int]:
    if points.shape[0] == 0:
        return 0.0, 0
    query_points = np.asarray(points, dtype=np.float64)
    if (
        igl is not None
        and boundary_vertices is not None
        and boundary_faces is not None
        and boundary_faces.size > 0
    ):
        squared_distances, _, _ = igl.point_mesh_squared_distance(
            query_points,
            np.asarray(boundary_vertices, dtype=np.float64),
            np.asarray(boundary_faces, dtype=np.int64),
        )
        distances = np.sqrt(np.maximum(np.asarray(squared_distances, dtype=np.float64).reshape(-1), 0.0))
    else:
        if boundary_tree is None:
            raise ValueError("boundary_tree must be provided when libigl boundary distance is unavailable.")
        distances, _ = boundary_tree.query(query_points, k=1)
        distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    near_mask = distances <= float(max_distance) + float(tolerance)
    near_count = int(np.count_nonzero(near_mask))
    near_fraction = float(near_count / points.shape[0]) if points.shape[0] > 0 else 0.0
    return near_fraction, near_count


def global_boundary_proximity_mask(
    points: np.ndarray,
    boundary_tree: cKDTree | None,
    *,
    boundary_vertices: np.ndarray | None = None,
    boundary_faces: np.ndarray | None = None,
    max_distance: float,
    tolerance: float,
) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    query_points = np.asarray(points, dtype=np.float64)
    if (
        igl is not None
        and boundary_vertices is not None
        and boundary_faces is not None
        and boundary_faces.size > 0
    ):
        squared_distances, _, _ = igl.point_mesh_squared_distance(
            query_points,
            np.asarray(boundary_vertices, dtype=np.float64),
            np.asarray(boundary_faces, dtype=np.int64),
        )
        distances = np.sqrt(np.maximum(np.asarray(squared_distances, dtype=np.float64).reshape(-1), 0.0))
    else:
        if boundary_tree is None:
            raise ValueError("boundary_tree must be provided when libigl boundary distance is unavailable.")
        distances, _ = boundary_tree.query(query_points, k=1)
        distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    return distances <= float(max_distance) + float(tolerance)


def batched_global_boundary_proximity(
    mesh_files: dict[LayerKey, Path],
    all_keys: list[LayerKey],
    *,
    axis: str,
    sample_mode: str,
    boundary_tree: cKDTree | None,
    boundary_vertices: np.ndarray | None,
    boundary_faces: np.ndarray | None,
    max_distance: float,
    tolerance: float,
) -> dict[LayerKey, tuple[float, int, int]]:
    reference_cache: dict[LayerKey, ReferenceMesh3Axis] = {}
    layer_sample_counts: dict[LayerKey, int] = {}
    stacked_points: list[np.ndarray] = []

    for key in all_keys:
        reference = reference_cache.get(key)
        if reference is None:
            reference = load_reference_mesh_3axis(mesh_files[key], key, axis=axis)
            reference_cache[key] = reference
        sample_points = sample_points_for_mesh(reference, mode=sample_mode)
        sample_count = int(sample_points.shape[0])
        layer_sample_counts[key] = sample_count
        if sample_count > 0:
            stacked_points.append(np.asarray(sample_points, dtype=np.float64))

    if not stacked_points:
        return {key: (0.0, 0, 0) for key in all_keys}

    all_points = np.vstack(stacked_points)
    near_mask = global_boundary_proximity_mask(
        all_points,
        boundary_tree,
        boundary_vertices=boundary_vertices,
        boundary_faces=boundary_faces,
        max_distance=max_distance,
        tolerance=tolerance,
    )

    results: dict[LayerKey, tuple[float, int, int]] = {}
    start = 0
    for key in all_keys:
        sample_count = layer_sample_counts[key]
        if sample_count <= 0:
            results[key] = (0.0, 0, 0)
            continue
        end = start + sample_count
        near_count = int(np.count_nonzero(near_mask[start:end]))
        near_fraction = float(near_count / sample_count)
        results[key] = (near_fraction, near_count, sample_count)
        start = end
    return results


def query_coverage_from_above(
    points: np.ndarray,
    reference_mesh: ReferenceMesh3Axis,
    *,
    k_nearest_vertices: int,
    tolerance: float,
    tree_query_workers: int,
) -> np.ndarray:
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

    covered = np.zeros((projected_points.shape[0],), dtype=bool)
    face_projected = np.asarray(reference_mesh.face_projected, dtype=np.float64)
    face_axis_values = np.asarray(reference_mesh.face_axis_values, dtype=np.float64)

    for point_idx, (projected_point, point_axis_value, point_vertex_indices) in enumerate(
        zip(projected_points, point_axis_values, nearest_vertex_indices, strict=False)
    ):
        incident_lists = [
            reference_mesh.vertex_incident_faces[int(vertex_idx)]
            for vertex_idx in point_vertex_indices
        ]
        if not incident_lists:
            continue
        candidate_faces = np.unique(np.concatenate(incident_lists))
        if candidate_faces.size == 0:
            continue

        triangles_projected = face_projected[candidate_faces]
        triangles_axis_values = face_axis_values[candidate_faces]

        a = triangles_projected[:, 0, :]
        b = triangles_projected[:, 1, :]
        c = triangles_projected[:, 2, :]
        v0 = b - a
        v1 = c - a
        v2 = projected_point[None, :] - a

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
        valid &= support_values >= point_axis_value - tolerance
        if np.any(valid):
            covered[point_idx] = True

    return covered


def classify_layers(
    mesh_files: dict[LayerKey, Path],
    *,
    axis: str,
    lookahead_main_count: int,
    min_uncovered_fraction: float,
    k_nearest_vertices: int,
    sample_mode: str,
    tolerance: float,
    layer_workers: int = 1,
    tree_query_workers: int = 1,
    boundary_precheck_enabled: bool = False,
    boundary_precheck_distance: float = 1.5,
    boundary_precheck_min_fraction: float = 0.30,
    global_boundary_mesh: Path | None = None,
    global_boundary_distance: float = 1.5,
    global_boundary_min_fraction: float = 0.30,
    normalize_global_boundary_base_to_zero: bool = False,
) -> list[dict[str, object]]:
    all_keys = sorted(mesh_files.keys(), key=lambda key: (key.main, key.sub))
    classify_start = time.perf_counter()
    axis_index = axis_to_index(axis)
    global_boundary_tree: cKDTree | None = None
    global_boundary_vertices: np.ndarray | None = None
    global_boundary_faces: np.ndarray | None = None
    global_boundary_enabled = False
    if global_boundary_mesh is not None:
        print(f"[1/4] Loading global boundary mesh: {global_boundary_mesh}")
        global_vertices, global_faces = load_obj_mesh(str(global_boundary_mesh))
        if normalize_global_boundary_base_to_zero:
            global_base = float(np.min(global_vertices[:, axis_index]))
            global_vertices = apply_axis_lift(global_vertices, axis=axis, amount=-global_base)
            print(f"[1/4] Normalizing global-boundary base: {axis}={global_base:.6g} -> 0")
        global_boundary_vertices = np.asarray(global_vertices, dtype=np.float64)
        global_boundary_faces = np.asarray(global_faces, dtype=np.int64)
        if global_boundary_vertices.shape[0] == 0:
            raise ValueError(f"Global boundary mesh has no vertices: {global_boundary_mesh}")
        if igl is None or global_boundary_faces.size == 0:
            global_boundary_tree = cKDTree(global_boundary_vertices)
        global_boundary_enabled = True
        print(
            f"[1/4] Global boundary ready "
            f"(vertices={global_boundary_vertices.shape[0]}, faces={global_boundary_faces.shape[0]})"
        )
    elif boundary_precheck_enabled:
        print("[1/4] Using local layer-boundary precheck only")
    else:
        print("[1/4] No coarse precheck enabled; all layers go directly to fine coverage test")
    if layer_workers <= 1:
        print(f"[2/4] Running serial classification for {len(all_keys)} layers")
        reference_cache: dict[LayerKey, ReferenceMesh3Axis] = {}
        results: list[dict[str, object]] = []

        def load_cached(key: LayerKey) -> ReferenceMesh3Axis:
            cached = reference_cache.get(key)
            if cached is None:
                cached = load_reference_mesh_3axis(mesh_files[key], key, axis=axis)
                reference_cache[key] = cached
            return cached

        for layer_index, key in enumerate(all_keys, start=1):
            print(f"[2/4] Fine coverage {layer_index}/{len(all_keys)}: {key.stem}", flush=True)
            current_mesh = load_cached(key)
            sample_points = sample_points_for_mesh(current_mesh, mode=sample_mode)
            boundary_candidate = True
            boundary_precheck_fraction = 0.0
            boundary_precheck_count = 0
            global_boundary_candidate = True
            global_boundary_fraction = 0.0
            global_boundary_count = 0
            if global_boundary_enabled:
                global_boundary_fraction, global_boundary_count = global_boundary_proximity_fraction(
                    sample_points,
                    global_boundary_tree,
                    boundary_vertices=global_boundary_vertices,
                    boundary_faces=global_boundary_faces,
                    max_distance=global_boundary_distance,
                    tolerance=tolerance,
                )
                global_boundary_candidate = bool(global_boundary_fraction >= global_boundary_min_fraction)
            elif boundary_precheck_enabled:
                boundary_precheck_fraction, boundary_precheck_count = boundary_proximity_fraction(
                    sample_points,
                    current_mesh,
                    max_distance=boundary_precheck_distance,
                    tolerance=tolerance,
                )
                boundary_candidate = bool(boundary_precheck_fraction >= boundary_precheck_min_fraction)
            if sample_points.shape[0] == 0:
                uncovered_fraction = 0.0
                uncovered_count = 0
                sample_count = 0
                top_like = False
                future_layers: list[str] = []
            elif not global_boundary_candidate or not boundary_candidate:
                uncovered_fraction = 0.0
                uncovered_count = 0
                sample_count = int(sample_points.shape[0])
                top_like = False
                future_layers = []
            else:
                candidate_mains = future_main_indices(all_keys, key.main, lookahead_main_count)
                future_keys = [future_key for future_key in all_keys if future_key.main in candidate_mains]
                covered = np.zeros((sample_points.shape[0],), dtype=bool)
                for future_key in future_keys:
                    future_mesh = load_cached(future_key)
                    covered |= query_coverage_from_above(
                        sample_points,
                        future_mesh,
                        k_nearest_vertices=k_nearest_vertices,
                        tolerance=tolerance,
                        tree_query_workers=tree_query_workers,
                    )
                    if np.all(covered):
                        break
                uncovered_mask = ~covered
                uncovered_count = int(np.count_nonzero(uncovered_mask))
                sample_count = int(sample_points.shape[0])
                uncovered_fraction = float(uncovered_count / sample_count) if sample_count > 0 else 0.0
                top_like = bool(uncovered_fraction >= min_uncovered_fraction)
                future_layers = [future_key.stem for future_key in future_keys]

            results.append(
                {
                    "layer_main": key.main,
                    "layer_sub": key.sub,
                    "mesh_file": mesh_files[key].name,
                    "sample_mode": sample_mode,
                    "sample_count": sample_count,
                    "global_boundary_precheck_enabled": bool(global_boundary_enabled),
                    "global_boundary_fraction": global_boundary_fraction,
                    "global_boundary_count": global_boundary_count,
                    "global_boundary_candidate": global_boundary_candidate,
                    "boundary_precheck_enabled": bool(boundary_precheck_enabled),
                    "boundary_precheck_fraction": boundary_precheck_fraction,
                    "boundary_precheck_count": boundary_precheck_count,
                    "boundary_candidate": boundary_candidate,
                    "uncovered_count": uncovered_count,
                    "uncovered_fraction": uncovered_fraction,
                    "lookahead_main_count": lookahead_main_count,
                    "future_layers_checked": future_layers,
                    "top_like": top_like,
                }
            )
        elapsed = time.perf_counter() - classify_start
        print(f"[4/4] Classification complete in {elapsed:.2f}s")
        return results

    mesh_path_lookup = {key.stem: str(path) for key, path in mesh_files.items()}
    jobs = []
    for key in all_keys:
        candidate_mains = future_main_indices(all_keys, key.main, lookahead_main_count)
        future_keys = [future_key for future_key in all_keys if future_key.main in candidate_mains]
        jobs.append(
            {
                "key_stem": key.stem,
                "mesh_path": mesh_path_lookup[key.stem],
                "future_key_stems": [future_key.stem for future_key in future_keys],
                "future_mesh_paths": [mesh_path_lookup[future_key.stem] for future_key in future_keys],
                "axis": axis,
                "k_nearest_vertices": k_nearest_vertices,
                "sample_mode": sample_mode,
                "tolerance": tolerance,
                "min_uncovered_fraction": min_uncovered_fraction,
                "lookahead_main_count": lookahead_main_count,
                "tree_query_workers": tree_query_workers,
                "boundary_precheck_enabled": boundary_precheck_enabled,
                "boundary_precheck_distance": boundary_precheck_distance,
                "boundary_precheck_min_fraction": boundary_precheck_min_fraction,
                "global_boundary_enabled": bool(global_boundary_enabled),
                "global_boundary_distance": global_boundary_distance,
                "global_boundary_min_fraction": global_boundary_min_fraction,
                "global_boundary_fraction": 0.0,
                "global_boundary_count": 0,
                "global_boundary_candidate": True,
            }
        )
    print(f"[2/4] Prepared {len(jobs)} fine-check jobs across {len(all_keys)} layers")
    if global_boundary_enabled:
        print("[3/4] Running coarse global-boundary candidate filter")
        print("[3/4] Collecting layer samples for one batched global-boundary distance pass")
        coarse_stats = batched_global_boundary_proximity(
            mesh_files,
            all_keys,
            axis=axis,
            sample_mode=sample_mode,
            boundary_tree=global_boundary_tree,
            boundary_vertices=global_boundary_vertices,
            boundary_faces=global_boundary_faces,
            max_distance=global_boundary_distance,
            tolerance=tolerance,
        )

        filtered_jobs: list[dict[str, object]] = []
        skipped_results: list[dict[str, object]] = []
        for job, key in zip(jobs, all_keys, strict=False):
            global_boundary_fraction, global_boundary_count, sample_count = coarse_stats[key]
            global_boundary_candidate = bool(global_boundary_fraction >= global_boundary_min_fraction)
            job["global_boundary_fraction"] = global_boundary_fraction
            job["global_boundary_count"] = global_boundary_count
            job["global_boundary_candidate"] = global_boundary_candidate
            if global_boundary_candidate:
                filtered_jobs.append(job)
            else:
                skipped_results.append(
                    {
                        "layer_main": key.main,
                        "layer_sub": key.sub,
                        "mesh_file": mesh_files[key].name,
                        "sample_mode": sample_mode,
                        "sample_count": sample_count,
                        "global_boundary_precheck_enabled": True,
                        "global_boundary_fraction": global_boundary_fraction,
                        "global_boundary_count": global_boundary_count,
                        "global_boundary_candidate": False,
                        "boundary_precheck_enabled": bool(boundary_precheck_enabled),
                        "boundary_precheck_fraction": 0.0,
                        "boundary_precheck_count": 0,
                        "boundary_candidate": True,
                        "uncovered_count": 0,
                        "uncovered_fraction": 0.0,
                        "lookahead_main_count": lookahead_main_count,
                        "future_layers_checked": [],
                        "top_like": False,
                    }
                )
        jobs = filtered_jobs
        print(
            f"[3/4] Coarse filter kept {len(jobs)} candidate layers and skipped {len(skipped_results)} layers"
        )
    elif boundary_precheck_enabled:
        print("[3/4] Fine jobs will apply local layer-boundary precheck inside each worker")
    else:
        print("[3/4] Sending all layers to fine coverage test")

    print(f"[4/4] Launching fine coverage workers (layer_workers={layer_workers})")
    with ProcessPoolExecutor(max_workers=layer_workers, mp_context=get_context('spawn')) as executor:
        futures = [executor.submit(_classify_layer_job, job) for job in jobs]
        results = []
        for completed_count, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[4/4] Fine coverage complete {completed_count}/{len(futures)}: "
                f"{result['layer_main']}_{result['layer_sub']}",
                flush=True,
            )
    if global_boundary_enabled:
        results.extend(skipped_results)
    results.sort(key=lambda item: (int(item["layer_main"]), int(item["layer_sub"])))
    elapsed = time.perf_counter() - classify_start
    print(f"[4/4] Classification complete in {elapsed:.2f}s")
    return results


def _classify_layer_job(job: dict[str, object]) -> dict[str, object]:
    key_stem = str(job["key_stem"])
    key_main_str, key_sub_str = key_stem.split("_", maxsplit=1)
    key = LayerKey(main=int(key_main_str), sub=int(key_sub_str))
    axis = str(job["axis"])
    tolerance = float(job["tolerance"])
    k_nearest_vertices = int(job["k_nearest_vertices"])
    sample_mode = str(job["sample_mode"])
    min_uncovered_fraction = float(job["min_uncovered_fraction"])
    lookahead_main_count = int(job["lookahead_main_count"])
    tree_query_workers = int(job["tree_query_workers"])
    boundary_precheck_enabled = bool(job["boundary_precheck_enabled"])
    boundary_precheck_distance = float(job["boundary_precheck_distance"])
    boundary_precheck_min_fraction = float(job["boundary_precheck_min_fraction"])
    global_boundary_enabled = bool(job["global_boundary_enabled"])
    global_boundary_fraction = float(job["global_boundary_fraction"])
    global_boundary_count = int(job["global_boundary_count"])
    global_boundary_candidate = bool(job["global_boundary_candidate"])
    current_mesh = load_reference_mesh_3axis(Path(str(job["mesh_path"])), key, axis=axis)
    sample_points = sample_points_for_mesh(current_mesh, mode=sample_mode)
    future_layers = [str(stem) for stem in job["future_key_stems"]]
    boundary_candidate = True
    boundary_precheck_fraction = 0.0
    boundary_precheck_count = 0
    if boundary_precheck_enabled:
        boundary_precheck_fraction, boundary_precheck_count = boundary_proximity_fraction(
            sample_points,
            current_mesh,
            max_distance=boundary_precheck_distance,
            tolerance=tolerance,
        )
        boundary_candidate = bool(boundary_precheck_fraction >= boundary_precheck_min_fraction)
    if sample_points.shape[0] == 0:
        uncovered_fraction = 0.0
        uncovered_count = 0
        sample_count = 0
        top_like = False
    elif not global_boundary_candidate or not boundary_candidate:
        uncovered_fraction = 0.0
        uncovered_count = 0
        sample_count = int(sample_points.shape[0])
        top_like = False
        future_layers = []
    else:
        covered = np.zeros((sample_points.shape[0],), dtype=bool)
        for future_stem, future_mesh_path in zip(job["future_key_stems"], job["future_mesh_paths"], strict=False):
            future_main_str, future_sub_str = str(future_stem).split("_", maxsplit=1)
            future_key = LayerKey(main=int(future_main_str), sub=int(future_sub_str))
            future_mesh = load_reference_mesh_3axis(Path(str(future_mesh_path)), future_key, axis=axis)
            covered |= query_coverage_from_above(
                sample_points,
                future_mesh,
                k_nearest_vertices=k_nearest_vertices,
                tolerance=tolerance,
                tree_query_workers=tree_query_workers,
            )
            if np.all(covered):
                break
        uncovered_mask = ~covered
        uncovered_count = int(np.count_nonzero(uncovered_mask))
        sample_count = int(sample_points.shape[0])
        uncovered_fraction = float(uncovered_count / sample_count) if sample_count > 0 else 0.0
        top_like = bool(uncovered_fraction >= min_uncovered_fraction)

    return {
        "layer_main": key.main,
        "layer_sub": key.sub,
        "mesh_file": Path(str(job["mesh_path"])).name,
        "sample_mode": sample_mode,
        "sample_count": sample_count,
        "global_boundary_precheck_enabled": global_boundary_enabled,
        "global_boundary_fraction": global_boundary_fraction,
        "global_boundary_count": global_boundary_count,
        "global_boundary_candidate": global_boundary_candidate,
        "boundary_precheck_enabled": boundary_precheck_enabled,
        "boundary_precheck_fraction": boundary_precheck_fraction,
        "boundary_precheck_count": boundary_precheck_count,
        "boundary_candidate": boundary_candidate,
        "uncovered_count": uncovered_count,
        "uncovered_fraction": uncovered_fraction,
        "lookahead_main_count": lookahead_main_count,
        "future_layers_checked": future_layers,
        "top_like": top_like,
    }


def layer_color(index: int, total: int) -> tuple[float, float, float]:
    if total <= 0:
        return (1.0, 1.0, 1.0)
    hue = index / max(total, 1)
    return colorsys.hsv_to_rgb(hue, 0.65, 0.9)


def preview_classification(
    mesh_files: dict[LayerKey, Path],
    results: list[dict[str, object]],
    *,
    top_color: str,
    other_color: str,
    top_opacity: float,
    other_opacity: float,
    axis: str,
) -> None:
    try:
        import pyvista as pv  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Preview requires the optional `pyvista` package.") from exc

    result_lookup = {
        (int(item["layer_main"]), int(item["layer_sub"])): item
        for item in results
    }
    ordered_keys = sorted(mesh_files.keys(), key=lambda key: (key.main, key.sub))
    print(f"[preview] Rendering {len(ordered_keys)} layers")
    plotter = pv.Plotter()
    reference_cache: dict[LayerKey, ReferenceMesh3Axis] = {}

    def load_cached(key: LayerKey) -> ReferenceMesh3Axis:
        cached = reference_cache.get(key)
        if cached is None:
            cached = load_reference_mesh_3axis(mesh_files[key], key, axis=axis)
            reference_cache[key] = cached
        return cached

    for idx, key in enumerate(ordered_keys):
        item = result_lookup[(key.main, key.sub)]
        reference = load_cached(key)
        vertices, faces = reference.vertices, reference.faces
        mesh = to_pyvista_polydata(vertices, faces)
        is_top = bool(item["top_like"])
        if is_top:
            color = top_color
            opacity = top_opacity
        else:
            color = other_color
            opacity = other_opacity
        plotter.add_mesh(
            mesh,
            color=color,
            opacity=opacity,
            show_edges=False,
        )
    plotter.add_text(
        "Top-like layers highlighted",
        position="upper_right",
        font_size=10,
    )
    plotter.show()


def main() -> None:
    args = build_parser().parse_args()
    overall_start = time.perf_counter()
    if args.lookahead_main_count < 0:
        raise ValueError("--lookahead-main-count must be non-negative.")
    if args.layer_workers <= 0:
        raise ValueError("--layer-workers must be positive.")
    if args.tree_query_workers == 0 or args.tree_query_workers < -1:
        raise ValueError("--tree-query-workers must be positive or -1.")
    if args.global_boundary_distance < 0.0:
        raise ValueError("--global-boundary-distance must be non-negative.")
    if not (0.0 <= args.global_boundary_min_fraction <= 1.0):
        raise ValueError("--global-boundary-min-fraction must lie in [0, 1].")
    if args.boundary_precheck_distance < 0.0:
        raise ValueError("--boundary-precheck-distance must be non-negative.")
    if not (0.0 <= args.boundary_precheck_min_fraction <= 1.0):
        raise ValueError("--boundary-precheck-min-fraction must lie in [0, 1].")
    if args.k_nearest_vertices <= 0:
        raise ValueError("--k-nearest-vertices must be positive.")
    if not (0.0 <= args.min_uncovered_fraction <= 1.0):
        raise ValueError("--min-uncovered-fraction must lie in [0, 1].")

    mesh_files = collect_layer_files(args.mesh_dir, "obj")
    if not mesh_files:
        raise FileNotFoundError(f"No layer OBJ files were found in {args.mesh_dir}")
    print(f"[setup] Loaded {len(mesh_files)} layer meshes from {args.mesh_dir}")

    results = classify_layers(
        mesh_files,
        axis=str(args.axis),
        lookahead_main_count=int(args.lookahead_main_count),
        min_uncovered_fraction=float(args.min_uncovered_fraction),
        k_nearest_vertices=int(args.k_nearest_vertices),
        sample_mode=str(args.sample_mode),
        tolerance=float(args.tolerance),
        layer_workers=int(args.layer_workers),
        tree_query_workers=int(args.tree_query_workers),
        boundary_precheck_enabled=bool(args.boundary_precheck_enabled),
        boundary_precheck_distance=float(args.boundary_precheck_distance),
        boundary_precheck_min_fraction=float(args.boundary_precheck_min_fraction),
        global_boundary_mesh=args.global_boundary_mesh.resolve() if args.global_boundary_mesh is not None else None,
        global_boundary_distance=float(args.global_boundary_distance),
        global_boundary_min_fraction=float(args.global_boundary_min_fraction),
        normalize_global_boundary_base_to_zero=bool(args.normalize_global_boundary_base_to_zero),
    )

    top_like_layers = [item for item in results if bool(item["top_like"])]
    print(f"Mesh folder: {args.mesh_dir}")
    print(f"Axis: {args.axis}")
    print(f"Lookahead main count: {args.lookahead_main_count}")
    print(f"Min uncovered fraction: {args.min_uncovered_fraction:.3f}")
    print(f"Sample mode: {args.sample_mode}")
    print(f"Layer workers: {args.layer_workers}")
    print(f"KD-tree query workers: {args.tree_query_workers}")
    print(f"Normalize global-boundary base to zero: {bool(args.normalize_global_boundary_base_to_zero)}")
    print(f"Global boundary precheck enabled: {args.global_boundary_mesh is not None}")
    if args.global_boundary_mesh is not None:
        print(f"Global boundary mesh: {args.global_boundary_mesh}")
        print(f"Global boundary distance: {args.global_boundary_distance}")
        print(f"Global boundary min fraction: {args.global_boundary_min_fraction:.3f}")
    print(f"Boundary precheck enabled: {bool(args.boundary_precheck_enabled)}")
    if args.boundary_precheck_enabled:
        print(f"Boundary precheck distance: {args.boundary_precheck_distance}")
        print(f"Boundary precheck min fraction: {args.boundary_precheck_min_fraction:.3f}")
    print(f"Top-like layers: {len(top_like_layers)} / {len(results)}")
    for item in top_like_layers:
        print(
            f"{int(item['layer_main'])}_{int(item['layer_sub'])}: "
            f"uncovered_fraction={float(item['uncovered_fraction']):.4f} "
            f"({int(item['uncovered_count'])}/{int(item['sample_count'])})"
        )

    if args.output_json is not None:
        print(f"[output] Writing JSON report: {args.output_json}")
        payload = {
            "mesh_dir": str(args.mesh_dir),
            "axis": str(args.axis),
            "lookahead_main_count": int(args.lookahead_main_count),
            "min_uncovered_fraction": float(args.min_uncovered_fraction),
            "k_nearest_vertices": int(args.k_nearest_vertices),
            "sample_mode": str(args.sample_mode),
            "tolerance": float(args.tolerance),
            "layer_workers": int(args.layer_workers),
            "tree_query_workers": int(args.tree_query_workers),
            "boundary_precheck_enabled": bool(args.boundary_precheck_enabled),
            "boundary_precheck_distance": float(args.boundary_precheck_distance),
            "boundary_precheck_min_fraction": float(args.boundary_precheck_min_fraction),
            "global_boundary_mesh": (
                str(args.global_boundary_mesh.resolve()) if args.global_boundary_mesh is not None else None
            ),
            "global_boundary_distance": float(args.global_boundary_distance),
            "global_boundary_min_fraction": float(args.global_boundary_min_fraction),
            "normalize_global_boundary_base_to_zero": bool(args.normalize_global_boundary_base_to_zero),
            "layers": results,
        }
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved JSON report: {args.output_json}")

    if args.preview:
        print("[preview] Opening classification preview")
        preview_classification(
            mesh_files,
            results,
            top_color=str(args.top_color),
            other_color=str(args.other_color),
            top_opacity=float(args.top_opacity),
            other_opacity=float(args.other_opacity),
            axis=str(args.axis),
        )
    overall_elapsed = time.perf_counter() - overall_start
    print(f"[done] Total runtime: {overall_elapsed:.2f}s")


if __name__ == "__main__":
    main()
