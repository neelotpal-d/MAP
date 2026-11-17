"""Nearest-surface 3-axis height stage for projected and intrinsic toolpaths.

Unlike the fixed-axis solver, this stage uses the shortest 3D point-to-triangle
distance to recent support layers. It processes contiguous layer groups in
separate processes so each worker can reuse its local support-mesh cache.
"""

from __future__ import annotations

from multiprocessing import get_context

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import shutil
import os
import time
from pathlib import Path

import numpy as np

from batch_compute_heights import (
    LayerKey,
    copy_path_breaks_for_layer,
    collect_layer_files,
    copy_path_tags_for_layer,
    copy_width_profile_for_layer,
    infer_model_lift_from_metadata,
    load_existing_metadata,
    previous_main_indices,
)
from extrusion_height import (
    apply_axis_lift,
    compute_point_heights_3axis,
    load_path_rows,
    load_reference_mesh_3axis,
    prepare_reference_mesh_3axis,
    save_path_with_heights,
)
from mesh_distance_workflow import load_obj_mesh

try:
    import igl  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("This experiment requires libigl Python bindings (`igl`).") from exc


def clear_directory(directory: Path) -> None:
    if not directory.exists():
        return
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def nearest_surface_heights(
    points: np.ndarray,
    reference_meshes: list[tuple[np.ndarray, np.ndarray]],
    *,
    reference_trees: list[object] | None = None,
    capture_queries: list | None = None,
    return_closest: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return exact distances and, by default, closest support points.

    An optional collector retains each reference's native distance/face arrays
    for an immediate correction pass on the same coordinates.
    """
    best_squared = np.full(points.shape[0], np.inf, dtype=np.float64)
    best_points = np.zeros_like(points, dtype=np.float64) if return_closest else None
    if reference_trees is not None and len(reference_trees) != len(reference_meshes):
        raise ValueError("Each reference mesh requires one nearest-distance tree.")
    for index, (vertices, faces) in enumerate(reference_meshes):
        squared, nearest_faces, closest = (igl.point_mesh_squared_distance(points, vertices, faces)
                              if reference_trees is None else
                              reference_trees[index].squared_distance(vertices, faces, points))
        squared = np.asarray(squared, dtype=np.float64).reshape(-1)
        if capture_queries is not None:
            capture_queries.append((squared, np.asarray(nearest_faces).reshape(-1)))
        replace = squared < best_squared
        best_squared[replace] = squared[replace]
        if return_closest:
            best_points[replace] = np.asarray(closest, dtype=np.float64)[replace]
    return np.sqrt(np.maximum(best_squared, 0.0)), best_points


def _split_contiguous_groups(items: list[dict[str, object]], worker_count: int) -> list[list[dict[str, object]]]:
    """Split sorted layer jobs so each process can reuse nearby support meshes."""
    group_count = min(max(int(worker_count), 1), len(items))
    group_size = (len(items) + group_count - 1) // group_count
    return [items[start : start + group_size] for start in range(0, len(items), group_size)]


def _compute_nearest_height_group(
    jobs: list[dict[str, object]],
    *,
    output_dir: Path,
    platform_value: float,
    max_platform_height: float | None,
    nearest_direct_height: float,
    model_lift: float,
    model_lift_axis: str,
    finalize_layer=None,
    progress_interval: float=0.,
    validation_config=None,
    validation_catalog=None,
) -> list[dict[str, object]]:
    """Compute a group with local caches and optional in-memory finalization.

    The finalizer runs while reference data is available. Only the current
    layer's query arrays are retained; none are returned through the pool.
    """
    surface_cache: dict[LayerKey, tuple[np.ndarray, np.ndarray]] = {}
    axis_cache: dict[LayerKey, object] = {}
    nearest_cache: dict[LayerKey, object] = {}
    results: list[dict[str, object]] = []
    began=last_report=time.perf_counter();processed_points=0;mesh_loads=0
    validator = None
    if validation_config and validation_config.get('enabled') and finalize_layer is None:
        from support_height_validation import HeightValidator
        validator = HeightValidator(validation_config, validation_catalog or [], platform_value=platform_value,
            max_platform_height=max_platform_height, lift=model_lift, lift_axis=model_lift_axis)
    # Scalar-window references expire quickly, but physical bases may be used
    # much later. Evict by the actual schedule, not by a fixed cache capacity.
    last_use = {key: index for index, job in enumerate(jobs) for key, _ in job['reference_items']}
    for job_index, job in enumerate(jobs):
        key = job["key"]
        path_path = job["path_path"]
        reference_items = job["reference_items"]
        if not isinstance(key, LayerKey) or not isinstance(path_path, Path) or not isinstance(reference_items, list):
            raise TypeError("Invalid nearest-height worker job.")

        references: list[tuple[np.ndarray, np.ndarray]] = []
        axis_references: list[object] = []
        reference_keys: list[LayerKey] = []
        reference_trees: list[object] = []
        for support_key, support_path in reference_items:
            if not isinstance(support_key, LayerKey) or not isinstance(support_path, Path):
                raise TypeError("Invalid nearest-height support reference.")
            surface = surface_cache.get(support_key)
            if surface is None:
                vertices, faces = load_obj_mesh(str(support_path))
                if abs(model_lift) > 0.0:
                    vertices = apply_axis_lift(vertices, axis=model_lift_axis, amount=model_lift)
                surface = np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)
                surface_cache[support_key] = surface
                mesh_loads+=1
            if support_key not in nearest_cache:
                tree = igl.AABB()
                tree.init(*surface)
                nearest_cache[support_key] = tree
            references.append(surface)
            reference_keys.append(support_key)
            reference_trees.append(nearest_cache[support_key])

        path_rows = load_path_rows(path_path)
        points = path_rows[:, 0:3]
        processed_points+=len(points)
        platform_mask = np.zeros(points.shape[0], dtype=bool)
        nearest_heights = np.empty(0, dtype=np.float64)
        nearest_queries = [] if finalize_layer is not None else None
        if references:
            nearest_heights, _ = nearest_surface_heights(points, references, reference_trees=reference_trees,
                capture_queries=nearest_queries, return_closest=False)
            heights = nearest_heights.copy()
            y_support_needed = nearest_heights > nearest_direct_height
            if np.any(y_support_needed):
                for support_key, support_path in reference_items:
                    if support_key not in axis_cache:
                        axis_cache[support_key] = prepare_reference_mesh_3axis(
                            *surface_cache[support_key], support_key, path=support_path, axis="y")
                    axis_references.append(axis_cache[support_key])
                axis_heights, axis_platform_mask = compute_point_heights_3axis(
                    points[y_support_needed],
                    axis_references,
                    axis="y",
                    platform_value=platform_value,
                    max_height=max_platform_height,
                    return_platform_mask=True,
                )
                checked_heights = np.minimum(nearest_heights[y_support_needed], axis_heights)
                checked_heights[axis_platform_mask] = axis_heights[axis_platform_mask]
                heights[y_support_needed] = checked_heights
                platform_mask[y_support_needed] = axis_platform_mask
        else:
            platform_mask[:] = True
            heights = np.maximum(0.0, points[:, 1] - platform_value)
        if max_platform_height is not None:
            heights[platform_mask] = np.minimum(heights[platform_mask], max_platform_height)

        output_path = output_dir / f"{key.stem}_3axis.txt"
        result = {
                "layer_key": key,
                "output_path": output_path,
                "reference_keys": reference_keys,
                "platform_support_count": int(platform_mask.sum()),
                "platform_support_fraction": float(platform_mask.mean()) if platform_mask.size else 0.0,
                "nearest_direct_count": int((nearest_heights <= nearest_direct_height).sum()) if references else 0,
                "fixed_y_check_count": int((nearest_heights > nearest_direct_height).sum()) if references else 0,
                "height_min": float(heights.min()) if heights.size else 0.0,
                "height_max": float(heights.max()) if heights.size else 0.0,
            }
        result['_platform_mask'] = platform_mask
        handled = (finalize_layer(job, result, path_rows, heights, references, reference_trees, nearest_queries)
                   if finalize_layer is not None else False)
        if validator is not None:
            rows = np.column_stack((path_rows[:, :6], heights))
            recent = {(ref.main, ref.sub): (v, f, {}, tree)
                for ref, (v, f), tree in zip(reference_keys, references, reference_trees)}
            result['support_validation'] = validator.validate(rows, source={'part': 'model'}, key=key,
                recent=recent, platform_mask=platform_mask, output_path=output_path)
            heights = rows[:, 6]
            result.update(height_min=float(heights.min()) if len(heights) else 0.,
                          height_max=float(heights.max()) if len(heights) else 0.)
        result.pop('_platform_mask', None)
        if not handled:
            save_path_with_heights(output_path, path_rows, heights)
        results.append(result)
        if progress_interval>0 and (time.perf_counter()-last_report>=progress_interval or job_index+1==len(jobs)):
            last_report=time.perf_counter()
            print(f'[heights worker] pid={os.getpid()} group={jobs[0]["key"].stem}..{jobs[-1]["key"].stem} '
                f'layers={job_index+1}/{len(jobs)} points={processed_points} elapsed={last_report-began:.1f}s',flush=True)
        for support_key in reference_keys:
            if last_use[support_key] == job_index:
                surface_cache.pop(support_key, None)
                nearest_cache.pop(support_key, None)
                axis_cache.pop(support_key, None)
    if progress_interval>0 and jobs:
        record=dict(pid=os.getpid(),first_layer=jobs[0]['key'].stem,last_layer=jobs[-1]['key'].stem,
            layers=len(jobs),path_points=processed_points,mesh_loads=mesh_loads,wall_seconds=time.perf_counter()-began)
        (output_dir/f'height_worker_{os.getpid()}_{jobs[0]["key"].stem}.json').write_text(json.dumps(record,indent=2))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experimental nearest-surface height computation for projected hybrid paths.",
    )
    parser.add_argument("mesh_dir", type=Path, help="Expanded support meshes named <main>_<sub>.obj.")
    parser.add_argument("path_dir", type=Path, help="Projected hybrid paths named <main>_<sub>.txt.")
    parser.add_argument("output_dir", type=Path, help="Output folder for *_3axis.txt rows and metadata.")
    parser.add_argument("--previous-main-window", type=int, default=2, help="Number of preceding main indices used as support.")
    parser.add_argument("--platform-value", type=float, default=0.0, help="Fixed y platform used only when no prior support exists.")
    parser.add_argument("--max-platform-height", type=float, default=None, help="Optional cap applied only to platform-fallback points.")
    parser.add_argument(
        "--nearest-direct-height",
        type=float,
        default=0.3,
        help="Use nearest-triangle height directly at or below this value; query fixed-y support only above it.",
    )
    parser.add_argument("--layer-workers", type=int, default=1, help="Process workers, each assigned a contiguous group of layers.")
    parser.add_argument("--clean-output", action="store_true", help="Clear the output folder before writing.")
    return parser


def run_nearest_surface_height_stage(args: argparse.Namespace) -> Path:
    """Run the nearest-triangle 3-axis height stage from CLI or pipeline config."""
    mesh_dir = args.mesh_dir.resolve()
    path_dir = args.path_dir.resolve()
    output_dir = args.output_dir.resolve()
    mesh_files = collect_layer_files(mesh_dir, "obj")
    path_files = collect_layer_files(path_dir, "txt")
    keys = sorted(set(mesh_files) & set(path_files), key=lambda key: (key.main, key.sub))
    if not keys:
        raise FileNotFoundError("No matching <main>_<sub> support mesh/path pairs were found.")
    if args.previous_main_window < 0:
        raise ValueError("previous_main_window must be non-negative.")
    if args.nearest_direct_height < 0.0:
        raise ValueError("nearest_direct_height must be non-negative.")
    if args.layer_workers <= 0:
        raise ValueError("layer_workers must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        clear_directory(output_dir)
        print(f"Cleared output directory: {output_dir}")
    source_metadata = load_existing_metadata(path_dir / "batch_paths.meta.json")
    validation_config = getattr(args, 'support_validation', {})
    validation_catalog = None
    if validation_config.get('enabled'):
        from support_height_validation import build_reference_catalog
        if not validation_config.get('model_solid'):
            print('[height validation] No original model solid supplied; model/material continuity will be unavailable.', flush=True)
        validation_catalog = build_reference_catalog(mesh_files, path_files, source_metadata.get('layers', []),
            lift=infer_model_lift_from_metadata(source_metadata)[0], lift_axis=infer_model_lift_from_metadata(source_metadata)[1],
            bounds_mesh_dir=source_metadata.get('input_dir'))
    source_width = source_metadata.get("width") if isinstance(source_metadata, dict) else None
    model_lift, model_lift_axis = infer_model_lift_from_metadata(source_metadata)
    all_mesh_keys = sorted(mesh_files, key=lambda key: (key.main, key.sub))
    print(f"Computing nearest-surface heights for {len(keys)} layers into {output_dir}")
    print(f"Support window: previous {args.previous_main_window} main layers; model lift: {model_lift} along {model_lift_axis}")
    print(f"Nearest-triangle layer-group processes: {args.layer_workers}")
    jobs: list[dict[str, object]] = []
    for key in keys:
        support_mains = previous_main_indices(all_mesh_keys, key.main, args.previous_main_window)
        support_keys = [mesh_key for mesh_key in all_mesh_keys if mesh_key.main in support_mains]
        jobs.append(
            {
                "key": key,
                "path_path": path_files[key],
                "reference_items": [(support_key, mesh_files[support_key]) for support_key in support_keys],
            }
        )
    groups = _split_contiguous_groups(jobs, int(args.layer_workers))
    print(f"Layer groups: {len(groups)}")
    worker_kwargs = {
        "output_dir": output_dir,
        "platform_value": float(args.platform_value),
        "max_platform_height": args.max_platform_height,
        "nearest_direct_height": float(args.nearest_direct_height),
        "model_lift": model_lift,
        "model_lift_axis": model_lift_axis,
        "validation_config": validation_config,
        "validation_catalog": validation_catalog,
    }
    if len(groups) == 1:
        grouped_results = [_compute_nearest_height_group(groups[0], **worker_kwargs)]
    else:
        with ProcessPoolExecutor(max_workers=len(groups), mp_context=get_context('spawn')) as executor:
            futures = [executor.submit(_compute_nearest_height_group, group, **worker_kwargs) for group in groups]
            grouped_results = [future.result() for future in futures]

    results = [result for group_results in grouped_results for result in group_results]
    results.sort(key=lambda result: (result["layer_key"].main, result["layer_key"].sub))
    layer_outputs: list[dict[str, object]] = []
    for result in results:
        key = result["layer_key"]
        if not isinstance(key, LayerKey):
            raise TypeError("Invalid nearest-height worker result.")
        width_file = copy_width_profile_for_layer(
            source_batch_metadata=source_metadata,
            key=key,
            source_dir=path_dir,
            output_dir=output_dir,
        )
        path_tag_file = copy_path_tags_for_layer(
            source_batch_metadata=source_metadata,
            key=key,
            source_dir=path_dir,
            output_dir=output_dir,
        )
        path_break_file = copy_path_breaks_for_layer(
            source_batch_metadata=source_metadata,
            key=key,
            source_dir=path_dir,
            output_dir=output_dir,
        )
        layer_outputs.append({
            "layer_main": key.main,
            "layer_sub": key.sub,
            "source_mesh": mesh_files[key].name,
            "source_path_file": path_files[key].name,
            "output_path_file": result["output_path"].name,
            "width_profile_file": width_file,
            "path_tag_file": path_tag_file,
            "path_break_file": path_break_file,
            "reference_layers": [support_key.stem for support_key in result["reference_keys"]],
            "platform_support_count": result["platform_support_count"],
            "platform_support_fraction": result["platform_support_fraction"],
            "height_measurement": "nearest_triangle_surface_euclidean",
            "nearest_direct_count": result["nearest_direct_count"],
            "fixed_y_check_count": result["fixed_y_check_count"],
            "height_min": result["height_min"],
            "height_max": result["height_max"],
        })
        if 'support_validation' in result:
            layer_outputs[-1]['support_validation'] = result['support_validation']
        print(f"Saved nearest heights: {result['output_path'].name} (references={len(result['reference_keys'])}, max={result['height_max']:.6g})")

    metadata = {
        "batch_kind": "3axis_nearest_surface_path_heights",
        "output_format": "x y z nx ny nz height",
        "height_mode": "nearest_triangle_surface_euclidean",
        "input_mesh_dir": str(mesh_dir),
        "input_path_dir": str(path_dir),
        "output_dir": str(output_dir),
        "previous_main_window": int(args.previous_main_window),
        "platform_value": float(args.platform_value),
        "max_platform_height": args.max_platform_height,
        "nearest_direct_height": float(args.nearest_direct_height),
        "nearest_triangle_layer_group_processes": len(groups),
        "model_lift": model_lift,
        "model_lift_axis": model_lift_axis,
        "width_profile_mode": "per_point_sidecar",
        "width": source_width,
        "source_path_metadata": source_metadata,
        "layers": layer_outputs,
    }
    if validation_config.get('enabled'):
        metadata['support_validation'] = dict(validation_config,
            unresolved_points=sum(r['support_validation']['unresolved_points'] for r in results),
            checked_points=sum(r['support_validation']['checked_points'] for r in results),
            recovered_points=sum(r['support_validation']['recovered_points'] for r in results))
        for name in ('normal_miss_fallback_points', 'normal_miss_actual_points', 'normal_miss_cap_points', 'clamped_height_points',
                     'threshold_fallback_points','unverified_support_points','invalid_height_points','geometrically_verified_points',
                     'uncapped_model_normal_points', 'above_cap_model_normal_points'):
            metadata['support_validation'][name] = sum(r['support_validation'][name] for r in results)
    metadata_path = output_dir / "batch_3axis.meta.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved height metadata: {metadata_path.name}")
    return output_dir


def main() -> None:
    run_nearest_surface_height_stage(build_parser().parse_args())


if __name__ == "__main__":
    main()
