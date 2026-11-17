"""
Batch-compute 3-axis extrusion heights for ordered path files.

Inputs:
- layer meshes named like <main>_<sub>.obj
- matching path files named like <main>_<sub>.txt

Outputs:
- augmented path files named like <main>_<sub>_3axis.txt
  with rows: x y z nx ny nz height
"""

from __future__ import annotations

from multiprocessing import get_context

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import re
import shutil
from pathlib import Path

import numpy as np

from extrusion_height import LayerKey
from extrusion_height import compute_cross_sectional_height_scale_3axis
from extrusion_height import compute_point_heights_3axis
from extrusion_height import load_path_rows
from extrusion_height import load_reference_mesh_3axis
from extrusion_height import save_path_with_heights


LAYER_PATTERN = re.compile(r"^(?P<main>\d+)_(?P<sub>\d+)\.(?P<ext>obj|txt)$", re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute 3-axis support heights for layer path files.",
    )
    parser.add_argument(
        "mesh_dir",
        type=Path,
        help="Folder containing layer meshes named like <main>_<sub>.obj.",
    )
    parser.add_argument(
        "path_dir",
        type=Path,
        help="Folder containing path files named like <main>_<sub>.txt.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Folder where <main>_<sub>_3axis.txt files will be written.",
    )
    parser.add_argument(
        "--previous-main-window",
        type=int,
        default=2,
        help="How many previous main-index layer groups to consider as support candidates.",
    )
    parser.add_argument(
        "--axis",
        type=str,
        default="y",
        help="Fixed 3-axis print direction. Default: y.",
    )
    parser.add_argument(
        "--platform-value",
        type=float,
        default=0.0,
        help="Platform plane value along the chosen axis. Default: 0.0.",
    )
    parser.add_argument(
        "--k-nearest-vertices",
        type=int,
        default=12,
        help="Number of nearby projected vertices used to gather candidate support triangles.",
    )
    parser.add_argument(
        "--max-height",
        type=float,
        default=None,
        help="Optional maximum allowed 3-axis height when a point falls back to the platform plane. Mesh-supported points are not clipped.",
    )
    parser.add_argument(
        "--uncapped-first-layer-count",
        type=int,
        default=0,
        help="Number of first main-index layers where --max-height is disabled.",
    )
    parser.add_argument(
        "--auto-uncap-base-layers",
        action="store_true",
        help="Disable --max-height for layers where at least --base-layer-min-fraction of path points fall back to platform support.",
    )
    parser.add_argument(
        "--base-layer-min-fraction",
        type=float,
        default=0.7,
        help="Fraction threshold for --auto-uncap-base-layers. Default 0.7 means at least 70 percent of points must use platform support.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker threads used for per-point height queries.",
    )
    parser.add_argument(
        "--tree-query-workers",
        type=int,
        default=1,
        help="Number of worker threads used internally by SciPy cKDTree.query. Use -1 for all CPU threads.",
    )
    parser.add_argument(
        "--layer-workers",
        type=int,
        default=1,
        help="Number of worker processes used to compute separate layers in parallel. Default keeps the original serial behavior.",
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


def parse_layer_name(path: Path) -> LayerKey | None:
    match = LAYER_PATTERN.match(path.name)
    if match is None:
        return None
    return LayerKey(main=int(match.group("main")), sub=int(match.group("sub")))


def collect_layer_files(folder: Path, expected_ext: str) -> dict[LayerKey, Path]:
    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Folder not found: {folder}")
    files: dict[LayerKey, Path] = {}
    for path in folder.iterdir():
        if not path.is_file():
            continue
        match = LAYER_PATTERN.match(path.name)
        if match is None or match.group("ext").lower() != expected_ext:
            continue
        key = parse_layer_name(path)
        if key is not None:
            files[key] = path
    return files


def previous_main_indices(keys: list[LayerKey], current_main: int, window: int) -> list[int]:
    unique_mains = sorted({key.main for key in keys if key.main < current_main})
    if window <= 0:
        return []
    return unique_mains[-window:]


def load_existing_metadata(metadata_path: Path) -> dict[str, object]:
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def find_layer_metadata_entry(
    batch_metadata: dict[str, object],
    key: LayerKey,
) -> dict[str, object] | None:
    layers = batch_metadata.get("layers")
    if not isinstance(layers, list):
        return None
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        if int(layer.get("layer_main", -1)) != key.main or int(layer.get("layer_sub", -1)) != key.sub:
            continue
        return layer
    return None


def copy_width_profile_for_layer(
    *,
    source_batch_metadata: dict[str, object],
    key: LayerKey,
    source_dir: Path,
    output_dir: Path,
) -> str | None:
    layer_meta = find_layer_metadata_entry(source_batch_metadata, key)
    if layer_meta is None:
        return None
    width_profile_name = layer_meta.get("width_profile_file")
    if not isinstance(width_profile_name, str) or not width_profile_name:
        return None
    source_path = source_dir / width_profile_name
    if not source_path.exists():
        return None
    output_path = output_dir / source_path.name
    if source_path.resolve() != output_path.resolve():
        shutil.copyfile(source_path, output_path)
    return output_path.name


def copy_path_tags_for_layer(
    *,
    source_batch_metadata: dict[str, object],
    key: LayerKey,
    source_dir: Path,
    output_dir: Path,
) -> str | None:
    layer_meta = find_layer_metadata_entry(source_batch_metadata, key)
    if layer_meta is None:
        return None
    tag_name = layer_meta.get("path_tag_file")
    if not isinstance(tag_name, str) or not tag_name:
        return None
    source_path = source_dir / tag_name
    if not source_path.exists():
        return None
    output_path = output_dir / source_path.name
    if source_path.resolve() != output_path.resolve():
        shutil.copyfile(source_path, output_path)
    return output_path.name


def copy_path_breaks_for_layer(
    *,
    source_batch_metadata: dict[str, object],
    key: LayerKey,
    source_dir: Path,
    output_dir: Path,
) -> str | None:
    layer_meta = find_layer_metadata_entry(source_batch_metadata, key)
    if layer_meta is None:
        return None
    break_name = layer_meta.get("path_break_file")
    if not isinstance(break_name, str) or not break_name:
        return None
    source_path = source_dir / break_name
    if not source_path.exists():
        return None
    output_path = output_dir / source_path.name
    if source_path.resolve() != output_path.resolve():
        shutil.copyfile(source_path, output_path)
    return output_path.name


def infer_model_lift_from_metadata(source_batch_metadata: dict[str, object]) -> tuple[float, str]:
    if not isinstance(source_batch_metadata, dict):
        return 0.0, "y"
    try:
        return float(source_batch_metadata.get("model_lift", 0.0)), str(source_batch_metadata.get("model_lift_axis", "y"))
    except Exception:
        return 0.0, "y"


def write_height_metadata(
    *,
    mesh_dir: Path,
    path_dir: Path,
    output_dir: Path,
    axis: str,
    previous_main_window: int,
    platform_value: float,
    k_nearest_vertices: int,
    max_height: float | None,
    uncapped_first_layer_count: int,
    auto_uncap_base_layers: bool,
    base_layer_min_fraction: float,
    num_workers: int,
    tree_query_workers: int,
    layer_workers: int,
    slope_correction: dict[str, object] | None,
    layer_outputs: list[dict[str, object]],
) -> dict[str, object]:
    width = None
    path_role_tag_convention = None
    for layer in layer_outputs:
        source_metadata = layer.get("source_metadata")
        if isinstance(source_metadata, dict):
            width_value = source_metadata.get("width", source_metadata.get("contour_spacing"))
            if width_value is not None:
                width = width_value
            if path_role_tag_convention is None:
                path_role_tag_convention = source_metadata.get("path_role_tag_convention")
                break
    model_lift = 0.0
    model_lift_axis = "y"
    for layer in layer_outputs:
        source_metadata = layer.get("source_metadata")
        if isinstance(source_metadata, dict):
            model_lift, model_lift_axis = infer_model_lift_from_metadata(source_metadata)
            break
    return {
        "batch_kind": "3axis_path_heights",
        "input_mesh_dir": str(mesh_dir),
        "input_path_dir": str(path_dir),
        "output_dir": str(output_dir),
        "output_format": "x y z nx ny nz height",
        "width_profile_mode": "per_point_sidecar",
        "path_role_tag_convention": path_role_tag_convention,
        "path_role_tag_format": "half_open_index_blocks" if path_role_tag_convention else None,
        "height_mode": "3axis",
        "height_axis": axis,
        "width": width,
        "model_lift": model_lift,
        "model_lift_axis": model_lift_axis,
        "previous_main_window": previous_main_window,
        "platform_value": platform_value,
        "k_nearest_vertices": k_nearest_vertices,
        "max_height": max_height,
        "uncapped_first_layer_count": uncapped_first_layer_count,
        "auto_uncap_base_layers": auto_uncap_base_layers,
        "base_layer_min_fraction": base_layer_min_fraction,
        "num_workers": num_workers,
        "tree_query_workers": tree_query_workers,
        "layer_workers": layer_workers,
        "slope_correction": slope_correction,
        "layers": layer_outputs,
    }


def _compute_layer_heights_job(
    *,
    key: LayerKey,
    mesh_path: Path,
    path_path: Path,
    output_dir: Path,
    reference_items: list[tuple[LayerKey, Path]],
    axis: str,
    platform_value: float,
    k_nearest_vertices: int,
    max_height: float | None,
    uncapped_by_first_layers: bool,
    auto_uncap_base_layers: bool,
    base_layer_min_fraction: float,
    num_workers: int,
    tree_query_workers: int,
    model_lift: float = 0.0,
    model_lift_axis: str = "y",
    slope_correction: dict[str, object] | None = None,
) -> dict[str, object]:
    reference_meshes = [
        load_reference_mesh_3axis(
            reference_path,
            reference_key,
            axis=axis,
            coordinate_lift=model_lift,
            coordinate_lift_axis=model_lift_axis,
        )
        for reference_key, reference_path in reference_items
    ]
    path_rows = load_path_rows(path_path)
    (
        heights,
        platform_mask,
        uncapped_by_base_layer,
        effective_max_height,
        slope_correction_stats,
    ) = compute_heights_with_effective_cap(
        path_rows[:, 0:3],
        reference_meshes,
        axis=axis,
        platform_value=platform_value,
        k_nearest_vertices=k_nearest_vertices,
        max_height=max_height,
        uncapped_by_first_layers=uncapped_by_first_layers,
        auto_uncap_base_layers=auto_uncap_base_layers,
        base_layer_min_fraction=base_layer_min_fraction,
        num_workers=num_workers,
        tree_query_workers=tree_query_workers,
        slope_correction=slope_correction,
    )

    output_path = output_dir / f"{key.stem}_3axis.txt"
    save_path_with_heights(output_path, path_rows, heights)
    return {
        "layer_key": key,
        "output_path": output_path,
        "reference_layers": [reference_key.stem for reference_key, _ in reference_items],
        "max_height": float(heights.max()) if heights.size else 0.0,
        "effective_max_height": effective_max_height,
        "uncapped_by_first_layers": uncapped_by_first_layers,
        "uncapped_by_base_layer": uncapped_by_base_layer,
        "slope_correction_stats": slope_correction_stats,
        "platform_support_count": int(platform_mask.sum()),
        "platform_support_fraction": float(platform_mask.mean()) if platform_mask.size else 0.0,
    }


def compute_heights_with_effective_cap(
    path_points,
    reference_meshes,
    *,
    axis: str,
    platform_value: float,
    k_nearest_vertices: int,
    max_height: float | None,
    uncapped_by_first_layers: bool,
    auto_uncap_base_layers: bool,
    base_layer_min_fraction: float,
    num_workers: int,
    tree_query_workers: int,
    slope_correction: dict[str, object] | None = None,
):
    uncapped_heights, platform_mask = compute_point_heights_3axis(
        path_points,
        reference_meshes,
        axis=axis,
        platform_value=platform_value,
        k_nearest_vertices=k_nearest_vertices,
        max_height=None,
        num_workers=num_workers,
        tree_query_workers=tree_query_workers,
        return_platform_mask=True,
    )
    platform_fraction = float(platform_mask.mean()) if platform_mask.size else 0.0
    uncapped_by_base_layer = bool(auto_uncap_base_layers and platform_fraction >= base_layer_min_fraction)
    effective_max_height = None if uncapped_by_first_layers or uncapped_by_base_layer else max_height
    heights = np.asarray(uncapped_heights, dtype=np.float64).copy()
    if effective_max_height is not None:
        heights[platform_mask] = np.minimum(heights[platform_mask], float(effective_max_height))
    slope_correction_stats: dict[str, float | int] | None = None
    if isinstance(slope_correction, dict) and bool(slope_correction.get("enabled", False)):
        min_scale = float(slope_correction.get("min_scale", 0.0))
        scales, slope_platform_mask = compute_cross_sectional_height_scale_3axis(
            path_points,
            reference_meshes,
            axis=axis,
            k_nearest_vertices=k_nearest_vertices,
            min_scale=min_scale,
        )
        correction_mask = ~slope_platform_mask
        if np.any(correction_mask):
            heights = np.asarray(heights, dtype=np.float64).copy()
            heights[correction_mask] *= scales[correction_mask]
            corrected_scales = np.asarray(scales[correction_mask], dtype=np.float64)
            slope_correction_stats = {
                "count": int(corrected_scales.shape[0]),
                "min": float(np.min(corrected_scales)),
                "max": float(np.max(corrected_scales)),
                "mean": float(np.mean(corrected_scales)),
                "std": float(np.std(corrected_scales)),
            }
    return heights, platform_mask, uncapped_by_base_layer, effective_max_height, slope_correction_stats


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    mesh_files = collect_layer_files(args.mesh_dir, "obj")
    path_files = collect_layer_files(args.path_dir, "txt")
    shared_keys = sorted(set(mesh_files) & set(path_files), key=lambda key: (key.main, key.sub))
    if not shared_keys:
        raise FileNotFoundError("No matching <main>_<sub> mesh/path pairs were found.")
    if args.layer_workers <= 0:
        raise ValueError("layer_workers must be positive.")
    if args.uncapped_first_layer_count < 0:
        raise ValueError("uncapped_first_layer_count must be non-negative.")
    if not (0.0 <= args.base_layer_min_fraction <= 1.0):
        raise ValueError("base_layer_min_fraction must be in [0, 1].")

    source_batch_metadata = load_existing_metadata(args.path_dir / "batch_paths.meta.json")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        clear_directory(args.output_dir)
        print(f"Cleared output directory: {args.output_dir}")

    reference_cache: dict[LayerKey, object] = {}
    all_mesh_keys = sorted(mesh_files.keys(), key=lambda key: (key.main, key.sub))

    print(f"Mesh folder: {args.mesh_dir}")
    print(f"Path folder: {args.path_dir}")
    print(f"Output folder: {args.output_dir}")
    print(f"Matched layers: {len(shared_keys)}")
    print(f"Previous main window: {args.previous_main_window}")
    print(f"3-axis direction: {args.axis}")
    print(f"KD-tree query workers: {args.tree_query_workers}")
    print(f"Layer workers: {args.layer_workers}")
    model_lift, model_lift_axis = infer_model_lift_from_metadata(source_batch_metadata)
    print(f"Model lift: {model_lift} along {model_lift_axis}")

    layer_outputs: list[dict[str, object]] = []
    layer_jobs = []
    main_layer_ranks = {main: rank for rank, main in enumerate(sorted({key.main for key in shared_keys}))}
    for key in shared_keys:
        candidate_mains = previous_main_indices(all_mesh_keys, key.main, args.previous_main_window)
        reference_keys = [mesh_key for mesh_key in all_mesh_keys if mesh_key.main in candidate_mains]
        layer_jobs.append(
            {
                "key": key,
                "mesh_path": mesh_files[key],
                "path_path": path_files[key],
                "reference_keys": reference_keys,
            }
        )

    if args.layer_workers == 1:
        for job in layer_jobs:
            key = job["key"]
            reference_keys = job["reference_keys"]
            reference_meshes = []
            for reference_key in reference_keys:
                cached = reference_cache.get(reference_key)
                if cached is None:
                    cached = load_reference_mesh_3axis(
                        mesh_files[reference_key],
                        reference_key,
                        axis=args.axis,
                        coordinate_lift=model_lift,
                        coordinate_lift_axis=model_lift_axis,
                    )
                    reference_cache[reference_key] = cached
                reference_meshes.append(cached)

            path_rows = load_path_rows(path_files[key])
            uncapped_by_first_layers = main_layer_ranks[key.main] < args.uncapped_first_layer_count
            (
                heights,
                platform_mask,
                uncapped_by_base_layer,
                effective_max_height,
                slope_correction_stats,
            ) = compute_heights_with_effective_cap(
                path_rows[:, 0:3],
                reference_meshes,
                axis=args.axis,
                platform_value=args.platform_value,
                k_nearest_vertices=args.k_nearest_vertices,
                max_height=args.max_height,
                uncapped_by_first_layers=uncapped_by_first_layers,
                auto_uncap_base_layers=bool(args.auto_uncap_base_layers),
                base_layer_min_fraction=float(args.base_layer_min_fraction),
                num_workers=args.num_workers,
                tree_query_workers=args.tree_query_workers,
            )
            platform_support_count = int(platform_mask.sum())
            platform_support_fraction = float(platform_mask.mean()) if platform_mask.size else 0.0

            output_path = args.output_dir / f"{key.stem}_3axis.txt"
            save_path_with_heights(output_path, path_rows, heights)
            width_profile_file = copy_width_profile_for_layer(
                source_batch_metadata=source_batch_metadata,
                key=key,
                source_dir=args.path_dir,
                output_dir=args.output_dir,
            )
            path_tag_file = copy_path_tags_for_layer(
                source_batch_metadata=source_batch_metadata,
                key=key,
                source_dir=args.path_dir,
                output_dir=args.output_dir,
            )
            path_break_file = copy_path_breaks_for_layer(
                source_batch_metadata=source_batch_metadata,
                key=key,
                source_dir=args.path_dir,
                output_dir=args.output_dir,
            )
            layer_outputs.append(
                {
                    "layer_main": key.main,
                    "layer_sub": key.sub,
                    "source_mesh": mesh_files[key].name,
                    "source_path_file": path_files[key].name,
                    "output_path_file": output_path.name,
                    "source_metadata_file": "batch_paths.meta.json",
                    "source_metadata": source_batch_metadata,
                    "width_profile_file": width_profile_file,
                    "path_tag_file": path_tag_file,
                    "path_break_file": path_break_file,
                    "reference_layers": [reference_key.stem for reference_key in reference_keys],
                    "effective_max_height": effective_max_height,
                    "uncapped_by_first_layers": uncapped_by_first_layers,
                    "uncapped_by_base_layer": uncapped_by_base_layer,
                    "slope_correction_stats": slope_correction_stats,
                    "platform_support_count": platform_support_count,
                    "platform_support_fraction": platform_support_fraction,
                }
            )
            print(
                f"Saved: {output_path.name} "
                f"(references={len(reference_keys)}, max_height={float(heights.max()) if heights.size else 0.0:.6g})"
            )
    else:
        with ProcessPoolExecutor(max_workers=args.layer_workers, mp_context=get_context('spawn')) as executor:
            futures = []
            for job in layer_jobs:
                key = job["key"]
                reference_items = [
                    (reference_key, mesh_files[reference_key])
                    for reference_key in job["reference_keys"]
                ]
                uncapped_by_first_layers = main_layer_ranks[key.main] < args.uncapped_first_layer_count
                futures.append(
                    executor.submit(
                        _compute_layer_heights_job,
                        key=key,
                        mesh_path=job["mesh_path"],
                        path_path=job["path_path"],
                        output_dir=args.output_dir,
                        reference_items=reference_items,
                        axis=args.axis,
                        platform_value=args.platform_value,
                        k_nearest_vertices=args.k_nearest_vertices,
                        max_height=args.max_height,
                        uncapped_by_first_layers=uncapped_by_first_layers,
                        auto_uncap_base_layers=bool(args.auto_uncap_base_layers),
                        base_layer_min_fraction=float(args.base_layer_min_fraction),
                        num_workers=args.num_workers,
                        tree_query_workers=args.tree_query_workers,
                        model_lift=model_lift,
                        model_lift_axis=model_lift_axis,
                        slope_correction=None,
                    )
                )

            results = [future.result() for future in futures]
        results.sort(key=lambda item: (item["layer_key"].main, item["layer_key"].sub))
        for result in results:
            key = result["layer_key"]
            output_path = result["output_path"]
            width_profile_file = copy_width_profile_for_layer(
                source_batch_metadata=source_batch_metadata,
                key=key,
                source_dir=args.path_dir,
                output_dir=args.output_dir,
            )
            path_tag_file = copy_path_tags_for_layer(
                source_batch_metadata=source_batch_metadata,
                key=key,
                source_dir=args.path_dir,
                output_dir=args.output_dir,
            )
            path_break_file = copy_path_breaks_for_layer(
                source_batch_metadata=source_batch_metadata,
                key=key,
                source_dir=args.path_dir,
                output_dir=args.output_dir,
            )
            layer_outputs.append(
                {
                    "layer_main": key.main,
                    "layer_sub": key.sub,
                    "source_mesh": mesh_files[key].name,
                    "source_path_file": path_files[key].name,
                    "output_path_file": output_path.name,
                    "source_metadata_file": "batch_paths.meta.json",
                    "source_metadata": source_batch_metadata,
                    "width_profile_file": width_profile_file,
                    "path_tag_file": path_tag_file,
                    "path_break_file": path_break_file,
                    "reference_layers": result["reference_layers"],
                    "effective_max_height": result["effective_max_height"],
                    "uncapped_by_first_layers": result["uncapped_by_first_layers"],
                    "uncapped_by_base_layer": result["uncapped_by_base_layer"],
                    "slope_correction_stats": result.get("slope_correction_stats"),
                    "platform_support_count": result["platform_support_count"],
                    "platform_support_fraction": result["platform_support_fraction"],
                }
            )
            print(
                f"Saved: {output_path.name} "
                f"(references={len(result['reference_layers'])}, max_height={result['max_height']:.6g})"
            )

    batch_metadata = write_height_metadata(
        mesh_dir=args.mesh_dir,
        path_dir=args.path_dir,
        output_dir=args.output_dir,
        axis=args.axis,
        previous_main_window=args.previous_main_window,
        platform_value=args.platform_value,
        k_nearest_vertices=args.k_nearest_vertices,
        max_height=args.max_height,
        uncapped_first_layer_count=args.uncapped_first_layer_count,
        auto_uncap_base_layers=bool(args.auto_uncap_base_layers),
        base_layer_min_fraction=float(args.base_layer_min_fraction),
        num_workers=args.num_workers,
        tree_query_workers=args.tree_query_workers,
        layer_workers=args.layer_workers,
        slope_correction=None,
        layer_outputs=layer_outputs,
    )
    batch_metadata_path = args.output_dir / "batch_3axis.meta.json"
    batch_metadata_path.write_text(json.dumps(batch_metadata, indent=2), encoding="utf-8")
    print(f"Saved batch metadata: {batch_metadata_path.name}")


if __name__ == "__main__":
    main()
