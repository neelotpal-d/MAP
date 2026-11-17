"""
Batch-generate hybrid toolpaths from layered OBJ meshes.

Hybrid toolpaths combine:
- boundary-distance contours near the outer shell
- signed-heat parallel contours in the interior

Output rows are saved as:
    <main index>_<sub index>.txt

with columns:
    x y z nx ny nz
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

from batch_generate_paths import iter_layer_meshes
from batch_generate_paths import layer_sort_key
from extrusion_height import apply_axis_lift
from mesh_distance_workflow import load_obj_mesh
from test_hybrid_contours import generate_hybrid_toolpath
from test_hybrid_contours_vectorized import apply_vectorized_extractor_patch
from test_hybrid_contours import encode_path_tag_blocks
from toolpath_export import compute_path_normals
from toolpath_export import save_oriented_path


LAYER_PATTERN = re.compile(r"^(?P<main>\d+)_(?P<sub>\d+)\.obj$", re.IGNORECASE)


def clear_directory(directory: Path) -> None:
    if not directory.exists():
        return
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _build_batch_metadata(
    *,
    input_dir: Path,
    output_dir: Path,
    spacing: float,
    parallel_spacing: float,
    resample_spacing: float,
    parallel_resample_spacing: float,
    boundary_count: int,
    sharp_angle_deg: float,
    boundary_field_backend: str,
    parallel_field_backend: str,
    axis_mode: str,
    axis: str,
    plane_value_mode: str,
    plane_value: float | None,
    level_set_constraint: str,
    soft_level_set_weight: float,
    contour_max_level_fraction: float | None,
    min_polyline_points: int,
    mild_smooth_all: bool,
    mild_smooth_iterations: int,
    mild_smooth_alpha: float,
    first_layers_lowest_y_start_count: int,
    boundary_start_offset_count: int,
    parallel_direction_mode: str,
    model_lift: float,
    model_lift_axis: str,
    use_vectorized_extractor: bool,
    layer_outputs: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "batch_kind": "hybrid_connected_paths",
        "output_format": "x y z nx ny nz",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "path_kind": "hybrid_boundary_parallel_connected",
        "toolpath_mode": "hybrid",
        "contour_spacing": spacing,
        "parallel_spacing": parallel_spacing,
        "resample_spacing": resample_spacing,
        "parallel_resample_spacing": parallel_resample_spacing,
        "sharp_angle_deg": sharp_angle_deg,
        "width": spacing,
        "boundary_count": boundary_count,
        "boundary_field_backend": boundary_field_backend,
        "parallel_field_backend": parallel_field_backend,
        "axis_mode": axis_mode,
        "configured_axis": axis,
        "parallel_axis": axis if axis_mode == "fixed" else None,
        "plane_value_mode": plane_value_mode,
        "plane_value": plane_value,
        "level_set_constraint": level_set_constraint,
        "soft_level_set_weight": soft_level_set_weight,
        "contour_max_level_fraction": contour_max_level_fraction,
        "min_polyline_points": min_polyline_points,
        "mild_smooth_all": mild_smooth_all,
        "mild_smooth_iterations": mild_smooth_iterations,
        "mild_smooth_alpha": mild_smooth_alpha,
        "first_layers_lowest_y_start_count": first_layers_lowest_y_start_count,
        "boundary_start_offset_count": boundary_start_offset_count,
        "parallel_direction_mode": parallel_direction_mode,
        "model_lift": model_lift,
        "model_lift_axis": model_lift_axis,
        "use_vectorized_extractor": use_vectorized_extractor,
        "normal_mode": "computed_from_mesh_after_hybrid_chaining",
        "width_profile_mode": "per_point_sidecar",
        "path_role_tag_convention": "layer2gcode.path-role-tags.v1",
        "path_role_tag_format": "half_open_index_blocks",
        "layers": layer_outputs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate hybrid connected path TXT files for all layered OBJ meshes in a folder.",
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing layer meshes named like <main>_<sub>.obj.")
    parser.add_argument("output_dir", type=Path, help="Folder where hybrid path TXT files will be written.")
    parser.add_argument("--spacing", type=float, required=True, help="Shared contour spacing for boundary and parallel families.")
    parser.add_argument("--parallel-spacing", type=float, default=None, help="Optional spacing between interior parallel/zig-zag lines. Defaults to --spacing.")
    parser.add_argument("--resample-spacing", type=float, default=None, help="Optional resampling spacing used before greedy chaining. Defaults to --spacing.")
    parser.add_argument("--parallel-resample-spacing", type=float, default=None, help="Optional resampling spacing for interior parallel paths. Defaults to --resample-spacing.")
    parser.add_argument("--sharp-angle-deg", type=float, default=35.0, help="Angle threshold in degrees for preserving sharp points during hybrid resampling.")
    parser.add_argument("--boundary-count", type=int, required=True, help="Number of boundary-family contours, counting the exact boundary as contour 0.")
    parser.add_argument("--boundary-field-backend", choices=("native", "potpourri3d_heat", "potpourri_fast_marching_unsigned"), default="native", help="Backend used for the hybrid boundary-distance field.")
    parser.add_argument("--parallel-field-backend", choices=("signed_heat", "fast_marching_unsigned", "vector_heat_stripe"), default="signed_heat", help="Backend used for the hybrid interior parallel field.")
    parser.add_argument("--axis-mode", choices=("fixed", "auto_tangent_alternating"), default="fixed", help="How to choose the parallel source-plane axis. 'fixed' uses --axis, 'auto_tangent_alternating' drops the dominant normal axis and alternates the remaining two by layer main index parity.")
    parser.add_argument("--axis", choices=("x", "y", "z"), default="x", help="Axis normal for the midpoint parallel source plane when --axis-mode=fixed.")
    parser.add_argument("--plane-value", type=float, default=None, help="Optional plane value override. Default uses midpoint extent.")
    parser.add_argument("--level-set-constraint", choices=("ZeroSet", "None", "Multiple"), default="None", help="Level-set constraint passed to MeshSignedHeatSolver.")
    parser.add_argument("--soft-level-set-weight", type=float, default=-1.0, help="Soft level-set weight passed to MeshSignedHeatSolver.")
    parser.add_argument("--contour-max-level-fraction", type=float, default=1.0, help="Optional fraction of field range used to trim hybrid contour extraction.")
    parser.add_argument("--min-polyline-points", type=int, default=2, help="Discard kept parallel contour polylines with fewer than this many points before chaining.")
    parser.add_argument("--mild-smooth-all", action="store_true", help="Apply a mild Laplacian smoothing pass to hybrid contour families before resampling/chaining.")
    parser.add_argument("--mild-smooth-iterations", type=int, default=8, help="Number of iterations for the optional mild hybrid contour smoothing pass.")
    parser.add_argument("--mild-smooth-alpha", type=float, default=0.2, help="Per-iteration Laplacian smoothing strength for the optional mild hybrid contour smoothing pass.")
    parser.add_argument("--first-layers-lowest-y-start-count", type=int, default=0, help="Number of first main-index layers whose hybrid boundary path should start at the lowest-Y outer point.")
    parser.add_argument("--boundary-start-offset-count", type=int, default=0, help="Number of boundary contour offsets to skip before chaining the hybrid boundary family. 1 skips the exact outer boundary.")
    parser.add_argument("--parallel-direction-mode", choices=("connect_nearest", "one_direction"), default="connect_nearest", help="How to orient interior parallel/hatching paths.")
    parser.add_argument("--use-vectorized-extractor", action="store_true", help="Use the vectorized parallel-segment extractor from test_hybrid_contours_vectorized.py.")
    parser.add_argument("--layer-workers", type=int, default=1, help="Number of worker processes used to generate separate layers in parallel.")
    parser.add_argument("--clean-output", action="store_true", help="Remove existing files in the output directory before writing.")
    parser.add_argument("--model-lift", type=float, default=0.0, help="Optional uniform lift applied to all mesh/path coordinates before toolpath generation.")
    parser.add_argument("--model-lift-axis", choices=("x", "y", "z"), default="y", help="Axis along which --model-lift is applied. Default: y.")
    return parser


def process_hybrid_layer(
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
    tolerance: float = 1e-8,
) -> tuple[Path, dict[str, object]]:
    del planar_infill  # Reserved for the projected planar-infill test backend.
    if use_vectorized_extractor:
        apply_vectorized_extractor_patch()
    V, F = load_obj_mesh(str(mesh_path))
    if abs(float(model_lift)) > 0.0:
        V = apply_axis_lift(V, axis=model_lift_axis, amount=float(model_lift))
    main_str, _ = mesh_path.stem.split("_", maxsplit=1)
    result = generate_hybrid_toolpath(
        V,
        F,
        spacing=spacing,
        parallel_spacing=parallel_spacing,
        resample_spacing=resample_spacing,
        parallel_resample_spacing=parallel_resample_spacing,
        boundary_count=boundary_count,
        sharp_angle_deg=sharp_angle_deg,
        boundary_field_backend=boundary_field_backend,
        parallel_field_backend=parallel_field_backend,
        axis_mode=axis_mode,
        axis=axis,
        layer_main_index=int(main_str),
        plane_value=plane_value,
        level_set_constraint=level_set_constraint,
        soft_level_set_weight=soft_level_set_weight,
        contour_max_level_fraction=contour_max_level_fraction,
        min_polyline_points=min_polyline_points,
        mild_smooth_all=mild_smooth_all,
        mild_smooth_iterations=mild_smooth_iterations,
        mild_smooth_alpha=mild_smooth_alpha,
        start_selection_mode=start_selection_mode,
        boundary_start_offset_count=boundary_start_offset_count,
        parallel_direction_mode=parallel_direction_mode,
        disable_parallel=disable_parallel,
        top_layer_width_check_enabled=top_layer_width_check_enabled,
        top_layer_width_check_max_factor=top_layer_width_check_max_factor,
        is_top_layer=is_top_layer,
        tolerance=tolerance,
    )
    path_points = np.asarray(result["path_points"], dtype=np.float64)
    path_widths = np.asarray(result.get("path_widths", np.empty((0,), dtype=np.float64)), dtype=np.float64).reshape(-1)
    path_tags = np.asarray(result.get("path_tags", np.empty((0,), dtype=str)), dtype=str).reshape(-1)
    path_normals = compute_path_normals(V, F, path_points) if path_points.shape[0] > 0 else np.empty((0, 3), dtype=np.float64)
    output_path = output_dir / f"{mesh_path.stem}.txt"
    save_oriented_path(str(output_path), path_points, path_normals)
    width_profile_path = output_dir / f"{mesh_path.stem}.widths.txt"
    path_tag_path = output_dir / f"{mesh_path.stem}.tags.txt"
    if path_widths.size > 0:
        if path_widths.shape[0] != path_points.shape[0]:
            raise ValueError("Hybrid width profile must match the number of path points.")
        np.savetxt(width_profile_path, path_widths, fmt="%.10f")
    if path_tags.size > 0:
        if path_tags.shape[0] != path_points.shape[0]:
            raise ValueError("Hybrid path tags must match the number of path points.")
        blocks = encode_path_tag_blocks(path_tags)
        path_tag_path.write_text(
            "".join(f"{start} {end} {tag}\n" for start, end, tag in blocks),
            encoding="utf-8",
        )
    metadata = {
        "boundary_cutoff": float(result["boundary_cutoff"]),
        "boundary_field_backend": str(result["boundary_field_backend"]),
        "parallel_field_backend": str(result["parallel_field_backend"]),
        "axis_mode": str(result["axis_mode"]),
        "parallel_axis": str(result["parallel_axis"]),
        "dominant_normal_axis": result["dominant_normal_axis"],
        "tangent_axes": result["tangent_axes"],
        "plane_value": float(result["plane_value"]),
        "parallel_spacing": float(result["parallel_spacing"]),
        "resample_spacing": float(result["resample_spacing"]),
        "parallel_resample_spacing": float(result["parallel_resample_spacing"]),
        "boundary_start_offset_count": int(result["boundary_start_offset_count"]),
        "parallel_direction_mode": str(result["parallel_direction_mode"]),
        "parallel_disabled": bool(result["parallel_disabled"]),
        "top_layer_width_check_enabled": bool(result.get("top_layer_width_check_enabled", False)),
        "is_top_layer": bool(is_top_layer),
        "top_layer_width_check_max_factor": float(result.get("top_layer_width_check_max_factor", 2.0)),
        "use_vectorized_extractor": bool(use_vectorized_extractor),
        "width_profile_file": width_profile_path.name if path_widths.size > 0 else None,
        "path_tag_file": path_tag_path.name if path_tags.size > 0 else None,
        "path_tag_format": "half_open_index_blocks",
        "path_tag_block_count": len(encode_path_tag_blocks(path_tags)),
        "path_tag_values": sorted({str(tag) for tag in path_tags.tolist()}),
        "width_profile_point_count": int(path_widths.shape[0]),
        "width_profile_min": float(np.min(path_widths)) if path_widths.size > 0 else None,
        "width_profile_max": float(np.max(path_widths)) if path_widths.size > 0 else None,
        "sharp_angle_deg": float(result["sharp_angle_deg"]),
        "source_curve_count": len(result["source_curves"]),
        "source_segment_count": len(result["source_segments"]),
        "boundary_loop_count": len(result["boundary_loops"]),
        "boundary_level_count": len(result["boundary_levels"]),
        "parallel_level_count": int(np.asarray(result["parallel_levels"]).size),
        "parallel_polyline_count": len(result["parallel_polylines"]),
        "chained_boundary_part_count": len(result["chained_boundary_parts"]),
        "chained_parallel_part_count": len(result["chained_parallel_parts"]),
        "start_selection_mode": start_selection_mode,
        "path_point_count": int(path_points.shape[0]),
    }
    metadata.update({key: float(value) for key, value in result.get("width_diagnostics", {}).items()})
    metadata.update({key: float(value) for key, value in result.get("wall_gap_diagnostics", {}).items()})
    metadata.update({key: float(value) for key, value in result.get("boundary_gap_width_diagnostics", {}).items()})
    metadata.update({key: float(value) for key, value in result.get("top_layer_width_diagnostics", {}).items()})
    return output_path, metadata


def main() -> None:
    args = build_parser().parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        clear_directory(output_dir)
        print(f"Cleared output directory: {output_dir}")

    layer_meshes = iter_layer_meshes(input_dir)
    if not layer_meshes:
        raise FileNotFoundError(f"No layer meshes named like <main>_<sub>.obj were found in {input_dir}")
    if args.layer_workers <= 0:
        raise ValueError("layer_workers must be positive.")
    if args.first_layers_lowest_y_start_count < 0:
        raise ValueError("first_layers_lowest_y_start_count must be non-negative.")
    if args.boundary_start_offset_count < 0:
        raise ValueError("boundary_start_offset_count must be non-negative.")

    print(f"Generating hybrid connected paths for {len(layer_meshes)} layers into {output_dir}")
    print(f"Hybrid layer workers: {args.layer_workers}")

    layer_outputs: list[dict[str, object]] = []
    plane_value_mode = "fixed" if args.plane_value is not None else "midpoint_extent"
    main_ranks = {main: rank for rank, main in enumerate(sorted({layer_sort_key(path)[0] for path in layer_meshes}))}
    if args.layer_workers == 1:
        for layer_mesh in layer_meshes:
            main_index, _ = layer_sort_key(layer_mesh)
            start_selection_mode = (
                "lowest_y"
                if main_ranks[main_index] < args.first_layers_lowest_y_start_count
                else "first"
            )
            output_path, layer_meta = process_hybrid_layer(
                layer_mesh,
                output_dir,
                spacing=args.spacing,
                parallel_spacing=args.parallel_spacing,
                resample_spacing=args.resample_spacing,
                parallel_resample_spacing=args.parallel_resample_spacing,
                boundary_count=args.boundary_count,
                sharp_angle_deg=args.sharp_angle_deg,
                boundary_field_backend=args.boundary_field_backend,
                parallel_field_backend=args.parallel_field_backend,
                axis_mode=args.axis_mode,
                axis=args.axis,
                plane_value=args.plane_value,
                level_set_constraint=args.level_set_constraint,
                soft_level_set_weight=args.soft_level_set_weight,
                contour_max_level_fraction=args.contour_max_level_fraction,
                min_polyline_points=args.min_polyline_points,
                mild_smooth_all=args.mild_smooth_all,
                mild_smooth_iterations=args.mild_smooth_iterations,
                mild_smooth_alpha=args.mild_smooth_alpha,
                start_selection_mode=start_selection_mode,
                boundary_start_offset_count=args.boundary_start_offset_count,
                parallel_direction_mode=args.parallel_direction_mode,
                model_lift=args.model_lift,
                model_lift_axis=args.model_lift_axis,
                use_vectorized_extractor=bool(args.use_vectorized_extractor),
            )
            main_str, sub_str = layer_mesh.stem.split("_", maxsplit=1)
            layer_outputs.append(
                {
                    "layer_main": int(main_str),
                    "layer_sub": int(sub_str),
                    "source_mesh": layer_mesh.name,
                    "path_file": output_path.name,
                    **layer_meta,
                }
            )
            print(f"Saved hybrid path: {output_path.name}")
    else:
        with ProcessPoolExecutor(max_workers=args.layer_workers, mp_context=get_context('spawn')) as executor:
            futures = [
                executor.submit(
                    process_hybrid_layer,
                    layer_mesh,
                    output_dir,
                    spacing=args.spacing,
                    parallel_spacing=args.parallel_spacing,
                    resample_spacing=args.resample_spacing,
                    parallel_resample_spacing=args.parallel_resample_spacing,
                    boundary_count=args.boundary_count,
                    sharp_angle_deg=args.sharp_angle_deg,
                    boundary_field_backend=args.boundary_field_backend,
                    parallel_field_backend=args.parallel_field_backend,
                    axis_mode=args.axis_mode,
                    axis=args.axis,
                    plane_value=args.plane_value,
                    level_set_constraint=args.level_set_constraint,
                    soft_level_set_weight=args.soft_level_set_weight,
                    contour_max_level_fraction=args.contour_max_level_fraction,
                    min_polyline_points=args.min_polyline_points,
                    mild_smooth_all=args.mild_smooth_all,
                    mild_smooth_iterations=args.mild_smooth_iterations,
                    mild_smooth_alpha=args.mild_smooth_alpha,
                    start_selection_mode=(
                        "lowest_y"
                        if main_ranks[layer_sort_key(layer_mesh)[0]] < args.first_layers_lowest_y_start_count
                        else "first"
                    ),
                    boundary_start_offset_count=args.boundary_start_offset_count,
                    parallel_direction_mode=args.parallel_direction_mode,
                    model_lift=args.model_lift,
                    model_lift_axis=args.model_lift_axis,
                    use_vectorized_extractor=bool(args.use_vectorized_extractor),
                )
                for layer_mesh in layer_meshes
            ]
            results = [future.result() for future in futures]
        results.sort(key=lambda item: layer_sort_key(item[0].with_suffix(".obj")))
        for output_path, layer_meta in results:
            main_str, sub_str = output_path.stem.split("_", maxsplit=1)
            layer_outputs.append(
                {
                    "layer_main": int(main_str),
                    "layer_sub": int(sub_str),
                    "source_mesh": f"{output_path.stem}.obj",
                    "path_file": output_path.name,
                    **layer_meta,
                }
            )
            print(f"Saved hybrid path: {output_path.name}")

    batch_metadata = _build_batch_metadata(
        input_dir=input_dir,
        output_dir=output_dir,
        spacing=args.spacing,
        parallel_spacing=float(args.parallel_spacing) if args.parallel_spacing is not None else float(args.spacing),
        resample_spacing=float(args.resample_spacing) if args.resample_spacing is not None else float(args.spacing),
        parallel_resample_spacing=(
            float(args.parallel_resample_spacing)
            if args.parallel_resample_spacing is not None
            else float(args.resample_spacing) if args.resample_spacing is not None else float(args.spacing)
        ),
        boundary_count=args.boundary_count,
        sharp_angle_deg=args.sharp_angle_deg,
        boundary_field_backend=args.boundary_field_backend,
        parallel_field_backend=args.parallel_field_backend,
        axis_mode=args.axis_mode,
        axis=args.axis,
        plane_value_mode=plane_value_mode,
        plane_value=args.plane_value,
        level_set_constraint=args.level_set_constraint,
        soft_level_set_weight=args.soft_level_set_weight,
        contour_max_level_fraction=args.contour_max_level_fraction,
        min_polyline_points=args.min_polyline_points,
        mild_smooth_all=args.mild_smooth_all,
        mild_smooth_iterations=args.mild_smooth_iterations,
        mild_smooth_alpha=args.mild_smooth_alpha,
        first_layers_lowest_y_start_count=args.first_layers_lowest_y_start_count,
        boundary_start_offset_count=args.boundary_start_offset_count,
        parallel_direction_mode=args.parallel_direction_mode,
        model_lift=float(args.model_lift),
        model_lift_axis=str(args.model_lift_axis),
        use_vectorized_extractor=bool(args.use_vectorized_extractor),
        layer_outputs=layer_outputs,
    )
    batch_metadata_path = output_dir / "batch_paths.meta.json"
    batch_metadata_path.write_text(json.dumps(batch_metadata, indent=2), encoding="utf-8")
    print(f"Saved path metadata: {batch_metadata_path.name}")


if __name__ == "__main__":
    main()
