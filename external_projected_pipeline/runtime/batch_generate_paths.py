"""
Batch-generate connected toolpaths from layered OBJ meshes.

Input meshes are expected to follow the naming convention:
    <main index>_<sub index>.obj

For each mesh, the script:
- loads the mesh
- solves the boundary-sourced heat distance field
- extracts equally spaced contours
- optionally resamples and filters them
- builds one connected path
- saves rows of x y z nx ny nz to the output folder as:
    <main index>_<sub index>.txt
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

from contour_extraction import build_connected_closed_loop_path_with_normals
from contour_extraction import filter_oriented_contour_levels_by_point_count
from contour_extraction import generate_boundary_distance_contours_with_normals
from contour_extraction import generate_boundary_loops_with_normals
from contour_extraction import resample_oriented_contour_levels
from contour_extraction import resample_oriented_polyline_equal_spacing
from contour_extraction import smooth_oriented_contour_levels_in_fraction_window
from heat_method_core import HeatMethodParams
from extrusion_height import apply_axis_lift
from mesh_distance_workflow import load_obj_mesh
from toolpath_export import save_oriented_path


LAYER_PATTERN = re.compile(r"^(?P<main>\d+)_(?P<sub>\d+)\.obj$", re.IGNORECASE)


def _build_batch_metadata(
    *,
    input_dir: Path,
    output_dir: Path,
    spacing: float,
    heat_params: HeatMethodParams,
    prefer_libigl: bool,
    heat_backend: str,
    max_level: float | None,
    max_level_fraction: float | None,
    resample_spacing: float | None,
    sharp_angle_deg: float,
    min_contour_points: int,
    mild_smooth_all: bool,
    mild_smooth_iterations: int,
    mild_smooth_alpha: float,
    smooth_level_frac_from: float | None,
    smooth_level_frac_to: float | None,
    smooth_iterations: int,
    smooth_alpha: float,
    outer_start_mode: str,
    first_layers_lowest_y_start_count: int,
    allow_branching_boundary_fallback: bool,
    model_lift: float,
    model_lift_axis: str,
    layer_outputs: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "batch_kind": "boundary_contours_connected_paths",
        "output_format": "x y z nx ny nz",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "path_kind": "boundary_contours_connected",
        "contour_spacing": spacing,
        "width": spacing,
        "outer_boundary_used": outer_start_mode == "boundary",
        "outer_start_mode": outer_start_mode,
        "closed_loop_fast": True,
        "normal_mode": "carried_from_contours",
        "prefer_libigl": prefer_libigl,
        "heat_backend": heat_backend,
        "timestep_scale": heat_params.timestep_scale,
        "use_direct_solver": heat_params.use_direct_solver,
        "enforce_dirichlet_on_sources": heat_params.enforce_dirichlet_on_sources,
        "max_level": max_level,
        "max_level_fraction": max_level_fraction,
        "resample_spacing": resample_spacing,
        "sharp_angle_deg": sharp_angle_deg,
        "min_contour_points": min_contour_points,
        "mild_smooth_all": mild_smooth_all,
        "mild_smooth_iterations": mild_smooth_iterations,
        "mild_smooth_alpha": mild_smooth_alpha,
        "smooth_level_frac_from": smooth_level_frac_from,
        "smooth_level_frac_to": smooth_level_frac_to,
        "smooth_iterations": smooth_iterations,
        "smooth_alpha": smooth_alpha,
        "allow_branching_boundary_fallback": allow_branching_boundary_fallback,
        "first_layers_lowest_y_start_count": first_layers_lowest_y_start_count,
        "model_lift": model_lift,
        "model_lift_axis": model_lift_axis,
        "layers": layer_outputs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate connected path TXT files for all layered OBJ meshes in a folder.",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Folder containing layer meshes named like <main>_<sub>.obj.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Folder where connected path TXT files will be written.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        required=True,
        help="Contour spacing in scalar-field units.",
    )
    parser.add_argument(
        "--timestep-scale",
        type=float,
        default=1.0,
        help="Scale factor for the default heat timestep h^2.",
    )
    parser.add_argument(
        "--iterative",
        action="store_true",
        help="Use conjugate gradient instead of direct sparse solves.",
    )
    parser.add_argument(
        "--no-libigl",
        action="store_true",
        help="Force the manual SciPy operator path even if libigl is installed.",
    )
    parser.add_argument(
        "--heat-backend",
        choices=("native", "potpourri3d"),
        default="native",
        help="Backend used to compute the boundary distance field.",
    )
    parser.add_argument(
        "--max-level",
        type=float,
        default=None,
        help="Optional upper bound on contour levels.",
    )
    parser.add_argument(
        "--max-level-fraction",
        type=float,
        default=None,
        help="Optional fraction of the maximum scalar value used to trim inner contours.",
    )
    parser.add_argument(
        "--resample-spacing",
        type=float,
        default=None,
        help="Optional arc-length spacing used to resample contours before chaining.",
    )
    parser.add_argument(
        "--sharp-angle-deg",
        type=float,
        default=35.0,
        help="Angle threshold in degrees for preserving sharp points during resampling.",
    )
    parser.add_argument(
        "--min-contour-points",
        type=int,
        default=4,
        help="Remove any contour polyline with fewer than this many points.",
    )
    parser.add_argument(
        "--mild-smooth-all",
        action="store_true",
        help="Apply a mild Laplacian smoothing pass to all contours before any inner-level smoothing window.",
    )
    parser.add_argument(
        "--mild-smooth-iterations",
        type=int,
        default=8,
        help="Number of iterations for the optional mild all-contour smoothing pass.",
    )
    parser.add_argument(
        "--mild-smooth-alpha",
        type=float,
        default=0.2,
        help="Per-iteration Laplacian smoothing strength for the optional mild all-contour smoothing pass.",
    )
    parser.add_argument(
        "--smooth-level-frac-from",
        type=float,
        default=None,
        help="Optional lower scalar fraction for applying strong contour smoothing before chaining.",
    )
    parser.add_argument(
        "--smooth-level-frac-to",
        type=float,
        default=None,
        help="Optional upper scalar fraction for applying strong contour smoothing before chaining.",
    )
    parser.add_argument(
        "--smooth-iterations",
        type=int,
        default=30,
        help="Number of Laplacian smoothing iterations when smoothing is enabled.",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.75,
        help="Laplacian smoothing strength per iteration when smoothing is enabled.",
    )
    parser.add_argument(
        "--outer-start-mode",
        type=str,
        default="boundary",
        choices=("boundary", "skip_boundary"),
        help="Outer path policy. 'boundary' uses the exact mesh boundary; 'skip_boundary' omits it and chains generated contours only.",
    )
    parser.add_argument(
        "--first-layers-lowest-y-start-count",
        type=int,
        default=0,
        help="Number of first main-index layers whose connected contour path should start at the lowest-Y outer point.",
    )
    parser.add_argument(
        "--layer-workers",
        type=int,
        default=1,
        help="Number of worker processes used to generate separate layers in parallel. Default keeps the original serial behavior.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing files in the output directory before writing.",
    )
    parser.add_argument(
        "--model-lift",
        type=float,
        default=0.0,
        help="Optional uniform lift applied to all mesh/path coordinates before toolpath generation.",
    )
    parser.add_argument(
        "--model-lift-axis",
        choices=("x", "y", "z"),
        default="y",
        help="Axis along which --model-lift is applied. Default: y.",
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


def layer_sort_key(path: Path) -> tuple[int, int]:
    match = LAYER_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected layer filename: {path.name}")
    return int(match.group("main")), int(match.group("sub"))


def iter_layer_meshes(input_dir: Path) -> list[Path]:
    meshes = [path for path in input_dir.iterdir() if path.is_file() and LAYER_PATTERN.match(path.name)]
    return sorted(meshes, key=layer_sort_key)


def process_layer(
    mesh_path: Path,
    output_dir: Path,
    *,
    spacing: float,
    heat_params: HeatMethodParams,
    prefer_libigl: bool,
    heat_backend: str,
    max_level: float | None,
    max_level_fraction: float | None,
    resample_spacing: float | None,
    sharp_angle_deg: float,
    min_contour_points: int,
    mild_smooth_all: bool,
    mild_smooth_iterations: int,
    mild_smooth_alpha: float,
    smooth_level_frac_from: float | None,
    smooth_level_frac_to: float | None,
    smooth_iterations: int,
    smooth_alpha: float,
    outer_start_mode: str = "boundary",
    start_selection_mode: str = "lexicographic",
    allow_branching_boundary_fallback: bool = True,
    model_lift: float = 0.0,
    model_lift_axis: str = "y",
) -> Path:
    if outer_start_mode not in {"boundary", "skip_boundary"}:
        raise ValueError("outer_start_mode must be either 'boundary' or 'skip_boundary'.")
    V, F = load_obj_mesh(str(mesh_path))
    if abs(float(model_lift)) > 0.0:
        V = apply_axis_lift(V, axis=model_lift_axis, amount=float(model_lift))
    scalar_field, oriented_contour_levels = generate_boundary_distance_contours_with_normals(
        V,
        F,
        spacing,
        heat_params=heat_params,
        prefer_libigl=prefer_libigl,
        heat_backend=heat_backend,
        max_level=max_level,
        max_level_fraction=max_level_fraction,
    )
    oriented_boundary_loops = []
    if outer_start_mode == "boundary":
        try:
            oriented_boundary_loops = generate_boundary_loops_with_normals(V, F)
        except ValueError as exc:
            if allow_branching_boundary_fallback and "Boundary graph is not chain/loop-like" in str(exc):
                print(
                    f"Warning: {mesh_path.name} has a branching boundary graph; "
                    "falling back to contour-only path chaining."
                )
            else:
                raise

    if resample_spacing is not None:
        oriented_contour_levels = resample_oriented_contour_levels(
            oriented_contour_levels,
            resample_spacing,
            angle_threshold_degrees=sharp_angle_deg,
        )
        oriented_boundary_loops = [
            resample_oriented_polyline_equal_spacing(
                polyline,
                resample_spacing,
                angle_threshold_degrees=sharp_angle_deg,
            )
            for polyline in oriented_boundary_loops
        ]

    scalar_max = float(np.max(scalar_field))
    if mild_smooth_all:
        oriented_contour_levels = smooth_oriented_contour_levels_in_fraction_window(
            oriented_contour_levels,
            scalar_max,
            fraction_from=0.0,
            fraction_to=1.0,
            iterations=mild_smooth_iterations,
            alpha=mild_smooth_alpha,
        )

    if smooth_level_frac_from is not None or smooth_level_frac_to is not None:
        if smooth_level_frac_from is None or smooth_level_frac_to is None:
            raise ValueError("Both smoothing fraction bounds must be provided together.")
        oriented_contour_levels = smooth_oriented_contour_levels_in_fraction_window(
            oriented_contour_levels,
            scalar_max,
            fraction_from=smooth_level_frac_from,
            fraction_to=smooth_level_frac_to,
            iterations=smooth_iterations,
            alpha=smooth_alpha,
        )

    oriented_contour_levels = filter_oriented_contour_levels_by_point_count(
        oriented_contour_levels,
        min_points=min_contour_points,
    )
    oriented_boundary_loops = [
        polyline
        for polyline in oriented_boundary_loops
        if polyline.points.shape[0] >= min_contour_points
    ]

    connected_path = build_connected_closed_loop_path_with_normals(
        oriented_contour_levels,
        boundary_polylines=oriented_boundary_loops,
        start_selection_mode=start_selection_mode,
    )

    output_path = output_dir / f"{mesh_path.stem}.txt"
    save_oriented_path(
        str(output_path),
        connected_path.points,
        connected_path.normals,
    )
    return output_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    if args.layer_workers <= 0:
        raise ValueError("layer_workers must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        clear_directory(output_dir)
        print(f"Cleared output directory: {output_dir}")

    layer_meshes = iter_layer_meshes(input_dir)
    if not layer_meshes:
        raise FileNotFoundError(
            f"No layer meshes named like <main>_<sub>.obj were found in {input_dir}",
        )

    heat_params = HeatMethodParams(
        timestep_scale=args.timestep_scale,
        use_direct_solver=not args.iterative,
    )

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Layer count: {len(layer_meshes)}")
    print(f"Layer workers: {args.layer_workers}")

    layer_outputs: list[dict[str, object]] = []
    if args.first_layers_lowest_y_start_count < 0:
        raise ValueError("first_layers_lowest_y_start_count must be non-negative.")
    main_ranks = {main: rank for rank, main in enumerate(sorted({layer_sort_key(path)[0] for path in layer_meshes}))}
    if args.layer_workers == 1:
        for layer_mesh in layer_meshes:
            main_index, _ = layer_sort_key(layer_mesh)
            start_selection_mode = (
                "lowest_y"
                if main_ranks[main_index] < args.first_layers_lowest_y_start_count
                else "lexicographic"
            )
            output_path = process_layer(
                layer_mesh,
                output_dir,
                spacing=args.spacing,
                heat_params=heat_params,
                prefer_libigl=not args.no_libigl,
                heat_backend=args.heat_backend,
                max_level=args.max_level,
                max_level_fraction=args.max_level_fraction,
                resample_spacing=args.resample_spacing,
                sharp_angle_deg=args.sharp_angle_deg,
                min_contour_points=args.min_contour_points,
                mild_smooth_all=args.mild_smooth_all,
                mild_smooth_iterations=args.mild_smooth_iterations,
                mild_smooth_alpha=args.mild_smooth_alpha,
                smooth_level_frac_from=args.smooth_level_frac_from,
                smooth_level_frac_to=args.smooth_level_frac_to,
                smooth_iterations=args.smooth_iterations,
                smooth_alpha=args.smooth_alpha,
                outer_start_mode=args.outer_start_mode,
                start_selection_mode=start_selection_mode,
                allow_branching_boundary_fallback=True,
                model_lift=args.model_lift,
                model_lift_axis=args.model_lift_axis,
            )
            main_str, sub_str = layer_mesh.stem.split("_", maxsplit=1)
            layer_outputs.append(
                {
                    "layer_main": int(main_str),
                    "layer_sub": int(sub_str),
                    "source_mesh": layer_mesh.name,
                    "path_file": output_path.name,
                }
            )
            print(f"Saved: {output_path.name}")
    else:
        with ProcessPoolExecutor(max_workers=args.layer_workers, mp_context=get_context('spawn')) as executor:
            futures = [
                executor.submit(
                    process_layer,
                    layer_mesh,
                    output_dir,
                    spacing=args.spacing,
                    heat_params=heat_params,
                    prefer_libigl=not args.no_libigl,
                    heat_backend=args.heat_backend,
                    max_level=args.max_level,
                    max_level_fraction=args.max_level_fraction,
                    resample_spacing=args.resample_spacing,
                    sharp_angle_deg=args.sharp_angle_deg,
                    min_contour_points=args.min_contour_points,
                    mild_smooth_all=args.mild_smooth_all,
                    mild_smooth_iterations=args.mild_smooth_iterations,
                    mild_smooth_alpha=args.mild_smooth_alpha,
                    smooth_level_frac_from=args.smooth_level_frac_from,
                    smooth_level_frac_to=args.smooth_level_frac_to,
                    smooth_iterations=args.smooth_iterations,
                    smooth_alpha=args.smooth_alpha,
                    outer_start_mode=args.outer_start_mode,
                    start_selection_mode=(
                        "lowest_y"
                        if main_ranks[layer_sort_key(layer_mesh)[0]] < args.first_layers_lowest_y_start_count
                        else "lexicographic"
                    ),
                    allow_branching_boundary_fallback=True,
                    model_lift=args.model_lift,
                    model_lift_axis=args.model_lift_axis,
                )
                for layer_mesh in layer_meshes
            ]
            output_paths = [future.result() for future in futures]
        output_paths.sort(key=lambda path: layer_sort_key(path.with_suffix(".obj")))
        for output_path in output_paths:
            main_str, sub_str = output_path.stem.split("_", maxsplit=1)
            layer_outputs.append(
                {
                    "layer_main": int(main_str),
                    "layer_sub": int(sub_str),
                    "source_mesh": f"{output_path.stem}.obj",
                    "path_file": output_path.name,
                }
            )
            print(f"Saved: {output_path.name}")

    batch_metadata = _build_batch_metadata(
        input_dir=input_dir,
        output_dir=output_dir,
        spacing=args.spacing,
        heat_params=heat_params,
        prefer_libigl=not args.no_libigl,
        heat_backend=args.heat_backend,
        max_level=args.max_level,
        max_level_fraction=args.max_level_fraction,
        resample_spacing=args.resample_spacing,
        sharp_angle_deg=args.sharp_angle_deg,
        min_contour_points=args.min_contour_points,
        mild_smooth_all=args.mild_smooth_all,
        mild_smooth_iterations=args.mild_smooth_iterations,
        mild_smooth_alpha=args.mild_smooth_alpha,
        smooth_level_frac_from=args.smooth_level_frac_from,
        smooth_level_frac_to=args.smooth_level_frac_to,
        smooth_iterations=args.smooth_iterations,
        smooth_alpha=args.smooth_alpha,
        outer_start_mode=args.outer_start_mode,
        first_layers_lowest_y_start_count=args.first_layers_lowest_y_start_count,
        allow_branching_boundary_fallback=True,
        model_lift=float(args.model_lift),
        model_lift_axis=str(args.model_lift_axis),
        layer_outputs=layer_outputs,
    )
    batch_metadata_path = output_dir / "batch_paths.meta.json"
    batch_metadata_path.write_text(json.dumps(batch_metadata, indent=2), encoding="utf-8")
    print(f"Saved batch metadata: {batch_metadata_path.name}")


if __name__ == "__main__":
    main()
