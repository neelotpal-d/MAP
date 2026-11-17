"""
Visualize layered meshes and generated path files together.

Expected naming convention:
    meshes: <main index>_<sub index>.obj
    paths:  <main index>_<sub index>.txt
            <main index>_<sub index>_3axis.txt
            <main index>_<sub index>_5axis.txt

Each matching layer is shown with a distinct color, and its path is rendered
with a brighter companion color.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import random
import re
from pathlib import Path

import numpy as np

from mesh_distance_workflow import load_obj_mesh
from mesh_distance_workflow import to_pyvista_polydata


MESH_PATTERN = re.compile(r"^(?P<main>\d+)_(?P<sub>\d+)\.obj$", re.IGNORECASE)
PATH_PATTERN = re.compile(r"^(?P<main>\d+)_(?P<sub>\d+)(?:_(?P<mode>3axis|5axis))?\.txt$", re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize layer meshes and matching path files from two folders.",
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
        "--show-normals",
        action="store_true",
        help="Display path normals as glyphs if the TXT file has nx ny nz columns.",
    )
    parser.add_argument(
        "--normal-scale",
        type=float,
        default=0.25,
        help="Scale factor for displayed normal glyphs.",
    )
    parser.add_argument(
        "--mesh-opacity",
        type=float,
        default=0.45,
        help="Opacity used for layer meshes.",
    )
    parser.add_argument(
        "--one-by-one",
        action="store_true",
        help="Show one layer/path pair at a time in separate viewer windows.",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=4.0,
        help="Line width used for the displayed path polyline.",
    )
    parser.add_argument(
        "--show-points",
        action="store_true",
        help="Display the sampled path points as rendered points.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=8.0,
        help="Point size used when --show-points is enabled.",
    )
    parser.add_argument(
        "--randomize-consecutive-colors",
        action="store_true",
        help="Assign consecutive layers shuffled random-looking colors for better visual separation.",
    )
    parser.add_argument(
        "--color-seed",
        type=int,
        default=12345,
        help="Seed used when --randomize-consecutive-colors is enabled.",
    )
    parser.add_argument(
        "--show-width-highlights",
        action="store_true",
        help="Overlay path segments whose width profile is above the nominal baseline.",
    )
    parser.add_argument(
        "--width-highlight-tolerance",
        type=float,
        default=1e-6,
        help="Minimum amount above the nominal baseline required to highlight a segment.",
    )
    parser.add_argument(
        "--width-highlight-color",
        type=str,
        default="yellow",
        help="Color used for highlighted wider-extrusion path segments.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Optional single layer to display, written as <main>_<sub> (for example 120_0).",
    )
    parser.add_argument(
        "--show-nearest-parallel-distance",
        action="store_true",
        help="Highlight a recovered projected raster pass, its nearest different pass, and their 3D/XZ separation.",
    )
    parser.add_argument(
        "--color-nearest-parallel-distance",
        action="store_true",
        help="Color recovered projected raster passes by their nearest different-pass 3D distance.",
    )
    parser.add_argument(
        "--color-path-sample-spacing",
        action="store_true",
        help=(
            "Color each path point by its local 3D distance to adjacent sampled points. "
            "This reveals sparse/adaptively decimated regions without treating path breaks as segments."
        ),
    )
    parser.add_argument(
        "--distance-parallel-index",
        type=int,
        default=None,
        help="Recovered raster-pass index for --show-nearest-parallel-distance. Default selects the middle pass.",
    )
    parser.add_argument(
        "--parallel-run-tolerance",
        type=float,
        default=1e-6,
        help="XZ tolerance used to recognize straight projected raster segments.",
    )
    return parser


def mesh_layer_key(path: Path) -> tuple[int, int]:
    match = MESH_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected layer file name: {path.name}")
    return int(match.group("main")), int(match.group("sub"))


def path_layer_key(path: Path) -> tuple[int, int]:
    match = PATH_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected path file name: {path.name}")
    return int(match.group("main")), int(match.group("sub"))


def collect_named_files(folder: Path, expected_ext: str) -> dict[tuple[int, int], Path]:
    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Folder not found: {folder}")

    files: dict[tuple[int, int], Path] = {}
    for path in folder.iterdir():
        if not path.is_file():
            continue
        if expected_ext == "obj":
            if MESH_PATTERN.match(path.name) is None:
                continue
            files[mesh_layer_key(path)] = path
        elif expected_ext == "txt":
            if PATH_PATTERN.match(path.name) is None:
                continue
            files[path_layer_key(path)] = path
        else:
            raise ValueError(f"Unsupported extension type: {expected_ext}")
    return files


def generate_layer_colors(count: int) -> list[tuple[float, float, float]]:
    if count <= 0:
        return []
    colors: list[tuple[float, float, float]] = []
    for idx in range(count):
        hue = idx / max(count, 1)
        saturation = 0.65
        value = 0.82
        colors.append(colorsys.hsv_to_rgb(hue, saturation, value))
    return colors


def generate_randomized_layer_colors(
    count: int,
    *,
    seed: int,
) -> list[tuple[float, float, float]]:
    if count <= 0:
        return []
    hues = [idx / max(count, 1) for idx in range(count)]
    rng = random.Random(seed)
    rng.shuffle(hues)
    colors: list[tuple[float, float, float]] = []
    for idx, hue in enumerate(hues):
        saturation = 0.6 + 0.18 * ((idx % 3) / 2.0)
        value = 0.76 + 0.16 * ((idx % 4) / 3.0)
        colors.append(colorsys.hsv_to_rgb(hue, min(1.0, saturation), min(1.0, value)))
    return colors


def brighten_color(color: tuple[float, float, float], factor: float = 1.2) -> tuple[float, float, float]:
    return tuple(min(1.0, channel * factor) for channel in color)


def load_path_points(path_file: Path) -> tuple[np.ndarray, np.ndarray | None]:
    if path_file.stat().st_size == 0:
        print(f"Warning: skipping empty path file: {path_file}")
        return np.empty((0, 3), dtype=np.float64), None

    data = np.loadtxt(path_file, dtype=np.float64)
    if data.size == 0:
        print(f"Warning: skipping empty path file: {path_file}")
        return np.empty((0, 3), dtype=np.float64), None

    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 3:
        print(f"Warning: skipping path file with fewer than x y z columns: {path_file}")
        return np.empty((0, 3), dtype=np.float64), None

    points = data[:, 0:3]
    normals = data[:, 3:6] if data.shape[1] >= 6 else None
    return points, normals


def infer_width_profile_path(path_file: Path) -> Path:
    stem = path_file.stem
    if stem.endswith("_3axis"):
        stem = stem[:-6]
    elif stem.endswith("_5axis"):
        stem = stem[:-6]
    return path_file.with_name(f"{stem}.widths.txt")


def load_width_profile(path_file: Path, point_count: int) -> np.ndarray | None:
    width_profile_path = infer_width_profile_path(path_file)
    if not width_profile_path.exists() or width_profile_path.stat().st_size == 0:
        return None
    values = np.loadtxt(width_profile_path, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape[0] != point_count:
        print(
            f"Warning: width profile point count mismatch for {path_file.name}: "
            f"{values.shape[0]} vs {point_count}"
        )
        return None
    return values


def load_path_break_indices(path_file: Path, point_count: int) -> list[int]:
    stem = path_file.stem
    if stem.endswith("_3axis"):
        stem = stem[: -len("_3axis")]
    break_path = path_file.with_name(f"{stem}.breaks.txt")
    if not break_path.exists():
        return []
    values = sorted({int(line.strip()) for line in break_path.read_text(encoding="utf-8").splitlines() if line.strip()})
    return [index for index in values if 0 < index < point_count]


def infer_nominal_width_from_metadata(path_file: Path) -> float | None:
    key = path_layer_key(path_file)
    candidate_metadata_files = [
        path_file.parent / "batch_paths.meta.json",
        path_file.parent / "batch_3axis.meta.json",
        path_file.parent / "batch_5axis.meta.json",
    ]
    for metadata_path in candidate_metadata_files:
        if not metadata_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        layers = metadata.get("layers")
        if isinstance(layers, list):
            for layer in layers:
                if not isinstance(layer, dict):
                    continue
                if int(layer.get("layer_main", -1)) != key[0] or int(layer.get("layer_sub", -1)) != key[1]:
                    continue
                for field in ("width", "contour_spacing", "parallel_spacing"):
                    value = layer.get(field)
                    if value is not None:
                        return float(value)
                source_metadata = layer.get("source_metadata")
                if isinstance(source_metadata, dict):
                    for field in ("width", "contour_spacing", "parallel_spacing"):
                        value = source_metadata.get(field)
                        if value is not None:
                            return float(value)
        for field in ("width", "contour_spacing", "parallel_spacing"):
            value = metadata.get(field)
            if value is not None:
                return float(value)
    return None


def infer_parallel_axis_from_metadata(path_file: Path) -> str | None:
    key = path_layer_key(path_file)
    for metadata_path in (path_file.parent / "batch_paths.meta.json", path_file.parent / "batch_3axis.meta.json"):
        if not metadata_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        layers = metadata.get("layers")
        if not isinstance(layers, list):
            continue
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            if int(layer.get("layer_main", -1)) != key[0] or int(layer.get("layer_sub", -1)) != key[1]:
                continue
            sources = (layer, layer.get("source_metadata"))
            for source in sources:
                if not isinstance(source, dict):
                    continue
                axis = str(source.get("parallel_axis", "")).lower()
                if axis in {"x", "z"}:
                    return axis
    return None


def recover_projected_parallel_runs(
    path_points: np.ndarray,
    *,
    parallel_axis: str,
    tolerance: float,
    min_length: float,
) -> list[np.ndarray]:
    """Recover long straight XZ raster runs from a connected projected path."""
    points = np.asarray(path_points, dtype=np.float64)
    if points.shape[0] < 2:
        return []
    along_index, across_index = (0, 2) if parallel_axis == "x" else (2, 0)
    deltas = np.diff(points[:, (along_index, across_index)], axis=0)
    is_raster_segment = (np.abs(deltas[:, 1]) <= tolerance) & (np.abs(deltas[:, 0]) > tolerance)
    runs: list[np.ndarray] = []
    start: int | None = None
    for segment_index, is_raster in enumerate(is_raster_segment):
        if is_raster and start is None:
            start = segment_index
        if start is not None and (not is_raster or segment_index == is_raster_segment.shape[0] - 1):
            end = segment_index + 1 if is_raster else segment_index
            indices = np.arange(start, end + 1, dtype=np.int64)
            length = float(np.sum(np.linalg.norm(np.diff(points[indices], axis=0), axis=1)))
            if length >= min_length:
                runs.append(indices)
            start = None
    return runs


def add_nearest_parallel_distance_diagnostic(
    plotter: object,
    *,
    path_points: np.ndarray,
    path_path: Path,
    line_width: float,
    requested_run_index: int | None,
    tolerance: float,
) -> None:
    import pyvista as pv
    from scipy.spatial import cKDTree

    parallel_axis = infer_parallel_axis_from_metadata(path_path)
    if parallel_axis is None:
        print(f"Nearest-parallel diagnostic skipped for {path_path.name}: projected parallel-axis metadata is unavailable.")
        return
    nominal_width = infer_nominal_width_from_metadata(path_path) or 0.0
    runs = recover_projected_parallel_runs(
        path_points,
        parallel_axis=parallel_axis,
        tolerance=float(tolerance),
        min_length=max(2.0 * nominal_width, 1e-6),
    )
    if len(runs) < 2:
        print(f"Nearest-parallel diagnostic skipped for {path_path.name}: fewer than two raster runs were recovered.")
        return
    across_index = 2 if parallel_axis == "x" else 0
    runs.sort(key=lambda indices: float(np.mean(path_points[indices, across_index])))
    selected_index = len(runs) // 2 if requested_run_index is None else int(requested_run_index)
    if selected_index < 0 or selected_index >= len(runs):
        raise ValueError(f"distance_parallel_index must be in [0, {len(runs) - 1}] for {path_path.name}.")
    selected_indices = runs[selected_index]
    other_indices = np.concatenate([indices for index, indices in enumerate(runs) if index != selected_index])
    tree = cKDTree(path_points[other_indices])
    distances, nearest_local = tree.query(path_points[selected_indices], k=1)
    source_local = int(np.argmin(distances))
    source_index = int(selected_indices[source_local])
    target_index = int(other_indices[int(nearest_local[source_local])])
    source_point = path_points[source_index]
    target_point = path_points[target_index]
    distance_3d = float(np.linalg.norm(target_point - source_point))
    distance_xz = float(np.linalg.norm((target_point - source_point)[[0, 2]]))
    target_run_index = next(index for index, indices in enumerate(runs) if target_index in indices)

    plotter.add_mesh(pv.lines_from_points(path_points[selected_indices], close=False), color="red", line_width=max(line_width * 1.6, line_width + 2.0))
    plotter.add_mesh(pv.lines_from_points(path_points[runs[target_run_index]], close=False), color="orange", line_width=max(line_width * 1.6, line_width + 2.0))
    connector = np.vstack((source_point, target_point))
    plotter.add_mesh(pv.lines_from_points(connector, close=False), color="yellow", line_width=max(line_width * 2.0, line_width + 3.0))
    plotter.add_mesh(pv.PolyData(connector), color="yellow", point_size=12.0, render_points_as_spheres=True)
    plotter.add_text(
        f"Projected raster direction: {parallel_axis}\n"
        f"Selected pass {selected_index} -> nearest pass {target_run_index}\n"
        f"3D separation: {distance_3d:.4f}\nXZ separation: {distance_xz:.4f}",
        position="upper_left",
        font_size=11,
    )
    print(
        f"{path_path.name}: selected raster pass {selected_index}, nearest pass {target_run_index}; "
        f"3D distance={distance_3d:.6g}, XZ distance={distance_xz:.6g}",
    )


def add_nearest_parallel_distance_coloring(
    plotter: object,
    *,
    path_points: np.ndarray,
    path_path: Path,
    line_width: float,
    point_size: float,
    tolerance: float,
) -> None:
    """Overlay raster passes with their local nearest-other-pass 3D distance."""
    import pyvista as pv
    from scipy.spatial import cKDTree

    parallel_axis = infer_parallel_axis_from_metadata(path_path)
    if parallel_axis is None:
        print(f"Nearest-parallel coloring skipped for {path_path.name}: projected parallel-axis metadata is unavailable.")
        return
    nominal_width = infer_nominal_width_from_metadata(path_path) or 0.0
    runs = recover_projected_parallel_runs(
        path_points,
        parallel_axis=parallel_axis,
        tolerance=float(tolerance),
        min_length=max(2.0 * nominal_width, 1e-6),
    )
    if len(runs) < 2:
        print(f"Nearest-parallel coloring skipped for {path_path.name}: fewer than two raster runs were recovered.")
        return

    distances_by_run: list[np.ndarray] = []
    for run_index, indices in enumerate(runs):
        other_indices = np.concatenate([other for index, other in enumerate(runs) if index != run_index])
        tree = cKDTree(path_points[other_indices])
        distances, _ = tree.query(path_points[indices], k=1)
        distances_by_run.append(np.asarray(distances, dtype=np.float64))
    all_distances = np.concatenate(distances_by_run)
    color_limits = (float(np.min(all_distances)), float(np.max(all_distances)))
    if color_limits[1] - color_limits[0] <= 1e-12:
        color_limits = (color_limits[0], color_limits[0] + 1e-12)

    for run_index, (indices, distances) in enumerate(zip(runs, distances_by_run)):
        raster_points = pv.PolyData(path_points[indices])
        raster_points["nearest_parallel_distance_3d"] = distances
        plotter.add_mesh(
            raster_points,
            scalars="nearest_parallel_distance_3d",
            cmap="viridis",
            clim=color_limits,
            point_size=max(point_size, line_width * 1.8),
            render_points_as_spheres=True,
            show_scalar_bar=run_index == 0,
            scalar_bar_args={"title": "Nearest raster\n3D distance"},
        )
    print(
        f"{path_path.name}: colored {len(runs)} recovered raster passes by nearest-pass 3D distance "
        f"(min={color_limits[0]:.6g}, max={color_limits[1]:.6g}).",
    )


def local_path_sample_spacing(path_points: np.ndarray, break_indices: list[int]) -> np.ndarray:
    """Return local 3D sample spacing, without bridging explicit path breaks."""
    point_count = path_points.shape[0]
    spacing = np.full(point_count, np.nan, dtype=np.float64)
    run_starts = [0, *break_indices]
    run_ends = [*break_indices, point_count]
    for start, end in zip(run_starts, run_ends, strict=False):
        run_count = end - start
        if run_count < 2:
            continue
        segment_lengths = np.linalg.norm(np.diff(path_points[start:end], axis=0), axis=1)
        spacing[start] = segment_lengths[0]
        spacing[end - 1] = segment_lengths[-1]
        if run_count > 2:
            # Mean adjacent segment length is the local sampling interval.
            spacing[start + 1:end - 1] = 0.5 * (segment_lengths[:-1] + segment_lengths[1:])
    return spacing


def add_path_sample_spacing_coloring(
    plotter: object,
    *,
    path_points: np.ndarray,
    path_path: Path,
    path_break_indices: list[int],
    point_size: float,
) -> None:
    """Overlay path samples colored by their local within-run 3D spacing."""
    import pyvista as pv

    spacing = local_path_sample_spacing(path_points, path_break_indices)
    valid = np.isfinite(spacing)
    if not np.any(valid):
        print(f"Sample-spacing coloring skipped for {path_path.name}: no path run has two points.")
        return
    values = spacing[valid]
    low = float(np.min(values))
    high = float(np.max(values))
    color_limits = (low, high if high - low > 1e-12 else low + 1e-12)
    point_cloud = pv.PolyData(path_points[valid])
    point_cloud["local_sample_spacing_3d"] = values
    plotter.add_mesh(
        point_cloud,
        scalars="local_sample_spacing_3d",
        cmap="plasma",
        clim=color_limits,
        point_size=point_size,
        render_points_as_spheres=True,
        scalar_bar_args={"title": "Local sample\nspacing (3D)"},
    )
    print(
        f"{path_path.name}: local 3D sample spacing over {values.shape[0]} points "
        f"(min={low:.6g}, mean={float(np.mean(values)):.6g}, max={high:.6g}).",
    )


def add_layer_to_plotter(
    plotter: object,
    *,
    mesh_path: Path,
    path_path: Path,
    layer_color: tuple[float, float, float],
    path_color: tuple[float, float, float],
    mesh_opacity: float,
    line_width: float,
    show_points: bool,
    point_size: float,
    show_normals: bool,
    normal_scale: float,
    show_width_highlights: bool,
    width_highlight_tolerance: float,
    width_highlight_color: str,
    show_nearest_parallel_distance: bool,
    color_nearest_parallel_distance: bool,
    color_path_sample_spacing: bool,
    distance_parallel_index: int | None,
    parallel_run_tolerance: float,
) -> None:
    import pyvista as pv

    V, F = load_obj_mesh(str(mesh_path))
    mesh = to_pyvista_polydata(V, F)
    plotter.add_mesh(
        mesh,
        color=layer_color,
        opacity=mesh_opacity,
        show_edges=False,
    )

    path_points, normals = load_path_points(path_path)
    width_profile = load_width_profile(path_path, path_points.shape[0]) if path_points.shape[0] > 0 else None
    path_break_indices = load_path_break_indices(path_path, path_points.shape[0]) if path_points.shape[0] > 0 else []
    if path_points.shape[0] >= 2:
        run_starts = [0, *path_break_indices]
        run_ends = [*path_break_indices, path_points.shape[0]]
        for start, end in zip(run_starts, run_ends, strict=False):
            if end - start >= 2:
                plotter.add_mesh(pv.lines_from_points(path_points[start:end], close=False), color=path_color, line_width=line_width)
        if show_width_highlights and width_profile is not None:
            segment_widths = 0.5 * (width_profile[:-1] + width_profile[1:])
            nominal_width = infer_nominal_width_from_metadata(path_path)
            if nominal_width is None:
                nominal_width = float(np.min(width_profile))
            wide_mask = segment_widths > nominal_width + float(width_highlight_tolerance)
            if np.any(wide_mask):
                for idx in np.flatnonzero(wide_mask):
                    segment_poly = pv.lines_from_points(path_points[idx:idx + 2], close=False)
                    plotter.add_mesh(
                        segment_poly,
                        color=width_highlight_color,
                        line_width=max(line_width * 1.8, line_width + 2.0),
                    )
    if show_points and path_points.shape[0] > 0:
        path_point_cloud = pv.PolyData(path_points)
        plotter.add_mesh(
            path_point_cloud,
            color=path_color,
            point_size=point_size,
            render_points_as_spheres=True,
        )

    if show_normals and normals is not None and path_points.shape[0] > 0:
        glyph_input = pv.PolyData(path_points)
        glyph_input["vectors"] = normals
        glyphs = glyph_input.glyph(
            orient="vectors",
            scale=False,
            factor=normal_scale,
        )
        plotter.add_mesh(glyphs, color=path_color)
    if show_nearest_parallel_distance and path_points.shape[0] > 0:
        add_nearest_parallel_distance_diagnostic(
            plotter,
            path_points=path_points,
            path_path=path_path,
            line_width=line_width,
            requested_run_index=distance_parallel_index,
            tolerance=parallel_run_tolerance,
        )
    if color_nearest_parallel_distance and path_points.shape[0] > 0:
        add_nearest_parallel_distance_coloring(
            plotter,
            path_points=path_points,
            path_path=path_path,
            line_width=line_width,
            point_size=point_size,
            tolerance=parallel_run_tolerance,
        )
    if color_path_sample_spacing and path_points.shape[0] > 0:
        add_path_sample_spacing_coloring(
            plotter,
            path_points=path_points,
            path_path=path_path,
            path_break_indices=path_break_indices,
            point_size=point_size,
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    mesh_files = collect_named_files(args.mesh_dir, "obj")
    path_files = collect_named_files(args.path_dir, "txt")
    shared_keys = sorted(set(mesh_files) & set(path_files))
    if args.layer is not None:
        match = re.fullmatch(r"(\d+)_(\d+)", args.layer.strip())
        if match is None:
            raise ValueError("--layer must be written as <main>_<sub>, for example 120_0.")
        requested_key = int(match.group(1)), int(match.group(2))
        if requested_key not in shared_keys:
            raise FileNotFoundError(f"Requested layer {args.layer} has no matching mesh and path file.")
        shared_keys = [requested_key]
    if not shared_keys:
        raise FileNotFoundError("No matching <main>_<sub> mesh/path pairs were found.")

    import pyvista as pv

    colors = (
        generate_randomized_layer_colors(len(shared_keys), seed=int(args.color_seed))
        if args.randomize_consecutive_colors
        else generate_layer_colors(len(shared_keys))
    )
    print(f"Mesh folder: {args.mesh_dir}")
    print(f"Path folder: {args.path_dir}")
    print(f"Matched layers: {len(shared_keys)}")

    if args.one_by_one:
        for idx, key in enumerate(shared_keys):
            mesh_path = mesh_files[key]
            path_path = path_files[key]
            layer_color = colors[idx]
            path_color = brighten_color(layer_color, factor=1.25)
            print(f"Layer {key[0]}_{key[1]}: mesh={mesh_path.name}, path={path_path.name}")
            plotter = pv.Plotter()
            add_layer_to_plotter(
                plotter,
                mesh_path=mesh_path,
                path_path=path_path,
                layer_color=layer_color,
                path_color=path_color,
                mesh_opacity=args.mesh_opacity,
                line_width=args.line_width,
                show_points=args.show_points,
                point_size=args.point_size,
                show_normals=args.show_normals,
                normal_scale=args.normal_scale,
                show_width_highlights=args.show_width_highlights,
                width_highlight_tolerance=args.width_highlight_tolerance,
                width_highlight_color=args.width_highlight_color,
                show_nearest_parallel_distance=args.show_nearest_parallel_distance,
                color_nearest_parallel_distance=args.color_nearest_parallel_distance,
                color_path_sample_spacing=args.color_path_sample_spacing,
                distance_parallel_index=args.distance_parallel_index,
                parallel_run_tolerance=args.parallel_run_tolerance,
            )
            plotter.add_text(f"Layer {key[0]}_{key[1]}", font_size=12)
            plotter.show()
    else:
        plotter = pv.Plotter()
        for idx, key in enumerate(shared_keys):
            mesh_path = mesh_files[key]
            path_path = path_files[key]
            layer_color = colors[idx]
            path_color = brighten_color(layer_color, factor=1.25)
            add_layer_to_plotter(
                plotter,
                mesh_path=mesh_path,
                path_path=path_path,
                layer_color=layer_color,
                path_color=path_color,
                mesh_opacity=args.mesh_opacity,
                line_width=args.line_width,
                show_points=args.show_points,
                point_size=args.point_size,
                show_normals=args.show_normals,
                normal_scale=args.normal_scale,
                show_width_highlights=args.show_width_highlights,
                width_highlight_tolerance=args.width_highlight_tolerance,
                width_highlight_color=args.width_highlight_color,
                show_nearest_parallel_distance=args.show_nearest_parallel_distance,
                color_nearest_parallel_distance=args.color_nearest_parallel_distance,
                color_path_sample_spacing=args.color_path_sample_spacing,
                distance_parallel_index=args.distance_parallel_index,
                parallel_run_tolerance=args.parallel_run_tolerance,
            )
            print(f"Layer {key[0]}_{key[1]}: mesh={mesh_path.name}, path={path_path.name}")
        plotter.show()


if __name__ == "__main__":
    main()
