"""Batched collision checking against a scalar field and plane-bounded tool region.

For each sampled origin point inside the model:
1. place the tool origin at that point,
2. express all sampled candidate points in the fixed `(u, v, n)` tool frame,
3. keep points inside the user-defined tool bounds,
4. mark collision if any included point has scalar value lower than the origin scalar.

The visualization mode shows:
- mesh / optional volume context,
- contours only up to the chosen colliding origin scalar value,
- one selected colliding origin,
- the transparent tool box/planes at that origin,
- all colliding sampled points for that origin.
"""

import argparse
import json
from pathlib import Path

from runtimeEnv import configure_openmp_runtime

configure_openmp_runtime()

import igl
import numpy as np
import pyvista as pv
import torch

from scalarNetUtils import load_scalar_net_from_checkpoint
from toolSampling import build_tool_frame
from axisymmetricTool import tool_contains, nominal_mesh, validate_nominal


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCALAR_NAME = "scalar"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check scalar-order collisions for a plane-bounded tool region.")
    parser.add_argument("--config-path", default=None, help="Optional JSON config file to load arguments from.")
    parser.add_argument(
        "--save-config-path",
        default=None,
        help="Optional JSON path to save the resolved arguments before running.",
    )
    parser.add_argument("--mesh-path", default=None, help="Surface mesh used for inside/outside sampling.")
    parser.add_argument("--volume-ele-path", default=None, help="Optional TetGen .ele file for contour visualization.")
    parser.add_argument("--volume-node-path", default=None, help="Optional TetGen .node file paired with --volume-ele-path.")
    parser.add_argument(
        "--contour-source",
        choices=("auto", "tet", "surface", "voxel"),
        default="auto",
        help="Layer-generation domain. 'auto' preserves the old TetGen-then-surface fallback.",
    )
    parser.add_argument("--voxel-spacing", type=float, default=1.0, help="World-space voxel spacing in voxel mode.")
    parser.add_argument("--voxel-padding", type=float, default=1.0, help="World-space bounds padding in voxel mode.")
    parser.add_argument(
        "--field-type",
        choices=("neural", "x", "y", "z"),
        default="neural",
        help="Scalar field source: checkpoint-backed neural field or normalized coordinate field.",
    )
    parser.add_argument("--checkpoint-path", default=None, help="Scalar-net checkpoint path for --field-type neural.")
    parser.add_argument(
        "--normalization-model-path",
        default=None,
        help="Optional mesh used to compute normalization. Defaults to --mesh-path.",
    )
    parser.add_argument("--device", default=DEVICE, help="Torch device.")
    parser.add_argument("--dim-hidden", type=int, default=128, help="Scalar-net hidden width.")
    parser.add_argument("--num-layers", type=int, default=5, help="Scalar-net depth.")
    parser.add_argument("--w0", type=float, default=1.0, help="SIREN hidden-layer frequency.")
    parser.add_argument("--w0-initial", type=float, default=1.0, help="SIREN first-layer frequency.")
    parser.add_argument("--scalar-batch-size", type=int, default=50000, help="Batch size for scalar evaluation.")
    parser.add_argument("--origin-batch-size", type=int, default=32, help="Batch size for origin collision checking.")
    parser.add_argument("--padding", type=float, default=0.6, help="Bounding-box padding for grid sampling.")
    parser.add_argument("--grid-spacing", type=float, default=2.0, help="Grid spacing for coarse interior sampling.")
    parser.add_argument(
        "--inside-method",
        choices=("vtk", "igl"),
        default="vtk",
        help="Use VTK's fast enclosed-point test (default) or exact libigl signed distance for root inclusion.",
    )
    parser.add_argument(
        "--candidate-oversampling",
        type=float,
        default=4.0,
        help="When --max-sampled-points caps roots, test at most this factor times that many raw grid points.",
    )
    parser.add_argument(
        "--inside-threshold",
        type=float,
        default=0.0,
        help="Keep points with signed distance below this threshold. Use 0 for strict interior.",
    )
    parser.add_argument("--max-sampled-points", type=int, default=100000, help="Optional cap on sampled points.")
    parser.add_argument("--build-direction", type=float, nargs=3, default=(0.0, 1.0, 0.0), metavar=("NX", "NY", "NZ"))
    parser.add_argument("--in-plane-u", type=float, nargs=3, default=(1.0, 0.0, 0.0), metavar=("UX", "UY", "UZ"))
    parser.add_argument("--tool-x-min", type=float, default=-70.0)
    parser.add_argument("--tool-x-max", type=float, default=20.0)
    parser.add_argument("--tool-y-min", type=float, default=0.0)
    parser.add_argument("--tool-y-max", type=float, default=4.0)
    parser.add_argument("--tool-z-min", type=float, default=-35.0)
    parser.add_argument("--tool-z-max", type=float, default=35.0)
    parser.add_argument(
        "--debug-viz",
        action="store_true",
        help="Show the simple setup only: sampled roots, contours, and one reference tool box, without computing collisions.",
    )
    parser.add_argument("--viz", action="store_true", help="Show a visualization for one colliding origin.")
    parser.add_argument(
        "--viz-all-colliding-origins",
        action="store_true",
        help="Show all colliding root/origin points as a point cloud after collision computation.",
    )
    parser.add_argument(
        "--viz-colliding-origin-rank",
        type=int,
        default=0,
        help="Choose the Nth colliding origin to visualize.",
    )
    parser.add_argument("--viz-contour-count", type=int, default=20, help="Number of contour levels for visualization.")
    parser.add_argument(
        "--viz-contour-step-world",
        type=float,
        default=0.2,
        help="Base world-space contour step used to offset above scalar min.",
    )
    parser.add_argument(
        "--viz-cross-section",
        action="store_true",
        help="Open an additional window showing a planar cross-section of the contours and tool.",
    )
    parser.add_argument(
        "--viz-cross-section-axis",
        choices=("x", "y", "z"),
        default="y",
        help="World axis normal to the optional cross-section plane.",
    )
    parser.add_argument(
        "--viz-cross-section-position-world",
        type=float,
        default=None,
        help="World coordinate of the optional cross-section; defaults to the model center on its selected axis.",
    )
    parser.set_defaults(axisymmetric_tool=None)
    args = parser.parse_args()
    return apply_json_config(parser, args)


def apply_json_config(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    """Load defaults from a JSON config and let explicit CLI args override them."""
    if args.config_path is None:
        return args

    config_path = Path(args.config_path)
    config_data = json.loads(config_path.read_text(encoding="utf-8"))

    explicit_cli_names = set()
    argv_tokens = __import__("sys").argv[1:]
    for token in argv_tokens:
        if token.startswith("--"):
            option_name = token[2:].replace("-", "_")
            explicit_cli_names.add(option_name)

    for key, value in config_data.items():
        if not hasattr(args, key):
            continue
        if key in explicit_cli_names:
            continue
        setattr(args, key, value)

    return args


def save_args_to_json(args: argparse.Namespace, output_path: str) -> None:
    """Save the resolved script arguments to a JSON config file."""
    config_dict = vars(args).copy()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")


def load_volume_mesh(node_path: str | None, ele_path: str | None) -> pv.DataSet | None:
    """Optionally load a TetGen volume mesh for contour visualization."""
    if not node_path or not ele_path:
        return None

    node_file = Path(node_path)
    ele_file = Path(ele_path)
    if not node_file.exists() or not ele_file.exists():
        return None

    try:
        return pv.read(str(ele_file))
    except Exception:
        return None


def build_padded_voxel_grid(mesh: pv.PolyData, spacing: float, padding: float) -> pv.ImageData:
    """Create the regular grid used as the scalar contour/layer domain."""
    if spacing <= 0.0:
        raise ValueError("--voxel-spacing must be positive.")
    if padding < 0.0:
        raise ValueError("--voxel-padding cannot be negative.")

    bounds = np.asarray(mesh.bounds, dtype=np.float64).reshape(3, 2)
    lower = bounds[:, 0] - padding
    upper = bounds[:, 1] + padding
    dimensions = np.ceil((upper - lower) / spacing).astype(np.int64) + 1
    return pv.ImageData(
        dimensions=tuple(int(value) for value in dimensions),
        spacing=(float(spacing),) * 3,
        origin=tuple(float(value) for value in lower),
    )


def select_contour_source(
    args: argparse.Namespace,
    surface_mesh: pv.PolyData,
    volume_mesh: pv.DataSet | None,
) -> tuple[pv.DataSet, pv.PolyData | None, np.ndarray | None]:
    """Select the layer domain and return optional voxel clipping metadata."""
    if args.contour_source == "tet":
        if volume_mesh is None:
            raise ValueError("--contour-source tet requires valid --volume-node-path and --volume-ele-path files.")
        return volume_mesh, None, None
    if args.contour_source == "surface":
        return surface_mesh, None, None
    if args.contour_source == "auto":
        return (volume_mesh if volume_mesh is not None else surface_mesh), None, None

    trim_mesh = surface_mesh.extract_surface().triangulate()
    trim_mesh.compute_normals(
        point_normals=True,
        cell_normals=True,
        consistent_normals=True,
        auto_orient_normals=True,
        inplace=True,
    )
    voxel_grid = build_padded_voxel_grid(trim_mesh, args.voxel_spacing, args.voxel_padding)
    inside_mask = np.asarray(voxel_grid.compute_implicit_distance(trim_mesh)["implicit_distance"]) <= 0.0
    if not np.any(inside_mask):
        raise ValueError("The voxel layer grid contains no points inside --mesh-path.")
    return voxel_grid, trim_mesh, inside_mask


def clip_voxel_contours(contours: pv.PolyData, trim_mesh: pv.PolyData | None) -> pv.PolyData:
    """Trim voxel-generated layers to the closed input mesh."""
    if trim_mesh is None or contours.n_points == 0 or contours.n_cells == 0:
        return contours
    return contours.clip_surface(trim_mesh, invert=True).extract_surface(algorithm="dataset_surface")


def compute_normalization(points: np.ndarray, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the centered isotropic scale used throughout the repo."""
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)

    min_vals = torch.tensor([x_min, y_min, z_min], dtype=torch.float32, device=device)
    max_vals = torch.tensor([x_max, y_max, z_max], dtype=torch.float32, device=device)
    mid_vals = 0.5 * (min_vals + max_vals)

    max_range = np.max([x_max - x_min, y_max - y_min, z_max - z_min])
    range_vals = 0.5 * torch.tensor([max_range, max_range, max_range], dtype=torch.float32, device=device)
    return mid_vals, range_vals


def normalize_world_points(
    points_world: np.ndarray,
    mid_vals: torch.Tensor,
    range_vals: torch.Tensor,
    device: str,
    requires_grad: bool,
) -> torch.Tensor:
    """Normalize world-space points into model coordinates."""
    points = torch.tensor(points_world, dtype=torch.float32, device=device)
    points = (points - mid_vals) / range_vals
    points = points.detach()
    points.requires_grad = requires_grad
    return points


def sample_candidate_points(
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    surface_mesh: pv.PolyData,
    padding: float,
    grid_spacing: float,
    inside_threshold: float,
    max_sampled_points: int | None,
    inside_method: str,
    candidate_oversampling: float,
) -> np.ndarray:
    """Sample a coarse grid and keep interior points (or a chosen inside band)."""
    x_min, y_min, z_min = np.min(mesh_vertices, axis=0)
    x_max, y_max, z_max = np.max(mesh_vertices, axis=0)

    x_vals = np.arange(x_min - padding, x_max + padding, grid_spacing)
    y_vals = np.arange(y_min - padding, y_max + padding, grid_spacing)
    z_vals = np.arange(z_min - padding, z_max + padding, grid_spacing)

    grid_x, grid_y, grid_z = np.meshgrid(x_vals, y_vals, z_vals)
    grid_points = np.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), axis=1)
    total_grid_points = len(grid_points)
    if max_sampled_points is not None:
        if candidate_oversampling < 1.0:
            raise ValueError("--candidate-oversampling must be at least one.")
        raw_limit = int(np.ceil(max_sampled_points * candidate_oversampling))
        if total_grid_points > raw_limit:
            selected = np.random.default_rng(0).choice(total_grid_points, raw_limit, replace=False)
            grid_points = grid_points[selected]
            print(f"Collision roots: testing {len(grid_points)}/{total_grid_points} raw grid points before inside filtering.")

    if inside_method == "vtk" and inside_threshold == 0.0:
        # The collision roots need only an inside/outside classification. VTK's
        # enclosed-point test avoids an exact closest-triangle distance query
        # for every grid point on very dense meshes.
        cloud = pv.PolyData(grid_points)
        if hasattr(cloud, "select_interior_points"):
            # Keep the legacy VTK locator/tolerance with the PyVista 0.47+ API.
            enclosed = cloud.select_interior_points(
                surface_mesh, method="cell_locator", locator_tolerance=1e-6, check_surface=False
            )
            inside = np.asarray(enclosed.point_data["selected_points"], dtype=bool)
        else:
            enclosed = cloud.select_enclosed_points(
                surface_mesh, tolerance=1e-6, check_surface=False
            )
            inside = np.asarray(enclosed.point_data["SelectedPoints"], dtype=bool)
        candidate_points = grid_points[inside]
    else:
        if inside_method == "vtk":
            print("Using exact libigl distance because --inside-threshold is not zero.")
        sdf = igl.signed_distance(grid_points, mesh_vertices, mesh_faces)[0]
        candidate_points = grid_points[sdf < inside_threshold]

    if max_sampled_points is not None and candidate_points.shape[0] > max_sampled_points:
        random_indices = np.random.default_rng(0).choice(candidate_points.shape[0], size=max_sampled_points, replace=False)
        candidate_points = candidate_points[random_indices]

    return candidate_points


def evaluate_scalar_values(
    scalar_net,
    points_world: np.ndarray,
    mid_vals: torch.Tensor,
    range_vals: torch.Tensor,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Evaluate a neural scalar field on world-space points in batches."""
    scalar_batches = []

    for start in range(0, points_world.shape[0], batch_size):
        batch_world = points_world[start : start + batch_size]
        batch_model = normalize_world_points(batch_world, mid_vals, range_vals, device, requires_grad=False)
        with torch.no_grad():
            try:
                output = scalar_net(batch_model, compute_grads=False)
            except TypeError:
                output = scalar_net(batch_model)
        scalars = output["scalars"] if isinstance(output, dict) else output
        scalar_batches.append(scalars.detach().to("cpu").numpy().reshape(-1))

    return np.concatenate(scalar_batches, axis=0)


def evaluate_coordinate_scalar_values(
    points_world: np.ndarray,
    mid_vals: torch.Tensor,
    range_vals: torch.Tensor,
    device: str,
    axis: str,
) -> np.ndarray:
    """Evaluate normalized coordinate fields x/scale, y/scale, or z/scale."""
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    points_model = normalize_world_points(points_world, mid_vals, range_vals, device, requires_grad=False)
    return points_model[:, axis_index].detach().to("cpu").numpy().reshape(-1)


def build_scalar_field(args: argparse.Namespace):
    """Return either the neural field or a built-in coordinate-field marker."""
    if args.field_type == "neural":
        if args.checkpoint_path is None:
            raise ValueError("--checkpoint-path is required when --field-type neural.")
        return load_scalar_net_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            dim_hidden=args.dim_hidden,
            num_layers=args.num_layers,
            w0=args.w0,
            w0_initial=args.w0_initial,
        )
    return None


def compute_collision_mask_batched(
    candidate_points_model: torch.Tensor,
    candidate_scalars: torch.Tensor,
    tool_frame: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tool_bounds: dict[str, float],
    origin_batch_size: int,
) -> torch.Tensor:
    """Return a boolean collision mask for all candidate origins using batched vectorization."""
    u, v, n = tool_frame
    num_points = candidate_points_model.shape[0]
    collision_mask = torch.zeros(num_points, dtype=torch.bool, device=candidate_points_model.device)

    for start in range(0, num_points, origin_batch_size):
        stop = min(start + origin_batch_size, num_points)
        origins = candidate_points_model[start:stop]
        origin_scalars = candidate_scalars[start:stop]

        offsets = candidate_points_model.unsqueeze(0) - origins.unsqueeze(1)
        a_coords = torch.einsum("bnc,c->bn", offsets, u)
        b_coords = torch.einsum("bnc,c->bn", offsets, v)
        h_coords = torch.einsum("bnc,c->bn", offsets, n)

        inside_tool = tool_contains(a_coords, b_coords, h_coords, tool_bounds)
        lower_scalar = candidate_scalars.unsqueeze(0) < origin_scalars.unsqueeze(1)
        collision_mask[start:stop] = torch.any(inside_tool & lower_scalar, dim=1)

    return collision_mask


def find_colliding_points_for_origin(
    origin_point_model: torch.Tensor,
    origin_scalar: torch.Tensor,
    candidate_points_model: torch.Tensor,
    candidate_points_world: np.ndarray,
    candidate_scalars: torch.Tensor,
    tool_frame: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tool_bounds: dict[str, float],
) -> np.ndarray:
    """Return all sampled world-space points that collide with one chosen origin."""
    u, v, n = tool_frame
    offsets = candidate_points_model - origin_point_model.unsqueeze(0)
    a_coords = offsets @ u
    b_coords = offsets @ v
    h_coords = offsets @ n

    inside_tool = tool_contains(a_coords, b_coords, h_coords, tool_bounds)
    lower_scalar = candidate_scalars < origin_scalar
    colliding_mask = inside_tool & lower_scalar
    return candidate_points_world[colliding_mask.detach().to("cpu").numpy()]


def build_tool_box_mesh(
    origin_point_world: np.ndarray,
    tool_frame: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tool_bounds: dict[str, float],
) -> pv.PolyData:
    """Build a transparent tool box/plane-bounded region mesh in world space."""
    if 'axisymmetric_tool' in tool_bounds:
        return nominal_mesh(tool_bounds['axisymmetric_tool'], origin_point_world, tool_frame)
    u, v, n = tool_frame
    u = u.detach().to("cpu")
    v = v.detach().to("cpu")
    n = n.detach().to("cpu")
    origin = torch.tensor(origin_point_world, dtype=torch.float32)

    x0, x1 = tool_bounds["x_min"], tool_bounds["x_max"]
    y0, y1 = tool_bounds["y_min"], tool_bounds["y_max"]
    z0, z1 = tool_bounds["z_min"], tool_bounds["z_max"]

    corners_local = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]

    corners_world = []
    for x_val, y_val, z_val in corners_local:
        corners_world.append(origin + x_val * u + y_val * n + z_val * v)
    corners_world = torch.stack(corners_world, dim=0).numpy()

    faces = np.hstack(
        (
            [4, 0, 1, 2, 3],
            [4, 4, 5, 6, 7],
            [4, 0, 1, 5, 4],
            [4, 1, 2, 6, 5],
            [4, 2, 3, 7, 6],
            [4, 3, 0, 4, 7],
        )
    )
    return pv.PolyData(corners_world, faces)


def build_tool_box_edges(tool_box_mesh: pv.PolyData) -> pv.PolyData:
    """Extract visible feature edges so the tool bounds remain readable."""
    return tool_box_mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=True,
        non_manifold_edges=True,
    )


def build_contours_until_origin(
    contour_source: pv.DataSet,
    origin_scalar: float,
    args: argparse.Namespace,
    model_scalars: np.ndarray,
    range_vals: torch.Tensor,
    scalar_bounds: np.ndarray | None = None,
    trim_mesh: pv.PolyData | None = None,
) -> pv.PolyData:
    """Build contours only up to the chosen origin scalar value."""
    contour_dataset = contour_source.copy(deep=True)
    contour_dataset.point_data[SCALAR_NAME] = model_scalars

    bounded_scalars = model_scalars if scalar_bounds is None else scalar_bounds
    scalar_min = float(bounded_scalars.min())
    scalar_max = float(bounded_scalars.max())
    contour_stop = min(origin_scalar, scalar_max)
    step = args.viz_contour_step_world / range_vals[0].detach().to("cpu").item()
    contour_start = scalar_min + step

    if contour_stop <= contour_start:
        contour_values = [contour_stop]
    else:
        contour_values = np.linspace(contour_start, contour_stop, args.viz_contour_count)

    return clip_voxel_contours(contour_dataset.contour(contour_values, scalars=SCALAR_NAME), trim_mesh)


def build_full_contours(
    contour_source: pv.DataSet,
    contour_scalars: np.ndarray,
    args: argparse.Namespace,
    range_vals: torch.Tensor,
    scalar_bounds: np.ndarray | None = None,
    trim_mesh: pv.PolyData | None = None,
) -> pv.PolyData:
    """Build a full contour stack across the available scalar range."""
    contour_dataset = contour_source.copy(deep=True)
    contour_dataset.point_data[SCALAR_NAME] = contour_scalars

    bounded_scalars = contour_scalars if scalar_bounds is None else scalar_bounds
    scalar_min = float(bounded_scalars.min())
    scalar_max = float(bounded_scalars.max())
    step = args.viz_contour_step_world / range_vals[0].detach().to("cpu").item()
    contour_start = scalar_min + step

    if scalar_max <= contour_start:
        contour_values = [scalar_max]
    else:
        contour_values = np.linspace(contour_start, scalar_max, args.viz_contour_count)

    return clip_voxel_contours(contour_dataset.contour(contour_values, scalars=SCALAR_NAME), trim_mesh)


def select_reference_origin(candidate_points_world: np.ndarray, build_direction_world: np.ndarray) -> np.ndarray:
    """Choose a stable reference origin for debug visualization."""
    build_direction_world = np.asarray(build_direction_world, dtype=np.float32)
    build_direction_world = build_direction_world / (np.linalg.norm(build_direction_world) + 1e-8)
    projected_height = candidate_points_world @ build_direction_world
    return candidate_points_world[int(np.argmin(projected_height))]


def visualize_cross_section(
    surface_mesh: pv.PolyData,
    contours: pv.PolyData,
    args: argparse.Namespace,
    tool_box_mesh: pv.PolyData | None = None,
    highlighted_contour: pv.PolyData | None = None,
) -> None:
    """Open a section-only view of the model, contour stack, and optional tool."""
    if not args.viz_cross_section:
        return

    axis_index = {"x": 0, "y": 1, "z": 2}[args.viz_cross_section_axis]
    normal = np.zeros(3, dtype=np.float64)
    normal[axis_index] = 1.0
    origin = np.asarray(surface_mesh.center, dtype=np.float64)
    if args.viz_cross_section_position_world is not None:
        origin[axis_index] = args.viz_cross_section_position_world

    surface_section = surface_mesh.slice(normal=normal, origin=origin)
    contour_section = contours.slice(normal=normal, origin=origin)
    highlight_section = (
        None
        if highlighted_contour is None
        else highlighted_contour.slice(normal=normal, origin=origin)
    )
    tool_section = None if tool_box_mesh is None else tool_box_mesh.slice(normal=normal, origin=origin)

    plotter = pv.Plotter(title=f"Collision-check cross-section ({args.viz_cross_section_axis} = {origin[axis_index]:.3f})")
    plotter.enable_anti_aliasing()
    plotter.add_axes()
    if surface_section.n_points > 0:
        plotter.add_mesh(surface_section, color="#8C8C8C", line_width=2.0, lighting=False)
    if contour_section.n_points > 0:
        plotter.add_mesh(contour_section, color="#8FB3D9", line_width=2.0, lighting=False)
    if highlight_section is not None and highlight_section.n_points > 0:
        plotter.add_mesh(highlight_section, color="#1F2A44", line_width=4.0, lighting=False)
    if tool_section is not None and tool_section.n_points > 0:
        plotter.add_mesh(tool_section, color="#CC6B00", line_width=4.0, lighting=False)

    if args.viz_cross_section_axis == "x":
        plotter.view_yz()
    elif args.viz_cross_section_axis == "y":
        plotter.view_xz()
    else:
        plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.show()


def visualize_debug_setup(
    surface_mesh: pv.PolyData,
    volume_mesh: pv.DataSet | None,
    contour_source: pv.DataSet,
    contour_scalars: np.ndarray,
    range_vals: torch.Tensor,
    args: argparse.Namespace,
    candidate_points_world: np.ndarray,
    tool_box_mesh: pv.PolyData,
    contour_scalar_bounds: np.ndarray | None,
    contour_trim_mesh: pv.PolyData | None,
) -> None:
    """Visualize the sampled roots and the tool frame without computing collisions."""
    contours = build_full_contours(
        contour_source, contour_scalars, args, range_vals, contour_scalar_bounds, contour_trim_mesh
    )
    tool_box_edges = build_tool_box_edges(tool_box_mesh)
    candidate_cloud = pv.PolyData(candidate_points_world)

    plotter = pv.Plotter()
    plotter.enable_anti_aliasing()
    plotter.add_axes()
    plotter.add_mesh(contours, color="#8FB3D9", opacity=0.35, line_width=2.0, lighting=False)

    if volume_mesh is not None:
        plotter.add_mesh(volume_mesh, color="lightgray", opacity=0.08, show_edges=True)
        plotter.add_mesh(surface_mesh, color="lightblue", opacity=0.05)
    else:
        plotter.add_mesh(surface_mesh, color="lightblue", opacity=0.1)

    plotter.add_mesh(candidate_cloud, color="#2E4057", opacity=0.5, point_size=6.0, render_points_as_spheres=True)
    plotter.add_mesh(tool_box_mesh, color="#F4B266", opacity=0.2, lighting=False)
    plotter.add_mesh(tool_box_edges, color="#CC6B00", line_width=5.0, lighting=False)
    plotter.enable_parallel_projection()
    plotter.show()
    visualize_cross_section(surface_mesh, contours, args, tool_box_mesh=tool_box_mesh)


def visualize_all_colliding_origins(
    surface_mesh: pv.PolyData,
    volume_mesh: pv.DataSet | None,
    contour_source: pv.DataSet,
    contour_scalars: np.ndarray,
    range_vals: torch.Tensor,
    args: argparse.Namespace,
    colliding_origins_world: np.ndarray,
    contour_scalar_bounds: np.ndarray | None,
    contour_trim_mesh: pv.PolyData | None,
) -> None:
    """Visualize every colliding origin/root point at once."""
    contours = build_full_contours(
        contour_source, contour_scalars, args, range_vals, contour_scalar_bounds, contour_trim_mesh
    )
    colliding_cloud = pv.PolyData(colliding_origins_world)

    plotter = pv.Plotter()
    plotter.enable_anti_aliasing()
    plotter.add_axes()
    plotter.add_mesh(contours, color="#8FB3D9", opacity=0.35, line_width=2.0, lighting=False)

    if volume_mesh is not None:
        plotter.add_mesh(volume_mesh, color="lightgray", opacity=0.08, show_edges=True)
        plotter.add_mesh(surface_mesh, color="lightblue", opacity=0.05)
    else:
        plotter.add_mesh(surface_mesh, color="lightblue", opacity=0.1)

    plotter.add_mesh(colliding_cloud, color="red", point_size=8.0, render_points_as_spheres=True)
    plotter.enable_parallel_projection()
    plotter.show()
    visualize_cross_section(surface_mesh, contours, args)


def visualize_selected_collision(
    surface_mesh: pv.PolyData,
    volume_mesh: pv.DataSet | None,
    contour_source: pv.DataSet,
    contour_scalars: np.ndarray,
    range_vals: torch.Tensor,
    args: argparse.Namespace,
    origin_point_world: np.ndarray,
    origin_scalar: float,
    colliding_points_world: np.ndarray,
    tool_box_mesh: pv.PolyData,
    contour_scalar_bounds: np.ndarray | None,
    contour_trim_mesh: pv.PolyData | None,
) -> None:
    """Visualize one selected colliding origin, its tool, colliding points, and contour stack."""
    contours = build_contours_until_origin(
        contour_source,
        origin_scalar,
        args,
        contour_scalars,
        range_vals,
        contour_scalar_bounds,
        contour_trim_mesh,
    )
    last_contour_source = contour_source.copy(deep=True)
    last_contour_source.point_data[SCALAR_NAME] = contour_scalars
    last_contour = clip_voxel_contours(
        last_contour_source.contour([origin_scalar], scalars=SCALAR_NAME), contour_trim_mesh
    )
    tool_box_edges = build_tool_box_edges(tool_box_mesh)

    plotter = pv.Plotter()
    plotter.enable_anti_aliasing()
    plotter.add_axes()
    plotter.add_mesh(contours, color="#8FB3D9", opacity=0.45, line_width=2.0, lighting=False)
    plotter.add_mesh(last_contour, color="#1F2A44", opacity=1.0, line_width=4.0, lighting=False)

    if volume_mesh is not None:
        plotter.add_mesh(volume_mesh, color="lightgray", opacity=0.08, show_edges=True)
        plotter.add_mesh(surface_mesh, color="lightblue", opacity=0.05)
    else:
        plotter.add_mesh(surface_mesh, color="lightblue", opacity=0.1)

    plotter.add_mesh(tool_box_mesh, color="#F4B266", opacity=0.2, lighting=False)
    plotter.add_mesh(tool_box_edges, color="#CC6B00", line_width=5.0, lighting=False)
    plotter.add_mesh(pv.PolyData(origin_point_world.reshape(1, 3)), color="purple", point_size=16.0, render_points_as_spheres=True)

    if colliding_points_world.shape[0] > 0:
        colliding_cloud = pv.PolyData(colliding_points_world)
        plotter.add_mesh(colliding_cloud, color="red", point_size=10.0, render_points_as_spheres=True)

    plotter.enable_parallel_projection()
    plotter.show()
    visualize_cross_section(
        surface_mesh,
        contours,
        args,
        tool_box_mesh=tool_box_mesh,
        highlighted_contour=last_contour,
    )


def main() -> None:
    args = parse_args()
    if args.save_config_path is not None:
        save_args_to_json(args, args.save_config_path)
    if args.mesh_path is None:
        raise ValueError("--mesh-path is required either on the command line or in --config-path.")

    surface_mesh = pv.read(args.mesh_path)
    # Collision detection needs only interior roots and their scalar values.
    # Preparing a voxel contour volume triggers an expensive full-mesh
    # implicit-distance query and was previously done even for headless checks.
    needs_contours = args.debug_viz or args.viz or args.viz_all_colliding_origins
    volume_mesh = None
    contour_source = contour_trim_mesh = inside_contour_mask = None
    if needs_contours:
        volume_mesh = load_volume_mesh(args.volume_node_path, args.volume_ele_path)
        contour_source, contour_trim_mesh, inside_contour_mask = select_contour_source(
            args, surface_mesh, volume_mesh
        )
    else:
        print("Headless collision check: skipping contour-volume construction and mesh trimming.")

    normalization_mesh = pv.read(args.normalization_model_path) if args.normalization_model_path else surface_mesh
    mesh_vertices = np.asarray(surface_mesh.points)
    mesh_faces = np.asarray(surface_mesh.faces).reshape((-1, 4))[:, 1:4]

    candidate_points_world = sample_candidate_points(
        mesh_vertices=mesh_vertices,
        mesh_faces=mesh_faces,
        padding=args.padding,
        grid_spacing=args.grid_spacing,
        inside_threshold=args.inside_threshold,
        max_sampled_points=args.max_sampled_points,
        surface_mesh=surface_mesh,
        inside_method=args.inside_method,
        candidate_oversampling=args.candidate_oversampling,
    )
    print(f"Sampled {candidate_points_world.shape[0]} candidate interior points.")

    mid_vals, range_vals = compute_normalization(np.asarray(normalization_mesh.points), args.device)
    scalar_field = build_scalar_field(args)

    if args.field_type == "neural":
        candidate_scalars_np = evaluate_scalar_values(
            scalar_net=scalar_field,
            points_world=candidate_points_world,
            mid_vals=mid_vals,
            range_vals=range_vals,
            device=args.device,
            batch_size=args.scalar_batch_size,
        )
        contour_scalars_np = (
            evaluate_scalar_values(
                scalar_net=scalar_field,
                points_world=np.asarray(contour_source.points),
                mid_vals=mid_vals,
                range_vals=range_vals,
                device=args.device,
                batch_size=args.scalar_batch_size,
            )
            if needs_contours else None
        )
    else:
        candidate_scalars_np = evaluate_coordinate_scalar_values(
            points_world=candidate_points_world,
            mid_vals=mid_vals,
            range_vals=range_vals,
            device=args.device,
            axis=args.field_type,
        )
        contour_scalars_np = (
            evaluate_coordinate_scalar_values(
                points_world=np.asarray(contour_source.points),
                mid_vals=mid_vals,
                range_vals=range_vals,
                device=args.device,
                axis=args.field_type,
            )
            if needs_contours else None
        )

    contour_scalar_bounds = (
        contour_scalars_np[inside_contour_mask] if inside_contour_mask is not None else None
    )

    candidate_points_model = normalize_world_points(
        candidate_points_world,
        mid_vals=mid_vals,
        range_vals=range_vals,
        device=args.device,
        requires_grad=False,
    )
    candidate_scalars = torch.tensor(candidate_scalars_np, dtype=torch.float32, device=args.device)

    u, v, n = build_tool_frame(
        build_direction_world=args.build_direction,
        in_plane_u_world=args.in_plane_u,
        device=args.device,
    )
    tool_bounds_model = {
        "x_min": args.tool_x_min / range_vals[0].item(),
        "x_max": args.tool_x_max / range_vals[0].item(),
        "y_min": args.tool_y_min / range_vals[0].item(),
        "y_max": args.tool_y_max / range_vals[0].item(),
        "z_min": args.tool_z_min / range_vals[0].item(),
        "z_max": args.tool_z_max / range_vals[0].item(),
    }
    tool_bounds_world = {
        "x_min": args.tool_x_min,
        "x_max": args.tool_x_max,
        "y_min": args.tool_y_min,
        "y_max": args.tool_y_max,
        "z_min": args.tool_z_min,
        "z_max": args.tool_z_max,
    }

    if args.axisymmetric_tool is not None:
        validate_nominal(args.axisymmetric_tool)
        tool_bounds_model = dict(axisymmetric_tool=args.axisymmetric_tool, scale_mm=range_vals[0].item())
        tool_bounds_world = dict(axisymmetric_tool=args.axisymmetric_tool)

    if args.debug_viz:
        reference_origin_world = select_reference_origin(candidate_points_world, np.asarray(args.build_direction))
        reference_tool_box_mesh = build_tool_box_mesh(reference_origin_world, (u, v, n), tool_bounds_world)
        visualize_debug_setup(
            surface_mesh=surface_mesh,
            volume_mesh=volume_mesh,
            contour_source=contour_source,
            contour_scalars=contour_scalars_np,
            range_vals=range_vals,
            args=args,
            candidate_points_world=candidate_points_world,
            tool_box_mesh=reference_tool_box_mesh,
            contour_scalar_bounds=contour_scalar_bounds,
            contour_trim_mesh=contour_trim_mesh,
        )
        return

    collision_mask = compute_collision_mask_batched(
        candidate_points_model=candidate_points_model,
        candidate_scalars=candidate_scalars,
        tool_frame=(u, v, n),
        tool_bounds=tool_bounds_model,
        origin_batch_size=args.origin_batch_size,
    )
    collision_mask_np = collision_mask.detach().to("cpu").numpy()
    colliding_origin_indices = np.nonzero(collision_mask_np)[0]

    print(f"Colliding origins: {colliding_origin_indices.size}/{candidate_points_world.shape[0]}")
    if colliding_origin_indices.size == 0:
        return

    if args.viz_all_colliding_origins:
        visualize_all_colliding_origins(
            surface_mesh=surface_mesh,
            volume_mesh=volume_mesh,
            contour_source=contour_source,
            contour_scalars=contour_scalars_np,
            range_vals=range_vals,
            args=args,
            colliding_origins_world=candidate_points_world[colliding_origin_indices],
            contour_scalar_bounds=contour_scalar_bounds,
            contour_trim_mesh=contour_trim_mesh,
        )

    if not args.viz:
        return

    chosen_rank = int(np.clip(args.viz_colliding_origin_rank, 0, colliding_origin_indices.size - 1))
    chosen_origin_index = int(colliding_origin_indices[chosen_rank])
    origin_point_world = candidate_points_world[chosen_origin_index]
    origin_point_model = candidate_points_model[chosen_origin_index]
    origin_scalar = candidate_scalars[chosen_origin_index]

    colliding_points_world = find_colliding_points_for_origin(
        origin_point_model=origin_point_model,
        origin_scalar=origin_scalar,
        candidate_points_model=candidate_points_model,
        candidate_points_world=candidate_points_world,
        candidate_scalars=candidate_scalars,
        tool_frame=(u, v, n),
        tool_bounds=tool_bounds_model,
    )
    tool_box_mesh = build_tool_box_mesh(origin_point_world, (u, v, n), tool_bounds_world)

    visualize_selected_collision(
        surface_mesh=surface_mesh,
        volume_mesh=volume_mesh,
        contour_source=contour_source,
        contour_scalars=contour_scalars_np,
        range_vals=range_vals,
        args=args,
        origin_point_world=origin_point_world,
        origin_scalar=float(origin_scalar.detach().to("cpu").item()),
        colliding_points_world=colliding_points_world,
        tool_box_mesh=tool_box_mesh,
        contour_scalar_bounds=contour_scalar_bounds,
        contour_trim_mesh=contour_trim_mesh,
    )


if __name__ == "__main__":
    main()
