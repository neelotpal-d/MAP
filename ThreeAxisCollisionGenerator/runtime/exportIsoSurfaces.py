"""Export scalar-field iso-surfaces from a TetGen mesh or padded voxel grid.

The iso-value walk starts at `scalar_min + epsilon` and stops before
`scalar_max - epsilon`. After each exported surface, the next iso-value is:

    current + average_grad_norm_on_surface * distance_world / scale

where `scale` is the scalar-field coordinate normalization scale.
"""

import argparse
from pathlib import Path

from runtimeEnv import configure_openmp_runtime

configure_openmp_runtime()

import numpy as np
import pyvista as pv
import torch

from scalarNetUtils import load_scalar_net_from_checkpoint


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCALAR_NAME = "scalar"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export scalar-field iso-surfaces as OBJ files.")
    parser.add_argument(
        "--field-type",
        choices=("neural", "x", "y", "z"),
        default="neural",
        help="Scalar field to use: checkpoint-backed neural field or normalized coordinate field.",
    )
    parser.add_argument("--checkpoint-path", default=None, help="Scalar-net checkpoint path for --field-type neural.")
    parser.add_argument(
        "--contour-source",
        choices=("tet", "voxel"),
        default="tet",
        help="Use a TetGen volume mesh or a padded regular voxel grid as the contour domain.",
    )
    parser.add_argument("--node-path", default=None, help="TetGen .node file paired with --ele-path for --contour-source tet.")
    parser.add_argument("--ele-path", default=None, help="TetGen .ele file used for --contour-source tet.")
    parser.add_argument(
        "--mesh-path",
        default=None,
        help="Closed input surface mesh required for --contour-source voxel and used to trim voxel contours.",
    )
    parser.add_argument("--voxel-spacing", type=float, default=1.0, help="Uniform world-space spacing for --contour-source voxel (legacy default).")
    parser.add_argument(
        "--voxel-spacing-xyz",
        type=float,
        nargs=3,
        default=None,
        metavar=("SX", "SY", "SZ"),
        help="Anisotropic voxel edge lengths in world units; overrides --voxel-spacing.",
    )
    parser.add_argument("--voxel-padding", type=float, default=1.0, help="World-space padding around --mesh-path in voxel mode.")
    parser.add_argument(
        "--voxel-min-region-area-fraction",
        type=float,
        default=0.001,
        help="Discard disconnected voxel regions smaller than this fraction of the largest region at the same iso-level; 0 disables it.",
    )
    parser.add_argument(
        "--voxel-preclip-volume",
        action="store_true",
        help="Clip the voxel volume to the OBJ before scalar evaluation instead of clipping every extracted layer.",
    )
    parser.add_argument(
        "--axis-conversion",
        choices=("none", "z-up-to-y-up", "y-up-to-z-up"),
        default="none",
        help="Optional right-handed axis conversion applied after scalar assignment and before export.",
    )
    parser.add_argument(
        "--print-direction",
        type=float,
        nargs=3,
        default=(0.0, 1.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help="Output-space print direction used to place the model minimum at zero; defaults to +Y.",
    )
    parser.add_argument(
        "--keep-export-origin",
        action="store_true",
        help="Do not translate exported layers so the input model minimum along --print-direction is zero.",
    )
    parser.add_argument(
        "--normalization-model-path",
        default=None,
        help="Optional mesh used to compute normalization. Defaults to the input mesh in voxel mode, else contour source.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for exported OBJ files.")
    parser.add_argument("--distance", type=float, required=True, help="Desired world-space spacing between slices.")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Scalar-space trim from min and max.")
    parser.add_argument(
        "--start-scalar",
        type=float,
        default=None,
        help="Optional scalar value for the first exported layer; defaults to scalar_min + epsilon.",
    )
    parser.add_argument(
        "--export-scalar-field-txt",
        action="store_true",
        help="Also export the scalar field as a TXT file in contour-source point order.",
    )
    parser.add_argument(
        "--scalar-field-txt-name",
        default="scalar_field.txt",
        help="Filename for the exported scalar-field TXT inside --output-dir.",
    )
    parser.add_argument(
        "--export-main-scalar-values-txt",
        action="store_true",
        help="Also export one scalar value per exported main layer as a TXT file.",
    )
    parser.add_argument(
        "--main-scalar-values-txt-name",
        default="main_scalar_values.txt",
        help="Filename for the exported main-layer scalar values TXT inside --output-dir.",
    )
    parser.add_argument("--device", default=DEVICE, help="Torch device.")
    parser.add_argument("--dim-hidden", type=int, default=128, help="Scalar-net hidden width.")
    parser.add_argument("--num-layers", type=int, default=5, help="Scalar-net depth.")
    parser.add_argument("--w0", type=float, default=1.0, help="SIREN hidden-layer frequency.")
    parser.add_argument("--w0-initial", type=float, default=1.0, help="SIREN first-layer frequency.")
    parser.add_argument("--scalar-batch-size", type=int, default=50000, help="Batch size for scalar evaluation.")
    parser.add_argument("--grad-batch-size", type=int, default=20000, help="Batch size for gradient evaluation.")
    parser.add_argument(
        "--verify-neural-iso-residuals",
        action="store_true",
        help="Evaluate the neural field at every exported iso-surface vertex and save residual diagnostics as VTP files.",
    )
    parser.add_argument(
        "--project-neural-iso",
        action="store_true",
        help="Apply guarded Newton updates along the neural world gradient before export; implies --verify-neural-iso-residuals.",
    )
    parser.add_argument("--neural-projection-iterations", type=int, default=3, help="Newton updates for --project-neural-iso.")
    parser.add_argument(
        "--neural-projection-max-step",
        type=float,
        default=0.25,
        help="Maximum world-space displacement per vertex per Newton update.",
    )
    parser.add_argument(
        "--max-surfaces",
        type=int,
        default=None,
        help="Optional safety cap on the number of exported iso-surfaces.",
    )
    return parser.parse_args()


def convert_points_axis(points: np.ndarray, axis_conversion: str) -> np.ndarray:
    """Convert point coordinates between common up-axis conventions."""
    if axis_conversion == "none":
        return points
    if axis_conversion == "z-up-to-y-up":
        # Right-handed rotation: old z becomes new y.
        return np.stack((points[:, 0], points[:, 2], -points[:, 1]), axis=1)
    if axis_conversion == "y-up-to-z-up":
        # Inverse of z-up-to-y-up.
        return np.stack((points[:, 0], -points[:, 2], points[:, 1]), axis=1)
    raise ValueError(f"Unsupported axis conversion: {axis_conversion}")


def convert_dataset_axis(dataset: pv.DataSet, axis_conversion: str) -> pv.DataSet:
    """Return a copy of a dataset with converted point coordinates."""
    converted_dataset = dataset.copy(deep=True)
    converted_dataset.points = convert_points_axis(np.asarray(dataset.points), axis_conversion)
    return converted_dataset


def compute_print_base_translation(
    reference_points: np.ndarray,
    print_direction: np.ndarray | tuple[float, float, float] | list[float],
) -> tuple[np.ndarray, float]:
    """Return the output translation placing the reference minimum at print coordinate zero."""

    direction = np.asarray(print_direction, dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction))
    if direction.shape != (3,) or direction_norm <= 1e-12:
        raise ValueError("--print-direction must be a non-zero three-component vector.")
    direction /= direction_norm
    minimum_projection = float(np.min(np.asarray(reference_points, dtype=np.float64) @ direction))
    return -minimum_projection * direction, minimum_projection


def compute_normalization(points: np.ndarray, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the same center and isotropic range scale used by the slicers."""
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)

    min_vals = torch.tensor([x_min, y_min, z_min], dtype=torch.float32, device=device)
    max_vals = torch.tensor([x_max, y_max, z_max], dtype=torch.float32, device=device)
    mid_vals = 0.5 * (min_vals + max_vals)

    max_range = np.max([x_max - x_min, y_max - y_min, z_max - z_min])
    range_vals = 0.5 * torch.tensor(
        [max_range, max_range, max_range],
        dtype=torch.float32,
        device=device,
    )
    return mid_vals, range_vals


def normalize_world_points(
    points_world: np.ndarray,
    mid_vals: torch.Tensor,
    range_vals: torch.Tensor,
    device: str,
    requires_grad: bool,
) -> torch.Tensor:
    points = torch.tensor(points_world, dtype=torch.float32, device=device)
    points = (points - mid_vals) / range_vals
    points = points.detach()
    points.requires_grad = requires_grad
    return points


def load_closed_surface_mesh(mesh_path: str) -> pv.PolyData:
    """Load the input surface used to bound and trim voxel-mode contours."""
    mesh = pv.read(mesh_path)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    mesh = mesh.triangulate()
    mesh.compute_normals(
        point_normals=True,
        cell_normals=True,
        consistent_normals=True,
        auto_orient_normals=True,
        inplace=True,
    )
    return mesh


def build_padded_voxel_grid(mesh: pv.PolyData, spacing: float | tuple[float, float, float] | list[float] | np.ndarray, padding: float) -> pv.ImageData:
    """Create a rectilinear, axis-aligned grid that overestimates closed mesh bounds."""
    spacing_xyz = np.asarray(spacing, dtype=np.float64)
    if spacing_xyz.ndim == 0:
        spacing_xyz = np.repeat(spacing_xyz, 3)
    if spacing_xyz.shape != (3,) or np.any(spacing_xyz <= 0.0):
        raise ValueError("Voxel spacing must contain three positive world-space values.")
    if padding < 0.0:
        raise ValueError("--voxel-padding cannot be negative.")

    bounds = np.asarray(mesh.bounds, dtype=np.float64).reshape(3, 2)
    lower = bounds[:, 0] - padding
    upper = bounds[:, 1] + padding
    dimensions = np.ceil((upper - lower) / spacing_xyz).astype(np.int64) + 1
    grid = pv.ImageData(
        dimensions=tuple(int(value) for value in dimensions),
        spacing=tuple(float(value) for value in spacing_xyz),
        origin=tuple(float(value) for value in lower),
    )
    return grid


def clip_iso_surface_to_mesh(surface: pv.PolyData, mesh: pv.PolyData) -> pv.PolyData:
    """Keep the portion of a voxel-extracted iso-surface inside the input mesh."""
    if surface.n_points == 0 or surface.n_cells == 0:
        return surface
    # VTK implicit distances are negative inside a closed surface; invert=True
    # retains that interior portion and creates the boundary intersection.
    return surface.clip_surface(mesh, invert=True).extract_surface(algorithm="dataset_surface")


def evaluate_scalar_values(
    scalar_net,
    points_world: np.ndarray,
    mid_vals: torch.Tensor,
    range_vals: torch.Tensor,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Evaluate scalar values on world-space points in batches."""
    scalar_batches = []

    for start in range(0, points_world.shape[0], batch_size):
        batch_world = points_world[start : start + batch_size]
        batch_model = normalize_world_points(
            batch_world,
            mid_vals=mid_vals,
            range_vals=range_vals,
            device=device,
            requires_grad=False,
        )
        with torch.no_grad():
            try:
                output = scalar_net(batch_model, compute_grads=False)
            except TypeError:
                output = scalar_net(batch_model)

        scalars = output["scalars"] if isinstance(output, dict) else output
        scalar_batches.append(scalars.detach().to("cpu").numpy().reshape(-1))

    return np.concatenate(scalar_batches, axis=0)


def evaluate_neural_values_and_world_gradients(
    scalar_net,
    points_world: np.ndarray,
    mid_vals: torch.Tensor,
    range_vals: torch.Tensor,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate f and df/dx_world at arbitrary world-space points."""
    scalar_batches: list[np.ndarray] = []
    gradient_batches: list[np.ndarray] = []
    for start in range(0, points_world.shape[0], batch_size):
        batch_model = normalize_world_points(
            points_world[start : start + batch_size], mid_vals, range_vals, device, requires_grad=True
        )
        output = scalar_net(batch_model)
        scalars = output["scalars"] if isinstance(output, dict) else output
        gradients_model = output["grads"]
        gradients_world = gradients_model / range_vals
        scalar_batches.append(scalars.detach().to("cpu").numpy().reshape(-1))
        gradient_batches.append(gradients_world.detach().to("cpu").numpy())
    return np.concatenate(scalar_batches), np.concatenate(gradient_batches)


def project_surface_to_neural_iso(
    surface: pv.PolyData,
    target_scalar: float,
    scalar_net,
    mid_vals: torch.Tensor,
    range_vals: torch.Tensor,
    device: str,
    batch_size: int,
    iterations: int,
    max_step: float,
) -> tuple[pv.PolyData, np.ndarray, np.ndarray]:
    """Project a sampled contour toward the true neural iso-surface safely."""
    refined = surface.copy(deep=True)
    points = np.asarray(refined.points, dtype=np.float64).copy()
    for _ in range(iterations):
        values, gradients = evaluate_neural_values_and_world_gradients(
            scalar_net, points, mid_vals, range_vals, device, batch_size
        )
        residual = target_scalar - values
        squared_norm = np.einsum("ij,ij->i", gradients, gradients)
        valid = squared_norm > 1e-16
        displacement = np.zeros_like(points)
        displacement[valid] = gradients[valid] * (residual[valid] / squared_norm[valid])[:, None]
        lengths = np.linalg.norm(displacement, axis=1)
        clipped = lengths > max_step
        if np.any(clipped):
            displacement[clipped] *= (max_step / lengths[clipped])[:, None]
        points += displacement
        if not np.any(valid) or float(np.max(np.abs(residual[valid]))) < 1e-6:
            break
    refined.points = points
    values, _ = evaluate_neural_values_and_world_gradients(
        scalar_net, points, mid_vals, range_vals, device, batch_size
    )
    return refined, values, values - target_scalar


def evaluate_coordinate_scalar_values(
    points_world: np.ndarray,
    mid_vals: torch.Tensor,
    range_vals: torch.Tensor,
    device: str,
    axis: str,
) -> np.ndarray:
    """Evaluate an in-house normalized coordinate scalar field."""
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    points_model = normalize_world_points(
        points_world,
        mid_vals=mid_vals,
        range_vals=range_vals,
        device=device,
        requires_grad=False,
    )
    return points_model[:, axis_index].detach().to("cpu").numpy().reshape(-1)


def evaluate_average_grad_norm(
    scalar_net,
    points_world: np.ndarray,
    mid_vals: torch.Tensor,
    range_vals: torch.Tensor,
    device: str,
    batch_size: int,
) -> float:
    """Evaluate average model-space gradient norm on an iso-surface."""
    grad_norm_sum = 0.0
    point_count = 0

    for start in range(0, points_world.shape[0], batch_size):
        batch_world = points_world[start : start + batch_size]
        batch_model = normalize_world_points(
            batch_world,
            mid_vals=mid_vals,
            range_vals=range_vals,
            device=device,
            requires_grad=True,
        )
        output = scalar_net(batch_model)
        grads = output["grads"]
        grad_norm = torch.norm(grads, dim=1)
        grad_norm_sum += grad_norm.detach().sum().item()
        point_count += grad_norm.numel()

    if point_count == 0:
        return 0.0
    return grad_norm_sum / point_count


def evaluate_coordinate_average_grad_norm() -> float:
    """Return the model-space gradient norm for x/scale, y/scale, or z/scale fields."""
    return 1.0


def extract_iso_surface(dataset: pv.DataSet, scalar_value: float) -> pv.PolyData:
    """Extract one iso-surface at the requested scalar value."""
    return dataset.contour([scalar_value], scalars=SCALAR_NAME)


def split_iso_surface_regions(surface: pv.PolyData) -> list[pv.PolyData]:
    """Split one iso-surface into disconnected connected components."""
    if surface.n_points == 0 or surface.n_cells == 0:
        return []

    all_regions = surface.connectivity("all")
    region_ids = np.unique(all_regions["RegionId"])

    regions = []
    for region_id in region_ids:
        region = all_regions.threshold(
            value=(float(region_id), float(region_id)),
            scalars="RegionId",
        ).extract_surface(algorithm="dataset_surface")
        if region.n_points > 0 and region.n_cells > 0:
            regions.append(region)
    return regions


def save_iso_surface(
    surface: pv.PolyData,
    output_dir: Path,
    main_index: int,
    sub_index: int = 0,
    export_translation: np.ndarray | None = None,
) -> Path:
    """Save one iso-surface using the current main/sub-index convention."""
    output_path = output_dir / f"{main_index}_{sub_index}.obj"
    output_surface = surface.triangulate()
    if export_translation is not None:
        output_surface = output_surface.copy(deep=True)
        output_surface.points = np.asarray(output_surface.points) + np.asarray(export_translation, dtype=np.float64)
    output_surface.save(output_path)
    return output_path


def save_scalar_field_txt(scalars: np.ndarray, output_path: Path) -> None:
    """Save one scalar value per row in the original TetGen point order."""
    np.savetxt(output_path, scalars.reshape(-1, 1), fmt="%.10f")


def save_main_scalar_values_txt(main_scalar_values: list[float], output_path: Path) -> None:
    """Save one exported main-layer scalar value per row."""
    np.savetxt(output_path, np.asarray(main_scalar_values, dtype=np.float64).reshape(-1, 1), fmt="%.10f")


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.voxel_min_region_area_fraction < 1.0:
        raise ValueError("--voxel-min-region-area-fraction must lie in [0, 1).")
    if args.neural_projection_iterations < 1:
        raise ValueError("--neural-projection-iterations must be positive.")
    if args.neural_projection_max_step <= 0.0:
        raise ValueError("--neural-projection-max-step must be positive.")
    if args.project_neural_iso and args.field_type != "neural":
        raise ValueError("--project-neural-iso requires --field-type neural.")
    if args.project_neural_iso:
        args.verify_neural_iso_residuals = True

    trim_mesh = None
    inside_voxel_mask = None
    if args.contour_source == "tet":
        if args.node_path is None or args.ele_path is None:
            raise ValueError("--node-path and --ele-path are required for --contour-source tet.")
        if not Path(args.node_path).exists():
            raise FileNotFoundError(f"Node file not found: {args.node_path}")
        if not Path(args.ele_path).exists():
            raise FileNotFoundError(f"Element file not found: {args.ele_path}")
        model_dataset = pv.read(args.ele_path)
    else:
        if args.mesh_path is None:
            raise ValueError("--mesh-path is required for --contour-source voxel.")
        if not Path(args.mesh_path).exists():
            raise FileNotFoundError(f"Input mesh not found: {args.mesh_path}")
        trim_mesh = load_closed_surface_mesh(args.mesh_path)
        voxel_spacing = args.voxel_spacing_xyz if args.voxel_spacing_xyz is not None else args.voxel_spacing
        model_dataset = build_padded_voxel_grid(trim_mesh, voxel_spacing, args.voxel_padding)
        if args.voxel_preclip_volume:
            # Removing exterior cells first also avoids neural evaluation at
            # padded grid points that cannot contribute to an exported layer.
            model_dataset = model_dataset.clip_surface(trim_mesh, invert=True)
            if model_dataset.n_cells == 0:
                raise ValueError("Pre-clipping the voxel volume produced no interior cells.")
            print("Pre-clipped the voxel volume to the OBJ before scalar evaluation.")
        else:
            voxel_distances = model_dataset.compute_implicit_distance(trim_mesh)["implicit_distance"]
            inside_voxel_mask = np.asarray(voxel_distances) <= 0.0
            if not np.any(inside_voxel_mask):
                raise ValueError("The padded voxel grid contains no points inside --mesh-path.")

    normalization_dataset = (
        pv.read(args.normalization_model_path)
        if args.normalization_model_path
        else (trim_mesh if trim_mesh is not None else model_dataset)
    )
    mid_vals, range_vals = compute_normalization(normalization_dataset.points, args.device)
    scale = range_vals[0]

    scalar_net = None
    if args.field_type == "neural":
        if args.checkpoint_path is None:
            raise ValueError("--checkpoint-path is required when --field-type neural.")
        scalar_net = load_scalar_net_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            dim_hidden=args.dim_hidden,
            num_layers=args.num_layers,
            w0=args.w0,
            w0_initial=args.w0_initial,
        )
        scalars = evaluate_scalar_values(
            scalar_net=scalar_net,
            points_world=model_dataset.points,
            mid_vals=mid_vals,
            range_vals=range_vals,
            device=args.device,
            batch_size=args.scalar_batch_size,
        )
    else:
        scalars = evaluate_coordinate_scalar_values(
            points_world=model_dataset.points,
            mid_vals=mid_vals,
            range_vals=range_vals,
            device=args.device,
            axis=args.field_type,
        )

    contour_dataset = model_dataset.copy(deep=True)
    contour_dataset.point_data[SCALAR_NAME] = scalars
    trim_each_surface = trim_mesh
    if trim_mesh is not None and args.voxel_preclip_volume:
        trim_each_surface = None
        print("Skipping per-layer surface clipping.")

    scalar_bounds = scalars if inside_voxel_mask is None else scalars[inside_voxel_mask]
    scalar_min = float(np.min(scalar_bounds))
    scalar_max = float(np.max(scalar_bounds))
    scalar_stop = scalar_max - args.epsilon
    default_start_scalar = scalar_min + args.epsilon
    current_scalar = default_start_scalar if args.start_scalar is None else float(args.start_scalar)
    if current_scalar < default_start_scalar or current_scalar > scalar_stop:
        raise ValueError(
            f"--start-scalar must lie in [{default_start_scalar:.8f}, {scalar_stop:.8f}], got {current_scalar:.8f}."
        )
    if args.start_scalar is not None:
        print(f"Starting adaptive layer export at scalar {current_scalar:.8f}.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_translation = None
    if not args.keep_export_origin:
        # Axis conversion is part of the output coordinate system, so measure
        # the base after that conversion rather than in field-evaluation space.
        export_reference = trim_mesh if trim_mesh is not None else normalization_dataset
        export_reference = convert_dataset_axis(export_reference, args.axis_conversion)
        export_translation, base_projection = compute_print_base_translation(
            export_reference.points, args.print_direction
        )
        print(
            "Exporting layers with the model print-base translated to zero: "
            f"min projection={base_projection:.8f}, translation={export_translation.tolist()}."
        )
    if args.export_scalar_field_txt:
        save_scalar_field_txt(scalars, output_dir / args.scalar_field_txt_name)

    main_index = 0
    main_scalar_values = []
    while current_scalar <= scalar_stop:
        if args.max_surfaces is not None and main_index >= args.max_surfaces:
            break

        surface = extract_iso_surface(contour_dataset, current_scalar)
        if trim_each_surface is not None:
            surface = clip_iso_surface_to_mesh(surface, trim_each_surface)
        if surface.n_points == 0 or surface.n_cells == 0:
            print(f"Skipping empty iso-surface at scalar {current_scalar:.6f}")
            current_scalar += args.distance / scale.item()
            continue

        residual_text = ""
        if args.verify_neural_iso_residuals:
            neural_values, _ = evaluate_neural_values_and_world_gradients(
                scalar_net, np.asarray(surface.points), mid_vals, range_vals, args.device, args.grad_batch_size
            )
            initial_residuals = neural_values - current_scalar
            surface.point_data["NeuralScalarBeforeProjection"] = neural_values
            surface.point_data["NeuralResidualBeforeProjection"] = initial_residuals
            if args.project_neural_iso:
                surface, final_values, final_residuals = project_surface_to_neural_iso(
                    surface,
                    current_scalar,
                    scalar_net,
                    mid_vals,
                    range_vals,
                    args.device,
                    args.grad_batch_size,
                    args.neural_projection_iterations,
                    args.neural_projection_max_step,
                )
                surface.point_data["NeuralScalar"] = final_values
                surface.point_data["NeuralResidual"] = final_residuals
                residual_text = (
                    f", neural_residual_abs_max={np.max(np.abs(initial_residuals)):.3e}"
                    f"->{np.max(np.abs(final_residuals)):.3e}"
                )
            else:
                surface.point_data["NeuralScalar"] = neural_values
                surface.point_data["NeuralResidual"] = initial_residuals
                residual_text = f", neural_residual_abs_max={np.max(np.abs(initial_residuals)):.3e}"

        split_surfaces = split_iso_surface_regions(surface)
        if args.contour_source == "voxel" and split_surfaces:
            largest_area = max(float(region.area) for region in split_surfaces)
            retained = [
                region for region in split_surfaces
                if float(region.area) >= largest_area * args.voxel_min_region_area_fraction
            ]
            removed = len(split_surfaces) - len(retained)
            if removed:
                print(f"Discarded {removed} small voxel region(s) at scalar {current_scalar:.6f}.")
            split_surfaces = retained
        if not split_surfaces:
            print(f"Skipping iso-surface with no connected regions at scalar {current_scalar:.6f}")
            current_scalar += args.distance / scale.item()
            continue

        output_paths = []
        for sub_index, split_surface in enumerate(split_surfaces):
            if args.verify_neural_iso_residuals:
                split_surface.save(output_dir / f"{main_index}_{sub_index}_neural_residuals.vtp")
            split_surface = convert_dataset_axis(split_surface, args.axis_conversion)
            output_paths.append(
                save_iso_surface(
                    split_surface,
                    output_dir,
                    main_index=main_index,
                    sub_index=sub_index,
                    export_translation=export_translation,
                )
            )
        main_scalar_values.append(float(current_scalar))

        if args.field_type == "neural":
            avg_grad_norm = evaluate_average_grad_norm(
                scalar_net=scalar_net,
                points_world=surface.points,
                mid_vals=mid_vals,
                range_vals=range_vals,
                device=args.device,
                batch_size=args.grad_batch_size,
            )
        else:
            avg_grad_norm = evaluate_coordinate_average_grad_norm()

        scalar_step = avg_grad_norm * args.distance / scale.item()
        print(
            f"{main_index}_*: scalar={current_scalar:.6f}, "
            f"regions={len(split_surfaces)}, avg_grad_norm={avg_grad_norm:.6f}, "
            f"next_step={scalar_step:.6f}{residual_text}, saved={[str(path) for path in output_paths]}"
        )

        if scalar_step <= 1e-8:
            print("Stopping because computed scalar step is too small.")
            break

        current_scalar += scalar_step
        main_index += 1

    if args.export_main_scalar_values_txt:
        save_main_scalar_values_txt(main_scalar_values, output_dir / args.main_scalar_values_txt_name)


if __name__ == "__main__":
    main()
