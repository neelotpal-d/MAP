"""Generate warm-start scalar-field initializers for a mesh and frequency pair."""

import argparse
from pathlib import Path

from runtimeEnv import configure_openmp_runtime

configure_openmp_runtime()

import igl
import numpy as np
import pyvista as pv
import torch

from fourierFeatureNet import FourierFeatureNet
from scalarNetUtils import initialize_scalar_net, warm_start_scalar_net_to_direction
from sdfField import compute_normalization


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The current experiments keep the scalar-network width and depth fixed.
SCALAR_NET_DIM_HIDDEN = 128
SCALAR_NET_NUM_LAYERS = 5
FOURIER_NUM_FEATURES = 128
FOURIER_SIGMA = 4.0
FOURIER_SEED = 0

BOUNDARY_PADDING = 1.1
GRID_SPACING = 2.0
NEAR_SURFACE_BAND = 1.0
OUTER_BAND = 1.5
MAX_INPUT_POINTS = 100000

DEFAULT_TARGET_DIRECTION = (0.0, 1.0, 0.0)
DEFAULT_EPOCHS = 150
DEFAULT_LR = 1e-4
DEFAULT_BATCH_SIZE = 4000
DEFAULT_DIRECTION_WEIGHT = 1.0
DEFAULT_SCALAR_ANCHOR_WEIGHT = 0.0
DEFAULT_CURVATURE_WEIGHT = 0.0
DEFAULT_CURVATURE_SAMPLE_FRACTION = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a warm-start scalar initializer checkpoint.")
    parser.add_argument("--model-name", required=True, help="Short model name used in the output checkpoint file.")
    parser.add_argument("--mesh-path", required=True, help="Path to the surface mesh used for sampling.")
    parser.add_argument("--backend", choices=("siren", "fourier"), default="siren", help="Scalar-field backend.")
    parser.add_argument("--w0", type=float, default=1.0, help="SIREN hidden-layer frequency.")
    parser.add_argument("--w0-initial", type=float, default=1.0, help="SIREN first-layer frequency.")
    parser.add_argument("--num-layers", type=int, default=SCALAR_NET_NUM_LAYERS, help="Number of SIREN hidden layers.")
    parser.add_argument('--seed', type=int, default=None, help='Optional reproducible sampling and network initialization seed')
    parser.add_argument("--fourier-num-features", type=int, default=FOURIER_NUM_FEATURES, help="Number of Fourier frequencies.")
    parser.add_argument("--fourier-sigma", type=float, default=FOURIER_SIGMA, help="Fourier frequency bandwidth.")
    parser.add_argument("--fourier-seed", type=int, default=FOURIER_SEED, help="Fixed random Fourier-feature seed.")
    parser.add_argument("--device", default=DEVICE, help="Torch device to use.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Warm-start optimization epochs.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Warm-start learning rate.")
    parser.add_argument(
        "--lr-warmup-epochs", type=int, default=0,
        help="Linearly ramp the initializer learning rate to --lr over this many initial epochs; 0 disables the ramp.",
    )
    parser.add_argument(
        "--lr-warmup-factor", type=float, default=1e-3,
        help="Learning-rate fraction used on the first warm-up epoch.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Warm-start batch size.")
    parser.add_argument(
        "--direction-weight",
        type=float,
        default=DEFAULT_DIRECTION_WEIGHT,
        help="Weight on the raw gradient-matching loss.",
    )
    parser.add_argument(
        "--scalar-anchor-weight",
        type=float,
        default=DEFAULT_SCALAR_ANCHOR_WEIGHT,
        help="Optional weight driving scalar values toward zero during warm start.",
    )
    parser.add_argument(
        "--curvature-weight",
        type=float,
        default=DEFAULT_CURVATURE_WEIGHT,
        help="Optional Hessian-squared penalty that suppresses curvature during warm start.",
    )
    parser.add_argument(
        "--curvature-sample-fraction",
        type=float,
        default=DEFAULT_CURVATURE_SAMPLE_FRACTION,
        help="Fraction of each direction batch randomly used for the curvature estimate.",
    )
    parser.add_argument("--debug-progress", action="store_true", help="Print progress for every warm-start batch.")
    parser.add_argument("--padding", type=float, default=BOUNDARY_PADDING, help="Grid padding in world units.")
    parser.add_argument("--grid-spacing", type=float, default=GRID_SPACING, help="Grid spacing in world units.")
    parser.add_argument(
        "--grid-spacing-xyz",
        type=float,
        nargs=3,
        default=None,
        metavar=("DX", "DY", "DZ"),
        help="Optional per-axis sampling spacings; overrides --grid-spacing.",
    )
    parser.add_argument("--min-grid-samples-per-axis", type=int, default=1, help="Minimum grid coordinates along every axis.")
    parser.add_argument(
        "--near-surface-band",
        type=float,
        default=NEAR_SURFACE_BAND,
        help="Keep all interior points plus points within this exterior SDF band.",
    )
    parser.add_argument(
        "--outer-band",
        type=float,
        default=OUTER_BAND,
        help="Reserved outer SDF band, retained for consistency with slicer sampling.",
    )
    parser.add_argument(
        "--max-input-points",
        type=int,
        default=MAX_INPUT_POINTS,
        help="Maximum sampled points used for warm-start optimization.",
    )
    parser.add_argument(
        "--target-direction",
        type=float,
        nargs=3,
        default=DEFAULT_TARGET_DIRECTION,
        metavar=("DX", "DY", "DZ"),
        help="Target gradient direction in model space.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where the initializer checkpoint will be written.",
    )
    args = parser.parse_args()
    if args.lr_warmup_epochs < 0:
        parser.error("--lr-warmup-epochs cannot be negative.")
    if not 0.0 < args.lr_warmup_factor <= 1.0:
        parser.error("--lr-warmup-factor must be in (0, 1].")
    return args


def format_frequency_value(value: float) -> str:
    """Format a frequency value for use in checkpoint file names."""
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def build_initializer_checkpoint_path(args: argparse.Namespace) -> Path:
    """Build a backend-specific initializer name without overwriting SIREN runs."""
    if args.backend == "fourier":
        filename = (
            f"scalar_init_{args.model_name}_fourier_f{args.fourier_num_features}"
            f"_s{format_frequency_value(args.fourier_sigma)}_seed{args.fourier_seed}.pt"
        )
    else:
        filename = (
            f"scalar_init_{args.model_name}_w{format_frequency_value(args.w0_initial)}"
            f"_w{format_frequency_value(args.w0)}.pt"
        )
    if args.num_layers != SCALAR_NET_NUM_LAYERS:
        filename = filename.removesuffix(".pt") + f"_l{args.num_layers}.pt"
    if args.curvature_weight > 0.0:
        filename = filename.removesuffix(".pt") + f"_curv{format_frequency_value(args.curvature_weight)}.pt"
    return Path(args.output_dir) / filename


def load_mesh_arrays(mesh_path: str) -> tuple[pv.PolyData, np.ndarray, np.ndarray]:
    mesh = pv.read(mesh_path)
    vertices = np.array(mesh.points)
    faces = np.array(mesh.faces).reshape((-1, 4))[:, 1:4]
    return mesh, vertices, faces


def sample_model_points(
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    padding: float,
    grid_spacing: float | tuple[float, float, float] | list[float] | np.ndarray,
    min_samples_per_axis: int,
    near_surface_band: float,
    outer_band: float,
    max_input_points: int,
) -> np.ndarray:
    x_min, y_min, z_min = np.min(mesh_vertices, axis=0)
    x_max, y_max, z_max = np.max(mesh_vertices, axis=0)

    axis_spacing = np.asarray(grid_spacing, dtype=np.float64)
    if axis_spacing.ndim == 0:
        axis_spacing = np.full(3, float(axis_spacing))
    if axis_spacing.shape != (3,) or np.any(axis_spacing <= 0.0) or min_samples_per_axis < 1:
        raise ValueError("Grid spacing and minimum axis sample count must be positive.")

    def axis_values(lower: float, upper: float, axis: int) -> np.ndarray:
        values = np.arange(lower - padding, upper + padding, axis_spacing[axis])
        if values.size >= min_samples_per_axis:
            return values
        return np.linspace(lower - padding, upper + padding, min_samples_per_axis)

    x_vals = axis_values(x_min, x_max, 0)
    y_vals = axis_values(y_min, y_max, 1)
    z_vals = axis_values(z_min, z_max, 2)

    grid_x, grid_y, grid_z = np.meshgrid(x_vals, y_vals, z_vals)
    grid_points = np.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), axis=1)
    print(
        "Initializer sampling grid: "
        f"axis_counts=({len(x_vals)}, {len(y_vals)}, {len(z_vals)}), total={len(grid_points)}.",
        flush=True,
    )

    sdf = igl.signed_distance(grid_points, mesh_vertices, mesh_faces)[0]
    mask = sdf < near_surface_band
    # Kept in step with build_interior_band_grid; this band is not sampled yet.
    _ = sdf < outer_band
    candidate_points = grid_points[mask]

    if candidate_points.shape[0] > max_input_points:
        random_indices = torch.randperm(candidate_points.shape[0])[:max_input_points].cpu().numpy()
        candidate_points = candidate_points[random_indices]

    print(
        f"Initializer retained {len(candidate_points)} point(s) from {int(mask.sum())} interior/near-surface candidates.",
        flush=True,
    )

    return candidate_points


def to_model_points(
    points: np.ndarray,
    mid_vals: torch.Tensor,
    range_vals: torch.Tensor,
    device: str,
) -> torch.Tensor:
    tensor_points = torch.tensor(points, dtype=torch.float32, device=device)
    tensor_points = (tensor_points - mid_vals) / range_vals
    tensor_points = tensor_points.detach()
    tensor_points.requires_grad = True
    return tensor_points


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    _, mesh_vertices, mesh_faces = load_mesh_arrays(args.mesh_path)
    sampled_points = sample_model_points(
        mesh_vertices=mesh_vertices,
        mesh_faces=mesh_faces,
        padding=args.padding,
        grid_spacing=(tuple(args.grid_spacing_xyz) if args.grid_spacing_xyz is not None else args.grid_spacing),
        min_samples_per_axis=args.min_grid_samples_per_axis,
        near_surface_band=args.near_surface_band,
        outer_band=args.outer_band,
        max_input_points=args.max_input_points,
    )

    mid_vals, range_vals = compute_normalization(mesh_vertices, args.device)
    input_points = to_model_points(sampled_points, mid_vals, range_vals, args.device)
    target_direction = torch.tensor(args.target_direction, dtype=torch.float32, device=args.device)

    if args.backend == "fourier":
        scalar_net = FourierFeatureNet(
            dim_hidden=SCALAR_NET_DIM_HIDDEN,
            num_layers=args.num_layers,
            num_features=args.fourier_num_features,
            sigma=args.fourier_sigma,
            seed=args.fourier_seed,
        ).to(args.device)
    else:
        scalar_net = initialize_scalar_net(
            device=args.device,
            dim_hidden=SCALAR_NET_DIM_HIDDEN,
            num_layers=args.num_layers,
            w0=args.w0,
            w0_initial=args.w0_initial,
        )

    warm_start_scalar_net_to_direction(
        scalar_net=scalar_net,
        sample_points=input_points,
        target_direction=target_direction,
        epochs=args.epochs,
        lr=args.lr,
        lr_warmup_epochs=args.lr_warmup_epochs,
        lr_warmup_factor=args.lr_warmup_factor,
        batch_size=args.batch_size,
        direction_weight=args.direction_weight,
        scalar_anchor_weight=args.scalar_anchor_weight,
        curvature_weight=args.curvature_weight,
        curvature_sample_fraction=args.curvature_sample_fraction,
        debug_progress=args.debug_progress,
    )

    output_path = build_initializer_checkpoint_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": scalar_net.state_dict(),
        "field_backend": args.backend,
        "model_name": args.model_name,
        "seed": args.seed,
        "mesh_path": args.mesh_path,
        "w0": args.w0,
        "w0_initial": args.w0_initial,
        "dim_hidden": SCALAR_NET_DIM_HIDDEN,
        "num_layers": args.num_layers,
        "target_direction": list(args.target_direction),
        "epochs": args.epochs,
        "lr": args.lr,
        "lr_warmup_epochs": args.lr_warmup_epochs,
        "lr_warmup_factor": args.lr_warmup_factor,
        "batch_size": args.batch_size,
        "padding": args.padding,
        "grid_spacing": args.grid_spacing_xyz if args.grid_spacing_xyz is not None else args.grid_spacing,
        "near_surface_band": args.near_surface_band,
        "outer_band": args.outer_band,
        "max_input_points": args.max_input_points,
        "curvature_weight": args.curvature_weight,
    }
    if args.backend == "fourier":
        checkpoint.update(
            {
                "fourier_num_features": args.fourier_num_features,
                "fourier_sigma": args.fourier_sigma,
                "fourier_seed": args.fourier_seed,
                "fourier_hidden_dim": SCALAR_NET_DIM_HIDDEN,
                "fourier_num_layers": args.num_layers,
            }
        )
    torch.save(checkpoint, output_path)
    print(f"Saved initializer checkpoint to {output_path}")


if __name__ == "__main__":
    main()
