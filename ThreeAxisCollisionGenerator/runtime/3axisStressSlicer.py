"""Experimental 3-axis slicing pipeline.

This script learns or loads a scalar field over a mesh and extracts contour
curves from that field. The contours act as candidate slicing paths for a
3-axis process.

The current implementation is intentionally still "research code", but it is
structured so the main stages are easier to inspect:
1. load the surface mesh
2. build a coarse domain near the mesh
3. choose surface/base constraints
4. optionally fine-tune the scalar field
5. evaluate the field on the mesh
6. extract contour curves
7. visualize the result

Future extension:
- collision-loss helpers live in `toolFieldOffsets.py`
- tool sampling helpers live in `toolSampling.py`

This variant replaces the surface-normal alignment term with a stress-driven
orthogonality term: at sampled stress points, the scalar-field gradient is
encouraged to be perpendicular to the local stress direction.
"""

from itertools import cycle
import json
from pathlib import Path

from runtimeEnv import configure_openmp_runtime

configure_openmp_runtime()

import igl
import numpy as np
import pyvista as pv
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

from sdfField import sdfModel
from scalarNetUtils import load_scalar_net_from_checkpoint
from siren_pytorch import SirenNet
from toolFieldOffsets import analyze_gated_translated_collisions
from toolFieldOffsets import compute_gated_translated_scalar_loss
from toolSampling import (
    build_tool_frame,
    orient_tool_points,
    sample_multi_rectangular_plane_points,
    sample_rectangular_plane_points,
    scale_tool_points,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "benchy"
SURFACE_MESH_PATH = "./inputShapes/TshapeBracketNew.obj"
VOLUME_NODE_PATH = "./inputShapes/TshapeBracketNew.node"
VOLUME_ELE_PATH = "./inputShapes/TshapeBracketNew.ele"
#CHECKPOINT_PATH = "./parameters_3axisTShapeHNC_Flsun2.pt"
CHECKPOINT_PATH = "./scalar_init_TshapeBracketNew_w4_w4z.pt"
#CHECKPOINT_PATH = "./parameters_3axisStressAlternating.pt"

#OUTPUT_CHECKPOINT_PATH = "./parameters_3axisTShapeHNC_Flsun3_y2.pt"
OUTPUT_CHECKPOINT_PATH = "./parameters_3axisTShapBambu_curved.pt"

SDF_GATE_CHECKPOINT_PATH = "./TshapeSDF.pt"
STRESS_FILE_PATH = "./stressField/stress_T_Full.txt"

BOUNDARY_PADDING = 1.0
GRID_SPACING = 1.0
NEAR_SURFACE_BAND = 0.5
OUTER_BAND = 1.5
MAX_INPUT_POINTS = 100000

UP_DIRECTION = np.array([[0.0, 0.0, 1.0]])
UPWARD_NORMAL_ANGLE_DEGREES = 20
BASE_Y_TOLERANCE = 1e-3

BATCH_SIZE = 500
LEARNING_RATE = 3e-5
STEP_SIZE = 100
STEP_GAMMA = 0.1
EPOCHS = 220
SAVE_EVERY = 1  
SCALAR_W0 = 4.0
SCALAR_W0_INITIAL = 4.0
ENABLE_COLLISION_LOSS = True
COLLISION_WEIGHT = 1e7
COLLISION_GATE_EPSILON_WORLD = 0.5
ENABLE_BASE_DIRECTION_LOSS = False
BASE_DIRECTION_WEIGHT = 10.0
ENABLE_BASE_SCALAR_LOSS = False
BASE_SCALAR_WEIGHT = 10.0

CONTOUR_COUNT = 100
CONTOUR_START_STEP = 0.2

TOOL_X_MIN = -40.0
TOOL_X_MAX = 50.0
TOOL_Z_MIN = -50.0
TOOL_Z_MAX = 10.0
TOOL_Y_VALUE = 10.0
TOOL_NUM_X = 20
TOOL_NUM_Z = 20
SECOND_TOOL_SURFACE_ENABLED = False
SECOND_TOOL_X_MIN = -7.0
SECOND_TOOL_X_MAX = 7.0
SECOND_TOOL_Z_MIN = -7.0
SECOND_TOOL_Z_MAX = 7.0
SECOND_TOOL_Y_VALUE = 8.0
SECOND_TOOL_NUM_X = 5
SECOND_TOOL_NUM_Z = 5
BUILD_DIRECTION_WORLD = (0.0, 0.0, 1.0)
IN_PLANE_U_WORLD = (1.0, 0.0, 0.0)
SHOW_LAYER_TOOL_VIS = True
SHOW_PLATFORM_IN_LAYER_VIS = True
SHOW_TOOL_IN_LAYER_VIS = True
PLATE_PADDING_WORLD = 20.0
DISPLAY_SCALAR_RANGE_PERCENT = 50.0
SLICER_VIS_CONFIG_PATH = "./slicerVisualizationConfig.json"
'''
TOOL_X_MIN = -25.0
TOOL_X_MAX = 25.0
TOOL_Z_MIN = -25.0
TOOL_Z_MAX = 25.0
TOOL_Y_VALUE = 10.0
TOOL_NUM_X = 20
TOOL_NUM_Z = 20
BUILD_DIRECTION_WORLD = (0.0, 0.0, 1.0)
IN_PLANE_U_WORLD = (1.0, 0.0, 0.0)
SHOW_LAYER_TOOL_VIS = True
SHOW_PLATFORM_IN_LAYER_VIS = True
SHOW_TOOL_IN_LAYER_VIS = True
PLATE_PADDING_WORLD = 20.0
DISPLAY_SCALAR_RANGE_PERCENT = 70.0
SLICER_VIS_CONFIG_PATH = "./slicerVisualizationConfig.json"
'''


def load_surface_mesh(path: str) -> pv.PolyData:
    """Load the mesh that will be sliced."""
    return pv.read(path)


def load_volume_mesh(node_path: str | None, ele_path: str | None) -> pv.DataSet | None:
    """Optionally load a paired volume mesh for visualization."""
    if not node_path or not ele_path:
        return None

    node_file = Path(node_path)
    ele_file = Path(ele_path)
    if not node_file.exists() or not ele_file.exists():
        print("Volume mesh files not found. Falling back to surface-mesh visualization.")
        return None

    try:
        return pv.read(str(ele_file))
    except Exception as exc:
        print(f"Failed to load volume mesh from {ele_file}: {exc}")
        return None


def extract_mesh_arrays(mesh: pv.PolyData) -> tuple[np.ndarray, np.ndarray]:
    """Return vertices and triangle faces in a form suitable for libigl."""
    vertices = np.array(mesh.points)
    faces = np.array(mesh.faces).reshape((-1, 4))[:, 1:4]
    return vertices, faces


def build_interior_band_grid(
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    padding: float,
    grid_spacing: float | tuple[float, float, float] | list[float] | np.ndarray,
    near_surface_band: float,
    outer_band: float,
    min_samples_per_axis: int = 1,
) -> tuple[pv.StructuredGrid, pv.PolyData, np.ndarray]:
    """Sample a coarse Cartesian domain and keep the interior plus a small outside band."""
    x_min, y_min, z_min = np.min(mesh_vertices, axis=0)
    x_max, y_max, z_max = np.max(mesh_vertices, axis=0)

    spacing = np.asarray(grid_spacing, dtype=np.float64)
    if spacing.ndim == 0:
        spacing = np.full(3, float(spacing))
    if spacing.shape != (3,) or np.any(spacing <= 0.0) or min_samples_per_axis < 1:
        raise ValueError("Grid spacing and minimum axis sample count must be positive.")

    def axis_values(lower: float, upper: float, axis: int) -> np.ndarray:
        values = np.arange(lower - padding, upper + padding, spacing[axis])
        if values.size >= min_samples_per_axis:
            return values
        return np.linspace(lower - padding, upper + padding, min_samples_per_axis)

    x_vals = axis_values(x_min, x_max, 0)
    y_vals = axis_values(y_min, y_max, 1)
    z_vals = axis_values(z_min, z_max, 2)

    grid_x, grid_y, grid_z = np.meshgrid(x_vals, y_vals, z_vals)
    mesh_grid = pv.StructuredGrid(grid_x.flatten(), grid_y.flatten(), grid_z.flatten())

    sdf = igl.signed_distance(mesh_grid.points, mesh_vertices, mesh_faces)[0]
    interior_band_mask = sdf < near_surface_band
    masked_grid = pv.PolyData(mesh_grid.points[interior_band_mask])

    # Preserved from the original script for future experiments.
    _ = sdf < outer_band

    return mesh_grid, masked_grid, sdf


def compute_base_points(masked_grid: pv.PolyData, y_tolerance: float) -> np.ndarray:
    """Pick the lowest points in the coarse interior domain as base constraints."""
    base_mask = masked_grid.points[:, 1] < (masked_grid.points[:, 1].min() + y_tolerance)
    return masked_grid.points[base_mask]


def compute_normalization(
    mesh_vertices: np.ndarray,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize points into a symmetric cube centered on the mesh."""
    x_min, y_min, z_min = np.min(mesh_vertices, axis=0)
    x_max, y_max, z_max = np.max(mesh_vertices, axis=0)

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


def to_model_points(points: np.ndarray, mid_vals: torch.Tensor, range_vals: torch.Tensor, device: str) -> torch.Tensor:
    """Move world-space points into the model's normalized coordinate system."""
    tensor_points = torch.tensor(points, dtype=torch.float32, device=device)
    tensor_points = (tensor_points - mid_vals) / range_vals
    tensor_points = tensor_points.detach()
    tensor_points.requires_grad = True
    return tensor_points


def build_tool_points(
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    y_value: float,
    num_x: int,
    num_z: int,
    build_direction_world,
    in_plane_u_world,
    range_vals: torch.Tensor,
    device: str,
    second_surface_enabled: bool = False,
    second_x_min: float | None = None,
    second_x_max: float | None = None,
    second_z_min: float | None = None,
    second_z_max: float | None = None,
    second_y_value: float | None = None,
    second_num_x: int | None = None,
    second_num_z: int | None = None,
) -> torch.Tensor:
    """Sample tool points in world units and scale them into model coordinates."""
    plane_specs = [
        {
            "x_min": x_min,
            "x_max": x_max,
            "z_min": z_min,
            "z_max": z_max,
            "y_value": y_value,
            "num_x": num_x,
            "num_z": num_z,
        }
    ]
    if second_surface_enabled:
        plane_specs.append(
            {
                "x_min": second_x_min,
                "x_max": second_x_max,
                "z_min": second_z_min,
                "z_max": second_z_max,
                "y_value": second_y_value,
                "num_x": second_num_x,
                "num_z": second_num_z,
            }
        )
    tool_points_local = sample_multi_rectangular_plane_points(plane_specs, device=device)
    tool_points_world = orient_tool_points(
        tool_points_local,
        build_direction_world=build_direction_world,
        in_plane_u_world=in_plane_u_world,
    )
    return scale_tool_points(tool_points_world, range_vals)


def build_tool_points_world(
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    y_value: float,
    num_x: int,
    num_z: int,
    build_direction_world,
    in_plane_u_world,
    root_point_world: np.ndarray,
    device: str | torch.device | None = None,
    second_surface_enabled: bool = False,
    second_x_min: float | None = None,
    second_x_max: float | None = None,
    second_z_min: float | None = None,
    second_z_max: float | None = None,
    second_y_value: float | None = None,
    second_num_x: int | None = None,
    second_num_z: int | None = None,
) -> np.ndarray:
    """Place the local rectangular tool at a chosen world-space root point."""
    plane_specs = [
        {
            "x_min": x_min,
            "x_max": x_max,
            "z_min": z_min,
            "z_max": z_max,
            "y_value": y_value,
            "num_x": num_x,
            "num_z": num_z,
        }
    ]
    if second_surface_enabled:
        plane_specs.append(
            {
                "x_min": second_x_min,
                "x_max": second_x_max,
                "z_min": second_z_min,
                "z_max": second_z_max,
                "y_value": second_y_value,
                "num_x": second_num_x,
                "num_z": second_num_z,
            }
        )
    tool_points_local = sample_multi_rectangular_plane_points(plane_specs, device=device)
    tool_points_world = orient_tool_points(
        tool_points_local,
        build_direction_world=build_direction_world,
        in_plane_u_world=in_plane_u_world,
    )
    root_point_world = torch.as_tensor(
        root_point_world,
        dtype=tool_points_world.dtype,
        device=tool_points_world.device,
    ).reshape(1, 3)
    return (tool_points_world + root_point_world).detach().to("cpu").numpy()


def build_plate_rectangle_world(
    mesh_points_world: np.ndarray,
    build_direction_world,
    in_plane_u_world,
    padding_world: float,
) -> pv.PolyData:
    """Build a plate rectangle under the mesh in the fixed `(u, v, n)` frame."""
    u, v, n = build_tool_frame(build_direction_world, in_plane_u_world)
    mesh_points = torch.tensor(mesh_points_world, dtype=torch.float32)

    a_coords = mesh_points @ u.cpu()
    b_coords = mesh_points @ v.cpu()
    h_coords = mesh_points @ n.cpu()

    a_min = a_coords.min().item() - padding_world
    a_max = a_coords.max().item() + padding_world
    b_min = b_coords.min().item() - padding_world
    b_max = b_coords.max().item() + padding_world
    h_plane = h_coords.min().item() - padding_world

    corners = torch.stack(
        (
            a_min * u.cpu() + b_min * v.cpu() + h_plane * n.cpu(),
            a_max * u.cpu() + b_min * v.cpu() + h_plane * n.cpu(),
            a_max * u.cpu() + b_max * v.cpu() + h_plane * n.cpu(),
            a_min * u.cpu() + b_max * v.cpu() + h_plane * n.cpu(),
        ),
        dim=0,
    ).numpy()
    faces = np.hstack(([4, 0, 1, 2, 3]))
    return pv.PolyData(corners, faces)


def select_lowest_point_on_surface(
    surface: pv.PolyData,
    build_direction_world,
) -> np.ndarray | None:
    """Select the lowest world-space point on a surface along the build direction."""
    if surface.n_points == 0:
        return None
    _, _, n = build_tool_frame(build_direction_world, IN_PLANE_U_WORLD)
    surface_points = np.asarray(surface.points)
    height_values = surface_points @ n.detach().to("cpu").numpy()
    root_index = int(np.argmin(height_values))
    return surface_points[root_index]


def load_stress_data(stress_file_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load stress sample points, directions, and magnitudes from text."""
    stress_rows = []
    with open(stress_file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            components = [float(component) for component in line.split()]
            if len(components) != 9:
                raise ValueError("Each stress-data row must contain exactly 9 values.")
            stress_rows.append(components)

    stress_array = np.array(stress_rows)
    stress_points = stress_array[:, 0:3]
    stress_directions = stress_array[:, 3:6]
    stress_magnitudes = stress_array[:, 6:9]
    return stress_points, stress_directions, stress_magnitudes


def farthest_point_sampling_indices(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Return indices chosen by farthest-point sampling."""
    num_points = points.shape[0]
    centroids = torch.zeros(int(num_samples), dtype=torch.long, device=points.device)
    distance = torch.ones(num_points, dtype=torch.float32, device=points.device) * 1e10
    farthest = torch.randint(0, num_points, (1,), dtype=torch.long, device=points.device)

    for i in range(num_samples):
        centroids[i] = farthest
        centroid = points[farthest]
        dist = torch.sum((points - centroid) ** 2, dim=1)
        distance = torch.minimum(distance, dist)
        farthest = torch.argmax(distance)

    return centroids


def prepare_stress_constraints(
    stress_file_path: str,
    mid_vals: torch.Tensor,
    range_vals: torch.Tensor,
    device: str,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load, normalize, and farthest-point-sample stress constraints."""
    stress_points_np, stress_directions_np, _ = load_stress_data(stress_file_path)

    mid_vals_cpu = mid_vals.detach().to("cpu")
    range_vals_cpu = range_vals.detach().to("cpu")

    stress_points_cpu = torch.tensor(stress_points_np, dtype=torch.float32)
    stress_points_cpu = (stress_points_cpu - mid_vals_cpu) / range_vals_cpu
    stress_points_cpu = stress_points_cpu.detach()

    stress_directions_cpu = torch.tensor(stress_directions_np, dtype=torch.float32)
    stress_directions_cpu = stress_directions_cpu / (torch.norm(stress_directions_cpu, dim=1, keepdim=True) + 1e-8)

    fps_count = min(int(num_samples), stress_points_cpu.shape[0])
    fps_indices = farthest_point_sampling_indices(stress_points_cpu, fps_count)

    sampled_stress_points = stress_points_cpu[fps_indices].detach().to(device)
    sampled_stress_directions = stress_directions_cpu[fps_indices].detach().to(device)

    del stress_points_cpu
    del stress_directions_cpu
    torch.cuda.empty_cache()

    sampled_stress_points.requires_grad = True
    return sampled_stress_points, sampled_stress_directions


def save_slicer_visualization_config(
    output_path: str,
    surface_mesh_path: str,
    scalar_checkpoint_path: str,
    tool_x_min: float,
    tool_x_max: float,
    tool_z_min: float,
    tool_z_max: float,
    tool_y_value: float,
    tool_num_x: int,
    tool_num_z: int,
    contour_count: int,
) -> None:
    """Save the standalone slicer-visualization configuration to JSON.

    The query/root point is intentionally omitted so it can be chosen later.
    """
    config = {
        "surface_mesh_path": surface_mesh_path,
        "scalar_checkpoint_path": scalar_checkpoint_path,
        "tool_x_min": tool_x_min,
        "tool_x_max": tool_x_max,
        "tool_z_min": tool_z_min,
        "tool_z_max": tool_z_max,
        "tool_y_value": tool_y_value,
        "tool_num_x": tool_num_x,
        "tool_num_z": tool_num_z,
        "contour_count": contour_count,
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def build_dataloaders(
    input_points: torch.Tensor,
    stress_points: torch.Tensor,
    stress_directions: torch.Tensor,
    base_points: torch.Tensor,
    batch_size: int,
):
    """Build finite loaders plus cycled training loaders for all constraints."""
    input_loader = DataLoader(input_points, batch_size=batch_size, shuffle=True)
    base_loader = DataLoader(base_points, batch_size=batch_size, shuffle=True)
    stress_dataset = TensorDataset(stress_points, stress_directions)
    stress_loader = DataLoader(stress_dataset, batch_size=batch_size, shuffle=True)

    input_loader_train = input_loader
    stress_loader_train = stress_loader
    base_loader_train = base_loader

    if len(stress_loader) > len(input_loader):
        input_loader_train = cycle(input_loader)
        base_loader_train = cycle(base_loader)
    elif len(stress_loader) < len(input_loader):
        stress_loader_train = cycle(stress_loader)
        base_loader_train = cycle(base_loader)

    return {
        "input_loader": input_loader,
        "stress_loader": stress_loader,
        "base_loader": base_loader,
        "input_loader_train": input_loader_train,
        "stress_loader_train": stress_loader_train,
        "base_loader_train": base_loader_train,
    }


def load_scalar_net(
    checkpoint_path: str | None,
    device: str,
    dim_hidden: int = 128,
    num_layers: int = 5,
    w0: float = 1.0,
    w0_initial: float = 1.0,
) -> SirenNet:
    """Create the scalar field network and optionally restore trained weights."""
    return load_scalar_net_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        dim_hidden=dim_hidden,
        num_layers=num_layers,
        w0=w0,
        w0_initial=w0_initial,
    )


def check_collision_for_root_points(
    scalar_net: SirenNet,
    root_points: torch.Tensor,
    tool_points_model: torch.Tensor,
    sdf_gate_model=None,
    root_scalars: torch.Tensor | None = None,
    gate_epsilon: float = 0.0,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
) -> dict[str, torch.Tensor]:
    """Check collision loss for a batch of root points in the current field state.

    This helper is intended for quick experiments where we want to test the
    current scalar field against a chosen set of root points, often the same
    points already used as the domain batch in training.
    """
    if root_scalars is None:
        root_scalars = scalar_net(root_points)["scalars"]

    return compute_gated_translated_scalar_loss(
        scalar_field=scalar_net,
        root_points=root_points,
        tool_points=tool_points_model,
        gating_field=sdf_gate_model,
        root_scalars=root_scalars,
        gate_epsilon=gate_epsilon,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
    )


def _extract_root_points_from_batch(batch) -> torch.Tensor:
    """Extract root-point tensors from a dataloader batch.

    If the batch is already a tensor, return it directly.
    If the batch is a tuple/list, use the first element, which matches common
    `(inputs, targets, ...)` loader conventions.
    """
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (tuple, list)):
        return batch[0]
    raise TypeError("Unsupported batch type for collision checking.")


def check_collision(
    scalar_net: SirenNet,
    root_source,
    tool_points_model: torch.Tensor,
    sdf_gate_model=None,
    gate_epsilon: float = 0.0,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
) -> dict[str, torch.Tensor]:
    """Check collision on either a root tensor or an iterable of root batches.

    For a tensor input, this delegates directly to `check_collision_for_root_points`.
    For a dataloader/iterable input, it evaluates each batch separately and
    combines the results using the exact masked-loss numerator and denominator.
    """
    if isinstance(root_source, torch.Tensor):
        return check_collision_for_root_points(
            scalar_net=scalar_net,
            root_points=root_source,
            tool_points_model=tool_points_model,
            sdf_gate_model=sdf_gate_model,
            gate_epsilon=gate_epsilon,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
        )

    total_masked_violation_sum = None
    total_num_included_points = None
    batch_outputs = []

    for i,batch in enumerate(root_source):
        print(f"{i}/{len(root_source)}")
        batch_root_points = _extract_root_points_from_batch(batch)
        batch_output = check_collision_for_root_points(
            scalar_net=scalar_net,
            root_points=batch_root_points,
            tool_points_model=tool_points_model,
            sdf_gate_model=sdf_gate_model,
            gate_epsilon=gate_epsilon,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
        )
        batch_outputs.append(batch_output)

        if total_masked_violation_sum is None:
            total_masked_violation_sum = batch_output["masked_violation_sum"]
            total_num_included_points = batch_output["num_included_points"]
        else:
            total_masked_violation_sum = total_masked_violation_sum + batch_output["masked_violation_sum"]
            total_num_included_points = total_num_included_points + batch_output["num_included_points"]

    if total_masked_violation_sum is None:
        zero = torch.tensor(0.0, device=tool_points_model.device, dtype=tool_points_model.dtype)
        return {
            "loss": zero,
            "masked_violation_sum": zero,
            "num_included_points": zero,
            "batch_outputs": [],
        }

    combined_loss = total_masked_violation_sum / (total_num_included_points + 1e-8)
    return {
        "loss": combined_loss,
        "masked_violation_sum": total_masked_violation_sum,
        "num_included_points": total_num_included_points,
        "batch_outputs": batch_outputs,
    }


def train_scalar_field(
    scalar_net: SirenNet,
    input_loader_train,
    stress_loader_train,
    base_loader_train,
    up_direction: torch.Tensor,
    epochs: int,
    learning_rate: float,
    step_size: int,
    step_gamma: float,
    tool_points_model: torch.Tensor | None = None,
    sdf_gate_model=None,
    collision_weight: float = 0.0,
    gate_epsilon: float = 0.0,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    checkpoint_path: str | None = None,
    save_every: int | None = None,
) -> None:
    """Fine-tune the scalar field used for slicing."""
    # This is the natural place to add collision-aware regularization later.
    # See `toolFieldOffsets.py` for translated tool-point difference and gated
    # collision-loss helpers that can be evaluated per batch.
    #
    # Potential collision-loss hook:
    # 1. sample tool offsets once outside the epoch loop with
    #    `sample_rectangular_plane_points(...)`
    # 2. scale those offsets with `scale_tool_points(...)`
    # 3. inside this batch loop, reuse `inp_outs["scalars"]` as `root_scalars`
    # 4. call `compute_gated_translated_scalar_loss(...)` on the chosen root
    #    points and add the returned loss term to `loss`
    optimizer = torch.optim.Adam(scalar_net.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=step_gamma)

    for epoch in range(epochs):
        epoch_loss = 0
        batch_sets = zip(input_loader_train, stress_loader_train, base_loader_train)

        for batch_input_points, (batch_stress_points, batch_stress_directions), batch_base_points in batch_sets:
            # `batch_input_points` is the most likely anchor set for a future
            # collision loss, since these are domain samples already expressed
            # in the same normalized coordinates as the scalar field.
            inp_outs = scalar_net(batch_input_points, compute_hess=False)
            inp_grads = inp_outs["grads"]
            collision_loss = None
            if tool_points_model is not None and collision_weight > 0.0:
                collision_out = compute_gated_translated_scalar_loss(
                    scalar_field=scalar_net,
                    root_points=batch_input_points,
                    tool_points=tool_points_model,
                    gating_field=sdf_gate_model,
                    root_scalars=inp_outs["scalars"],
                    gate_epsilon=gate_epsilon,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    z_min=z_min,
                    z_max=z_max,
                )
                collision_loss = collision_out["loss"]

            inp_norm = torch.sum(inp_grads * inp_grads, dim=1)
            inp_norm_error_1 = torch.relu(inp_norm - 1)
            inp_norm_error_2 = torch.relu(0.25 - inp_norm)
            norm_loss = 0.5 * torch.mean(inp_norm_error_1 * inp_norm_error_1)
            norm_loss += 0.5 * torch.mean(inp_norm_error_2 * inp_norm_error_2)

            stress_outs = scalar_net(batch_stress_points, compute_hess=False)
            stress_grads = stress_outs["grads"]
            stress_grads = stress_grads / (torch.norm(stress_grads, dim=1, keepdim=True) + 1e-5)
            stress_dot = torch.sum(stress_grads * batch_stress_directions, dim=1, keepdim=True)
            stress_loss = torch.mean(stress_dot * stress_dot)

            base_direction_loss = None
            base_scalar_loss = None
            if ENABLE_BASE_DIRECTION_LOSS or ENABLE_BASE_SCALAR_LOSS:
                base_outs = scalar_net(batch_base_points, compute_hess=False)
                if ENABLE_BASE_DIRECTION_LOSS:
                    base_grads = base_outs["grads"]
                    base_error = base_grads - up_direction
                    base_direction_loss = torch.mean(base_error * base_error)
                if ENABLE_BASE_SCALAR_LOSS:
                    base_scal_error = base_outs["scalars"]
                    base_scalar_loss = torch.mean(base_scal_error * base_scal_error)

            loss = stress_loss + 0.01 * norm_loss
            if collision_loss is not None:
                loss = loss + collision_weight * collision_loss
            if base_direction_loss is not None:
                loss = loss + BASE_DIRECTION_WEIGHT * base_direction_loss
            if base_scalar_loss is not None:
                loss = loss + BASE_SCALAR_WEIGHT * base_scalar_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss = loss.detach()

        scheduler.step()
        print(epoch_loss)

        if checkpoint_path and save_every and ((epoch + 1) % save_every == 0):
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": scalar_net.state_dict(),
            }

            torch.save(checkpoint, checkpoint_path)

    if checkpoint_path and epochs > 0:
        checkpoint = {
            "epoch": epochs,
            "model_state_dict": scalar_net.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)


def evaluate_scalar_field_on_dataset_points(
    dataset: pv.DataSet,
    scalar_net: SirenNet,
    mid_vals: torch.Tensor,
    range_vals: torch.Tensor,
    device: str,
) -> np.ndarray:
    """Evaluate the scalar field on every dataset point."""
    model_points = torch.tensor(dataset.points, dtype=torch.float32, device=device)
    model_points = (model_points - mid_vals) / range_vals
    model_points = model_points.detach()
    model_points.requires_grad = True
    return scalar_net(model_points)["scalars"].detach().to("cpu").numpy()


def visualize_colliding_root_points(
    mesh: pv.PolyData,
    scalar_net: SirenNet,
    root_source,
    tool_points_model: torch.Tensor,
    mid_vals: torch.Tensor,
    range_vals: torch.Tensor,
    sdf_gate_model=None,
    gate_epsilon: float = 0.0,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
) -> None:
    """Visualize colliding root points from either a tensor or a finite loader."""
    colliding_root_batches = []

    if isinstance(root_source, torch.Tensor):
        collision_out = analyze_gated_translated_collisions(
            scalar_field=scalar_net,
            root_points=root_source,
            tool_points=tool_points_model,
            gating_field=sdf_gate_model,
            gate_epsilon=gate_epsilon,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
        )
        colliding_root_batches.append(collision_out["colliding_root_points"])
    else:
        for batch in root_source:
            batch_root_points = _extract_root_points_from_batch(batch)
            collision_out = analyze_gated_translated_collisions(
                scalar_field=scalar_net,
                root_points=batch_root_points,
                tool_points=tool_points_model,
                gating_field=sdf_gate_model,
                gate_epsilon=gate_epsilon,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                z_min=z_min,
                z_max=z_max,
            )
            if collision_out["colliding_root_points"].shape[0] > 0:
                colliding_root_batches.append(collision_out["colliding_root_points"])

    if len(colliding_root_batches) == 0:
        print("No colliding root points found.")
        return

    colliding_root_points = torch.cat(colliding_root_batches, dim=0)
    colliding_root_points_world = colliding_root_points * range_vals + mid_vals
    colliding_root_cloud = pv.PolyData(colliding_root_points_world.detach().to("cpu").numpy())

    plotter = pv.Plotter()
    plotter.enable_anti_aliasing()
    plotter.add_axes()
    plotter.add_mesh(mesh, color="lightblue", opacity=0.2)
    plotter.add_mesh(colliding_root_cloud, color="red", point_size=10.0, render_points_as_spheres=True)
    plotter.show()


def visualize_colliding_tool_points_for_root(
    mesh: pv.PolyData,
    scalar_net: SirenNet,
    root_source,
    tool_points_model: torch.Tensor,
    mid_vals: torch.Tensor,
    range_vals: torch.Tensor,
    sdf_gate_model=None,
    gate_epsilon: float = 0.0,
    collision_root_index: int = 0,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
) -> None:
    """Visualize colliding translated tool points for one colliding root point."""
    selected_root_point = None
    selected_translated_points = None
    colliding_root_counter = 0

    if isinstance(root_source, torch.Tensor):
        root_batches = [root_source]
    else:
        root_batches = root_source

    for batch in root_batches:
        batch_root_points = _extract_root_points_from_batch(batch)
        collision_out = analyze_gated_translated_collisions(
            scalar_field=scalar_net,
            root_points=batch_root_points,
            tool_points=tool_points_model,
            gating_field=sdf_gate_model,
            gate_epsilon=gate_epsilon,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
        )

        root_collision_mask = collision_out["root_collision_mask"][:, 0]
        if not torch.any(root_collision_mask):
            continue

        translated_points = collision_out["translated_points"]
        point_collision_mask = collision_out["point_collision_mask"]

        colliding_indices = torch.nonzero(root_collision_mask, as_tuple=False).flatten()
        for root_idx in colliding_indices:
            if colliding_root_counter == collision_root_index:
                selected_root_point = batch_root_points[root_idx : root_idx + 1]
                selected_translated_points = translated_points[root_idx][point_collision_mask[root_idx, :, 0]]
                break
            colliding_root_counter += 1

        if selected_root_point is not None:
            break

    if selected_root_point is None or selected_translated_points is None or selected_translated_points.shape[0] == 0:
        print("No colliding tool points found for the requested colliding root index.")
        return

    selected_root_point_world = (selected_root_point * range_vals + mid_vals).detach().to("cpu").numpy()
    selected_translated_points_world = (selected_translated_points * range_vals + mid_vals).detach().to("cpu").numpy()

    root_cloud = pv.PolyData(selected_root_point_world)
    tool_cloud = pv.PolyData(selected_translated_points_world)

    plotter = pv.Plotter()
    plotter.enable_anti_aliasing()
    plotter.add_axes()
    plotter.add_mesh(mesh, color="lightblue", opacity=0.2)
    plotter.add_mesh(tool_cloud, color="orange", point_size=10.0, render_points_as_spheres=True)
    plotter.add_mesh(root_cloud, color="red", point_size=14.0, render_points_as_spheres=True)
    plotter.show()


def build_contour_dataset(dataset: pv.DataSet, model_scalars: np.ndarray) -> pv.DataSet:
    """Attach the scalar field to a copy of the source dataset."""
    dataset = dataset.copy(deep=True)
    dataset.point_data["scalar"] = model_scalars
    return dataset


def build_contour_values(model_scalars: np.ndarray, range_vals: torch.Tensor) -> np.ndarray:
    """Build the contour values used for the layer stack."""
    step = CONTOUR_START_STEP / range_vals[0].detach().to("cpu")
    scalar_min = float(model_scalars.min())
    scalar_max = float(model_scalars.max())
    scalar_range = scalar_max - scalar_min

    display_fraction = np.clip(DISPLAY_SCALAR_RANGE_PERCENT / 100.0, 0.0, 1.0)
    displayed_scalar_max = scalar_min + display_fraction * scalar_range
    contour_start = scalar_min + step
    contour_stop = min(displayed_scalar_max, scalar_max + 1e-3)

    if contour_stop <= contour_start:
        return np.array([contour_start], dtype=np.float64)
    return np.linspace(contour_start, contour_stop, CONTOUR_COUNT)


def extract_contours(contour_dataset: pv.DataSet, contour_values: np.ndarray) -> pv.PolyData:
    """Extract the full contour stack from the prepared contour dataset."""
    return contour_dataset.contour(contour_values, scalars="scalar")


def extract_last_iso_surface(contour_dataset: pv.DataSet, contour_values: np.ndarray) -> pv.PolyData | None:
    """Extract only the last iso-surface from the contour stack."""
    if contour_values.size == 0:
        return None
    last_surface = contour_dataset.contour([float(contour_values[-1])], scalars="scalar")
    if last_surface.n_points == 0 or last_surface.n_cells == 0:
        return None
    return last_surface


def visualize_contours(
    mesh: pv.PolyData,
    contours: pv.PolyData,
    volume_mesh: pv.DataSet | None = None,
    plate_mesh: pv.PolyData | None = None,
    last_surface: pv.PolyData | None = None,
    root_point_world: np.ndarray | None = None,
    tool_points_world: np.ndarray | None = None,
) -> None:
    """Display the contour stack and optionally the plate, last surface, and tool placement."""
    plotter = pv.Plotter()
    plotter.enable_anti_aliasing()
    plotter.add_axes()
    plotter.add_mesh(contours, line_width=4)
    plotter.enable_parallel_projection()
    if volume_mesh is not None:
        plotter.add_mesh(volume_mesh, color="lightgray", opacity=0.08, show_edges=True)
        plotter.add_mesh(mesh, color="lightblue", opacity=0.06)
    else:
        plotter.add_mesh(mesh, color="lightblue", opacity=0.1)
    if plate_mesh is not None:
        plotter.add_mesh(plate_mesh, color="lightgreen", opacity=0.35)
    if last_surface is not None:
        plotter.add_mesh(last_surface, color="white", opacity=0.2)
    if tool_points_world is not None:
        tool_cloud = pv.PolyData(tool_points_world)
        plotter.add_mesh(tool_cloud, color="orange", point_size=8.0, render_points_as_spheres=True)
    if root_point_world is not None:
        root_cloud = pv.PolyData(np.asarray(root_point_world).reshape(1, 3))
        plotter.add_mesh(root_cloud, color="purple", point_size=14.0, render_points_as_spheres=True)
    plotter.show()


def main() -> None:
    """Run the full experimental slicing workflow."""
    _ = MODEL_NAME, OUTPUT_CHECKPOINT_PATH

    surface_mesh = load_surface_mesh(SURFACE_MESH_PATH)
    volume_mesh = load_volume_mesh(VOLUME_NODE_PATH, VOLUME_ELE_PATH)
    mesh_vertices, mesh_faces = extract_mesh_arrays(surface_mesh)

    _, masked_grid, _ = build_interior_band_grid(
        mesh_vertices,
        mesh_faces,
        padding=BOUNDARY_PADDING,
        grid_spacing=GRID_SPACING,
        near_surface_band=NEAR_SURFACE_BAND,
        outer_band=OUTER_BAND,
    )

    base_points_np = compute_base_points(masked_grid, BASE_Y_TOLERANCE)

    random_indices = torch.randperm(masked_grid.points.shape[0])
    sampled_points = masked_grid.points[random_indices][:MAX_INPUT_POINTS]

    mid_vals, range_vals = compute_normalization(mesh_vertices, DEVICE)
    sdf_scalar_scale = range_vals[0]
    collision_gate_epsilon_model = COLLISION_GATE_EPSILON_WORLD / sdf_scalar_scale

    input_points = to_model_points(sampled_points, mid_vals, range_vals, DEVICE)
    stress_points, stress_directions = prepare_stress_constraints(
        STRESS_FILE_PATH,
        mid_vals=mid_vals,
        range_vals=range_vals,
        device=DEVICE,
        num_samples=input_points.shape[0],
    )
    base_points = to_model_points(base_points_np, mid_vals, range_vals, DEVICE)

    dataloaders = build_dataloaders(
        input_points,
        stress_points,
        stress_directions,
        base_points,
        batch_size=BATCH_SIZE,
    )
    # If we add tool collision constraints, the tool offsets should be sampled
    # and scaled in `toolSampling.py`, then compared here against the learned
    # scalar field using the helpers in `toolFieldOffsets.py`.
    # Potential setup hook:
    # tool_points_world = sample_rectangular_plane_points(...)
    # tool_points_model = scale_tool_points(tool_points_world, range_vals)
    # sdf_gate_model = sdfModel(device=DEVICE, model_load_path=SDF_GATE_CHECKPOINT_PATH)
    tool_points = None
    sdf_gate_model = None
    collision_weight = 0.0
    if ENABLE_COLLISION_LOSS:
        tool_points = build_tool_points(
            TOOL_X_MIN,
            TOOL_X_MAX,
            TOOL_Z_MIN,
            TOOL_Z_MAX,
            TOOL_Y_VALUE,
            TOOL_NUM_X,
            TOOL_NUM_Z,
            BUILD_DIRECTION_WORLD,
            IN_PLANE_U_WORLD,
            range_vals,
            DEVICE,
            second_surface_enabled=SECOND_TOOL_SURFACE_ENABLED,
            second_x_min=SECOND_TOOL_X_MIN,
            second_x_max=SECOND_TOOL_X_MAX,
            second_z_min=SECOND_TOOL_Z_MIN,
            second_z_max=SECOND_TOOL_Z_MAX,
            second_y_value=SECOND_TOOL_Y_VALUE,
            second_num_x=SECOND_TOOL_NUM_X,
            second_num_z=SECOND_TOOL_NUM_Z,
        )
        sdf_gate_model = sdfModel(device=DEVICE, model_load_path=SDF_GATE_CHECKPOINT_PATH)
        collision_weight = COLLISION_WEIGHT

    scalar_net = load_scalar_net(
        CHECKPOINT_PATH,
        DEVICE,
        w0=SCALAR_W0,
        w0_initial=SCALAR_W0_INITIAL,
    )
    up_direction = torch.tensor(UP_DIRECTION, dtype=torch.float32, device=DEVICE)
    up_direction = up_direction / (torch.norm(up_direction, dim=1, keepdim=True) + 1e-8)
    _ = tool_points, sdf_gate_model

    x_min, y_min, z_min = torch.min(input_points, dim=0)[0]
    x_max, y_max, z_max = torch.max(input_points, dim=0)[0]

    train_scalar_field(
        scalar_net,
        dataloaders["input_loader_train"],
        dataloaders["stress_loader_train"],
        dataloaders["base_loader_train"],
        up_direction=up_direction,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        step_size=STEP_SIZE,
        step_gamma=STEP_GAMMA,
        tool_points_model=tool_points,
        sdf_gate_model=sdf_gate_model,
        collision_weight=collision_weight,
        gate_epsilon=collision_gate_epsilon_model,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
        checkpoint_path=OUTPUT_CHECKPOINT_PATH,
        save_every=SAVE_EVERY,
    )
    save_slicer_visualization_config(
        output_path=SLICER_VIS_CONFIG_PATH,
        surface_mesh_path=SURFACE_MESH_PATH,
        scalar_checkpoint_path=OUTPUT_CHECKPOINT_PATH,
        tool_x_min=TOOL_X_MIN,
        tool_x_max=TOOL_X_MAX,
        tool_z_min=TOOL_Z_MIN,
        tool_z_max=TOOL_Z_MAX,
        tool_y_value=TOOL_Y_VALUE,
        tool_num_x=TOOL_NUM_X,
        tool_num_z=TOOL_NUM_Z,
        contour_count=CONTOUR_COUNT,
    )
    x_min, y_min, z_min = torch.min(input_points, dim=0)[0]
    x_max, y_max, z_max = torch.max(input_points, dim=0)[0]
    # visualize_colliding_root_points(
    #     mesh=surface_mesh,
    #     scalar_net=scalar_net,
    #     root_source=dataloaders["input_loader"],
    #     tool_points_model=tool_points,
    #     mid_vals=mid_vals,
    #     range_vals=range_vals,
    #     sdf_gate_model=sdf_gate_model,
    #     x_min=x_min,
    #     x_max=x_max,
    #     y_min=y_min,
    #     y_max=y_max,
    #     z_min=z_min,
    #     z_max=z_max,
    # )
    # Potential detailed collision-visualization hook:
    # visualize_colliding_tool_points_for_root(
    #     mesh=surface_mesh,
    #     scalar_net=scalar_net,
    #     root_source=dataloaders["input_loader"],
    #     tool_points_model=tool_points,
    #     mid_vals=mid_vals,
    #     range_vals=range_vals,
    #     sdf_gate_model=sdf_gate_model,
    #     collision_root_index=0,
    #     x_min=x_min,
    #     x_max=x_max,
    #     y_min=y_min,
    #     y_max=y_max,
    #     z_min=z_min,
    #     z_max=z_max,
    # )
    contour_source = volume_mesh if volume_mesh is not None else surface_mesh
    model_scalars = evaluate_scalar_field_on_dataset_points(
        contour_source,
        scalar_net,
        mid_vals,
        range_vals,
        DEVICE,
    )
    contour_dataset = build_contour_dataset(contour_source, model_scalars)
    contour_values = build_contour_values(model_scalars, range_vals)
    contours = extract_contours(contour_dataset, contour_values)

    plate_mesh = None
    last_surface = None
    root_point_world = None
    tool_points_world = None

    if SHOW_LAYER_TOOL_VIS:
        if SHOW_PLATFORM_IN_LAYER_VIS:
            plate_mesh = build_plate_rectangle_world(
                mesh_points_world=surface_mesh.points,
                build_direction_world=BUILD_DIRECTION_WORLD,
                in_plane_u_world=IN_PLANE_U_WORLD,
                padding_world=PLATE_PADDING_WORLD,
        )
        last_surface = extract_last_iso_surface(contour_dataset, contour_values)
        if last_surface is not None and SHOW_TOOL_IN_LAYER_VIS:
            root_point_world = select_lowest_point_on_surface(last_surface, BUILD_DIRECTION_WORLD)
            if root_point_world is not None:
                tool_points_world = build_tool_points_world(
                    TOOL_X_MIN,
                    TOOL_X_MAX,
                    TOOL_Z_MIN,
                    TOOL_Z_MAX,
                    TOOL_Y_VALUE,
                    TOOL_NUM_X,
                    TOOL_NUM_Z,
                    BUILD_DIRECTION_WORLD,
                    IN_PLANE_U_WORLD,
                    root_point_world=root_point_world,
                    device=DEVICE,
                    second_surface_enabled=SECOND_TOOL_SURFACE_ENABLED,
                    second_x_min=SECOND_TOOL_X_MIN,
                    second_x_max=SECOND_TOOL_X_MAX,
                    second_z_min=SECOND_TOOL_Z_MIN,
                    second_z_max=SECOND_TOOL_Z_MAX,
                    second_y_value=SECOND_TOOL_Y_VALUE,
                    second_num_x=SECOND_TOOL_NUM_X,
                    second_num_z=SECOND_TOOL_NUM_Z,
                )

    visualize_contours(
        surface_mesh,
        contours,
        volume_mesh=volume_mesh,
        plate_mesh=plate_mesh,
        last_surface=last_surface,
        root_point_world=root_point_world,
        tool_points_world=tool_points_world,
    )


if __name__ == "__main__":
    main()
