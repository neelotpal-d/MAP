"""Convex trial using a sampled within-layer gradient-ratio regularizer.

This reuses the stress slicer's contour and collision helpers, but the
optimization itself is not stress-driven. It adds:
1. surface-quality constraints on upward-inclined face centers
2. optional base fixing over all domain points within a height band
3. gradient-magnitude regularization

The file is intentionally separate from `3axisStressSlicer.py`.
"""

from __future__ import annotations

import heapq
from itertools import cycle
import importlib.util
import json
from pathlib import Path
from datetime import datetime, timezone

from runtimeEnv import configure_openmp_runtime

configure_openmp_runtime()

import numpy as np
import pyvista as pv
import torch
from scipy.spatial import cKDTree
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

from fourierFeatureNet import FourierFeatureNet
from scalarNetUtils import AffineResidualScalarField
from layerThicknessControl import (
    BandGradientReference,
    sampled_gradient_ratio_loss,
)
from toolFieldOffsets import compute_gated_translated_scalar_loss


# This copied configuration lives below ``experiments/`` rather than beside the
# shared base slicer.
BASE_MODULE_PATH = Path(__file__).resolve().parents[2] / "3axisStressSlicer.py"

DEVICE = "cuda"
SURFACE_MESH_PATH = "./inputShapes/convex.obj"
VOLUME_NODE_PATH = "./inputShapes/convex.node"
VOLUME_ELE_PATH = "./inputShapes/convex.ele"
CHECKPOINT_PATH = "./scalar_init_convex_w10_w10_curv1em05.pt"
#CHECKPOINT_PATH = "./experiments/convex_constraint_warmup/parameters_convex_warmup_only.pt"
OUTPUT_CHECKPOINT_PATH = "./experiments/convex_hard_collision/parameters_convex_hard_collision.pt"
SDF_GATE_CHECKPOINT_PATH = "./convexSDF.pt"



BOUNDARY_PADDING = 1.1
GRID_SPACING = 2.0
MIN_INPUT_GRID_SAMPLES_PER_AXIS = 1
NEAR_SURFACE_BAND = 1.0
OUTER_BAND = 1.5
MAX_INPUT_POINTS = 150000
# When the candidate grid is denser than MAX_INPUT_POINTS, redraw a fresh
# random input regularization subset every epoch instead of reusing one cap.
RESAMPLE_INPUT_REGULARIZATION_POINTS_EACH_EPOCH = False
INPUT_REGULARIZATION_POINTS_PER_EPOCH = MAX_INPUT_POINTS
INPUT_REGULARIZATION_RESAMPLING_SEED = 2026

UP_DIRECTION = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
# Manual preserved patches may be selected on downward-facing geometry. Flip a
# whole connected group's target normals when its aggregate direction is below UP_DIRECTION.
ORIENT_PRESERVED_GROUP_NORMALS_TOWARD_UP = True
SURFACE_NORMAL_MAX_ANGLE_DEGREES = 40.0
SURFACE_HEIGHT_MIN_WORLD = 23
SURFACE_HEIGHT_MAX_WORLD = 76.2
SURFACE_REGION_AREA_RATIO_MIN = 0.1
HEIGHT_SELECTION_MODE = "face-center"
BASE_HEIGHT_WORLD = 1.0
BASE_NORMAL_MAX_ANGLE_DEGREES = 40.0
SHOW_SURFACE_QUALITY_POINTS_BEFORE_TRAINING = True
SHOW_SURFACE_QUALITY_NORMALS_BEFORE_TRAINING = True
SURFACE_QUALITY_NORMAL_ARROW_STRIDE = 25
SURFACE_QUALITY_NORMAL_ARROW_LENGTH_WORLD = 1.0
SHOW_SURFACE_CONSTRAINT_BATCHES_BEFORE_TRAINING = True
SHOW_ALL_LOSS_EVALUATION_POINTS_BEFORE_TRAINING = True
SHOW_SURFACE_EDGE_PAIRS_BEFORE_TRAINING = False
SHOW_INPUT_REGULARIZATION_BATCHES_BEFORE_TRAINING = True
INPUT_REGULARIZATION_VIS_BATCH_COUNT = 8
INPUT_REGULARIZATION_VIS_POINT_SIZE = 4.0
ENABLE_INTERACTIVE_MANUAL_SURFACE_REGIONS = True
MANUAL_SURFACE_FLOOD_MAX_NEIGHBOR_NORMAL_ANGLE_DEGREES = 1.0
MANUAL_SURFACE_ERASE_RADIUS_WORLD = 2.0
MANUAL_SURFACE_REGIONS_PATH = "./convex_manual_surface_regions.json"
LOAD_MANUAL_SURFACE_REGIONS = False
SAVE_MANUAL_SURFACE_REGIONS = False
SURFACE_CONSTRAINT_METADATA_PATH = "./convex_surface_constraint_regions.json"
SAVE_SURFACE_CONSTRAINT_METADATA = True
# Optional pipeline hooks. They keep this standalone script's original vertex
# constraints unless an orchestrator explicitly supplies resampling callbacks.
SURFACE_CONSTRAINT_RESAMPLER = None
SURFACE_EDGE_PAIR_RESAMPLER = None
BASE_CONSTRAINT_RESAMPLER = None
PRE_TRAIN_CONSTRAINT_PREVIEW_CALLBACK = None
STRENGTH_LOSS_PROVIDER = None
# A pipeline may offer interactive adjustment of automatic face-selection
# thresholds before this standalone script creates the constraint dataset.
AUTOMATIC_SELECTION_CONFIGURATOR = None

BATCH_SIZE = 500
LEARNING_RATE = 1e-5
# Start below the configured learning rate for the first epoch, then restore
# LEARNING_RATE before the second epoch. Set the factor to 1.0 to disable.
LEARNING_RATE_WARMUP_EPOCHS = 15
LEARNING_RATE_WARMUP_FACTOR = 1e-5
# After the constant low-LR warm-up, optionally linearly ramp to LEARNING_RATE.
# Zero preserves the legacy immediate restoration behavior.
LEARNING_RATE_RAMP_EPOCHS = 0
STEP_SIZE = 250
STEP_GAMMA = 0.1
EPOCHS = 250
SAVE_EVERY = 1
DEBUG_PRINT_LOSS_BREAKDOWN = True
LOSS_BREAKDOWN_EVERY_EPOCHS = 1
PRINT_INITIAL_LOSS_GRADIENT_CONTRIBUTIONS = True
INITIAL_LOSS_GRADIENT_DIAGNOSTIC_EPOCHS = 15
PRINT_INITIALIZER_BASE_DIRECTION_DIAGNOSTIC = True
SCALAR_W0 = 10.0
SCALAR_W0_INITIAL = 10.0
SCALAR_NUM_LAYERS = 5
# Keep "siren" as the default so existing checkpoints remain compatible.
# Set to "fourier" to train the experimental Fourier-feature field below.
FIELD_BACKEND = "siren"
# Optional exact normalized-coordinate affine baseline. The pipeline sets this
# to [0, 1, 0] for its y-field residual experiment.
AFFINE_RESIDUAL_DIRECTION = None
AFFINE_RESIDUAL_INITIAL_SCALE = 0.0
FOURIER_NUM_FEATURES = 128
FOURIER_SIGMA = 4.0
FOURIER_SEED = 0
FOURIER_HIDDEN_DIM = 128
FOURIER_NUM_LAYERS = 5

ENABLE_SURFACE_DIRECTION_LOSS = True
SURFACE_DIRECTION_WEIGHT = 100.0
# On side surfaces, suppress only the tangent component that turns away from
# projected UP_DIRECTION. A local batch is accepted only if its parameter
# gradient does not substantially oppose the current combined objective.
ENABLE_TANGENT_TWIST_LOSS = True
TANGENT_TWIST_WEIGHT = 1.0
TANGENT_TWIST_MIN_PROJECTED_UP_NORM = 0.1
TANGENT_TWIST_MIN_GRADIENT_COSINE = -0.1
TANGENT_TWIST_COSINE_CHECK_EVERY_BATCHES = 10
ENABLE_SUPPORT_FREE_SURFACE_LOSS = False
SUPPORT_FREE_SURFACE_WEIGHT = 200.0
SUPPORT_FREE_SURFACE_EXTRA_ANGLE_DEGREES = 60.0
# Heights are measured from the model's lowest point along UP_DIRECTION.
# Leave either bound as None to keep that side unbounded.
SUPPORT_FREE_SURFACE_HEIGHT_MIN_WORLD = 5
SUPPORT_FREE_SURFACE_HEIGHT_MAX_WORLD = 12
ENABLE_SURFACE_GROUP_SCALAR_LOSS = True
SURFACE_GROUP_SCALAR_WEIGHT = 50.0
# Zero applies group scalar consistency immediately; use N to start it at the
# zero-based epoch N after direction/base constraints have settled.
SURFACE_GROUP_SCALAR_START_EPOCH = 20
SURFACE_SCALAR_RAMP_EPOCHS = 0
# "l1" is outlier-tolerant; "l2" emphasizes larger within-group scalar gaps.
SURFACE_GROUP_SCALAR_LOSS_TYPE = "l2"
# Convert group scalar differences to estimated world distances using the
# alignment-weighted mean gradient magnitude. This denominator is trainable.
SURFACE_GROUP_GRADIENT_DENOMINATOR_EPSILON = 1e-4
SURFACE_SCALAR_DIVIDE_BY_GRADIENT = True
# Optional A/B test: retain distance normalization while preventing scalar
# consistency from changing gradient magnitude through its denominator.
SURFACE_SCALAR_GRADIENT_DENOMINATOR_DETACH = False
SURFACE_GROUP_ALIGNMENT_MAX_ANGLE_DEGREES = 3.0
# Smoothness of the alignment-based scalar-consistency weighting in dot-product
# space. Smaller values make this approach the previous hard aligned-point mask.
SURFACE_GROUP_ALIGNMENT_WEIGHT_TEMPERATURE = 0.02
SURFACE_GROUP_BATCH_SIZE = 8
# Balance the normal-alignment batches by the angle between their target
# surface normal and UP_DIRECTION. This affects only the pointwise surface
# direction loss; whole connected groups remain intact for group-scalar loss.
SURFACE_NORMAL_STRATIFICATION_ENABLED = False
SURFACE_NORMAL_STRATIFICATION_BIN_COUNT = 4
SURFACE_NORMAL_STRATIFICATION_MAX_ANGLE_DEGREES = SURFACE_NORMAL_MAX_ANGLE_DEGREES
SURFACE_NORMAL_STRATIFICATION_SEED = 12_345
ENABLE_SURFACE_GROUP_ANGLE_CURRICULUM = False
SURFACE_GROUP_ANGLE_CURRICULUM_BIN_COUNT = 4
SURFACE_GROUP_ANGLE_CURRICULUM_CHECK_EVERY_EPOCHS = 10
SURFACE_GROUP_ANGLE_CURRICULUM_LOSS_THRESHOLD = 0.02
SURFACE_GROUP_ANGLE_CURRICULUM_EMA_DECAY = 0.8
SURFACE_GROUP_ANGLE_CURRICULUM_REQUIRED_CHECKS = 2
SURFACE_GROUP_ANGLE_CURRICULUM_EVAL_BATCH_SIZE = 5000
ENABLE_SURFACE_EDGE_SCALAR_LOSS = False
# When True, pairwise edge consistency replaces the group-wide scalar loss.
SURFACE_EDGE_REPLACES_GROUP_SCALAR_LOSS = False
SURFACE_EDGE_SCALAR_WEIGHT = 1.0
# Start local edge equality only after the direction field has settled.
SURFACE_EDGE_SCALAR_START_EPOCH = 20
SURFACE_EDGE_BATCH_SIZE = 5000
SURFACE_EDGE_USE_SAMPLED_NEIGHBORS = False
SURFACE_EDGE_NEIGHBOR_RADIUS_WORLD = 0.0
SURFACE_EDGE_NEIGHBOR_NORMAL_ANGLE_DEGREES = 20.0
SURFACE_EDGE_LONG_RANGE_PAIRS_PER_GROUP = 0
SURFACE_EDGE_LONG_RANGE_MIN_DISTANCE_WORLD = 0.0
SURFACE_EDGE_LONG_RANGE_SEED = 1234
SURFACE_EDGE_ALIGNMENT_MAX_ANGLE_DEGREES = 6.0
SURFACE_EDGE_ALIGNMENT_WEIGHT_TEMPERATURE = 0.002
SURFACE_EDGE_POSITIVE_WEIGHT_TEMPERATURE = 0.05
SURFACE_EDGE_ALIGNMENT_PAIR_WEIGHT = "product"
ENABLE_BASE_DIRECTION_LOSS = True
BASE_DIRECTION_WEIGHT = 100.0
BASE_DIRECTION_PARALLEL_ONLY = False
ENABLE_BASE_SCALAR_LOSS = False
BASE_SCALAR_WEIGHT = 1.0
# Measure whether the weighted base objective reinforces or conflicts with all
# other losses. The diagnostic observes one batch and never changes its update.
PRINT_BASE_GRADIENT_INTERFERENCE = True
BASE_GRADIENT_INTERFERENCE_EVERY_EPOCHS = 1
# First establish an approximately unit gradient field, then relax to a
# bounded [0.5, 1.0] magnitude range while retaining modest band consistency.
ENABLE_GRADIENT_NORM_CURRICULUM = False
GRADIENT_NORM_UNIT_EPOCHS = 20
GRADIENT_NORM_UNIT_WEIGHT = 10.0
GRADIENT_NORM_LATE_RANGE_WEIGHT = 5.0
GRADIENT_NORM_LATE_RATIO_WEIGHT = 5.0
GRADIENT_NORM_MIN_SQUARED = 0.25
GRADIENT_NORM_MAX_SQUARED = 1.0
GRADIENT_BAND_RATIO_WEIGHT = 5.0
GRADIENT_BAND_COUNT = 8
# q is ||grad f||^2. A squared-ratio cap of four permits a factor-of-two
# variation in gradient magnitude within a band.
GRADIENT_BAND_MAX_SQUARED_RATIO = 4.0
# Retained for the separate legacy comparison runner's explicit loss override.
GRADIENT_BAND_SOFT_EXTREMA_TEMPERATURE = 0.1
GRADIENT_BAND_LOG_EPSILON = 1e-8
# Ratio control is scale invariant. An absolute floor is optional and disabled.
GRADIENT_BAND_SQUARED_FLOOR = 0.25
GRADIENT_BAND_FLOOR_WEIGHT = 0.5
ENABLE_BALANCED_NORM_BATCHES = False
NORM_POOL_SIZE = 20_000
NORM_POINTS_PER_BAND = 1_000
SHOW_NORM_CONTROL_BATCHES_EACH_EPOCH = False
NORM_CONTROL_BATCH_VIS_MAX_BATCHES = 32
_LAST_NORM_CONTROL_SAMPLE = None
ENABLE_INPUT_CURVATURE_LOSS = True
# Exact Hessians are evaluated only on this random fraction of each input batch.
INPUT_CURVATURE_WEIGHT = 1e-3
INPUT_CURVATURE_SAMPLE_FRACTION = 0.2
INPUT_CURVATURE_MODE = 'raw-hessian'
INPUT_CURVATURE_RADIUS_MM = 35.0
ENABLE_BUILD_DIRECTION_SLOPE_LOSS = False
BUILD_DIRECTION_SLOPE_WEIGHT = 10.0
BUILD_DIRECTION_SLOPE_MAX_ANGLE_DEGREES = 60.0

ENABLE_COLLISION_LOSS = True
COLLISION_WEIGHT = 1e6
COLLISION_GATE_EPSILON_WORLD = 0.5
# "global_mean" preserves the original loss. "per_tool_topk" selects the
# strongest violations independently for every root and every tool surface.
COLLISION_LOSS_REDUCTION = "per_tool_topk"#"global_mean"
COLLISION_TOPK_FRACTION = 0.30
NORMALIZE_COLLISION_BY_GRADIENT = False
COLLISION_GRADIENT_EPSILON = 1e-6
COLLISION_TOOL_WEIGHTS = (1.0, 1.0)
# Keep the collision quadrature points fixed while optimizing.  Regenerating
# them per epoch changes the objective and makes the collision loss noisy.
PERTURB_TOOL_POINTS_EACH_EPOCH = True
TOOL_PERTURB_SEED = 0


def surface_scalar_loss_ramp_factor(epoch: int) -> float:
    """Return the scalar-consistency multiplier for a zero-based epoch."""
    if epoch < SURFACE_GROUP_SCALAR_START_EPOCH:
        return 0.0
    if SURFACE_SCALAR_RAMP_EPOCHS <= 0:
        return 1.0
    return min(1.0, (epoch - SURFACE_GROUP_SCALAR_START_EPOCH + 1) / SURFACE_SCALAR_RAMP_EPOCHS)

CONTOUR_COUNT = 200
CONTOUR_START_STEP = 0.2
DISPLAY_SCALAR_RANGE_PERCENT = 60.0
SLICER_VIS_CONFIG_PATH = "./slicerVisualizationConfig.json"

# TOOL_X_MIN = -20.0
# TOOL_X_MAX = 20.0
# TOOL_Z_MIN = -20.0
# TOOL_Z_MAX = 20.0
# TOOL_Y_VALUE = 8.0
# TOOL_NUM_X = 20
# TOOL_NUM_Z = 20
TOOL_X_MIN = -28.5
TOOL_X_MAX = 28.5
TOOL_Z_MIN = -61.0
TOOL_Z_MAX = 15.6
TOOL_Y_VALUE = 3.60
TOOL_NUM_X = 20
TOOL_NUM_Z = 22
SECOND_TOOL_SURFACE_ENABLED = False
SECOND_TOOL_X_MIN = -20.0
SECOND_TOOL_X_MAX = 28.0
SECOND_TOOL_Z_MIN = -20.0
SECOND_TOOL_Z_MAX = 5.0
SECOND_TOOL_Y_VALUE = 2.55
SECOND_TOOL_NUM_X = 10
SECOND_TOOL_NUM_Z = 10
BUILD_DIRECTION_WORLD = (0.0, 1.0, 0.0)
IN_PLANE_U_WORLD = (1.0, 0.0, 0.0)
SHOW_LAYER_TOOL_VIS = True
SHOW_FINAL_CONTOUR_VIS = True
SHOW_PLATFORM_IN_LAYER_VIS = True
SHOW_TOOL_IN_LAYER_VIS = True
PLATE_PADDING_WORLD = 20.0


def load_base_module():
    spec = importlib.util.spec_from_file_location("stress_slicer_base_surface_quality", BASE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load base module from {BASE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_scalar_field(base):
    """Load the selected field backend and reject incompatible checkpoints."""
    if FIELD_BACKEND == "siren":
        scalar_net = base.load_scalar_net(
            CHECKPOINT_PATH,
            DEVICE,
            num_layers=SCALAR_NUM_LAYERS,
            w0=SCALAR_W0,
            w0_initial=SCALAR_W0_INITIAL,
        )
        if AFFINE_RESIDUAL_DIRECTION is not None and not isinstance(scalar_net, AffineResidualScalarField):
            scalar_net = AffineResidualScalarField(
                scalar_net, AFFINE_RESIDUAL_DIRECTION, AFFINE_RESIDUAL_INITIAL_SCALE
            ).to(DEVICE)
        metadata = {"field_backend": "siren", 'scalar_network': {
            'num_layers': SCALAR_NUM_LAYERS, 'dim_hidden': 128,
            'w0': SCALAR_W0, 'w0_initial': SCALAR_W0_INITIAL}}
        if isinstance(scalar_net, AffineResidualScalarField):
            metadata.update(
                {
                    "affine_residual_direction": scalar_net.affine_direction.detach().cpu().tolist(),
                    "affine_residual_initial_scale": float(AFFINE_RESIDUAL_INITIAL_SCALE),
                }
            )
        return scalar_net, metadata

    if FIELD_BACKEND != "fourier":
        raise ValueError(f"Unsupported FIELD_BACKEND: {FIELD_BACKEND!r}")

    scalar_net = FourierFeatureNet(
        dim_hidden=FOURIER_HIDDEN_DIM,
        num_layers=FOURIER_NUM_LAYERS,
        num_features=FOURIER_NUM_FEATURES,
        sigma=FOURIER_SIGMA,
        seed=FOURIER_SEED,
    ).to(DEVICE)
    checkpoint_metadata = {
        "field_backend": "fourier",
        "fourier_num_features": FOURIER_NUM_FEATURES,
        "fourier_sigma": FOURIER_SIGMA,
        "fourier_seed": FOURIER_SEED,
        "fourier_hidden_dim": FOURIER_HIDDEN_DIM,
        "fourier_num_layers": FOURIER_NUM_LAYERS,
    }
    if CHECKPOINT_PATH:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        if checkpoint.get("field_backend") != "fourier":
            raise ValueError(
                "A Fourier field requires a Fourier checkpoint. Set CHECKPOINT_PATH = None "
                "for a fresh experiment, or select a checkpoint saved with FIELD_BACKEND = 'fourier'."
            )
        scalar_net.load_state_dict(checkpoint["model_state_dict"])
    return scalar_net, checkpoint_metadata


def build_tool_perturbation_template(
    local_tool_points: torch.Tensor,
    seed: int = TOOL_PERTURB_SEED,
) -> tuple[torch.Tensor, float, float]:
    """Build one cached local `(x, z)` jitter template plus the sampling steps."""
    unique_x = torch.unique(local_tool_points[:, 0])
    unique_z = torch.unique(local_tool_points[:, 2])

    x_step = 0.0 if unique_x.numel() < 2 else float(unique_x[1] - unique_x[0])
    z_step = 0.0 if unique_z.numel() < 2 else float(unique_z[1] - unique_z[0])

    generator = torch.Generator(device=local_tool_points.device)
    generator.manual_seed(int(seed))
    jitter_template = torch.rand(
        (local_tool_points.shape[0], 2),
        device=local_tool_points.device,
        dtype=local_tool_points.dtype,
        generator=generator,
    ) - 0.5
    return jitter_template, x_step, z_step


def perturb_tool_points_for_epoch(
    base_local_tool_points: torch.Tensor,
    jitter_template: torch.Tensor | None,
    x_step: float,
    z_step: float,
) -> torch.Tensor:
    """Apply cheap per-epoch in-plane jitter from a fixed local tool grid."""
    if jitter_template is None:
        return base_local_tool_points

    perturbed_tool_points = base_local_tool_points.clone()
    x_scale = torch.rand((), device=base_local_tool_points.device, dtype=base_local_tool_points.dtype)
    z_scale = torch.rand((), device=base_local_tool_points.device, dtype=base_local_tool_points.dtype)
    perturbed_tool_points[:, 0] = perturbed_tool_points[:, 0] + x_scale * jitter_template[:, 0] * x_step
    perturbed_tool_points[:, 2] = perturbed_tool_points[:, 2] + z_scale * jitter_template[:, 1] * z_step
    return perturbed_tool_points


def build_fixed_frame_tool_points_model(
    base,
    local_tool_points: torch.Tensor,
    range_vals: torch.Tensor,
    fixed_frame: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Orient local tool samples in the fixed frame and scale them into model space."""
    u, v, n = fixed_frame
    x_vals = local_tool_points[:, 0:1]
    y_vals = local_tool_points[:, 1:2]
    z_vals = local_tool_points[:, 2:3]
    tool_points_world = x_vals * u.unsqueeze(0) + y_vals * n.unsqueeze(0) + z_vals * v.unsqueeze(0)
    return base.scale_tool_points(tool_points_world, range_vals)


def reduce_collision_loss(
    collision_out: dict[str, torch.Tensor],
    tool_point_counts: list[int],
    reduction: str,
    topk_fraction: float,
    tool_weights: tuple[float, ...],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Reduce one shared collision evaluation independently by tool when requested."""
    if reduction == "global_mean":
        return collision_out["loss"], [collision_out["loss"]]
    if reduction != "per_tool_topk":
        raise ValueError(f"Unsupported collision loss reduction: {reduction!r}.")
    if not 0.0 < topk_fraction <= 1.0:
        raise ValueError("COLLISION_TOPK_FRACTION must be in (0, 1].")
    if len(tool_weights) != len(tool_point_counts):
        raise ValueError("COLLISION_TOOL_WEIGHTS must contain one weight per tool surface.")

    scalar_differences = collision_out["scalar_differences"].squeeze(-1)
    inside_mask = collision_out["inside_mask"].squeeze(-1)
    squared_violations = torch.relu(-scalar_differences).square() * inside_mask

    per_tool_losses = []
    start = 0
    for point_count in tool_point_counts:
        stop = start + point_count
        tool_violations = squared_violations[:, start:stop]
        k = max(1, int(np.ceil(topk_fraction * point_count)))
        per_root_loss = torch.topk(tool_violations, k=k, dim=1).values.mean(dim=1)
        per_tool_losses.append(per_root_loss.mean())
        start = stop

    if start != squared_violations.shape[1]:
        raise ValueError("Tool point counts do not match the combined collision samples.")

    weighted_losses = [weight * loss for weight, loss in zip(tool_weights, per_tool_losses)]
    return torch.stack(weighted_losses).sum(), per_tool_losses


def compute_surface_quality_constraints(
    mesh,
    up_direction: np.ndarray,
    max_angle_degrees: float,
    min_height_world: float | None = None,
    max_height_world: float | None = None,
    min_region_area_ratio: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select vertices incident to faces within the given inclination threshold.

    Faces are selected first. Every vertex incident to a selected face becomes
    a constraint sample, and its target normal is the normalized average of
    exactly those selected incident face normals. Optional height bounds are
    evaluated at face centers along `up_direction` from the lowest mesh vertex.
    """
    mesh = mesh.copy(deep=True)
    mesh = mesh.triangulate()
    mesh.compute_normals(point_normals=False, cell_normals=True, inplace=True)

    faces = np.asarray(mesh.faces).reshape((-1, 4))[:, 1:4]
    face_normals = np.asarray(mesh.cell_data["Normals"], dtype=np.float32)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8
    mesh_points = np.asarray(mesh.points, dtype=np.float32)
    face_centers = np.mean(mesh_points[faces], axis=1)

    unit_up = up_direction.reshape(1, 3).astype(np.float32)
    unit_up = unit_up / (np.linalg.norm(unit_up, axis=1, keepdims=True) + 1e-8)
    cos_threshold = np.cos(np.deg2rad(max_angle_degrees))
    selected_faces = np.sum(face_normals * unit_up, axis=1) > cos_threshold

    if min_height_world is not None or max_height_world is not None:
        mesh_projected = (mesh_points @ unit_up.reshape(3)).astype(np.float32)
        min_projected_height = float(np.min(mesh_projected))
        vertex_relative_heights = mesh_projected[faces] - min_projected_height
        if HEIGHT_SELECTION_MODE == "clipped":
            if min_height_world is not None:
                selected_faces &= np.max(vertex_relative_heights, axis=1) >= float(min_height_world)
            if max_height_world is not None:
                selected_faces &= np.min(vertex_relative_heights, axis=1) <= float(max_height_world)
        else:
            face_relative_height = (face_centers @ unit_up.reshape(3)).astype(np.float32) - min_projected_height
            if min_height_world is not None:
                selected_faces &= face_relative_height >= float(min_height_world)
            if max_height_world is not None:
                selected_faces &= face_relative_height <= float(max_height_world)

    if min_region_area_ratio is not None:
        selected_faces = filter_selected_face_regions(
            mesh=mesh,
            selected_faces=selected_faces,
            min_region_area_ratio=min_region_area_ratio,
        )

    selected_mask = np.zeros(mesh.n_points, dtype=bool)
    selected_mask[faces[selected_faces].reshape(-1)] = True
    selected_face_normal_sum = np.zeros((mesh.n_points, 3), dtype=np.float32)
    for face, face_normal in zip(faces[selected_faces], face_normals[selected_faces]):
        selected_face_normal_sum[face] += face_normal
    averaged_normals = selected_face_normal_sum[selected_mask]
    averaged_normals /= np.linalg.norm(averaged_normals, axis=1, keepdims=True) + 1e-8
    selected_group_ids = build_selected_group_ids(mesh, selected_mask)
    return mesh_points[selected_mask], averaged_normals, selected_group_ids


def build_face_adjacency(faces: np.ndarray) -> list[set[int]]:
    """Build face adjacency across shared edges, not merely shared vertices."""

    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted((int(first), int(second))))
            edge_to_faces.setdefault(edge, []).append(face_index)
    neighbors: list[set[int]] = [set() for _ in range(faces.shape[0])]
    for incident_faces in edge_to_faces.values():
        for face_index in incident_faces:
            neighbors[face_index].update(other for other in incident_faces if other != face_index)
    return neighbors


def filter_selected_face_regions(
    mesh: pv.PolyData,
    selected_faces: np.ndarray,
    min_region_area_ratio: float,
) -> np.ndarray:
    """Remove small components using face connectivity and actual triangle area."""

    if min_region_area_ratio <= 0.0 or not np.any(selected_faces):
        return selected_faces
    faces = np.asarray(mesh.faces).reshape((-1, 4))[:, 1:4]
    points = np.asarray(mesh.points, dtype=np.float64)
    triangle_points = points[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangle_points[:, 1] - triangle_points[:, 0], triangle_points[:, 2] - triangle_points[:, 0]),
        axis=1,
    )
    neighbors = build_face_adjacency(faces)
    visited = np.zeros(faces.shape[0], dtype=bool)
    components: list[list[int]] = []
    for start_index in np.flatnonzero(selected_faces):
        if visited[start_index]:
            continue
        visited[start_index] = True
        component = []
        stack = [int(start_index)]
        while stack:
            current_index = stack.pop()
            component.append(current_index)
            for neighbor_index in neighbors[current_index]:
                if selected_faces[neighbor_index] and not visited[neighbor_index]:
                    visited[neighbor_index] = True
                    stack.append(neighbor_index)
        components.append(component)
    component_areas = [float(np.sum(areas[component])) for component in components]
    threshold = min_region_area_ratio * max(component_areas)
    filtered_faces = np.zeros_like(selected_faces)
    for component, component_area in zip(components, component_areas):
        if component_area >= threshold:
            filtered_faces[np.asarray(component, dtype=np.int64)] = True
    return filtered_faces


def build_selected_components(mesh, selected_mask: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    """Return connected components of selected mesh points and per-point area weights."""
    mesh_faces = np.asarray(mesh.faces).reshape((-1, 4))[:, 1:4]
    point_to_neighbors: list[set[int]] = [set() for _ in range(mesh.n_points)]
    for a_idx, b_idx, c_idx in mesh_faces:
        point_to_neighbors[a_idx].update((int(b_idx), int(c_idx)))
        point_to_neighbors[b_idx].update((int(a_idx), int(c_idx)))
        point_to_neighbors[c_idx].update((int(a_idx), int(b_idx)))

    point_areas = np.zeros(mesh.n_points, dtype=np.float64)
    triangle_points = np.asarray(mesh.points, dtype=np.float64)[mesh_faces]
    triangle_areas = 0.5 * np.linalg.norm(
        np.cross(triangle_points[:, 1] - triangle_points[:, 0], triangle_points[:, 2] - triangle_points[:, 0]),
        axis=1,
    )
    for face_idx, face in enumerate(mesh_faces):
        shared_area = float(triangle_areas[face_idx]) / 3.0
        point_areas[face] += shared_area

    selected_indices = np.nonzero(selected_mask)[0]
    visited = np.zeros(mesh.n_points, dtype=bool)
    components: list[list[int]] = []
    for start_idx in selected_indices:
        if visited[start_idx]:
            continue
        stack = [int(start_idx)]
        visited[start_idx] = True
        component: list[int] = []
        while stack:
            current_idx = stack.pop()
            component.append(current_idx)
            for neighbor_idx in point_to_neighbors[current_idx]:
                if selected_mask[neighbor_idx] and not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    stack.append(neighbor_idx)
        components.append(component)

    return components, point_areas


def build_selected_group_ids(mesh, selected_mask: np.ndarray) -> np.ndarray:
    """Return consecutive group ids for each selected point."""
    selected_indices = np.nonzero(selected_mask)[0]
    if selected_indices.size == 0:
        return np.zeros((0,), dtype=np.int64)

    components, _ = build_selected_components(mesh, selected_mask)
    group_ids_full = np.full(mesh.n_points, fill_value=-1, dtype=np.int64)
    for group_index, component in enumerate(components):
        group_ids_full[np.asarray(component, dtype=np.int64)] = group_index
    return group_ids_full[selected_indices]


def orient_preserved_group_normals_toward_up(
    normals: np.ndarray,
    group_ids: np.ndarray,
    up_direction: np.ndarray,
) -> np.ndarray:
    """Flip a connected preserved group only when its aggregate normal is downward."""
    if normals.size == 0 or group_ids.size == 0:
        return normals
    unit_up = np.asarray(up_direction, dtype=np.float32).reshape(3)
    unit_up /= np.linalg.norm(unit_up) + 1e-8
    oriented_normals = normals.copy()
    for group_id in np.unique(group_ids):
        group_mask = group_ids == group_id
        aggregate_normal = np.sum(oriented_normals[group_mask], axis=0)
        aggregate_length = float(np.linalg.norm(aggregate_normal))
        if aggregate_length <= 1e-8:
            print(f"Preserved group {int(group_id)}: skipped normal flip because its aggregate normal is near zero.")
            continue
        up_dot = float(np.dot(aggregate_normal / aggregate_length, unit_up))
        if up_dot < 0.0:
            oriented_normals[group_mask] *= -1.0
            print(f"Preserved group {int(group_id)}: flipped normals toward UP_DIRECTION (aggregate dot={up_dot:.4f}).")
    return oriented_normals


def filter_surface_quality_regions(
    mesh,
    selected_mask: np.ndarray,
    min_region_area_ratio: float,
) -> np.ndarray:
    """Keep only connected selected point regions whose area exceeds a fraction of the largest region."""
    if min_region_area_ratio <= 0.0 or not np.any(selected_mask):
        return selected_mask

    components, point_areas = build_selected_components(mesh, selected_mask)
    if not components:
        return selected_mask

    area_by_component = [float(np.sum(point_areas[component])) for component in components]
    max_region_area = max(area_by_component)
    min_allowed_area = float(min_region_area_ratio) * max_region_area

    filtered_mask = np.zeros_like(selected_mask, dtype=bool)
    for component, component_area in zip(components, area_by_component):
        if component_area >= min_allowed_area:
            filtered_mask[np.asarray(component, dtype=np.int64)] = True
    return filtered_mask


def compute_base_constraints(
    surface_mesh: pv.PolyData,
    up_direction: np.ndarray,
    base_height_world: float,
    scalar_scale: float,
    max_normal_angle_degrees: float = BASE_NORMAL_MAX_ANGLE_DEGREES,
) -> tuple[np.ndarray, np.ndarray]:
    """Select low vertices incident to downward-facing faces and assign scalar targets."""
    if base_height_world <= 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 1), dtype=np.float32)
    mesh = surface_mesh.copy(deep=True).triangulate()
    mesh.compute_normals(point_normals=False, cell_normals=True, inplace=True)
    points = np.asarray(mesh.points, dtype=np.float32)
    unit_up = up_direction.reshape(1, 3).astype(np.float32)
    unit_up = unit_up / (np.linalg.norm(unit_up, axis=1, keepdims=True) + 1e-8)

    projected_height = (points @ unit_up.reshape(3)).astype(np.float32)
    min_height = float(np.min(projected_height))
    relative_height = projected_height - min_height
    faces = np.asarray(mesh.faces).reshape((-1, 4))[:, 1:4]
    face_normals = np.asarray(mesh.cell_data["Normals"], dtype=np.float32)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8
    downward_faces = face_normals @ (-unit_up.reshape(3)) > np.cos(np.deg2rad(max_normal_angle_degrees))
    downward_vertex_mask = np.zeros(mesh.n_points, dtype=bool)
    downward_vertex_mask[faces[downward_faces].reshape(-1)] = True
    base_mask = (relative_height <= base_height_world) & downward_vertex_mask

    base_points = points[base_mask]
    base_target_scalars = (relative_height[base_mask] / float(scalar_scale)).reshape(-1, 1)
    return base_points, base_target_scalars.astype(np.float32)


def compute_support_free_surface_constraints(
    mesh,
    up_direction: np.ndarray,
    base_height_world: float,
    min_height_world: float | None = None,
    max_height_world: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select support-free surface points within an optional height interval."""
    mesh = mesh.copy(deep=True)
    mesh = mesh.triangulate()
    mesh.compute_normals(point_normals=True, cell_normals=False, inplace=True)

    point_normals = np.asarray(mesh.point_data["Normals"], dtype=np.float32)
    point_normals = point_normals / (np.linalg.norm(point_normals, axis=1, keepdims=True) + 1e-8)
    mesh_points = np.asarray(mesh.points, dtype=np.float32)

    unit_up = up_direction.reshape(1, 3).astype(np.float32)
    unit_up = unit_up / (np.linalg.norm(unit_up, axis=1, keepdims=True) + 1e-8)
    projected_height = (mesh_points @ unit_up.reshape(3)).astype(np.float32)
    min_projected_height = float(np.min(projected_height))
    relative_height = projected_height - min_projected_height
    selected_mask = (relative_height > float(base_height_world)) if base_height_world > 0 else np.ones(len(mesh_points), dtype=bool)
    if min_height_world is not None:
        selected_mask &= relative_height >= float(min_height_world)
    if max_height_world is not None:
        selected_mask &= relative_height <= float(max_height_world)
    return mesh_points[selected_mask], point_normals[selected_mask]


def visualize_surface_quality_points(
    mesh,
    points: np.ndarray,
    normals: np.ndarray,
    group_ids: np.ndarray,
) -> None:
    """Show the exact point-normal constraints selected before optimization."""
    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.enable_anti_aliasing()
    plotter.add_mesh(mesh, color="lightgray", opacity=0.18)

    selected_cloud = pv.PolyData(points)
    if group_ids.size == points.shape[0] and group_ids.size > 0:
        selected_cloud.point_data["surface_group"] = group_ids
        plotter.add_mesh(
            selected_cloud,
            scalars="surface_group",
            cmap="turbo",
            point_size=8.0,
            render_points_as_spheres=True,
            show_scalar_bar=False,
        )
    else:
        plotter.add_mesh(selected_cloud, color="tomato", point_size=8.0, render_points_as_spheres=True)

    if SHOW_SURFACE_QUALITY_NORMALS_BEFORE_TRAINING and points.shape[0] > 0:
        arrow_stride = max(1, int(SURFACE_QUALITY_NORMAL_ARROW_STRIDE))
        arrow_indices = np.arange(0, points.shape[0], arrow_stride)
        plotter.add_arrows(
            points[arrow_indices],
            normals[arrow_indices],
            mag=SURFACE_QUALITY_NORMAL_ARROW_LENGTH_WORLD,
            color="deepskyblue",
        )
    plotter.add_text(
        "Surface-quality constraints: colored points; cyan arrows: selected point normals",
        position="upper_left",
        font_size=10,
    )

    plotter.show()


def visualize_surface_constraint_groups_and_batches(
    mesh: pv.PolyData,
    points: np.ndarray,
    normals: np.ndarray,
    group_ids: np.ndarray,
    up_direction: np.ndarray,
) -> None:
    """Display connected scalar groups and the slope batches used for normal loss."""
    if points.shape[0] == 0:
        print("Surface constraint batch visualization skipped: no selected points.")
        return

    unique_groups, group_counts = np.unique(group_ids, return_counts=True)
    print(
        "Surface scalar groups: "
        + ", ".join(f"group {group_id}: {count} points" for group_id, count in zip(unique_groups, group_counts))
    )
    group_cloud = pv.PolyData(points)
    group_cloud.point_data["connected_group"] = group_ids.astype(np.int32)
    group_plotter = pv.Plotter()
    group_plotter.add_axes()
    group_plotter.add_mesh(mesh, color="lightgray", opacity=0.15)
    group_plotter.add_mesh(
        group_cloud,
        scalars="connected_group",
        cmap="turbo",
        point_size=8.0,
        render_points_as_spheres=True,
        show_scalar_bar=True,
    )
    group_plotter.add_text("Connected surface-constraint groups", position="upper_left", font_size=10)
    group_plotter.show()

    if not SURFACE_NORMAL_STRATIFICATION_ENABLED:
        print("Surface-normal stratification is disabled, so normal-loss batches are ordinary shuffled batches.")
        return

    unit_up = up_direction.reshape(3).astype(np.float32)
    unit_up /= np.linalg.norm(unit_up) + 1e-8
    normal_dot = np.clip(normals @ unit_up, -1.0, 1.0)
    angles = np.degrees(np.arccos(normal_dot))
    bin_edges = np.linspace(
        0.0,
        SURFACE_NORMAL_STRATIFICATION_MAX_ANGLE_DEGREES,
        SURFACE_NORMAL_STRATIFICATION_BIN_COUNT + 1,
    )
    batch_ids = np.digitize(angles, bin_edges[1:-1], right=False).astype(np.int32)
    unique_batches, batch_counts = np.unique(batch_ids, return_counts=True)
    batch_summary = []
    for batch_id, count in zip(unique_batches, batch_counts):
        lower = bin_edges[batch_id]
        upper = bin_edges[batch_id + 1]
        if batch_id < SURFACE_NORMAL_STRATIFICATION_BIN_COUNT - 1:
            angle_text = f"{lower:.1f}-{upper:.1f} deg"
        else:
            angle_text = f">={lower:.1f} deg"
        batch_summary.append(f"batch {batch_id} ({angle_text}): {count} points")
    print("Surface normal-loss batches: " + ", ".join(batch_summary))

    batch_cloud = pv.PolyData(points)
    batch_cloud.point_data["normal_loss_batch"] = batch_ids
    batch_plotter = pv.Plotter()
    batch_plotter.add_axes()
    batch_plotter.add_mesh(mesh, color="lightgray", opacity=0.15)
    batch_plotter.add_mesh(
        batch_cloud,
        scalars="normal_loss_batch",
        cmap="Set1",
        point_size=8.0,
        render_points_as_spheres=True,
        show_scalar_bar=True,
    )
    batch_plotter.add_text(
        "Slope-coherent normal-loss batches (each color is one whole batch)",
        position="upper_left",
        font_size=10,
    )
    batch_plotter.show()


def visualize_input_regularization_batches(
    mesh: pv.PolyData,
    candidate_points: np.ndarray,
    batch_size: int,
    points_per_epoch: int,
) -> None:
    """Show representative input regularization batches in separate figures."""
    if candidate_points.shape[0] == 0:
        print("Input-regularization batch visualization skipped: no sampled grid points.")
        return
    if batch_size <= 0:
        raise ValueError("BATCH_SIZE must be positive for input regularization visualization.")

    epoch_point_count = candidate_points.shape[0] if points_per_epoch == 0 else min(
        points_per_epoch, candidate_points.shape[0]
    )
    preview_batch_count = min(
        INPUT_REGULARIZATION_VIS_BATCH_COUNT,
        int(np.ceil(epoch_point_count / batch_size)),
    )
    # The actual loader redraws this order each epoch. This fixed preview makes
    # the first few otherwise spatially scattered batches inspectable.
    preview_order = np.random.default_rng(INPUT_REGULARIZATION_RESAMPLING_SEED).permutation(candidate_points.shape[0])
    epoch_points = candidate_points[preview_order[:epoch_point_count]]
    print(
        f"Input regularization: {candidate_points.shape[0]} candidate points, "
        f"{epoch_point_count} points/epoch, {int(np.ceil(epoch_point_count / batch_size))} batches/epoch; "
        f"showing {preview_batch_count} batches in separate figures."
    )
    for batch_index in range(preview_batch_count):
        start = batch_index * batch_size
        batch_points = epoch_points[start : start + batch_size]
        plotter = pv.Plotter()
        plotter.add_axes()
        plotter.add_mesh(mesh, color="lightgray", opacity=0.14)
        plotter.add_mesh(
            pv.PolyData(batch_points),
            color="tomato",
            point_size=INPUT_REGULARIZATION_VIS_POINT_SIZE,
            render_points_as_spheres=True,
        )
        plotter.add_text(
            f"Input regularization batch {batch_index + 1}/{int(np.ceil(epoch_point_count / batch_size))}: "
            f"{batch_points.shape[0]} points",
            position="upper_left",
            font_size=10,
        )
        plotter.show()


def visualize_all_loss_evaluation_points(
    mesh: pv.PolyData,
    input_points: np.ndarray,
    surface_points: np.ndarray,
    base_points: np.ndarray,
    support_free_points: np.ndarray,
    edge_points_a: np.ndarray,
    edge_points_b: np.ndarray,
) -> None:
    """Show every point pool from which the currently configured losses draw."""
    point_sets = [
        ("Input regularizers / curvature / collision roots", input_points, "slategray", 2.0, 0.20),
        ("Surface direction + group scalar", surface_points, "tomato", 7.0, 1.0),
        ("Base direction + scalar", base_points, "deepskyblue", 7.0, 1.0),
        ("Support-free surface", support_free_points, "gold", 7.0, 1.0),
        ("Surface edge scalar", edge_points_a, "magenta", 7.0, 1.0),
        ("Surface edge scalar", edge_points_b, "magenta", 7.0, 1.0),
    ]
    print(
        "Loss evaluation pools: "
        + ", ".join(f"{label}={points.shape[0]}" for label, points, _, _, _ in point_sets)
    )
    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.add_mesh(mesh, color="lightgray", opacity=0.12)
    added_labels: set[str] = set()
    for label, points, color, point_size, opacity in point_sets:
        if points.shape[0] == 0:
            continue
        plotter.add_mesh(
            pv.PolyData(points),
            color=color,
            opacity=opacity,
            point_size=point_size,
            render_points_as_spheres=True,
            label=label if label not in added_labels else None,
        )
        added_labels.add(label)
    if added_labels:
        plotter.add_legend(size=(0.36, 0.20), bcolor="white")
    plotter.add_text(
        "All loss pools; curvature samples a fresh subset of the gray input pool each step",
        position="upper_left",
        font_size=10,
    )
    plotter.show()


def visualize_surface_edge_pairs(mesh: pv.PolyData, points_a: np.ndarray, points_b: np.ndarray) -> None:
    """Show the actual local scalar-loss pairs as line segments."""
    if len(points_a) == 0:
        print("Surface edge-pair preview: no accepted pairs.")
        return
    pair_points = np.empty((2 * len(points_a), 3), dtype=np.float32)
    pair_points[0::2], pair_points[1::2] = points_a, points_b
    lines = np.column_stack((np.full(len(points_a), 2, dtype=np.int64), np.arange(2 * len(points_a), dtype=np.int64).reshape((-1, 2)))).reshape(-1)
    plotter = pv.Plotter(title="Sampled surface neighbour pairs")
    plotter.add_axes()
    plotter.add_mesh(mesh, color="lightgray", opacity=0.15)
    plotter.add_mesh(pv.PolyData(pair_points, lines=lines), color="magenta", line_width=2)
    plotter.add_mesh(pv.PolyData(pair_points), color="gold", point_size=5, render_points_as_spheres=True)
    plotter.add_text(f"Accepted sampled neighbour pairs: {len(points_a)}", position="upper_left", font_size=10)
    plotter.show()


def visualize_norm_control_batches(batch_snapshots: list[tuple[np.ndarray, np.ndarray]], epoch: int) -> None:
    """Display the norm-control points sampled in each batch of an epoch."""
    if not batch_snapshots:
        return
    snapshots = batch_snapshots[:NORM_CONTROL_BATCH_VIS_MAX_BATCHES]
    columns = min(4, len(snapshots))
    rows = int(np.ceil(len(snapshots) / columns))
    plotter = pv.Plotter(shape=(rows, columns), title=f"Norm-control batches, epoch {epoch + 1}")
    for batch_index, (points, bands) in enumerate(snapshots):
        plotter.subplot(batch_index // columns, batch_index % columns)
        cloud = pv.PolyData(points)
        cloud["band"] = bands.astype(np.int32)
        plotter.add_mesh(cloud, scalars="band", cmap="turbo", point_size=5.0,
                         render_points_as_spheres=True)
        plotter.add_text(f"batch {batch_index + 1}\npoints={len(points)}", font_size=9)
        plotter.add_axes()
    plotter.show()


def build_mesh_vertex_neighbors(mesh: pv.PolyData) -> list[set[int]]:
    """Build vertex adjacency from a triangulated surface mesh."""
    faces = np.asarray(mesh.faces).reshape((-1, 4))[:, 1:4]
    neighbors: list[set[int]] = [set() for _ in range(mesh.n_points)]
    for first, second, third in faces:
        neighbors[first].update((int(second), int(third)))
        neighbors[second].update((int(first), int(third)))
        neighbors[third].update((int(first), int(second)))
    return neighbors


def flood_smooth_surface_region(
    seed_index: int,
    faces: np.ndarray,
    face_normals: np.ndarray,
    face_neighbors: list[set[int]],
    max_neighbor_angle_degrees: float,
) -> np.ndarray:
    """Flood incident faces across edges whose face normals remain smooth."""

    minimum_dot = float(np.cos(np.deg2rad(max_neighbor_angle_degrees)))
    seed_faces = np.flatnonzero(np.any(faces == int(seed_index), axis=1))
    if seed_faces.size == 0:
        return np.empty(0, dtype=np.int64)
    visited = np.zeros(faces.shape[0], dtype=bool)
    visited[seed_faces] = True
    stack = [int(face_index) for face_index in seed_faces]

    while stack:
        current_face = stack.pop()
        for neighbor_index in face_neighbors[current_face]:
            if visited[neighbor_index]:
                continue
            normal_dot = float(np.dot(face_normals[current_face], face_normals[neighbor_index]))
            if normal_dot >= minimum_dot:
                visited[neighbor_index] = True
                stack.append(neighbor_index)

    return np.unique(faces[visited].reshape(-1))


def select_surface_erase_brush(
    seed_index: int,
    points: np.ndarray,
    vertex_neighbors: list[set[int]],
    radius_world: float,
) -> np.ndarray:
    """Select a local connected patch by geodesic distance along mesh edges."""
    if radius_world <= 0.0:
        return np.asarray([seed_index], dtype=np.int64)

    distances = np.full(points.shape[0], np.inf, dtype=np.float64)
    distances[seed_index] = 0.0
    queue: list[tuple[float, int]] = [(0.0, int(seed_index))]

    while queue:
        distance, current_index = heapq.heappop(queue)
        if distance != distances[current_index] or distance > radius_world:
            continue
        for neighbor_index in vertex_neighbors[current_index]:
            edge_length = float(np.linalg.norm(points[current_index] - points[neighbor_index]))
            candidate_distance = distance + edge_length
            if candidate_distance <= radius_world and candidate_distance < distances[neighbor_index]:
                distances[neighbor_index] = candidate_distance
                heapq.heappush(queue, (candidate_distance, neighbor_index))

    return np.flatnonzero(distances <= radius_world)


def split_manual_regions_into_connected_components(
    regions: list[np.ndarray],
    vertex_neighbors: list[set[int]],
) -> list[np.ndarray]:
    """Split edited manual patches so each returned scalar group is connected."""
    connected_regions: list[np.ndarray] = []
    for region in regions:
        allowed = np.zeros(len(vertex_neighbors), dtype=bool)
        allowed[region] = True
        visited = np.zeros_like(allowed)
        for seed_index in region:
            if visited[seed_index]:
                continue
            component: list[int] = []
            stack = [int(seed_index)]
            visited[seed_index] = True
            while stack:
                current_index = stack.pop()
                component.append(current_index)
                for neighbor_index in vertex_neighbors[current_index]:
                    if allowed[neighbor_index] and not visited[neighbor_index]:
                        visited[neighbor_index] = True
                        stack.append(neighbor_index)
            connected_regions.append(np.asarray(component, dtype=np.int64))
    return connected_regions


def save_manual_surface_regions(
    path: str | Path,
    regions: list[np.ndarray],
    n_points: int,
    model_name: str,
) -> None:
    """Save final manual mesh-vertex groups in a portable, inspectable JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_points": int(n_points),
        "regions": [region.astype(int).tolist() for region in regions],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {len(regions)} manual surface group(s) to {output_path}")


def load_manual_surface_regions(path: str | Path, n_points: int, model_name: str) -> list[np.ndarray]:
    """Load cached manual groups and ensure they belong to this mesh."""
    input_path = Path(path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    cached_model_name = payload.get("model_name")
    if cached_model_name is not None and cached_model_name != model_name:
        raise ValueError(
            f"Manual region cache {input_path} is for model {cached_model_name!r}, "
            f"not {model_name!r}."
        )
    if int(payload.get("n_points", -1)) != n_points:
        raise ValueError(
            f"Manual region cache {input_path} has {payload.get('n_points')} mesh vertices, "
            f"but the current mesh has {n_points}."
        )
    regions = [np.unique(np.asarray(region, dtype=np.int64)) for region in payload.get("regions", [])]
    if any(np.any(region < 0) or np.any(region >= n_points) for region in regions):
        raise ValueError(f"Manual region cache {input_path} contains invalid mesh vertex indices.")
    print(f"Loaded {len(regions)} manual surface group(s) from {input_path}")
    return regions


def save_surface_constraint_metadata(
    path: str | Path,
    mesh: pv.PolyData,
    constrained_points: np.ndarray,
    constrained_group_ids: np.ndarray,
    automatic_group_ids: set[int],
    model_name: str,
    automatic_max_angle_degrees: float | None = None,
    automatic_min_height_world: float | None = None,
    automatic_max_height_world: float | None = None,
    automatic_min_region_area_ratio: float | None = None,
    cap_geometry: dict[str, np.ndarray] | None = None,
) -> None:
    """Save constraint groups and, when available, their exact clipped cap geometry."""
    metadata_mesh = mesh.copy(deep=True).triangulate()
    mesh_points = np.asarray(metadata_mesh.points, dtype=np.float32)
    point_to_index = {tuple(point): index for index, point in enumerate(mesh_points)}
    regions = []
    for group_id in np.unique(constrained_group_ids):
        group_points = constrained_points[constrained_group_ids == group_id]
        vertex_indices = sorted(
            {
                point_to_index[tuple(point)]
                for point in group_points
                if tuple(point) in point_to_index
            }
        )
        if vertex_indices:
            regions.append(
                {
                    "group_id": int(group_id),
                    "source": "automatic" if int(group_id) in automatic_group_ids else "manual",
                    "vertex_indices": vertex_indices,
                }
            )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap_geometry_payload = None
    if cap_geometry is not None and len(cap_geometry.get("triangles", [])):
        triangles = np.asarray(cap_geometry["triangles"], dtype=np.float32)
        group_ids = np.asarray(cap_geometry["group_ids"], dtype=np.int64)
        parent_face_ids = np.asarray(cap_geometry["parent_face_ids"], dtype=np.int64)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
            raise ValueError("Cap geometry triangles must have shape (N, 3, 3).")
        if len(group_ids) != len(triangles) or len(parent_face_ids) != len(triangles):
            raise ValueError("Cap geometry arrays must have one group and parent face per triangle.")
        # Keep the metadata self-contained. Unlike a vertex-index list, these
        # triangles represent partial height/angle boundary clips exactly.
        cap_geometry_payload = {
            "format": "clipped_parent_triangles_v1",
            "triangles": triangles.tolist(),
            "group_ids": group_ids.tolist(),
            "parent_face_ids": parent_face_ids.tolist(),
        }
        # A clipped patch can have no original mesh vertex in its interior.
        # Preserve those geometry-only groups explicitly instead of silently
        # dropping them from the human-readable region list.
        saved_group_ids = {int(region["group_id"]) for region in regions}
        for group_id in np.unique(group_ids):
            group_id = int(group_id)
            if group_id not in saved_group_ids:
                regions.append(
                    {
                        "group_id": group_id,
                        "source": "automatic" if group_id in automatic_group_ids else "manual",
                        "vertex_indices": [],
                        "geometry_only": True,
                    }
                )
        regions.sort(key=lambda region: int(region["group_id"]))
    output_path.write_text(
        json.dumps(
            {
                "metadata_type": "surface_constraint_regions",
                "model_name": model_name,
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                "n_points": int(metadata_mesh.n_points),
                "automatic_selection": {
                    "max_angle_degrees": automatic_max_angle_degrees,
                    "min_height_world": automatic_min_height_world,
                    "max_height_world": automatic_max_height_world,
                    "min_region_area_ratio": automatic_min_region_area_ratio,
                },
                # Vertex indices alone cannot represent partial boundary clips.
                "cap_geometry": cap_geometry_payload,
                "regions": regions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {len(regions)} automatic/manual surface constraint group(s) to {output_path}")


def select_manual_surface_regions(mesh: pv.PolyData) -> tuple[list[np.ndarray], np.ndarray]:
    """Interactively add regions using face-normal smoothness for flooding."""
    selection_mesh = mesh.copy(deep=True).triangulate()
    selection_mesh.compute_normals(point_normals=True, cell_normals=True, inplace=True)
    point_normals = np.asarray(selection_mesh.point_data["Normals"], dtype=np.float32)
    point_normals /= np.linalg.norm(point_normals, axis=1, keepdims=True) + 1e-8
    face_normals = np.asarray(selection_mesh.cell_data["Normals"], dtype=np.float32)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8
    faces = np.asarray(selection_mesh.faces).reshape((-1, 4))[:, 1:4]
    face_neighbors = build_face_adjacency(faces)
    vertex_neighbors = build_mesh_vertex_neighbors(selection_mesh)
    points = np.asarray(selection_mesh.points, dtype=np.float32)

    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.enable_anti_aliasing()
    plotter.add_mesh(selection_mesh, color="lightgray", opacity=0.22, name="input_mesh", pickable=True)
    state: dict[str, object] = {
        "preview": np.empty(0, dtype=np.int64),
        "accepted": [],
        "mode": "flood",
    }

    def refresh_instructions() -> None:
        if state["mode"] == "erase":
            message = (
                "ERASER MODE (red)\n"
                "Left-click: immediately erase a small connected area\n"
                "B: return to flood mode | P: local add brush | C: clear all\n"
                "Close window: continue training"
            )
        elif state["mode"] == "paint":
            message = (
                "LOCAL ADD MODE (blue)\n"
                "Left-click: immediately add a small connected area\n"
                "P: return to flood mode | B: eraser | C: clear all\n"
                "Close window: continue training"
            )
        else:
            message = (
                "ADD MODE (green)\n"
                "Left-click: preview a smooth flooded region\n"
                "A: add the green preview | R: discard preview\n"
                "B: switch to eraser | P: switch to local add brush\n"
                "Close window: continue training"
            )
        plotter.add_text(message, position="upper_left", font_size=10, name="selection_instructions")

    def show_indices(name: str, indices: np.ndarray, color: str, point_size: float) -> None:
        plotter.remove_actor(name, render=False)
        if indices.size > 0:
            plotter.add_mesh(
                pv.PolyData(points[indices]),
                color=color,
                point_size=point_size,
                render_points_as_spheres=True,
                name=name,
                reset_camera=False,
                pickable=False,
            )

    def refresh_preview() -> None:
        color = {"erase": "crimson", "paint": "dodgerblue", "flood": "limegreen"}[state["mode"]]
        show_indices("manual_preview", state["preview"], color, 7.0)
        plotter.render()

    def refresh_accepted() -> None:
        accepted = state["accepted"]
        accepted_indices = np.concatenate(accepted) if accepted else np.empty(0, dtype=np.int64)
        show_indices("manual_accepted", accepted_indices, "darkorange", 5.0)
        plotter.render()

    def on_pick(_picked_point, picker) -> None:
        point_index = int(picker.GetPointId())
        if point_index < 0:
            return
        if state["mode"] in {"erase", "paint"}:
            state["preview"] = select_surface_erase_brush(
                seed_index=point_index,
                points=points,
                vertex_neighbors=vertex_neighbors,
                radius_world=MANUAL_SURFACE_ERASE_RADIUS_WORLD,
            )
            if state["mode"] == "erase":
                erase_preview_from_accepted()
            else:
                accept_preview()
        else:
            state["preview"] = flood_smooth_surface_region(
                seed_index=point_index,
                faces=faces,
                face_normals=face_normals,
                face_neighbors=face_neighbors,
                max_neighbor_angle_degrees=MANUAL_SURFACE_FLOOD_MAX_NEIGHBOR_NORMAL_ANGLE_DEGREES,
            )
        refresh_preview()

    def accept_preview() -> None:
        preview = state["preview"]
        if preview.size == 0:
            return
        accepted = state["accepted"]
        already_accepted = np.concatenate(accepted) if accepted else np.empty(0, dtype=np.int64)
        new_indices = np.setdiff1d(preview, already_accepted, assume_unique=False)
        if new_indices.size > 0:
            accepted.append(new_indices)
        state["preview"] = np.empty(0, dtype=np.int64)
        refresh_preview()
        refresh_accepted()

    def clear_preview() -> None:
        state["preview"] = np.empty(0, dtype=np.int64)
        refresh_preview()

    def switch_mode(next_mode: str) -> None:
        entering_brush = state["mode"] == "flood" and next_mode != "flood"
        # Preserve a visible green flood preview before entering either brush mode.
        if entering_brush and state["preview"].size > 0:
            accept_preview()
        state["mode"] = next_mode
        clear_preview()
        refresh_instructions()
        print(f"Manual surface selection mode: {next_mode}")

    def toggle_eraser_mode() -> None:
        switch_mode("flood" if state["mode"] == "erase" else "erase")

    def toggle_paint_mode() -> None:
        switch_mode("flood" if state["mode"] == "paint" else "paint")

    def erase_preview_from_accepted() -> None:
        """Subtract the current smooth flood from every accepted patch."""
        preview = state["preview"]
        accepted = state["accepted"]
        if preview.size == 0 or not accepted:
            return

        preview_mask = np.zeros(points.shape[0], dtype=bool)
        preview_mask[preview] = True
        state["accepted"] = [
            region[~preview_mask[region]]
            for region in accepted
            if np.any(~preview_mask[region])
        ]
        clear_preview()
        refresh_accepted()

    def clear_accepted() -> None:
        state["accepted"] = []
        clear_preview()
        refresh_accepted()

    plotter.enable_point_picking(
        callback=on_pick,
        picker="point",
        use_picker=True,
        left_clicking=True,
        show_message=False,
        point_size=12.0,
        color="red",
    )
    plotter.add_key_event("a", accept_preview)
    plotter.add_key_event("b", toggle_eraser_mode)
    plotter.add_key_event("p", toggle_paint_mode)
    plotter.add_key_event("r", clear_preview)
    plotter.add_key_event("c", clear_accepted)
    refresh_instructions()
    plotter.show()
    accepted = split_manual_regions_into_connected_components(state["accepted"], vertex_neighbors)
    print(f"Manual surface selection: {len(accepted)} connected constraint group(s).")
    return accepted, point_normals


def append_manual_surface_regions(
    points: np.ndarray,
    normals: np.ndarray,
    group_ids: np.ndarray,
    mesh_points: np.ndarray,
    mesh_normals: np.ndarray,
    manual_regions: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Append accepted interactive regions as separate surface-quality groups."""
    if not manual_regions:
        return points, normals, group_ids

    next_group_id = 0 if group_ids.size == 0 else int(np.max(group_ids)) + 1
    manual_points = []
    manual_normals = []
    manual_group_ids = []
    for region_index, region in enumerate(manual_regions):
        if region.size == 0:
            continue
        manual_points.append(mesh_points[region])
        manual_normals.append(mesh_normals[region])
        manual_group_ids.append(np.full(region.size, next_group_id + region_index, dtype=np.int64))

    if not manual_points:
        return points, normals, group_ids
    return (
        np.concatenate((points, *manual_points), axis=0),
        np.concatenate((normals, *manual_normals), axis=0),
        np.concatenate((group_ids, *manual_group_ids), axis=0),
    )


def build_constrained_surface_edges(
    mesh: pv.PolyData,
    constrained_points: np.ndarray,
    constrained_normals: np.ndarray,
    constrained_group_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return constrained mesh edges with their selected target normals."""
    edge_mesh = mesh.copy(deep=True).triangulate()
    mesh_points = np.asarray(edge_mesh.points, dtype=np.float32)

    point_to_index = {tuple(point): index for index, point in enumerate(mesh_points)}
    vertex_groups: list[set[int]] = [set() for _ in range(edge_mesh.n_points)]
    vertex_group_normal_sums: dict[tuple[int, int], np.ndarray] = {}
    vertex_group_normal_counts: dict[tuple[int, int], int] = {}
    for point, normal, group_id in zip(constrained_points, constrained_normals, constrained_group_ids):
        point_index = point_to_index.get(tuple(point))
        if point_index is not None:
            normalized_group_id = int(group_id)
            vertex_groups[point_index].add(normalized_group_id)
            key = (point_index, normalized_group_id)
            vertex_group_normal_sums[key] = vertex_group_normal_sums.get(key, np.zeros(3, dtype=np.float32)) + normal
            vertex_group_normal_counts[key] = vertex_group_normal_counts.get(key, 0) + 1

    faces = np.asarray(edge_mesh.faces).reshape((-1, 4))[:, 1:4]
    all_edges = np.concatenate((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0)
    all_edges.sort(axis=1)
    unique_edges = np.unique(all_edges, axis=0)
    selected_edges: list[np.ndarray] = []
    normals_a: list[np.ndarray] = []
    normals_b: list[np.ndarray] = []
    for edge in unique_edges:
        first, second = int(edge[0]), int(edge[1])
        for group_id in vertex_groups[first] & vertex_groups[second]:
            first_key = (first, group_id)
            second_key = (second, group_id)
            first_normal = vertex_group_normal_sums[first_key] / vertex_group_normal_counts[first_key]
            second_normal = vertex_group_normal_sums[second_key] / vertex_group_normal_counts[second_key]
            first_normal /= np.linalg.norm(first_normal) + 1e-8
            second_normal /= np.linalg.norm(second_normal) + 1e-8
            selected_edges.append(edge)
            normals_a.append(first_normal)
            normals_b.append(second_normal)
    selected_edges = np.asarray(selected_edges, dtype=np.int64)
    print(f"Surface edge constraints: {selected_edges.shape[0]} selected edges.")
    if selected_edges.size == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty, empty, empty
    return (
        mesh_points[selected_edges[:, 0]],
        mesh_points[selected_edges[:, 1]],
        np.asarray(normals_a, dtype=np.float32),
        np.asarray(normals_b, dtype=np.float32),
    )


def build_sampled_surface_neighbor_edges(points: np.ndarray, normals: np.ndarray, group_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pair each sampled point with its nearest compatible sampled neighbour."""
    pairs = set()
    rejected_distance = 0
    rejected_normal = 0
    max_angle_cosine = float(np.cos(np.deg2rad(SURFACE_EDGE_NEIGHBOR_NORMAL_ANGLE_DEGREES)))
    for group_id in np.unique(group_ids):
        indices = np.flatnonzero(group_ids == group_id)
        if len(indices) < 2:
            continue
        distances, local_neighbors = cKDTree(points[indices]).query(points[indices], k=2)
        for local_index, (distance, neighbor) in enumerate(zip(distances[:, 1], local_neighbors[:, 1])):
            first, second = int(indices[local_index]), int(indices[neighbor])
            if distance > SURFACE_EDGE_NEIGHBOR_RADIUS_WORLD:
                rejected_distance += 1
            elif np.dot(normals[first], normals[second]) < max_angle_cosine:
                rejected_normal += 1
            else:
                pairs.add(tuple(sorted((first, second))))
    local_pair_count = len(pairs)
    if SURFACE_EDGE_LONG_RANGE_PAIRS_PER_GROUP > 0:
        rng = np.random.default_rng(SURFACE_EDGE_LONG_RANGE_SEED)
        long_range_added = 0
        for group_id in np.unique(group_ids):
            indices = np.flatnonzero(group_ids == group_id)
            if len(indices) < 2:
                continue
            target_count = min(int(SURFACE_EDGE_LONG_RANGE_PAIRS_PER_GROUP), len(indices) * (len(indices) - 1) // 2)
            accepted = 0
            attempts = 0
            # Randomly connect distant samples in the same preserved patch.
            # The loss itself continuously gates each endpoint by its current
            # scalar-gradient/target-normal alignment.
            while accepted < target_count and attempts < target_count * 40:
                first_local, second_local = rng.integers(0, len(indices), size=2)
                attempts += 1
                if first_local == second_local:
                    continue
                first, second = sorted((int(indices[first_local]), int(indices[second_local])))
                if np.linalg.norm(points[first] - points[second]) < SURFACE_EDGE_LONG_RANGE_MIN_DISTANCE_WORLD:
                    continue
                pair = (first, second)
                if pair in pairs:
                    continue
                pairs.add(pair)
                accepted += 1
                long_range_added += 1
        print(
            "Random long-range surface pairs: "
            f"added={long_range_added}, per_group={SURFACE_EDGE_LONG_RANGE_PAIRS_PER_GROUP}, "
            f"minimum_distance={SURFACE_EDGE_LONG_RANGE_MIN_DISTANCE_WORLD:.4f}."
        )
    if not pairs:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty, empty, empty
    pair_indices = np.asarray(sorted(pairs), dtype=np.int64)
    covered = np.unique(pair_indices.reshape(-1)).size
    print(
        "Sampled surface edge constraints: "
        f"pairs={len(pair_indices)} (local={local_pair_count}, long_range={len(pair_indices) - local_pair_count}), "
        f"covered={covered}/{len(points)} ({100.0 * covered / max(len(points), 1):.1f}%), "
        f"radius={SURFACE_EDGE_NEIGHBOR_RADIUS_WORLD:.4f}, "
        f"rejected_distance={rejected_distance}, rejected_normal={rejected_normal}."
    )
    return points[pair_indices[:, 0]], points[pair_indices[:, 1]], normals[pair_indices[:, 0]], normals[pair_indices[:, 1]]


def build_contour_values(model_scalars: np.ndarray, range_vals: torch.Tensor) -> np.ndarray:
    """Build contour values using this script's own visualization range settings."""
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


class SlopeGroupedSurfaceNormalLoader:
    """Yield every non-empty surface-inclination bin as one normal-loss batch."""

    def __init__(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
        up_direction: np.ndarray,
        bin_count: int,
        max_angle_degrees: float,
        seed: int,
    ) -> None:
        if points.shape[0] != normals.shape[0]:
            raise ValueError("Surface points and normals must have matching lengths.")
        if bin_count <= 0 or max_angle_degrees <= 0.0:
            raise ValueError("Surface-normal grouping requires positive bin and angle values.")

        self.points = points
        self.normals = normals
        self._iteration = 0
        self.seed = seed

        up = torch.as_tensor(up_direction, dtype=normals.dtype, device=normals.device).reshape(3)
        up = up / (torch.linalg.vector_norm(up) + 1e-8)
        normal_dot = torch.clamp(torch.sum(normals * up, dim=1), -1.0, 1.0)
        angles = torch.rad2deg(torch.arccos(normal_dot))
        # Values above the configured range remain in the steepest bin, so
        # manual regions cannot silently disappear from the normal loss.
        edges = torch.linspace(
            0.0,
            max_angle_degrees,
            bin_count + 1,
            dtype=angles.dtype,
            device=angles.device,
        )
        bin_ids = torch.bucketize(angles, edges[1:-1])
        self.bin_indices = [torch.nonzero(bin_ids == bin_index, as_tuple=False).flatten() for bin_index in range(bin_count)]
        self.bin_indices = [indices for indices in self.bin_indices if indices.numel() > 0]
        if not self.bin_indices:
            raise ValueError("No surface-normal points are available for stratification.")

        counts = [int(indices.numel()) for indices in self.bin_indices]
        print(
            "Surface-normal slope groups: "
            f"{len(self.bin_indices)} whole groups used as normal-loss batches with point counts {counts}."
        )

    def __len__(self) -> int:
        return len(self.bin_indices)

    def __iter__(self):
        generator = torch.Generator(device=self.points.device)
        generator.manual_seed(self.seed + self._iteration)
        self._iteration += 1
        group_order = torch.randperm(len(self.bin_indices), generator=generator, device=self.points.device)
        for group_index in group_order.tolist():
            batch_indices = self.bin_indices[group_index]
            yield self.points[batch_indices], self.normals[batch_indices]


class SurfaceGroupAngleCurriculum:
    """Unlock steeper normal-angle bins only after each group's frontier settles."""

    def __init__(self, points, normals, group_ids, up_direction) -> None:
        self.points, self.normals, self.group_ids = points, normals, group_ids
        up = torch.as_tensor(up_direction, dtype=normals.dtype, device=normals.device).reshape(3)
        self.up = up / (torch.linalg.vector_norm(up) + 1e-8)
        self.cells: dict[int, list[torch.Tensor]] = {}
        self.active_bin: dict[int, int] = {}
        self.ema: dict[int, float] = {}
        self.passes: dict[int, int] = {}
        for group in torch.unique(group_ids, sorted=True).tolist():
            indices = torch.nonzero(group_ids == group, as_tuple=False).flatten()
            angles = torch.rad2deg(torch.arccos(torch.clamp(torch.sum(normals[indices] * self.up, dim=1), -1.0, 1.0)))
            low, high = float(angles.min()), float(angles.max())
            if high - low < 1e-6:
                bins = [indices]
            else:
                edges = torch.linspace(low, high, SURFACE_GROUP_ANGLE_CURRICULUM_BIN_COUNT + 1, device=points.device)
                ids = torch.bucketize(angles, edges[1:-1])
                bins = [indices[ids == bin_id] for bin_id in range(SURFACE_GROUP_ANGLE_CURRICULUM_BIN_COUNT)]
                bins = [cell for cell in bins if cell.numel() > 0]
            self.cells[int(group)] = bins
            self.active_bin[int(group)] = 0
            self.passes[int(group)] = 0
        print("Surface angle curriculum: " + ", ".join(f"group {group}: {len(cells)} bins" for group, cells in self.cells.items()))

    def active_cells(self):
        return [cell for group, cells in self.cells.items() for cell in cells[: self.active_bin[group] + 1]]

    def active_group_tensors(self):
        result = []
        for group, cells in self.cells.items():
            indices = torch.cat(cells[: self.active_bin[group] + 1])
            result.append((self.points[indices].detach(), self.normals[indices].detach()))
        return result

    def __len__(self):
        return max(1, len(self.active_cells()))

    def __iter__(self):
        for indices in self.active_cells():
            yield self.points[indices].detach().requires_grad_(True), self.normals[indices]

    def update(self, scalar_net, epoch: int) -> None:
        if (epoch + 1) % SURFACE_GROUP_ANGLE_CURRICULUM_CHECK_EVERY_EPOCHS:
            return
        for group, cells in self.cells.items():
            frontier = self.active_bin[group]
            if frontier >= len(cells) - 1:
                continue
            indices = cells[frontier]
            loss_sum, count = 0.0, 0
            for start in range(0, indices.numel(), SURFACE_GROUP_ANGLE_CURRICULUM_EVAL_BATCH_SIZE):
                batch = self.points[indices[start : start + SURFACE_GROUP_ANGLE_CURRICULUM_EVAL_BATCH_SIZE]].detach().requires_grad_(True)
                normals = self.normals[indices[start : start + SURFACE_GROUP_ANGLE_CURRICULUM_EVAL_BATCH_SIZE]]
                grads = scalar_net(batch, compute_hess=False)["grads"]
                error = grads / (torch.linalg.vector_norm(grads, dim=1, keepdim=True) + 1e-5) - normals
                loss_sum += float(error.square().sum().detach().cpu())
                count += int(error.numel())
            value = loss_sum / max(count, 1)
            previous = self.ema.get(group, value)
            self.ema[group] = SURFACE_GROUP_ANGLE_CURRICULUM_EMA_DECAY * previous + (1.0 - SURFACE_GROUP_ANGLE_CURRICULUM_EMA_DECAY) * value
            self.passes[group] = self.passes[group] + 1 if self.ema[group] <= SURFACE_GROUP_ANGLE_CURRICULUM_LOSS_THRESHOLD else 0
            if self.passes[group] >= SURFACE_GROUP_ANGLE_CURRICULUM_REQUIRED_CHECKS:
                self.active_bin[group] += 1
                self.passes[group] = 0
                print(f"Surface curriculum unlocked group {group} bin {self.active_bin[group] + 1}/{len(cells)} at epoch {epoch + 1}.")


class ResampledInputPointLoader:
    """Redraw a random subset of the full input grid at every epoch iteration."""

    def __init__(
        self,
        points: torch.Tensor,
        batch_size: int,
        points_per_epoch: int,
        seed: int,
    ) -> None:
        if batch_size <= 0 or points_per_epoch < 0:
            raise ValueError("Input resampling requires a positive batch size and nonnegative point count.")
        self.points = points
        self.batch_size = batch_size
        self.points_per_epoch = points.shape[0] if points_per_epoch == 0 else min(points_per_epoch, points.shape[0])
        self.seed = seed
        self._iteration = 0

    def __len__(self) -> int:
        return int(np.ceil(self.points_per_epoch / self.batch_size))

    def __iter__(self):
        generator = torch.Generator(device=self.points.device)
        generator.manual_seed(self.seed + self._iteration)
        self._iteration += 1
        indices = torch.randperm(self.points.shape[0], generator=generator, device=self.points.device)
        indices = indices[: self.points_per_epoch]
        for start in range(0, self.points_per_epoch, self.batch_size):
            batch_points = self.points[indices[start : start + self.batch_size]]
            yield batch_points.detach().requires_grad_(True)


def build_hybrid_dataloaders(
    input_points: torch.Tensor,
    surface_points: torch.Tensor,
    surface_normals: torch.Tensor,
    support_free_surface_points: torch.Tensor,
    support_free_surface_normals: torch.Tensor,
    surface_group_ids: torch.Tensor | None,
    surface_edge_points_a: torch.Tensor | None,
    surface_edge_points_b: torch.Tensor | None,
    surface_edge_normals_a: torch.Tensor | None,
    surface_edge_normals_b: torch.Tensor | None,
    base_points: torch.Tensor,
    base_target_scalars: torch.Tensor,
    batch_size: int,
    surface_group_batch_size: int,
    surface_edge_batch_size: int,
    resample_input_points_each_epoch: bool = False,
    input_points_per_epoch: int = 0,
    input_resampling_seed: int = 0,
):
    surface_curriculum = None
    if resample_input_points_each_epoch:
        input_loader = ResampledInputPointLoader(
            points=input_points,
            batch_size=batch_size,
            points_per_epoch=input_points_per_epoch,
            seed=input_resampling_seed,
        )
    else:
        input_loader = DataLoader(input_points, batch_size=batch_size, shuffle=True)
    def optional_loader(tensors, enabled, label):
        if tensors[0].shape[0] == 0:
            if enabled:
                raise ValueError(f'No {label} samples for an enabled loss; adjust the constraint selection or disable that objective.')
            # A finite empty batch lets zip/cycling advance without asking
            # RandomSampler to shuffle an empty, inactive constraint pool.
            return [tuple(tensors)]
        return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True)

    if surface_points.shape[0] == 0:
        surface_loader = optional_loader((surface_points, surface_normals),
            ENABLE_SURFACE_DIRECTION_LOSS or ENABLE_SURFACE_GROUP_SCALAR_LOSS or
            ENABLE_SURFACE_EDGE_SCALAR_LOSS or SURFACE_EDGE_REPLACES_GROUP_SCALAR_LOSS, 'surface constraint')
    elif ENABLE_SURFACE_GROUP_ANGLE_CURRICULUM:
        surface_curriculum = SurfaceGroupAngleCurriculum(surface_points, surface_normals, surface_group_ids, UP_DIRECTION)
        surface_loader = surface_curriculum
    elif SURFACE_NORMAL_STRATIFICATION_ENABLED:
        surface_loader = SlopeGroupedSurfaceNormalLoader(
            points=surface_points,
            normals=surface_normals,
            up_direction=UP_DIRECTION,
            bin_count=SURFACE_NORMAL_STRATIFICATION_BIN_COUNT,
            max_angle_degrees=SURFACE_NORMAL_STRATIFICATION_MAX_ANGLE_DEGREES,
            seed=SURFACE_NORMAL_STRATIFICATION_SEED,
        )
    else:
        surface_loader = DataLoader(TensorDataset(surface_points, surface_normals), batch_size=batch_size, shuffle=True)
    support_free_surface_loader = optional_loader(
        (support_free_surface_points, support_free_surface_normals), ENABLE_SUPPORT_FREE_SURFACE_LOSS, 'support-free surface')
    base_loader = optional_loader((base_points, base_target_scalars),
        ENABLE_BASE_DIRECTION_LOSS or ENABLE_BASE_SCALAR_LOSS, 'base constraint')
    surface_group_loader = None
    if surface_curriculum is not None:
        class ActiveGroupLoader:
            def __len__(self):
                return 1

            def __iter__(self):
                yield surface_curriculum.active_group_tensors()

        surface_group_loader = ActiveGroupLoader()
    elif surface_group_ids is not None and surface_group_ids.numel() > 0:
        unique_group_ids = torch.unique(surface_group_ids, sorted=True)
        surface_group_dataset = [
            (
                surface_points[surface_group_ids == group_id].detach().clone(),
                surface_normals[surface_group_ids == group_id].detach().clone(),
            )
            for group_id in unique_group_ids
        ]
        surface_group_loader = DataLoader(
            surface_group_dataset,
            batch_size=surface_group_batch_size,
            shuffle=True,
            collate_fn=lambda batch: batch,
        )

    surface_edge_loader = None
    if surface_edge_points_a is not None and surface_edge_points_a.shape[0] > 0:
        surface_edge_loader = DataLoader(
            TensorDataset(
                surface_edge_points_a,
                surface_edge_points_b,
                surface_edge_normals_a,
                surface_edge_normals_b,
            ),
            batch_size=surface_edge_batch_size,
            shuffle=True,
        )

    loader_lengths = [len(input_loader), len(surface_loader), len(support_free_surface_loader), len(base_loader)]
    if surface_group_loader is not None:
        loader_lengths.append(len(surface_group_loader))
    if surface_edge_loader is not None:
        loader_lengths.append(len(surface_edge_loader))
    max_len = max(loader_lengths)

    def maybe_cycle(loader):
        if len(loader) >= max_len:
            return loader

        def repeat_loader():
            while True:
                yield from loader

        return repeat_loader()

    return {
        "input_loader": input_loader,
        "surface_loader": surface_loader,
        "support_free_surface_loader": support_free_surface_loader,
        "base_loader": base_loader,
        "surface_group_loader": surface_group_loader,
        "surface_edge_loader": surface_edge_loader,
        "input_loader_train": maybe_cycle(input_loader),
        "surface_loader_train": maybe_cycle(surface_loader),
        "support_free_surface_loader_train": maybe_cycle(support_free_surface_loader),
        "base_loader_train": maybe_cycle(base_loader),
        "surface_group_loader_train": None if surface_group_loader is None else maybe_cycle(surface_group_loader),
        "surface_edge_loader_train": None if surface_edge_loader is None else maybe_cycle(surface_edge_loader),
        "surface_curriculum": surface_curriculum,
        "num_train_batches": max_len,
    }


def compute_surface_group_scalar_loss(
    scalar_net,
    batch_surface_groups: list[tuple[torch.Tensor, torch.Tensor]],
    scalar_scale: torch.Tensor,
) -> torch.Tensor:
    """Encourage aligned points to share similar estimated layer heights.

    Every point in each connected group remains in the loss. Its contribution
    is a smooth, detached weight based on gradient-normal alignment, so points
    near the alignment threshold transition gradually instead of appearing or
    disappearing from the scalar constraint between optimization steps.
    """
    if not batch_surface_groups:
        first_param = next(scalar_net.parameters())
        return torch.tensor(0.0, device=first_param.device, dtype=first_param.dtype)

    first_group_points, _ = batch_surface_groups[0]
    group_sizes = torch.tensor(
        [group_points.shape[0] for group_points, _ in batch_surface_groups],
        device=first_group_points.device,
    )
    if int(group_sizes.sum().item()) == 0:
        return torch.tensor(0.0, device=first_group_points.device, dtype=first_group_points.dtype)

    # DataLoader/group construction may strip requires_grad; SIREN computes
    # spatial gradients internally, so this independent input leaf is needed.
    flattened_points = torch.cat([group_points for group_points, _ in batch_surface_groups], dim=0)
    flattened_points = flattened_points.detach().requires_grad_(True)
    flattened_normals = torch.cat([group_normals for _, group_normals in batch_surface_groups], dim=0)
    group_ids = torch.repeat_interleave(
        torch.arange(len(batch_surface_groups), device=flattened_points.device, dtype=torch.long),
        group_sizes,
    )

    group_outputs = scalar_net(flattened_points, compute_hess=False)
    group_scalars = group_outputs["scalars"].reshape(-1)
    group_grads = group_outputs["grads"]
    group_grad_norms = torch.norm(group_grads, dim=1)
    group_grad_directions = group_grads / (group_grad_norms.unsqueeze(1) + 1e-5)

    min_allowed_dot = float(np.cos(np.deg2rad(SURFACE_GROUP_ALIGNMENT_MAX_ANGLE_DEGREES)))
    alignment_dot = torch.sum(group_grad_directions * flattened_normals, dim=1)
    alignment_weights = torch.sigmoid(
        (alignment_dot - min_allowed_dot) / SURFACE_GROUP_ALIGNMENT_WEIGHT_TEMPERATURE
    ).detach()

    num_groups = len(batch_surface_groups)
    group_sum = torch.zeros(num_groups, device=flattened_points.device, dtype=group_scalars.dtype)
    group_weight_sum = torch.zeros(num_groups, device=flattened_points.device, dtype=group_scalars.dtype)
    group_sum.index_add_(0, group_ids, alignment_weights * group_scalars)
    group_weight_sum.index_add_(0, group_ids, alignment_weights)

    group_mean = group_sum / torch.clamp(group_weight_sum, min=1e-8)
    centered_difference = group_scalars - group_mean[group_ids]
    world_scale = scalar_scale.to(device=flattened_points.device, dtype=group_scalars.dtype)
    group_grad_norm_sum = torch.zeros(num_groups, device=flattened_points.device, dtype=group_scalars.dtype)
    group_grad_norm_sum.index_add_(0, group_ids, alignment_weights * group_grad_norms)
    group_mean_grad_norm = group_grad_norm_sum / torch.clamp(group_weight_sum, min=1e-8)
    if SURFACE_SCALAR_DIVIDE_BY_GRADIENT:
        denominator = group_mean_grad_norm[group_ids]
        if SURFACE_SCALAR_GRADIENT_DENOMINATOR_DETACH:
            denominator = denominator.detach()
        estimated_world_height_difference = world_scale * centered_difference / torch.clamp(
            denominator,
            min=SURFACE_GROUP_GRADIENT_DENOMINATOR_EPSILON,
        )
    else:
        estimated_world_height_difference = world_scale * centered_difference
    if SURFACE_GROUP_SCALAR_LOSS_TYPE == "l1":
        per_point_error = torch.abs(estimated_world_height_difference)
    elif SURFACE_GROUP_SCALAR_LOSS_TYPE == "l2":
        per_point_error = estimated_world_height_difference * estimated_world_height_difference
    else:
        raise ValueError(
            "SURFACE_GROUP_SCALAR_LOSS_TYPE must be 'l1' or 'l2', "
            f"got {SURFACE_GROUP_SCALAR_LOSS_TYPE!r}."
        )

    group_error_sum = torch.zeros(num_groups, device=flattened_points.device, dtype=group_scalars.dtype)
    group_error_sum.index_add_(0, group_ids, alignment_weights * per_point_error)
    group_loss = group_error_sum / torch.clamp(group_weight_sum, min=1e-8)
    return torch.mean(group_loss)


def compute_surface_edge_scalar_loss(
    scalar_net,
    edge_points_a: torch.Tensor,
    edge_points_b: torch.Tensor,
    edge_normals_a: torch.Tensor,
    edge_normals_b: torch.Tensor,
    scalar_scale: torch.Tensor,
) -> torch.Tensor:
    """Penalize scalar jumps across constrained edges with smooth alignment weights."""
    point_count = edge_points_a.shape[0]
    if point_count == 0:
        return torch.tensor(0.0, device=edge_points_a.device, dtype=edge_points_a.dtype)

    edge_points = torch.cat((edge_points_a, edge_points_b), dim=0).detach().requires_grad_(True)
    edge_normals = torch.cat((edge_normals_a, edge_normals_b), dim=0)
    outputs = scalar_net(edge_points, compute_hess=False)
    scalars = outputs["scalars"].reshape(-1)
    gradients = outputs["grads"]
    normal_components = torch.sum(gradients * edge_normals, dim=1)
    gradient_sq_norms = torch.sum(gradients * gradients, dim=1)
    cosine_sq = float(np.cos(np.deg2rad(SURFACE_EDGE_ALIGNMENT_MAX_ANGLE_DEGREES)) ** 2)
    alignment_sq = normal_components.square() / torch.clamp(gradient_sq_norms, min=1e-8)
    angular_weights = torch.sigmoid(
        (alignment_sq - cosine_sq) / SURFACE_EDGE_ALIGNMENT_WEIGHT_TEMPERATURE
    )
    positive_weights = torch.sigmoid(normal_components / SURFACE_EDGE_POSITIVE_WEIGHT_TEMPERATURE)
    point_weights = (angular_weights * positive_weights).detach()
    first_weights = point_weights[:point_count]
    second_weights = point_weights[point_count:]
    if SURFACE_EDGE_ALIGNMENT_PAIR_WEIGHT == "product":
        edge_weights = first_weights * second_weights
    elif SURFACE_EDGE_ALIGNMENT_PAIR_WEIGHT == "min":
        # A scalar-consistency link is only as trustworthy as its less
        # aligned endpoint, without attenuating two moderate confidences twice.
        edge_weights = torch.minimum(first_weights, second_weights)
    else:
        raise ValueError(
            "SURFACE_EDGE_ALIGNMENT_PAIR_WEIGHT must be 'product' or 'min', "
            f"got {SURFACE_EDGE_ALIGNMENT_PAIR_WEIGHT!r}."
        )
    scalar_difference = scalars[:point_count] - scalars[point_count:]
    world_scale = scalar_scale.to(device=scalars.device, dtype=scalars.dtype)
    # Use the average squared endpoint gradient directly. This avoids a
    # square-root while retaining the intended inverse-gradient scaling.
    if SURFACE_SCALAR_DIVIDE_BY_GRADIENT:
        mean_endpoint_gradient_sq = 0.5 * (
            gradient_sq_norms[:point_count] + gradient_sq_norms[point_count:]
        )
        if SURFACE_SCALAR_GRADIENT_DENOMINATOR_DETACH:
            mean_endpoint_gradient_sq = mean_endpoint_gradient_sq.detach()
        denominator_epsilon_sq = SURFACE_GROUP_GRADIENT_DENOMINATOR_EPSILON ** 2
        edge_error = world_scale.square() * scalar_difference.square() / torch.clamp(
            mean_endpoint_gradient_sq,
            min=denominator_epsilon_sq,
        )
    else:
        edge_error = world_scale.square() * scalar_difference.square()
    # Keep the accumulated pair weight. Dividing by sum(edge_weights) turns
    # the loss into a weighted mean and hides how many compatible neighbours
    # are actually contributing in a batch.
    return torch.sum(edge_weights * edge_error)


def compute_sampled_input_curvature_loss(
    scalar_net,
    input_points: torch.Tensor,
    sample_fraction: float,
    scalar_scale=None,
) -> torch.Tensor:
    """Evaluate raw Hessian or physical layer curvature on a random subset."""
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError("INPUT_CURVATURE_SAMPLE_FRACTION must be in (0, 1].")
    if sample_fraction == 1.:
        sampled_points = input_points.detach().requires_grad_(True)
    else:
        sample_count = max(1, int(round(input_points.shape[0] * sample_fraction)))
        sample_indices = torch.randperm(input_points.shape[0], device=input_points.device)[:sample_count]
        sampled_points = input_points[sample_indices].detach().requires_grad_(True)
    outputs = scalar_net(sampled_points, compute_hess=True)
    hessian = torch.stack((outputs["HX2"], outputs["HY2"], outputs["HZ2"]), dim=1)
    if INPUT_CURVATURE_MODE == 'geometric':
        if scalar_scale is None:
            raise ValueError('Geometric curvature requires the world coordinate scale')
        from layerCurvature import layer_curvature_loss
        return layer_curvature_loss(outputs['grads'], hessian, scalar_scale, INPUT_CURVATURE_RADIUS_MM)
    return torch.mean(hessian * hessian)


def compute_tangent_twist_loss(
    surface_grads: torch.Tensor,
    surface_normals: torch.Tensor,
    up_direction: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Penalize sideways tangent gradients where projected up is well defined.

    The direction ``cross(N, project_tangent(up))`` is tangent to the surface
    and perpendicular to the desired projected build direction. Its squared
    gradient component suppresses twist without fixing the gradient magnitude.
    """
    projected_up = up_direction - torch.sum(surface_normals * up_direction, dim=1, keepdim=True) * surface_normals
    projected_norm = torch.linalg.vector_norm(projected_up, dim=1)
    eligible = projected_norm >= TANGENT_TWIST_MIN_PROJECTED_UP_NORM
    eligible_count = int(eligible.sum().detach().cpu())
    if eligible_count == 0:
        return torch.zeros((), dtype=surface_grads.dtype, device=surface_grads.device), 0

    tangent_up = projected_up[eligible] / projected_norm[eligible].unsqueeze(1)
    side_direction = torch.linalg.cross(surface_normals[eligible], tangent_up, dim=1)
    side_direction = side_direction / (torch.linalg.vector_norm(side_direction, dim=1, keepdim=True) + 1e-8)
    sideways_gradient = torch.sum(surface_grads[eligible] * side_direction, dim=1)
    return torch.mean(sideways_gradient.square()), eligible_count


def parameter_gradient_cosine(
    first_loss: torch.Tensor,
    second_loss: torch.Tensor,
    scalar_net: torch.nn.Module,
) -> float:
    """Measure whether two losses reinforce one another in parameter space."""
    parameters = [parameter for parameter in scalar_net.parameters() if parameter.requires_grad]
    first_grads = torch.autograd.grad(first_loss, parameters, retain_graph=True, allow_unused=True)
    second_grads = torch.autograd.grad(second_loss, parameters, retain_graph=True, allow_unused=True)
    dot = torch.zeros((), device=first_loss.device, dtype=first_loss.dtype)
    first_norm_sq = torch.zeros_like(dot)
    second_norm_sq = torch.zeros_like(dot)
    for first_grad, second_grad in zip(first_grads, second_grads):
        if first_grad is None or second_grad is None:
            continue
        dot = dot + torch.sum(first_grad * second_grad)
        first_norm_sq = first_norm_sq + torch.sum(first_grad.square())
        second_norm_sq = second_norm_sq + torch.sum(second_grad.square())
    cosine = dot / (torch.sqrt(first_norm_sq * second_norm_sq) + 1e-12)
    return float(cosine.detach().cpu())


def print_loss_gradient_contributions(weighted_components: dict[str, torch.Tensor], scalar_net: torch.nn.Module, epoch: int) -> None:
    """Report which early-training loss has the strongest parameter-space pull."""
    active_components = {name: value for name, value in weighted_components.items() if value.requires_grad}
    if not active_components:
        return
    parameters = [parameter for parameter in scalar_net.parameters() if parameter.requires_grad]
    total_loss = sum(active_components.values())
    total_grads = torch.autograd.grad(total_loss, parameters, retain_graph=True, allow_unused=True)
    rows = []
    for name, component in active_components.items():
        component_grads = torch.autograd.grad(component, parameters, retain_graph=True, allow_unused=True)
        component_norm_sq = torch.zeros((), device=total_loss.device)
        other_norm_sq = torch.zeros_like(component_norm_sq)
        dot_other = torch.zeros_like(component_norm_sq)
        for component_grad, total_grad in zip(component_grads, total_grads):
            if component_grad is None or total_grad is None:
                continue
            other_grad = total_grad - component_grad
            component_norm_sq += torch.sum(component_grad.square())
            other_norm_sq += torch.sum(other_grad.square())
            dot_other += torch.sum(component_grad * other_grad)
        component_norm = torch.sqrt(component_norm_sq)
        cosine = dot_other / (component_norm * torch.sqrt(other_norm_sq) + 1e-12)
        rows.append(f"{name}:norm={float(component_norm.detach().cpu()):.2e},cos_other={float(cosine.detach().cpu()):+.3f}")
    print(f"initial_loss_gradients epoch={epoch + 1}: " + " | ".join(rows), flush=True)


def compute_soft_banded_gradient_ratio_loss(
    scalar_values: torch.Tensor,
    gradients: torch.Tensor,
    scalar_range: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standalone complete-set pointwise loss; trainers use shared references."""
    ratio_loss, _ = sampled_gradient_ratio_loss(
        scalar_values, gradients, GRADIENT_BAND_COUNT,
        float(np.sqrt(GRADIENT_BAND_MAX_SQUARED_RATIO)), scalar_range,
    )
    squared_gradient_norms = torch.sum(gradients * gradients, dim=1)
    floor_loss = torch.mean(torch.relu(GRADIENT_BAND_SQUARED_FLOOR - squared_gradient_norms).square())
    return ratio_loss, floor_loss


_DEFAULT_GRADIENT_RATIO_LOSS = compute_soft_banded_gradient_ratio_loss


def compute_gradient_ratio_regularization(
    scalar_net,
    input_outputs: dict[str, torch.Tensor],
    norm_control_points: torch.Tensor | None,
    reference: BandGradientReference | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared sampling and objective for the fused and separate trainers."""
    global _LAST_NORM_CONTROL_SAMPLE
    _LAST_NORM_CONTROL_SAMPLE = None
    outputs = input_outputs
    memberships = None
    if ENABLE_BALANCED_NORM_BATCHES and reference is not None:
        points, memberships = reference.sample()
        _LAST_NORM_CONTROL_SAMPLE = (points.detach(), memberships[0].detach())
        outputs = scalar_net(points, compute_hess=False)
    if reference is not None:
        ratio_loss, observed_ratio = reference.loss(outputs["scalars"], outputs["grads"], memberships)
    else:
        ratio_loss, observed_ratio = sampled_gradient_ratio_loss(
            outputs["scalars"], outputs["grads"], GRADIENT_BAND_COUNT,
            float(np.sqrt(GRADIENT_BAND_MAX_SQUARED_RATIO)),
        )
    squared_norms = outputs["grads"].square().sum(dim=1)
    floor_loss = torch.relu(GRADIENT_BAND_SQUARED_FLOOR - squared_norms).square().mean()
    # Comparison runners deliberately replace this public loss hook. Keep their
    # requested objective while reporting the same actual sampled ratio metric.
    if compute_soft_banded_gradient_ratio_loss is not _DEFAULT_GRADIENT_RATIO_LOSS:
        ratio_loss, floor_loss = compute_soft_banded_gradient_ratio_loss(outputs["scalars"], outputs["grads"])
    return ratio_loss, floor_loss, observed_ratio


def compute_squared_gradient_norm_curriculum_losses(gradients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unit-norm and bounded-range losses without square roots."""
    squared_norms = torch.sum(gradients * gradients, dim=1)
    unit_loss = torch.mean((squared_norms - 1.0).square())
    range_loss = torch.mean(
        torch.relu(GRADIENT_NORM_MIN_SQUARED - squared_norms).square()
        + torch.relu(squared_norms - GRADIENT_NORM_MAX_SQUARED).square()
    )
    return unit_loss, range_loss


def train_scalar_field(
    scalar_net,
    input_loader_train,
    surface_loader_train,
    support_free_surface_loader_train,
    base_loader_train,
    surface_group_loader_train,
    surface_edge_loader_train,
    norm_control_points: torch.Tensor | None,
    scalar_scale: torch.Tensor,
    up_direction: torch.Tensor,
    build_direction: torch.Tensor,
    epochs: int,
    learning_rate: float,
    step_size: int,
    step_gamma: float,
    tool_points_model: torch.Tensor | None = None,
    local_tool_point_sets: list[torch.Tensor] | None = None,
    collision_tool_point_counts: list[int] | None = None,
    range_vals: torch.Tensor | None = None,
    fixed_frame: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    tool_jitter_specs: list[tuple[torch.Tensor, float, float]] | None = None,
    base_module=None,
    sdf_gate_model=None,
    collision_weight: float = 0.0,
    gate_epsilon: float = 0.0,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    z_min=None,
    z_max=None,
    checkpoint_path: str | None = None,
    save_every: int | None = None,
    checkpoint_metadata: dict | None = None,
    surface_curriculum: SurfaceGroupAngleCurriculum | None = None,
    norm_steps_per_epoch: int | None = None,
) -> None:
    if ENABLE_TANGENT_TWIST_LOSS and TANGENT_TWIST_COSINE_CHECK_EVERY_BATCHES < 1:
        raise ValueError("TANGENT_TWIST_COSINE_CHECK_EVERY_BATCHES must be at least 1.")
    if SURFACE_GROUP_SCALAR_START_EPOCH < 0:
        raise ValueError("SURFACE_GROUP_SCALAR_START_EPOCH cannot be negative.")
    if LEARNING_RATE_RAMP_EPOCHS < 0:
        raise ValueError("LEARNING_RATE_RAMP_EPOCHS cannot be negative.")
    optimizer = torch.optim.Adam(scalar_net.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=step_gamma)
    if LEARNING_RATE_WARMUP_EPOCHS > 0:
        if not 0.0 < LEARNING_RATE_WARMUP_FACTOR <= 1.0:
            raise ValueError("LEARNING_RATE_WARMUP_FACTOR must be in (0, 1].")
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate * LEARNING_RATE_WARMUP_FACTOR

    for epoch in range(epochs):
        if STRENGTH_LOSS_PROVIDER is not None:
            STRENGTH_LOSS_PROVIDER.begin_epoch(epoch)
        gradient_reference = (
            BandGradientReference(
                scalar_net, norm_control_points, GRADIENT_BAND_COUNT,
                float(np.sqrt(GRADIENT_BAND_MAX_SQUARED_RATIO)), NORM_POOL_SIZE,
                NORM_POINTS_PER_BAND, norm_steps_per_epoch or len(input_loader_train),
            )
            if norm_control_points is not None else None
        )
        if gradient_reference is not None:
            print(f"norm_reference_points={len(norm_control_points)} "
                  f"reference_ratio_max={float(gradient_reference.probe_max_ratio.cpu()):.4f}")
        epoch_max_height_ratio = scalar_scale.new_tensor(1.0)
        if LEARNING_RATE_RAMP_EPOCHS > 0 and (
            LEARNING_RATE_WARMUP_EPOCHS <= epoch < LEARNING_RATE_WARMUP_EPOCHS + LEARNING_RATE_RAMP_EPOCHS
        ):
            ramp_progress = (epoch - LEARNING_RATE_WARMUP_EPOCHS + 1) / LEARNING_RATE_RAMP_EPOCHS
            ramp_lr = learning_rate * (
                LEARNING_RATE_WARMUP_FACTOR + (1.0 - LEARNING_RATE_WARMUP_FACTOR) * ramp_progress
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = ramp_lr
            print(
                f"Learning-rate ramp epoch {epoch - LEARNING_RATE_WARMUP_EPOCHS + 1}/"
                f"{LEARNING_RATE_RAMP_EPOCHS}: learning rate={ramp_lr:.3e}"
            )
        elif epoch == LEARNING_RATE_WARMUP_EPOCHS:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            message = "Warm-up complete" if LEARNING_RATE_RAMP_EPOCHS == 0 else "Learning-rate ramp complete"
            print(f"{message}: learning rate restored to {learning_rate:.3e}")
        # Accumulate only detached values so logging does not synchronize the
        # GPU once per batch or retain completed autograd graphs.
        epoch_loss = torch.zeros((), device=scalar_scale.device, dtype=scalar_scale.dtype)
        batch_count = 0
        loss_sums = {
            "surface_direction": 0.0,
            "support_free": 0.0,
            "group_scalar": 0.0,
            "edge_scalar": 0.0,
            "gradient_unit": 0.0,
            "gradient_range": 0.0,
            "gradient_ratio": 0.0,
            "gradient_floor": 0.0,
            "input_curvature": 0.0,
            "build_slope": 0.0,
            "base_direction": 0.0,
            "base_scalar": 0.0,
            "collision_raw": 0.0,
            "collision": 0.0,
            "tangent_twist": 0.0,
        }
        twist_cosine_sum = 0.0
        twist_cosine_checks = 0
        twist_accepted_groups = 0
        norm_batch_snapshots = []
        epoch_tool_points_model = tool_points_model
        if local_tool_point_sets is not None and range_vals is not None and fixed_frame is not None and base_module is not None:
            epoch_local_tool_sets = []
            for tool_index, base_tool_points in enumerate(local_tool_point_sets):
                jitter_template, x_step, z_step = (
                    tool_jitter_specs[tool_index]
                    if tool_jitter_specs is not None
                    else (None, 0.0, 0.0)
                )
                epoch_local_tool_sets.append(
                    perturb_tool_points_for_epoch(
                        base_local_tool_points=base_tool_points,
                        jitter_template=jitter_template,
                        x_step=x_step,
                        z_step=z_step,
                    )
                )
            if globals().get('TOOL_AXISYMMETRIC_SPEC') is not None:
                from axisymmetricTool import sample_tool
                epoch_local_tool_sets = sample_tool(TOOL_AXISYMMETRIC_SPEC, scalar_scale.device, seed=TOOL_PERTURB_SEED+epoch)
            epoch_local_tool_points = torch.cat(epoch_local_tool_sets, dim=0)
            epoch_tool_points_model = build_fixed_frame_tool_points_model(
                base=base_module,
                local_tool_points=epoch_local_tool_points,
                range_vals=range_vals,
                fixed_frame=fixed_frame,
            )
        group_source = surface_group_loader_train if surface_group_loader_train is not None else cycle([[]])
        edge_source = (
            surface_edge_loader_train
            if surface_edge_loader_train is not None
            else cycle([(None, None, None, None)])
        )
        batch_sets = zip(
            input_loader_train,
            surface_loader_train,
            support_free_surface_loader_train,
            base_loader_train,
            group_source,
            edge_source,
        )

        for batch_index, (
            batch_input_points,
            (batch_surface_points, batch_surface_normals),
            (batch_support_free_surface_points, batch_support_free_surface_normals),
            (batch_base_points, batch_base_target_scalars),
            batch_surface_groups,
            batch_surface_edges,
        ) in enumerate(batch_sets):
            inp_outs = scalar_net(batch_input_points, compute_hess=False)
            inp_grads = inp_outs["grads"]

            collision_loss = None
            if epoch_tool_points_model is not None and collision_weight > 0.0:
                collision_out = compute_gated_translated_scalar_loss(
                    scalar_field=scalar_net,
                    root_points=batch_input_points,
                    tool_points=epoch_tool_points_model,
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
                active_tool_counts = collision_tool_point_counts or [epoch_tool_points_model.shape[0]]
                collision_loss, _ = reduce_collision_loss(
                    collision_out=collision_out,
                    tool_point_counts=active_tool_counts,
                    reduction=COLLISION_LOSS_REDUCTION,
                    topk_fraction=COLLISION_TOPK_FRACTION,
                    tool_weights=COLLISION_TOOL_WEIGHTS[: len(active_tool_counts)],
                )

            band_gradient_ratio_loss, band_gradient_floor_loss, observed_height_ratio = compute_gradient_ratio_regularization(
                scalar_net, inp_outs, norm_control_points, gradient_reference,
            )
            if SHOW_NORM_CONTROL_BATCHES_EACH_EPOCH and ENABLE_BALANCED_NORM_BATCHES and gradient_reference is not None:
                if _LAST_NORM_CONTROL_SAMPLE is not None:
                    sampled_points, sampled_bands = _LAST_NORM_CONTROL_SAMPLE
                    norm_batch_snapshots.append(
                        (sampled_points.cpu().numpy(), sampled_bands.cpu().numpy())
                    )
            epoch_max_height_ratio = torch.maximum(epoch_max_height_ratio, observed_height_ratio)
            gradient_unit_loss, gradient_range_loss = compute_squared_gradient_norm_curriculum_losses(inp_grads)
            if ENABLE_GRADIENT_NORM_CURRICULUM:
                unit_norm_weight = GRADIENT_NORM_UNIT_WEIGHT if epoch < GRADIENT_NORM_UNIT_EPOCHS else 0.0
                range_norm_weight = GRADIENT_NORM_LATE_RANGE_WEIGHT if epoch >= GRADIENT_NORM_UNIT_EPOCHS else 0.0
                band_ratio_weight = GRADIENT_NORM_LATE_RATIO_WEIGHT if epoch >= GRADIENT_NORM_UNIT_EPOCHS else 0.0
                band_floor_weight = 0.0
            else:
                unit_norm_weight = 0.0
                range_norm_weight = 0.0
                band_ratio_weight = GRADIENT_BAND_RATIO_WEIGHT
                band_floor_weight = GRADIENT_BAND_FLOOR_WEIGHT

            input_curvature_loss = torch.tensor(0.0, device=batch_input_points.device)
            if ENABLE_INPUT_CURVATURE_LOSS:
                input_curvature_loss = compute_sampled_input_curvature_loss(
                    scalar_net,
                    batch_input_points,
                    INPUT_CURVATURE_SAMPLE_FRACTION,
                    scalar_scale=scalar_scale,
                )

            build_direction_slope_loss = torch.tensor(0.0, device=batch_input_points.device)
            if ENABLE_BUILD_DIRECTION_SLOPE_LOSS:
                inp_grad_unit = inp_grads / (torch.norm(inp_grads, dim=1, keepdim=True) + 1e-8)
                build_dot = torch.sum(inp_grad_unit * build_direction, dim=1)
                min_allowed_dot = float(np.cos(np.deg2rad(BUILD_DIRECTION_SLOPE_MAX_ANGLE_DEGREES)))
                build_direction_slope_error = torch.relu(min_allowed_dot - build_dot)
                build_direction_slope_loss = torch.mean(build_direction_slope_error * build_direction_slope_error)

            surface_loss = torch.tensor(0.0, device=batch_input_points.device)
            surface_grads_raw = None
            if ENABLE_SURFACE_DIRECTION_LOSS:
                surface_outs = scalar_net(batch_surface_points, compute_hess=False)
                surface_grads_raw = surface_outs["grads"]
                surface_grads = surface_grads_raw / (torch.norm(surface_grads_raw, dim=1, keepdim=True) + 1e-5)
                # Keep the normal-alignment term as an L2-style penalty so
                # strongly misaligned points are corrected aggressively.
                surface_error = surface_grads - batch_surface_normals
                surface_loss = torch.mean(surface_error * surface_error)

            support_free_surface_loss = torch.tensor(0.0, device=batch_input_points.device)
            if ENABLE_SUPPORT_FREE_SURFACE_LOSS:
                support_free_outs = scalar_net(batch_support_free_surface_points, compute_hess=False)
                support_free_grads = support_free_outs["grads"]
                support_free_grads = support_free_grads / (torch.norm(support_free_grads, dim=1, keepdim=True) + 1e-5)
                min_allowed_dot = float(np.cos(np.deg2rad(90.0 + SUPPORT_FREE_SURFACE_EXTRA_ANGLE_DEGREES)))
                surface_dot = torch.sum(support_free_grads * batch_support_free_surface_normals, dim=1)
                support_free_error = torch.relu(min_allowed_dot - surface_dot)
                support_free_surface_loss = torch.mean(support_free_error * support_free_error)
                #print(support_free_surface_loss)
            surface_group_scalar_loss = torch.tensor(0.0, device=batch_input_points.device)
            group_scalar_is_active = (
                ENABLE_SURFACE_GROUP_SCALAR_LOSS
                and not SURFACE_EDGE_REPLACES_GROUP_SCALAR_LOSS
                and epoch >= SURFACE_GROUP_SCALAR_START_EPOCH
            )
            if group_scalar_is_active:
                surface_group_scalar_loss = compute_surface_group_scalar_loss(
                    scalar_net,
                    batch_surface_groups,
                    scalar_scale=scalar_scale,
                )

            surface_edge_scalar_loss = torch.tensor(0.0, device=batch_input_points.device)
            should_compute_edge_loss = (
                (ENABLE_SURFACE_EDGE_SCALAR_LOSS or SURFACE_EDGE_REPLACES_GROUP_SCALAR_LOSS)
                and batch_surface_edges[0] is not None
                and epoch >= SURFACE_EDGE_SCALAR_START_EPOCH
            )
            if should_compute_edge_loss:
                surface_edge_scalar_loss = compute_surface_edge_scalar_loss(
                    scalar_net,
                    *batch_surface_edges,
                    scalar_scale=scalar_scale,
                )

            base_direction_loss = torch.tensor(0.0, device=batch_input_points.device)
            base_scalar_loss = torch.tensor(0.0, device=batch_input_points.device)
            if ENABLE_BASE_DIRECTION_LOSS or ENABLE_BASE_SCALAR_LOSS:
                base_outs = scalar_net(batch_base_points, compute_hess=False)
                if ENABLE_BASE_DIRECTION_LOSS:
                    base_grads = base_outs["grads"]
                    if BASE_DIRECTION_PARALLEL_ONLY:
                        base_up_dot = torch.sum(base_grads * up_direction, dim=1, keepdim=True)
                        up_projection = base_up_dot * up_direction
                        base_tangent = base_grads - up_projection
                        downward_error = torch.relu(-base_up_dot)
                        base_direction_loss = torch.mean(base_tangent.square() + downward_error.square())
                    else:
                        base_error = base_grads - up_direction
                        base_direction_loss = torch.mean(base_error * base_error)
                if ENABLE_BASE_SCALAR_LOSS:
                    base_scal_error = base_outs["scalars"] - batch_base_target_scalars
                    base_scalar_loss = torch.mean(base_scal_error * base_scal_error)

            scalar_loss_factor = surface_scalar_loss_ramp_factor(epoch)
            weighted_loss_components = {
                "surface_direction": SURFACE_DIRECTION_WEIGHT * surface_loss,
                "support_free": SUPPORT_FREE_SURFACE_WEIGHT * support_free_surface_loss,
                "group_scalar": scalar_loss_factor * SURFACE_GROUP_SCALAR_WEIGHT * surface_group_scalar_loss,
                "edge_scalar": scalar_loss_factor * SURFACE_EDGE_SCALAR_WEIGHT * surface_edge_scalar_loss,
                "gradient_unit": unit_norm_weight * gradient_unit_loss,
                "gradient_range": range_norm_weight * gradient_range_loss,
                "gradient_ratio": band_ratio_weight * band_gradient_ratio_loss,
                "gradient_floor": band_floor_weight * band_gradient_floor_loss,
                "input_curvature": INPUT_CURVATURE_WEIGHT * input_curvature_loss,
                "build_slope": BUILD_DIRECTION_SLOPE_WEIGHT * build_direction_slope_loss,
                "base_direction": BASE_DIRECTION_WEIGHT * base_direction_loss,
                "base_scalar": BASE_SCALAR_WEIGHT * base_scalar_loss,
            }
            loss = sum(weighted_loss_components.values())
            if STRENGTH_LOSS_PROVIDER is not None:
                strength_loss = STRENGTH_LOSS_PROVIDER(batch_input_points, inp_grads)
                weighted_loss_components['strength'] = strength_loss
                loss = loss+strength_loss
            if collision_loss is not None:
                weighted_loss_components["collision"] = collision_weight * collision_loss
                loss = loss + weighted_loss_components["collision"]

            tangent_twist_loss = torch.tensor(0.0, device=batch_input_points.device)
            applied_tangent_twist_loss = torch.tensor(0.0, device=batch_input_points.device)
            should_probe_twist = (
                ENABLE_TANGENT_TWIST_LOSS
                and ENABLE_SURFACE_DIRECTION_LOSS
                and TANGENT_TWIST_WEIGHT > 0.0
                and batch_index % TANGENT_TWIST_COSINE_CHECK_EVERY_BATCHES == 0
            )
            if should_probe_twist:
                tangent_twist_loss, eligible_twist_points = compute_tangent_twist_loss(
                    surface_grads_raw,
                    batch_surface_normals,
                    up_direction,
                )
                if eligible_twist_points > 0:
                    weighted_twist_loss = TANGENT_TWIST_WEIGHT * tangent_twist_loss
                    twist_cosine = parameter_gradient_cosine(loss, weighted_twist_loss, scalar_net)
                    twist_cosine_sum += twist_cosine
                    twist_cosine_checks += 1
                    if twist_cosine >= TANGENT_TWIST_MIN_GRADIENT_COSINE:
                        loss = loss + weighted_twist_loss
                        weighted_loss_components["tangent_twist"] = weighted_twist_loss
                        applied_tangent_twist_loss = tangent_twist_loss
                        twist_accepted_groups += 1

            if (
                PRINT_INITIAL_LOSS_GRADIENT_CONTRIBUTIONS
                and epoch < INITIAL_LOSS_GRADIENT_DIAGNOSTIC_EPOCHS
                and batch_index == 0
            ):
                print_loss_gradient_contributions(weighted_loss_components, scalar_net, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach()
            if DEBUG_PRINT_LOSS_BREAKDOWN:
                component_values = {
                    "surface_direction": SURFACE_DIRECTION_WEIGHT * surface_loss,
                    "support_free": SUPPORT_FREE_SURFACE_WEIGHT * support_free_surface_loss,
                    "group_scalar": scalar_loss_factor * SURFACE_GROUP_SCALAR_WEIGHT * surface_group_scalar_loss,
                    "edge_scalar": scalar_loss_factor * SURFACE_EDGE_SCALAR_WEIGHT * surface_edge_scalar_loss,
                    "gradient_unit": unit_norm_weight * gradient_unit_loss,
                    "gradient_range": range_norm_weight * gradient_range_loss,
                    "gradient_ratio": band_ratio_weight * band_gradient_ratio_loss,
                    "gradient_floor": band_floor_weight * band_gradient_floor_loss,
                    "input_curvature": INPUT_CURVATURE_WEIGHT * input_curvature_loss,
                    "build_slope": BUILD_DIRECTION_SLOPE_WEIGHT * build_direction_slope_loss,
                    "base_direction": BASE_DIRECTION_WEIGHT * base_direction_loss,
                    "base_scalar": BASE_SCALAR_WEIGHT * base_scalar_loss,
                    "collision_raw": torch.tensor(0.0, device=loss.device)
                    if collision_loss is None
                    else collision_loss,
                    "collision": torch.tensor(0.0, device=loss.device)
                    if collision_loss is None
                    else collision_weight * collision_loss,
                    "tangent_twist": TANGENT_TWIST_WEIGHT * applied_tangent_twist_loss,
                }
                for name, value in component_values.items():
                    loss_sums[name] += float(value.detach().to("cpu").item())
            batch_count += 1

        # Do not let StepLR count the warm-up/ramp epochs: its first decay is
        # measured from stable target-LR training instead.
        if epoch >= LEARNING_RATE_WARMUP_EPOCHS + LEARNING_RATE_RAMP_EPOCHS:
            scheduler.step()
        if SHOW_NORM_CONTROL_BATCHES_EACH_EPOCH:
            visualize_norm_control_batches(norm_batch_snapshots, epoch)
        if surface_curriculum is not None:
            surface_curriculum.update(scalar_net, epoch)
        if batch_count > 0:
            epoch_loss = float((epoch_loss / batch_count).detach().cpu().item())
        if (
            DEBUG_PRINT_LOSS_BREAKDOWN
            and batch_count > 0
            and (epoch + 1) % LOSS_BREAKDOWN_EVERY_EPOCHS == 0
        ):
            breakdown = " ".join(
                f"{name}={value / batch_count:.4e}"
                for name, value in loss_sums.items()
            )
            twist_status = ""
            if ENABLE_TANGENT_TWIST_LOSS:
                mean_cosine = twist_cosine_sum / max(twist_cosine_checks, 1)
                twist_status = (
                    f" tangent_twist_groups={twist_accepted_groups}/{twist_cosine_checks}"
                    f" tangent_twist_cosine={mean_cosine:.3f}"
                )
            print(f"epoch={epoch + 1}/{epochs} total={epoch_loss:.4e} {breakdown}{twist_status}")
        else:
            print(epoch_loss)

        if batch_count:
            if STRENGTH_LOSS_PROVIDER is not None:
                STRENGTH_LOSS_PROVIDER.end_epoch(scalar_net)
            print(
                f"sampled_band_height_ratio_max={float(epoch_max_height_ratio.cpu()):.4f} "
                f"target={np.sqrt(GRADIENT_BAND_MAX_SQUARED_RATIO):.4f}"
            )
            if ENABLE_BALANCED_NORM_BATCHES and gradient_reference is not None:
                print(f"norm_reference_coverage={int(gradient_reference.seen.sum().cpu())}/{len(norm_control_points)}")

        if epoch < LEARNING_RATE_WARMUP_EPOCHS:
            print(
                f"Warm-up epoch {epoch + 1}/{LEARNING_RATE_WARMUP_EPOCHS}: "
                f"learning rate={optimizer.param_groups[0]['lr']:.3e}"
            )

        if checkpoint_path and save_every and ((epoch + 1) % save_every == 0):
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": scalar_net.state_dict(),
            }
            if checkpoint_metadata:
                checkpoint.update(checkpoint_metadata)
            torch.save(checkpoint, checkpoint_path)

    if checkpoint_path and epochs > 0:
        checkpoint = {
            "epoch": epochs,
            "model_state_dict": scalar_net.state_dict(),
        }
        if checkpoint_metadata:
            checkpoint.update(checkpoint_metadata)
        torch.save(checkpoint, checkpoint_path)


def main() -> None:
    base = load_base_module()

    surface_mesh = base.load_surface_mesh(SURFACE_MESH_PATH)
    volume_mesh = base.load_volume_mesh(VOLUME_NODE_PATH, VOLUME_ELE_PATH)
    mesh_vertices, mesh_faces = base.extract_mesh_arrays(surface_mesh)

    _, masked_grid, _ = base.build_interior_band_grid(
        mesh_vertices,
        mesh_faces,
        padding=BOUNDARY_PADDING,
        grid_spacing=GRID_SPACING,
        near_surface_band=NEAR_SURFACE_BAND,
        outer_band=OUTER_BAND,
        min_samples_per_axis=MIN_INPUT_GRID_SAMPLES_PER_AXIS,
    )

    automatic_angle = SURFACE_NORMAL_MAX_ANGLE_DEGREES
    automatic_min_height = SURFACE_HEIGHT_MIN_WORLD
    automatic_max_height = SURFACE_HEIGHT_MAX_WORLD
    if AUTOMATIC_SELECTION_CONFIGURATOR is not None:
        automatic_angle, automatic_min_height, automatic_max_height = AUTOMATIC_SELECTION_CONFIGURATOR(
            surface_mesh,
            UP_DIRECTION,
            automatic_angle,
            automatic_min_height,
            automatic_max_height,
        )
    surface_constraint_points_np, surface_constraint_normals_np, surface_constraint_group_ids_np = compute_surface_quality_constraints(
        surface_mesh,
        up_direction=UP_DIRECTION,
        max_angle_degrees=automatic_angle,
        min_height_world=automatic_min_height,
        max_height_world=automatic_max_height,
        min_region_area_ratio=SURFACE_REGION_AREA_RATIO_MIN,
    )
    automatic_surface_group_ids = set(np.unique(surface_constraint_group_ids_np).tolist())
    support_free_surface_points_np, support_free_surface_normals_np = compute_support_free_surface_constraints(
        surface_mesh,
        up_direction=UP_DIRECTION,
        base_height_world=BASE_HEIGHT_WORLD,
        min_height_world=SUPPORT_FREE_SURFACE_HEIGHT_MIN_WORLD,
        max_height_world=SUPPORT_FREE_SURFACE_HEIGHT_MAX_WORLD,
    )
    if ENABLE_INTERACTIVE_MANUAL_SURFACE_REGIONS:
        if LOAD_MANUAL_SURFACE_REGIONS:
            manual_regions = load_manual_surface_regions(
                MANUAL_SURFACE_REGIONS_PATH,
                surface_mesh.n_points,
                Path(SURFACE_MESH_PATH).stem,
            )
            selection_mesh = surface_mesh.copy(deep=True).triangulate()
            selection_mesh.compute_normals(point_normals=True, cell_normals=False, inplace=True)
            mesh_point_normals = np.asarray(selection_mesh.point_data["Normals"], dtype=np.float32)
            mesh_point_normals /= np.linalg.norm(mesh_point_normals, axis=1, keepdims=True) + 1e-8
        else:
            manual_regions, mesh_point_normals = select_manual_surface_regions(surface_mesh)
            if SAVE_MANUAL_SURFACE_REGIONS:
                save_manual_surface_regions(
                    MANUAL_SURFACE_REGIONS_PATH,
                    manual_regions,
                    surface_mesh.n_points,
                    Path(SURFACE_MESH_PATH).stem,
                )
        (
            surface_constraint_points_np,
            surface_constraint_normals_np,
            surface_constraint_group_ids_np,
        ) = append_manual_surface_regions(
            points=surface_constraint_points_np,
            normals=surface_constraint_normals_np,
            group_ids=surface_constraint_group_ids_np,
            mesh_points=np.asarray(surface_mesh.points, dtype=np.float32),
            mesh_normals=mesh_point_normals,
            manual_regions=manual_regions,
        )
        print(f"Added {len(manual_regions)} interactive surface-quality regions.")
    if ORIENT_PRESERVED_GROUP_NORMALS_TOWARD_UP:
        surface_constraint_normals_np = orient_preserved_group_normals_toward_up(
            surface_constraint_normals_np,
            surface_constraint_group_ids_np,
            UP_DIRECTION,
        )
    use_surface_edge_scalar_loss = (
        ENABLE_SURFACE_EDGE_SCALAR_LOSS or SURFACE_EDGE_REPLACES_GROUP_SCALAR_LOSS
    )
    # The preview is a diagnostic of the sampled-pair construction and should
    # remain available even when the interactive dialog disables edge loss.
    build_surface_edge_pairs = use_surface_edge_scalar_loss or SHOW_SURFACE_EDGE_PAIRS_BEFORE_TRAINING
    if build_surface_edge_pairs and not SURFACE_EDGE_USE_SAMPLED_NEIGHBORS:
        (
            surface_edge_points_a_np,
            surface_edge_points_b_np,
            surface_edge_normals_a_np,
            surface_edge_normals_b_np,
        ) = build_constrained_surface_edges(
            surface_mesh,
            surface_constraint_points_np,
            surface_constraint_normals_np,
            surface_constraint_group_ids_np,
        )
    else:
        surface_edge_points_a_np = np.empty((0, 3), dtype=np.float32)
        surface_edge_points_b_np = np.empty((0, 3), dtype=np.float32)
        surface_edge_normals_a_np = np.empty((0, 3), dtype=np.float32)
        surface_edge_normals_b_np = np.empty((0, 3), dtype=np.float32)
    # Keep original mesh vertices for legacy metadata consumers. Resampling
    # creates interior points, which cannot be mapped back to vertex indices.
    metadata_constraint_points_np = surface_constraint_points_np.copy()
    metadata_constraint_group_ids_np = surface_constraint_group_ids_np.copy()
    if SURFACE_CONSTRAINT_RESAMPLER is not None:
        (
            surface_constraint_points_np,
            surface_constraint_normals_np,
            surface_constraint_group_ids_np,
        ) = SURFACE_CONSTRAINT_RESAMPLER(
            surface_mesh,
            surface_constraint_points_np,
            surface_constraint_normals_np,
            surface_constraint_group_ids_np,
            automatic_surface_group_ids,
            automatic_angle,
            automatic_min_height,
            automatic_max_height,
        )
        # Partial height/angle clipping can split an automatic vertex group
        # into several disconnected surface patches. The resampler publishes
        # the final automatic IDs so training and cap metadata stay aligned.
        resampled_automatic_groups = getattr(SURFACE_CONSTRAINT_RESAMPLER, "automatic_group_ids", None)
        if resampled_automatic_groups is not None:
            automatic_surface_group_ids = set(resampled_automatic_groups)
    if SAVE_SURFACE_CONSTRAINT_METADATA:
        save_surface_constraint_metadata(
            SURFACE_CONSTRAINT_METADATA_PATH,
            surface_mesh,
            metadata_constraint_points_np,
            metadata_constraint_group_ids_np,
            automatic_surface_group_ids,
            Path(SURFACE_MESH_PATH).stem,
            automatic_max_angle_degrees=automatic_angle,
            automatic_min_height_world=automatic_min_height,
            automatic_max_height_world=automatic_max_height,
            automatic_min_region_area_ratio=SURFACE_REGION_AREA_RATIO_MIN,
            cap_geometry=getattr(SURFACE_CONSTRAINT_RESAMPLER, "cap_geometry", None),
        )
    if build_surface_edge_pairs and callable(SURFACE_EDGE_PAIR_RESAMPLER):
        (
            surface_edge_points_a_np,
            surface_edge_points_b_np,
            surface_edge_normals_a_np,
            surface_edge_normals_b_np,
        ) = SURFACE_EDGE_PAIR_RESAMPLER(
            surface_mesh,
            surface_constraint_points_np,
            surface_constraint_normals_np,
            surface_constraint_group_ids_np,
        )
    elif build_surface_edge_pairs and SURFACE_EDGE_USE_SAMPLED_NEIGHBORS:
        (
            surface_edge_points_a_np,
            surface_edge_points_b_np,
            surface_edge_normals_a_np,
            surface_edge_normals_b_np,
        ) = build_sampled_surface_neighbor_edges(
            surface_constraint_points_np,
            surface_constraint_normals_np,
            surface_constraint_group_ids_np,
        )
    if SHOW_SURFACE_QUALITY_POINTS_BEFORE_TRAINING:
        visualize_surface_quality_points(
            mesh=surface_mesh,
            points=surface_constraint_points_np,
            normals=surface_constraint_normals_np,
            group_ids=surface_constraint_group_ids_np,
        )
    if SHOW_SURFACE_EDGE_PAIRS_BEFORE_TRAINING:
        visualize_surface_edge_pairs(surface_mesh, surface_edge_points_a_np, surface_edge_points_b_np)
    if SHOW_SURFACE_CONSTRAINT_BATCHES_BEFORE_TRAINING:
        visualize_surface_constraint_groups_and_batches(
            mesh=surface_mesh,
            points=surface_constraint_points_np,
            normals=surface_constraint_normals_np,
            group_ids=surface_constraint_group_ids_np,
            up_direction=UP_DIRECTION,
        )

    use_surface_group_scalar_loss = (
        ENABLE_SURFACE_GROUP_SCALAR_LOSS and not SURFACE_EDGE_REPLACES_GROUP_SCALAR_LOSS
    )
    mid_vals, range_vals = base.compute_normalization(mesh_vertices, DEVICE)
    scalar_scale = float(range_vals[0].detach().to("cpu").item())
    base_points_np, base_target_scalars_np = compute_base_constraints(
        surface_mesh,
        up_direction=UP_DIRECTION,
        base_height_world=BASE_HEIGHT_WORLD,
        scalar_scale=scalar_scale,
        max_normal_angle_degrees=BASE_NORMAL_MAX_ANGLE_DEGREES,
    )
    if BASE_CONSTRAINT_RESAMPLER is not None:
        base_points_np, base_target_scalars_np = BASE_CONSTRAINT_RESAMPLER(
            surface_mesh,
            base_points_np,
            base_target_scalars_np,
            UP_DIRECTION,
            BASE_HEIGHT_WORLD,
            BASE_NORMAL_MAX_ANGLE_DEGREES,
            scalar_scale,
        )
    print(
        f"Base constraints: {base_points_np.shape[0]} samples within "
        f"{BASE_HEIGHT_WORLD:.3f} world units of the lowest point and within "
        f"{BASE_NORMAL_MAX_ANGLE_DEGREES:.1f} degrees of -UP_DIRECTION."
    )
    if PRE_TRAIN_CONSTRAINT_PREVIEW_CALLBACK is not None:
        PRE_TRAIN_CONSTRAINT_PREVIEW_CALLBACK(
            surface_mesh,
            surface_constraint_points_np,
            surface_constraint_normals_np,
            surface_constraint_group_ids_np,
            base_points_np,
            UP_DIRECTION,
        )
    collision_gate_epsilon_model = COLLISION_GATE_EPSILON_WORLD / range_vals[0]

    input_candidate_points = masked_grid.points
    if RESAMPLE_INPUT_REGULARIZATION_POINTS_EACH_EPOCH:
        sampled_points = input_candidate_points
        input_points_per_epoch = INPUT_REGULARIZATION_POINTS_PER_EPOCH
        print(
            f"Input regularization resampling: {input_candidate_points.shape[0]} candidates, "
            f"{min(input_points_per_epoch, input_candidate_points.shape[0]) if input_points_per_epoch else input_candidate_points.shape[0]} "
            "fresh points per epoch."
        )
    else:
        random_indices = torch.randperm(input_candidate_points.shape[0])
        sampled_points = input_candidate_points[random_indices][:MAX_INPUT_POINTS]
        input_points_per_epoch = 0
    if SHOW_ALL_LOSS_EVALUATION_POINTS_BEFORE_TRAINING:
        visualize_all_loss_evaluation_points(
            mesh=surface_mesh,
            input_points=sampled_points,
            surface_points=surface_constraint_points_np,
            base_points=base_points_np,
            support_free_points=support_free_surface_points_np,
            edge_points_a=surface_edge_points_a_np,
            edge_points_b=surface_edge_points_b_np,
        )
    if SHOW_INPUT_REGULARIZATION_BATCHES_BEFORE_TRAINING:
        visualize_input_regularization_batches(
            mesh=surface_mesh,
            candidate_points=sampled_points,
            batch_size=BATCH_SIZE,
            points_per_epoch=input_points_per_epoch,
        )

    input_points = base.to_model_points(sampled_points, mid_vals, range_vals, DEVICE)
    globals()['INPUT_CURVATURE_BOUNDARY_POINTS'] = (
        base.to_model_points(mesh_vertices, mid_vals, range_vals, DEVICE)
        if globals().get('INPUT_CURVATURE_BOUNDARY_COVERAGE', False) else None
    )
    surface_points = base.to_model_points(surface_constraint_points_np, mid_vals, range_vals, DEVICE)
    surface_normals = torch.tensor(surface_constraint_normals_np, dtype=torch.float32, device=DEVICE)
    support_free_surface_points = base.to_model_points(support_free_surface_points_np, mid_vals, range_vals, DEVICE)
    support_free_surface_normals = torch.tensor(support_free_surface_normals_np, dtype=torch.float32, device=DEVICE)
    surface_group_ids = torch.tensor(surface_constraint_group_ids_np, dtype=torch.long, device=DEVICE)
    surface_edge_points_a = base.to_model_points(surface_edge_points_a_np, mid_vals, range_vals, DEVICE)
    surface_edge_points_b = base.to_model_points(surface_edge_points_b_np, mid_vals, range_vals, DEVICE)
    surface_edge_normals_a = torch.tensor(surface_edge_normals_a_np, dtype=torch.float32, device=DEVICE)
    surface_edge_normals_b = torch.tensor(surface_edge_normals_b_np, dtype=torch.float32, device=DEVICE)
    base_points = base.to_model_points(base_points_np, mid_vals, range_vals, DEVICE)
    base_target_scalars = torch.tensor(base_target_scalars_np, dtype=torch.float32, device=DEVICE)
    dataloaders = build_hybrid_dataloaders(
        input_points=input_points,
        surface_points=surface_points,
        surface_normals=surface_normals,
        support_free_surface_points=support_free_surface_points,
        support_free_surface_normals=support_free_surface_normals,
        surface_group_ids=surface_group_ids if (use_surface_group_scalar_loss or ENABLE_SURFACE_GROUP_ANGLE_CURRICULUM) else None,
        surface_edge_points_a=surface_edge_points_a,
        surface_edge_points_b=surface_edge_points_b,
        surface_edge_normals_a=surface_edge_normals_a,
        surface_edge_normals_b=surface_edge_normals_b,
        base_points=base_points,
        base_target_scalars=base_target_scalars,
        batch_size=BATCH_SIZE,
        surface_group_batch_size=SURFACE_GROUP_BATCH_SIZE,
        surface_edge_batch_size=SURFACE_EDGE_BATCH_SIZE,
        resample_input_points_each_epoch=RESAMPLE_INPUT_REGULARIZATION_POINTS_EACH_EPOCH,
        input_points_per_epoch=input_points_per_epoch,
        input_resampling_seed=INPUT_REGULARIZATION_RESAMPLING_SEED,
    )

    tool_points = None
    local_tool_point_sets = None
    collision_tool_point_counts = None
    fixed_frame = None
    tool_jitter_specs = None
    sdf_gate_model = None
    collision_weight = 0.0
    if ENABLE_COLLISION_LOSS:
        plane_specs = [
            {
                "x_min": TOOL_X_MIN,
                "x_max": TOOL_X_MAX,
                "z_min": TOOL_Z_MIN,
                "z_max": TOOL_Z_MAX,
                "y_value": TOOL_Y_VALUE,
                "num_x": TOOL_NUM_X,
                "num_z": TOOL_NUM_Z,
            }
        ]
        if SECOND_TOOL_SURFACE_ENABLED:
            plane_specs.append(
                {
                    "x_min": SECOND_TOOL_X_MIN,
                    "x_max": SECOND_TOOL_X_MAX,
                    "z_min": SECOND_TOOL_Z_MIN,
                    "z_max": SECOND_TOOL_Z_MAX,
                    "y_value": SECOND_TOOL_Y_VALUE,
                    "num_x": SECOND_TOOL_NUM_X,
                    "num_z": SECOND_TOOL_NUM_Z,
                }
            )
        local_tool_point_sets = [
            base.sample_rectangular_plane_points(**plane_spec, device=DEVICE)
            for plane_spec in plane_specs
        ]
        if globals().get('TOOL_BOX_SPEC') is not None:
            from boxToolSampling import sample_box
            local_tool_point_sets = sample_box(TOOL_BOX_SPEC, DEVICE, seed=TOOL_PERTURB_SEED)
        if globals().get('TOOL_AXISYMMETRIC_SPEC') is not None:
            from axisymmetricTool import sample_tool
            local_tool_point_sets = sample_tool(TOOL_AXISYMMETRIC_SPEC, DEVICE, seed=TOOL_PERTURB_SEED)
        local_tool_points = torch.cat(local_tool_point_sets, dim=0)
        collision_tool_point_counts = [int(points.shape[0]) for points in local_tool_point_sets]
        fixed_frame = base.build_tool_frame(
            build_direction_world=BUILD_DIRECTION_WORLD,
            in_plane_u_world=IN_PLANE_U_WORLD,
            device=DEVICE,
        )
        if PERTURB_TOOL_POINTS_EACH_EPOCH and globals().get('TOOL_BOX_SPEC') is None and globals().get('TOOL_AXISYMMETRIC_SPEC') is None:
            tool_jitter_specs = [
                build_tool_perturbation_template(
                    tool_points_for_surface,
                    seed=TOOL_PERTURB_SEED + tool_index,
                )
                for tool_index, tool_points_for_surface in enumerate(local_tool_point_sets)
            ]
        tool_points = build_fixed_frame_tool_points_model(
            base=base,
            local_tool_points=local_tool_points,
            range_vals=range_vals,
            fixed_frame=fixed_frame,
        )
        sdf_gate_model = base.sdfModel(device=DEVICE, model_load_path=SDF_GATE_CHECKPOINT_PATH)
        collision_weight = COLLISION_WEIGHT

    scalar_net, checkpoint_metadata = load_scalar_field(base)
    if PRINT_INITIALIZER_BASE_DIRECTION_DIAGNOSTIC and ENABLE_BASE_DIRECTION_LOSS and base_points.numel() > 0:
        diagnostic_points = base_points.detach().requires_grad_(True)
        diagnostic_outputs = scalar_net(diagnostic_points, compute_hess=False)
        diagnostic_up_direction = torch.tensor(UP_DIRECTION, dtype=torch.float32, device=DEVICE)
        diagnostic_up_direction = diagnostic_up_direction / (torch.norm(diagnostic_up_direction, dim=1, keepdim=True) + 1e-8)
        diagnostic_error = diagnostic_outputs["grads"] - diagnostic_up_direction
        diagnostic_raw_loss = torch.mean(diagnostic_error.square())
        print(
            "initializer_base_direction: "
            f"points={len(diagnostic_points)} "
            f"raw={float(diagnostic_raw_loss.detach().cpu()):.6e} "
            f"weighted={float((BASE_DIRECTION_WEIGHT * diagnostic_raw_loss).detach().cpu()):.6e} "
            f"rms={float(torch.sqrt(diagnostic_raw_loss).detach().cpu()):.6e}"
        )
    if globals().get('STRESS_ORIENTATION_INITIALIZATION') is not None:
        checkpoint_metadata['stress_orientation_initialization'] = STRESS_ORIENTATION_INITIALIZATION
    checkpoint_metadata["layer_thickness_control"] = {
        "method": (
            "shared_band_pointwise_interval"
            if compute_soft_banded_gradient_ratio_loss is _DEFAULT_GRADIENT_RATIO_LOSS
            else "custom_gradient_ratio_loss"
        ),
        "max_height_ratio": float(np.sqrt(GRADIENT_BAND_MAX_SQUARED_RATIO)),
        "band_count": GRADIENT_BAND_COUNT,
        "balanced_sampling": ENABLE_BALANCED_NORM_BATCHES,
        "reference_scan_batch_size": NORM_POOL_SIZE,
        "points_per_band_minimum": NORM_POINTS_PER_BAND,
        "reference_point_count": int(input_points.shape[0]),
    }
    up_direction = torch.tensor(UP_DIRECTION, dtype=torch.float32, device=DEVICE)
    if STRENGTH_LOSS_PROVIDER is not None:
        checkpoint_metadata['strength_constraints'] = STRENGTH_LOSS_PROVIDER.metadata
    checkpoint_metadata['input_curvature'] = {
        'enabled': ENABLE_INPUT_CURVATURE_LOSS, 'mode': INPUT_CURVATURE_MODE,
        'weight': INPUT_CURVATURE_WEIGHT, 'sample_fraction': INPUT_CURVATURE_SAMPLE_FRACTION,
        'radius_mm': INPUT_CURVATURE_RADIUS_MM,
        'boundary_coverage': globals().get('INPUT_CURVATURE_BOUNDARY_COVERAGE', False),
        'boundary_point_count': (len(INPUT_CURVATURE_BOUNDARY_POINTS)
                                 if INPUT_CURVATURE_BOUNDARY_POINTS is not None else 0),
    }
    checkpoint_metadata['regularization_profile'] = globals().get('REGULARIZATION_PROFILE', 'banded')
    if checkpoint_metadata['regularization_profile'] == 'integrated-five-axis':
        checkpoint_metadata['integrated_regularization'] = dict(
            gradient_variation_weight=GRADIENT_VARIATION_WEIGHT,
            small_gradient_weight=SMALL_GRADIENT_WEIGHT,
            variation='mean(||H n||^2 / ||g||) on interior samples',
            small_gradient='mean(exp(-100 ||g||^2)) on interior samples',
            base_alignment=False, banded_norm_control=False,
            curvature_pool='interior plus covered boundary vertices')
    checkpoint_metadata['collision_normalization'] = {
        'enabled': NORMALIZE_COLLISION_BY_GRADIENT,
        'gradient_epsilon': COLLISION_GRADIENT_EPSILON,
        'denominator_detached': False,
        'coordinates': 'normalized model coordinates',
    }
    up_direction = up_direction / (torch.norm(up_direction, dim=1, keepdim=True) + 1e-8)
    build_direction = torch.tensor([BUILD_DIRECTION_WORLD], dtype=torch.float32, device=DEVICE)
    build_direction = build_direction / (torch.norm(build_direction, dim=1, keepdim=True) + 1e-8)

    x_min, y_min, z_min = torch.min(input_points, dim=0)[0]
    x_max, y_max, z_max = torch.max(input_points, dim=0)[0]

    train_scalar_field(
        scalar_net=scalar_net,
        input_loader_train=dataloaders["input_loader_train"],
        surface_loader_train=dataloaders["surface_loader_train"],
        support_free_surface_loader_train=dataloaders["support_free_surface_loader_train"],
        base_loader_train=dataloaders["base_loader_train"],
        surface_group_loader_train=dataloaders["surface_group_loader_train"],
        surface_edge_loader_train=dataloaders["surface_edge_loader_train"],
        norm_control_points=input_points,
        norm_steps_per_epoch=dataloaders["num_train_batches"],
        scalar_scale=range_vals[0],
        up_direction=up_direction,
        build_direction=build_direction,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        step_size=STEP_SIZE,
        step_gamma=STEP_GAMMA,
        tool_points_model=tool_points,
        local_tool_point_sets=local_tool_point_sets,
        collision_tool_point_counts=collision_tool_point_counts,
        range_vals=range_vals,
        fixed_frame=fixed_frame,
        tool_jitter_specs=tool_jitter_specs,
        base_module=base,
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
        checkpoint_metadata=checkpoint_metadata,
        surface_curriculum=dataloaders["surface_curriculum"],
    )

    base.save_slicer_visualization_config(
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

    if not SHOW_FINAL_CONTOUR_VIS:
        return

    contour_source = volume_mesh if volume_mesh is not None else surface_mesh
    model_scalars = base.evaluate_scalar_field_on_dataset_points(
        contour_source,
        scalar_net=scalar_net,
        mid_vals=mid_vals,
        range_vals=range_vals,
        device=DEVICE,
    )
    contour_dataset = base.build_contour_dataset(contour_source, model_scalars)
    contour_values = build_contour_values(model_scalars, range_vals)
    contours = base.extract_contours(contour_dataset, contour_values)

    plate_mesh = None
    last_surface = None
    root_point_world = None
    tool_points_world = None
    if SHOW_LAYER_TOOL_VIS:
        if SHOW_PLATFORM_IN_LAYER_VIS:
            plate_mesh = base.build_plate_rectangle_world(
                mesh_points_world=surface_mesh.points,
                build_direction_world=BUILD_DIRECTION_WORLD,
                in_plane_u_world=IN_PLANE_U_WORLD,
                padding_world=PLATE_PADDING_WORLD,
            )
        last_surface = base.extract_last_iso_surface(contour_dataset, contour_values)
        if last_surface is not None and SHOW_TOOL_IN_LAYER_VIS:
            root_point_world = base.select_lowest_point_on_surface(last_surface, BUILD_DIRECTION_WORLD)
            if root_point_world is not None:
                tool_points_world = base.build_tool_points_world(
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

    base.visualize_contours(
        mesh=surface_mesh,
        contours=contours,
        volume_mesh=volume_mesh,
        plate_mesh=plate_mesh,
        last_surface=last_surface,
        root_point_world=root_point_world,
        tool_points_world=tool_points_world,
    )


if __name__ == "__main__":
    main()
