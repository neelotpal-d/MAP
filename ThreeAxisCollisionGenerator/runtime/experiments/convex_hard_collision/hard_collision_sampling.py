"""Runtime sampling extensions for the isolated Convex surface-quality slicer.

This module deliberately patches an imported copy of the existing Convex
script at runtime. The original slicer remains unchanged, while the experiment
can reuse its geometry preparation, losses, checkpoint format, and visualizer.
"""

from __future__ import annotations

import math
import inspect
import time
import json
from pathlib import Path
from dataclasses import dataclass
from itertools import cycle, islice
from typing import Callable

import numpy as np
import torch
from torch.optim import lr_scheduler
from globalOrientation import scheduled_value


@dataclass(frozen=True)
class HardCollisionConfig:
    """Sampling and refresh controls for the isolated experiment."""

    input_points_per_epoch: int = 30_000
    collision_roots_per_step: int = 128
    hard_fraction: float = 0.50
    positive_fraction: float = 0.25
    refresh_every_epochs: int = 5
    mining_candidate_count: int = 50_000
    hard_pool_size: int = 8_192
    validation_count: int = 4_096
    scoring_root_batch_size: int = 128
    spatial_bins_per_axis: int = 16
    full_initial_mining: bool = True
    sdf_gate_batch_size: int = 50_000
    field_query_batch_size: int = 16_384
    fuse_regular_losses: bool = False
    seed: int = 12345

    def __post_init__(self) -> None:
        positive_counts = (
            self.input_points_per_epoch,
            self.collision_roots_per_step,
            self.refresh_every_epochs,
            self.mining_candidate_count,
            self.hard_pool_size,
            self.scoring_root_batch_size,
            self.spatial_bins_per_axis,
            self.sdf_gate_batch_size,
            self.field_query_batch_size,
        )
        if any(value <= 0 for value in positive_counts):
            raise ValueError("Sampling counts, batch sizes, and refresh interval must be positive.")
        if not 0.0 <= self.hard_fraction <= 1.0 or not 0.0 <= self.positive_fraction <= 1.0:
            raise ValueError("Collision sampling fractions must lie in [0, 1].")
        if self.hard_fraction + self.positive_fraction > 1.0:
            raise ValueError("Hard and positive collision fractions cannot sum to more than one.")
        if self.validation_count < 0:
            raise ValueError("Validation count cannot be negative.")


class EpochSubsetLoader:
    """Yield a fresh, without-replacement subset of a tensor each epoch."""

    def __init__(
        self,
        points: torch.Tensor,
        batch_size: int,
        points_per_epoch: int,
        epoch_callback: Callable[[int], None],
    ) -> None:
        self.dataset = points
        self.batch_size = int(batch_size)
        self.points_per_epoch = min(int(points_per_epoch), int(points.shape[0]))
        self.epoch_callback = epoch_callback
        self.epoch = -1

    def __len__(self) -> int:
        return math.ceil(self.points_per_epoch / self.batch_size)

    def __iter__(self):
        self.epoch += 1
        self.epoch_callback(self.epoch)
        indices = torch.randperm(self.dataset.shape[0], device=self.dataset.device)
        indices = indices[: self.points_per_epoch]
        self.seen = (torch.zeros(len(self.dataset), dtype=torch.bool, device=self.dataset.device)
                     if getattr(self, 'track_coverage', False) else None)
        for start in range(0, self.points_per_epoch, self.batch_size):
            ids = indices[start : start + self.batch_size]
            if self.seen is not None:
                self.seen[ids] = True
            yield self.dataset[ids]


class HardCollisionRootManager:
    """Mine collision roots periodically and draw biased training batches."""

    def __init__(self, config: HardCollisionConfig) -> None:
        self.config = config
        self.root_pool: torch.Tensor | None = None
        self.training_indices: torch.Tensor | None = None
        self.validation_indices: torch.Tensor | None = None
        self.hard_indices: torch.Tensor | None = None
        self.positive_indices: torch.Tensor | None = None
        self.current_epoch = -1
        self.last_refresh_epoch = -1
        self.refresh_pending = False
        self.generator: torch.Generator | None = None
        self.numpy_generator = np.random.default_rng(config.seed)
        self.spatial_buckets: list[np.ndarray] = []
    def set_root_pool(self, root_pool: torch.Tensor) -> None:
        self.root_pool = root_pool.detach()
        device = root_pool.device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(self.config.seed)
        permutation = torch.randperm(root_pool.shape[0], device=device, generator=self.generator)
        validation_count = min(self.config.validation_count, max(0, root_pool.shape[0] // 10))
        self.validation_indices = permutation[:validation_count]
        # Collision monitoring must not permanently remove roots from training.
        # Independent nominal checking supplies separate geometric evidence.
        self.training_indices = permutation
        if self.training_indices.numel() == 0:
            self.training_indices = permutation
            self.validation_indices = permutation[:0]
        self._build_spatial_buckets()
        self.uniform_order = self.training_indices[torch.randperm(len(self.training_indices), device=device, generator=self.generator)]
        self.uniform_cursor = 0

    def _build_spatial_buckets(self) -> None:
        """Partition training roots into fixed spatial cells for balanced mining."""

        if self.root_pool is None or self.training_indices is None:
            return
        training_indices_cpu = self.training_indices.detach().to("cpu").numpy()
        points = self.root_pool[self.training_indices].detach().to("cpu").numpy()
        point_min = points.min(axis=0)
        point_range = np.maximum(points.max(axis=0) - point_min, 1e-12)
        normalized = np.clip((points - point_min) / point_range, 0.0, 1.0)
        bins = self.config.spatial_bins_per_axis
        coordinates = np.minimum((normalized * bins).astype(np.int64), bins - 1)
        cell_ids = coordinates[:, 0] + bins * (coordinates[:, 1] + bins * coordinates[:, 2])

        order = np.argsort(cell_ids, kind="stable")
        sorted_cells = cell_ids[order]
        _, starts = np.unique(sorted_cells, return_index=True)
        stops = np.r_[starts[1:], sorted_cells.size]
        self.spatial_buckets = [training_indices_cpu[order[start:stop]] for start, stop in zip(starts, stops)]

    def _sample_spatially_balanced_indices(self, count: int) -> torch.Tensor:
        """Sample near-equal numbers from occupied spatial cells without replacement."""

        if self.training_indices is None or not self.spatial_buckets:
            raise RuntimeError("Spatial collision buckets have not been initialized.")
        count = min(count, self.training_indices.numel())
        quota = max(1, math.ceil(count / len(self.spatial_buckets)))
        selected_parts = []
        for bucket in self.spatial_buckets:
            take = min(quota, bucket.size)
            if take == bucket.size:
                selected_parts.append(bucket)
            else:
                selected_parts.append(self.numpy_generator.choice(bucket, size=take, replace=False))
        selected = np.unique(np.concatenate(selected_parts))

        if selected.size > count:
            selected = self.numpy_generator.choice(selected, size=count, replace=False)
        elif selected.size < count:
            training_cpu = self.training_indices.detach().to("cpu").numpy()
            remaining = np.setdiff1d(training_cpu, selected, assume_unique=False)
            fill_count = min(count - selected.size, remaining.size)
            if fill_count:
                fill = self.numpy_generator.choice(remaining, size=fill_count, replace=False)
                selected = np.concatenate((selected, fill))
        return torch.as_tensor(selected, device=self.training_indices.device, dtype=torch.long)

    def start_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        interval = max(1, self.config.refresh_every_epochs)
        self.refresh_pending = self.last_refresh_epoch < 0 or epoch - self.last_refresh_epoch >= interval

    def _random_from(self, indices: torch.Tensor, count: int) -> torch.Tensor:
        if count <= 0:
            return indices[:0]
        choices = torch.randint(
            indices.numel(),
            (count,),
            device=indices.device,
            generator=self.generator,
        )
        return indices[choices]

    def _uniform_stream(self, count):
        parts = []
        while count > 0:
            if self.uniform_cursor == len(self.uniform_order):
                self.uniform_order = self.training_indices[torch.randperm(len(self.training_indices), device=self.training_indices.device, generator=self.generator)]
                self.uniform_cursor = 0
            take = min(count, len(self.uniform_order)-self.uniform_cursor)
            parts.append(self.uniform_order[self.uniform_cursor:self.uniform_cursor+take])
            self.uniform_cursor += take
            count -= take
        return torch.cat(parts) if parts else self.training_indices[:0]

    def sample_training_roots(self) -> torch.Tensor:
        if self.root_pool is None or self.training_indices is None:
            raise RuntimeError("Collision root pool has not been initialized.")

        total = self.config.collision_roots_per_step
        hard_count = round(total * self.config.hard_fraction) if self.hard_indices is not None else 0
        positive_count = round(total * self.config.positive_fraction) if self.positive_indices is not None else 0
        uniform_count = total - hard_count - positive_count

        selected = [self._uniform_stream(uniform_count)]
        if hard_count:
            selected.append(self._random_from(self.hard_indices, hard_count))
        if positive_count:
            selected.append(self._random_from(self.positive_indices, positive_count))
        indices = torch.cat(selected)
        order = torch.randperm(indices.numel(), device=indices.device, generator=self.generator)
        return self.root_pool[indices[order]]

    @staticmethod
    def _per_root_scores(module, collision_out, tool_point_counts) -> torch.Tensor:
        differences = collision_out["scalar_differences"].squeeze(-1)
        inside_mask = collision_out["inside_mask"].squeeze(-1)
        violations = torch.relu(-differences).square() * inside_mask
        tool_weights = module.COLLISION_TOOL_WEIGHTS[: len(tool_point_counts)]

        scores = torch.zeros(violations.shape[0], device=violations.device, dtype=violations.dtype)
        start = 0
        for point_count, weight in zip(tool_point_counts, tool_weights):
            stop = start + point_count
            tool_violations = violations[:, start:stop]
            k = max(1, int(np.ceil(module.COLLISION_TOPK_FRACTION * point_count)))
            scores = scores + weight * torch.topk(tool_violations, k=k, dim=1).values.mean(dim=1)
            start = stop
        if start != violations.shape[1]:
            raise ValueError("Tool point counts do not match collision samples during root scoring.")
        return scores

    def _score_indices(
        self,
        module,
        original_collision_function,
        scalar_field,
        indices: torch.Tensor,
        tool_points: torch.Tensor,
        tool_point_counts: list[int],
        collision_kwargs: dict,
    ) -> torch.Tensor:
        scores = []
        with torch.no_grad():
            for start in range(0, indices.numel(), self.config.scoring_root_batch_size):
                batch_indices = indices[start : start + self.config.scoring_root_batch_size]
                roots = self.root_pool[batch_indices]
                root_scalars = (None if getattr(module, "NORMALIZE_COLLISION_BY_GRADIENT", False)
                                else scalar_field(roots, compute_grads=False)["scalars"])
                collision_out = original_collision_function(
                    scalar_field=scalar_field,
                    root_points=roots,
                    tool_points=tool_points,
                    root_scalars=root_scalars,
                    **collision_kwargs,
                )
                scores.append(self._per_root_scores(module, collision_out, tool_point_counts))
        return torch.cat(scores) if scores else torch.empty(0, device=self.root_pool.device)

    def refresh_if_needed(
        self,
        module,
        original_collision_function,
        scalar_field,
        tool_points: torch.Tensor,
        tool_point_counts: list[int],
        collision_kwargs: dict,
    ) -> None:
        if not self.refresh_pending:
            return
        if self.training_indices is None or self.validation_indices is None:
            raise RuntimeError("Collision root pool has not been initialized.")

        is_initial_refresh = self.last_refresh_epoch < 0
        if is_initial_refresh and self.config.full_initial_mining:
            candidate_indices = self.training_indices
            refresh_kind = "full"
        else:
            candidate_indices = self._sample_spatially_balanced_indices(
                self.config.mining_candidate_count
            )
            if self.hard_indices is not None:
                # Previously hard roots are always rescored, even when their
                # spatial cells were not selected in this refresh.
                candidate_indices = torch.unique(torch.cat((candidate_indices, self.hard_indices)))
            refresh_kind = "spatial"
        candidate_count = candidate_indices.numel()
        candidate_scores = self._score_indices(
            module,
            original_collision_function,
            scalar_field,
            candidate_indices,
            tool_points,
            tool_point_counts,
            collision_kwargs,
        )

        positive_mask = candidate_scores > 0
        positive_indices = candidate_indices[positive_mask]
        positive_scores = candidate_scores[positive_mask]
        hard_count = min(self.config.hard_pool_size, positive_indices.numel())
        if hard_count:
            hard_order = torch.topk(positive_scores, k=hard_count).indices
            self.hard_indices = positive_indices[hard_order]
            hard_mask = torch.ones(positive_indices.numel(), dtype=torch.bool, device=positive_indices.device)
            hard_mask[hard_order] = False
            remaining_positive = positive_indices[hard_mask]
            self.positive_indices = remaining_positive if remaining_positive.numel() else self.hard_indices
        else:
            self.hard_indices = None
            self.positive_indices = None

        validation_scores = self._score_indices(
            module,
            original_collision_function,
            scalar_field,
            self.validation_indices,
            tool_points,
            tool_point_counts,
            collision_kwargs,
        )
        validation_positive = validation_scores > 0
        collision_rate = validation_positive.float().mean().item() if validation_scores.numel() else 0.0
        positive_mean = (
            validation_scores[validation_positive].mean().item() if validation_positive.any() else 0.0
        )
        worst = validation_scores.max().item() if validation_scores.numel() else 0.0
        print(
            f"collision_refresh epoch={self.current_epoch + 1} "
            f"kind={refresh_kind} candidates={candidate_count} "
            f"positives={positive_indices.numel()} hard={hard_count} "
            f"validation_rate={collision_rate:.4f} validation_positive_mean={positive_mean:.4e} "
            f"validation_worst={worst:.4e}"
        )
        self.last_refresh_epoch = self.current_epoch
        self.refresh_pending = False


def _extract_scalars(field_output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    """Accept the scalar-only output conventions used by local field models."""

    return field_output["scalars"] if isinstance(field_output, dict) else field_output


def _bbox_mask(
    points: torch.Tensor,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    z_min=None,
    z_max=None,
) -> torch.Tensor:
    """Return the boolean model-domain mask without allocating float masks."""

    mask = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
    if x_min is not None:
        mask &= points[:, 0] >= x_min
    if x_max is not None:
        mask &= points[:, 0] <= x_max
    if y_min is not None:
        mask &= points[:, 1] >= y_min
    if y_max is not None:
        mask &= points[:, 1] <= y_max
    if z_min is not None:
        mask &= points[:, 2] >= z_min
    if z_max is not None:
        mask &= points[:, 2] <= z_max
    return mask


def compute_bbox_culled_collision_loss(
    scalar_field,
    root_points: torch.Tensor,
    tool_points: torch.Tensor,
    gating_field=None,
    root_scalars: torch.Tensor | None = None,
    gate_epsilon: float = 0.0,
    normalize_by_mask: bool = True,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    z_min=None,
    z_max=None,
    sdf_gate_batch_size: int = 50_000,
    field_query_batch_size: int = 16_384,
) -> dict[str, torch.Tensor]:
    """Evaluate scalar values only for translated points that pass both gates.

    The SDF is a fixed boolean gate, so its evaluation is safely inference-only.
    This preserves scalar-field gradients at included root/tool pairs while
    avoiding scalar-field work for points outside the model volume.
    """

    root_count, tool_count = root_points.shape[0], tool_points.shape[0]
    translated_points = (root_points[:, None, :] + tool_points[None, :, :]).reshape(-1, 3)
    bbox_mask_flat = _bbox_mask(
        translated_points,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
    )
    bbox_indices = torch.nonzero(bbox_mask_flat, as_tuple=False).reshape(-1)
    inside_mask_flat = torch.zeros(
        translated_points.shape[0], 1, device=translated_points.device, dtype=root_points.dtype
    )

    if bbox_indices.numel() and gating_field is not None:
        # The SDF only determines a hard mask, so its graph is never useful.
        with torch.inference_mode():
            candidate_points = translated_points[bbox_indices]
            if hasattr(gating_field, "predictOuts"):
                gate_values = _extract_scalars(
                    gating_field.predictOuts(candidate_points, batch_size=sdf_gate_batch_size)
                )
            else:
                try:
                    gate_values = _extract_scalars(gating_field(candidate_points, compute_grads=False))
                except TypeError:
                    gate_values = _extract_scalars(gating_field(candidate_points))
            gate_inside = gate_values.reshape(-1) < gate_epsilon
        inside_indices = bbox_indices[gate_inside]
    else:
        inside_indices = bbox_indices

    if inside_indices.numel():
        if root_scalars is None:
            root_scalars = _extract_scalars(scalar_field(root_points, compute_grads=False))
        root_values = root_scalars.reshape(root_count, -1)[:, 0]
        active_root_indices = torch.div(inside_indices, tool_count, rounding_mode="floor")
        active_translated_scalars = torch.cat([
            _extract_scalars(scalar_field(translated_points[batch], compute_grads=False)).reshape(-1)
            for batch in inside_indices.split(field_query_batch_size)
        ])
        active_differences = active_translated_scalars - root_values[active_root_indices]
        scalar_differences_flat = torch.zeros_like(inside_mask_flat).index_copy(
            0, inside_indices, active_differences.unsqueeze(1)
        )
        inside_mask_flat = inside_mask_flat.index_fill(0, inside_indices, 1.0)
    else:
        scalar_differences_flat = torch.zeros_like(inside_mask_flat)

    scalar_differences = scalar_differences_flat.reshape(root_count, tool_count, 1)
    inside_mask = inside_mask_flat.reshape(root_count, tool_count, 1)
    bounding_box_mask = bbox_mask_flat.to(dtype=root_points.dtype).reshape(root_count, tool_count, 1)
    masked_violation = torch.relu(-scalar_differences).square() * inside_mask
    if normalize_by_mask:
        loss = masked_violation.sum() / (inside_mask.sum() + 1e-8)
    else:
        loss = masked_violation.mean()
    return {
        "loss": loss,
        "scalar_differences": scalar_differences,
        "inside_mask": inside_mask,
        "bounding_box_mask": bounding_box_mask,
        "masked_violation_sum": masked_violation.sum(),
        "num_included_points": inside_mask.sum(),
    }


def _group_scalar_loss_from_outputs(
    module,
    group_scalars: torch.Tensor,
    group_grads: torch.Tensor,
    group_normals: torch.Tensor,
    group_ids: torch.Tensor,
    group_count: int,
    scalar_scale: torch.Tensor,
) -> torch.Tensor:
    """Apply the configured group-scalar equation to fused outputs."""

    if group_count == 0:
        return torch.zeros((), device=group_scalars.device, dtype=group_scalars.dtype)
    scalars = group_scalars.reshape(-1)
    grad_norms = torch.norm(group_grads, dim=1)
    grad_directions = group_grads / (grad_norms.unsqueeze(1) + 1e-5)
    min_allowed_dot = float(np.cos(np.deg2rad(module.SURFACE_GROUP_ALIGNMENT_MAX_ANGLE_DEGREES)))
    alignment_dot = torch.sum(grad_directions * group_normals, dim=1)
    weights = torch.sigmoid(
        (alignment_dot - min_allowed_dot) / module.SURFACE_GROUP_ALIGNMENT_WEIGHT_TEMPERATURE
    ).detach()

    group_sum = torch.zeros(group_count, device=scalars.device, dtype=scalars.dtype)
    group_weight_sum = torch.zeros_like(group_sum)
    group_sum.index_add_(0, group_ids, weights * scalars)
    group_weight_sum.index_add_(0, group_ids, weights)
    group_mean = group_sum / torch.clamp(group_weight_sum, min=1e-8)

    grad_norm_sum = torch.zeros_like(group_sum)
    grad_norm_sum.index_add_(0, group_ids, weights * grad_norms)
    mean_grad_norm = grad_norm_sum / torch.clamp(group_weight_sum, min=1e-8)
    world_scale = scalar_scale.to(device=scalars.device, dtype=scalars.dtype)
    world_difference = world_scale * (scalars - group_mean[group_ids]) / torch.clamp(
        mean_grad_norm[group_ids], min=module.SURFACE_GROUP_GRADIENT_DENOMINATOR_EPSILON
    )
    if module.SURFACE_GROUP_SCALAR_LOSS_TYPE == "l1":
        point_error = torch.abs(world_difference)
    elif module.SURFACE_GROUP_SCALAR_LOSS_TYPE == "l2":
        point_error = world_difference.square()
    else:
        raise ValueError(f"Unsupported group scalar loss: {module.SURFACE_GROUP_SCALAR_LOSS_TYPE!r}")
    group_error_sum = torch.zeros_like(group_sum)
    group_error_sum.index_add_(0, group_ids, weights * point_error)
    return torch.mean(group_error_sum / torch.clamp(group_weight_sum, min=1e-8))


def _fused_train_scalar_field(module, *args, **kwargs) -> None:
    """Train with one first-derivative forward for compatible regular losses."""

    signature = inspect.signature(module._hard_collision_original_train)
    bound = signature.bind_partial(*args, **kwargs)
    values = bound.arguments
    scalar_net = values["scalar_net"]
    orientation = getattr(module, 'GLOBAL_ORIENTATION', None)
    if orientation is not None:
        orientation.bind(scalar_net, values['fixed_frame'], values['range_vals'])
    optimizer = (orientation.optimizer(scalar_net, values['learning_rate']) if orientation is not None
                 else torch.optim.Adam(scalar_net.parameters(), lr=values['learning_rate']))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=values["step_size"], gamma=values["step_gamma"])
    if module.LEARNING_RATE_WARMUP_EPOCHS > 0:
        for parameter_group in optimizer.param_groups[:1]:
            parameter_group["lr"] = values["learning_rate"] * module.LEARNING_RATE_WARMUP_FACTOR

    input_loader = values["input_loader_train"]
    surface_loader = values["surface_loader_train"]
    support_loader = values["support_free_surface_loader_train"]
    base_loader = values["base_loader_train"]
    group_loader = values["surface_group_loader_train"]
    edge_loader = values["surface_edge_loader_train"]
    scalar_scale = values["scalar_scale"]
    up_direction = values["up_direction"]
    build_direction = values["build_direction"]
    collision_counts = values.get("collision_tool_point_counts")
    provider = getattr(module, 'STRENGTH_LOSS_PROVIDER', None)
    boundary_points = getattr(module, 'INPUT_CURVATURE_BOUNDARY_POINTS', None)
    boundary_stream = None
    if boundary_points is not None:
        from stressSampling import PartitionedEpochBatches
        boundary_stream = PartitionedEpochBatches(len(boundary_points), 65537)
    if provider is not None and provider.sampling == 'fps':
        reference = values.get('norm_control_points')
        if reference is None:
            reference = input_loader.dataset
        provider.prepare_fps(len(reference), input_loader.batch_size, next(scalar_net.parameters()).device)
    start_time = time.perf_counter()
    trainable_parameters = [parameter for parameter in scalar_net.parameters() if parameter.requires_grad]

    try:
        for epoch in range(values["epochs"]):
            if orientation is not None:
                orientation.begin_epoch(epoch, scalar_net, optimizer)
            frame_only = orientation is not None and orientation.phase == 'frame'
            epoch_collision_weight = scheduled_value(getattr(module, 'COLLISION_WEIGHT_SCHEDULE', [(1, values['collision_weight'])]), epoch+1)
            if frame_only and epoch_collision_weight <= 0:
                raise ValueError('Frame-only phases require a positive collision weight')
            trainable_parameters = [parameter for parameter in scalar_net.parameters() if parameter.requires_grad]
            strength_provider = None if frame_only else getattr(module, 'STRENGTH_LOSS_PROVIDER', None)
            if strength_provider is not None:
                strength_provider.begin_epoch(epoch)
            fps_active = strength_provider is not None and strength_provider.sampling == 'fps'
            epoch_steps = max(len(input_loader), len(strength_provider.stream)) if fps_active else len(input_loader)
            if boundary_stream is not None and not frame_only:
                boundary_stream.begin_epoch(epoch, epoch_steps)
            curvature_interior_evaluations = 0
            norm_active = module.GRADIENT_BAND_RATIO_WEIGHT > 0 or module.GRADIENT_BAND_FLOOR_WEIGHT > 0
            norm_control_points = values.get("norm_control_points") if norm_active and not frame_only else None
            gradient_reference = (
                module.BandGradientReference(
                    scalar_net, norm_control_points, module.GRADIENT_BAND_COUNT,
                    float(np.sqrt(module.GRADIENT_BAND_MAX_SQUARED_RATIO)), module.NORM_POOL_SIZE,
                    module.NORM_POINTS_PER_BAND, epoch_steps if fps_active else (values.get("norm_steps_per_epoch") or len(input_loader)),
                )
                if norm_control_points is not None else None
            )
            if gradient_reference is not None:
                print(f"norm_reference_points={len(norm_control_points)} "
                      f"reference_ratio_max={float(gradient_reference.probe_max_ratio.cpu()):.4f}")
            epoch_max_height_ratio = scalar_scale.new_tensor(1.0)
            if epoch == module.LEARNING_RATE_WARMUP_EPOCHS:
                for parameter_group in optimizer.param_groups[:1]:
                    parameter_group["lr"] = values["learning_rate"]
                print(f"Warm-up complete: learning rate restored to {values['learning_rate']:.3e}")
            elif epoch < module.LEARNING_RATE_WARMUP_EPOCHS:
                print(
                    f"Warm-up epoch {epoch + 1}/{module.LEARNING_RATE_WARMUP_EPOCHS}: "
                    f"learning rate={optimizer.param_groups[0]['lr']:.3e}"
                )

            epoch_tool_points = values.get("tool_points_model")
            local_sets = values.get("local_tool_point_sets")
            if (
                local_sets is not None
                and values.get("range_vals") is not None
                and values.get("fixed_frame") is not None
                and values.get("base_module") is not None
            ):
                epoch_local_sets = []
                for tool_index, base_tool_points in enumerate(local_sets):
                    jitter_specs = values.get("tool_jitter_specs")
                    jitter_template, x_step, z_step = (
                        jitter_specs[tool_index] if jitter_specs is not None else (None, 0.0, 0.0)
                    )
                    epoch_local_sets.append(
                        module.perturb_tool_points_for_epoch(base_tool_points, jitter_template, x_step, z_step)
                    )
                if getattr(module, 'TOOL_BOX_SPEC', None) is not None:
                    from boxToolSampling import sample_box
                    epoch_local_sets = sample_box(module.TOOL_BOX_SPEC, scalar_scale.device, seed=module.TOOL_PERTURB_SEED+epoch)
                if getattr(module, 'TOOL_AXISYMMETRIC_SPEC', None) is not None:
                    from axisymmetricTool import sample_tool
                    epoch_local_sets = sample_tool(module.TOOL_AXISYMMETRIC_SPEC, scalar_scale.device, seed=module.TOOL_PERTURB_SEED+epoch)
                epoch_tool_points = module.build_fixed_frame_tool_points_model(
                    values["base_module"],
                    torch.cat(epoch_local_sets, dim=0),
                    values["range_vals"],
                    values["fixed_frame"],
                )

            group_source = group_loader if group_loader is not None else cycle([[]])
            edge_source = edge_loader if edge_loader is not None else cycle([(None, None, None, None)])
            epoch_loss = scalar_scale.new_zeros(())
            batch_count = 0
            base_gradient_diagnostic = None
            norm_batch_snapshots = []
            loss_sums = {
                "base_direction": None,
                "base_scalar": None,
                "surface_direction": None,
                "support_free": None,
                "group_scalar": None,
                "edge_scalar": None,
                "gradient_ratio": None,
                "gradient_floor": None,
                "gradient_variation": None,
                "small_gradient": None,
                "build_slope": None,
                "input_curvature": None,
                "collision": None,
            }
            streams = (input_loader, surface_loader, support_loader, base_loader, group_source, edge_source)
            batches_epoch = (islice(zip(*(cycle(stream) for stream in streams)), epoch_steps)
                             if fps_active else zip(*streams))
            for batches in batches_epoch:
                (
                    batch_input,
                    (surface_points, surface_normals),
                    (support_points, support_normals),
                    (base_points, base_targets),
                    batch_groups,
                    batch_edges,
                ) = batches

                if frame_only:
                    collision_out = module.compute_gated_translated_scalar_loss(
                        scalar_field=scalar_net, root_points=batch_input, tool_points=epoch_tool_points,
                        gating_field=values.get('sdf_gate_model'), gate_epsilon=values['gate_epsilon'],
                        x_min=values.get('x_min'), x_max=values.get('x_max'),
                        y_min=values.get('y_min'), y_max=values.get('y_max'),
                        z_min=values.get('z_min'), z_max=values.get('z_max'))
                    collision_loss, _ = module.reduce_collision_loss(collision_out,
                        collision_counts or [len(epoch_tool_points)], module.COLLISION_LOSS_REDUCTION,
                        module.COLLISION_TOPK_FRACTION, module.COLLISION_TOOL_WEIGHTS[:len(collision_counts or [len(epoch_tool_points)])])
                    loss = epoch_collision_weight*collision_loss+orientation.parameter.sum()*0
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    orientation.after_step()
                    epoch_loss += loss.detach()
                    batch_count += 1
                    for name in loss_sums:
                        value = loss.detach() if name == 'collision' else loss.detach()*0
                        loss_sums[name] = value if loss_sums[name] is None else loss_sums[name]+value
                    continue
                if orientation is not None:
                    build_direction = orientation.direction().detach().reshape(1,3)
                group_points = None
                group_normals = None
                group_ids = None
                group_count = 0
                if (
                    module.ENABLE_SURFACE_GROUP_SCALAR_LOSS
                    and not module.SURFACE_EDGE_REPLACES_GROUP_SCALAR_LOSS
                    and epoch >= getattr(module, "SURFACE_GROUP_SCALAR_START_EPOCH", 0)
                ):
                    if batch_groups:
                        group_sizes = [points.shape[0] for points, _ in batch_groups]
                        group_points = torch.cat([points for points, _ in batch_groups], dim=0)
                        group_normals = torch.cat([normals for _, normals in batch_groups], dim=0)
                        group_count = len(batch_groups)
                        group_ids = torch.repeat_interleave(
                            torch.arange(group_count, device=group_points.device),
                            torch.tensor(group_sizes, device=group_points.device),
                        )

                sections = [("input", batch_input)]
                if fps_active:
                    sections.append(('stress', strength_provider.next_fps_batch()))
                if module.ENABLE_SURFACE_DIRECTION_LOSS:
                    sections.append(("surface", surface_points))
                if module.ENABLE_SUPPORT_FREE_SURFACE_LOSS:
                    sections.append(("support", support_points))
                if module.ENABLE_BASE_DIRECTION_LOSS or module.ENABLE_BASE_SCALAR_LOSS:
                    sections.append(("base", base_points))
                if group_points is not None:
                    sections.append(("group", group_points))
                fused_points = torch.cat([points for _, points in sections], dim=0).detach().requires_grad_(True)
                fused_outputs = scalar_net(fused_points, compute_hess=False)
                slices = {}
                offset = 0
                for name, points in sections:
                    slices[name] = slice(offset, offset + points.shape[0])
                    offset += points.shape[0]
                scalars = fused_outputs["scalars"]
                grads = fused_outputs["grads"]
                input_scalars, input_grads = scalars[slices["input"]], grads[slices["input"]]
                zero = torch.zeros((), device=fused_points.device, dtype=fused_points.dtype)

                if norm_active:
                    ratio_loss, floor_loss, observed_height_ratio = module.compute_gradient_ratio_regularization(
                        scalar_net, {"scalars": input_scalars, "grads": input_grads},
                        norm_control_points, gradient_reference,
                    )
                else:
                    ratio_loss, floor_loss = zero, zero
                    with torch.no_grad():
                        norms = input_grads.norm(dim=1)
                        observed_height_ratio = norms.max()/norms.min().clamp_min(1e-8)
                if (
                    getattr(module, "SHOW_NORM_CONTROL_BATCHES_EACH_EPOCH", False)
                    and getattr(module, "ENABLE_BALANCED_NORM_BATCHES", False)
                    and gradient_reference is not None
                    and getattr(module, "_LAST_NORM_CONTROL_SAMPLE", None) is not None
                ):
                    sampled_points, sampled_bands = module._LAST_NORM_CONTROL_SAMPLE
                    norm_batch_snapshots.append(
                        (sampled_points.cpu().numpy(), sampled_bands.cpu().numpy())
                    )
                epoch_max_height_ratio = torch.maximum(epoch_max_height_ratio, observed_height_ratio)
                build_slope_loss = zero
                if module.ENABLE_BUILD_DIRECTION_SLOPE_LOSS:
                    grad_unit = input_grads / (torch.norm(input_grads, dim=1, keepdim=True) + 1e-8)
                    error = torch.relu(
                        float(np.cos(np.deg2rad(module.BUILD_DIRECTION_SLOPE_MAX_ANGLE_DEGREES)))
                        - torch.sum(grad_unit * build_direction, dim=1)
                    )
                    build_slope_loss = torch.mean(error.square())

                surface_loss = zero
                if "surface" in slices:
                    surface_grads = grads[slices["surface"]]
                    surface_grads = surface_grads / (torch.norm(surface_grads, dim=1, keepdim=True) + 1e-5)
                    surface_loss = torch.mean((surface_grads - surface_normals).square())

                support_loss = zero
                if "support" in slices:
                    support_grads = grads[slices["support"]]
                    support_grads = support_grads / (torch.norm(support_grads, dim=1, keepdim=True) + 1e-5)
                    minimum = float(np.cos(np.deg2rad(90.0 + module.SUPPORT_FREE_SURFACE_EXTRA_ANGLE_DEGREES)))
                    support_loss = torch.mean(torch.relu(minimum - torch.sum(support_grads * support_normals, dim=1)).square())

                group_loss = zero
                if "group" in slices:
                    group_loss = _group_scalar_loss_from_outputs(
                        module,
                        scalars[slices["group"]],
                        grads[slices["group"]],
                        group_normals,
                        group_ids,
                        group_count,
                        scalar_scale,
                    )

                base_direction_loss = zero
                base_scalar_loss = zero
                if "base" in slices:
                    base_scalars, base_grads = scalars[slices["base"]], grads[slices["base"]]
                    if module.ENABLE_BASE_DIRECTION_LOSS:
                        if getattr(module, "BASE_DIRECTION_PARALLEL_ONLY", False):
                            base_up_dot = torch.sum(base_grads * up_direction, dim=1, keepdim=True)
                            up_projection = base_up_dot * up_direction
                            base_tangent = base_grads - up_projection
                            downward_error = torch.relu(-base_up_dot)
                            base_direction_loss = torch.mean(base_tangent.square() + downward_error.square())
                        else:
                            base_direction_loss = torch.mean((base_grads - up_direction).square())
                    if module.ENABLE_BASE_SCALAR_LOSS:
                        base_scalar_loss = torch.mean((base_scalars - base_targets).square())

                edge_loss = zero
                if (
                    (module.ENABLE_SURFACE_EDGE_SCALAR_LOSS or module.SURFACE_EDGE_REPLACES_GROUP_SCALAR_LOSS)
                    and batch_edges[0] is not None
                    and epoch >= getattr(module, "SURFACE_EDGE_SCALAR_START_EPOCH", 0)
                ):
                    edge_loss = module.compute_surface_edge_scalar_loss(scalar_net, *batch_edges, scalar_scale=scalar_scale)

                ramp_function = getattr(module, "surface_scalar_loss_ramp_factor", None)
                scalar_loss_factor = (
                    ramp_function(epoch)
                    if ramp_function is not None
                    else 1.0
                )

                curvature_loss = zero
                variation_loss = small_gradient_loss = zero
                if module.ENABLE_INPUT_CURVATURE_LOSS:
                    curvature_points = batch_input
                    if boundary_stream is not None:
                        boundary_ids = boundary_stream.draw()
                        if len(boundary_ids):
                            boundary_ids = torch.as_tensor(boundary_ids, device=boundary_points.device)
                            curvature_points = torch.cat([batch_input, boundary_points[boundary_ids]])
                    curvature_interior_evaluations += round(len(batch_input)*module.INPUT_CURVATURE_SAMPLE_FRACTION)
                    if getattr(module, 'REGULARIZATION_PROFILE', 'banded') == 'integrated-five-axis':
                        from integratedLosses import five_axis_regularizers
                        x_curvature = curvature_points.detach().requires_grad_(True)
                        out_curvature = scalar_net(x_curvature, compute_hess=True)
                        hessian = torch.stack([out_curvature['HX2'], out_curvature['HY2'], out_curvature['HZ2']], dim=1)
                        variation_loss, small_gradient_loss, curvature_loss = five_axis_regularizers(
                            out_curvature['grads'], hessian, scalar_scale,
                            module.INPUT_CURVATURE_RADIUS_MM, interior_count=len(batch_input))
                    else:
                        curvature_loss = module.compute_sampled_input_curvature_loss(
                            scalar_net, curvature_points, module.INPUT_CURVATURE_SAMPLE_FRACTION,
                            scalar_scale=scalar_scale,
                        )

                collision_loss = None
                if epoch_tool_points is not None and epoch_collision_weight > 0.0:
                    collision_out = module.compute_gated_translated_scalar_loss(
                        scalar_field=scalar_net,
                        root_points=batch_input,
                        tool_points=epoch_tool_points,
                        gating_field=values.get("sdf_gate_model"),
                        root_scalars=input_scalars,
                        gate_epsilon=values["gate_epsilon"],
                        x_min=values.get("x_min"), x_max=values.get("x_max"),
                        y_min=values.get("y_min"), y_max=values.get("y_max"),
                        z_min=values.get("z_min"), z_max=values.get("z_max"),
                    )
                    active_counts = collision_counts or [epoch_tool_points.shape[0]]
                    collision_loss, _ = module.reduce_collision_loss(
                        collision_out, active_counts, module.COLLISION_LOSS_REDUCTION,
                        module.COLLISION_TOPK_FRACTION, module.COLLISION_TOOL_WEIGHTS[: len(active_counts)],
                    )

                loss = (
                    module.SURFACE_DIRECTION_WEIGHT * surface_loss
                    + module.SUPPORT_FREE_SURFACE_WEIGHT * support_loss
                    + scalar_loss_factor * module.SURFACE_GROUP_SCALAR_WEIGHT * group_loss
                    + scalar_loss_factor * module.SURFACE_EDGE_SCALAR_WEIGHT * edge_loss
                    + module.GRADIENT_BAND_RATIO_WEIGHT * ratio_loss
                    + module.GRADIENT_BAND_FLOOR_WEIGHT * floor_loss
                    + module.INPUT_CURVATURE_WEIGHT * curvature_loss
                    + getattr(module, 'GRADIENT_VARIATION_WEIGHT', 0.) * variation_loss
                    + getattr(module, 'SMALL_GRADIENT_WEIGHT', 0.) * small_gradient_loss
                    + module.BUILD_DIRECTION_SLOPE_WEIGHT * build_slope_loss
                    + module.BASE_DIRECTION_WEIGHT * base_direction_loss
                    + module.BASE_SCALAR_WEIGHT * base_scalar_loss
                )
                if collision_loss is not None:
                    loss = loss + epoch_collision_weight * collision_loss
                if strength_provider is not None:
                    stress_gradients = grads[slices['stress']] if fps_active else input_grads
                    loss = loss+strength_provider(batch_input, stress_gradients)
                should_measure_base_interference = (
                    module.PRINT_BASE_GRADIENT_INTERFERENCE
                    and (epoch + 1) % module.BASE_GRADIENT_INTERFERENCE_EVERY_EPOCHS == 0
                    and batch_count == 0
                    and (module.ENABLE_BASE_DIRECTION_LOSS or module.ENABLE_BASE_SCALAR_LOSS)
                )
                if should_measure_base_interference:
                    weighted_base_loss = (
                        module.BASE_DIRECTION_WEIGHT * base_direction_loss
                        + module.BASE_SCALAR_WEIGHT * base_scalar_loss
                    )
                    other_loss = loss - weighted_base_loss
                    if weighted_base_loss.requires_grad and other_loss.requires_grad:
                        base_grads = torch.autograd.grad(
                            weighted_base_loss,
                            trainable_parameters,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        other_grads = torch.autograd.grad(
                            other_loss,
                            trainable_parameters,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        dot = torch.zeros((), device=loss.device)
                        base_norm_sq = torch.zeros((), device=loss.device)
                        other_norm_sq = torch.zeros((), device=loss.device)
                        for base_grad, other_grad in zip(base_grads, other_grads):
                            if base_grad is not None:
                                base_norm_sq = base_norm_sq + torch.sum(base_grad.square())
                            if other_grad is not None:
                                other_norm_sq = other_norm_sq + torch.sum(other_grad.square())
                            if base_grad is not None and other_grad is not None:
                                dot = dot + torch.sum(base_grad * other_grad)
                        base_norm = torch.sqrt(base_norm_sq)
                        other_norm = torch.sqrt(other_norm_sq)
                        total_norm = torch.sqrt(torch.clamp(base_norm_sq + other_norm_sq + 2.0 * dot, min=0.0))
                        base_gradient_diagnostic = (
                            float(base_norm.detach().cpu().item()),
                            float(other_norm.detach().cpu().item()),
                            float(total_norm.detach().cpu().item()),
                            float((dot / (base_norm * other_norm + 1e-12)).detach().cpu().item()),
                            float((base_norm / (total_norm + 1e-12)).detach().cpu().item()),
                        )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if orientation is not None:
                    orientation.after_step()
                epoch_loss += loss.detach()
                batch_count += 1

                # Accumulate detached GPU values and synchronize only once per
                # reporting epoch, rather than once per component per batch.
                if module.DEBUG_PRINT_LOSS_BREAKDOWN:
                    components = {
                        "base_direction": module.BASE_DIRECTION_WEIGHT * base_direction_loss,
                        "base_scalar": module.BASE_SCALAR_WEIGHT * base_scalar_loss,
                        "surface_direction": module.SURFACE_DIRECTION_WEIGHT * surface_loss,
                        "support_free": module.SUPPORT_FREE_SURFACE_WEIGHT * support_loss,
                        "group_scalar": scalar_loss_factor * module.SURFACE_GROUP_SCALAR_WEIGHT * group_loss,
                        "edge_scalar": scalar_loss_factor * module.SURFACE_EDGE_SCALAR_WEIGHT * edge_loss,
                        "gradient_ratio": module.GRADIENT_BAND_RATIO_WEIGHT * ratio_loss,
                        "gradient_floor": module.GRADIENT_BAND_FLOOR_WEIGHT * floor_loss,
                        "build_slope": module.BUILD_DIRECTION_SLOPE_WEIGHT * build_slope_loss,
                        "input_curvature": module.INPUT_CURVATURE_WEIGHT * curvature_loss,
                        "gradient_variation": getattr(module, 'GRADIENT_VARIATION_WEIGHT', 0.) * variation_loss,
                        "small_gradient": getattr(module, 'SMALL_GRADIENT_WEIGHT', 0.) * small_gradient_loss,
                        "collision": zero if collision_loss is None else epoch_collision_weight * collision_loss,
                    }
                    for name, component in components.items():
                        detached_component = component.detach()
                        loss_sums[name] = (
                            detached_component
                            if loss_sums[name] is None
                            else loss_sums[name] + detached_component
                        )

            scheduler.step()
            if module.ENABLE_INPUT_CURVATURE_LOSS and module.INPUT_CURVATURE_SAMPLE_FRACTION == 1. and not frame_only:
                interior_seen = getattr(input_loader, 'seen', None)
                coverage = dict(epoch=epoch+1, interior_evaluations=curvature_interior_evaluations,
                    interior_unique_seen=int(interior_seen.sum().item()) if interior_seen is not None else None,
                    interior_pool=len(input_loader.dataset),
                    boundary_unique_seen=int(boundary_stream.seen.sum()) if boundary_stream is not None else 0,
                    boundary_pool=len(boundary_points) if boundary_points is not None else 0)
                if boundary_stream is not None and not boundary_stream.seen.all():
                    raise RuntimeError('Incomplete boundary curvature coverage')
                print('curvature_coverage '+json.dumps(coverage), flush=True)
                if values.get('checkpoint_path'):
                    with (Path(values['checkpoint_path']).parent/'curvature_coverage.jsonl').open('a') as handle:
                        handle.write(json.dumps(coverage)+'\n')
            if orientation is not None:
                update=getattr(orientation,'intermediate_update',None)
                if update is not None:
                    update(module,scalar_net,optimizer,batch_input,epoch_tool_points,values,collision_counts,epoch_collision_weight)
                orientation.end_epoch(epoch_collision_weight)
            if getattr(module, "SHOW_NORM_CONTROL_BATCHES_EACH_EPOCH", False):
                module.visualize_norm_control_batches(norm_batch_snapshots, epoch)
            mean_epoch_loss = float((epoch_loss / max(batch_count, 1)).cpu())
            if strength_provider is not None:
                strength_provider.end_epoch(scalar_net)
            if (
                module.DEBUG_PRINT_LOSS_BREAKDOWN
                and (epoch + 1) % module.LOSS_BREAKDOWN_EVERY_EPOCHS == 0
            ):
                breakdown = " ".join(
                    f"{name}={float((value / max(batch_count, 1)).cpu().item()) if value is not None else 0.:.3e}"
                    for name, value in loss_sums.items()
                )
                print(f"epoch={epoch + 1}/{values['epochs']} total={mean_epoch_loss:.4e} {breakdown}")
            else:
                print(mean_epoch_loss)
            if batch_count:
                ratio_target = (f"target={np.sqrt(module.GRADIENT_BAND_MAX_SQUARED_RATIO):.4f}"
                                if norm_active else 'diagnostic_only; no band constraint')
                print(
                    f"{'sampled_band_height_ratio_max' if norm_active else 'sampled_batch_norm_ratio_max'}={float(epoch_max_height_ratio.cpu()):.4f} "
                    f"{ratio_target}"
                )
                if module.ENABLE_BALANCED_NORM_BATCHES and gradient_reference is not None:
                    print(f"norm_reference_coverage={int(gradient_reference.seen.sum().cpu())}/{len(norm_control_points)}")
            if base_gradient_diagnostic is not None:
                base_norm, other_norm, total_norm, cosine, base_share = base_gradient_diagnostic
                print(
                    "base_gradient_interference "
                    f"base_norm={base_norm:.3e} other_norm={other_norm:.3e} "
                    f"total_norm={total_norm:.3e} base_other_cosine={cosine:.3f} "
                    f"base_share={base_share:.3f}"
                )
            if values.get("checkpoint_path") and values.get("save_every") and (epoch + 1) % values["save_every"] == 0:
                checkpoint = {"epoch": epoch + 1, "model_state_dict": scalar_net.state_dict()}
                if values.get("checkpoint_metadata"):
                    checkpoint.update(values["checkpoint_metadata"])
                if orientation is not None:
                    checkpoint.update(orientation.checkpoint())
                torch.save(checkpoint, values["checkpoint_path"])
        if values.get("checkpoint_path") and values["epochs"] > 0:
            checkpoint = {"epoch": values["epochs"], "model_state_dict": scalar_net.state_dict()}
            if values.get("checkpoint_metadata"):
                checkpoint.update(values["checkpoint_metadata"])
            if orientation is not None:
                checkpoint.update(orientation.checkpoint())
            torch.save(checkpoint, values["checkpoint_path"])
        if orientation is not None:
            orientation.finish(scalar_net)
    finally:
        print(f"training_wall_time_seconds={time.perf_counter() - start_time:.2f}")


def install(module, config: HardCollisionConfig | None = None) -> HardCollisionRootManager:
    """Install the isolated samplers into an imported Convex module."""

    config = config or HardCollisionConfig()
    manager = HardCollisionRootManager(config)
    module._hard_collision_manager = manager
    original_build_dataloaders = module.build_hybrid_dataloaders
    original_collision_function = module.compute_gated_translated_scalar_loss

    def fast_collision_function(*args, **kwargs):
        """Use the experiment's inference-only SDF and pre-scalar culling path."""

        bound = inspect.signature(original_collision_function).bind_partial(*args, **kwargs)
        orientation = getattr(module, 'GLOBAL_ORIENTATION', None)
        if orientation is not None:
            bound.arguments['tool_points'] = orientation.rotate_offsets(bound.arguments['tool_points'])
        normalize = getattr(module, 'NORMALIZE_COLLISION_BY_GRADIENT', False)
        if normalize:
            # Reuse one root forward for values and gradients. Diagnostics may
            # run under no_grad; they still need input derivatives, not graphs.
            field_trainable = any(parameter.requires_grad for parameter in bound.arguments['scalar_field'].parameters())
            training_graph = torch.is_grad_enabled() and field_trainable
            with torch.enable_grad():
                roots = bound.arguments['root_points'].detach().requires_grad_(True)
                output = bound.arguments['scalar_field'](roots, compute_grads=False)
                values = output['scalars']
                gradients = torch.autograd.grad(values.sum(), roots, create_graph=training_graph)[0]
            bound.arguments['root_points'] = roots
            bound.arguments['root_scalars'] = values if field_trainable else values.detach()
        result = compute_bbox_culled_collision_loss(
            **bound.arguments,
            sdf_gate_batch_size=config.sdf_gate_batch_size,
            field_query_batch_size=config.field_query_batch_size,
        )
        if normalize:
            denominator = gradients.norm(dim=1).clamp_min(module.COLLISION_GRADIENT_EPSILON)
            result['scalar_differences'] = result['scalar_differences'] / denominator[:, None, None]
            violations = torch.relu(-result['scalar_differences']).square() * result['inside_mask']
            result['masked_violation_sum'] = violations.sum()
            if bound.arguments.get('normalize_by_mask', True):
                result['loss'] = violations.sum() / (result['inside_mask'].sum() + 1e-8)
            else:
                result['loss'] = violations.mean()
        return result

    def build_hybrid_dataloaders(*args, **kwargs):
        loaders = original_build_dataloaders(*args, **kwargs)
        bound = inspect.signature(original_build_dataloaders).bind_partial(*args, **kwargs)
        input_points = bound.arguments.get("input_points")
        batch_size = bound.arguments.get("batch_size", module.BATCH_SIZE)
        if input_points is None:
            raise RuntimeError("Could not obtain input points for the epoch sampler.")
        manager.set_root_pool(input_points)
        epoch_loader = EpochSubsetLoader(
            points=input_points,
            batch_size=batch_size,
            points_per_epoch=config.input_points_per_epoch,
            epoch_callback=manager.start_epoch,
        )
        epoch_loader.track_coverage = (module.ENABLE_INPUT_CURVATURE_LOSS
                                       and module.INPUT_CURVATURE_SAMPLE_FRACTION == 1.)
        loaders["input_loader_train"] = epoch_loader
        loaders["num_train_batches"] = len(epoch_loader)
        collision_status = (f"{config.collision_roots_per_step} collision roots/step"
                            if module.ENABLE_COLLISION_LOSS else "collision disabled")
        print(
            f"Training sampler: {len(epoch_loader)} training batches/epoch, "
            f"{epoch_loader.points_per_epoch}/{input_points.shape[0]} interior points/epoch, "
            f"{collision_status}."
        )
        return loaders

    def compute_gated_translated_scalar_loss(*args, **kwargs):
        scalar_field = kwargs.get("scalar_field", args[0] if args else None)
        tool_points = kwargs.get("tool_points", args[2] if len(args) > 2 else None)
        if scalar_field is None or tool_points is None:
            return fast_collision_function(*args, **kwargs)

        tool_point_counts = getattr(module, "_hard_collision_tool_point_counts", None)
        if not tool_point_counts or sum(tool_point_counts) != tool_points.shape[0]:
            tool_point_counts = [tool_points.shape[0]]

        forwarded = dict(kwargs)
        forwarded.pop("scalar_field", None)
        forwarded.pop("root_points", None)
        forwarded.pop("tool_points", None)
        forwarded.pop("root_scalars", None)
        orientation = getattr(module, 'GLOBAL_ORIENTATION', None)
        if orientation is not None and orientation.parameter.requires_grad and orientation.steps-orientation.last_mining_step >= orientation.args.orientation_refresh_steps:
            manager.refresh_pending = True
        was_pending = manager.refresh_pending
        manager.refresh_if_needed(
            module=module,
            original_collision_function=fast_collision_function,
            scalar_field=scalar_field,
            tool_points=tool_points,
            tool_point_counts=tool_point_counts,
            collision_kwargs=forwarded,
        )
        if orientation is not None and was_pending:
            orientation.last_mining_step = orientation.steps
        roots = manager.sample_training_roots()
        root_scalars = (None if getattr(module, "NORMALIZE_COLLISION_BY_GRADIENT", False)
                                else scalar_field(roots, compute_grads=False)["scalars"])
        return fast_collision_function(
            scalar_field=scalar_field,
            root_points=roots,
            tool_points=tool_points,
            root_scalars=root_scalars,
            **forwarded,
        )

    module.build_hybrid_dataloaders = build_hybrid_dataloaders
    module.compute_gated_translated_scalar_loss = compute_gated_translated_scalar_loss

    original_train = module.train_scalar_field
    module._hard_collision_original_train = original_train

    def train_scalar_field(*args, **kwargs):
        bound = inspect.signature(original_train).bind_partial(*args, **kwargs)
        counts = bound.arguments.get("collision_tool_point_counts")
        module._hard_collision_tool_point_counts = counts
        start_time = time.perf_counter()
        try:
            return original_train(*args, **kwargs)
        finally:
            elapsed_seconds = time.perf_counter() - start_time
            print(f"training_wall_time_seconds={elapsed_seconds:.2f}")

    def fused_train_scalar_field(*args, **kwargs):
        bound = inspect.signature(original_train).bind_partial(*args, **kwargs)
        module._hard_collision_tool_point_counts = bound.arguments.get("collision_tool_point_counts")
        return _fused_train_scalar_field(module, *args, **kwargs)

    module.train_scalar_field = fused_train_scalar_field if config.fuse_regular_losses else train_scalar_field
    return manager
