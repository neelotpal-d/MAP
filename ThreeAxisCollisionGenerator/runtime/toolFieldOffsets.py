"""Helpers for evaluating translated tool samples in a scalar field.

The main operation in this module is:
1. take a batch of root points with shape (n, 3)
2. take a set of tool-relative points with shape (k, 3)
3. translate the tool points to every root point to get shape (n, k, 3)
4. evaluate the scalar field both at the translated points and at the roots
5. return the translated values, root values, and their difference

This code is written to preserve gradients. Nothing is detached.
"""

import torch


def _extract_scalar_values(field_output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    """Normalize scalar-field outputs to a tensor of shape (..., 1).

    The helper accepts either:
    - a raw tensor returned by the caller's scalar field function, or
    - a dictionary with a `"scalars"` entry, which matches the local SIREN API.
    """
    if isinstance(field_output, dict):
        return field_output["scalars"]
    return field_output


def _evaluate_field(field, points: torch.Tensor) -> torch.Tensor:
    """Evaluate a field-like object on points and return scalar values.

    Supported inputs:
    - a callable returning a tensor or a dict with `"scalars"`
    - an object exposing a `predictOuts(points)` method
    """
    if hasattr(field, "predictOuts"):
        return _extract_scalar_values(field.predictOuts(points))
    try:
        return _extract_scalar_values(field(points, compute_grads=False))
    except TypeError:
        return _extract_scalar_values(field(points))


def _build_bounding_box_mask(
    points: torch.Tensor,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
) -> torch.Tensor:
    """Return a float mask for points inside an axis-aligned bounding box."""
    inside_mask = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)

    if x_min is not None:
        inside_mask = inside_mask * (points[:, 0:1] >= x_min).to(dtype=points.dtype)
    if x_max is not None:
        inside_mask = inside_mask * (points[:, 0:1] <= x_max).to(dtype=points.dtype)
    if y_min is not None:
        inside_mask = inside_mask * (points[:, 1:2] >= y_min).to(dtype=points.dtype)
    if y_max is not None:
        inside_mask = inside_mask * (points[:, 1:2] <= y_max).to(dtype=points.dtype)
    if z_min is not None:
        inside_mask = inside_mask * (points[:, 2:3] >= z_min).to(dtype=points.dtype)
    if z_max is not None:
        inside_mask = inside_mask * (points[:, 2:3] <= z_max).to(dtype=points.dtype)

    return inside_mask


def translate_tool_points(root_points: torch.Tensor, tool_points: torch.Tensor) -> torch.Tensor:
    """Translate tool-relative points to every root point.

    Parameters
    ----------
    root_points:
        Tensor with shape (n, 3).
    tool_points:
        Tensor with shape (k, 3).

    Returns
    -------
    torch.Tensor
        Tensor with shape (n, k, 3), where each tool point cloud is translated
        so its origin sits on the corresponding root point.
    """
    return root_points[:, None, :] + tool_points[None, :, :]


def evaluate_translated_scalar_differences(
    scalar_field,
    root_points: torch.Tensor,
    tool_points: torch.Tensor,
    root_scalars: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate scalar differences between roots and translated tool points.

    Parameters
    ----------
    scalar_field:
        Callable that accepts a tensor of shape (..., 3) flattened to (m, 3)
        and returns either:
        - a tensor of shape (m, 1), or
        - a dict containing `"scalars"` with shape (m, 1).
    root_points:
        Tensor of shape (n, 3).
    tool_points:
        Tensor of shape (k, 3).
    root_scalars:
        Optional tensor of shape (n, 1) or (n, 1, 1) containing precomputed
        scalar-field values at the root points. Supplying this avoids an extra
        root-point field evaluation when those values are already available.

    Returns
    -------
    dict[str, torch.Tensor]
        A dictionary with:
        - `"translated_points"`: shape (n, k, 3)
        - `"translated_scalars"`: shape (n, k, 1)
        - `"root_scalars"`: shape (n, 1, 1)
        - `"scalar_differences"`: shape (n, k, 1)

    Notes
    -----
    The difference is computed as:
        scalar_field(root + tool_offset) - scalar_field(root)
    No tensors are detached, so gradients can flow back through both the root
    points and the scalar field parameters.
    """
    translated_points = translate_tool_points(root_points, tool_points)

    num_roots, num_tool_points, _ = translated_points.shape
    translated_points_flat = translated_points.reshape(num_roots * num_tool_points, 3)

    translated_scalars_flat = _evaluate_field(scalar_field, translated_points_flat)
    translated_scalars = translated_scalars_flat.reshape(num_roots, num_tool_points, 1)

    if root_scalars is None:
        root_scalars = _evaluate_field(scalar_field, root_points)

    root_scalars = root_scalars.reshape(num_roots, 1, 1)
    scalar_differences = translated_scalars - root_scalars

    return {
        "translated_points": translated_points,
        "translated_scalars": translated_scalars,
        "root_scalars": root_scalars,
        "scalar_differences": scalar_differences,
    }


def compute_translated_scalar_differences(
    scalar_field,
    root_points: torch.Tensor,
    tool_points: torch.Tensor,
    root_scalars: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return only the translated-minus-root scalar differences.

    This is a convenience helper for loss construction when the translated
    point cloud and intermediate scalar tensors are not needed explicitly.
    """
    return evaluate_translated_scalar_differences(
        scalar_field=scalar_field,
        root_points=root_points,
        tool_points=tool_points,
        root_scalars=root_scalars,
    )["scalar_differences"]


def compute_gated_translated_scalar_loss(
    scalar_field,
    root_points: torch.Tensor,
    tool_points: torch.Tensor,
    gating_field=None,
    root_scalars: torch.Tensor | None = None,
    gate_epsilon: float = 0.0,
    normalize_by_mask: bool = True,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the gated hinge-style loss on translated scalar differences.

    The loss is:
        mean((ReLU(-difference))^2)
    evaluated only at translated tool points whose gating SDF is below the
    requested threshold.

    Parameters
    ----------
    scalar_field:
        Scalar field used to compute the translated-minus-root difference.
    root_points:
        Tensor of shape (n, 3).
    tool_points:
        Tensor of shape (k, 3).
    gating_field:
        Optional callable used to decide which translated tool points are
        included in the loss. If omitted, every translated point is included.
    root_scalars:
        Optional precomputed scalar-field values at the roots.
    gate_epsilon:
        Threshold used for the gating field. With the local sign convention,
        negative means inside, so `gate_epsilon=0` keeps strictly inside points
        and positive values include a small outside band.
    normalize_by_mask:
        If True, normalize by the number of included points. If False, use a
        simple mean over the full `(n, k, 1)` tensor after masking.
    x_min, x_max, y_min, y_max, z_min, z_max:
        Optional axis-aligned bounding-box limits for the SDF domain. Points
        outside this box are treated as outside immediately and excluded from
        the loss regardless of the gating field prediction.
    """
    outputs = evaluate_translated_scalar_differences(
        scalar_field=scalar_field,
        root_points=root_points,
        tool_points=tool_points,
        root_scalars=root_scalars,
    )

    scalar_differences = outputs["scalar_differences"]
    translated_points = outputs["translated_points"]
    num_roots, num_tool_points, _ = translated_points.shape

    translated_points_flat = translated_points.reshape(num_roots * num_tool_points, 3)
    bounding_box_mask = _build_bounding_box_mask(
        translated_points_flat,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
    ).reshape(num_roots, num_tool_points, 1)

    if gating_field is None:
        inside_mask = bounding_box_mask
    else:
        gating_scalars = _evaluate_field(gating_field, translated_points_flat).reshape(num_roots, num_tool_points, 1)
        sdf_inside_mask = (gating_scalars < gate_epsilon).to(dtype=scalar_differences.dtype)
        inside_mask = bounding_box_mask * sdf_inside_mask

    violation = torch.relu(-scalar_differences) ** 2
    masked_violation = violation * inside_mask

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


def analyze_gated_translated_collisions(
    scalar_field,
    root_points: torch.Tensor,
    tool_points: torch.Tensor,
    gating_field=None,
    root_scalars: torch.Tensor | None = None,
    gate_epsilon: float = 0.0,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
) -> dict[str, torch.Tensor]:
    """Return diagnostic collision information for translated tool placements."""
    loss_outputs = compute_gated_translated_scalar_loss(
        scalar_field=scalar_field,
        root_points=root_points,
        tool_points=tool_points,
        gating_field=gating_field,
        root_scalars=root_scalars,
        gate_epsilon=gate_epsilon,
        normalize_by_mask=True,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
    )

    translated_outputs = evaluate_translated_scalar_differences(
        scalar_field=scalar_field,
        root_points=root_points,
        tool_points=tool_points,
        root_scalars=root_scalars,
    )

    translated_points = translated_outputs["translated_points"]
    scalar_differences = loss_outputs["scalar_differences"]
    inside_mask = loss_outputs["inside_mask"]

    point_collision_mask = (scalar_differences < 0) & (inside_mask > 0)
    root_collision_mask = point_collision_mask.any(dim=1)
    collision_counts = point_collision_mask.to(dtype=scalar_differences.dtype).sum(dim=1)

    return {
        "translated_points": translated_outputs["translated_points"],
        "translated_scalars": translated_outputs["translated_scalars"],
        "root_scalars": translated_outputs["root_scalars"],
        "scalar_differences": scalar_differences,
        "inside_mask": inside_mask,
        "bounding_box_mask": loss_outputs["bounding_box_mask"],
        "point_collision_mask": point_collision_mask,
        "root_collision_mask": root_collision_mask,
        "collision_counts": collision_counts,
        "colliding_root_points": root_points[root_collision_mask[:, 0]],
        "colliding_translated_points": translated_points[point_collision_mask.expand_as(translated_points)].reshape(-1, 3),
        "loss": loss_outputs["loss"],
    }


class ToolFieldOffsetEvaluator:
    """Stateful helper for translated tool-point field evaluations.

    This is convenient when the same scalar field and optional gating SDF are
    reused across many calls during training.
    """

    def __init__(self, scalar_field, gating_field=None):
        self.scalar_field = scalar_field
        self.gating_field = gating_field

    def evaluate(
        self,
        root_points: torch.Tensor,
        tool_points: torch.Tensor,
        root_scalars: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return translated points, scalar values, and scalar differences."""
        return evaluate_translated_scalar_differences(
            scalar_field=self.scalar_field,
            root_points=root_points,
            tool_points=tool_points,
            root_scalars=root_scalars,
        )

    def differences(
        self,
        root_points: torch.Tensor,
        tool_points: torch.Tensor,
        root_scalars: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return only the translated-minus-root scalar differences."""
        return compute_translated_scalar_differences(
            scalar_field=self.scalar_field,
            root_points=root_points,
            tool_points=tool_points,
            root_scalars=root_scalars,
        )

    def gated_loss(
        self,
        root_points: torch.Tensor,
        tool_points: torch.Tensor,
        root_scalars: torch.Tensor | None = None,
        gating_field=None,
        gate_epsilon: float = 0.0,
        normalize_by_mask: bool = True,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        z_min: float | None = None,
        z_max: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return the gated hinge-style loss and its intermediate tensors."""
        active_gating_field = self.gating_field if gating_field is None else gating_field
        return compute_gated_translated_scalar_loss(
            scalar_field=self.scalar_field,
            root_points=root_points,
            tool_points=tool_points,
            gating_field=active_gating_field,
            root_scalars=root_scalars,
            gate_epsilon=gate_epsilon,
            normalize_by_mask=normalize_by_mask,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
        )
