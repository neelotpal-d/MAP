"""Support/model extension for the XZ-projected hybrid path planner.

Support layers use only offset perimeter loops.  Their resampling pitch is
derived from the projected footprint area so small cylindrical supports retain
enough points to remain well resolved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from batch_generate_projected_hybrid_paths import (
    ProjectedSurfaceLifter,
    _planar_region_from_projected_triangles,
    process_projected_hybrid_layer,
)
from extrusion_height import apply_axis_lift
from mesh_distance_workflow import load_obj_mesh


def support_resample_spacing(
    projected_area: float,
    *,
    base_spacing: float,
    min_spacing: float,
    min_points_per_contour: int,
) -> tuple[float, dict[str, float]]:
    """Choose a support perimeter pitch from its equivalent circular size."""
    if not np.isfinite([projected_area,base_spacing,min_spacing]).all():
        raise ValueError('Support area and resampling spacings must be finite.')
    if base_spacing <= 0.0 or min_spacing <= 0.0:
        raise ValueError("Support resampling spacings must be positive.")
    if min_spacing > base_spacing:
        raise ValueError("support_model.min_resample_spacing cannot exceed resample_spacing.")
    if min_points_per_contour < 3:
        raise ValueError("support_model.min_points_per_contour must be at least 3.")

    area = max(0.0, float(projected_area))
    equivalent_circumference = 2.0 * float(np.sqrt(np.pi * area))
    target_pitch = (
        equivalent_circumference / float(min_points_per_contour)
        if equivalent_circumference > 0.0
        else base_spacing
    )
    effective_spacing = max(min_spacing, min(base_spacing, target_pitch))
    return effective_spacing, {
        "support_projected_area": area,
        "support_equivalent_circumference": equivalent_circumference,
        "support_target_resample_spacing": target_pitch,
        "support_resample_spacing": effective_spacing,
        "support_min_points_per_contour": float(min_points_per_contour),
    }


def process_projected_support_layer(
    mesh_path: Path,
    output_dir: Path,
    *,
    spacing: float,
    resample_spacing: float | None,
    contour_count: int,
    min_resample_spacing: float,
    min_points_per_contour: int,
    model_lift: float = 0.0,
    model_lift_axis: str = "y",
) -> tuple[Path, dict[str, object]]:
    """Write two (by default) projected offset loops for one support layer."""
    if contour_count < 1:
        raise ValueError("support_model.contour_count must be at least 1.")
    vertices, faces = load_obj_mesh(str(mesh_path))
    if abs(float(model_lift)) > 0.0:
        vertices = apply_axis_lift(vertices, axis=model_lift_axis, amount=float(model_lift))
    region = _planar_region_from_projected_triangles(ProjectedSurfaceLifter.build(vertices, faces))
    base_spacing = float(resample_spacing if resample_spacing is not None else spacing)
    effective_spacing, diagnostics = support_resample_spacing(
        float(region.area),
        base_spacing=base_spacing,
        min_spacing=float(min_resample_spacing),
        min_points_per_contour=int(min_points_per_contour),
    )

    output_path, metadata = process_projected_hybrid_layer(
        mesh_path,
        output_dir,
        spacing=float(spacing),
        parallel_spacing=float(spacing),
        resample_spacing=effective_spacing,
        parallel_resample_spacing=effective_spacing,
        boundary_count=int(contour_count),
        sharp_angle_deg=0.0,
        boundary_field_backend="planar_polygon_offset",
        parallel_field_backend="planar_raster_clip",
        axis_mode="fixed",
        axis="x",
        plane_value=None,
        level_set_constraint="None",
        soft_level_set_weight=-1.0,
        contour_max_level_fraction=None,
        min_polyline_points=2,
        mild_smooth_all=False,
        mild_smooth_iterations=0,
        mild_smooth_alpha=0.0,
        disable_parallel=True,
        model_lift=float(model_lift),
        model_lift_axis=model_lift_axis,
    )

    tag_path = output_dir / f"{mesh_path.stem}.tags.txt"
    if tag_path.exists():
        tag_path.write_text(
            tag_path.read_text(encoding="utf-8").replace("boundary_loop", "support_boundary_loop"),
            encoding="utf-8",
        )
    metadata.update(
        {
            "layer_role": "support",
            "support_contour_count": int(contour_count),
            "parallel_disabled": True,
            "path_tag_values": ["support_boundary_loop"],
            **diagnostics,
        }
    )
    return output_path, metadata
