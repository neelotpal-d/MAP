from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .model_field import SurfaceSamples


@dataclass(frozen=True)
class ContactSet:
    points: np.ndarray
    normals: np.ndarray
    phases: np.ndarray
    terminal_scalars: np.ndarray
    areas: np.ndarray
    source_face_count: int
    selected_face_count: int
    roof_centers: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    roof_normals: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    roof_areas: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    roof_phases: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))


def load_layer_values(path: str | Path) -> np.ndarray:
    values = np.asarray(np.loadtxt(path), dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("At least two layer scalar values are required.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Layer scalar values contain NaN or infinity.")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("Layer scalar values must be strictly increasing.")
    return values


def derive_layer_values(
    scalar_samples: np.ndarray,
    gradient_samples_world: np.ndarray,
    world_spacing: float,
    epsilon: float,
    maximum_count: int,
    probe_count: int,
) -> np.ndarray:
    """Derive a nominal model schedule without evaluating the SIREN outside the model.

    This mirrors the repository's adaptive scalar stepping, but estimates the
    local gradient norm from trusted surface samples near the current scalar.
    Supplying the exporter's main_scalar_values.txt remains preferable.
    This median estimate is not a maximum physical-spacing bound. The curved
    pipeline separately refines support-only levels using actual support-field
    samples in SupportGenerationCurved.support_schedule.
    """

    scalar_samples = np.asarray(scalar_samples, dtype=np.float64).reshape(-1)
    gradient_norms = np.linalg.norm(np.asarray(gradient_samples_world, dtype=np.float64), axis=1)
    valid = np.isfinite(scalar_samples) & np.isfinite(gradient_norms) & (gradient_norms > 1e-9)
    if np.count_nonzero(valid) < 2:
        raise ValueError("Too few valid trusted samples to derive layer scalar values.")
    scalar_samples = scalar_samples[valid]
    gradient_norms = gradient_norms[valid]
    lower = float(np.min(scalar_samples) + epsilon)
    upper = float(np.max(scalar_samples) - epsilon)
    if lower >= upper:
        raise ValueError("Layer epsilon removes the complete trusted scalar range.")

    order = np.argsort(scalar_samples)
    sorted_scalars = scalar_samples[order]
    sorted_norms = gradient_norms[order]
    count = min(max(8, int(probe_count)), len(sorted_scalars))
    values = [lower]
    scalar_span = upper - lower
    minimum_step = max(np.finfo(np.float64).eps * max(abs(lower), abs(upper), 1.0), scalar_span * 1e-7)
    while values[-1] < upper and len(values) < int(maximum_count):
        current = values[-1]
        insertion = int(np.searchsorted(sorted_scalars, current))
        start = max(0, min(len(sorted_scalars) - count, insertion - count // 2))
        local_norm = float(np.median(sorted_norms[start : start + count]))
        step = max(local_norm * float(world_spacing), minimum_step)
        next_value = current + step
        if next_value >= upper:
            break
        values.append(next_value)
    if len(values) >= int(maximum_count) and values[-1] < upper:
        raise RuntimeError("Automatic layer schedule reached layers.maximum_count.")
    if len(values) < 2:
        values.append(upper)
    return np.asarray(values, dtype=np.float64)


def scalar_to_phase(scalars: np.ndarray, layer_values: np.ndarray) -> np.ndarray:
    layer_indices = np.arange(len(layer_values), dtype=np.float64)
    return np.interp(np.asarray(scalars, dtype=np.float64), layer_values, layer_indices)


def _coverage_preserving_subset(
    records: list[tuple[float, np.ndarray, np.ndarray, float, float]],
    maximum_count: int,
) -> list[tuple[float, np.ndarray, np.ndarray, float, float]]:
    """Keep a deterministic, spatially distributed subset of roof terminals.

    Selecting only the largest-area bins makes the hard terminal cap punch
    holes in large overhangs.  Start from the most significant bin, then use
    farthest-point sampling in world space so the cap limits routing work
    without silently turning a dense roof demand back into sparse contacts.
    """

    if maximum_count <= 0 or len(records) <= maximum_count:
        return records
    points = np.stack([record[1] for record in records])
    areas = np.asarray([record[0] for record in records], dtype=np.float64)
    phases = np.asarray([record[3] for record in records], dtype=np.float64)
    # Stable tie-break: area, then earlier layer phase, then original bucket
    # order (which is itself sorted deterministically).
    first = int(np.lexsort((np.arange(len(records)), phases, -areas))[0])
    chosen = [first]
    nearest_squared = np.sum((points - points[first]) ** 2, axis=1)
    nearest_squared[first] = -np.inf
    for _ in range(1, maximum_count):
        candidate = int(np.argmax(nearest_squared))
        if not np.isfinite(nearest_squared[candidate]):
            break
        chosen.append(candidate)
        squared = np.sum((points - points[candidate]) ** 2, axis=1)
        nearest_squared = np.minimum(nearest_squared, squared)
        nearest_squared[np.asarray(chosen, dtype=np.int64)] = -np.inf
    return [records[index] for index in chosen]


def phase_to_scalar(phases: np.ndarray, layer_values: np.ndarray) -> np.ndarray:
    """Map the continuous support layer coordinate back to model scalar units."""

    layer_indices = np.arange(len(layer_values), dtype=np.float64)
    return np.interp(np.asarray(phases, dtype=np.float64), layer_indices, layer_values)


def detect_contacts(
    samples: SurfaceSamples,
    layer_values: np.ndarray,
    build_direction: np.ndarray,
    build_plate_projection: float,
    cfg: dict,
) -> ContactSet:
    gradients = np.asarray(samples.gradients_world, dtype=np.float64)
    grad_norm = np.linalg.norm(gradients, axis=1)
    valid_gradient = grad_norm > 1e-10
    gradient_direction = np.zeros_like(gradients)
    gradient_direction[valid_gradient] = gradients[valid_gradient] / grad_norm[valid_gradient, None]
    cosine = np.einsum("ij,ij->i", gradient_direction, samples.normals)
    # The user-facing threshold is measured from the downward field direction:
    # 45 degrees therefore selects normals within 45 degrees of -grad(phi).
    # Keep the legacy normal-vs-gradient angle as a resolved compatibility
    # value for old configuration files.
    support_threshold = cfg.get("support_threshold_angle_degrees")
    legacy_angle = float(cfg["overhang_angle_degrees"])
    normal_gradient_angle = 180.0 - float(support_threshold) if support_threshold is not None else legacy_angle
    threshold_cosine = float(np.cos(np.deg2rad(normal_gradient_angle)))
    projection = samples.centers @ build_direction
    phase = scalar_to_phase(samples.scalars, layer_values)
    selected = (
        valid_gradient
        & (cosine <= threshold_cosine)
        & (projection >= build_plate_projection + float(cfg["base_exclusion"]))
        & (phase >= float(cfg["minimum_layer"]))
        & (samples.scalars >= layer_values[0])
        & (samples.scalars <= layer_values[-1])
    )
    selected_ids = np.flatnonzero(selected)
    if selected_ids.size == 0:
        return ContactSet(
            points=np.empty((0, 3)),
            normals=np.empty((0, 3)),
            phases=np.empty(0),
            terminal_scalars=np.empty(0),
            areas=np.empty(0),
            source_face_count=len(samples.centers),
            selected_face_count=0,
        )

    # ``tip_spacing`` is a physical coverage distance, not the side length
    # of an arbitrary XYZ hash cell.  A cell of that side length has a
    # 2.5*sqrt(3) diagonal, which is why the old 2.5-mm setting visibly
    # produced much wider contact gaps.  Bound every source point and its
    # bucket representative by the requested distance in the worst-oriented
    # case by using the cube diagonal as the bin diameter.
    spacing = float(cfg["tip_spacing"])
    bin_width = spacing / np.sqrt(3.0)
    points = samples.centers[selected_ids]
    normals = samples.normals[selected_ids]
    weights = samples.areas[selected_ids]
    origin = points.min(axis=0)
    bins = np.floor((points - origin) / bin_width).astype(np.int64)

    # A terminal is a physical roof/contact patch, not one seed per sampled
    # face *or* per scalar-layer index. Including phase in this key made the
    # same spatial patch reappear as multiple near-coincident requests on an
    # analytic Y field. The phase/scalar below remains the terminal condition.
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for local_index, xyz in enumerate(bins):
        key = (int(xyz[0]), int(xyz[1]), int(xyz[2]))
        buckets.setdefault(key, []).append(local_index)

    records: list[tuple[float, np.ndarray, np.ndarray, float, float]] = []
    for key in sorted(buckets):
        ids = np.asarray(buckets[key], dtype=np.int64)
        bucket_weights = np.maximum(weights[ids], 1e-12)
        total_area = float(np.sum(weights[ids]))
        if total_area < float(cfg["minimum_bucket_area"]):
            continue
        point = np.average(points[ids], axis=0, weights=bucket_weights)
        normal = np.average(normals[ids], axis=0, weights=bucket_weights)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        # The scalar is sampled only on the trusted model surface.  The outer
        # tip is offset geometrically, but carries this source scalar as its
        # sole compatibility constraint; the SIREN is never queried outside
        # the model to invent a value there.
        terminal_scalar = float(np.average(samples.scalars[selected_ids][ids], weights=bucket_weights))
        terminal_phase = float(scalar_to_phase(np.asarray([terminal_scalar]), layer_values)[0])
        point = point + float(cfg["surface_offset"]) * normal
        records.append((total_area, point, normal, terminal_phase, terminal_scalar))

    maximum_seeds = int(cfg["maximum_seeds"])
    if maximum_seeds > 0 and len(records) > maximum_seeds:
        records = _coverage_preserving_subset(records, maximum_seeds)
    records.sort(key=lambda item: (item[3], *item[1].tolist()))
    if not records:
        return ContactSet(
            points=np.empty((0, 3)), normals=np.empty((0, 3)), phases=np.empty(0), terminal_scalars=np.empty(0),
            areas=np.empty(0),
            source_face_count=len(samples.centers), selected_face_count=len(selected_ids),
        )
    return ContactSet(
        points=np.stack([item[1] for item in records]),
        normals=np.stack([item[2] for item in records]),
        phases=np.asarray([item[3] for item in records]),
        terminal_scalars=np.asarray([item[4] for item in records]),
        areas=np.asarray([item[0] for item in records]),
        source_face_count=len(samples.centers),
        selected_face_count=len(selected_ids),
        roof_centers=np.asarray(samples.centers[selected_ids], dtype=np.float64),
        roof_normals=np.asarray(samples.normals[selected_ids], dtype=np.float64),
        roof_areas=np.asarray(samples.areas[selected_ids], dtype=np.float64),
        roof_phases=np.asarray(phase[selected_ids], dtype=np.float64),
    )
