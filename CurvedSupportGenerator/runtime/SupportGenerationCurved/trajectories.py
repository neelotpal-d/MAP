"""Envelope guide trajectories from trusted surface gradients, without a tree."""
from __future__ import annotations

import numpy as np
import pyvista as pv


def trace_to_plate(points, gradients, direction, plate, step, turn_degrees=2.25):
    """Start at -grad(phi), gradually rotate toward downward build direction.

    This is geometric guidance, not exterior neural evaluation or a printable
    branch graph. Turning also handles horizontal and initially upward rays.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    gradients = np.asarray(gradients, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    if gradients.shape != points.shape or not np.all(np.isfinite(gradients)):
        raise ValueError("Trajectory gradients must be finite and match points.")
    if step <= 0 or not 0 < turn_degrees <= 90:
        raise ValueError("Trajectory step must be positive and turn angle in (0,90].")
    norms = np.linalg.norm(gradients, axis=1)
    eligible = (norms > 1e-10) & (points @ direction > plate + 1e-9)
    selected = np.flatnonzero(eligible)
    paths = []
    target = -direction
    delta_max = np.deg2rad(turn_degrees)
    for index in selected:
        position = points[index].copy()
        moving = -gradients[index] / norms[index]
        path = [position.copy()]
        max_steps = int(np.ceil((position @ direction - plate) / step)) + int(np.ceil(np.pi / delta_max)) * 2 + 10
        for _ in range(max_steps):
            next_point = position + step * moving
            height = float(position @ direction - plate)
            next_height = float(next_point @ direction - plate)
            if next_height <= 0:
                endpoint = position + (height / (height - next_height)) * (next_point - position)
                endpoint -= (endpoint @ direction - plate) * direction
                path.append(endpoint)
                break
            path.append(next_point)
            position = next_point
            cosine = float(np.clip(moving @ target, -1, 1))
            angle = np.arccos(cosine)
            if angle <= delta_max:
                moving = target.copy()
            else:
                tangent = target - cosine * moving
                if np.linalg.norm(tangent) < 1e-10:
                    axis = np.eye(3)[np.argmin(np.abs(moving))]
                    tangent = axis - (axis @ moving) * moving
                tangent /= np.linalg.norm(tangent)
                moving = np.cos(delta_max) * moving + np.sin(delta_max) * tangent
                moving /= np.linalg.norm(moving)
        else:
            raise RuntimeError(f"Trajectory {index} did not reach the plate within its bounded step budget.")
        paths.append(np.asarray(path))
    if not paths:
        raise ValueError("No valid above-plate trajectory starts were selected.")
    vertices = np.vstack(paths)
    lines = []
    offset = 0
    for path in paths:
        lines.extend((len(path), *range(offset, offset + len(path))))
        offset += len(path)
    poly = pv.PolyData(vertices)
    poly.lines = np.asarray(lines, dtype=np.int64)
    endpoint_errors = [abs(path[-1] @ direction - plate) for path in paths]
    return poly, {"trajectory_count": len(paths), "trajectory_points": len(vertices),
                  "step": step, "turn_degrees": turn_degrees,
                  "plate_endpoint_error_max": float(max(endpoint_errors)),
                  "invalid_gradient_count": int(np.count_nonzero(norms <= 1e-10))}


def overhang_mask(normals, gradients, threshold_degrees=45):
    gradients = np.asarray(gradients, dtype=float)
    norms = np.linalg.norm(gradients, axis=1)
    return (norms > 1e-10) & (np.einsum("ni,ni->n", normals, gradients) <= -np.cos(np.deg2rad(threshold_degrees)) * norms)
