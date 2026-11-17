"""Direct cavity-node sampling with the original Taylor method as a backup."""
import numpy as np
from .transfer import blended_taylor


def sample_boundary(mesh, points, evaluator, samples, mode="auto"):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    _, projected = mesh.find_closest_cell(points, return_closest_point=True)
    projected = np.asarray(projected, dtype=float)
    distances = np.linalg.norm(points - projected, axis=1)
    if mode == "auto":
        mode = "direct" if evaluator is not None else "sample_taylor"
    if mode == "direct":
        if evaluator is None:
            raise ValueError("Direct boundary sampling requires a callable field; use sample_taylor for stored samples.")
        values, gradients = evaluator.evaluate_scalars_and_gradients(points)
    elif mode == "projected_taylor":
        if evaluator is None:
            raise ValueError("Projected Taylor backup requires a callable field; use sample_taylor for stored samples.")
        values, gradients = evaluator.evaluate_scalars_and_gradients(projected)
        values = values + np.einsum("ni,ni->n", gradients, points - projected)
    elif mode == "sample_taylor":
        values, gradients = blended_taylor(points, *samples)
    else:
        raise ValueError(f"Unknown boundary mode: {mode}")
    values = np.asarray(values, dtype=float).reshape(-1)
    gradients = np.asarray(gradients, dtype=float)
    if values.shape != (len(points),) or gradients.shape != points.shape or not np.all(np.isfinite(values)) or not np.all(np.isfinite(gradients)):
        raise ValueError("Boundary evaluator returned invalid scalars or gradients.")
    return values, gradients, distances, {"mode": mode, "query_count": len(points),
        "query_distance_to_model_max": float(distances.max()),
        "query_distance_to_model_rms": float(np.sqrt(np.mean(distances ** 2))),
        "uses_scalar_extrapolation": mode != "direct"}
