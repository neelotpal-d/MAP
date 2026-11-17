"""Geometry-based SDF fitting samples, independent of the layer-field grid."""
import igl
import numpy as np


def volume_grid(vertices, spacing):
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Volume spacing must be positive and finite.')
    low, high = vertices.min(0), vertices.max(0)
    padding = .1 * float(np.ptp(vertices, axis=0).max())
    axes = [np.arange(a-padding, b+padding+spacing*.01, spacing) for a,b in zip(low,high)]
    if np.prod([len(a) for a in axes], dtype=np.int64) > 2_000_000:
        raise ValueError('SDF grid exceeds two million points; increase volume spacing.')
    return np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1,3)


def surface_samples(vertices, faces, spacing, rng, perturb_mm):
    if not np.isfinite(spacing) or spacing <= 0 or not np.isfinite(perturb_mm) or perturb_mm <= 0:
        raise ValueError('Surface spacing and perturbation scale must be positive and finite.')
    # libigl uses Euclidean Poisson-disk separation, including across nearby
    # faces. Its C++ random generator is not exposed; persist the actual points.
    _, _, surface = igl.blue_noise(vertices, faces, spacing)
    if not len(surface):
        raise ValueError('No surface samples; reduce surface spacing.')
    perturbed = surface + rng.normal(scale=perturb_mm, size=surface.shape)
    return surface, perturbed
