"""Validation helpers for the first shared analytic non-planar field experiment."""
import numpy as np


def field_values(evaluator,points):
    points=np.asarray(points,dtype=float).reshape(-1,3)
    values,gradients=evaluator.evaluate_scalars_and_gradients(points)
    values=np.asarray(values,dtype=float).reshape(-1)
    gradients=np.asarray(gradients,dtype=float)
    if len(values)!=len(points) or gradients.shape!=points.shape or not np.isfinite(values).all() or not np.isfinite(gradients).all():
        raise ValueError('Nonplanar field must return finite scalars and gradients at every requested point.')
    if np.any(np.linalg.norm(gradients,axis=1)<1e-10):
        raise ValueError('Nonplanar experiment requires nonzero field gradients.')
    return values,gradients


def slice_error(surface,evaluator,value):
    """Measure analytic scalar residual on vertices and triangle interiors.

    Linear contour/trim facets approximate a nonlinear isosurface; these
    diagnostics expose that discretization error rather than claiming exactness.
    """
    if not surface.n_cells:
        return dict(vertex_scalar_error_max=0.,interior_scalar_error_max=0.)
    triangles=np.asarray(surface.points)[surface.regular_faces]
    scalar=lambda points: np.asarray(evaluator.evaluate_scalars(points),dtype=float) if hasattr(evaluator,'evaluate_scalars') else field_values(evaluator,points)[0]
    probes=np.concatenate([triangles.mean(axis=1)[:,None,:],
        (triangles+np.roll(triangles,-1,axis=1))/2],axis=1).reshape(-1,3)
    # Shared edges contribute the same midpoint twice. Evaluate every physical
    # location once and retain the original vertex/interior error definitions.
    points=np.vstack((surface.points,probes))
    unique,inverse=np.unique(points,axis=0,return_inverse=True)
    values=scalar(unique)
    vertices=values[inverse[:surface.n_points]]
    interior=values[inverse[surface.n_points:]]
    if not np.isfinite(vertices).all() or not np.isfinite(interior).all():
        raise ValueError('Nonfinite scalar residual samples.')
    return dict(vertex_scalar_error_max=float(np.max(np.abs(vertices-value))),
        interior_scalar_error_max=float(np.max(np.abs(interior-value))))
