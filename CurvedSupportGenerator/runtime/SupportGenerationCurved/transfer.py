"""Unnormalized vector diffusion and constrained FEM scalar integration."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, cg
from scipy.spatial import cKDTree


def validate_samples(points, scalars, gradients):
    points = np.asarray(points, dtype=float)
    scalars = np.asarray(scalars, dtype=float).reshape(-1)
    gradients = np.asarray(gradients, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or len(scalars) != len(points) or gradients.shape != points.shape:
        raise ValueError("Samples require points (N,3), scalars (N,), gradients (N,3).")
    if not all(np.all(np.isfinite(a)) for a in (points, scalars, gradients)):
        raise ValueError("Samples must be finite.")
    return points, scalars, gradients


def blended_taylor(query, points, scalars, gradients, neighbors=8, batch_size=50_000):
    """Inverse-distance partition of unity; reproduces all affine fields.

    Finite k-neighbor changes may still introduce small discontinuities. This
    evaluator is a baseline/initialization, not a guaranteed globally C1 field.
    """
    points, scalars, gradients = validate_samples(points, scalars, gradients)
    query = np.asarray(query, dtype=float).reshape(-1, 3)
    tree = cKDTree(points)
    result = np.empty(len(query))
    vectors = np.empty_like(query)
    k = min(neighbors, len(points))
    for start in range(0, len(query), batch_size):
        q = query[start:start + batch_size]
        distance, ids = tree.query(q, k=k)
        distance, ids = distance.reshape(len(q), k), ids.reshape(len(q), k)
        weights = 1 / np.maximum(distance, 1e-10) ** 2
        weights /= weights.sum(axis=1, keepdims=True)
        values = scalars[ids] + np.einsum("nki,nki->nk", gradients[ids], q[:, None, :] - points[ids])
        result[start:start + len(q)] = np.sum(weights * values, axis=1)
        vectors[start:start + len(q)] = np.einsum("nk,nki->ni", weights, gradients[ids])
    return result, vectors


def assemble_fem(points, cells):
    xyz = np.asarray(points, dtype=float)[cells]
    edges = xyz[:, 1:] - xyz[:, :1]
    determinant = np.linalg.det(edges)
    volume = np.abs(determinant) / 6
    # Remove geometrically degenerate clipping slivers before inversion.
    scale = np.max(np.linalg.norm(edges, axis=2), axis=1)
    valid = volume > np.maximum(scale ** 3 * 1e-12, 1e-18)
    if not np.all(valid):
        raise ValueError("Degenerate tetrahedra must be removed before assembly.")
    # Local coordinates avoid cancellation for translated model coordinates.
    basis = np.concatenate((np.ones((len(cells), 4, 1)), xyz - xyz[:, :1]), axis=2)
    grad = np.linalg.inv(basis)[:, 1:, :].transpose(0, 2, 1)
    local = volume[:, None, None] * np.einsum("nik,njk->nij", grad, grad)
    rows = np.broadcast_to(cells[:, :, None], local.shape).ravel()
    cols = np.broadcast_to(cells[:, None, :], local.shape).ravel()
    matrix = coo_matrix((local.ravel(), (rows, cols)), shape=(len(points), len(points))).tocsr()
    return matrix, grad, volume


def solve_dirichlet(matrix, rhs, anchors, values, initial=None, tolerance=1e-11, maximum_iterations=5000):
    anchors = np.asarray(anchors, dtype=bool)
    _, labels = connected_components(matrix, directed=False)
    if set(np.unique(labels)) - set(labels[anchors]):
        raise ValueError("Every connected envelope component needs a scalar/vector anchor.")
    free = ~anchors
    result = np.asarray(values, dtype=float).copy()
    if not np.any(free):
        return result, {"iterations": 0, "relative_residual": 0.0}
    reduced = matrix[free][:, free]
    target = np.asarray(rhs)[free] - matrix[free][:, anchors] @ result[anchors]
    diagonal = reduced.diagonal()
    if np.any(diagonal <= 0):
        raise ValueError("Nonpositive FEM diagonal.")
    preconditioner = LinearOperator(reduced.shape, matvec=lambda x: x / diagonal)
    iterations = [0]
    def counted(_):
        iterations[0] += 1
    solution, info = cg(reduced, target, x0=None if initial is None else initial[free],
                        rtol=tolerance, atol=1e-12, maxiter=maximum_iterations,
                        M=preconditioner, callback=counted)
    residual = float(np.linalg.norm(reduced @ solution - target) / max(np.linalg.norm(target), 1e-12))
    if info != 0 or not np.all(np.isfinite(solution)):
        raise RuntimeError(f"FEM CG failed: info={info}, residual={residual:g}.")
    result[free] = solution
    return result, {"iterations": iterations[0], "relative_residual": residual}


def reconstruct(domain, anchor_scalars, anchor_gradients, baseline,
                upward_length=0.,build_direction=(0,1,0),platform_upward=False,upward_ramp=0.):
    """Diffuse vectors, optionally attracting free vectors toward build-up.

    Positive upward_length (mm) minimizes integral |grad v|^2 +
    |v - magnitude*up|^2 / upward_length^2 with exact model vector anchors.
    Smaller length gives a stronger upward preference. Zero is legacy diffusion.
    Model scalar anchors remain exact during the subsequent gradient fit.
    """
    up=np.asarray(build_direction,dtype=float)
    if (not np.isfinite(upward_length) or upward_length<0 or not np.isfinite(upward_ramp) or upward_ramp<0 or up.shape!=(3,)
            or not np.isfinite(up).all() or np.linalg.norm(up)<1e-12):
        raise ValueError('Require a nonnegative finite upward length and a finite nonzero build direction.')
    up=up/np.linalg.norm(up)
    cells = np.asarray(domain.cells).reshape(-1, 5)[:, 1:]
    matrix, grad, volume = assemble_fem(domain.points, cells)
    anchors = np.asarray(domain["model_anchor"], dtype=bool)
    component_count, labels = connected_components(matrix, directed=False)
    plate_nodes = np.abs(np.asarray(domain["plate_distance"])) < 1e-5 if "plate_distance" in domain.point_data else np.zeros(domain.n_points, dtype=bool)
    without_plate = set(np.unique(labels)) - set(labels[plate_nodes])
    vector = np.zeros((domain.n_points, 3))
    vector[anchors] = anchor_gradients
    # Nearest anchor initialization preserves a constant vector exactly and
    # avoids resolving it from zero on ill-conditioned clipped tetrahedra.
    # This applies to any affine field, without an analytic-y special case.
    anchor_distance,nearest = cKDTree(np.asarray(domain.points)[anchors]).query(domain.points)
    vector_initial = anchor_gradients[nearest]
    magnitudes=np.linalg.norm(anchor_gradients,axis=1)
    positive=magnitudes[magnitudes>1e-12]
    magnitude=float(np.median(positive)) if len(positive) else 0.
    if (upward_length>0 or platform_upward) and magnitude==0:
        raise ValueError('Upward regularization requires a nonzero boundary gradient scale.')
    target=magnitude*up
    vector_anchors=anchors.copy()
    platform_fixed=plate_nodes&~anchors if platform_upward else np.zeros(domain.n_points,dtype=bool)
    vector[platform_fixed]=target
    vector_anchors|=platform_fixed
    vector_matrix=matrix
    vector_rhs=np.zeros((domain.n_points,3))
    if upward_length>0:
        mass=np.zeros(domain.n_points)
        np.add.at(mass,cells.ravel(),np.repeat(volume/4.,4))
        screening=mass/upward_length**2
        if upward_ramp>0:
            # Quintic onset has zero first/second derivatives at both ends.
            # Distance is to sampled anchor vertices, not an exact model SDF.
            t=np.clip(anchor_distance/upward_ramp,0.,1.)
            screening*=t**3*(10+t*(-15+6*t))
        vector_matrix=matrix+diags(screening)
        vector_rhs=screening[:,None]*target
    reports = []
    # Natural outer boundaries; optional platform vectors never override model
    # anchors. Preserve model gradient units rather than normalizing to one.
    for axis in range(3):
        vector[:, axis], report = solve_dirichlet(vector_matrix, vector_rhs[:,axis], vector_anchors,
                                                  vector[:, axis], initial=vector_initial[:, axis])
        reports.append(report)
    target_vector = vector[cells].mean(axis=1)
    local_rhs = volume[:, None] * np.einsum("nik,nk->ni", grad, target_vector)
    rhs = np.zeros(domain.n_points)
    np.add.at(rhs, cells.ravel(), local_rhs.ravel())
    values = np.zeros(domain.n_points)
    values[anchors] = anchor_scalars
    scalar, scalar_report = solve_dirichlet(matrix, rhs, anchors, values, initial=baseline)
    actual_gradient = np.einsum("ni,nik->nk", scalar[cells], grad)
    residual = np.linalg.norm(actual_gradient - target_vector, axis=1)
    vector_norm=np.linalg.norm(vector,axis=1)
    gradient_norm=np.linalg.norm(actual_gradient,axis=1)
    return scalar, vector, actual_gradient, {"vector_solves": reports, "scalar_solve": scalar_report,
        "upward_regularization":dict(length_mm=float(upward_length),platform_upward=bool(platform_upward),
            ramp_mm=float(upward_ramp),ramp_distance='nearest model-anchor vertex',
            platform_fixed_nodes=int(platform_fixed.sum()),build_direction=up.tolist(),target_gradient_magnitude=magnitude,
            boundary_vector_error_max=float(np.max(np.linalg.norm(vector[anchors]-anchor_gradients,axis=1))),
            minimum_vector_up_dot=float(np.min((vector@up)/np.maximum(vector_norm,1e-15))),
            minimum_reconstructed_up_dot=float(np.min((actual_gradient@up)/np.maximum(gradient_norm,1e-15))),
            non_upward_reconstructed_cells=int(np.count_nonzero(actual_gradient@up<=0))),
        "domain_component_count": int(component_count), "domain_components_without_plate": len(without_plate),
        "gradient_fit_rms": float(np.sqrt(np.sum(volume * residual ** 2) / volume.sum())),
        "minimum_gradient_norm": float(np.linalg.norm(actual_gradient, axis=1).min()),
        "maximum_gradient_norm": float(np.linalg.norm(actual_gradient, axis=1).max()),
        "anchor_scalar_error_max": float(np.max(np.abs(scalar[anchors] - anchor_scalars)))}
