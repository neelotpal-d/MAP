"""
Core skeleton for a reusable heat-method geodesic solver on triangle meshes.

This module keeps the numerical API focused on plain vertex/face arrays so it
can be reused independently of mesh IO, boundary extraction, and visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

try:
    import igl  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    igl = None


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(slots=True)
class SourceSpec:
    vertex_indices: IntArray
    values: FloatArray | None = None


@dataclass(slots=True)
class HeatMethodParams:
    timestep_scale: float = 1.0
    epsilon: float = 1e-12
    use_direct_solver: bool = True
    cache_factorization: bool = False
    enforce_dirichlet_on_sources: bool = True


@dataclass(slots=True)
class HeatMethodOperators:
    laplacian: Any
    mass_matrix: Any
    gradient_operator: Any | None = None
    divergence_operator: Any | None = None
    face_areas: FloatArray | None = None


@dataclass(slots=True)
class HeatMethodResult:
    distance: FloatArray  # shape: (n,)
    heat: FloatArray | None = None  # shape: (n,)
    face_vector_field: FloatArray | None = None  # shape: (m, 3)
    timestep: float | None = None


def _as_float_array(values: NDArray[Any]) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def _as_int_array(values: NDArray[Any]) -> IntArray:
    return np.asarray(values, dtype=np.int64)


def _triangle_double_areas(V: FloatArray, F: IntArray) -> FloatArray:
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    return np.linalg.norm(np.cross(e1, e2), axis=1)


def _compute_face_gradients(
    V: FloatArray,
    F: IntArray,
    values: FloatArray,
    face_areas: FloatArray,
) -> FloatArray:
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    normals_twice = np.cross(v1 - v0, v2 - v0)
    safe_denom = np.maximum(face_areas, np.finfo(np.float64).eps)

    grad_phi0 = np.cross(normals_twice, v2 - v1) / safe_denom[:, None]
    grad_phi1 = np.cross(normals_twice, v0 - v2) / safe_denom[:, None]
    grad_phi2 = np.cross(normals_twice, v1 - v0) / safe_denom[:, None]

    return (
        values[F[:, 0], None] * grad_phi0
        + values[F[:, 1], None] * grad_phi1
        + values[F[:, 2], None] * grad_phi2
    )


def _build_gradient_operator_manual(
    V: FloatArray,
    F: IntArray,
    face_areas: FloatArray,
) -> sparse.csr_matrix:
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    normals_twice = np.cross(v1 - v0, v2 - v0)
    safe_denom = np.maximum(face_areas, np.finfo(np.float64).eps)

    grad_phi0 = np.cross(normals_twice, v2 - v1) / safe_denom[:, None]
    grad_phi1 = np.cross(normals_twice, v0 - v2) / safe_denom[:, None]
    grad_phi2 = np.cross(normals_twice, v1 - v0) / safe_denom[:, None]

    num_faces = F.shape[0]
    num_vertices = V.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for axis in range(3):
        base = axis * num_faces
        for local_idx, grads in enumerate((grad_phi0, grad_phi1, grad_phi2)):
            rows.extend(base + np.arange(num_faces))
            cols.extend(F[:, local_idx])
            data.extend(grads[:, axis])

    return sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(3 * num_faces, num_vertices),
        dtype=np.float64,
    ).tocsr()


def _build_divergence_operator_from_gradient(
    gradient_operator: sparse.spmatrix,
    face_areas: FloatArray,
) -> sparse.csr_matrix:
    area_weights = np.repeat(0.5 * face_areas, 3)
    area_matrix = sparse.diags(area_weights, format="csr")
    return (-gradient_operator.transpose() @ area_matrix).tocsr()


def _build_operators_manual(V: FloatArray, F: IntArray) -> HeatMethodOperators:
    num_vertices = V.shape[0]
    edge_pairs = np.vstack((F[:, [1, 2]], F[:, [2, 0]], F[:, [0, 1]]))
    edge_pairs = np.sort(edge_pairs, axis=1)
    unique_edges = np.unique(edge_pairs, axis=0)

    edge_lengths = np.linalg.norm(
        V[unique_edges[:, 0]] - V[unique_edges[:, 1]],
        axis=1,
    )
    if np.any(edge_lengths <= np.finfo(np.float64).eps):
        raise ValueError("Mesh contains zero-length edges.")

    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    face_double_areas = _triangle_double_areas(V, F)
    if np.any(face_double_areas <= np.finfo(np.float64).eps):
        raise ValueError("Mesh contains degenerate faces with zero area.")

    # Cotangents are computed from the two edge vectors incident to each
    # triangle vertex, using dot(u, v) / ||u x v||.
    cot0 = np.einsum("ij,ij->i", v1 - v0, v2 - v0) / face_double_areas
    cot1 = np.einsum("ij,ij->i", v0 - v1, v2 - v1) / face_double_areas
    cot2 = np.einsum("ij,ij->i", v0 - v2, v1 - v2) / face_double_areas

    I = np.concatenate(
        (F[:, 1], F[:, 2], F[:, 2], F[:, 0], F[:, 0], F[:, 1]),
    )
    J = np.concatenate(
        (F[:, 2], F[:, 1], F[:, 0], F[:, 2], F[:, 1], F[:, 0]),
    )
    W = 0.5 * np.concatenate((cot0, cot0, cot1, cot1, cot2, cot2))

    off_diag = sparse.coo_matrix((W, (I, J)), shape=(num_vertices, num_vertices))
    off_diag = off_diag.tocsr()
    diagonal = np.asarray(off_diag.sum(axis=1)).ravel()
    laplacian = sparse.diags(diagonal, format="csr") - off_diag

    vertex_areas = np.zeros(num_vertices, dtype=np.float64)
    np.add.at(vertex_areas, F[:, 0], face_double_areas / 6.0)
    np.add.at(vertex_areas, F[:, 1], face_double_areas / 6.0)
    np.add.at(vertex_areas, F[:, 2], face_double_areas / 6.0)
    mass_matrix = sparse.diags(vertex_areas, format="csr")

    gradient_operator = _build_gradient_operator_manual(V, F, face_double_areas)
    divergence_operator = _build_divergence_operator_from_gradient(
        gradient_operator,
        face_double_areas,
    )
    return HeatMethodOperators(
        laplacian=laplacian,
        mass_matrix=mass_matrix,
        gradient_operator=gradient_operator,
        divergence_operator=divergence_operator,
        face_areas=0.5 * face_double_areas,
    )


def _apply_dirichlet_constraints(
    matrix: sparse.spmatrix,
    rhs: FloatArray,
    constrained_indices: IntArray,
    constrained_values: FloatArray,
) -> tuple[sparse.csr_matrix, FloatArray]:
    """
    Apply strong Dirichlet constraints by replacing constrained rows only.

    This preserves the simpler "identity row" enforcement pattern, at the cost
    of losing symmetry in the modified system. If we later want to use strictly
    symmetric solvers throughout, switch to a reduced solve or row/column
    elimination with RHS correction.
    """
    A = matrix.tolil(copy=True)
    b = np.asarray(rhs, dtype=np.float64).copy()
    indices = np.asarray(constrained_indices, dtype=np.int64)
    values = np.asarray(constrained_values, dtype=np.float64)

    if indices.ndim != 1 or values.ndim != 1 or indices.shape != values.shape:
        raise ValueError("Dirichlet indices and values must be matching 1D arrays.")
    if indices.size == 0:
        return A.tocsr(), b

    # Use unique constrained rows to avoid double-applying repeated source indices.
    unique_indices, unique_pos = np.unique(indices, return_index=True)
    unique_values = values[unique_pos]

    for idx, value in zip(unique_indices, unique_values):
        A[idx, :] = 0.0
        A[idx, idx] = 1.0
        b[idx] = value

    return A.tocsr(), b


def validate_triangle_mesh(V: FloatArray, F: IntArray) -> None:
    """Validate array shapes and basic triangle mesh assumptions."""
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError("V must have shape (n, 3).")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("F must have shape (m, 3).")
    if V.shape[0] == 0:
        raise ValueError("Mesh must contain at least one vertex.")
    if F.shape[0] == 0:
        raise ValueError("Mesh must contain at least one triangle.")
    if not np.isfinite(V).all():
        raise ValueError("Vertex coordinates must be finite.")
    if np.issubdtype(F.dtype, np.floating):
        raise ValueError("F must contain integer triangle indices.")
    if np.min(F) < 0 or np.max(F) >= V.shape[0]:
        raise ValueError("Triangle indices are out of bounds.")
    if np.any(F[:, 0] == F[:, 1]) or np.any(F[:, 1] == F[:, 2]) or np.any(F[:, 2] == F[:, 0]):
        raise ValueError("Faces must reference three distinct vertices.")


def mean_edge_length(V: FloatArray, F: IntArray) -> float:
    """Return the mean unique-edge length used to derive the heat timestep."""
    validate_triangle_mesh(V, F)
    edges = np.vstack((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    lengths = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    if lengths.size == 0:
        raise ValueError("Unable to compute mean edge length from mesh.")
    return float(np.mean(lengths))


def choose_heat_timestep(
    V: FloatArray,
    F: IntArray,
    params: HeatMethodParams,
) -> float:
    """Choose a timestep, typically proportional to mean edge length squared."""
    h = mean_edge_length(V, F)
    return params.timestep_scale * h * h


def build_heat_operators(
    V: FloatArray,
    F: IntArray,
    *,
    prefer_libigl: bool = True,
) -> HeatMethodOperators:
    """
    Assemble the discrete operators required by the heat method.

    Preferred path:
    - use libigl bindings for cotangent Laplacian, mass matrix, and gradient

    Fallback path:
    - assemble operators manually with NumPy/SciPy
    """
    validate_triangle_mesh(V, F)
    V = _as_float_array(V)
    F = _as_int_array(F)

    if prefer_libigl and igl is not None:
        laplacian = igl.cotmatrix(V, F).tocsr()
        mass_matrix = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI).tocsr()
        gradient_operator = igl.grad(V, F).tocsr()
        face_areas = 0.5 * _triangle_double_areas(V, F)
        divergence_operator = _build_divergence_operator_from_gradient(
            gradient_operator,
            2.0 * face_areas,
        )
        return HeatMethodOperators(
            laplacian=-laplacian,
            mass_matrix=mass_matrix,
            gradient_operator=gradient_operator,
            divergence_operator=divergence_operator,
            face_areas=face_areas,
        )

    return _build_operators_manual(V, F)


def build_source_vector(
    num_vertices: int,
    sources: SourceSpec,
) -> FloatArray:
    """
    Build the source indicator vector u0.

    First version uses source vertices only.
    """
    if num_vertices <= 0:
        raise ValueError("num_vertices must be positive.")

    indices = np.asarray(sources.vertex_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("Source vertex indices must be a 1D array.")
    if indices.size == 0:
        raise ValueError("At least one source vertex is required.")
    if np.min(indices) < 0 or np.max(indices) >= num_vertices:
        raise ValueError("Source vertex indices are out of bounds.")

    source_vector = np.zeros(num_vertices, dtype=np.float64)
    if sources.values is None:
        source_vector[np.unique(indices)] = 1.0
        return source_vector

    values = np.asarray(sources.values, dtype=np.float64)
    if values.shape != indices.shape:
        raise ValueError("Source values must match the shape of source vertex indices.")
    source_vector[indices] = values
    return source_vector


def solve_heat_step(
    operators: HeatMethodOperators,
    source_vector: FloatArray,
    timestep: float,
    params: HeatMethodParams,
) -> FloatArray:
    """
    Solve the short-time heat system:

        (M + tL) u = M u0
    """
    system_matrix = (operators.mass_matrix + timestep * operators.laplacian).tocsr()
    rhs = operators.mass_matrix @ source_vector
    if params.use_direct_solver:
        return _as_float_array(sparse_linalg.spsolve(system_matrix, rhs))

    solution, info = sparse_linalg.cg(system_matrix, rhs)
    if info != 0:
        raise RuntimeError(f"Conjugate-gradient heat solve failed with info={info}.")
    return _as_float_array(solution)


def compute_normalized_face_vector_field(
    V: FloatArray,
    F: IntArray,
    operators: HeatMethodOperators,
    heat_values: FloatArray,
    params: HeatMethodParams,
) -> FloatArray:
    """
    Compute the normalized descent vector field:

        X = -grad(u) / ||grad(u)||
    """
    if operators.face_areas is None:
        raise ValueError("Face areas are required to compute the vector field.")

    if operators.gradient_operator is not None:
        gradients = operators.gradient_operator @ heat_values
        num_faces = F.shape[0]
        face_gradients = np.column_stack(
            (
                gradients[0:num_faces],
                gradients[num_faces : 2 * num_faces],
                gradients[2 * num_faces : 3 * num_faces],
            ),
        )
    else:
        face_gradients = _compute_face_gradients(V, F, heat_values, 2.0 * operators.face_areas)

    norms = np.linalg.norm(face_gradients, axis=1)
    safe_norms = np.maximum(norms, params.epsilon)
    return -face_gradients / safe_norms[:, None]


def solve_distance_poisson(
    operators: HeatMethodOperators,
    face_vector_field: FloatArray,
    sources: SourceSpec,
    params: HeatMethodParams,
) -> FloatArray:
    """
    Reconstruct the scalar distance field from the normalized vector field.

    With the positive stiffness matrix L = G^T A G and divergence operator
    Div = -G^T A, the least-squares reconstruction is:

        L phi = -Div(X)
    """
    if operators.divergence_operator is None:
        raise ValueError("Divergence operator is required for Poisson reconstruction.")

    #rhs = (-operators.divergence_operator @ face_vector_field.reshape(-1))
    rhs = -(operators.divergence_operator @ face_vector_field.T.reshape(-1))
    rhs = np.asarray(rhs, dtype=np.float64)
    laplacian = operators.laplacian.tocsr()
    if params.enforce_dirichlet_on_sources:
        constrained_values = np.zeros_like(
            np.asarray(sources.vertex_indices, dtype=np.int64),
            dtype=np.float64,
        )
        laplacian, rhs = _apply_dirichlet_constraints(
            laplacian,
            rhs,
            np.asarray(sources.vertex_indices, dtype=np.int64),
            constrained_values,
        )
    else:
        # Fix the nullspace for the pure Neumann-style solve by anchoring one vertex.
        laplacian, rhs = _apply_dirichlet_constraints(
            laplacian,
            rhs,
            np.array([0], dtype=np.int64),
            np.array([0.0], dtype=np.float64),
        )
    #print(laplacian)
    #print(rhs)
    if params.use_direct_solver:
        return _as_float_array(sparse_linalg.spsolve(laplacian, rhs))
    solution, info = sparse_linalg.cg(laplacian, rhs)
    if info != 0:
        raise RuntimeError(f"Conjugate-gradient Poisson solve failed with info={info}.")
    return _as_float_array(solution)


def shift_distance_to_sources(
    distance: FloatArray,
    sources: SourceSpec,
) -> FloatArray:
    """Shift the field so source values are zero or as close to zero as possible."""
    shifted = np.asarray(distance, dtype=np.float64).copy()
    source_values = shifted[np.asarray(sources.vertex_indices, dtype=np.int64)]
    shifted -= np.min(source_values)
    shifted[shifted < 0.0] = 0.0
    return shifted


def solve_heat_geodesic(
    V: FloatArray,
    F: IntArray,
    sources: SourceSpec,
    params: HeatMethodParams | None = None,
    *,
    prefer_libigl: bool = True,
    return_intermediates: bool = False,
) -> HeatMethodResult:
    """
    End-to-end heat-method geodesic solve for a triangle mesh.

    This is the main reusable entry point for contour and toolpath planning.
    """
    V = _as_float_array(V)
    F = _as_int_array(F)
    validate_triangle_mesh(V, F)

    if params is None:
        params = HeatMethodParams()

    operators = build_heat_operators(V, F, prefer_libigl=prefer_libigl)
    timestep = choose_heat_timestep(V, F, params)
    source_vector = build_source_vector(V.shape[0], sources)
    heat_values = solve_heat_step(operators, source_vector, timestep, params)
    face_vector_field = compute_normalized_face_vector_field(
        V,
        F,
        operators,
        heat_values,
        params,
    )
    distance = solve_distance_poisson(operators, face_vector_field, sources, params)
    distance = shift_distance_to_sources(distance, sources)

    if return_intermediates:
        return HeatMethodResult(
            distance=distance,
            heat=heat_values,
            face_vector_field=face_vector_field,
            timestep=timestep,
        )

    return HeatMethodResult(distance=distance, timestep=timestep)
