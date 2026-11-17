"""
Workflow helpers layered on top of the heat-method numerical core.

This module owns mesh IO, boundary detection, source construction, and debug
preview utilities. It should stay separate from the PDE implementation so the
core solver remains reusable for non-boundary source configurations.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import TYPE_CHECKING

import numpy as np

from heat_method_core import _as_float_array
from heat_method_core import _as_int_array
from heat_method_core import validate_triangle_mesh

from heat_method_core import FloatArray
from heat_method_core import HeatMethodParams
from heat_method_core import HeatMethodResult
from heat_method_core import IntArray
from heat_method_core import SourceSpec
from heat_method_core import solve_heat_geodesic

if TYPE_CHECKING:
    import pyvista as pv

try:
    import potpourri3d as pp3d  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pp3d = None


_OBJ_NUMBER = rb'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?'
_OBJ_V_RECORD = re.compile(rb'(?m)^[ \t\v\f]*v(?=[ \t\v\f\r\n]|$)')
_OBJ_F_RECORD = re.compile(rb'(?m)^[ \t\v\f]*f(?=[ \t\v\f\r\n]|$)')
_OBJ_V = re.compile(rb'(?m)^[ \t]*v[ \t]+('+_OBJ_NUMBER+rb'[ \t]+'+_OBJ_NUMBER+rb'[ \t]+'+_OBJ_NUMBER+rb')[ \t]*\r?$')
_OBJ_INDEX = rb'[1-9][0-9]{0,17}'
_OBJ_F = re.compile(rb'(?m)^[ \t]*f[ \t]+('+_OBJ_INDEX+rb'[ \t]+'+_OBJ_INDEX+rb'[ \t]+'+_OBJ_INDEX+rb')[ \t]*\r?$')
_OBJ_UNUSUAL_WHITESPACE = re.compile(rb'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]|\r(?!\n)')


def _load_text_obj_geometry(mesh_path: Path) -> tuple[FloatArray, IntArray]:
    """Batch-parse ordinary generated triangles; retain the full OBJ fallback.

    Every vertex/face record must match the strict fast syntax, with vertices
    preceding faces. Negative/slash/polygon indices, extra vertex attributes,
    interleaved records and malformed input retain the original parser/errors.
    """
    data = mesh_path.read_bytes()
    first_face = _OBJ_F_RECORD.search(data)
    if (data.isascii() and not _OBJ_UNUSUAL_WHITESPACE.search(data) and first_face is not None
            and _OBJ_V_RECORD.search(data, first_face.end()) is None):
        vertices, faces = _OBJ_V.findall(data), _OBJ_F.findall(data)
        if (vertices and faces and len(vertices) == len(_OBJ_V_RECORD.findall(data))
                and len(faces) == len(_OBJ_F_RECORD.findall(data))):
            V = np.fromstring(b' '.join(vertices), dtype=np.float64, sep=' ').reshape(-1, 3)
            F = np.fromstring(b' '.join(faces), dtype=np.int64, sep=' ').reshape(-1, 3) - 1
            if np.isfinite(V).all() and F.min() >= 0 and F.max() < len(V):
                validate_triangle_mesh(V, F)
                return V, F
    return _load_general_text_obj_geometry(mesh_path)


def _load_general_text_obj_geometry(mesh_path: Path) -> tuple[FloatArray, IntArray]:
    """Read only OBJ vertex/face geometry, ignoring renderer-specific metadata."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with mesh_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.strip().split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "v":
                if len(fields) < 4:
                    raise ValueError(f"OBJ vertex on line {line_number} has fewer than three coordinates.")
                try:
                    vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
                except ValueError as exc:
                    raise ValueError(f"Invalid OBJ vertex coordinates on line {line_number}.") from exc
                continue
            if fields[0] != "f":
                continue
            if len(fields) < 4:
                raise ValueError(f"OBJ face on line {line_number} has fewer than three vertices.")
            indices: list[int] = []
            for token in fields[1:]:
                index_token = token.split("/", maxsplit=1)[0]
                try:
                    raw_index = int(index_token)
                except ValueError as exc:
                    raise ValueError(f"Invalid OBJ face index on line {line_number}.") from exc
                index = raw_index - 1 if raw_index > 0 else len(vertices) + raw_index
                if index < 0 or index >= len(vertices):
                    raise ValueError(f"OBJ face index on line {line_number} is outside the current vertex range.")
                indices.append(index)
            # OBJ may contain polygons. Fan triangulation retains the geometry
            # expected by the numerical routines without relying on VTK data.
            faces.extend([indices[0], indices[offset], indices[offset + 1]] for offset in range(1, len(indices) - 1))
    if not vertices or not faces:
        raise ValueError("OBJ contains no usable vertex/face geometry.")
    V = _as_float_array(np.asarray(vertices, dtype=np.float64))
    F = _as_int_array(np.asarray(faces, dtype=np.int64))
    validate_triangle_mesh(V, F)
    return V, F


def load_obj_mesh(path: str) -> tuple[FloatArray, IntArray]:
    """
    Load an OBJ triangle mesh and return plain V/F arrays.

    Text OBJ files use a small native geometry parser, so unrelated PyVista/VTK
    arrays cannot invalidate otherwise usable V/F data. Other formats retain
    the PyVista/trimesh fallback behavior.
    """
    mesh_path = Path(path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    if mesh_path.suffix.lower() == ".obj":
        return _load_text_obj_geometry(mesh_path)

    pyvista_mesh = None
    try:
        import pyvista as pv  # type: ignore

        pyvista_mesh = pv.read(str(mesh_path))
    except ImportError:
        pyvista_mesh = None

    if pyvista_mesh is not None:
        if pyvista_mesh.n_cells == 0:
            raise ValueError("Loaded mesh contains no cells.")
        if not pyvista_mesh.is_all_triangles:
            pyvista_mesh = pyvista_mesh.triangulate()

        V = _as_float_array(np.asarray(pyvista_mesh.points))
        faces = np.asarray(pyvista_mesh.faces)
        if faces.size % 4 != 0:
            raise ValueError("Unexpected PyVista face encoding for triangle mesh.")
        faces = faces.reshape(-1, 4)
        if not np.all(faces[:, 0] == 3):
            raise ValueError("Mesh contains non-triangular faces after triangulation.")
        F = _as_int_array(faces[:, 1:4])
        validate_triangle_mesh(V, F)
        return V, F

    try:
        import trimesh  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Loading meshes requires either pyvista or trimesh to be installed.",
        ) from exc

    trimesh_mesh = trimesh.load_mesh(str(mesh_path), process=False)
    if not isinstance(trimesh_mesh, trimesh.Trimesh):
        raise ValueError("Expected a single triangle mesh, not a mesh scene.")
    if trimesh_mesh.faces.shape[1] != 3:
        raise ValueError("Loaded mesh is not triangulated.")

    V = _as_float_array(np.asarray(trimesh_mesh.vertices))
    F = _as_int_array(np.asarray(trimesh_mesh.faces))
    validate_triangle_mesh(V, F)
    return V, F


def compute_unique_edges(F: IntArray) -> IntArray:
    """Return the unique undirected edges implied by the triangle indices."""
    F = _as_int_array(F)
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("F must have shape (m, 3).")
    edges = np.vstack((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]))
    return _as_int_array(np.unique(np.sort(edges, axis=1), axis=0))


def compute_edge_face_incidence(F: IntArray) -> tuple[IntArray, IntArray]:
    """
    Return unique edges and the number of incident faces for each edge.

    Edges with incidence count 1 are mesh boundary edges.
    """
    F = _as_int_array(F)
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("F must have shape (m, 3).")
    edges = np.vstack((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    return _as_int_array(unique_edges), _as_int_array(counts)


def find_boundary_edges(F: IntArray) -> IntArray:
    """
    Return mesh boundary edges as pairs of vertex indices.

    Boundary edges are incident to exactly one triangle.
    """
    edges, counts = compute_edge_face_incidence(F)
    return _as_int_array(edges[counts == 1])


def find_boundary_vertices(F: IntArray) -> IntArray:
    """Return the unique vertex indices that lie on the mesh boundary."""
    boundary_edges = find_boundary_edges(F)
    if boundary_edges.size == 0:
        return _as_int_array(np.array([], dtype=np.int64))
    return _as_int_array(np.unique(boundary_edges.reshape(-1)))


def build_vertex_adjacency(F: IntArray) -> dict[int, set[int]]:
    """Build a lightweight vertex-neighbor adjacency map."""
    F = _as_int_array(F)
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("F must have shape (m, 3).")

    adjacency: dict[int, set[int]] = {}
    for a, b, c in F:
        adjacency.setdefault(int(a), set()).update((int(b), int(c)))
        adjacency.setdefault(int(b), set()).update((int(a), int(c)))
        adjacency.setdefault(int(c), set()).update((int(a), int(b)))
    return adjacency


def _build_edge_adjacency(edges: IntArray) -> dict[int, list[int]]:
    adjacency: dict[int, list[int]] = {}
    for a, b in edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    return adjacency


def _walk_boundary_component(
    start: int,
    next_vertex: int,
    adjacency: dict[int, list[int]],
    edge_pool: set[tuple[int, int]],
) -> IntArray:
    path = [start, next_vertex]
    edge_pool.remove(tuple(sorted((start, next_vertex))))
    prev = start
    current = next_vertex

    while True:
        candidates = [
            neighbor
            for neighbor in adjacency.get(current, [])
            if neighbor != prev and tuple(sorted((current, neighbor))) in edge_pool
        ]
        if not candidates:
            break

        if len(candidates) > 1:
            raise ValueError(
                "Boundary graph is not chain/loop-like; encountered a branching boundary vertex.",
            )

        nxt = candidates[0]
        path.append(nxt)
        edge_pool.remove(tuple(sorted((current, nxt))))
        prev, current = current, nxt
        if current == start:
            break

    return _as_int_array(np.asarray(path, dtype=np.int64))


def stitch_boundary_loops(boundary_edges: IntArray) -> list[IntArray]:
    """
    Stitch boundary edges into ordered polylines or loops.

    This is intended for exact boundary toolpath extraction.
    """
    boundary_edges = _as_int_array(boundary_edges)
    if boundary_edges.size == 0:
        return []
    if boundary_edges.ndim != 2 or boundary_edges.shape[1] != 2:
        raise ValueError("boundary_edges must have shape (k, 2).")

    adjacency = _build_edge_adjacency(boundary_edges)
    edge_pool = {tuple(sorted(map(int, edge))) for edge in boundary_edges}
    loops: list[IntArray] = []

    degree_one_vertices = [vertex for vertex, neighbors in adjacency.items() if len(neighbors) == 1]
    start_vertices = degree_one_vertices + [
        vertex for vertex in adjacency if vertex not in degree_one_vertices
    ]

    for start in start_vertices:
        for neighbor in adjacency.get(start, []):
            edge_key = tuple(sorted((start, neighbor)))
            if edge_key not in edge_pool:
                continue
            loops.append(_walk_boundary_component(start, neighbor, adjacency, edge_pool))

    return loops


def build_boundary_sources(F: IntArray) -> SourceSpec:
    """Create a source specification from all boundary vertices."""
    boundary_vertices = find_boundary_vertices(F)
    if boundary_vertices.size == 0:
        raise ValueError("Mesh has no boundary vertices to use as sources.")
    return SourceSpec(vertex_indices=boundary_vertices)


def solve_boundary_distance_field(
    V: FloatArray,
    F: IntArray,
    params: HeatMethodParams | None = None,
    *,
    prefer_libigl: bool = True,
    backend: str = "native",
    return_intermediates: bool = False,
) -> HeatMethodResult:
    """
    Convenience wrapper for the common case where the mesh boundary is the source.
    """
    sources = build_boundary_sources(F)
    backend_normalized = backend.strip().lower()
    if backend_normalized not in {"native", "potpourri3d"}:
        raise ValueError("backend must be either 'native' or 'potpourri3d'.")

    if backend_normalized == "potpourri3d":
        if pp3d is None:
            raise ImportError(
                "Boundary distance backend 'potpourri3d' requires the optional `potpourri3d` package.",
            )
        distances = pp3d.compute_distance_multisource(
            np.asarray(V, dtype=np.float64),
            np.asarray(F, dtype=np.int64),
            np.asarray(sources.vertex_indices, dtype=np.int64).tolist(),
        )
        return HeatMethodResult(distance=_as_float_array(np.asarray(distances, dtype=np.float64)))

    return solve_heat_geodesic(
        V,
        F,
        sources,
        params=params,
        prefer_libigl=prefer_libigl,
        return_intermediates=return_intermediates,
    )


def to_pyvista_polydata(V: FloatArray, F: IntArray) -> "pv.PolyData":
    """Convert V/F arrays into a PyVista mesh for inspection or plotting."""
    try:
        import pyvista as pv  # type: ignore
    except ImportError as exc:
        raise ImportError("PyVista is required for PolyData conversion.") from exc

    V = _as_float_array(V)
    F = _as_int_array(F)
    validate_triangle_mesh(V, F)

    face_array = np.hstack(
        (
            np.full((F.shape[0], 1), 3, dtype=np.int64),
            F,
        ),
    ).reshape(-1)
    return pv.PolyData(V, face_array)


def preview_scalar_field(
    V: FloatArray,
    F: IntArray,
    scalar_field: FloatArray,
    *,
    title: str = "Heat Method Distance",
) -> None:
    """
    Visualize a per-vertex scalar field, likely using pyvista for debugging.
    """
    mesh = to_pyvista_polydata(V, F)
    scalars = _as_float_array(scalar_field)
    if scalars.shape != (V.shape[0],):
        raise ValueError("scalar_field must have shape (n_vertices,).")

    mesh.point_data["scalar_field"] = scalars
    mesh.plot(
        scalars="scalar_field",
        show_edges=False,
        cmap="viridis",
        scalar_bar_args={"title": title},
        return_cpos=False,
    )
