"""Tree-independent exterior tetrahedral computation domain.

The background lattice is only a meshing aid. Its tetrahedra are cut against
continuous signed-distance samples, rather than retained as binary voxels.
The cavity approximates the original mesh; it is not a conforming remesh.
"""
from __future__ import annotations

import itertools
import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree


def distance_samples(dataset,surface,threads):
    if threads==0:
        return np.asarray(dataset.compute_implicit_distance(surface)['implicit_distance']).copy()
    from .tree_geometry import configure_geometry_threads,sample_mesh_distance
    from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance
    from vtkmodules.vtkFiltersGeneral import vtkSampleImplicitFunctionFilter
    configure_geometry_threads(threads)
    function=vtkImplicitPolyDataDistance();function.SetInput(surface)
    sampler=vtkSampleImplicitFunctionFilter();sampler.SetImplicitFunction(function)
    sampler.SetComputeGradients(False)
    return sample_mesh_distance(function,sampler,dataset.points)


def build_envelope(mesh: pv.PolyData, spacing: float, padding: float,
                   direction: np.ndarray, plate: float | None = None,
                   maximum_points: int = 500_000,
                   trajectory_points: np.ndarray | None = None,
                   distance_threads: int = 8,
                   model_overlap: float = 0.) -> tuple[pv.UnstructuredGrid, dict]:
    if spacing <= 0 or padding <= 0:
        raise ValueError("Spacing and padding must be positive.")
    if not np.isfinite(model_overlap) or model_overlap<0:
        raise ValueError('Model overlap must be finite and nonnegative.')
    direction = np.asarray(direction, dtype=float)
    if not np.all(np.isfinite(direction)) or np.linalg.norm(direction) < 1e-12:
        raise ValueError("Build direction must be finite and nonzero.")
    direction = direction / np.linalg.norm(direction)
    points = np.asarray(mesh.points, dtype=float)
    plate = float(np.min(points @ direction)) if plate is None else float(plate)
    if not np.isfinite(plate):
        raise ValueError("Build plate must be finite.")
    if trajectory_points is not None:
        trajectory_points = np.asarray(trajectory_points, dtype=float).reshape(-1, 3)
        if not len(trajectory_points) or not np.all(np.isfinite(trajectory_points)):
            raise ValueError("Trajectory points must be nonempty and finite.")
        seed = np.vstack((points, trajectory_points))
        lower, upper = seed.min(axis=0) - padding, seed.max(axis=0) + padding
        shell = None
    else:
        projected = points - ((points @ direction) - plate)[:, None] * direction
        seed = np.vstack((points, projected))
        # Minkowski expansion by a box is deliberately conservative. Clip the
        # expanded envelope back to the exact plate plane afterward.
        hull0 = ConvexHull(seed)
        seed = seed[hull0.vertices]
        offsets = padding * np.asarray(list(itertools.product((-1, 1), repeat=3)))
        expanded = (seed[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        hull = ConvexHull(expanded)
        shell = pv.PolyData(expanded, np.column_stack((np.full(len(hull.simplices), 3), hull.simplices)).ravel())
        shell = shell.clean().compute_normals(consistent_normals=True, auto_orient_normals=True)
        lower, upper = expanded.min(axis=0), expanded.max(axis=0)
    shape = np.ceil((upper - lower) / spacing).astype(int) + 1
    count = int(np.prod(shape))
    if count > maximum_points:
        raise ValueError(f"Envelope needs {count:,} background points; limit is {maximum_points:,}. Increase --spacing.")
    grid = pv.ImageData(dimensions=tuple(shape), spacing=(spacing,) * 3, origin=lower)
    if shell is None:
        grid['implicit_distance']=distance_samples(grid,mesh,distance_threads)
        grid.point_data["model_distance"] = np.asarray(grid["implicit_distance"]).copy()
        # An alpha-like offset volume: union of the padded model and balls
        # around trajectory samples. This is not a Delaunay alpha complex.
        ray_distance = cKDTree(trajectory_points).query(grid.points)[0]
        envelope_distance = np.minimum(grid["model_distance"] - padding, ray_distance - padding)
        grid.point_data["envelope_distance"] = np.asarray(envelope_distance).copy()
    else:
        grid['implicit_distance']=distance_samples(grid,shell,distance_threads)
        grid.point_data["envelope_distance"] = np.asarray(grid["implicit_distance"]).copy()
    # Copy before deleting the VTK-backed array; a popped view can otherwise
    # alias freed storage when the next point array is allocated.
    del grid.point_data["implicit_distance"]
    grid.point_data["plate_distance"] = np.asarray(grid.points) @ direction - plate
    domain = grid.to_tetrahedra(tetra_per_cell=6)
    domain = domain.clip_scalar(scalars="envelope_distance", value=0, invert=True)
    domain = domain.clip_scalar(scalars="plate_distance", value=0, invert=False)
    if shell is not None:
        domain['implicit_distance']=distance_samples(domain,mesh,distance_threads)
        domain.point_data["model_distance"] = np.asarray(domain["implicit_distance"]).copy()
        del domain.point_data["implicit_distance"]
    domain = domain.clip_scalar(scalars="model_distance", value=-model_overlap, invert=False)
    domain = domain.triangulate().clean()
    # Intersections coincident with the plate can produce triangles/lines.
    # They carry no volume and must not enter the volume FEM system.
    if np.any(domain.celltypes != pv.CellType.TETRA):
        domain = domain.extract_cells(domain.celltypes == pv.CellType.TETRA).clean()
    if not domain.n_cells or np.any(domain.celltypes != pv.CellType.TETRA):
        raise RuntimeError("Envelope clipping did not produce a nonempty tetrahedral domain.")
    cells = np.asarray(domain.cells).reshape(-1, 5)[:, 1:]
    xyz = np.asarray(domain.points, dtype=float)[cells]
    edges = xyz[:, 1:] - xyz[:, :1]
    volume = np.abs(np.linalg.det(edges)) / 6
    scale = np.max(np.linalg.norm(edges, axis=2), axis=1)
    valid = volume > np.maximum(scale ** 3 * 1e-12, 1e-18)
    dropped = int(np.count_nonzero(~valid))
    if dropped:
        domain = domain.extract_cells(valid).clean()
    if not domain.n_cells:
        raise RuntimeError("Envelope contains no nondegenerate volume cells.")
    if model_overlap>0:
        # A signed-distance function is 1-Lipschitz. Linear interpolation on
        # background tetrahedra errs by at most their diameter sqrt(3)*spacing.
        # Insetting the cavity by that amount retains the true exterior.
        # Re-evaluate distances: constraints belong to actual model-interior
        # nodes, not to the inaccurate zero crossing used for meshing.
        domain['model_distance']=distance_samples(domain,mesh,distance_threads)
        anchors=np.asarray(domain['model_distance'])<=max(1e-9,spacing*1e-7)
    else:
        anchors = np.abs(np.asarray(domain["model_distance"])) < max(1e-9, spacing * 1e-7)
    if not np.any(anchors):
        raise RuntimeError("No model-side cavity anchors were created; reduce spacing.")
    domain.point_data["model_anchor"] = anchors.astype(np.uint8)
    return domain, {"background_points": count, "points": domain.n_points,
                    "distance_threads":distance_threads,
                    "tetrahedra": domain.n_cells, "model_anchors": int(anchors.sum()),
                    "spacing": spacing, "padding": padding, "plate": plate,
                    "model_overlap_mm":float(model_overlap),
                    "exterior_coverage_bound_mm":float(np.sqrt(3)*spacing),
                    "degenerate_tetrahedra_removed": dropped,
                    "build_direction": direction.tolist(),
                    "envelope_mode": "gradient" if trajectory_points is not None else "convex",
                    "outer_representation": "padded model + trajectory sample balls" if trajectory_points is not None else "expanded convex hull",
                    "boundary_representation": "inset computation cavity with actual model-interior anchors" if model_overlap>0 else "piecewise-linear signed-distance cavity"}
