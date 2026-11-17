"""Layer sampling independent of the display grid, with local mesh refinement."""
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


def fem_layer_source(domain):
    """Preserve the actual piecewise-linear field instead of resampling it."""
    result = pv.UnstructuredGrid(domain.cells, domain.celltypes, np.asarray(domain.points, dtype=float))
    result['scalar'] = np.asarray(domain['scalar'], dtype=float)
    return result


def near_sweep_faces(surface, geometry):
    accelerator=getattr(geometry,'_exact_bvh',None)
    if accelerator is not None:
        return accelerator.near_sweep_faces(surface)
    """Conservative triangle/primitive bounds; no signed-distance queries."""
    triangles = np.asarray(surface.points)[surface.regular_faces]
    centers = triangles.mean(axis=1)
    half = np.max(np.abs(triangles-centers[:, None]), axis=1)
    radius = np.linalg.norm(triangles-centers[:, None], axis=2).max(axis=1)
    keep = np.zeros(len(triangles), dtype=bool)
    for start in range(0, len(triangles), 2048):
        stop = min(start+2048, len(triangles))
        query = cKDTree(centers[start:stop])
        for tree, ids, bound in geometry._primitive_buckets:
            pairs = query.sparse_distance_matrix(tree, float(radius[start:stop].max())+bound+1e-9, output_type='ndarray')
            if not len(pairs):
                continue
            rows = start+pairs['i']; primitives = ids[pairs['j']]
            possible = pairs['v'] <= radius[rows]+geometry.primitive_search_radius[primitives]+1e-9
            rows, primitives = rows[possible], primitives[possible]
            overlap = np.all(np.abs(centers[rows]-geometry.midpoints.data[primitives]) <=
                             half[rows]+geometry.primitive_half_extent[primitives]+1e-9, axis=1)
            keep[rows[overlap]] = True
    return keep


def adaptive_candidate(surface, geometry, tolerance=.05, maximum_passes=5, maximum_triangles=1_000_000):
    """Find thin features and resolve curved outlines without a fine volume grid.

    Split shared edges conformingly. Faces away from every swept primitive
    are discarded using conservative bounds, then only unresolved small
    features and coarse crossing faces request further refinement.
    """
    from .tree_geometry import subset_triangles, subdivide_selected_faces
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Adaptive layer tolerance must be finite and positive.')
    surface = surface.triangulate()
    original = surface.n_cells
    passes = 0
    budget_reached = False
    minimum_radius = float(min(np.min(geometry.ra), np.min(geometry.rb)))
    hidden_edge = max(1e-4, min(minimum_radius*.5, np.sqrt(8*tolerance*minimum_radius)))
    for number in range(maximum_passes+1):
        if not surface.n_cells:
            break
        keep = near_sweep_faces(surface, geometry)
        if not keep.all():
            surface = subset_triangles(surface, keep)
        if not surface.n_cells:
            break
        faces = surface.regular_faces
        points = np.asarray(surface.points)
        inside = geometry.evaluate(points) <= 0
        count = inside[faces].sum(axis=1)
        triangles = points[faces]
        longest = np.linalg.norm(triangles-np.roll(triangles, -1, axis=1), axis=2).max(axis=1)
        # Trunks need fewer facets than small tips for the same chord target.
        # A global minimum tip radius would over-refine the entire forest.
        _, nearest = geometry.midpoints.query(triangles.mean(axis=1))
        local_radius = np.minimum(geometry.ra[nearest], geometry.rb[nearest])
        outline_edge = np.sqrt(8*tolerance*np.maximum(local_radius,minimum_radius))
        selected = ((count == 0) & (longest > hidden_edge)) | ((count > 0) & (count < 3) & (longest > outline_edge))
        if not selected.any() or number == maximum_passes:
            break
        # A marked face becomes four faces; neighbours add at most three.
        # Bound each layer's mesh growth rather than refining a volume grid.
        if surface.n_cells+12*int(selected.sum())>maximum_triangles:
            budget_reached=True
            break
        surface = subdivide_selected_faces(surface, selected)
        passes += 1
    return surface, dict(source_triangles=original, candidate_triangles=surface.n_cells,
                         refinement_passes=passes, tolerance_mm=float(tolerance),
                         triangle_budget_reached=budget_reached)
