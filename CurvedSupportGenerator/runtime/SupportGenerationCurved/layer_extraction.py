"""Layer sampling independent of the display grid, with local mesh refinement."""
from time import perf_counter

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


def _model_clearance_rejections(triangles, geometry):
    """Certify whole triangles inside the model keepout using exact distance.

    For centroid c and farthest-vertex radius R, signed distance is
    1-Lipschitz, so d(c) + R bounds the distance at every triangle point.
    The 0.0001 mm margin reserves room for numerical distance error. This
    is a rejection proof, never a replacement for normal clipping.
    """
    triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    rejected = np.zeros(len(triangles), dtype=bool)
    for start in range(0, len(triangles), 10000):
        block = triangles[start:start + 10000]
        finite = np.isfinite(block).all(axis=(1, 2))
        ids = np.flatnonzero(finite)
        if not len(ids):
            continue
        block = block[ids]
        centers = block.mean(axis=1)
        radius = np.linalg.norm(block - centers[:, None], axis=2).max(axis=1)
        # Do not use the combined support field, its outside sentinels, or
        # bounded/clipped distances: this proof needs actual model distance.
        distance = np.asarray(geometry.distance(centers), dtype=float)
        rejected[start + ids] = (np.isfinite(distance) &
            (distance + radius + 1e-4 < float(geometry.side_gap)))
    return rejected


def adaptive_candidate(surface, geometry, tolerance=.05, maximum_passes=8, maximum_triangles=1_000_000, *, reject_model_clearance=False, proof_cache_entries=131072):
    """Find thin features and resolve curved outlines without a fine volume grid.

    Split shared edges conformingly. Faces away from every swept primitive
    are discarded using conservative bounds, then only unresolved small
    features and coarse crossing faces request further refinement.
    """
    from .tree_geometry import subset_triangles, subdivide_selected_faces
    from .triangle_proof_cache import TriangleProofCache
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Adaptive layer tolerance must be finite and positive.')
    surface = surface.triangulate()
    original = surface.n_cells
    # Only initial layer candidates opt in. Later nonlinear/detail repairs
    # retain their existing classifier and refinement policy.
    eligible = (reject_model_clearance and not hasattr(geometry, 'roof') and
        getattr(geometry, 'geometry_mode', None) == 'hard' and
        geometry.side_gap == geometry.contact_gap and
        np.isfinite(geometry.side_gap) and geometry.side_gap >= 0 and
        callable(getattr(geometry, 'distance', None)) and
        hasattr(geometry.mesh, 'n_open_edges') and geometry.mesh.n_open_edges == 0)
    rejection_report = dict(enabled=bool(eligible), tested_triangles=0,
                            rejected_triangles=0, seconds=0.)
    passes = 0
    budget_reached = False
    certify = getattr(geometry, 'certified_inside_triangles', None) if not hasattr(geometry, 'roof') else None
    # The geometry is fixed throughout this call. Local lifetime prevents stale
    # proofs after a layer/clearance/geometry change; no rounded spatial keys.
    proof_cache = TriangleProofCache(certify, proof_cache_entries) if certify is not None else None
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
        outside=np.flatnonzero(count==0)
        intersect=getattr(geometry,'possible_intersection_triangles',None) if not hasattr(geometry,'roof') else None
        if len(outside) and intersect is not None:
            keep=np.ones(len(faces),bool);keep[outside]=intersect(triangles[outside])
            if eligible:
                ids = outside[keep[outside]]
                if len(ids):
                    started = perf_counter()
                    rejected = _model_clearance_rejections(triangles[ids], geometry)
                    keep[ids[rejected]] = False
                    rejection_report['tested_triangles'] += len(ids)
                    rejection_report['rejected_triangles'] += int(rejected.sum())
                    rejection_report['seconds'] += perf_counter() - started
            if not keep.all():
                surface=subset_triangles(surface,keep)
                if not surface.n_cells:break
                faces=surface.regular_faces;points=np.asarray(surface.points)
                inside=geometry.evaluate(points)<=0;count=inside[faces].sum(axis=1);triangles=points[faces]
        longest = np.linalg.norm(triangles-np.roll(triangles, -1, axis=1), axis=2).max(axis=1)
        # Every unclassified face can contain several disconnected pieces.
        # Resolve those to the requested physical tolerance before clipping.
        selected = (count<3)&(longest>tolerance)
        # An all-inside face may still span a gap between different branches.
        # Do not pass those coarse faces to one-crossing edge clipping unless
        # one convex primitive certifies their entire interior.
        uncertain=np.flatnonzero((count==3)&(longest>tolerance))
        if len(uncertain):
            known=proof_cache(triangles[uncertain]) if proof_cache is not None else np.zeros(len(uncertain),bool)
            selected[uncertain[~known]]=True
        if not selected.any() or number == maximum_passes:
            break
        # Count actual shared split edges. The former 12-faces-per-marked-face
        # upper bound stopped well below the budget on dense marked regions.
        refined=subdivide_selected_faces(surface,selected,maximum_triangles=maximum_triangles)
        if refined is None:
            budget_reached=True
            break
        surface = refined
        passes += 1
    return surface, dict(source_triangles=original, candidate_triangles=surface.n_cells,
                         refinement_passes=passes, tolerance_mm=float(tolerance),
                         triangle_budget_reached=budget_reached,
                         interior_proof_cache=proof_cache.report() if proof_cache is not None else None,
                         early_clearance_rejection=rejection_report)
