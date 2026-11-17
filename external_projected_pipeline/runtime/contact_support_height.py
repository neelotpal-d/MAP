"""Contact-gap correction in both directions at model/support interfaces."""
from __future__ import annotations

import numpy as np
import igl

RAY_QUERY_BATCH_SIZE = 65536


def validate_contact_height_config(config: dict) -> dict:
    mode = str(config.get('direction_mode', 'model_normal'))
    gap = float(config.get('gap', 0.0))
    max_distance = float(config.get('max_query_distance', 2.0))
    if mode not in {'model_normal', 'nearest'}:
        raise ValueError('contact_support_height.direction_mode must be model_normal or nearest.')
    if not np.isfinite([gap, max_distance]).all() or gap < 0 or max_distance <= 0:
        raise ValueError('Contact gap must be non-negative and max_query_distance positive.')
    maximum = config.get('maximum_bead_height')
    if maximum is not None and (not np.isfinite(float(maximum)) or float(maximum) <= 0):
        raise ValueError('Contact maximum_bead_height must be finite and positive.')
    return dict(config, direction_mode=mode, gap=gap, max_query_distance=max_distance)


def _triangle_ray_distances(points, directions, triangles):
    """Double-precision, barycentrically checked ray hits on supplied triangles."""
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(directions, edge2)
    determinant = np.einsum('ij,ij->i', edge1, cross)
    scale = np.linalg.norm(edge1, axis=1) * np.linalg.norm(edge2, axis=1)
    valid = np.abs(determinant) > 1e-14 * np.maximum(scale, 1e-30)
    inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=valid)
    delta = points - triangles[:, 0]
    u = np.einsum('ij,ij->i', delta, cross) * inverse
    q = np.cross(delta, edge1)
    v = np.einsum('ij,ij->i', directions, q) * inverse
    distance = np.einsum('ij,ij->i', edge2, q) * inverse
    valid &= (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1. + 1e-9) & (distance > 1e-7)
    return np.where(valid & np.isfinite(distance), distance, np.inf)


def correct_contact_heights(rows: np.ndarray, references: list[tuple], config: dict,
                            *, source_layer: dict | None = None,
                            nearest_queries: list | None = None,
                            selected_contact_mask: np.ndarray | None = None) -> dict:
    """Remove the air gap at model/contact-support and support/model-foot interfaces.

    Reference tuples contain (vertices, faces, layer_metadata, cached_igl_aabb).
    The legacy ``model_normal`` mode casts along the current path's downward
    surface normal, also for a support footing. The original slicing field is
    not part of the OBJ export. Geometry and nozzle positions remain unchanged.
    Omitting source_layer retains the original model-on-support API behavior.
    Optional nearest_queries contain per-reference squared distances and face
    indices for exactly these point coordinates; rounding is checked by callers.
    Optional selected_contact_mask receives the per-point contact selection so
    downstream recovery can retire a superseded platform-fallback flag.
    """
    config = validate_contact_height_config(config)
    source_layer = {'part': 'model'} if source_layer is None else source_layer
    source_part = source_layer.get('part')
    model_footing = (source_part == 'support' and source_layer.get('is_model_base_layer', False)
                    and not source_layer.get('is_platform_layer', False)
                    and not source_layer.get('is_planar_base_layer', False))
    interface = 'model_on_support' if source_part == 'model' else ('support_on_model' if model_footing else 'none')
    count = len(rows)
    if selected_contact_mask is not None:
        if selected_contact_mask.shape != (count,) or selected_contact_mask.dtype != np.bool_:
            raise ValueError('Contact selection output must be a boolean array matching the points.')
        selected_contact_mask[:] = False
    if nearest_queries is not None:
        if len(nearest_queries) != len(references) or any(
                len(squared) != count or len(faces) != count for squared, faces in nearest_queries):
            raise ValueError('Cached contact queries must match every reference and point.')
    best = np.full(count, np.inf)
    winner = np.full(count, -1, dtype=int)
    normals = rows[:, 3:6].copy()
    sizes = np.linalg.norm(normals, axis=1)
    valid_normals = np.isfinite(normals).all(axis=1) & (sizes > 1e-12)
    normals /= np.maximum(sizes[:, None], 1e-12)
    normals[normals[:, 1] < 0] *= -1
    points = rows[:, :3]
    epsilon = 1e-7
    if config['direction_mode'] == 'model_normal':
        ids = np.flatnonzero(valid_normals)
        directions = -normals
        origins = points - epsilon * normals
    recovered_ray_queries = 0
    rejected_distant_interface_candidates = 0
    interface_gaps = []
    for _, _, metadata, _ in references:
        eligible = ((source_part == 'model' and metadata.get('part') == 'support' and metadata.get('is_contact_layer', False))
            or (model_footing and metadata.get('part') == 'model'))
        if not eligible:
            interface_gaps.append(None)
            continue
        owner = source_layer if model_footing else metadata
        field = config.get('gap_metadata_field')
        gap = float(owner.get('upstream_metadata', {}).get(field, config['gap']) if field else config['gap'])
        if not np.isfinite(gap) or gap < 0:
            raise ValueError('Invalid interface gap')
        interface_gaps.append(gap)
    for index, (vertices, faces, metadata, tree) in enumerate(references):
        if config['direction_mode'] == 'nearest':
            if nearest_queries is None:
                squared, _, _ = tree.squared_distance(vertices, faces, points)
            else:
                squared = nearest_queries[index][0]
            distances = np.sqrt(np.maximum(np.asarray(squared), 0.0)).reshape(-1)
        else:
            distances = np.full(count, np.inf)
            for start in range(0, len(ids), RAY_QUERY_BATCH_SIZE):
                chunk = ids[start:start + RAY_QUERY_BATCH_SIZE]
                hits, _, _ = tree.intersect_ray_first(
                    vertices, faces, origins[chunk], directions[chunk])
                hits = np.asarray(hits).reshape(-1)
                accepted = (hits >= 0) & (hits < len(faces))
                selected_ids = chunk[accepted]
                distances[selected_ids] = _triangle_ray_distances(points[selected_ids], directions[selected_ids],
                    vertices[faces[hits[accepted]]])
                # Some native ray queries miss exact shared edges. A nearest
                # triangle is only a candidate: require a true directional hit
                # inside that triangle before treating it as underlying support.
                missing = chunk[~np.isfinite(distances[chunk])]
                if len(missing):
                    if nearest_queries is None:
                        _, nearest, _ = tree.squared_distance(vertices, faces, points[missing])
                        nearest = np.asarray(nearest).reshape(-1)
                    else:
                        nearest = nearest_queries[index][1][missing]
                    usable = (nearest >= 0) & (nearest < len(faces))
                    missing = missing[usable]
                    fallback = _triangle_ray_distances(points[missing], directions[missing], vertices[faces[nearest[usable]]])
                    distances[missing] = fallback
                    recovered_ray_queries += int(np.isfinite(fallback).sum())
        interface_gap = interface_gaps[index]
        if interface_gap is not None and config.get('maximum_bead_height') is not None:
            reach = min(config['max_query_distance'], interface_gap + float(config['maximum_bead_height']))
            rejected_distant_interface_candidates += int(np.count_nonzero(np.isfinite(distances) & (distances > reach)))
            distances = np.where((distances > interface_gap + epsilon) & (distances <= reach), distances, np.inf)
        replace = distances < best - 1e-9
        winner[replace] = index
        best[replace] = distances[replace]
        # Same-material support on an exact tie takes priority over an air-gap
        # interface in either direction. This preserves ordinary bead thickness.
        if metadata.get('part') == source_part:
            finite = np.flatnonzero(np.isfinite(distances) & np.isfinite(best))
            winner[finite[np.abs(distances[finite] - best[finite]) <= 1e-9]] = index
    old = rows[:, 6].copy()
    selected = np.zeros(count, dtype=bool)
    gaps = np.zeros(count)
    gap_field = config.get('gap_metadata_field')
    for index, (_, _, metadata, _) in enumerate(references):
        eligible = ((source_part == 'model' and metadata.get('part') == 'support'
                     and metadata.get('is_contact_layer', False))
                    or (model_footing and metadata.get('part') == 'model'))
        if not eligible:
            continue
        # Interface gap metadata belongs to the support side: the reference
        # contact for model-on-support, or the current footing for the reverse.
        gap_owner = source_layer if model_footing else metadata
        gap_value = gap_owner.get('upstream_metadata', {}).get(gap_field, config['gap']) if gap_field else config['gap']
        gap = float(gap_value)
        if not np.isfinite(gap) or gap < 0:
            raise ValueError(f'Invalid contact gap in {gap_owner.get("source_mesh")}: {gap_value}')
        mask = (winner == index) & (best <= config['max_query_distance'])
        if config['direction_mode'] == 'nearest':
            # A fixed-Y/platform fallback must not have a nearest-gap subtracted.
            mask &= np.isclose(old, best, atol=1e-8, rtol=1e-6)
        selected |= mask
        gaps[mask] = gap
    rows[selected, 6] = np.maximum(0.0, best[selected] - gaps[selected])
    if selected_contact_mask is not None:
        selected_contact_mask[:] = selected
    return dict(enabled=True, direction_mode=config['direction_mode'], configured_gap=config['gap'], interface=interface,
                selected_contact_points=int(selected.sum()), changed_points=int(np.count_nonzero(np.abs(rows[:, 6] - old) > 1e-9)),
                selected_model_on_support_points=int(selected.sum()) if interface == 'model_on_support' else 0,
                selected_support_on_model_points=int(selected.sum()) if interface == 'support_on_model' else 0,
                recovered_ray_triangle_queries=recovered_ray_queries,
                rejected_distant_interface_candidates=rejected_distant_interface_candidates,
                zero_after_correction=int(np.count_nonzero(selected & (rows[:, 6] <= 0))),
                no_directional_hit_points=int(np.count_nonzero(~np.isfinite(best))),
                max_subtracted_gap=float(gaps.max(initial=0.0)),
                max_measured_contact_distance=float(best[selected].max(initial=0.0)))
