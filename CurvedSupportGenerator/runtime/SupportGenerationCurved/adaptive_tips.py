"""Opt-in radii saved by a coverage-sizing experiment; routes stay immutable."""
import time
import numpy as np


def apply_tip_profile(skeleton, seed_count, radii, body_radii, radius_cap, parent,
                      default_tip_radius):
    """Blend enlarged contact tips into existing stems over saved arc lengths.

    Only graphs with adaptive_tip_radius opt in. This does not fit the radii,
    move nodes, resize lower root contacts, or promise roof-layer coverage.
    """
    started=time.perf_counter()
    saved=np.asarray(skeleton.point_data['adaptive_tip_radius'],float)
    if saved.shape!=(skeleton.n_points,) or not np.isfinite(saved[:seed_count]).all() or np.any(saved[:seed_count]<=0):
        raise ValueError('Adaptive tip radii require finite positive values at every contact seed.')
    target=np.maximum(saved[:seed_count],default_tip_radius)
    if np.any(target>radius_cap[:seed_count]+1e-9):
        raise ValueError('Adaptive tip radii exceed their branch diameter cap.')
    lengths=np.asarray(skeleton.point_data.get('adaptive_tip_transition_mm',np.full(skeleton.n_points,3.)),float)
    if lengths.shape!=(skeleton.n_points,) or not np.isfinite(lengths[:seed_count]).all() or np.any(lengths[:seed_count]<=0):
        raise ValueError('Adaptive tip transition lengths must be positive and finite.')
    points=np.asarray(skeleton.points);before=np.asarray(radii).copy();result=before.copy()
    roots=np.asarray(skeleton.point_data.get('is_root',parent<0),bool)
    short=0;visits=0
    # TODO(performance): if recorded profile_seconds becomes material, replace
    # this bounded per-seed neck walk with batched CSR propagation and cache the
    # profile by graph/radius hashes across the two sweep passes and workers.
    # Never repeat the surface coverage fit during individual layer extraction.
    for seed in range(seed_count):
        node=seed;arc=0.;length=float(lengths[seed])
        while node>=0 and arc<length+1e-9:
            if roots[node] and node!=seed:
                short+=1;break
            w=np.clip(1-arc/length,0,1);w=w*w*(3-2*w)
            requested=body_radii[node]+max(0.,target[seed]-body_radii[node])*w
            if node==seed:requested=target[seed]
            result[node]=max(result[node],requested);visits+=1
            below=int(parent[node])
            if below<0:break
            arc+=float(np.linalg.norm(points[below]-points[node]));node=below
    result=np.minimum(result,radius_cap)
    # Existing root-foot construction and manual larger tips are authoritative.
    result[roots]=before[roots]
    if np.any(result<before-1e-9):
        raise ValueError('Adaptive tip sizing must not shrink existing geometry.')
    return result,dict(seed_radius_min_mm=float(target.min()),seed_radius_max_mm=float(target.max()),
        enlarged_seeds=int((target>default_tip_radius+1e-9).sum()),changed_nodes=int((result>before+1e-9).sum()),
        transition_min_mm=float(lengths[:seed_count].min()),transition_max_mm=float(lengths[:seed_count].max()),
        short_necks_reaching_root=short,profile_node_visits=visits,profile_seconds=time.perf_counter()-started,
        policy='Saved per-seed radii, smoothstep fade toward the existing body over saved arc length; lower roots unchanged.')
