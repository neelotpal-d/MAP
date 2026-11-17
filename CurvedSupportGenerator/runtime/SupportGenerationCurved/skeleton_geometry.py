"""Convert a saved, fully rooted progressive skeleton to tapered sweep primitives."""
from pathlib import Path
import json
import numpy as np
import pyvista as pv


def load_skeleton(source,allow_stalled=False,skip_unrooted=False):
    source=Path(source)
    folder=source if source.is_dir() else source.parent
    path=folder/'support_centerlines.vtp' if source.is_dir() else source
    report=json.loads((folder/'module_report.json').read_text())
    if report.get('unrooted_contacts',1)!=0 and not (allow_stalled or skip_unrooted):
        raise ValueError('Skeleton has stalled contacts; use --allow-stalled-skeleton to retain existing branches and continue, or resolve the contacts first.')
    report=dict(report,stalled_skeleton_allowed=bool(allow_stalled))
    graph=pv.read(path)
    if skip_unrooted and report.get('unrooted_contacts',0):
        graph,report=omit_unrooted(graph,report)
    return graph,report


def omit_unrooted(graph,report):
    """Copy the rooted forest; retain seed-prefix ordering and source ids."""
    edges=graph.lines.reshape(-1,3)
    if np.any(edges[:,0]!=2):raise ValueError('Expected two-point skeleton edges.')
    links=edges[:,1:]
    parent=np.full(graph.n_points,-1,dtype=int)
    if len(np.unique(links[:,0]))!=len(links):raise ValueError('Multiple skeleton parents.')
    parent[links[:,0]]=links[:,1]
    rooted=np.asarray(graph.point_data['is_root'],dtype=bool)
    state=np.zeros(graph.n_points,dtype=np.uint8)
    keep=np.zeros(graph.n_points,dtype=bool)
    for start in range(graph.n_points):
        node=start;chain=[]
        while node>=0 and state[node]==0:
            state[node]=1;chain.append(node);node=int(parent[node])
        if node>=0 and state[node]==1:raise ValueError('Skeleton contains a cycle.')
        valid=bool(keep[node]) if node>=0 else bool(rooted[chain[-1]])
        for item in chain:keep[item]=valid;state[item]=2
    seeds=int(report['seed_count'])
    excluded=np.flatnonzero(~keep[:seeds])
    retained=int(np.count_nonzero(keep[:seeds]))
    if not retained:raise ValueError('No rooted contacts remain; there is no support forest to slice.')
    ids=np.flatnonzero(keep)
    remap=np.full(graph.n_points,-1,dtype=int);remap[ids]=np.arange(len(ids))
    edge_mask=keep[links].all(axis=1)
    copied=pv.PolyData(graph.points[ids],lines=np.column_stack((np.full(edge_mask.sum(),2),remap[links[edge_mask]])).ravel())
    for name,array in graph.point_data.items():copied.point_data[name]=np.asarray(array)[ids]
    for name,array in graph.cell_data.items():copied.cell_data[name]=np.asarray(array)[edge_mask]
    for name,array in graph.field_data.items():copied.field_data[name]=array
    copied.point_data['source_node_id']=ids
    result=dict(report,original_seed_count=seeds,original_unrooted_contacts=report['unrooted_contacts'],
        excluded_contact_ids=excluded.tolist(),excluded_unrooted_contacts=len(excluded),
        excluded_segments=int((~edge_mask).sum()),seed_count=retained,rooted_contacts=retained,
        unrooted_contacts=0,stalled_fronts=0,nodes=copied.n_points,edges=copied.n_lines,
        ignored_contacts_remain_unsupported=True)
    print(f'WARNING: Omitting {len(excluded)} unrooted contact(s) and {int((~edge_mask).sum())} unfinished segment(s); '
          f'generating supports for {retained}/{seeds} contacts. The omitted region remains unsupported.',flush=True)
    return copied,result


def sweep_rows(skeleton,seed_count,base_radius=1.2,tip_radius=2.5,maximum_radius=2.5,direction=(0,1,0),allow_stalled=False,maximum_contact_radius=None,
               plate=None,platform_base_radius_factor=0.,platform_base_height=.6,branch_sizing='preserved',branch_diameter_angle=5.,planar_platform_bases=False,maximum_platform_radius=None):
    points=np.asarray(skeleton.points,dtype=float)
    edges=skeleton.lines.reshape(-1,3)
    if not len(edges) or np.any(edges[:,0]!=2):raise ValueError('Expected two-point directed skeleton edges.')
    edges=edges[:,1:];upper,lower=edges[:,0],edges[:,1]
    if not 0<seed_count<=len(points) or min(base_radius,tip_radius,maximum_radius)<=0:
        raise ValueError('Invalid skeleton contact count or radii.')
    if (not np.isfinite([platform_base_radius_factor,platform_base_height]).all()
            or platform_base_radius_factor<0 or platform_base_height<=0):
        raise ValueError('Invalid platform base settings.')
    out_degree=np.bincount(upper,minlength=len(points))
    if np.any(out_degree>1):raise ValueError('Skeleton nodes must have at most one lower parent.')
    height=points@np.asarray(direction,dtype=float)
    if np.any(height[upper]<=height[lower]+1e-8):raise ValueError('Imported skeleton must descend build height.')
    if branch_sizing not in ('preserved','bambu'):
        raise ValueError('Unknown branch sizing mode.')
    load=np.zeros(len(points));load[:seed_count]=1
    contact_top=np.full(len(points),-np.inf);contact_top[:seed_count]=height[:seed_count]
    parent=np.full(len(points),-1,dtype=int);parent[upper]=lower
    for node in np.argsort(height)[::-1]:
        if parent[node]>=0:
            load[parent[node]]+=load[node]
            contact_top[parent[node]]=max(contact_top[parent[node]],contact_top[node])
    roots=np.flatnonzero((parent<0)&(load>0))
    if (not allow_stalled and np.any(parent[:seed_count]<0)) or int(load[roots].sum())!=seed_count:
        raise ValueError('Skeleton contacts do not all reach terminal roots.')
    stalled=np.asarray(skeleton.point_data.get('is_stalled',np.asarray(skeleton.point_data.get('node_kind',np.zeros(len(points))))==3),dtype=bool)
    if allow_stalled:
        # Explicit root flags disambiguate terminal junctions and isolated seeds.
        accepted=np.asarray(skeleton.point_data.get('is_root',~stalled),dtype=bool)
        contact_roots=roots[accepted[roots]&~stalled[roots]&(parent[roots]<0)]
    else:contact_roots=roots
    platform_radius=maximum_radius if maximum_platform_radius is None else maximum_platform_radius
    if not np.isfinite(platform_radius) or platform_radius<base_radius:
        raise ValueError('Platform trunk radius cap must be finite and at least the minimum branch radius.')
    if maximum_platform_radius is not None and plate is None:
        raise ValueError('A platform trunk cap requires an explicit build plate.')
    root_of=np.arange(len(points));length_to_root=np.zeros(len(points))
    for node in np.argsort(height):
        if parent[node]>=0:
            root_of[node]=root_of[parent[node]]
            length_to_root[node]=length_to_root[parent[node]]+np.linalg.norm(points[node]-points[parent[node]])
    platform_roots=(contact_roots[np.abs(height[contact_roots]-float(plate))<=1e-6]
                    if plate is not None else np.empty(0,dtype=int))
    platform_connected=np.isin(root_of,platform_roots)
    radius_cap=np.where(platform_connected,platform_radius,maximum_radius)
    body_radii=np.minimum(radius_cap,base_radius*np.sqrt(np.maximum(load,1)))
    if branch_sizing=='bambu':
        from .bambu_sizing import branch_radius
        # Use the existing Bambu vertical-drop rule as an additional lower
        # bound, retaining load thickening and the user's trunk diameter cap.
        drop=np.where(np.isfinite(contact_top),np.maximum(0,contact_top-height),0.)
        widening=branch_radius(drop,diameter=2*base_radius,angle=branch_diameter_angle)
        body_radii=np.minimum(radius_cap,np.maximum(body_radii,widening))
    radii=body_radii.copy()
    radii[:seed_count]=tip_radius;radii[contact_roots]=tip_radius
    adaptive_count=0
    if 'root_contact_radius' in skeleton.point_data:
        saved=np.asarray(skeleton.point_data['root_contact_radius'],dtype=float)
        if not np.isfinite(saved).all() or np.any(saved<0):raise ValueError('Invalid saved root contact radii.')
        adaptive=saved[root_of]>0
        adaptive_count=int(np.count_nonzero(saved[roots]>0))
        # Spread the radius transition over the last 3mm of the existing path;
        # changing radii does not move the validated skeleton or its endpoints.
        lengths=np.asarray(skeleton.point_data.get('root_foot_transition_mm',np.zeros(len(points))),dtype=float)
        if not np.isfinite(lengths).all() or np.any(lengths<0):raise ValueError('Invalid root transition lengths.')
        lengths=np.maximum(3.,lengths[root_of])
        weight=np.clip(1-length_to_root/lengths,0,1);weight=weight*weight*(3-2*weight)
        weight[~adaptive]=0
        contact_limit=maximum_radius if maximum_contact_radius is None else maximum_contact_radius
        if not np.isfinite(contact_limit) or contact_limit<=0:raise ValueError('Invalid maximum contact radius.')
        contact=np.minimum(saved[root_of],contact_limit)
        radii=radii*(1-weight)+contact*weight
        radii[:seed_count]=tip_radius
    if maximum_platform_radius is not None:
        # A plate endpoint is a trunk, even when the separate flange is off.
        # Do not taper the newly enlarged stem back to a model-contact tip.
        radii[platform_roots]=body_radii[platform_roots]
    tip_profile=None
    if 'adaptive_tip_radius' in skeleton.point_data:
        from .adaptive_tips import apply_tip_profile
        radii,tip_profile=apply_tip_profile(skeleton,seed_count,radii,body_radii,radius_cap,parent,tip_radius)
    terminal=(upper<seed_count).astype(float)
    rows=np.column_stack((points[lower],points[upper],radii[lower],radii[upper],terminal))
    # A platform root is identified from the actual routing endpoint, not from
    # the field/layer value. The added vertical primitive meets the ordinary
    # trunk at the root and is clipped by TaperedTree's build-plate plane,
    # leaving an exactly horizontal circular contact face.
    platform_roots=np.empty(0,dtype=int)
    base_count=0
    if plate is not None:
        platform_roots=contact_roots[np.abs(height[contact_roots]-float(plate))<=1e-6]
        if platform_base_radius_factor>0 and len(platform_roots):
            # A plate root is a trunk endpoint, not a model-contact tip. Use
            # its load-based body radius for both the foot and its attachment.
            radii[platform_roots]=body_radii[platform_roots]
            rows=np.column_stack((points[lower],points[upper],radii[lower],radii[upper],terminal))
            enlarged=body_radii[platform_roots]*max(1.,platform_base_radius_factor)
            upper_points=points[platform_roots]+np.asarray(direction,dtype=float)[None,:]*platform_base_height
            base_rows=np.column_stack((points[platform_roots],upper_points,enlarged,radii[platform_roots],np.full(len(platform_roots),3 if planar_platform_bases else 0)))
            rows=np.vstack((rows,base_rows));base_count=len(base_rows)
    # An asymmetric root lobe is a short tapered primitive added beside the
    # unchanged root.  It is deliberately non-terminal: its rounded end is
    # clipped by the ordinary model-clearance field, not by a planar contact cap.
    lobe_count=0
    if 'root_foot_lobe_radius' in skeleton.point_data:
        lobe_radius=np.asarray(skeleton.point_data['root_foot_lobe_radius'],dtype=float)
        lobe_center=np.asarray(skeleton.point_data.get('root_foot_lobe_center',np.zeros_like(points)),dtype=float)
        lobe_nodes=contact_roots[lobe_radius[contact_roots]>1e-8]
        if len(lobe_nodes):
            extra=np.column_stack((lobe_center[lobe_nodes],points[lobe_nodes],lobe_radius[lobe_nodes],radii[lobe_nodes],np.zeros(len(lobe_nodes))))
            rows=np.vstack((rows,extra));lobe_count=len(lobe_nodes)
    delta=rows[:,:3]-rows[:,3:6]
    normals=delta/np.linalg.norm(delta,axis=1)[:,None]
    model_roots=np.setdiff1d(contact_roots,platform_roots,assume_unique=True)
    sizing=dict(seed_count=seed_count,root_count=len(contact_roots),
        stalled_terminal_count=len(roots)-len(contact_roots),
        isolated_contacts_without_segments=int(np.count_nonzero(parent[:seed_count]<0)),
        segment_count=len(rows),asymmetric_root_lobe_count=lobe_count,maximum_body_radius_mm=float(np.max(radius_cap)),minimum_body_radius_mm=base_radius,
        maximum_model_body_radius_mm=maximum_radius,maximum_platform_body_radius_mm=float(platform_radius),
        contact_radius_mm=tip_radius,adaptive_root_contacts=adaptive_count,
        maximum_contact_radius_mm=maximum_radius if maximum_contact_radius is None else maximum_contact_radius,
        root_transition_length_mm=float(np.max(lengths)) if adaptive_count else 0.,
        platform_root_count=len(platform_roots),platform_root_nodes=platform_roots.tolist(),
        platform_base_count=base_count,platform_base_radius_factor=platform_base_radius_factor,
        platform_base_height_mm=platform_base_height if base_count else 0.,
        branch_sizing=branch_sizing,branch_diameter_angle_degrees=branch_diameter_angle if branch_sizing=='bambu' else 0.,
        radius_rule='body radius=min(maximum,max(base*sqrt(descendant contacts),optional Bambu vertical-drop widening)); upper tips fixed; optional model-root radii with a default 3mm or longer screened-foot transition')
    if tip_profile is not None:sizing['adaptive_tip_profile']=tip_profile
    return rows,normals,points[model_roots],sizing
