"""Experimental conservative agglomeration of model-rooted contact struts.

This is a local forest planner, not a port of Bambu's complete tree planner.
Only uniform gaps and rounded contacts are supported initially.
"""
import time
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


def centerlines(segments):
    rows=np.asarray(segments,dtype=float).reshape(-1,9)
    points=rows[:,:6].reshape(-1,3)
    lines=np.column_stack((np.full(len(rows),2),np.arange(0,len(points),2),np.arange(1,len(points),2)))
    result=pv.PolyData(points,lines=lines.ravel())
    result.cell_data['kind']=rows[:,8].astype(np.int32)
    result.cell_data['radius_a']=rows[:,6]
    result.cell_data['radius_b']=rows[:,7]
    return result


def plan(geometry,field,approach=3.,neighbor_distance=12.,maximum_radius=2.5,maximum_trials=5000):
    """Return a forest with explicit junctions and individually retained tips.

    Body edges are tested with an enclosing radius plus sampling margin, and
    must rise at <=45 degrees from the build direction with increasing scalar.
    Failed/short input struts are retained verbatim; contacts are never dropped.
    """
    started=time.perf_counter()
    if geometry.side_gap!=geometry.contact_gap or geometry.terminal_cap!='rounded':
        raise ValueError('Merged model trees currently require equal gaps and --terminal-cap rounded.')
    if not np.all(geometry.model_bridge):
        raise ValueError('Initial merged planner supports bridge-only sources; retain struts for mixed platform/model sources.')
    direction=np.asarray(geometry.direction,dtype=float)
    scalar=lambda p: np.asarray(field.evaluate_scalars(np.asarray(p)),dtype=float)
    segments=[];normals=[];roots=[];groups=[];fallback=0;trials=0;merges=0
    rejected=dict(angle=0,clearance=0,scalar=0,length=0)

    def edge(a,b,ra,rb,kind=0,normal=None):
        return ([*a,*b,float(ra),float(rb),kind],
                np.asarray(normal if normal is not None else (a-b)/np.linalg.norm(b-a)))

    def body_ok(a,b,radius):
        delta=b-a;length=float(np.linalg.norm(delta));rise=float(delta@direction)
        if rise<=1e-5 or np.linalg.norm(delta-rise*direction)>rise+1e-8:
            rejected['angle']+=1;return False
        count=max(3,int(np.ceil(length/.5))+1)
        points=a+np.linspace(0,1,count)[:,None]*delta
        # Distance is 1-Lipschitz: half the sample interval bounds the
        # clearance loss between samples for this enclosing body radius.
        margin=length/(count-1)/2
        if np.min(geometry.distance(points))<geometry.side_gap+radius+margin:
            rejected['clearance']+=1;return False
        values=scalar(points)
        if not np.isfinite(values).all() or np.any(np.diff(values)<=1e-8):
            rejected['scalar']+=1;return False
        return True

    lowers=geometry.a.copy()
    if hasattr(field,'evaluate_scalars_and_gradients'):
        _,gradients=field.evaluate_scalars_and_gradients(lowers)
        gradients=np.asarray(gradients,dtype=float)
        gradients=gradients/np.maximum(np.linalg.norm(gradients,axis=1)[:,None],1e-15)
    else:
        gradients=geometry.edge_direction.copy()
    for i,(lower,upper) in enumerate(zip(geometry.a,geometry.b)):
        length=np.linalg.norm(upper-lower)
        radius=min(float(geometry.ra[i]),maximum_radius)
        head=upper-approach*geometry.edge_direction[i]
        root=lower+approach*gradients[i]
        # Do not require the old straight connector to fit: routing away from
        # that connector is precisely why merging can help. Newly proposed
        # connectors are checked below; unmerged inputs stay unchanged.
        if length<=2*approach+2:
            segments.append(geometry.segments[i].tolist());normals.append(geometry.tip_normals[i]);roots.append(lower)
            fallback+=1;continue
        tip=edge(head,upper,radius,geometry.rb[i],1,geometry.tip_normals[i])
        groups.append(dict(head=head,root=root,lower=lower,radius=radius,count=1,
                           edges=[tip],root_tip_radius=float(geometry.rb[i]),original=i))

    while len(groups)>1 and trials<maximum_trials:
        heads=np.asarray([g['head'] for g in groups])
        pairs=list(cKDTree(heads).query_pairs(neighbor_distance))
        pairs.sort(key=lambda pair:(np.linalg.norm(heads[pair[0]]-heads[pair[1]]),pair))
        accepted=False
        for ia,ib in pairs:
            a,b=groups[ia],groups[ib];trials+=1
            if trials>maximum_trials:break
            radius=min(maximum_radius,float(geometry.ra.min())*np.sqrt(a['count']+b['count']))
            middle=(a['head']+b['head'])/2
            horizontal=[np.linalg.norm(g['head']-middle-((g['head']-middle)@direction)*direction) for g in (a,b)]
            height=min(a['head']@direction,b['head']@direction)-max(max(horizontal),.75)-.05
            junction=middle+(height-middle@direction)*direction
            candidates=sorted((a,b),key=lambda g:np.linalg.norm(junction-g['root']))
            chosen=None
            old_length=sum(np.linalg.norm(g['head']-g['root']) for g in (a,b))
            for root_group in candidates:
                new_length=sum(np.linalg.norm(g['head']-junction) for g in (a,b))+np.linalg.norm(junction-root_group['root'])
                if new_length>=old_length-.2:
                    rejected['length']+=1;continue
                if (body_ok(root_group['root'],junction,radius)
                    and body_ok(junction,a['head'],radius) and body_ok(junction,b['head'],radius)):
                    chosen=root_group;break
            if chosen is None:continue
            edges=a['edges']+b['edges']+[
                edge(junction,a['head'],radius,a['radius']),edge(junction,b['head'],radius,b['radius'])]
            merged=dict(head=junction,root=chosen['root'],lower=chosen['lower'],radius=radius,
                        count=a['count']+b['count'],edges=edges,root_tip_radius=chosen['root_tip_radius'])
            groups=[g for j,g in enumerate(groups) if j not in (ia,ib)]+[merged]
            merges+=1;accepted=True;break
        if not accepted:break

    for g in groups:
        if g['count']==1:
            i=g['original']
            segments.append(geometry.segments[i].tolist());normals.append(geometry.tip_normals[i]);roots.append(g['lower'])
            fallback+=1
            continue
        edges=g['edges']+[edge(g['root'],g['head'],g['radius'],g['radius']),
                           edge(g['lower'],g['root'],g['root_tip_radius'],g['radius'])]
        for row,normal in edges:segments.append(row);normals.append(normal)
        roots.append(g['lower'])
    report=dict(mode='merged_model_forest',input_contacts=len(geometry.a),merges=merges,
        retained_roots=len(roots),fallback_struts=fallback,merged_group_sizes=[g['count'] for g in groups],
        trials=trials,rejections=rejected,seconds=time.perf_counter()-started,
        limitations=['Local greedy merging, not the complete Bambu planner.',
            'Uniform gaps only; offset contact patches and initial-layer overlap checks are not implemented.',
            'Retained fallback struts have the previous geometry and are not certified by the body-edge checks.',
            'Envelope coverage and contact-approach printability require downstream checks.'])
    return np.asarray(segments),np.asarray(normals),np.asarray(roots),report
