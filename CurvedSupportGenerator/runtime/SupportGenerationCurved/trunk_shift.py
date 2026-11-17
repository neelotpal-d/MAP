"""Optional lower-trunk displacement fading from a moved root to a fixed fork."""
import time
import numpy as np
from .stalled_recovery import valid_connections


def shift_trunks(graph,field,distance,landing_check,query,normals,direction=(0,1,0),plate=0.,
                 maximum_shift=2.,gap=.3,angle=45.,body_radius=2.5,maximum_roots=8):
    if not np.isfinite(maximum_shift) or maximum_shift<=0 or maximum_roots<1:
        raise ValueError('Require positive finite shift and a positive root budget.')
    if landing_check is None:raise ValueError('Trunk shifting requires model landing checks.')
    started=time.perf_counter();out=graph.copy(deep=True);xyz=np.asarray(out.points).copy()
    edges=graph.lines.reshape(-1,3)[:,1:];children={}
    for child,parent in edges:children.setdefault(int(parent),[]).append(int(child))
    roots=np.flatnonzero(np.asarray(graph.point_data.get('is_root',np.zeros(len(xyz))),dtype=bool)
        &~np.asarray(graph.point_data.get('is_stalled',np.zeros(len(xyz))),dtype=bool))
    trunks=[]
    for root in roots:
        path=[int(root)]
        while len(children.get(path[-1],[]))==1:
            child=children[path[-1]][0]
            if child in path:raise ValueError('Cycle in saved trunk.')
            path.append(child)
        # Restrict to shared trunks ending at a real fork, not single twigs.
        if len(path)<3 or len(children.get(path[-1],[]))<2:continue
        s=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz[path],axis=0),axis=1))]
        if s[-1]<3.:continue
        trunks.append((float(s[-1]),path,s))
    records=[]
    for total,path,s in sorted(trunks,reverse=True)[:maximum_roots]:
        root=path[0];original=xyz[path].copy();t=s/total
        weight=1-t**3*(10+t*(-15+6*t))
        # Endpoint+midpoint samples see nearby walls between saved nodes.
        sample=np.vstack((original,(original[:-1]+original[1:])*.5))
        weights=np.r_[weight,(weight[:-1]+weight[1:])*.5]
        arclength=np.r_[s,(s[:-1]+s[1:])*.5]
        def wall_score(points):
            _,face=query(points);n=np.asarray(normals)[face]
            _,g=field(points);g=np.asarray(g)
            unit=g/np.maximum(np.linalg.norm(g,axis=1)[:,None],1e-15)
            side=(np.abs(np.sum(n*unit,axis=1))<.7)&(arclength>max(.6,2*gap))&(weights>1e-5)
            d=np.asarray(distance(points))
            deficit=np.maximum(body_radius+gap-d,0.)
            score=float(np.sum(deficit*weights*side)/max(float(np.sum(weights)),1e-15))
            return score,side,n,d
        before,side,n,d=wall_score(sample)
        record=dict(root_node=int(root),fixed_branch_node=int(path[-1]),trunk_length_mm=total,
                    before_wall_penalty=before,shifted=False,trials=[])
        if not side.any() or before<=1e-6:
            record['reason']='no_nearby_side_wall';records.append(record);continue
        closest,face=query(original[:1]);base_normal=np.asarray(normals[face[0]],dtype=float)
        base_normal/=max(np.linalg.norm(base_normal),1e-15)
        deficit=np.maximum(body_radius+gap-d,0.)*weights*side
        away=np.sum(n*deficit[:,None],axis=0)
        away-=float(away@base_normal)*base_normal
        length=np.linalg.norm(away)
        if length<1e-8:
            record['reason']='opposing_walls_or_no_tangent_escape';records.append(record);continue
        offset=away/length*min(maximum_shift,float(np.max(np.maximum(body_radius+gap-d[side],0.))))
        for fraction in (1.,.75,.5,.25):
            trial=dict(fraction=fraction)
            base=original[0]+fraction*offset
            for _ in range(2):
                surface,face=query(base[None,:]);base=surface[0]+gap*np.asarray(normals[face[0]])
            displacement=base-original[0]
            if np.linalg.norm(displacement)>maximum_shift+1e-6:
                trial['reason']='root_projection_exceeds_shift';record['trials'].append(trial);continue
            candidate=original+weight[:,None]*displacement
            candidate[-1]=original[-1]
            if not landing_check(candidate[:1])[0]:
                trial['reason']='landing_rejected';record['trials'].append(trial);continue
            detail={}
            ok,_=valid_connections(candidate[1:],candidate[:-1],field,distance,gap,angle,plate,
                                   np.asarray(direction,dtype=float),diagnostics=detail)
            if not ok.all():
                trial.update(reason='edge_constraints',**detail);record['trials'].append(trial);continue
            probes=np.vstack((candidate,(candidate[:-1]+candidate[1:])*.5))
            after,_,_,_=wall_score(probes)
            if after>=before-1e-4:
                trial['reason']='no_wall_clearance_improvement';record['trials'].append(trial);continue
            xyz[path]=candidate
            record.update(shifted=True,after_wall_penalty=after,root_displacement_mm=float(np.linalg.norm(displacement)),
                old_root=original[0].tolist(),new_root=base.tolist(),reason='accepted')
            break
        if not record['shifted']:record['reason']='no_valid_gradual_shift'
        records.append(record)
    out.points=xyz
    achieved=np.asarray(out.point_data.get('landing_achieved_up_dot',np.full(len(xyz),np.nan))).copy()
    fallback=np.asarray(out.point_data.get('landing_is_fallback',np.zeros(len(xyz),dtype=np.uint8))).copy()
    for record in records:
        if record['shifted']:
            node=record['root_node'];achieved[node]=landing_check.alignment(xyz[node]);fallback[node]=0
    out.point_data['landing_achieved_up_dot']=achieved
    out.point_data['landing_is_fallback']=fallback
    key='routing_rank' if getattr(field,'direction_only',False) else 'scalar'
    out.point_data[key]=np.asarray(field(xyz)[0])
    return out,dict(maximum_shift_mm=maximum_shift,shifted_trunks=sum(r['shifted'] for r in records),
        examined_trunks=len(records),seconds=time.perf_counter()-started,records=records,
        displacement_profile='quintic decay from relocated model root to fixed first fork',
        limitation='Sampled centreline/landing checks using routing guide; final relaxed layers and swept-volume clearance still require checking.')
