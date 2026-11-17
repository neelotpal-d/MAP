"""Bounded alternative contact-transition paths; never changes the fixed neck.

The search retains several spatially distinct paths so a vertical preference
cannot discard every early detour. Callers commit only a complete attachment.
"""
import time
import numpy as np
from SupportGenerationCurved.progressive_skeleton import staged_axis,cone_direction


def transition_paths(start,age,contact_down,field,distance,direction,gap,angle,
                     fixed_length,transition_length,deadline,step=.3,width=32,landing_check=None,
                     previous_direction=None,previous_length=None,min_bend_radius=5.):
    end=fixed_length+transition_length; maximum=np.deg2rad(angle)
    start=np.asarray(start,float); direction=np.asarray(direction,float)
    # Full histories are short (at most the transition length / step).
    states=[dict(points=[start],age=float(age),previous=previous_direction,previous_length=previous_length,score=0.)]
    completed=[]; landings=[]
    # Keep a compact rejection tally.  A protected-stage stall used to report
    # only how many trial edges survived, which made it impossible to tell
    # whether the cause was the fixed-stage geometry, the curvature bound, the
    # field, or model clearance.  These counts are diagnostic only: they do
    # not alter the proposal set or any acceptance inequality.
    stats=dict(expanded_edges=0,accepted_edges=0,levels=0,budget_stopped=False,
        proposed_moves=0,rejected_non_descending_moves=0,rejected_bend_moves=0,
        rejected_invalid_field=0,rejected_field_cone=0,rejected_scalar_descent=0,
        rejected_native_scalar=0,rejected_clearance=0)
    phi=np.arange(16)*2*np.pi/16
    for level in range(int(np.ceil(max(0.,end-age)/step))+2):
        if time.perf_counter()>=deadline:stats['budget_stopped']=True;break
        if not states:break
        stats['levels']+=1
        positions=np.asarray([s['points'][-1] for s in states]);_,g=field(positions)
        edges=[]
        for k,s in enumerate(states):
            if s['age']>=end-1e-8:completed.append(s);continue
            norm=np.linalg.norm(g[k])
            if not np.isfinite(norm) or norm<1e-8:continue
            local=-g[k]/norm; progress=np.clip((s['age']-fixed_length)/max(transition_length,1e-15),0,1)
            axis=staged_axis(contact_down,local,progress); allowed=maximum*progress**2*(3-2*progress)
            length=min(step,end-s['age'])
            if s['age']<fixed_length-1e-9:length=min(length,fixed_length-s['age'])
            tangent=np.cross(axis,[1.,0.,0.])
            if np.linalg.norm(tangent)<1e-8:tangent=np.cross(axis,[0.,0.,1.])
            tangent/=np.linalg.norm(tangent); other=np.cross(axis,tangent)
            ring=np.cos(phi)[:,None]*tangent+np.sin(phi)[:,None]*other
            moves=[cone_direction(-direction,axis,allowed),axis]
            if allowed>1e-8:
                moves.extend(np.cos(allowed*f)*axis+np.sin(allowed*f)*ring for f in (.5,.85,.995))
            if s['previous'] is not None and min_bend_radius>0:
                previous=s['previous'];turn=2*np.arcsin(min(1.,(length+s['previous_length'])/(4*min_bend_radius)))
                tangent0=np.cross(previous,[1.,0.,0.])
                if np.linalg.norm(tangent0)<1e-8:tangent0=np.cross(previous,[0.,0.,1.])
                tangent0/=np.linalg.norm(tangent0);other0=np.cross(previous,tangent0)
                ring0=np.cos(phi)[:,None]*tangent0+np.sin(phi)[:,None]*other0
                for f in (.5,.995):
                    moves.extend([cone_direction(v,axis,allowed) for v in np.cos(turn*f)*previous+np.sin(turn*f)*ring0])
                moves.append(cone_direction(previous,axis,allowed))
            moves=np.unique(np.vstack(moves),axis=0)
            stats['proposed_moves']+=len(moves)
            descending=(moves@direction)<-1e-8
            stats['rejected_non_descending_moves']+=int((~descending).sum())
            moves=moves[descending]
            if s['previous'] is not None and min_bend_radius>0:
                bend_ok=moves@s['previous']>=np.cos(turn)-1e-10
                stats['rejected_bend_moves']+=int((~bend_ok).sum())
                moves=moves[bend_ok]
            for move in moves:edges.append((k,move,length))
        if not edges:break
        which=np.asarray([v[0] for v in edges]);moves=np.asarray([v[1] for v in edges]);lengths=np.asarray([v[2] for v in edges])
        a=positions[which];b=a+lengths[:,None]*moves
        probes=a[:,None]+np.linspace(0,1,17)[None,:,None]*(b-a)[:,None]
        rank,gradient=field(probes.reshape(-1,3)); rank=np.asarray(rank).reshape(-1,17);gradient=np.asarray(gradient).reshape(-1,17,3)
        norm=np.linalg.norm(gradient,axis=2); cosine=np.sum(-gradient*moves[:,None],axis=2)/np.maximum(norm,1e-15)
        finite=np.isfinite(gradient).all(axis=(1,2))&(norm.min(axis=1)>1e-8)&np.isfinite(rank).all(axis=1)
        cone=cosine.min(axis=1)>=np.cos(maximum)-1e-5
        scalar=np.all(np.diff(rank,axis=1)<-1e-9,axis=1)
        native=field.native_descent_mask(probes) if hasattr(field,'native_descent_mask') else np.ones(len(edges),bool)
        # Categories are deliberately non-exclusive: for example, an edge
        # inside the model may also have poor scalar descent.  That is more
        # useful than arbitrarily assigning a single root cause.
        stats['rejected_invalid_field']+=int((~finite).sum())
        stats['rejected_field_cone']+=int((~cone).sum())
        stats['rejected_scalar_descent']+=int((~scalar).sum())
        stats['rejected_native_scalar']+=int((~native).sum())
        ok=finite&cone&scalar&native
        ids=np.flatnonzero(ok);clearance=np.full(len(edges),-np.inf)
        if len(ids):
            ds=distance(probes[ids].reshape(-1,3)).reshape(-1,17);clearance[ids]=ds.min(axis=1)
            crossing=np.flatnonzero((ds<gap).any(axis=1))
            if landing_check is not None and len(crossing):
                cross_ids=ids[crossing];first=np.argmax(ds[crossing]<gap,axis=1)
                left=np.maximum(0,first-1)/16;right=first/16
                for _ in range(24):
                    mid=(left+right)/2;outside=distance(a[cross_ids]+mid[:,None]*(b[cross_ids]-a[cross_ids]))>=gap
                    left=np.where(outside,mid,left);right=np.where(outside,right,mid)
                contact_points=a[cross_ids]+left[:,None]*(b[cross_ids]-a[cross_ids])
                accepted=landing_check(contact_points)
                for index,point in zip(cross_ids[accepted],contact_points[accepted]):
                    if np.linalg.norm(point-a[index])<1e-7:continue
                    landings.append(dict(points=np.asarray(states[which[index]]['points']+[point]),
                        kind='model',score=float(np.linalg.norm(point-start))))
        clearance_ok=clearance>=gap+2e-5
        stats['rejected_clearance']+=int((~clearance_ok).sum())
        ok&=clearance_ok
        stats['expanded_edges']+=len(edges); stats['accepted_edges']+=int(ok.sum())
        survivors=[]
        for i in np.flatnonzero(ok):
            s=states[which[i]];previous=s['previous'];turn=0. if previous is None else 1-float(moves[i]@previous)
            # Clearance takes priority near an obstacle; down remains a soft
            # preference. Accumulated turn discourages rapidly changing headings.
            score=float(min(clearance[i]-gap,2.)+.15*((start-b[i])@direction)-.15*turn)
            survivors.append(dict(points=s['points']+[b[i]],age=s['age']+lengths[i],previous=moves[i],previous_length=lengths[i],score=score))
        survivors.sort(key=lambda s:-s['score']);states=[];seen=set();sectors={}
        for s in survivors:
            offset=s['points'][-1]-start
            # Heading diversity matters before the positions separate. A coarse
            # position-only bin would erase every gradual early detour.
            key=(*np.round(offset/.025).astype(int),*np.round(s['previous']/.04).astype(int))
            if key in seen:continue
            lateral=offset-(offset@direction)*direction
            u=np.cross(direction,[1.,0.,0.])
            if np.linalg.norm(u)<1e-8:u=np.cross(direction,[0.,0.,1.])
            u/=np.linalg.norm(u);v=np.cross(direction,u)
            sector=int(np.floor((np.arctan2(lateral@v,lateral@u)+np.pi)*12/(2*np.pi)))%12
            if sectors.get(sector,0)>=3:continue
            sectors[sector]=sectors.get(sector,0)+1
            seen.add(key);states.append(s)
            if len(states)>=width:break
    completed.extend(s for s in states if s['age']>=end-1e-8)
    unique={}
    for s in sorted(completed,key=lambda s:-s['score']):unique.setdefault(np.asarray(s['points']).tobytes(),np.asarray(s['points']))
    solutions=[dict(points=p,kind='continue') for p in unique.values()]
    distinct={}
    for s in sorted(landings,key=lambda s:s['score']):distinct.setdefault(tuple(np.round(s['points'][-1]/.1).astype(int)),s)
    solutions.extend(list(distinct.values())[:16])
    stats['valid_model_landings']=len(landings)
    return solutions,stats


def checked_transition(path,age,contact_down,field,distance,direction,gap,angle,
                       fixed_length,transition_length,plate=0.,previous_direction=None,previous_length=None,min_bend_radius=5.):
    """Independent dense acceptance after the bounded proposal search."""
    p=np.asarray(path);delta=np.diff(p,axis=0);length=np.linalg.norm(delta,axis=1)
    if not len(delta) or np.any(length<1e-7):return False,{}
    probes=p[:-1,None]+np.linspace(0,1,65)[None,:,None]*delta[:,None]
    rank,g=field(probes.reshape(-1,3));rank=np.asarray(rank).reshape(-1,65);g=np.asarray(g).reshape(-1,65,3)
    norm=np.linalg.norm(g,axis=2);unit=delta/length[:,None]
    cosine=np.sum(-g*unit[:,None],axis=2)/np.maximum(norm,1e-15)
    age=age+np.r_[0.,np.cumsum(length)[:-1]]
    progress=np.clip((age-fixed_length)/max(transition_length,1e-15),0,1)
    axes=np.asarray([staged_axis(contact_down,-v/max(np.linalg.norm(v),1e-15),s) for v,s in zip(g[:,0],progress)])
    allowed=np.deg2rad(angle)*progress**2*(3-2*progress)
    clearance=distance(probes.reshape(-1,3)).reshape(-1,65)
    flags=dict(finite=bool(np.isfinite(rank).all() and np.isfinite(g).all() and norm.min()>1e-8),
        field_cone=bool(cosine.min()>=np.cos(np.deg2rad(angle))-1e-5),
        contact_stage=bool(np.all(np.sum(axes*unit,axis=1)>=np.cos(allowed)-1e-8)),
        scalar_descent=bool(np.all(np.diff(rank,axis=1)<-1e-9)),
        height_descent=bool(np.all(delta@direction<-1e-8)),
        clearance=bool(clearance.min()>=gap-2e-6),platform=bool((probes@direction).min()>=plate-1e-6))
    if hasattr(field,'native_descent_mask'):flags['native_scalar_descent']=bool(field.native_descent_mask(probes).all())
    heading=unit if previous_direction is None else np.vstack((previous_direction,unit))
    segment_length=length if previous_direction is None else np.r_[previous_length,length]
    if len(heading)>1 and min_bend_radius>0:
        sine=np.sqrt(np.maximum(0.,(1-np.sum(heading[:-1]*heading[1:],axis=1))/2))
        discrete_radius=(segment_length[:-1]+segment_length[1:])/np.maximum(4*sine,1e-15)
        flags['discrete_bend_radius']=bool(discrete_radius.min()>=min_bend_radius-1e-6)
    else:discrete_radius=np.array([np.inf])
    return all(flags.values()),dict(checks=flags,minimum_clearance_mm=float(clearance.min()),
        minimum_discrete_bend_radius_mm=float(discrete_radius.min()),
        maximum_field_angle_degrees=float(np.rad2deg(np.arccos(np.clip(cosine.min(),-1,1)))))
