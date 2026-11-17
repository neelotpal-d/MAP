"""Bounded nearby-landing recovery for failed skeleton components only."""
import copy
import time
import numpy as np
import pyvista as pv
from SupportGenerationCurved.landing_checks import tangent_ring
from SupportGenerationCurved.progressive_skeleton import cone_direction, cone_directions


def diverse_trials(trials,budget,anchor_scores):
    """Cover rewind anchors and landing patches before repeating either."""
    if not trials:return []
    anchors=sorted({t[4] for t in trials},key=lambda a:anchor_scores[a],reverse=True)
    # Include both earlier clear-body anchors and near-contact alternatives.
    if len(anchors)>8:anchors=[anchors[i] for i in np.linspace(0,len(anchors)-1,8,dtype=int)]
    candidates=list(dict.fromkeys(t[5] for t in sorted(trials)))[:8]
    lookup={(t[4],t[5]):t for t in trials};selected=[];seen=set()
    for shift in range(max(len(anchors),len(candidates))):
        for i,anchor in enumerate(anchors):
            candidate=candidates[(i+shift)%len(candidates)]
            key=(anchor,candidate)
            if key in lookup and key not in seen:
                selected.append(lookup[key]);seen.add(key)
                if len(selected)>=budget:return selected
    for trial in sorted(trials):
        if (trial[4],trial[5]) not in seen:
            selected.append(trial);seen.add((trial[4],trial[5]))
            if len(selected)>=budget:break
    return selected


def guided_connection(start,target,field,distance,gap,angle,plate,direction,deadline,
                      step=.6,maximum_steps=160,diagnostics=None):
    """Bounded target attraction inside local cones; return only complete paths.

    This is not a global obstacle planner. A failed trial has no graph effects.
    """
    path=[np.asarray(start,dtype=float).copy()];ranks=[]
    def fail(reason):
        if diagnostics is not None:diagnostics.update(reason=reason,completed_steps=len(path)-1)
        return None
    for _ in range(maximum_steps):
        if time.perf_counter()>=deadline:return fail('time_budget')
        a=path[-1];delta=target-a;remaining=np.linalg.norm(delta)
        if remaining<=step:
            b=target.copy()
        else:
            _,g=field(a[None,:]);norm=np.linalg.norm(g[0])
            if not np.isfinite(norm) or norm<1e-8:return fail('invalid_field')
            move=cone_direction(delta/remaining,-g[0]/norm,np.deg2rad(max(0.,angle-2)))
            b=a+step*move
            if np.linalg.norm(target-b)>=remaining-1e-8:return fail('no_target_progress')
        ok,rank=valid_connections(a[None,:],b[None,:],field,distance,gap,angle,plate,direction,diagnostics=diagnostics)
        if not ok[0]:return fail('edge_constraints')
        path.append(b);ranks.append(float(rank[0]))
        if remaining<=step:return np.asarray(path),np.asarray(ranks)
    return fail('step_budget')


def valid_connections(a,b,field,distance,gap,angle,plate,direction,diagnostics=None):
    delta=b-a;length=np.linalg.norm(delta,axis=1)
    probes=a[:,None,:]+np.linspace(0,1,33)[None,:,None]*delta[:,None,:]
    rank,g=field(probes.reshape(-1,3));rank=rank.reshape(-1,33);g=g.reshape(-1,33,3)
    norms=np.linalg.norm(g,axis=2)
    cosine=np.sum(-g*delta[:,None,:],axis=2)/np.maximum(norms*length[:,None],1e-15)
    clearance=distance(probes.reshape(-1,3)).reshape(-1,33)
    ok=((length>1e-7)&np.isfinite(rank).all(axis=1)&np.isfinite(g).all(axis=(1,2))
        &(norms>1e-8).all(axis=1)&(np.diff(rank,axis=1)<-1e-9).all(axis=1)
        &(cosine.min(axis=1)>=np.cos(np.deg2rad(max(0.,angle-1)))-1e-5)
        &(clearance.min(axis=1)>=gap-2e-6)&((probes@direction).min(axis=1)>=plate-1e-6))
    native_ok=field.native_descent_mask(probes) if hasattr(field,'native_descent_mask') else np.ones(len(a),dtype=bool)
    ok &= native_ok
    if diagnostics is not None and not ok.all():
        diagnostics['violations']=[name for name,bad in (
            ('field_cone',cosine.min(axis=1)<np.cos(np.deg2rad(max(0.,angle-1)))-1e-5),
            ('model_clearance',clearance.min(axis=1)<gap-2e-6),
            ('ordering',~(np.diff(rank,axis=1)<-1e-9).all(axis=1)),
            ('native_scalar',~native_ok),
            ('platform',((probes@direction).min(axis=1)<plate-1e-6)),
            ('invalid_field',~np.isfinite(g).all(axis=(1,2))|(norms<=1e-8).any(axis=1))) if np.any(bad)]
    return ok,rank[:,-1]


def platform_connection(start,field,distance,gap,angle,plate,direction,deadline,
                        step=.6,maximum_steps=450,diagnostics=None):
    """Return a fully checked path to any platform point, or no path.

    Unlike a model target, a platform landing has no prescribed XY position.
    Partial attempts are discarded, and the caller's recovery budget applies.
    """
    direction=np.asarray(direction,dtype=float)
    path=[np.asarray(start,dtype=float).copy()];ranks=[]
    # Trial paths can rewind their own tail; the caller's graph is untouched.
    # Quantized visited endpoints prevent repeated detours into the same trap.
    visited=set();rewound=0.;force_cone=False;escapes=0;backtracks=0;branch_trials={}
    def key(p):return tuple(np.round(np.asarray(p)/(step/32)).astype(np.int64))
    visited.add(key(path[0]))
    def finish(reason):
        if diagnostics is not None:
            diagnostics.update(reason=reason,cone_escape_steps=escapes,rewind_steps=backtracks,
                               completed_steps=len(path)-1,rewound_mm=rewound)
        return None
    for _ in range(maximum_steps+24):
        if time.perf_counter()>=deadline:return finish('time_budget')
        if len(path)>maximum_steps:return finish('step_budget')
        a=path[-1];_,g=field(a[None,:]);norm=np.linalg.norm(g[0])
        if not np.isfinite(norm) or norm<1e-8:return finish('invalid_field')
        axis=-g[0]/norm
        moves=[cone_direction(-direction,axis,np.deg2rad(max(0.,angle-2))),axis]
        ends=[]
        for move in moves:
            for length in step/2.**np.arange(4):
                b=a+length*move
                if b@direction<plate:
                    drop=(b-a)@direction
                    if drop>=-1e-12:continue
                    b=a+(plate-a@direction)/drop*(b-a)
                ends.append(b)
        if not ends:return finish('no_candidates')
        ends=np.asarray(ends)
        ok,values=valid_connections(np.repeat(a[None,:],len(ends),axis=0),ends,
            field,distance,gap,angle,plate,direction)
        valid=np.flatnonzero(ok)
        valid=[int(i) for i in valid if key(ends[i]) not in visited]
        chosen=None if force_cone or not valid else (ends[valid[0]],float(values[valid[0]]))
        if chosen is None and branch_trials.get(key(a),0)<2:
            moves=cone_directions(axis,np.deg2rad(max(0.,angle-1)))
            for length in step/2.**np.arange(4):
                if time.perf_counter()>=deadline:return finish('time_budget')
                candidates=a+length*moves
                below=candidates@direction<plate
                if below.any():
                    delta=candidates[below]-a
                    candidates[below]=a+((plate-a@direction)/(delta@direction))[:,None]*delta
                probes=a+np.linspace(0,1,9)[None,:,None]*(candidates-a)[:,None,:]
                clearance=distance(probes.reshape(-1,3)).reshape(-1,9)
                ids=[i for i in np.flatnonzero(clearance.min(axis=1)>=gap-2e-6)
                     if key(candidates[i]) not in visited]
                ids=np.asarray(ids,dtype=int)
                score=np.minimum(clearance[ids,-1],gap+step)+.5*((a-candidates[ids])@direction)
                ids=ids[np.argsort(-score,kind='stable')]
                for first in range(0,len(ids),24):
                    if time.perf_counter()>=deadline:return finish('time_budget')
                    batch=ids[first:first+24]
                    ok,values=valid_connections(np.repeat(a[None,:],len(batch),axis=0),candidates[batch],
                        field,distance,gap,angle,plate,direction)
                    good=np.flatnonzero(ok)
                    if len(good):
                        i=int(good[0]);chosen=(candidates[batch[i]],float(values[i]));escapes+=1
                        branch_trials[key(a)]=branch_trials.get(key(a),0)+1
                        break
                if chosen is not None:break
        if chosen is None:
            if len(path)<2 or backtracks>=24:return finish('blocked')
            length=float(np.linalg.norm(path[-1]-path[-2]))
            if rewound+length>3.+1e-8:return finish('rewind_budget')
            path.pop();ranks.pop();rewound+=length;backtracks+=1;force_cone=True
            continue
        b,value=chosen;path.append(b);ranks.append(value);visited.add(key(b));force_cone=False
        if b@direction<=plate+1e-6:
            finish('platform_reached')
            return np.asarray(path),np.asarray(ranks)
    return finish('step_budget')


def recover_stalled(graph,source_report,field,distance,landing_check,direction=(0,1,0),plate=0.,
                    radius=1.5,time_budget=5.,maximum_components=None,edge_budget=24,
                    search_mode='local',prefer_platform=None,transition_rewind=True):
    """Attempt every failed group by default, with time_budget for each group.

    An explicit maximum_components is retained only for bounded diagnostic
    calls. Normal planning/recovery has no component-count cap.
    """
    if not np.isfinite(radius) or not np.isfinite(time_budget) or radius<=0 or time_budget<=0 or edge_budget<1:
        raise ValueError('Recovery budgets must be positive and finite.')
    if maximum_components is not None and (isinstance(maximum_components,bool) or
            not isinstance(maximum_components,(int,np.integer)) or maximum_components<1):
        raise ValueError('maximum_components must be None or a positive integer.')
    if search_mode not in ('local','wide'):raise ValueError('Unknown recovery search mode.')
    started=time.perf_counter();deadline=started+time_budget
    if landing_check is None:raise ValueError('Stalled recovery requires model landing checks.')
    points=list(np.asarray(graph.points).copy());n_original=len(points);seed_count=source_report['seed_count']
    original_edges=graph.lines.reshape(-1,3)[:,1:]
    parents={int(a):int(b) for a,b in original_edges}
    incoming={}
    for a,b in original_edges:incoming.setdefault(int(b),[]).append(int(a))
    kinds=np.asarray(graph['node_kind']);original_radius=np.asarray(graph.point_data.get('root_contact_radius',np.zeros(len(points))))
    roots=set(np.flatnonzero((kinds==2)|(original_radius>0)))
    if 'is_root' in graph.point_data:roots.update(np.flatnonzero(graph['is_root']))
    roots={i for i in roots if i not in parents}
    terminals={}
    for seed in range(seed_count):
        node=seed;seen=set()
        while node in parents:
            if node in seen:raise ValueError('Cycle in source skeleton.')
            seen.add(node);node=parents[node]
        terminals[node]=terminals.get(node,0)+1
    failed=sorted((node for node in terminals if node not in roots),key=lambda node:-terminals[node])
    records=[];new_roots={};new_ranks={};attempted=0;edge_checks=0;candidate_count=0;budget_stop=False
    direction=np.asarray(direction,dtype=float);direction/=np.linalg.norm(direction)
    gap=source_report['clearance_mm'];angle=source_report['local_cone_angle_degrees']
    stage_end=float(source_report.get('merge_start_length_mm',0.))
    if stage_end>0 and 'contact_path_length' not in graph.point_data:
        raise ValueError('Staged recovery requires saved contact path lengths.')
    travelled=list(np.asarray(graph.point_data.get('contact_path_length',np.zeros(len(points))),dtype=float))
    prefer_platform=bool(source_report.get('prefer_platform',False) if prefer_platform is None else prefer_platform)
    for stalled in failed:
        if maximum_components is not None and attempted>=maximum_components:
            budget_stop=True;break
        deadline=time.perf_counter()+time_budget
        attempted+=1;endpoint=np.asarray(points[stalled])
        if travelled[stalled]<stage_end-1e-9:
            if not transition_rewind:
                records.append(dict(source_stalled_node=int(stalled),recovered=False,
                    reason='protected_contact_stage',contact_path_length_mm=float(travelled[stalled]),
                    required_length_mm=stage_end))
                continue
            from SupportGenerationCurved.transition_recovery import transition_paths, checked_transition
            fixed=float(source_report.get('contact_gradient_length_mm',stage_end))
            transition=float(source_report.get('contact_transition_length_mm',max(0.,stage_end-fixed)))
            anchors=[];node=stalled
            while True:
                if fixed-1e-8<=travelled[node]<travelled[stalled]-.20:
                    anchors.append(node)
                upper=incoming.get(node,[])
                if len(upper)!=1 or travelled[node]<=fixed+1e-8:break
                node=upper[0]
                if travelled[stalled]-travelled[node]>2*radius+1e-8:break
            if len(anchors)>4:anchors=[anchors[i] for i in np.linspace(0,len(anchors)-1,4,dtype=int)]
            chosen=None;model_option=None;attempts=[]
            if transition>0 and 'contact_downhill_direction' in graph.point_data:
                for anchor in anchors:
                    if time.perf_counter()>=deadline:break
                    contact=np.asarray(graph['contact_downhill_direction'][anchor],float)
                    contact/=max(np.linalg.norm(contact),1e-15)
                    upper=incoming.get(anchor,[])
                    previous=None;previous_length=None
                    if len(upper)==1:
                        previous=np.asarray(points[anchor])-points[upper[0]];previous_length=float(np.linalg.norm(previous))
                        previous=previous/max(previous_length,1e-15)
                    options,statistics=transition_paths(points[anchor],travelled[anchor],contact,
                        field,distance,direction,gap,angle,fixed,transition,deadline,landing_check=landing_check,
                        previous_direction=previous,previous_length=previous_length)
                    attempts.append(dict(anchor=int(anchor),travel_mm=float(travelled[anchor]),**statistics))
                    for option in options:
                        if time.perf_counter()>=deadline:break
                        path=option['points']
                        valid,checks=checked_transition(path,travelled[anchor],contact,field,distance,
                            direction,gap,angle,fixed,transition,plate,previous_direction=previous,previous_length=previous_length)
                        if not valid:continue
                        if option['kind']=='model':
                            if model_option is None:model_option=(anchor,path,'model',checks)
                        elif prefer_platform:
                            suffix=platform_connection(path[-1],field,distance,gap,angle,plate,direction,deadline)
                            if suffix is not None:
                                joined=np.vstack((path,suffix[0][1:]))
                                valid,joined_checks=checked_transition(joined,travelled[anchor],contact,field,distance,
                                    direction,gap,angle,fixed,transition,plate,
                                    previous_direction=previous,previous_length=previous_length)
                                if valid:chosen=(anchor,joined,'platform',joined_checks);break
                    if chosen is not None:break
                    if model_option is not None and not prefer_platform:break
            if chosen is None:chosen=model_option
            if chosen is not None:
                anchor,path,kind,checks=chosen;previous=anchor
                ranks,_=field(path[1:])
                for point,value in zip(path[1:],ranks):
                    j=len(points);points.append(point.copy());parents[previous]=j;new_ranks[j]=float(value)
                    travelled.append(travelled[previous]+float(np.linalg.norm(point-points[previous])));previous=j
                roots.add(j);new_roots[j]=dict(rank=float(ranks[-1]),anchor=anchor,point=path[-1].copy(),kind=kind)
                records.append(dict(source_stalled_node=int(stalled),source_anchor_node=int(anchor),recovered=True,
                    reason='protected_transition_rewind',route_kind=kind,affected_contacts=terminals[stalled],
                    rewound_mm=float(travelled[stalled]-travelled[anchor]),candidate_point=path[-1].tolist(),
                    new_path_segments=len(path)-1,transition_checks=checks,attempts=attempts))
                continue
            records.append(dict(source_stalled_node=int(stalled),recovered=False,
                reason='protected_contact_stage',contact_path_length_mm=float(travelled[stalled]),
                required_length_mm=stage_end,transition_rewind_attempts=attempts))
            continue
        if prefer_platform:
            # Try the stalled endpoint first, then a small number of tail
            # rewinds. Never cross a junction or bypass a protected neck.
            bed_anchors=[];node=stalled;tail_length=0.
            while len(bed_anchors)<5:
                if travelled[node]>=stage_end-1e-9:bed_anchors.append(node)
                upper=incoming.get(node,[])
                if len(upper)!=1:break
                other=upper[0];tail_length+=np.linalg.norm(points[other]-points[node])
                if tail_length>radius*2:break
                node=other
            bed_path=None
            # Keep part of the component's remaining time for model landings.
            bed_deadline=min(deadline,time.perf_counter()+max(.01,(deadline-time.perf_counter())*.5))
            platform_attempts=[]
            for index,anchor in enumerate(bed_anchors):
                # Share this component's platform budget across rewind anchors;
                # the first failed endpoint must not consume all alternatives.
                remaining=bed_deadline-time.perf_counter()
                if remaining<=0:break
                anchor_deadline=min(bed_deadline,time.perf_counter()+remaining/(len(bed_anchors)-index))
                diagnostic=dict(source_anchor_node=int(anchor))
                bed_path=platform_connection(np.asarray(points[anchor]),field,distance,gap,angle,
                    plate,direction,anchor_deadline,step=source_report.get('step_mm',.6),diagnostics=diagnostic)
                platform_attempts.append(diagnostic)
                if bed_path is not None:break
                if time.perf_counter()>=bed_deadline:break
            if bed_path is not None:
                path,path_ranks=bed_path;previous=anchor
                for point,value in zip(path[1:],path_ranks):
                    j=len(points);points.append(point.copy());parents[previous]=j;new_ranks[j]=float(value)
                    travelled.append(travelled[previous]+float(np.linalg.norm(point-points[previous])))
                    previous=j
                target=path[-1];roots.add(j)
                new_roots[j]=dict(rank=float(path_ranks[-1]),anchor=anchor,point=target.copy(),kind='platform')
                records.append(dict(source_stalled_node=int(stalled),source_anchor_node=int(anchor),
                    recovered=True,root_kind='platform',candidate_point=target.tolist(),
                    affected_contacts=terminals[stalled],new_path_segments=len(path)-1,
                    platform_attempts=platform_attempts))
                continue
        closest,faces=landing_check.query(endpoint[None,:]);normal=landing_check.normals[faces[0]]
        normal=normal/np.linalg.norm(normal)
        # Two rotated six-point tangent rings, at four distances: fixed49
        # proposals including the original nearest patch, before deduplication.
        ring=tangent_ring(normal[None,:])[0]
        rotated=(ring+np.roll(ring,-1,axis=0));rotated/=np.linalg.norm(rotated,axis=1)[:,None]
        ring=np.vstack((ring,rotated))
        query=np.vstack((closest, (closest[:,None,:]+radius*np.array([.15,.35,.65,1.])[:,None,None]*ring[None,:,:]).reshape(-1,3)))
        if search_mode=='wide':
            # Look below the tip too: nearest projection of only its tangent
            # plane would repeatedly rediscover the nearby neck wall.
            horizontal=tangent_ring(direction[None,:])[0]
            samples=[endpoint-radius*drop*direction+radius*r*horizontal
                     for drop in (.25,.5,1.) for r in (0.,.25,.5,.85)]
            query=np.vstack((query,*samples))
        surface,face=landing_check.query(query)
        candidates=surface+gap*landing_check.normals[face]
        for _ in range(2):
            surface,_=landing_check.query(candidates);delta=candidates-surface
            candidates=surface+gap*delta/np.maximum(np.linalg.norm(delta,axis=1)[:,None],1e-15)
        candidates=np.unique(candidates,axis=0)
        candidates=candidates[np.linalg.norm(candidates-endpoint,axis=1)<=radius+gap*.1]
        candidate_count+=len(candidates)
        preferred=landing_check(candidates)
        hard=landing_check.fallback_mask(candidates)
        feasible=np.flatnonzero(preferred|hard)
        anchors=[];node=stalled;tail_length=0.
        while len(anchors)<(64 if search_mode=='wide' else 5):
            anchors.append(node)
            # Rewind only the failed tail; never cross an upstream junction.
            upper=incoming.get(node,[])
            if len(upper)!=1:break
            other=upper[0];tail_length+=np.linalg.norm(points[other]-points[node])
            if tail_length>radius*2:break
            node=other
        anchors=[node for node in anchors if travelled[node]>=stage_end-1e-9]
        # Only relaxed-orientation landings use the longest path length. Avoid
        # evaluating the entire field and forest for preferred landings.
        longest=None
        if np.any(~preferred[feasible]):
            ranks,_=field(np.asarray(points))
            longest=np.zeros(len(points))
            for child in np.argsort(ranks)[::-1]:
                if int(child) in parents:
                    lower=parents[int(child)]
                    longest[lower]=max(longest[lower],longest[child]+np.linalg.norm(points[child]-points[lower]))
        trials=[]
        for anchor in anchors:
            a=np.asarray(points[anchor]);_,g=field(a[None,:]);axis=-g[0]/max(np.linalg.norm(g[0]),1e-15)
            for i in feasible:
                delta=candidates[i]-a;length=np.linalg.norm(delta)
                if length<1e-7:continue
                if search_mode=='local' and delta@axis/length<np.cos(np.deg2rad(angle-1)):continue
                if search_mode=='wide' and delta@direction>=-1e-7:continue
                if not preferred[i] and longest[anchor]+length<landing_check.fallback_min_length:continue
                trials.append((not bool(preferred[i]),-landing_check.alignment(candidates[i]),
                               float(np.linalg.norm(candidates[i]-endpoint)),length,int(anchor),int(i)))
        trials.sort()
        if search_mode=='wide' and trials:
            anchor_distances=distance(np.asarray([points[i] for i in anchors]))
            scores={a:(float(min(d,3.)),float(np.asarray(points[a])@direction))
                    for a,d in zip(anchors,anchor_distances)}
            trials=diverse_trials(trials,edge_budget,scores)
        else:trials=trials[:edge_budget]
        chosen=None;chosen_path=None;tested=0
        path_attempts=[]
        # Small batches allow the clock budget to stop recovery between groups.
        for start in range(0,len(trials),8):
            if time.perf_counter()>=deadline:budget_stop=True;break
            batch=trials[start:start+8]
            a=np.asarray([points[item[4]] for item in batch]);b=np.asarray([candidates[item[5]] for item in batch])
            ok,rank=valid_connections(a,b,field,distance,gap,angle,plate,direction)
            tested+=len(batch);edge_checks+=len(batch)
            if ok.any():
                i=int(np.flatnonzero(ok)[0]);chosen=(batch[i],b[i],rank[i]);break
        if chosen is None and search_mode=='wide':
            # At most eight complete-path attempts per component. Candidates
            # and rewind anchors are ranked before this more expensive stage.
            for trial in trials[:8]:
                if time.perf_counter()>=deadline:budget_stop=True;break
                diagnostic=dict(source_anchor_node=int(trial[4]),candidate_point=candidates[trial[5]].tolist())
                result=guided_connection(np.asarray(points[trial[4]]),candidates[trial[5]],
                    field,distance,gap,angle,plate,direction,deadline,diagnostics=diagnostic)
                path_attempts.append(diagnostic)
                if result is not None:
                    chosen_path,path_ranks=result
                    chosen=(trial,chosen_path[-1],path_ranks[-1]);break
        if chosen is None:
            records.append(dict(source_stalled_node=int(stalled),recovered=False,candidates=len(candidates),
                landing_checks_passed=len(feasible),cone_candidates=len(trials),edge_checks=tested,
                rewind_anchor_count=len(anchors),tested_anchor_nodes=sorted({int(t[4]) for t in trials}),
                tested_landing_count=len({t[5] for t in trials}),path_attempts=path_attempts,
                platform_attempts=platform_attempts if prefer_platform else []));continue
        trial,target,rank=chosen;anchor=trial[4]
        if chosen_path is None:chosen_path=np.asarray([points[anchor],target]);path_ranks=np.asarray([rank])
        previous=anchor
        for point,value in zip(chosen_path[1:],path_ranks):
            j=len(points);points.append(point.copy());parents[previous]=j;new_ranks[j]=float(value)
            travelled.append(travelled[previous]+float(np.linalg.norm(point-points[previous])))
            previous=j
        roots.add(j);new_roots[j]=dict(rank=float(rank),anchor=anchor,point=target.copy())
        records.append(dict(source_stalled_node=int(stalled),source_anchor_node=anchor,recovered=True,
            candidate_point=target.tolist(),achieved_up_dot=landing_check.alignment(target),
            orientation_fallback=bool(trial[0]),affected_contacts=terminals[stalled],edge_checks=tested,
            new_path_segments=len(chosen_path)-1))
    # Remove superseded tails by retaining exactly the paths reachable from seeds.
    used=set();end_counts={}
    for seed in range(seed_count):
        node=seed;seen=set();used.add(node)
        while node in parents:
            if node in seen:raise ValueError('Cycle after recovery.')
            seen.add(node);node=parents[node];used.add(node)
        end_counts[node]=end_counts.get(node,0)+1
    order=np.asarray(list(range(seed_count))+sorted(used-set(range(seed_count))),dtype=int)
    mapping={int(node):i for i,node in enumerate(order)}
    retained_edges=[(a,b) for a,b in parents.items() if a in used]
    xyz=np.asarray(points);lines=np.c_[np.full(len(retained_edges),2,dtype=np.int64),np.asarray([[mapping[a],mapping[b]] for a,b in retained_edges],dtype=np.int64).reshape(-1,2)]
    output=pv.PolyData(xyz[order],lines=lines.ravel())
    for key in graph.point_data:
        original=np.asarray(graph.point_data[key]);extended=np.zeros((len(points),*original.shape[1:]),dtype=original.dtype)
        extended[:n_original]=original
        if key=='contact_path_length':extended=np.asarray(travelled)
        if key=='contact_downhill_direction':
            for a,b in retained_edges:
                if b>=n_original:extended[b]=extended[a]
        if key.startswith('source_'):extended[n_original:]=-1
        output.point_data[key]=extended[order]
    old_edge_index={(int(a),int(b)):i for i,(a,b) in enumerate(original_edges)}
    for key in graph.cell_data:
        original=np.asarray(graph.cell_data[key]);data=np.zeros((len(retained_edges),*original.shape[1:]),dtype=original.dtype)
        for i,edge in enumerate(retained_edges):
            if edge in old_edge_index:data[i]=original[old_edge_index[edge]]
        output.cell_data[key]=data
    edge_kind=np.asarray(output.cell_data.get('edge_kind',np.zeros(len(retained_edges))),dtype=np.int32)
    for i,(a,b) in enumerate(retained_edges):
        if b in new_ranks:edge_kind[i]=4
    output.cell_data['edge_kind']=edge_kind
    root_mask=np.asarray([node in roots for node in order]);stalled_mask=np.asarray([node in end_counts and node not in roots for node in order])
    incoming_counts=np.bincount([mapping[b] for a,b in retained_edges],minlength=len(order))
    kind=np.zeros(len(order),dtype=np.int32);kind[:seed_count]=1;kind[root_mask]=2;kind[stalled_mask]=3;kind[incoming_counts>1]=4
    output.point_data['node_kind']=kind;output.point_data['is_root']=root_mask.astype(np.uint8);output.point_data['is_stalled']=stalled_mask.astype(np.uint8)
    for key,default in (('root_contact_radius',0.),('landing_achieved_up_dot',np.nan),('landing_is_fallback',0)):
        data=np.asarray(output.point_data.get(key,np.full(len(order),default))).copy()
        for old,record in new_roots.items():
            point=record['point'];index=mapping[old]
            if record.get('kind')=='platform':
                data[index]=np.nan if key=='landing_achieved_up_dot' else 0
                continue
            if key=='root_contact_radius':data[index]=landing_check.contact_radii.get(point.tobytes(),0.)
            elif key=='landing_achieved_up_dot':data[index]=landing_check.alignment(point)
            else:data[index]=landing_check.alignment(point)<=landing_check.min_up_dot
        output.point_data[key]=data
    rank_key='routing_rank' if getattr(field,'direction_only',False) else 'scalar'
    rank=np.asarray(output.point_data.get(rank_key,np.zeros(len(order)))).copy()
    for old,value in new_ranks.items():rank[mapping[old]]=value
    output.point_data[rank_key]=rank
    report=copy.deepcopy(source_report)
    rooted=sum(count for node,count in end_counts.items() if node in roots)
    platform_roots={node for node in roots if abs(np.dot(points[node],direction)-plate)<=1e-6}
    model_roots=roots-platform_roots
    report.update(nodes=len(order),edges=len(retained_edges),rooted_contacts=rooted,unrooted_contacts=seed_count-rooted,
        platform_contacts=sum(count for node,count in end_counts.items() if node in platform_roots),
        model_contacts=sum(count for node,count in end_counts.items() if node in model_roots),
        platform_roots=len(platform_roots),model_roots=len(model_roots),
        prefer_platform=prefer_platform,
        stalled_fronts=int(stalled_mask.sum()),model_or_platform_roots=int(root_mask.sum()),
        root_group_sizes=sorted([count for node,count in end_counts.items() if node in roots],reverse=True),
        unrooted_group_sizes=sorted([count for node,count in end_counts.items() if node not in roots],reverse=True))
    if stage_end>0:
        report['stalled_in_contact_stage']=[int(i) for i in np.flatnonzero(stalled_mask & (np.asarray(travelled)[order]<stage_end-1e-9))]
    report['fallback_landings']=[dict(node=int(i),point=output.points[i].tolist(),achieved_up_dot=float(output['landing_achieved_up_dot'][i])) for i in np.flatnonzero(output['landing_is_fallback'])]
    report['stalled_recovery']=dict(attempted_components=attempted,recovered_components=len(new_roots),
        failed_components_before=len(failed),candidate_points=candidate_count,edge_checks=edge_checks,
        radius_mm=radius,time_budget_seconds=time_budget,seconds=time.perf_counter()-started,budget_stopped=budget_stop,
        time_budget_scope='per_failed_component',unattempted_components=len(failed)-attempted,
        search_mode=search_mode,prefer_platform=prefer_platform,
        transition_rewind_enabled=bool(transition_rewind),
        transition_rewind_recovered_components=sum(r.get('reason')=='protected_transition_rewind' and r['recovered'] for r in records),
        maximum_components=maximum_components,edge_budget_per_component=edge_budget,records=records,
        limitation='Bounded landing search and optional target-guided paths; not a global obstacle planner. Sampled centreline/landing checks, not branch-solid or first-layer certification.')
    report['model_landing_check']=landing_check.report()
    return output,report
