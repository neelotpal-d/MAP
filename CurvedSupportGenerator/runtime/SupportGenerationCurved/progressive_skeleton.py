"""Skeleton-only descending growth with a local field cone and vertical preference."""
import time
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


def cone_direction(preferred,axis,angle):
    axis=np.asarray(axis,dtype=float);axis=axis/max(np.linalg.norm(axis),1e-15)
    preferred=np.asarray(preferred,dtype=float);preferred=preferred/max(np.linalg.norm(preferred),1e-15)
    cosine=float(np.clip(preferred@axis,-1,1))
    if cosine>=np.cos(angle):return preferred
    tangent=preferred-cosine*axis
    if np.linalg.norm(tangent)<1e-10:
        tangent=np.cross(axis,[1.,0,0])
        if np.linalg.norm(tangent)<1e-10:tangent=np.cross(axis,[0,0,1.])
    return np.cos(angle)*axis+np.sin(angle)*tangent/np.linalg.norm(tangent)


def staged_axis(contact_down, local_down, progress):
    """Rotate the fixed contact direction smoothly to the local field axis."""
    if progress <= 0:return contact_down.copy()
    if progress >= 1:return local_down.copy()
    weight=progress*progress*(3.-2.*progress)
    cosine=float(np.clip(contact_down@local_down,-1.,1.))
    theta=np.arccos(cosine)
    tangent=local_down-cosine*contact_down
    if np.linalg.norm(tangent)<1e-10:
        if cosine>0:return contact_down.copy()
        tangent=np.cross(contact_down,[1.,0,0])
        if np.linalg.norm(tangent)<1e-10:tangent=np.cross(contact_down,[0,0,1.])
    tangent/=np.linalg.norm(tangent)
    return np.cos(theta*weight)*contact_down+np.sin(theta*weight)*tangent


def cone_directions(axis,angle):
    """Deterministic azimuth coverage, with extra rings near the cone rim."""
    axis=np.asarray(axis,dtype=float);axis=axis/max(np.linalg.norm(axis),1e-15)
    if angle<=1e-10:return axis[None,:]
    tangent=np.cross(axis,[1.,0.,0.])
    if np.linalg.norm(tangent)<1e-8:tangent=np.cross(axis,[0.,0.,1.])
    tangent/=np.linalg.norm(tangent);other=np.cross(axis,tangent)
    phi=np.arange(32)*2*np.pi/32
    ring=np.cos(phi)[:,None]*tangent+np.sin(phi)[:,None]*other
    return np.vstack([axis[None,:]]+[
        np.cos(angle*f)*axis+np.sin(angle*f)*ring for f in (.25,.5,.75,.9375,1.)])


def grow(seeds,field,distance,direction=(0,1,0),plate=0.,gap=.3,step=.6,
         angle_degrees=45.,merge_distance=6.,max_steps=450,progress=None,landing_check=None,
         growth_preference='vertical',clearance_escape_band=1.5,clearance_escape_weight=.45,
         platform_downward_bias=0.,contact_gradient_length=0.,contact_transition_length=0.,contact_gradients=None,
         prefer_platform=True):
    """Construct a directed forest; accepted edges descend their ordering rank.

    field(points) -> (scalar, gradient). Only centerlines are collision checked;
    branch radii, offset patches, footprint support and layers are deferred.
    growth_preference='field' uses local downhill as the unsteered direction;
    'vertical' preserves the original vertical preference inside the field cone.
    """
    started=time.perf_counter();seeds=np.asarray(seeds,dtype=float).reshape(-1,3)
    direction=np.asarray(direction,dtype=float);direction/=np.linalg.norm(direction)
    if (step<=0 or gap<0 or not 0<angle_degrees<90 or merge_distance<=0 or max_steps<1
            or clearance_escape_band<0 or not 0<=clearance_escape_weight<=1
            or not np.isfinite(platform_downward_bias) or not 0<=platform_downward_bias<=1):
        raise ValueError('Invalid skeleton growth settings.')
    if not all(np.isfinite(v) and v>=0 for v in (contact_gradient_length,contact_transition_length)):
        raise ValueError('Contact stage lengths must be finite and nonnegative.')
    if growth_preference not in ('vertical','field'):
        raise ValueError('Growth preference must be vertical or field.')
    max_angle=np.deg2rad(angle_degrees)
    points=list(seeds.copy());values,initial_gradients=field(seeds)
    initial_gradients=np.asarray(initial_gradients if contact_gradients is None else contact_gradients,dtype=float)
    if initial_gradients.shape!=seeds.shape or not np.isfinite(initial_gradients).all() or np.any(np.linalg.norm(initial_gradients,axis=1)<1e-8):
        raise ValueError('Contacts need finite nonzero gradients.')
    contact_down=-initial_gradients/np.linalg.norm(initial_gradients,axis=1)[:,None]
    travelled=[0.]*len(seeds)
    stage_end=contact_gradient_length+contact_transition_length
    values=list(np.asarray(values,dtype=float));owners=list(range(len(seeds)))
    parents=[-1]*len(seeds);edges=[];kinds=[];roots=[];stalled=[];active=list(range(len(seeds)))
    components=list(range(len(seeds)));merges=0;rejections=0
    modelward_moves_avoided=0;downward_retry_steps=0
    cone_escape_attempts=0;cone_escape_steps=0

    def find(i):
        while components[i]!=i:
            components[i]=components[components[i]];i=components[i]
        return i

    def stage(node, local_down):
        progress=(np.clip((travelled[node]-contact_gradient_length)/contact_transition_length,0.,1.)
                  if contact_transition_length>0 else float(travelled[node]>=contact_gradient_length-1e-9))
        axis=staged_axis(contact_down[owners[node]],local_down,progress)
        return axis,max_angle*progress*progress*(3.-2.*progress),progress

    def valid_many(a,b,nodes):
        nonlocal rejections
        delta=b-a;length=np.linalg.norm(delta,axis=1)
        probes=a[:,None,:]+np.linspace(0,1,17)[None,:,None]*delta[:,None,:]
        flat=probes.reshape(-1,3);d=distance(flat).reshape(-1,17);s,g=field(flat)
        s=np.asarray(s).reshape(-1,17);g=np.asarray(g).reshape(-1,17,3)
        lengths=np.linalg.norm(g,axis=2)
        alignment=np.sum((-g/np.maximum(lengths[:,:,None],1e-15))*(delta/np.maximum(length[:,None],1e-15))[:,None,:],axis=2)
        ok=((length>1e-7)&np.isfinite(s).all(axis=1)&np.isfinite(g).all(axis=(1,2))
            &(lengths>1e-8).all(axis=1)&(np.diff(s,axis=1)<-1e-9).all(axis=1)
            &(alignment.min(axis=1)>=np.cos(max_angle)-1e-5)
            &(d.min(axis=1)>=gap-2e-6)&((probes@direction).min(axis=1)>=plate-1e-6))
        if hasattr(field,'native_descent_mask'):ok &= field.native_descent_mask(probes)
        # Refine only accepted edges close to the cone rim. Nearest-boundary
        # FEM directions can change between coarse samples; a coarse pass must
        # not hide a small angular violation inside a grazing segment.
        refine=np.flatnonzero(ok & (alignment.min(axis=1)<np.cos(max(0.,max_angle-np.deg2rad(.5)))))
        if len(refine):
            # Match the independent routing audit's samples for near-rim
            # candidates. A narrow field turn between the former 33 samples
            # could pass growth and then fail the 65-probe saved-tree audit.
            fine=a[refine,None,:]+np.linspace(0,1,65)[None,:,None]*delta[refine,None,:]
            fs,fg=field(fine.reshape(-1,3));fs=np.asarray(fs).reshape(-1,65)
            fg=np.asarray(fg).reshape(-1,65,3);fn=np.linalg.norm(fg,axis=2)
            fd=distance(fine.reshape(-1,3)).reshape(-1,65)
            fa=np.sum(-fg*delta[refine,None,:],axis=2)/np.maximum(fn*length[refine,None],1e-15)
            fine_ok=(np.isfinite(fs).all(axis=1)&np.isfinite(fg).all(axis=(1,2))
                &(fn>1e-8).all(axis=1)&(np.diff(fs,axis=1)<-1e-9).all(axis=1)
                &(fa.min(axis=1)>=np.cos(max_angle)-1e-5)&(fd.min(axis=1)>=gap-2e-6))
            if hasattr(field,'native_descent_mask'):fine_ok &= field.native_descent_mask(fine)
            ok[refine] &= fine_ok
        for i,node in enumerate(nodes):
            axis,allowed,_=stage(node,-g[i,0]/max(lengths[i,0],1e-15))
            ok[i] &= (delta[i]/max(length[i],1e-15))@axis >= np.cos(allowed)-1e-8
        rejections+=int((~ok).sum())
        return ok,s[:,-1]

    def append(a,b,value,kind):
        j=len(points);points.append(np.asarray(b));values.append(float(value));owners.append(owners[a]);parents.append(-1)
        edges.append((a,j));kinds.append(kind);parents[a]=j
        travelled.append(travelled[a]+float(np.linalg.norm(points[a]-b)))
        return j

    completed=0;front_merges=0
    for iteration in range(max_steps):
        if not active:break
        completed+=1
        xyz=np.asarray(points);history=cKDTree(xyz)
        current=xyz[active];scalars,gradients=field(current);ds=distance(current)
        active_rows={node:row for row,node in enumerate(active)}
        down_moves=[]
        for row,node in enumerate(active):
            if not np.isfinite(gradients[row]).all() or np.linalg.norm(gradients[row])<1e-8:
                down_moves.append(np.zeros(3));continue
            local=-gradients[row]/max(np.linalg.norm(gradients[row]),1e-15)
            axis,allowed,_=stage(node,local)
            down_moves.append(cone_direction(-direction,axis,allowed))
        down_moves=np.asarray(down_moves).reshape(-1,3)
        down_ok=np.zeros(len(active),dtype=bool);down_distance=ds.copy()
        if prefer_platform:
            lengths=np.full(len(active),step)
            for boundary in (contact_gradient_length,stage_end):
                remaining=boundary-np.asarray([travelled[n] for n in active])
                mask=remaining>1e-9;lengths[mask]=np.minimum(lengths[mask],remaining[mask])
            down_ends=current+lengths[:,None]*down_moves
            below=(down_ends@direction)<plate
            if below.any():
                t=(plate-current[below]@direction)/((down_ends[below]-current[below])@direction)
                down_ends[below]=current[below]+t[:,None]*(down_ends[below]-current[below])
            down_ok,_=valid_many(current,down_ends,active)
            down_distance=distance(down_ends)

        def modelward_over_clear_down(ends,nodes):
            """Do not sacrifice an available downward clearance corridor to merge."""
            nonlocal modelward_moves_avoided
            if not prefer_platform:return np.zeros(len(nodes),dtype=bool)
            rows=np.asarray([active_rows[n] for n in nodes])
            available=down_ok[rows]
            avoid=np.zeros(len(nodes),dtype=bool)
            if available.any():
                proposed=distance(np.asarray(ends)[available])
                reference=np.minimum(ds[rows[available]],down_distance[rows[available]])
                avoid[available]=proposed<reference-max(1e-4,.05*step)
            modelward_moves_avoided+=int(avoid.sum())
            return avoid
        # Radius queries cannot be crowded out by the branch's own history.
        neighbors=history.query_ball_point(current,merge_distance)
        front_tree=cKDTree(current)
        front_neighbors=front_tree.query_ball_point(current,merge_distance)
        loads=np.bincount([find(i) for i in range(len(seeds))],minlength=len(seeds))
        consumed=set();next_active=[]
        # Equal-height fronts cannot connect horizontally. Try a shared point
        # below both, validating BOTH incoming edges against the local cones.
        pairs=sorted(front_tree.query_pairs(min(merge_distance,step*2.5)),
                     key=lambda pair:(np.linalg.norm(current[pair[0]]-current[pair[1]]),pair))
        for left,right in pairs:
            na,nb=active[left],active[right]
            if na in consumed or nb in consumed or find(owners[na])==find(owners[nb]):continue
            if min(travelled[na],travelled[nb])<stage_end-1e-9:continue
            a,b=current[left],current[right];mid=(a+b)*.5
            midpoint_height=min(a@direction,b@direction)
            drops=(step,step*.5,step*.25,step*1.5) if prefer_platform else (step*.25,step*.5,step,step*1.5)
            for drop in drops:
                junction=mid+direction*(midpoint_height-mid@direction-drop)
                if max(np.linalg.norm(junction-a),np.linalg.norm(junction-b))>step*1.25:continue
                ok,sv=valid_many(np.asarray([a,b]),np.asarray([junction,junction]),[na,nb])
                ok &= ~modelward_over_clear_down(np.asarray([junction,junction]),[na,nb])
                if not ok.all():continue
                j=append(na,junction,sv[0],2)
                travelled[j]=min(travelled[j],travelled[nb]+float(np.linalg.norm(junction-b)))
                edges.append((nb,j));kinds.append(2);parents[nb]=j
                components[find(owners[nb])]=find(owners[na])
                consumed.update((na,nb));next_active.append(j)
                merges+=1;front_merges+=1;break
        pending=[];moves=[];fallback_candidates={};deferred_moves={}
        for row,node in enumerate(active):
            if node in consumed:continue
            a=np.asarray(points[node]);gradient=gradients[row];norm=np.linalg.norm(gradient)
            if not np.isfinite(norm) or norm<1e-8 or ds[row]<gap-2e-6:
                stalled.append(node);continue
            local_axis=-gradient/norm
            axis,allowed_angle,stage_progress=stage(node,local_axis)
            merge_ready=travelled[node]>=stage_end-1e-9
            target=None
            candidates=[int(other) for other in neighbors[row]
                        if merge_ready and travelled[other]>=stage_end-1e-9 and find(owners[other])!=find(owners[node]) and values[other]<scalars[row]-1e-6]
            # Distance dominates; contact load gives a bounded preference to
            # established trunks among similarly close candidates.
            score=lambda other:(np.linalg.norm(xyz[other]-a)/(1+.12*np.log2(1+loads[find(owners[other])])),other)
            candidates.sort(key=score)
            close=[other for other in candidates if np.linalg.norm(xyz[other]-a)<=step*1.25]
            if close:
                ok,_=valid_many(np.repeat(a[None,:],len(close),axis=0),xyz[close],[node]*len(close))
                ok &= ~modelward_over_clear_down(xyz[close],[node]*len(close))
                feasible=np.flatnonzero(ok)
                if len(feasible):
                    target=close[feasible[0]]
                    edges.append((node,target));kinds.append(2);parents[node]=target
                    components[find(owners[node])]=find(owners[target]);merges+=1
                    continue
            if candidates:target=candidates[0]
            # The local field is always the cone axis. Vertical and merge
            # preferences may be projected anywhere inside the requested cone;
            # do not silently narrow the user's 45 degree limit near a model.
            preferred=axis.copy() if growth_preference=='field' else -direction
            # Also attract fronts at the same height towards a rendezvous below
            # them; restricting attraction to lower history strands misses these.
            fronts=[i for i in front_neighbors[row] if merge_ready and travelled[active[i]]>=stage_end-1e-9 and active[i] not in consumed
                    and find(owners[active[i]])!=find(owners[node])]
            if fronts:
                other=min(fronts,key=lambda i:(np.linalg.norm(current[i]-a)/(1+.12*np.log2(1+loads[find(owners[active[i]])])),i))
                b=current[other];mid=(a+b)*.5
                lateral=np.linalg.norm((b-a)-((b-a)@direction)*direction)
                drop=max(step,lateral/(2*np.tan(max(1e-6,max_angle))))
                rendezvous=mid+direction*(min(a@direction,b@direction)-mid@direction-drop)
                if target is None or np.linalg.norm(b-a)<=np.linalg.norm(xyz[target]-a)*1.2:
                    toward=rendezvous-a;toward/=np.linalg.norm(toward)
                    preferred=.1*preferred+.9*toward
                    target=None
            if target is not None:
                toward=xyz[target]-a;toward/=np.linalg.norm(toward)
                preferred=.25*preferred+.75*toward
            # Reassert platform-down after attraction, on every active front
            # including shared trunks. Clearance escape and cone checks retain
            # priority; this is a soft preference, not a new acceptance rule.
            if platform_downward_bias > 0:
                preferred/=max(np.linalg.norm(preferred),1e-15)
                bias=platform_downward_bias*stage_progress
                preferred=(1.-bias)*preferred-bias*direction
            # Ask for an outward model normal only for a front that is already
            # close to the clearance boundary. The final direction is still
            # projected into the unchanged local field cone below.
            if landing_check is not None and ds[row] < gap+clearance_escape_band:
                outward=landing_check.outward(a[None,:])[0]
                preferred=(1.-clearance_escape_weight)*preferred+clearance_escape_weight*outward
            move=cone_direction(preferred,axis,allowed_angle)
            if modelward_over_clear_down((a+step*move)[None,:],[node])[0]:
                move=down_moves[row].copy()
            pending.append(node);moves.append(move)
        moves=np.asarray(moves).reshape(-1,3)
        for attempt in range(12 if prefer_platform else 8):
            if not pending:break
            a=np.asarray([points[i] for i in pending])
            if prefer_platform and 4<=attempt<8:
                # Explicitly try the most downward direction permitted by
                # the contact stage. Wall steering in earlier retries must
                # not overwrite this independent candidate family.
                moves=np.asarray([down_moves[active_rows[n]].copy() for n in pending])
            if attempt==(8 if prefer_platform else 4):
                # If vertical/merge bias fails, try pure field descent before
                # declaring a front stalled. The acceptance cone is unchanged.
                _,g=field(a)
                moves=np.asarray([stage(node,-gradient/max(np.linalg.norm(gradient),1e-15))[0] for node,gradient in zip(pending,g)])
            lengths=np.full(len(pending),step/(2**(attempt%4)))
            for i,node in enumerate(pending):
                for boundary in (contact_gradient_length,stage_end):
                    remaining=boundary-travelled[node]
                    if remaining>1e-9:lengths[i]=min(lengths[i],remaining)
            b=a+lengths[:,None]*moves
            land=(b@direction)<plate
            if land.any():
                t=(plate-a[land]@direction)/((b[land]-a[land])@direction)
                b[land]=a[land]+t[:,None]*(b[land]-a[land])
            ts=np.linspace(0,1,9)
            probe=a[:,None,:]+ts[None,:,None]*(b-a)[:,None,:]
            ds_edge=distance(probe.reshape(-1,3)).reshape(-1,9)
            crossed=(ds_edge<gap).any(axis=1)
            if crossed.any():
                ids=np.flatnonzero(crossed);first=np.argmax(ds_edge[ids]<gap,axis=1)
                lo=ts[np.maximum(0,first-1)].copy();hi=ts[first].copy()
                for _ in range(20):
                    mid=(lo+hi)/2
                    outside=distance(a[ids]+mid[:,None]*(b[ids]-a[ids]))>=gap
                    lo=np.where(outside,mid,lo);hi=np.where(outside,hi,mid)
                b[ids]=a[ids]+lo[:,None]*(b[ids]-a[ids]);land[ids]=True
            landing_ok=np.ones(len(b),dtype=bool)
            model_land=land&~((b@direction)<=plate+1e-6)
            if landing_check is not None and model_land.any():
                ids=np.flatnonzero(model_land)
                landing_ok[ids]=landing_check(b[ids])
                if prefer_platform:
                    # A model attachment is secondary to an already verified
                    # free downward step. Retain the model route when down is
                    # blocked; do not force unsupported platform connections.
                    landing_ok[ids] &= ~np.asarray([down_ok[active_rows[pending[i]]] for i in ids])
                if hasattr(landing_check,'fallback_mask'):
                    for i,allowed in zip(ids,landing_check.fallback_mask(b[ids])):
                        if allowed and not landing_ok[i]:
                            fallback_candidates.setdefault(pending[i],{})[b[i].tobytes()]=b[i].copy()
                rejected=ids[~landing_ok[ids]]
                if len(rejected):
                    # Try sliding away from the unsuitable wall within the
                    # same cone and existing bounded backtracking budget.
                    _,g=field(a[rejected]);axes=-g/np.maximum(np.linalg.norm(g,axis=1)[:,None],1e-15)
                    outward=landing_check.outward(b[rejected])
                    for i,axis,n in zip(rejected,axes,outward):
                        guided_axis,allowed,_=stage(pending[i],axis)
                        moves[i]=cone_direction(axis+n,guided_axis,allowed)
            ok,end_scalar=valid_many(a,b,pending)
            ok &= landing_ok
            for i in np.flatnonzero(ok):
                if pending[i] in fallback_candidates and not land[i]:
                    # Finish the small preferred-candidate search before
                    # choosing a fallback or escaping along the model wall.
                    deferred_moves.setdefault(pending[i],(b[i].copy(),end_scalar[i]))
                    ok[i]=False;continue
                j=append(pending[i],b[i],end_scalar[i],3 if land[i] else 0)
                if prefer_platform and 4<=attempt<8 and not land[i]:downward_retry_steps+=1
                if land[i]:roots.append(j)
                else:next_active.append(j)
            pending=[node for i,node in enumerate(pending) if not ok[i]]
            moves=moves[~ok]
        if pending and fallback_candidates:
            # Consider fallback only AFTER the ordinary cone-limited retries.
            # Rank candidates already visited; no extra geometric search.
            longest=np.zeros(len(points))
            for child in np.argsort(values)[::-1]:
                lower=parents[child]
                if lower>=0:longest[lower]=max(longest[lower],longest[child]+np.linalg.norm(points[child]-points[lower]))
            unresolved=[]
            for node in pending:
                if prefer_platform and node in deferred_moves:
                    point,value=deferred_moves[node];next_active.append(append(node,point,value,0))
                    continue
                candidates=list(fallback_candidates.get(node,{}).values())
                candidates=[p for p in candidates if longest[node]+np.linalg.norm(p-points[node])>=landing_check.fallback_min_length]
                if not candidates:
                    if node in deferred_moves:
                        point,value=deferred_moves[node];next_active.append(append(node,point,value,0))
                    else:unresolved.append(node)
                    continue
                candidates.sort(key=lambda p:landing_check.alignment(p),reverse=True)
                b=np.asarray(candidates);a=np.repeat(np.asarray(points[node])[None,:],len(b),axis=0)
                ok,scalar=valid_many(a,b,[node]*len(b));feasible=np.flatnonzero(ok)
                if len(feasible):
                    i=feasible[0];j=append(node,b[i],scalar[i],3);roots.append(j)
                elif node in deferred_moves:
                    point,value=deferred_moves[node];next_active.append(append(node,point,value,0))
                else:unresolved.append(node)
            pending=unresolved
        # Only failed ordinary retries pay for azimuth coverage. Preserve the
        # fixed neck and transition exactly; this search starts on mature tails.
        unresolved=[]
        for node in pending:
            if travelled[node]<stage_end-1e-9:
                unresolved.append(node);continue
            cone_escape_attempts+=1
            a=np.asarray(points[node]);_,g=field(a[None,:]);norm=np.linalg.norm(g[0])
            if not np.isfinite(norm) or norm<1e-8:
                unresolved.append(node);continue
            axis=-g[0]/norm
            # Keep a one-degree margin to avoid repeatedly grazing the limit.
            moves=cone_directions(axis,np.deg2rad(max(0.,angle_degrees-1)))
            chosen=None
            for length in step/2.**np.arange(4):
                ends=a+length*moves
                below=ends@direction<plate
                if below.any():
                    delta=ends[below]-a
                    ends[below]=a+((plate-a@direction)/(delta@direction))[:,None]*delta
                probes=a+np.linspace(0,1,9)[None,:,None]*(ends-a)[:,None,:]
                clearance=distance(probes.reshape(-1,3)).reshape(-1,9)
                ids=np.flatnonzero(clearance.min(axis=1)>=gap-2e-6)
                # Prefer clearance corridors with downward progress. Do not
                # reward large clearance distances once safely away from walls.
                score=np.minimum(clearance[ids,-1],gap+step)+.5*((a-ends[ids])@direction)
                ids=ids[np.argsort(-score,kind='stable')]
                for first in range(0,len(ids),24):
                    batch=ids[first:first+24]
                    ok,rank=valid_many(np.repeat(a[None,:],len(batch),axis=0),ends[batch],[node]*len(batch))
                    good=np.flatnonzero(ok)
                    if len(good):
                        i=int(good[0]);chosen=(ends[batch[i]],rank[i]);break
                if chosen is not None:break
            if chosen is None:unresolved.append(node)
            else:
                point,value=chosen;on_plate=point@direction<=plate+1e-6
                j=append(node,point,value,3 if on_plate else 0)
                if on_plate:roots.append(j)
                else:next_active.append(j)
                cone_escape_steps+=1
        stalled.extend(unresolved)
        active=next_active
        if progress is not None and (iteration%20==0 or not active):
            progress(dict(iteration=iteration+1,active=len(active),merges=merges,roots=len(roots),stalled=len(stalled)))
    stalled.extend(active)
    xyz=np.asarray(points);lines=np.asarray([[2,a,b] for a,b in edges],dtype=np.int64).ravel()
    mesh=pv.PolyData(xyz,lines=lines)
    mesh.point_data['routing_rank' if getattr(field,'direction_only',False) else 'scalar']=np.asarray(values)
    incoming=np.bincount(np.asarray(edges,dtype=int)[:,1],minlength=len(points)) if edges else np.zeros(len(points),dtype=int)
    point_kind=np.zeros(len(points),dtype=np.int32);point_kind[:len(seeds)]=1
    point_kind[roots]=2;point_kind[stalled]=3;point_kind[incoming>1]=4
    mesh.point_data['node_kind']=point_kind
    mesh.point_data['is_root']=np.isin(np.arange(len(points)),roots).astype(np.uint8)
    mesh.point_data['is_stalled']=np.isin(np.arange(len(points)),stalled).astype(np.uint8)
    mesh.point_data['contact_path_length']=np.asarray(travelled)
    mesh.point_data['contact_downhill_direction']=contact_down[np.asarray(owners,dtype=np.int64)]
    if landing_check is not None and hasattr(landing_check,'results'):
        achieved=np.full(len(points),np.nan);fallback=np.zeros(len(points),dtype=np.uint8)
        for node in roots:
            record=landing_check.results.get(np.asarray(points[node],dtype=float).tobytes())
            if record:
                achieved[node]=record['achieved_up_dot']
                fallback[node]=achieved[node]<=landing_check.min_up_dot
        mesh.point_data['landing_achieved_up_dot']=achieved
        mesh.point_data['landing_is_fallback']=fallback
    mesh.cell_data['edge_kind']=np.asarray(kinds,dtype=np.int32)
    # Follow actual graph pointers: a seed can be merged into another branch
    # which later stalls. Such a seed must not be counted as successfully rooted.
    root_set=set(roots);rooted=0;terminal_counts={}
    for seed in range(len(seeds)):
        node=seed;visited=set()
        while parents[node]!=-1:
            if node in visited:raise RuntimeError('Cycle in descending skeleton.')
            visited.add(node);node=parents[node]
        rooted+=int(node in root_set);terminal_counts[node]=terminal_counts.get(node,0)+1
    report=dict(seed_count=len(seeds),nodes=len(points),edges=len(edges),merges=merges,
        model_or_platform_roots=len(roots),stalled_fronts=len(set(stalled)),rooted_contacts=rooted,
        unrooted_contacts=len(seeds)-rooted,
        root_group_sizes=sorted([n for i,n in terminal_counts.items() if i in root_set],reverse=True),
        unrooted_group_sizes=sorted([n for i,n in terminal_counts.items() if i not in root_set],reverse=True),
        rejected_edges=rejections,iterations=completed,
        planning_seconds=time.perf_counter()-started,step_mm=step,local_cone_angle_degrees=angle_degrees,
        shared_front_junctions=front_merges,merge_policy='cone_checked_front_rendezvous_and_history',
        clearance_mm=gap,vertical_preference=growth_preference=='vertical',
        growth_preference=growth_preference,clearance_escape_band_mm=clearance_escape_band,
        platform_downward_bias=platform_downward_bias,
        prefer_platform=bool(prefer_platform),
        modelward_moves_avoided=modelward_moves_avoided,downward_retry_steps=downward_retry_steps,
        cone_escape_attempts=cone_escape_attempts,cone_escape_steps=cone_escape_steps,
        routing_candidate_policy=('steered_then_downward_then_field_then_cone' if prefer_platform
                                  else 'steered_then_field_then_cone'),
        contact_gradient_length_mm=contact_gradient_length,contact_transition_length_mm=contact_transition_length,
        contact_stage_measure='branch arc length from offset contact seed',merge_start_length_mm=stage_end,
        stalled_in_contact_stage=[int(node) for node in set(stalled) if travelled[node]<stage_end-1e-9],
        clearance_escape_weight=clearance_escape_weight,
        ordering_parameter='build_height' if getattr(field,'direction_only',False) else 'transferred_scalar',
        limitations=['Skeleton-only experiment: centerline sampling, not branch-volume clearance certification.',
                     'No contact footprints, offset trimming, envelope rebuild, layers or toolpaths generated.'])
    if report['unrooted_contacts']:
        report['warnings']=[dict(
            code='unrooted_contacts_require_model_lift_review',
            contacts=report['unrooted_contacts'],
            message='Some required support contacts have neither an accepted model-root landing nor a plate path.',
            action='Review the contact patches; omit genuinely small patches, enable bounded recovery, or lift/reorient the model if the remaining contacts are required.'
        )]
        report['model_lift_review_recommended']=True
    else:
        report['warnings']=[]
        report['model_lift_review_recommended']=False
    if landing_check is not None:
        report['model_landing_check']=landing_check.report()
        if 'landing_is_fallback' in mesh.point_data:
            ids=np.flatnonzero(mesh['landing_is_fallback'])
            report['fallback_landings']=[dict(node=int(i),point=xyz[i].tolist(),achieved_up_dot=float(mesh['landing_achieved_up_dot'][i])) for i in ids]
    return mesh,report
