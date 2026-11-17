"""Vectorized refinement of bounded path chords; retains existing vertices."""
import time
import numpy as np


def part_anchors(parts,roles,minimum_boundary_points=4):
    if len(parts)!=len(roles):raise ValueError('Unaligned path parts and roles.')
    protected=[];offset=0;previous=None
    for part,role in zip(parts,roles):
        if not len(part):continue
        shared=previous is not None and np.linalg.norm(previous-part[0])<=1e-8
        local=np.arange(len(part))+offset-int(shared)
        protected.extend((int(local[0]),int(local[-1])))
        if 'boundary' in role:
            # Preserve the configured small-contour resolution as well as its
            # two opened seam endpoints. No support-tip polygon is collapsed.
            count=min(len(part),minimum_boundary_points)
            protected.extend(local[np.unique(np.linspace(0,len(part)-1,count).round().astype(int))].tolist())
        offset+=len(part)-int(shared);previous=part[-1]
    return protected


def _fragment_runs(points,tags,breaks,config):
    """Carry the existing downstream short-fragment policy through reduction."""
    if not config.get('enabled',False) or len(points)<2:return []
    minimum=float(config.get('min_length',2.));count=config.get('min_waypoints',5)
    count=None if count is None else int(count);jump=config.get('long_jump_threshold',2.)
    if minimum<=0 or not np.isfinite(minimum) or (count is not None and count<2):raise ValueError('Invalid fragment policy.')
    lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
    boundaries=set(breaks)
    if jump is not None:boundaries.update((np.flatnonzero(lengths>float(jump))+1).tolist())
    bounds=[0,*sorted(boundaries),len(points)];runs=[]
    for a,b in zip(bounds[:-1],bounds[1:]):
        if all(('infill' in str(t) or 'raster' in str(t)) and 'boundary' not in str(t) for t in tags[a:b]):
            old_length=float(lengths[a:max(a,b-1)].sum())
            runs.append((a,b,minimum,count,old_length<minimum and (count is None or b-a<count)))
    return runs


def select(points,normals,widths,tags,breaks,*,protected=(),max_spacing=1.,chord_error=.005,normal_degrees=2.,max_relative_length_loss=.005,region=None):
    start=time.perf_counter();p=np.asarray(points,float);n=len(p)
    if not np.isfinite([max_spacing,chord_error,normal_degrees]).all() or min(max_spacing,chord_error,normal_degrees)<=0 or normal_degrees>=90:
        raise ValueError('Point reduction requires finite positive distances and a normal angle below90degrees.')
    if not np.isfinite(max_relative_length_loss) or not 0<=max_relative_length_loss<1:raise ValueError('Invalid path-length loss bound.')
    normals=np.asarray(normals,float);widths=np.asarray(widths,float);tags=np.asarray(tags)
    if p.shape!=(n,3) or normals.shape!=p.shape or widths.shape!=(n,) or tags.shape!=(n,):raise ValueError('Unaligned path arrays.')
    if not np.isfinite(p).all() or not np.isfinite(normals).all() or not np.isfinite(widths).all():raise ValueError('Nonfinite path arrays.')
    if n<3:return np.arange(n),dict(input_points=n,output_points=n,seconds=time.perf_counter()-start)
    keep=np.zeros(n,bool);keep[[0,n-1]]=True
    protected=np.asarray(protected,int)
    if (protected<0).any() or (protected>=n).any():raise ValueError('Invalid protected path index.')
    keep[protected]=True
    breaks=np.asarray(breaks,int)
    if (breaks<=0).any() or (breaks>=n).any():raise ValueError('Invalid travel break.')
    keep[breaks]=True;keep[breaks-1]=True
    changes=np.flatnonzero((tags[1:]!=tags[:-1])|(abs(np.diff(widths))>1e-9))+1
    keep[changes]=True;keep[changes-1]=True
    d=np.diff(p,axis=0);length=np.linalg.norm(d,axis=1);arc=np.r_[0.,np.cumsum(length)]
    u=d/np.maximum(length[:,None],1e-20)
    turn=np.flatnonzero((length[:-1]>1e-8)&(length[1:]>1e-8)&(np.einsum('ij,ij->i',u[:-1],u[1:])<np.cos(np.deg2rad(10.))))+1
    keep[turn]=True
    # Keep hatch-to-wall junctions even when their projected turn is shallow.
    # Curvature in the lifting direction does not create a false hatch end.
    projected=d[:,[0,2]];plen=np.linalg.norm(projected,axis=1);pu=projected/np.maximum(plen[:,None],1e-20)
    hatch=np.array(['infill' in str(t) or 'raster' in str(t) for t in tags])
    junction=np.flatnonzero(hatch[1:-1]&(plen[:-1]>1e-8)&(plen[1:]>1e-8)&(np.einsum('ij,ij->i',pu[:-1],pu[1:])<np.cos(np.deg2rad(.25))))+1
    keep[junction]=True
    keep[np.array(['ironing' in str(t) for t in tags])]=True
    norm=np.linalg.norm(normals,axis=1);keep[norm<=1e-12]=True
    normals=normals/np.maximum(norm[:,None],1e-12)
    ids=np.arange(n);cos_scale=1.-np.cos(np.deg2rad(normal_degrees))
    if region is not None:
        import shapely
        region=region.buffer(1e-8);shapely.prepare(region)
    fallback=0;iterations=0
    for iterations in range(33):
        knots=np.flatnonzero(keep);left,right=knots[:-1],knots[1:];span=right-left
        group=np.minimum(np.searchsorted(knots,ids,side='right')-1,len(left)-1)
        a,b=left[group],right[group];v=p[b]-p[a];squared=np.einsum('ij,ij->i',v,v)
        t=np.clip(np.einsum('ij,ij->i',p-p[a],v)/np.maximum(squared,1e-30),0,1)
        error=p-(p[a]+t[:,None]*v);score=np.einsum('ij,ij->i',error,error)/(chord_error*chord_error)
        expected=normals[a]*(1.-t[:,None])+normals[b]*t[:,None];expected/=np.maximum(np.linalg.norm(expected,axis=1,keepdims=True),1e-20)
        score=np.maximum(score,np.maximum(0.,1.-np.einsum('ij,ij->i',expected,normals))/cos_scale)
        score[keep]=0.
        maximum=np.maximum.reduceat(score[:-1],left)
        bad_shape=(maximum>1.+1e-8)&(span>1)
        arc_distance=arc[right]-arc[left]
        bad_step=(arc_distance>max_spacing+1e-9)&(span>1)
        # A collinear retracing move has zero chord error but still deposits
        # material. Bound lost arc length as well as spatial deviation.
        bad_length=(np.linalg.norm(p[right]-p[left],axis=1)+1e-10<(1.-max_relative_length_loss)*arc_distance)&(span>1)
        bad_region=np.zeros(len(left),bool)
        if region is not None:
            import shapely
            check=np.flatnonzero((span>1)&~bad_shape&~bad_step&~bad_length)
            if len(check):
                lines=shapely.linestrings(np.stack((p[left[check]][:,[0,2]],p[right[check]][:,[0,2]]),axis=1))
                bad_region[check]=~shapely.covers(region,lines)
        bad=bad_shape|bad_step|bad_length|bad_region
        if not bad.any():break
        if iterations==32:
            keep[bad[group]]=True;fallback=int(bad.sum());break
        candidate=(score>=maximum[group]-1e-12)&(ids>a)&(ids<b)
        winner=np.minimum.reduceat(np.where(candidate,ids,n)[:-1],left)
        midpoint=np.searchsorted(arc,(arc[left]+arc[right])*.5)
        midpoint=np.clip(midpoint,left+1,right-1)
        chosen=np.where(bad_shape&(winner<n),winner,midpoint)
        keep[chosen[bad]]=True
    selected=np.flatnonzero(keep)
    return selected,dict(input_points=n,output_points=len(selected),removed_points=n-len(selected),
        max_spacing_mm=max_spacing,chord_error_mm=chord_error,normal_angle_degrees=normal_degrees,
        max_relative_length_loss_fraction=max_relative_length_loss,
        iterations=iterations+1,bounded_fallback_intervals=fallback,seconds=time.perf_counter()-start)


def reduce_generated(points,normals,widths,tags,breaks,*,parts,part_roles,target_distance,settings=None,region=None,minimum_boundary_points=4):
    """Compact aligned generator arrays before decimal serialization/heights.

    Disabled when absent in a saved legacy config. New UI/configs opt in.
    Only existing points are selected; no endpoint, seam or travel is moved.
    """
    if settings is None:settings={}
    if not isinstance(settings,dict):raise ValueError('point_reduction must be a JSON object.')
    if not isinstance(settings.get('enabled',False),bool):raise ValueError('point_reduction.enabled must be boolean.')
    if not settings.get('enabled',False):
        return points,normals,widths,tags,breaks,dict(enabled=False)
    began=time.perf_counter()
    spacing=float(target_distance)*float(settings.get('max_spacing_factor',2.))
    fragment=settings.get('fragment_filter',{})
    if not isinstance(fragment,dict):raise ValueError('point_reduction.fragment_filter must be an object.')
    protected=part_anchors(parts,part_roles,minimum_boundary_points)
    runs=_fragment_runs(points,tags,breaks,fragment)
    for a,b,*_ in runs:protected.extend((a,b-1))
    if fragment.get('enabled') and fragment.get('long_jump_threshold',2.) is not None:
        spacing=min(spacing,float(fragment.get('long_jump_threshold',2.))*(1.-1e-9))
        jumps=np.flatnonzero(np.linalg.norm(np.diff(points,axis=0),axis=1)>float(fragment.get('long_jump_threshold',2.)))+1
        protected.extend(jumps);protected.extend(jumps-1)
    selected,report=select(points,normals,widths,tags,breaks,
        protected=protected,max_spacing=spacing,
        chord_error=float(settings.get('chord_error_mm',.005)),normal_degrees=float(settings.get('normal_angle_degrees',2.)),
        max_relative_length_loss=float(settings.get('max_relative_length_loss',.005)),region=region)
    restore=[]
    for a,b,minimum,count,old_prune in runs:
        local=selected[(selected>=a)&(selected<b)]
        length=float(np.linalg.norm(np.diff(points[local],axis=0),axis=1).sum())
        if not old_prune and length<minimum and (count is None or len(local)<count):restore.extend(range(a,b))
    original_selected=len(selected)
    if restore:selected=np.union1d(selected,restore).astype(int)
    rewritten=np.searchsorted(selected,breaks).astype(int).tolist()
    if breaks:
        assert np.array_equal(selected[np.asarray(rewritten)],breaks)
        assert np.array_equal(selected[np.asarray(rewritten)-1],np.asarray(breaks)-1)
    report.update(enabled=True,stage='generated_arrays_before_writing_and_heights',target_distance_mm=float(target_distance),
        fragment_policy_retained_points=len(selected)-original_selected,output_points=len(selected),removed_points=len(points)-len(selected),seconds=time.perf_counter()-began)
    return points[selected],normals[selected],widths[selected],tags[selected],rewritten,report
