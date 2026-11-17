"""Support-only scalar refinement using the actual support field's gradients.

Model levels are immutable. Local support statistics are computed independently
of model sampling density. The maximum-spacing guard is a sampled gradient
estimate, not a certificate of printed backing or exact normal-ray separation.
"""
import time
import numpy as np


def refine_support_schedule(model_levels, values, gradients, *, target_spacing=.2,
                            maximum_spacing=.5, maximum_count=10000):
    levels=np.asarray(model_levels,dtype=float).reshape(-1)
    values=np.asarray(values,dtype=float).reshape(-1)
    gradients=np.asarray(gradients,dtype=float)
    if (len(levels)<2 or not np.isfinite(levels).all() or np.any(np.diff(levels)<=0)
            or gradients.shape!=(len(values),3) or not np.isfinite(values).all()
            or not np.isfinite(gradients).all()):
        raise ValueError('Support scheduling requires increasing finite model levels and finite scalar/gradient samples.')
    if (not np.isfinite([target_spacing,maximum_spacing]).all() or target_spacing<=0
            or maximum_spacing<target_spacing or maximum_count<len(levels)):
        raise ValueError('Require 0 < support target spacing <= maximum spacing and enough allowed layers.')
    norms=np.linalg.norm(gradients,axis=1)
    if np.any(norms<=1e-10):
        raise ValueError('Support field has near-zero sampled gradients; physical spacing cannot be estimated reliably.')
    inside=(values>=levels[0]-1e-9)&(values<=levels[-1]+1e-9)
    ids=np.clip(np.searchsorted(levels,values[inside],side='right')-1,0,len(levels)-2)
    norms=norms[inside];factors=np.ones(len(levels)-1,dtype=int);stats=[];total=len(levels)
    # A probe on/near a slice also constrains the neighbouring interval. This
    # avoids making support outside the model's sampled scalar bins invisible.
    order=np.argsort(ids);ids=ids[order];norms=norms[order]
    for i,step in enumerate(np.diff(levels)):
        left=np.searchsorted(ids,max(0,i-1),side='left');right=np.searchsorted(ids,min(len(levels)-2,i+1),side='right')
        local=norms[left:right]
        if not len(local):continue
        median=float(np.median(local));minimum=float(local.min())
        limit=min(target_spacing*median,maximum_spacing*minimum)
        n=max(1,int(np.ceil(step/limit-1e-12)))
        if n>maximum_count or total+n-1>maximum_count:
            raise ValueError('Support gradient refinement exceeds the maximum layer count; inspect low-gradient regions.')
        total+=n-1
        factors[i]=n
        stats.append(dict(model_interval=i,samples=len(local),gradient_median=median,gradient_minimum=minimum,subintervals=n))
    if int(factors.sum())+1>maximum_count:
        raise ValueError('Support gradient refinement exceeds the maximum layer count.')
    refined=np.r_[np.concatenate([np.linspace(a,b,int(n)+1)[:-1] for a,b,n in zip(levels[:-1],levels[1:],factors)]),levels[-1]]
    return refined,dict(enabled=True,model_level_count=len(levels),support_level_count=len(refined),
        added_support_levels=len(refined)-len(levels),model_levels_preserved=bool(np.isin(levels,refined).all()),
        support_gradient_samples=len(values),samples_in_schedule_range=int(inside.sum()),
        target_support_spacing_mm=target_spacing,maximum_estimated_support_spacing_mm=maximum_spacing,
        statistic='Separate local support median; local minimum only guards estimated maximum spacing.',
        gradient_norm_minimum=float(norms.min()) if len(norms) else None,
        gradient_norm_median=float(np.median(norms)) if len(norms) else None,intervals=stats,
        limitation='Sampled gradient estimate; downstream actual normal gaps and printed-path backing still require checking.')


def select_support_indices(model_levels, support_levels, model_indices):
    """Include inserted support levels immediately preceding selected model levels."""
    model=np.asarray(model_levels);support=np.asarray(support_levels);selected=set()
    for index in np.asarray(model_indices,dtype=int):
        if not 0<=index<len(model):raise ValueError('Selected model index outside schedule.')
        lo=np.searchsorted(support,model[index-1],side='right') if index else 0
        hi=np.searchsorted(support,model[index],side='right')
        selected.update(range(lo,hi))
    return np.array(sorted(selected),dtype=int)


def sample_support_gradients(geometry, evaluator, *, spacing=2., batch_size=8192, analytic_y=False):
    """Probe every swept branch at centres and four interior radial locations."""
    if not np.isfinite(spacing) or spacing<=0:raise ValueError('Support gradient probe spacing must be positive.')
    a,b=np.asarray(geometry.a),np.asarray(geometry.b)
    if not len(a):return np.empty(0),np.empty((0,3)),dict(probes=0)
    if analytic_y:
        p=np.vstack([a,b]);return p[:,1],np.tile([0.,1.,0.],(len(p),1)),dict(probes=len(p),method='Exact constant analytic-Y gradient')
    length=np.linalg.norm(b-a,axis=1);steps=np.maximum(1,np.ceil(length/spacing).astype(int));count=steps+1
    ids=np.repeat(np.arange(len(a)),count);starts=np.repeat(np.r_[0,np.cumsum(count)[:-1]],count)
    t=(np.arange(len(ids))-starts)/steps[ids];axis=b-a;axis/=np.maximum(length[:,None],1e-20)
    axis[length<1e-12]=[0,1,0]
    reference=np.tile([1.,0,0],(len(a),1));reference[abs(axis[:,0])>.9]=[0,0,1]
    u=np.cross(axis,reference);u/=np.linalg.norm(u,axis=1)[:,None];v=np.cross(axis,u)
    centres=a[ids]+t[:,None]*(b-a)[ids]
    radius=.8*(np.asarray(geometry.ra)[ids]*(1-t)+np.asarray(geometry.rb)[ids]*t)
    probes=np.vstack([centres,centres+radius[:,None]*u[ids],centres-radius[:,None]*u[ids],
        centres+radius[:,None]*v[ids],centres-radius[:,None]*v[ids]])
    probes=np.unique(probes,axis=0);values=[];gradients=[];retained=0;started=time.perf_counter()
    for start in range(0,len(probes),batch_size):
        q=probes[start:start+batch_size];q=q[np.asarray(geometry.evaluate(q))<=0]
        if not len(q):continue
        if hasattr(evaluator,'evaluate_scalars_and_gradients'):
            f,g=evaluator.evaluate_scalars_and_gradients(q)
        else:
            # Differentiate the ACTUAL transferred/blended support scalar,
            # never project a model gradient onto exterior support samples.
            epsilon=.01;f=evaluator.evaluate_scalars(q);g=np.empty_like(q)
            for k in range(3):
                delta=np.zeros(3);delta[k]=epsilon
                g[:,k]=(evaluator.evaluate_scalars(q+delta)-evaluator.evaluate_scalars(q-delta))/(2*epsilon)
        values.append(f);gradients.append(g);retained+=len(q)
    if not retained:raise ValueError('No material support probes available for gradient-aware slicing.')
    return np.concatenate(values),np.vstack(gradients),dict(proposed_probes=len(probes),material_probes=retained,
        axial_spacing_mm=spacing,radial_fraction=.8,seconds=time.perf_counter()-started,
        method='Actual support scalar gradients; analytic when available, otherwise 0.01mm central differences.')
