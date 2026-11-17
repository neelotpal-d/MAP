"""Reusable router definitions promoted from batched_route_curves.py; see promotion_sources.json."""
import numpy as np

def evaluate_chains(points,sizes,first,last,min_radius=5.,chunk_size=64):
    """Return [(path, max_curvature) or None], in input order.

    Every candidate uses the scalar implementation's .15mm chord grid and
    minimum 25 samples. Padding is excluded from all acceptance reductions.
    """
    points=np.asarray(points,float);sizes=np.asarray(sizes,int)
    first=np.asarray(first,float);last=np.asarray(last,float)
    if not len(points):return []
    chain=[];part=[];aa=[];bb=[];tt0=[];tt1=[]
    for i,n in enumerate(sizes):
        pts=points[i,:n];tangents=[first[i]]
        for j in range(1,n-1):
            tangent=pts[j+1]-pts[j-1];tangents.append(tangent/np.linalg.norm(tangent))
        tangents.append(last[i])
        for j in range(n-1):
            chain.append(i);part.append(j);aa.append(pts[j]);bb.append(pts[j+1])
            tt0.append(tangents[j]);tt1.append(tangents[j+1])
    aa=np.asarray(aa);bb=np.asarray(bb);tt0=np.asarray(tt0);tt1=np.asarray(tt1)
    delta=bb-aa;chord=np.linalg.norm(delta,axis=1);drop=-delta[:,1]
    h=np.minimum(.42*chord,.96*drop/np.maximum(abs(tt0[:,1])+abs(tt1[:,1]),1e-6))
    cc=aa+h[:,None]*tt0;dd=bb-h[:,None]*tt1
    count=np.maximum(25,np.ceil(chord/.15).astype(int)+1)
    segment_ok=(chord>=.1)&(drop>.01);segment_paths=[None]*len(aa);curvature=np.zeros(len(aa))
    ids=np.flatnonzero(segment_ok)
    for offset in range(0,len(ids),chunk_size):
        selected=ids[offset:offset+chunk_size];n=count[selected]
        columns=np.arange(int(n.max()))[None,:]
        t=np.minimum(columns/(n[:,None]-1),1.);u=1.-t
        a=aa[selected,None];b=bb[selected,None];c=cc[selected,None];d=dd[selected,None]
        p=u[:,:,None]**3*a+3*u[:,:,None]**2*t[:,:,None]*c+3*u[:,:,None]*t[:,:,None]**2*d+t[:,:,None]**3*b
        v=3*u[:,:,None]**2*(c-a)+6*u[:,:,None]*t[:,:,None]*(d-c)+3*t[:,:,None]**2*(b-d)
        acceleration=6*u[:,:,None]*(d-2*c+a)+6*t[:,:,None]*(b-2*d+c)
        k=np.linalg.norm(np.cross(v,acceleration),axis=2)/np.maximum(np.linalg.norm(v,axis=2)**3,1e-15)
        k=np.where(columns<n[:,None],k,0.)
        max_k=k.max(axis=1)
        bad_drop=((np.diff(p[:,:,1],axis=1)>=-1e-8)&(columns[:,:-1]<(n-1)[:,None])).any(axis=1)
        ok=(max_k<=1./min_radius)&~bad_drop
        segment_ok[selected]=ok;curvature[selected]=max_k
        for row,i in enumerate(selected):
            if ok[row]:segment_paths[i]=p[row,:n[row]].copy()
    result=[[] for _ in sizes];valid=np.ones(len(sizes),bool);maximum=np.zeros(len(sizes))
    for j,i in enumerate(chain):
        if not segment_ok[j]:valid[i]=False;continue
        result[i].append(segment_paths[j] if part[j]==0 else segment_paths[j][1:])
        maximum[i]=max(maximum[i],curvature[j])
    return [(np.vstack(result[i]),float(maximum[i])) if valid[i] else None for i in range(len(sizes))]

def evaluate_specs(specifications,min_radius=5.):
    """Attach paths to successful candidate dictionaries without reordering."""
    if not specifications:return []
    points=np.zeros((len(specifications),4,3));sizes=[];first=[];last=[]
    for i,spec in enumerate(specifications):
        n=len(spec['points']);points[i,:n]=spec['points'];sizes.append(n)
        first.append(spec['first']);last.append(spec['last'])
    evaluated=evaluate_chains(points,sizes,first,last,min_radius)
    result=[]
    for spec,answer in zip(specifications,evaluated):
        if answer is None:continue
        c={k:v for k,v in spec.items() if k not in ('points','first','last')}
        c['path'],c['k']=answer;result.append(c)
    return result
