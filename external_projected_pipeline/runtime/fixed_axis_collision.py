"""Continuous fixed-axis box versus finite bead enclosure, with causal BVH search.

All coordinates are Y-up. Positive findings are conservative potential hits:
Euclidean bead capsules are enclosed by L-infinity tubes. Rotating tools are
not covered. No machine envelope is supplied implicitly.
"""
import json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def load_tool(path):
    data=json.loads(Path(path).read_text())
    if not np.allclose(data.get('build_direction',[0,1,0]),[0,1,0],atol=1e-10):
        raise ValueError('Fixed-axis collision tool must use Y-up build_direction.')
    lower=np.array([data['tool_'+a+'_min'] for a in 'xyz'],float)
    upper=np.array([data['tool_'+a+'_max'] for a in 'xyz'],float)
    if not np.isfinite([lower,upper]).all() or (upper<=lower).any():raise ValueError('Invalid tool box.')
    return lower,upper


def swept_hits(start,end,lower,upper,a,b,radius,*,rotating=False):
    """Vectorized separating-axis test for two swept line/box enclosures.

    Equivalent to the reviewed two-variable LP: the Minkowski difference is a
    zonotope with five generators. Its facet normals are the cross products
    of generator pairs. Complete straight translations are checked, not poses.
    """
    if rotating:raise ValueError('Rotating tools require orientation enclosures.')
    start,end,lower,upper=map(lambda x:np.asarray(x,float),(start,end,lower,upper))
    a=np.asarray(a,float).reshape(-1,3);b=np.asarray(b,float).reshape(-1,3)
    radius=np.broadcast_to(np.asarray(radius,float),(len(a),))
    if any(x.shape!=(3,) for x in (start,end,lower,upper)) or a.shape!=b.shape or not all(np.isfinite(x).all() for x in (start,end,lower,upper,a,b,radius)) or (radius<0).any() or (upper<lower).any():
        raise ValueError('Invalid swept collision geometry.')
    move=end-start;bead=b-a;difference=(a+b)/2-(start+end)/2-(lower+upper)/2
    eye=np.eye(3);axes=[np.broadcast_to(eye,(len(a),3,3)),
        np.broadcast_to(np.cross(move,eye),(len(a),3,3)),np.cross(bead[:,None,:],eye[None]),
        np.cross(np.broadcast_to(move,bead.shape),bead)[:,None,:]]
    axes=np.concatenate(axes,axis=1)
    extent=np.einsum('nki,ni->nk',abs(axes),np.broadcast_to((upper-lower)/2,(len(a),3))+radius[:,None]+1e-7)
    extent+=.5*abs(axes@move)+.5*abs(np.einsum('nki,ni->nk',axes,bead))
    projection=abs(np.einsum('nki,ni->nk',axes,difference))
    return ~np.any(projection>extent+1e-10*np.linalg.norm(axes,axis=2),axis=1)


def audit(moves,move_orders,beads,bead_orders,radii,lower,upper,*,max_pairs=5_000_000):
    """KD broad phase + continuous narrow phase, excluding current/future beads."""
    started=time.perf_counter();moves=np.asarray(moves,float).reshape(-1,2,3);beads=np.asarray(beads,float).reshape(-1,2,3)
    move_orders=np.asarray(move_orders);bead_orders=np.asarray(bead_orders);radii=np.asarray(radii,float)
    if len(move_orders)!=len(moves) or len(bead_orders)!=len(beads) or len(radii)!=len(beads) or not all(np.isfinite(x).all() for x in (moves,beads,move_orders,bead_orders,radii)) or (radii<0).any():raise ValueError('Invalid chronological collision records.')
    lower,upper=np.asarray(lower),np.asarray(upper)
    if lower.shape!=(3,) or upper.shape!=(3,) or not np.isfinite([lower,upper]).all() or (upper<=lower).any():raise ValueError('Invalid tool bounds.')
    pairs=0;findings=[];groups=[]
    midpoint=beads.mean(axis=1);half=np.linalg.norm(beads[:,1]-beads[:,0],axis=1)/2+np.sqrt(3)*radii
    buckets=np.ceil(np.log2(np.maximum(half,.01))).astype(int)
    for group in np.unique(buckets):
        ids=np.flatnonzero(buckets==group);groups.append((ids,cKDTree(midpoint[ids]),float(half[ids].max())))
    extent=np.linalg.norm((upper-lower)/2)
    status='passed'
    for offset in range(0,len(moves),2048):
        chunk=moves[offset:offset+2048];centers=chunk.mean(axis=1)+(lower+upper)/2
        reach=extent+np.linalg.norm(chunk[:,1]-chunk[:,0],axis=1)/2
        for ids,tree,bound in groups:
            for local,near in enumerate(tree.query_ball_point(centers,reach+bound)):
                i=offset+local;targets=ids[near];targets=targets[bead_orders[targets]<move_orders[i]]
                pairs+=len(targets)
                if pairs>max_pairs:status='incomplete';break
                if not len(targets):continue
                mask=swept_hits(*moves[i],lower,upper,beads[targets,0],beads[targets,1],radii[targets])
                for j in targets[mask][:20]:
                    if len(findings)<1000:findings.append(dict(move=i,bead=int(j),move_order=int(move_orders[i]),bead_order=int(bead_orders[j])))
                if mask.any() and status=='passed':status='potential_collision'
            if status=='incomplete':break
        if status=='incomplete':break
    return dict(status=status,findings=findings,candidate_pairs=pairs,maximum_pairs=max_pairs,seconds=time.perf_counter()-started,
        scope='Continuous straight translations of the supplied fixed-axis box against earlier finite bead outer enclosures. Potential hits include enclosure false positives; rotating tools are not covered.')
