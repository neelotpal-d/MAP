"""Reusable router definitions promoted from support_direction_certificate.py; see promotion_sources.json."""
import collections

from pathlib import Path

import time

import numpy as np

from scipy.spatial import cKDTree

class DirectionCertificate:
    def __init__(self,guide,spacing=8.,maximum_edge=.65):
        started=time.perf_counter();self.guide=guide;self.maximum_edge=maximum_edge
        native=guide.native.native;path=Path(native.path);stat=path.stat()
        key=(str(path.resolve()),stat.st_mtime_ns,stat.st_size,tuple(guide.up),guide.outer_band,spacing,maximum_edge)
        self.stats=collections.Counter()
        if key in _CACHE:
            self.origin,self.shape,self.low,self.high=_CACHE[key];self.spacing=spacing
            self.initialization_seconds=time.perf_counter()-started;return
        points=native.points;tetra=points[native.cells]
        # Bound complete cell AABBs, not only tetrahedron interiors: VTK's
        # FindCell may choose an adjacent cell near a parametric boundary,
        # but tests the cell's AABB before EvaluatePosition (VTK 9.6.2).
        lower=tetra.min(axis=1);upper=tetra.max(axis=1);centers=(lower+upper)*.5
        radius=np.linalg.norm(upper-lower,axis=1)*.5
        norm=np.linalg.norm(native.gradients,axis=1)
        unit=native.gradients/np.maximum(norm[:,None],1e-15)
        unit[(norm<1e-8)|~np.isfinite(unit).all(axis=1)]=0.
        pad=guide.outer_band+maximum_edge
        self.origin=np.floor((points.min(axis=0)-pad)/spacing)*spacing
        self.shape=np.ceil((points.max(axis=0)+pad-self.origin)/spacing).astype(int)
        ids=np.indices(self.shape).reshape(3,-1).T
        locations=self.origin+(ids+.5)*spacing
        low=np.tile(guide.up,(len(locations),1));high=low.copy()
        bucket=np.floor(np.log2(np.maximum(radius,.125))).astype(int)
        reach=guide.outer_band+np.sqrt(3)*spacing*.5+maximum_edge*.5+1e-4
        for value in np.unique(bucket):
            selected=np.flatnonzero(bucket==value);tree=cKDTree(centers[selected])
            neighbours=tree.query_ball_point(locations,reach+radius[selected].max())
            for i,near in enumerate(neighbours):
                if not near:continue
                normals=unit[selected[near]]
                low[i]=np.minimum(low[i],normals.min(axis=0));high[i]=np.maximum(high[i],normals.max(axis=0))
        self.low=low;self.high=high;self.spacing=spacing
        _CACHE[key]=(self.origin,self.shape,self.low,self.high)
        self.initialization_seconds=time.perf_counter()-started

    def accepts(self,a,b,threshold):
        delta=b-a;length=np.linalg.norm(delta,axis=1)
        direction=-delta/length[:,None]
        cells=np.floor(((a+b)*.5-self.origin)/self.spacing).astype(int)
        inside=((cells>=0)&(cells<self.shape)).all(axis=1)
        low=np.tile(self.guide.up,(len(a),1));high=low.copy()
        if inside.any():
            ids=np.ravel_multi_index(cells[inside].T,self.shape)
            low[inside]=self.low[ids];high[inside]=self.high[ids]
        lower=np.sum(np.where(direction>=0,direction*low,direction*high),axis=1)
        accepted=(lower>=threshold+1e-10)&(length<=self.maximum_edge)
        self.stats['edges_considered']+=len(a);self.stats['directions_certified']+=int(accepted.sum())
        return accepted

def certified_valid_path(self,path):
    """Keep original fine checks wherever whole-edge bounds cannot prove safety."""
    if self.field.scalar_band!=0:
        from .fast_platform_grafts import GraftTrial
        return GraftTrial.valid_path(self,path)
    started=time.perf_counter();d=self.distance(path);p=np.asarray(path)
    if d.min()<self.settings['clearance']-2e-6:self.stats['coarse_clearance_rejected']+=1;return False
    delta=np.diff(p,axis=0);length=np.linalg.norm(delta,axis=1)
    if np.any(delta[:,1]>=-1e-8):return False
    lower=np.minimum(d[:-1],d[1:])-length*.5
    far=lower>self.field.outer_band+1e-4
    threshold=np.cos(np.deg2rad(44.75))-1e-5
    if np.any(-delta[far,1]/length[far]<threshold):return False
    proven=(lower>.3+1e-4)&self.direction_certificate.accepts(p[:-1],p[1:],threshold)
    remaining=~(far|proven)
    self.stats['continuous_near_edges_certified']+=int((proven&~far).sum())
    if remaining.any():
        links=np.flatnonzero(remaining);nodes=np.unique(np.r_[links,links+1])
        _,g=self.field(p[nodes]);lookup=dict(zip(nodes,g));norm=np.linalg.norm(g,axis=1)
        if not np.isfinite(g).all() or norm.min()<1e-8:return False
        g0=np.asarray([lookup[i] for i in links]);g1=np.asarray([lookup[i+1] for i in links])
        cosine0=np.sum(-g0*delta[links],axis=1)/(np.linalg.norm(g0,axis=1)*length[links])
        cosine1=np.sum(-g1*delta[links],axis=1)/(np.linalg.norm(g1,axis=1)*length[links])
        if min(cosine0.min(),cosine1.min())<threshold:
            self.stats['coarse_field_rejected']+=1;return False
        probes=p[links,None]+np.linspace(0,1,33)[None,:,None]*delta[links,None]
        fine=self.distance(probes.reshape(-1,3)).reshape(-1,33)
        if fine.min()<self.settings['clearance']-2e-6:return False
        rank,g=self.field(probes.reshape(-1,3));rank=rank.reshape(-1,33);g=g.reshape(-1,33,3)
        norms=np.linalg.norm(g,axis=2);cosine=np.sum(-g*delta[links,None],axis=2)/(norms*length[links,None])
        if (not np.isfinite(g).all() or norms.min()<1e-8 or cosine.min()<threshold or
            np.any(np.diff(rank,axis=1)>=-1e-9)):return False
        if self.field.scalar_band>0 and not self.field.native_descent_mask(probes).all():return False
    self.profile['accepted_validation_seconds']+=time.perf_counter()-started
    return True
