"""Small batched model-root checks; not a complete contact-footprint certificate."""
import time
import numpy as np


def tangent_ring(normals):
    tangent=np.cross(normals,[1.,0,0]);bad=np.linalg.norm(tangent,axis=1)<1e-8
    tangent[bad]=np.cross(normals[bad],[0.,0,1])
    tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    other=np.cross(normals,tangent);theta=np.arange(6)*np.pi/3
    return (np.cos(theta)[None,:,None]*tangent[:,None,:]
            +np.sin(theta)[None,:,None]*other[:,None,:])


class ModelLandingCheck:
    def __init__(self,query,normals,field,distance,gap=.3,min_up_dot=.15,footprint_radius=0.,contact_radius=None,minimum_contact_radius=.6,hard_min_up_dot=None,fallback_min_length=1.2):
        if not 0<=min_up_dot<1 or footprint_radius<0:raise ValueError('Invalid landing settings.')
        if hard_min_up_dot is not None and not 0<hard_min_up_dot<=min_up_dot:
            raise ValueError('Fallback hard minimum must be positive and no greater than the preferred threshold.')
        if fallback_min_length<0 or not np.isfinite(fallback_min_length):raise ValueError('Invalid fallback path length.')
        if contact_radius is not None and not 0<minimum_contact_radius<=contact_radius:
            raise ValueError('Require positive ordered contact radii.')
        self.query=query;self.normals=np.asarray(normals);self.field=field;self.distance=distance
        self.gap=gap;self.min_up_dot=min_up_dot;self.radius=footprint_radius
        self.contact_radius=contact_radius;self.minimum_contact_radius=minimum_contact_radius
        self.hard_min_up_dot=hard_min_up_dot;self.fallback_min_length=fallback_min_length
        self.results={}
        self.contact_radii={};self.backing_query_points=0
        self.queries=0;self.rejected=0;self.seconds=0.;self.escape_queries=0

    def __call__(self,points):
        start=time.perf_counter();points=np.asarray(points,dtype=float).reshape(-1,3)
        _,ids=self.query(points);n=self.normals[ids]
        _,g=self.field(points);g=np.asarray(g);length=np.linalg.norm(g,axis=1)
        up=g/np.maximum(length[:,None],1e-15);dot=np.sum(n*up,axis=1)
        threshold=self.min_up_dot if self.hard_min_up_dot is None else self.hard_min_up_dot
        ok=np.isfinite(dot)&(length>1e-8)&(dot>threshold)
        selected=np.flatnonzero(ok)
        if len(selected):
            u=up[selected];normal_dot=dot[selected]
            # The point below the gap must lie in model material, rather than
            # merely having a favourable nearest-triangle normal.
            below=points[selected]-u*((self.gap+.05)/normal_dot)[:,None]
            offsets=np.zeros((len(selected),1,3))
            if self.radius:offsets=np.concatenate((offsets,self.radius*tangent_ring(u)),axis=1)
            probes=below[:,None,:]+offsets
            self.backing_query_points+=len(selected)*offsets.shape[1]
            inside=np.asarray(self.distance(probes.reshape(-1,3))).reshape(probes.shape[:2])<0
            ok[selected]=inside.all(axis=1)
            if self.contact_radius is not None:
                # Contact-patch sizing is in the MODEL tangent plane. A layer-
                # plane disk only .05mm below its centre rejects legitimate
                # inclined planes, even with unlimited backing. This patch
                # screen does not certify the first printed layer's footprint.
                surface_normal=n[selected]/np.linalg.norm(n[selected],axis=1)[:,None]
                unit_ring=tangent_ring(surface_normal)
                # Three nested rings avoid accepting a wide contact whose
                # perimeter alone is backed while its interior crosses a hole.
                unresolved=np.flatnonzero(ok[selected]);chosen=np.zeros(len(selected))
                radius=self.contact_radius
                while len(unresolved):
                    stencil=unit_ring[unresolved,None,:,:]*np.array([1/3,2/3,1.])[None,:,None,None]*radius
                    probes=below[unresolved,None,None,:]+stencil
                    self.backing_query_points+=len(unresolved)*18
                    backed=(np.asarray(self.distance(probes.reshape(-1,3))).reshape(-1,18)<0).all(axis=1)
                    chosen[unresolved[backed]]=radius
                    unresolved=unresolved[~backed]
                    if radius<=self.minimum_contact_radius:break
                    radius=max(self.minimum_contact_radius,radius*.5)
                ok[selected] &= chosen>0
                for i in np.flatnonzero(chosen>0):
                    self.contact_radii[points[selected[i]].tobytes()]=float(chosen[i])
        for point,alignment,hard_ok in zip(points,dot,ok):
            self.results[point.tobytes()]=dict(achieved_up_dot=float(alignment),hard_checks_pass=bool(hard_ok))
        preferred_ok=ok&(dot>self.min_up_dot)
        self.queries+=len(points);self.rejected+=int((~preferred_ok).sum());self.seconds+=time.perf_counter()-start
        return preferred_ok

    def fallback_mask(self,points):
        if self.hard_min_up_dot is None:return np.zeros(len(points),dtype=bool)
        return np.asarray([self.results[p.tobytes()]['hard_checks_pass'] for p in np.asarray(points,dtype=float)])

    def alignment(self,point):return self.results[np.asarray(point,dtype=float).tobytes()]['achieved_up_dot']

    def outward(self,points):
        start=time.perf_counter()
        _,ids=self.query(points)
        self.escape_queries+=len(points);self.seconds+=time.perf_counter()-start
        return self.normals[ids]

    def report(self):
        return dict(candidate_checks=self.queries,rejected_checks=self.rejected,
                    escape_query_points=self.escape_queries,
                    backing_query_points=self.backing_query_points,
                    adaptive_contact_radius_mm=self.contact_radius,
                    minimum_contact_radius_mm=self.minimum_contact_radius,
                    adaptive_patch_basis='model_tangent',
                    checking_seconds=self.seconds,min_up_dot=self.min_up_dot,
                    preferred_min_up_dot=self.min_up_dot,hard_min_up_dot=self.hard_min_up_dot,
                    fallback_minimum_path_length_mm=self.fallback_min_length,
                    footprint_radius_mm=self.radius,
                    method='Upward face and backing centre; adaptive model-tangent contact-patch rings; optional fixed layer-plane ring.',
                    limitation='Sampled local test; branch-volume and first-layer printability are not certified.')
