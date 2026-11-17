"""Model-local print directions transitioning to a geometric vertical guide.

This is a routing direction field, NOT a reconstructed exterior scalar field.
"""
import numpy as np


class ContactDirectionGuide:
    direction_only=True

    def __init__(self,native,distance,project,direction=(0,1,0),inner_band=1.,outer_band=3.,scalar_band=None):
        if not 0<=inner_band<outer_band:raise ValueError('Require ordered contact bands.')
        scalar_band=inner_band if scalar_band is None else scalar_band
        if not np.isfinite(scalar_band) or not 0<=scalar_band<=inner_band:
            raise ValueError('Scalar descent band must lie inside the near-model direction band.')
        self.native=native;self.distance=distance;self.project=project
        self.up=np.asarray(direction,dtype=float);self.up/=np.linalg.norm(self.up)
        self.inner_band=inner_band;self.outer_band=outer_band
        self.scalar_band=float(scalar_band)
        self.cache={};self.evaluated=0;self.native_query_points=0;self.projected_points=0

    def __call__(self,points):
        points=np.asarray(points,dtype=float).reshape(-1,3)
        unique,inverse=np.unique(points,axis=0,return_inverse=True)
        keys=[p.tobytes() for p in unique];missing=[i for i,k in enumerate(keys) if k not in self.cache]
        if missing:
            q=unique[missing];d=self.distance(q);normal=np.tile(self.up,(len(q),1));phi=np.full(len(q),np.nan)
            near=d<self.outer_band
            if near.any():
                ids=np.flatnonzero(near);query=q[near].copy();projected=d[near]>self.inner_band
                if projected.any():
                    closest=self.project(query[projected])
                    offset=query[projected]-closest
                    # Clamp the gradient evaluation point to the near-model
                    # shell, continuously at inner_band. Jumping straight to
                    # the surface here would make a direction discontinuity.
                    query[projected]=closest+self.inner_band*offset/np.maximum(np.linalg.norm(offset,axis=1)[:,None],1e-15)
                values,gradient=self.native.evaluate_scalars_and_gradients(query)
                gradient=np.asarray(gradient,dtype=float);norm=np.linalg.norm(gradient,axis=1)
                unit=gradient/np.maximum(norm[:,None],1e-15)
                # Direction guidance may extend farther than the scalar
                # continuation can be trusted. A nearest-boundary Taylor field
                # can jump when the closest patch changes; increasing the
                # direction band must not silently widen scalar enforcement.
                local_phi=np.where(d[near]<=self.scalar_band,np.asarray(values,dtype=float),np.nan)
                phi[ids]=local_phi
                s=np.clip((d[near]-self.inner_band)/(self.outer_band-self.inner_band),0,1)
                s=s*s*(3-2*s)
                # Rotate directions, rather than blending scalar VALUES. This
                # avoids the extra spatial-weight derivative that stalled the
                # previous native/FEM scalar blend.
                cosine=np.clip(unit@self.up,-1,1);theta=np.arccos(cosine)
                tangent=self.up[None,:]-cosine[:,None]*unit
                tangent_norm=np.linalg.norm(tangent,axis=1)
                opposite=(tangent_norm<1e-8)&(cosine<0)
                if opposite.any():
                    tangent[opposite]=np.cross(unit[opposite],[1.,0,0])
                    bad=opposite&(np.linalg.norm(tangent,axis=1)<1e-8)
                    tangent[bad]=np.cross(unit[bad],[0.,0,1])
                tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-15)
                rotated=np.cos(theta*s)[:,None]*unit+np.sin(theta*s)[:,None]*tangent
                invalid=(norm<1e-8)|~np.isfinite(gradient).all(axis=1)
                rotated[invalid]=0
                normal[ids]=rotated
                self.native_query_points+=len(ids);self.projected_points+=int(projected.sum())
            for i,n,value in zip(missing,normal,phi):self.cache[keys[i]]=(unique[i]@self.up,n,value)
            self.evaluated+=len(missing)
        rank=np.asarray([self.cache[k][0] for k in keys])
        normal=np.asarray([self.cache[k][1] for k in keys]).reshape(-1,3)
        return rank[inverse],normal[inverse]

    def native_descent_mask(self,probes):
        """Check native scalar where adjacent samples are in the scalar band."""
        flat=np.asarray(probes,dtype=float).reshape(-1,3)
        self(flat)
        phi=np.asarray([self.cache[p.tobytes()][2] for p in flat]).reshape(probes.shape[:2])
        paired=np.isfinite(phi[:,:-1])&np.isfinite(phi[:,1:])
        return np.all(~paired|(np.diff(phi,axis=1)<-1e-9),axis=1)
