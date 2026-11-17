"""Reusable router definitions promoted from batched_support_field.py; see promotion_sources.json."""
import collections

import time

import numpy as np

import igl

class BatchedSurfaceQuery:
    def __init__(self,mesh,original):
        self.original=original;self.vertices=np.asarray(mesh.points,dtype=float)
        self.faces=np.asarray(mesh.regular_faces,dtype=np.int64)
        self.triangles=self.vertices[self.faces];self.tree=igl.AABB();self.tree.init(self.vertices,self.faces)
        self.e0=self.triangles[:,1]-self.triangles[:,0];self.e1=self.triangles[:,2]-self.triangles[:,0]
        self.a=np.sum(self.e0*self.e0,axis=1);self.b=np.sum(self.e0*self.e1,axis=1);self.c=np.sum(self.e1*self.e1,axis=1)
        self.denom=self.a*self.c-self.b*self.b;self.stats=collections.Counter()
    def query(self,points):
        p=np.asarray(points,float).reshape(-1,3)
        if not len(p):return p.copy(),np.empty(0,int),np.empty(0,bool)
        _,ids,feet=self.tree.squared_distance(self.vertices,self.faces,p)
        d=feet-self.triangles[ids,0];d0=np.sum(d*self.e0[ids],axis=1);d1=np.sum(d*self.e1[ids],axis=1)
        u=(self.c[ids]*d0-self.b[ids]*d1)/self.denom[ids]
        v=(self.a[ids]*d1-self.b[ids]*d0)/self.denom[ids]
        edge=np.minimum(np.minimum(u,v),1.-u-v)<.01
        # Keep the established VTK tie-breaking on triangle edges/vertices.
        if edge.any():feet[edge],ids[edge]=self.original(p[edge])
        self.stats['points']+=len(p);self.stats['vtk_edge_tie_queries']+=int(edge.sum())
        return feet,ids,edge
    def __call__(self,points):
        feet,ids,_=self.query(points);return feet,ids

class BatchedBoundaryField:
    def __init__(self,original):
        self.original=original;self.mesh=original.mesh;self.native=original.native
        self.query=BatchedSurfaceQuery(self.mesh,original.query);self.stats=collections.Counter()
    def evaluate_scalars_and_gradients(self,points):
        p=np.asarray(points,float).reshape(-1,3);feet,ids,edge=self.query.query(p)
        # FindCell's tolerance can select an adjacent tetrahedron even near
        # the interior of a surface face. Per-face gradient substitution is
        # therefore unsafe; preserve the native owner for every query.
        values,gradient=self.native.evaluate_scalars_and_gradients(feet)
        values=values+np.sum(gradient*(p-feet),axis=1)
        self.stats['points']+=len(p);self.stats['native_tetra_queries']+=len(p)
        return values,gradient

def accelerate_guide(guide):
    started=time.perf_counter();affine=BatchedBoundaryField(guide.native)
    guide.native=affine;guide.project=lambda p:affine.query(p)[0]
    return dict(initialization_seconds=time.perf_counter()-started,faces=len(affine.query.faces),
                method='Batched libigl nearest faces; original native tetrahedral ownership and near-edge handling')
