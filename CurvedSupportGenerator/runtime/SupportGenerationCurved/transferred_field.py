"""Continuous near-model neural band joined to an exterior FEM scalar field."""
import numpy as np
import pyvista as pv
from vtkmodules.vtkFiltersCore import vtkProbeFilter
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator
from .tree_geometry import sample_mesh_distance


class TransferredSupportField:
    def __init__(self,domain,native,geometry,inner_band=1.,outer_band=4.,mode='native_blend'):
        if not 0<=inner_band<outer_band:
            raise ValueError('Require 0 <= inner band < outer band.')
        if mode not in ('native_blend','relaxed'):raise ValueError('Unknown support scalar field mode.')
        self.mode=mode
        self.domain=domain
        self.boundary=domain.extract_surface(algorithm='dataset_surface').triangulate()
        self.native=native
        self.geometry=geometry
        self.inner_band,self.outer_band=inner_band,outer_band
        self.probe_points=pv.PolyData()
        self.probe=vtkProbeFilter()
        self.probe.SetSourceData(domain)
        self.probe.SetCellLocatorPrototype(vtkStaticCellLocator())
        self.probe.SetInputData(self.probe_points)
        self.probe.ComputeToleranceOff();self.probe.SetTolerance(1e-6)
        self.report=dict(source='Native neural field in near-model band; smoothly joined to reconstructed exterior scalar.',
            mode=mode,
            inner_band_mm=inner_band,outer_band_mm=outer_band,far_support_points_outside_domain=0,
            queried_points=0,boundary_extension_points=0,exterior_query_points=0,native_query_points=0)
        if mode=='relaxed':
            self.report.update(source='Relaxed reconstructed scalar directly; no native scalar blend.',
                               inner_band_mm=None,outer_band_mm=None)

    def boundary_values(self,points):
        ids,closest=self.boundary.find_closest_cell(points,return_closest_point=True)
        faces=self.boundary.regular_faces[ids]
        triangles=np.asarray(self.boundary.points)[faces]
        a=triangles[:,1]-triangles[:,0];b=triangles[:,2]-triangles[:,0];v=closest-triangles[:,0]
        aa=np.sum(a*a,axis=1);ab=np.sum(a*b,axis=1);bb=np.sum(b*b,axis=1)
        va=np.sum(v*a,axis=1);vb=np.sum(v*b,axis=1);den=aa*bb-ab*ab
        u=np.divide(bb*va-ab*vb,den,out=np.zeros_like(den),where=np.abs(den)>1e-24)
        w=np.divide(aa*vb-ab*va,den,out=np.zeros_like(den),where=np.abs(den)>1e-24)
        values=np.asarray(self.boundary['scalar'])[faces]
        return (1-u-w)*values[:,0]+u*values[:,1]+w*values[:,2]

    def evaluate_scalars(self,points):
        points=np.asarray(points,dtype=float).reshape(-1,3)
        if not len(points):
            return np.empty(0)
        distance=(np.full(len(points),np.inf) if self.mode=='relaxed' else
                  (self.geometry.bounded_distance(points,self.outer_band) if hasattr(self.geometry,'bounded_distance')
                   else self.geometry.distance(points)).astype(float))
        exterior=np.zeros(len(points))
        needed=distance>self.inner_band
        if needed.any():
            self.probe_points.points=points[needed]
            self.probe.Update()
            query=pv.wrap(self.probe.GetOutput())
            valid=np.asarray(query['vtkValidPointMask'],dtype=bool)
            sampled=np.asarray(query['scalar'],dtype=float).copy()
            indices=np.flatnonzero(needed)
            invalid=~valid
            if invalid.any():
                bad=indices[invalid]
                sampled[invalid]=self.boundary_values(points[bad])
                self.report['boundary_extension_points']+=int(invalid.sum())
                far=bad[distance[bad]>=self.outer_band]
                if len(far):
                    violations=int(np.count_nonzero(self.geometry.evaluate(points[far])<=0))
                    self.report['far_support_points_outside_domain']+=violations
                    if violations:
                        raise ValueError(f'{violations} support points lie outside the transfer domain; enlarge the tree envelope.')
            exterior[needed]=sampled
            self.report['exterior_query_points']+=int(needed.sum())
        near=distance<self.outer_band
        if near.any():
            native=self.native.evaluate_scalars(points[near])
            s=np.clip((distance[near]-self.inner_band)/(self.outer_band-self.inner_band),0,1)
            weight=1-s*s*(3-2*s)
            exterior[near]=weight*native+(1-weight)*exterior[near]
            self.report['native_query_points']+=int(near.sum())
        self.report['queried_points']+=len(points)
        if not np.isfinite(exterior).all():
            raise ValueError('Transferred support field contains nonfinite values.')
        return exterior
