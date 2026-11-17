"""Model volume sampling and neural isosurface projection for meshes without tets."""
import numpy as np
import pyvista as pv
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance
from vtkmodules.vtkFiltersGeneral import vtkSampleImplicitFunctionFilter
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator
from vtkmodules.vtkCommonCore import reference


def native_volume(mesh,evaluator,spacing=1.,maximum_points=3_000_000):
    from .tree_geometry import sample_mesh_distance
    lower=np.asarray(mesh.bounds)[::2]-spacing
    upper=np.asarray(mesh.bounds)[1::2]+spacing
    shape=np.ceil((upper-lower)/spacing).astype(int)+1
    if np.prod(shape)>maximum_points:
        raise ValueError('Native model grid exceeds point limit; increase model spacing.')
    grid=pv.ImageData(dimensions=shape,spacing=(spacing,)*3,origin=lower)
    print(f'Native model: sampling {grid.n_points:,} grid points...',flush=True)
    function=vtkImplicitPolyDataDistance();function.SetInput(mesh)
    sampler=vtkSampleImplicitFunctionFilter();sampler.SetImplicitFunction(function);sampler.SetComputeGradients(False)
    grid['signed_model_distance']=sample_mesh_distance(function,sampler,grid.points)
    volume=grid.clip_scalar(scalars='signed_model_distance',value=0,invert=True)
    values=evaluator.evaluate_scalars(volume.points)
    volume['compatible_scalar']=values
    return volume


class ModelIsoProjector:
    def __init__(self,mesh,evaluator,query_backend='vtk',query_threads=8):
        self.mesh=mesh;self.evaluator=evaluator
        self.normals=np.asarray(mesh.cell_normals)
        self.bulk=None
        if query_backend=='libigl':
            from .mesh_queries import LibiglClosest
            self.bulk=LibiglClosest(mesh)
        elif query_backend=='open3d':
            from .mesh_queries import Open3DClosest
            self.bulk=Open3DClosest(mesh,threads=query_threads)
        elif query_backend=='vtk':
            self.locator=vtkStaticCellLocator();self.locator.SetDataSet(mesh);self.locator.BuildLocator()
        else:raise ValueError('Unknown model query backend.')

    def closest(self,points):
        if self.bulk is not None:
            closest,ids=self.bulk.query(points)
            return closest,self.normals[ids]
        closest=np.empty_like(points,dtype=float);ids=np.empty(len(points),dtype=int)
        cell=reference(0);sub=reference(0);distance=reference(0.)
        q=[0.,0.,0.]
        for i,p in enumerate(points):
            self.locator.FindClosestPoint(p,q,cell,sub,distance)
            closest[i]=q;ids[i]=int(cell)
        return closest,self.normals[ids]

    def project(self,surface,value):
        faces=surface.regular_faces
        edges=np.sort(np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
        unique,count=np.unique(edges,axis=0,return_counts=True)
        boundary=np.unique(unique[count==1])
        points=np.asarray(surface.points,dtype=float).copy()
        normals=None
        if len(boundary):
            points[boundary],normals=self.closest(points[boundary])
        for _ in range(4):
            scalar,gradient=self.evaluator.evaluate_scalars_and_gradients(points)
            if len(boundary):
                gradient[boundary]-=np.sum(gradient[boundary]*normals,axis=1)[:,None]*normals
            norm=np.sum(gradient*gradient,axis=1)
            step=np.divide(scalar-value,norm,out=np.zeros_like(norm),where=norm>1e-16)[:,None]*gradient
            lengths=np.linalg.norm(step,axis=1)
            step*=np.minimum(1.,.25/np.maximum(lengths,1e-15))[:,None]
            points-=step
            if len(boundary):
                points[boundary],normals=self.closest(points[boundary])
        result=surface.copy();result.points=points
        return result
