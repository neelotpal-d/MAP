"""Immutable trilinear support scalar with analytic gradients in physical mm."""
from pathlib import Path
import numpy as np


def basis(points, origin, spacing, shape, derivatives=True):
    points=np.asarray(points,dtype=float).reshape(-1,3)
    if not np.isfinite(points).all():raise ValueError('Nonfinite support-field query.')
    p=(points-origin)/spacing
    if np.any(p < -1e-7) or np.any(p > np.asarray(shape)-1+1e-7):
        raise ValueError('Field query outside grid; extrapolation is forbidden.')
    cell=np.minimum(np.maximum(np.floor(p).astype(int),0),np.asarray(shape)-2)
    local=np.clip(p-cell,0.,1.)
    offsets=np.asarray([[i,j,k] for i in (0,1) for j in (0,1) for k in (0,1)])
    xyz=cell[:,None,:]+offsets
    ids=np.ravel_multi_index(xyz.transpose(2,0,1),shape)
    f=np.where(offsets[None,:,:],local[:,None,:],1-local[:,None,:]);weights=np.prod(f,axis=2)
    if not derivatives:return ids,weights,None
    deriv=np.empty((len(points),8,3))
    for a in range(3):
        other=[i for i in range(3) if i!=a]
        deriv[:,:,a]=(2*offsets[:,a]-1)*np.prod(f[:,:,other],axis=2)/spacing
    return ids,weights,deriv


class GridField:
    def __init__(self,data,values):
        self.origin=np.array(data['origin'],dtype=float,copy=True)
        self.spacing=float(data['spacing']);shape=np.asarray(data['shape'])
        if (self.origin.shape!=(3,) or shape.shape!=(3,) or not np.isfinite(self.origin).all()
                or not np.isfinite(shape).all() or np.any(shape!=np.rint(shape)) or np.any(shape<2)
                or not np.isfinite(self.spacing) or self.spacing<=0):
            raise ValueError('Require a finite 3D support grid with positive spacing and dimensions >=2.')
        self.shape=tuple(map(int,shape));self.values=np.array(values,dtype=float,copy=True).reshape(-1)
        if self.values.size!=np.prod(self.shape) or not np.isfinite(self.values).all():
            raise ValueError('Support grid scalar count/values are invalid.')
        self.origin.setflags(write=False);self.values.setflags(write=False)

    def evaluate_scalars_and_gradients(self,points):
        ids,w,d=basis(points,self.origin,self.spacing,self.shape);v=self.values[ids]
        return np.sum(v*w,axis=1),np.einsum('ni,nia->na',v,d)

    def evaluate_scalars(self,points):
        ids,w,_=basis(points,self.origin,self.spacing,self.shape,derivatives=False)
        return np.sum(self.values[ids]*w,axis=1)

    def volume(self):
        import pyvista as pv
        grid=pv.ImageData(dimensions=self.shape,spacing=(self.spacing,)*3,origin=self.origin)
        grid['scalar']=self.values.reshape(self.shape).ravel(order='F')
        return grid

    @classmethod
    def load(cls,folder):
        folder=Path(folder)
        with np.load(folder/'domain.npz') as d:meta={k:d[k].copy() for k in ('origin','spacing','shape')}
        with np.load(folder/'optimized_field.npz') as d:return cls(meta,d['values'])
