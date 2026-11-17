import numpy as np
from .distance_certificate import DistanceCertificate as OriginalCertificate

class GridDistanceCertificate(OriginalCertificate):
    def __init__(self,mesh,exact,**kwargs):
        super().__init__(mesh,exact,**kwargs)
        bounds=np.asarray(mesh.bounds).reshape(3,2)
        self.box_lower=bounds[:,0];self.box_upper=bounds[:,1]
        self.grid_origin=np.floor(self.box_lower/self.spacing)*self.spacing
        self.grid_shape=np.ceil((self.box_upper-self.grid_origin)/self.spacing).astype(int)+1
        xyz=np.indices(tuple(self.grid_shape)).reshape(3,-1).T*self.spacing+self.grid_origin
        values=[]
        for offset in range(0,len(xyz),20000):values.append(np.asarray(exact(xyz[offset:offset+20000]),float))
        self.grid_distances=np.concatenate(values).reshape(tuple(self.grid_shape))
        self.report.update(box_certified_points=0,distance_grid_certified_points=0,
            exact_distance_grid_nodes=len(xyz),grid_certificate='Signed-distance Lipschitz bound; float32 error margin. No distance interpolation.')

    def evaluate(self,points,band):
        points=np.asarray(points,dtype=float).reshape(-1,3)
        if not len(points):return np.empty(0,dtype=np.float32)
        if not np.isfinite(band) or band<0:raise ValueError('Invalid distance band.')
        result=np.empty(len(points),np.float32)
        displacement=np.maximum(self.box_lower-points,0)+np.maximum(points-self.box_upper,0)
        outside=np.linalg.norm(displacement,axis=1)>band+1e-8
        result[outside]=band+1.
        ids=np.flatnonzero(~outside)
        nodes=np.rint((points[ids]-self.grid_origin)/self.spacing).astype(int)
        in_grid=np.all((nodes>=0)&(nodes<self.grid_shape),axis=1)
        candidates=ids[in_grid];nodes=nodes[in_grid]
        d=self.grid_distances[tuple(nodes.T)]
        radius=np.linalg.norm(points[candidates]-(nodes*self.spacing+self.grid_origin),axis=1)
        error=4*np.finfo(np.float32).eps*np.maximum(1.,abs(d))+1e-6
        certified=abs(d)-radius-error>band
        accepted=candidates[certified]
        result[accepted]=np.sign(d[certified])*(band+1.)
        done=outside.copy();done[accepted]=True
        if (~done).any():result[~done]=super().evaluate(points[~done],band)
        self.report['box_certified_points']+=int(outside.sum())
        self.report['distance_grid_certified_points']+=len(accepted)
        self.report['queries']+=int(done.sum());self.report['certified_points']+=int(done.sum())
        return result
