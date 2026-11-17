"""Conservative surface-distance bounds with exact cached sign queries."""
import numpy as np
import time
from scipy.spatial import cKDTree
from .array_uniqueness import unique_rows_inverse


class DistanceCertificate:
    """Cap only distances proved farther than the caller's requested band.

    Every original triangle is covered by balls of radius R. Thus
    nearest_center_distance - R is a lower bound on distance to the surface.
    A nearby lattice node inside that empty ball has the same inside/outside
    sign as the query. Node signs come from the original exact mesh evaluator.
    No interpolated distance or coarse occupancy approximation is used.
    """
    def __init__(self,mesh,exact,radius=.5,spacing=2.,maximum_centers=5_000_000):
        started=time.perf_counter()
        self.exact=exact;self.spacing=float(spacing)
        step=float(radius)/2.
        self.radius=float(radius)+np.sqrt(3.)*step/2.
        self.signs={}
        self.report=dict(queries=0,certified_points=0,exact_points=0,sign_node_queries=0,
            covering_radius_mm=self.radius,sign_lattice_spacing_mm=self.spacing)
        triangles=np.asarray(mesh.points,dtype=float)[mesh.regular_faces]
        lower=np.floor(np.asarray(mesh.bounds)[::2]/step).astype(np.int64)-1
        shape=np.ceil(np.asarray(mesh.bounds)[1::2]/step).astype(np.int64)-lower+2
        keys=[]
        def collect(points):
            ids=np.rint(points/step).astype(np.int64)-lower
            keys.append(np.unique(np.ravel_multi_index(ids.T,shape)))
        centers=triangles.mean(axis=1)
        small=np.max(np.linalg.norm(triangles-centers[:,None,:],axis=2),axis=1)<=radius
        collect(centers[small])
        coarse=triangles[~small]
        # Sample rows parallel to each triangle's longest edge. Their spacing
        # perpendicular to that edge is <= sqrt(3)*r/2; samples on a row are
        # <= r apart. Projection onto the wider row below proves covering
        # distance <= r. Snapping adds at most sqrt(3)*step/2 to that bound.
        for start in range(0,len(coarse),512):
            tri=coarse[start:start+512]
            edges=np.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,1],tri[:,0]-tri[:,2]),axis=1)
            longest=np.argmax(np.sum(edges*edges,axis=2),axis=1);rows=np.arange(len(tri))
            a=tri[rows,longest];b=tri[rows,(longest+1)%3];c=tri[rows,(longest+2)%3]
            length=np.linalg.norm(b-a,axis=1)
            height=np.linalg.norm(np.cross(b-a,c-a),axis=1)/length
            divisions=np.maximum(1,np.ceil(height/(np.sqrt(3.)*radius/2)).astype(int))
            counts=divisions+1;ids=np.repeat(rows,counts)
            row_number=np.arange(counts.sum())-np.repeat(np.cumsum(counts)-counts,counts)
            t=row_number/divisions[ids]
            left=a[ids]+t[:,None]*(c[ids]-a[ids]);right=b[ids]+t[:,None]*(c[ids]-b[ids])
            segments=np.ceil(np.linalg.norm(right-left,axis=1)/radius).astype(int)
            sizes=segments+1;which=np.repeat(np.arange(len(left)),sizes)
            number=np.arange(sizes.sum())-np.repeat(np.cumsum(sizes)-sizes,sizes)
            fraction=np.divide(number,segments[which],out=np.zeros(len(which)),where=segments[which]>0)
            collect(left[which]+fraction[:,None]*(right[which]-left[which]))
        unique=np.unique(np.concatenate(keys))
        if len(unique)>maximum_centers:raise ValueError('Surface distance certificate exceeds its point budget.')
        positions=(np.column_stack(np.unravel_index(unique,shape))+lower)*step
        self.centers=cKDTree(positions)
        self.report['covering_centers']=self.centers.n
        self.report['initialization_seconds']=time.perf_counter()-started

    def evaluate(self,points,band):
        points=np.asarray(points,dtype=float).reshape(-1,3)
        if not len(points):return np.empty(0,dtype=np.float32)
        if not np.isfinite(band) or band<0:raise ValueError('Distance band must be finite and nonnegative.')
        lower=self.centers.query(points)[0]-self.radius
        nodes=np.rint(points/self.spacing).astype(np.int64)
        positions=nodes*self.spacing
        displacement=np.linalg.norm(positions-points,axis=1)
        certified=lower>np.maximum(band,displacement)+1e-8
        result=np.empty(len(points),dtype=np.float32)
        if np.any(certified):
            unique,inverse=unique_rows_inverse(nodes[certified])
            keys=[node.tobytes() for node in unique]
            missing=np.array([i for i,key in enumerate(keys) if key not in self.signs],dtype=int)
            if len(missing):
                values=self.exact(unique[missing]*self.spacing)
                for i,value in zip(missing,values):self.signs[keys[i]]=1 if value>=0 else -1
                self.report['sign_node_queries']+=len(missing)
            result[certified]=np.array([self.signs[key] for key in keys],dtype=np.float32)[inverse]*(band+1.)
        if np.any(~certified):result[~certified]=self.exact(points[~certified])
        self.report['queries']+=len(points)
        self.report['certified_points']+=int(certified.sum())
        self.report['exact_points']+=int((~certified).sum())
        return result
