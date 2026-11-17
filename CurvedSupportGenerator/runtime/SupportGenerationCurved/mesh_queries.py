"""Optional compiled mesh queries; retain the existing VTK fallback."""
import numpy as np


def import_open3d():
    import sys
    from pathlib import Path
    from runtimeEnv import configure_openmp_runtime
    configure_openmp_runtime()
    local=Path(__file__).resolve().parent/'.deps'
    if local.is_dir() and str(local) not in sys.path:sys.path.append(str(local))
    import open3d
    return open3d


class Open3DClosest:
    """Reusable Embree-backed scene, with an explicit CPU thread budget."""
    def __init__(self,mesh,threads=8):
        if threads<1:raise ValueError('Query thread count must be positive.')
        self.o3d=import_open3d();self.threads=int(threads)
        self.vertices=np.asarray(mesh.points,dtype=float)
        self.faces=np.asarray(mesh.regular_faces,dtype=np.int64)
        flat=self.faces.ravel();order=np.argsort(flat,kind='stable')
        self.incident=(order//3).astype(np.int32)
        self.offsets=np.r_[0,np.cumsum(np.bincount(flat,minlength=len(self.vertices)))]
        self.scene=self.o3d.t.geometry.RaycastingScene(nthreads=self.threads)
        self.scene.add_triangles(self.o3d.core.Tensor(np.asarray(mesh.points,dtype=np.float32)),
            self.o3d.core.Tensor(np.asarray(mesh.regular_faces,dtype=np.uint32)))

    def query(self,points):
        points=np.asarray(points,dtype=float).reshape(-1,3)
        if not len(points):return np.empty((0,3)),np.empty(0,dtype=np.int64)
        result=self.scene.compute_closest_points(self.o3d.core.Tensor(points.astype(np.float32)),nthreads=self.threads)
        ids=result['primitive_ids'].numpy().astype(np.int64)
        if np.any(ids>=len(self.faces)):raise ValueError('Invalid Open3D closest triangle.')
        # Float32 BVH lookup can select the neighboring side of an almost tied
        # edge. Search its vertex one-ring in float64 before choosing the point
        # and face normal used by neural projection.
        vertices=self.faces[ids].ravel()
        counts=self.offsets[vertices+1]-self.offsets[vertices]
        starts=self.offsets[vertices]
        entries=np.repeat(starts,counts)+np.arange(counts.sum())-np.repeat(np.cumsum(counts)-counts,counts)
        candidates=self.incident[entries]
        rows=np.repeat(np.repeat(np.arange(len(points)),3),counts)
        closest=closest_on_triangles(points[rows],self.vertices[self.faces[candidates]])
        squared=np.sum((closest-points[rows])**2,axis=1)
        group_starts=np.r_[0,np.cumsum(counts.reshape(-1,3).sum(axis=1))[:-1]]
        minimum=np.minimum.reduceat(squared,group_starts)
        chosen=np.minimum.reduceat(np.where(squared==minimum[rows],np.arange(len(rows)),len(rows)),group_starts)
        return closest[chosen],candidates[chosen].astype(np.int64)


def closest_on_triangles(points,triangles):
    a=triangles[:,0];ab=triangles[:,1]-a;ac=triangles[:,2]-a
    normal=np.cross(ab,ac);square=np.sum(normal*normal,axis=1)
    height=np.divide(np.sum((points-a)*normal,axis=1),square,out=np.zeros(len(points)),where=square>0)
    projected=points-height[:,None]*normal;offset=projected-a
    u=np.divide(np.sum(np.cross(offset,ac)*normal,axis=1),square,out=np.full(len(points),-1.),where=square>0)
    v=np.divide(np.sum(np.cross(ab,offset)*normal,axis=1),square,out=np.full(len(points),-1.),where=square>0)
    outside=(u<0)|(v<0)|(u+v>1)
    if np.any(outside):
        tri=triangles[outside];starts=tri;delta=np.roll(tri,-1,axis=1)-tri
        lengths=np.sum(delta*delta,axis=2)
        t=np.divide(np.sum((points[outside,None]-starts)*delta,axis=2),lengths,out=np.zeros_like(lengths),where=lengths>0)
        candidates=starts+np.clip(t,0,1)[:,:,None]*delta
        which=np.argmin(np.sum((candidates-points[outside,None])**2,axis=2),axis=1)
        projected[outside]=candidates[np.arange(len(which)),which]
    return projected


def arrays(mesh):
    # Eigen's column-major layout avoids repeated input-layout conversion.
    return np.asfortranarray(mesh.points,dtype=np.float64),np.asfortranarray(mesh.regular_faces,dtype=np.int64)


def bulk_signed_distance(mesh,points):
    """One large call: libigl constructs its signing data/tree for this call."""
    import igl
    points=np.asfortranarray(np.asarray(points,dtype=float).reshape(-1,3))
    if not len(points):return np.empty(0,dtype=np.float32)
    vertices,faces=arrays(mesh)
    values=igl.signed_distance(points,vertices,faces,igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)[0]
    if not np.isfinite(values).all():raise ValueError('Nonfinite libigl signed distance.')
    return np.asarray(values,dtype=np.float32)


class LibiglClosest:
    """Build one AABB and reuse it for all boundary projection batches."""
    def __init__(self,mesh):
        import igl
        self.vertices,self.faces=arrays(mesh)
        self.tree=igl.AABB();self.tree.init(self.vertices,self.faces)

    def query(self,points):
        points=np.asfortranarray(np.asarray(points,dtype=float).reshape(-1,3))
        if not len(points):return np.empty((0,3)),np.empty(0,dtype=np.int64)
        squared,ids,closest=self.tree.squared_distance(self.vertices,self.faces,points)
        if np.any(ids<0) or not np.isfinite(squared).all():raise ValueError('Invalid libigl closest-point query.')
        return closest,ids
