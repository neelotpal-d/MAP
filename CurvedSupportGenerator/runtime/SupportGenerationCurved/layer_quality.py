"""Layer diagnostics and bounded recontouring of the same trilinear field."""
import numpy as np


def mesh_quality(vertices, faces):
    vertices=np.asarray(vertices,dtype=float);faces=np.asarray(faces,dtype=np.int64)
    if faces.size==0:faces=np.empty((0,3),dtype=np.int64)
    if faces.ndim!=2 or faces.shape[1]!=3 or (len(faces) and (faces.min()<0 or faces.max()>=len(vertices))):
        raise ValueError('Layer quality requires valid triangle indices.')
    if not np.isfinite(vertices).all():raise ValueError('Layer mesh has nonfinite vertices.')
    if not len(faces):return dict(triangles=0,minimum_angle_degrees=None,shortest_edge_mm=None,faces_below_1_degree=0,degenerate_faces=0,ill_conditioned=False)
    tri=vertices[faces];u=tri[:,1]-tri[:,0];v=tri[:,2]-tri[:,0]
    area=np.linalg.norm(np.cross(u,v),axis=1)
    lengths=np.column_stack((np.linalg.norm(u,axis=1),np.linalg.norm(v,axis=1),np.linalg.norm(v-u,axis=1)))
    angles=np.column_stack((np.arctan2(area,np.einsum('ij,ij->i',u,v)),
        np.arctan2(area,np.einsum('ij,ij->i',-u,v-u)),np.arctan2(area,np.einsum('ij,ij->i',-v,u-v))))*180/np.pi
    minimum=float(angles.min());shortest=float(lengths.min())
    return dict(triangles=len(faces),minimum_angle_degrees=minimum,shortest_edge_mm=shortest,
        faces_below_1_degree=int((angles.min(axis=1)<1.).sum()),degenerate_faces=int((area<=1e-14).sum()),
        ill_conditioned=bool(minimum<1e-4 or shortest<1e-8))


def field_normal_consistency(vertices,faces,field):
    """Apply the planner's strict per-face gradient-sign/shared-edge contract."""
    vertices=np.asarray(vertices,dtype=float);faces=np.asarray(faces,dtype=np.int64)
    if not len(faces):return dict(valid=True,triangles=0,minimum_absolute_alignment=None,inconsistent_shared_edges=0,nonmanifold_edges=0)
    tri=vertices[faces];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(normal,axis=1)
    centers=tri.mean(axis=1)
    gradients=np.vstack([field.evaluate_scalars_and_gradients(centers[k:k+25000])[1] for k in range(0,len(faces),25000)])
    lengths=np.linalg.norm(gradients,axis=1)
    if not np.isfinite(gradients).all() or not np.isfinite(area).all() or np.any(area<=1e-14) or np.any(lengths<=1e-15):
        return dict(valid=False,triangles=len(faces),reason='Zero, nonfinite or degenerate gradient / triangle normal.')
    alignment=np.einsum('ij,ij->i',normal/area[:,None],gradients/lengths[:,None])
    oriented=faces.copy();flip=alignment<0;oriented[flip]=oriented[flip][:,[0,2,1]]
    edges=np.vstack((oriented[:,[0,1]],oriented[:,[1,2]],oriented[:,[2,0]]))
    _,inverse,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    directions=np.bincount(inverse,weights=np.where(edges[:,0]<edges[:,1],1.,-1.))
    conflicts=int(((counts==2)&(directions!=0)).sum());nonmanifold=int((counts>2).sum())
    minimum=float(np.abs(alignment).min());tangential=int((np.abs(alignment)<1e-6).sum())
    return dict(valid=not(conflicts or nonmanifold or tangential),triangles=len(faces),
        minimum_absolute_alignment=minimum,inconsistent_shared_edges=conflicts,
        nonmanifold_edges=nonmanifold,tangential_faces=tangential)


def refined_field_grid(field,factor,*,maximum_points=8_000_000):
    """Sample the original trilinear interpolant; do not optimize or smooth it."""
    import pyvista as pv
    if isinstance(factor,bool) or not isinstance(factor,int) or factor<1:raise ValueError('Field contour refinement factor must be a positive integer.')
    shape=(np.asarray(field.shape,dtype=np.int64)-1)*factor+1
    count=int(np.prod(shape))
    if count>maximum_points:raise ValueError(f'Finer layer contour needs {count} grid points, over the {maximum_points} point budget.')
    values=np.asarray(field.values).reshape(field.shape)
    for axis in range(3):
        p=np.arange(shape[axis],dtype=float)/factor;lower=np.minimum(p.astype(int),field.shape[axis]-2)
        weight=p-lower;broadcast=[1,1,1];broadcast[axis]=len(weight);weight=weight.reshape(broadcast)
        values=(1-weight)*np.take(values,lower,axis=axis)+weight*np.take(values,lower+1,axis=axis)
    grid=pv.ImageData(dimensions=tuple(map(int,shape)),spacing=(field.spacing/factor,)*3,origin=field.origin)
    grid['scalar']=values.ravel(order='F')
    return grid


def summarize(records):
    qualities=[r['mesh_quality'] for r in records if r.get('mesh_quality')]
    normals=[r['field_normal_consistency'] for r in records if r.get('field_normal_consistency')]
    retries=[dict(layer=r['layer'],**r['contour_source_refinement']) for r in records if r.get('contour_source_refinement',{}).get('factor',1)>1]
    return dict(layers_audited=len(qualities),ill_conditioned_layers=sum(bool(r['ill_conditioned']) for r in qualities),
        faces_below_1_degree=sum(r['faces_below_1_degree'] for r in qualities),
        field_normal_layers_checked=len(normals),field_normal_failures=sum(not r['valid'] for r in normals),
        recontoured_layers=retries,
        scope='Mesh quality and strict saved-gradient normal consistency. This is not a physical support-height, collision or continuous coverage certificate.')
