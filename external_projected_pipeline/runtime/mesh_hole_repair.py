"""Boundary-constrained support-hole patches without an implicit scalar field.

Neelotpal Dutta. Additional author details may be added here.
"""
from pathlib import Path
import numpy as np
import shapely
from scipy.spatial import cKDTree


def boundary_patch(holes):
    """Triangulate each approved ring using only its own existing 3D vertices."""
    vertices=[];faces=[];offset=0
    for hole in holes:
        ring=np.asarray(hole['ring'],dtype=float)
        if ring.ndim!=2 or ring.shape[1]!=3 or not np.isfinite(ring).all():
            raise ValueError('Repair boundary requires finite 3D points.')
        if len(ring)>1 and np.array_equal(ring[0],ring[-1]):ring=ring[:-1]
        polygon=shapely.Polygon(ring[:,[0,2]])
        if not polygon.is_valid or polygon.is_empty or polygon.area<=1e-12:
            raise ValueError('Repair boundary is not a simple projected ring.')
        triangles=shapely.get_parts(shapely.constrained_delaunay_triangles(polygon))
        xyz=np.asarray([np.asarray(t.exterior.coords)[:3] for t in triangles if t.area>1e-12])
        if not len(xyz):raise ValueError('Repair boundary did not produce a patch.')
        distance,ids=cKDTree(ring[:,[0,2]]).query(xyz.reshape(-1,2))
        if np.max(distance)>1e-7:
            raise ValueError('Repair triangulation introduced a vertex without a boundary height.')
        # Boundary vertices and heights are retained exactly. Heights within
        # each triangle are subsequently barycentric, including path endpoints.
        local=ring[ids].reshape(-1,3,3)
        vertices.append(local.reshape(-1,3))
        faces.append(np.arange(len(local)*3).reshape(-1,3)+offset);offset+=len(local)*3
    return (np.vstack(vertices),np.vstack(faces)) if vertices else (np.empty((0,3)),np.empty((0,3),int))


class MeshRepairLifter:
    """Use source facets everywhere except the explicitly approved hole masks."""
    def __init__(self,base,mask,context,holes):
        from batch_generate_projected_hybrid_paths import ProjectedSurfaceLifter
        self.base,self.mask,self.context=base,mask,context
        self.mask_count=len(holes);self.points=0;self.clearance=float('inf')
        self.reference_mesh=boundary_patch(holes)
        self.patch=ProjectedSurfaceLifter.build(*self.reference_mesh) if holes else None
    def __getattr__(self,name):return getattr(self.base,name)

    def validate_new_points(self,points):
        from support_hole_repair import _context,_distances,_confirmed_inside
        points=np.asarray(points,float)
        if not len(points):return
        _,model=_context(self.context['settings'])
        distance=_distances(model,points)
        required=self.context['settings']['minimum_new_path_clearance_mm']
        if (distance<required).any() or _confirmed_inside(model,points).any():
            raise ValueError('Interpolated support-hole patch enters the model keepout.')
        self.clearance=min(self.clearance,float(distance.min()))

    def lift(self,points):
        points=np.asarray(points,float).reshape(-1,2)
        if self.patch is None:return self.base.lift(points)
        mask=shapely.covers(self.mask,shapely.points(points))
        if not mask.any():return self.base.lift(points)
        result=np.empty((len(points),3));ambiguous=snapped=0
        if (~mask).any():result[~mask],ambiguous,snapped=self.base.lift(points[~mask])
        result[mask],extra,near=self.patch.lift(points[mask])
        if extra:raise ValueError('Interpolated hole patch has conflicting heights.')
        self.validate_new_points(result[mask]);self.points+=int(mask.sum())
        return result,ambiguous+extra,snapped+near

    def normals(self,points,normals):
        if self.patch is None:return normals
        mask=shapely.covers(self.mask,shapely.points(points[:,[0,2]]))
        if mask.any():
            ids=self.patch.tree.nearest(shapely.points(points[mask][:,[0,2]]))
            xyz=self.patch.vertices[self.patch.faces[self.patch.face_indices[ids]]]
            normal=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
            normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-30)
            normal*=np.where(np.einsum('ij,ij->i',normal,normals[mask])<0,-1.,1.)[:,None]
            normals[mask]=normal
        return normals

    def report(self):
        return dict(method='mesh_interpolation',mask_count=self.mask_count,interpolated_points=self.points,
            field_lift_points=0,field_required=False,source_mesh_unchanged=True,
            minimum_model_distance_mm=self.clearance if np.isfinite(self.clearance) else None)


def save_reference(lifter,output_dir,stem):
    """Write the exact same piecewise-planar patch used for path heights."""
    vertices,faces=lifter.reference_mesh
    if not len(faces):return None
    triangles=vertices[faces]
    # Test the added surface too, including interior samples. Only this small
    # patch is sampled; original source faces are never reprojected or refitted.
    spacing=lifter.context['settings']['boundary_sample_mm']
    for _ in range(9):
        longest=np.linalg.norm(triangles-np.roll(triangles,1,axis=1),axis=2).max(axis=1)
        selected=longest>spacing
        if not selected.any():break
        if len(triangles)+3*int(selected.sum())>100_000:
            raise ValueError('Interpolated hole validation exceeded its bounded budget.')
        a,b,c=triangles[selected,0],triangles[selected,1],triangles[selected,2]
        ab,bc,ca=(a+b)/2,(b+c)/2,(c+a)/2
        children=np.stack((np.stack((a,ab,ca),axis=1),np.stack((ab,b,bc),axis=1),
                           np.stack((ca,bc,c),axis=1),np.stack((ab,bc,ca),axis=1)),axis=1).reshape(-1,3,3)
        triangles=np.concatenate((triangles[~selected],children))
    else:raise ValueError('Interpolated hole validation refinement did not converge.')
    lifter.validate_new_points(np.vstack((triangles.reshape(-1,3),triangles.mean(axis=1))))
    path=Path(output_dir)/(stem+'.repair_reference.obj');path.parent.mkdir(parents=True,exist_ok=True)
    # Store the original constrained triangulation, not a different resampled
    # fit; height references and emitted paths must describe identical facets.
    with path.open('w',encoding='utf8') as stream:
        np.savetxt(stream,vertices,fmt='v %.17g %.17g %.17g')
        np.savetxt(stream,faces+1,fmt='f %d %d %d')
    return str(path.resolve())
