"""Contact-gap struts between scalar-ordered model surfaces."""
import numpy as np
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance
from scipy.spatial import cKDTree


def model_bridges(mesh,contacts,field,offset,plate,gap=.2,radius=2.5):
    function=vtkImplicitPolyDataDistance();function.SetInput(mesh)
    centers=contacts.points-offset*contacts.normals
    _,gradients=field.evaluate_scalars_and_gradients(centers)
    gradients/=np.linalg.norm(gradients,axis=1)[:,None]
    triangles=np.asarray(mesh.points)[mesh.regular_faces]
    face_centers=triangles.mean(axis=1)
    face_normals=np.asarray(mesh.cell_normals)
    locator=cKDTree(face_centers)
    records=[];remaining=[];tiny=[];base_faces=set()
    for i,(center,normal,gradient) in enumerate(zip(centers,contacts.normals,gradients)):
        if i%25==0:
            print(f'Model-contact bridges {i}/{len(centers)}',flush=True)
        # Start just outside the upper surface; follow descending print scalar.
        start=center+.025*normal
        axis=-gradient
        if axis[1]>=-1e-5:
            remaining.append(i);continue
        length=(plate-start[1])/axis[1]
        hit,cells=mesh.ray_trace(start,start+axis*length,first_point=True)
        if not len(hit):
            remaining.append(i);continue
        length=float(np.linalg.norm(hit-start))
        if length<1e-5:
            remaining.append(i);continue
        probes=start+np.linspace(0,length,33)[:,None]*axis
        distance=np.array([function.EvaluateFunction(p) for p in probes])
        inside=np.flatnonzero(distance>=gap)
        if not len(inside):
            tiny.append(dict(contact_index=i,reason='No room between surfaces after two contact gaps.',free_length_mm=length))
            continue
        # Find the two gap isosurfaces on the same scalar-gradient ray.
        def root(outside_t,inside_t):
            for _ in range(22):
                middle=(outside_t+inside_t)/2
                if function.EvaluateFunction(start+axis*middle)>=gap:
                    inside_t=middle
                else:
                    outside_t=middle
            return start+axis*inside_t
        first,last=int(inside[0]),int(inside[-1])
        upper=root(max(0,(first-1)*length/32),first*length/32)
        lower=root(min(length,(last+1)*length/32),last*length/32)
        scalar=field.evaluate_scalars(np.array([lower,upper]))
        if scalar[0]>=scalar[1]-1e-7 or (upper-lower)@gradient<=.02:
            remaining.append(i);continue
        ids=locator.query_ball_point(hit,radius+2.)
        floor_normal=face_normals[int(cells[0])]
        ids=np.asarray(ids,dtype=int)
        ids=ids[face_normals[ids]@floor_normal>.707]
        base_faces.update(ids);base_faces.add(int(cells[0]))
        records.append(dict(contact_index=i,lower=lower.tolist(),upper=upper.tolist(),normal=normal.tolist(),radius=radius,
            base_scalar=float(scalar[0]),tip_scalar=float(scalar[1]),length_mm=float(np.linalg.norm(upper-lower))))
    return records,np.asarray(remaining,dtype=int),tiny,np.asarray(sorted(base_faces),dtype=int)
