"""Physical whole-face bounds for explicitly tolerated support-trim residuals.

Neelotpal Dutta. Additional author details may be added here.
"""
import numpy as np

NOISE=1e-6

def configured(value):
    if isinstance(value,(bool,np.bool_)):
        raise ValueError('Trim boundary tolerance must be a finite nonnegative distance in mm.')
    value=float(value)
    if not np.isfinite(value) or value<0:
        raise ValueError('Trim boundary tolerance must be a finite nonnegative distance in mm.')
    return value

def classify_faces(surface,geometry,*,include_vertices=False,vertex_values=None):
    """Retain an outlier only when its entire face is within a known-solid ball.

    For an inside vertex q, if every triangle vertex is within epsilon of q,
    convexity of a Euclidean ball puts the WHOLE triangle within epsilon of
    valid material. This is a conservative upper bound, not a field/SDF scaling
    assumption. Larger faces remain strict and continue through refinement.
    """
    budget=configured(getattr(geometry,'trim_boundary_tolerance_mm',0.))
    faces=np.asarray(surface.regular_faces);points=np.asarray(surface.points)
    xyz=points[faces]
    probes=np.concatenate((xyz.mean(axis=1)[:,None,:],(xyz+np.roll(xyz,-1,axis=1))/2),axis=1)
    values=np.asarray(geometry.evaluate(probes.reshape(-1,3))).reshape(-1,4)
    raw=(~np.isfinite(values)|(values>NOISE)).any(axis=1)
    if include_vertices:
        if vertex_values is None:vertex_values=np.asarray(geometry.evaluate(points))
        v=vertex_values[faces]
        raw|=(~np.isfinite(v)|(v>NOISE)).any(axis=1)
    accepted=np.zeros(len(faces),bool);bound=np.zeros(len(faces))
    if budget and raw.any():
        ids=np.flatnonzero(raw)
        if vertex_values is None:
            used,inverse=np.unique(faces[ids],return_inverse=True)
            vv=np.asarray(geometry.evaluate(points[used]))[inverse].reshape(-1,3)
        else:vv=vertex_values[faces[ids]]
        for start in range(0,len(ids),8192):
            chosen=ids[start:start+8192];tri=xyz[chosen];v=vv[start:start+8192]
            radii=np.linalg.norm(tri[:,:,None,:]-tri[:,None,:,:],axis=3).max(axis=2)
            radii[~(np.isfinite(v)&(v<=0))]=np.inf
            radius=radii.min(axis=1)
            # NaN means invalid geometry evaluation, not an outside sentinel.
            valid=(radius<=budget)&np.isfinite(tri).all(axis=(1,2))&(~np.isnan(values[chosen]).any(axis=1))
            accepted[chosen[valid]]=True;bound[chosen[valid]]=radius[valid]
    selected=values[accepted]
    finite=selected[np.isfinite(selected)]
    report=dict(tolerance_mm=budget,accepted_faces=int(accepted.sum()),
        maximum_deviation_bound_mm=float(bound.max(initial=0)),
        maximum_sampled_field_residual=float(finite.max(initial=0)),
        accepted_outside_sentinels=bool(np.isinf(selected).any()),
        remaining_bad_faces=int((raw&~accepted).sum()),
        policy='Retained whole triangles within a Euclidean ball around an original inside vertex; field residual is not used as distance.')
    return raw&~accepted,accepted,report

def record(surface,report):
    surface.field_data['trim_boundary_tolerance_mm']=[report['tolerance_mm']]
    surface.field_data['trim_accepted_faces']=[report['accepted_faces']]
    surface.field_data['trim_maximum_deviation_bound_mm']=[report['maximum_deviation_bound_mm']]
    surface.field_data['trim_geometry_approximation']=[bool(report['accepted_faces'])]
