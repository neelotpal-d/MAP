"""Split point-touching surface fans without changing any triangle geometry."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def surface_components(vertices,faces):
    vertices=np.asarray(vertices,dtype=np.float64);faces=np.asarray(faces,dtype=np.int64)
    if not len(faces):return
    if faces.ndim!=2 or faces.shape[1]!=3 or faces.min()<0 or faces.max()>=len(vertices):
        raise ValueError('Surface patch extraction requires valid triangle indices.')
    n=len(faces)
    edges=np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1)
    unique,inverse,counts=np.unique(edges,axis=0,return_inverse=True,return_counts=True)
    if np.any(counts>2):
        raise ValueError('Surface has an edge shared by more than two triangles; repair its non-manifold sheets before planning.')
    order=np.argsort(inverse,kind='stable');starts=np.r_[0,np.cumsum(counts)[:-1]]
    paired=np.flatnonzero(counts==2);owners=np.tile(np.arange(n),3)
    a=owners[order[starts[paired]]];b=owners[order[starts[paired]+1]]
    graph=coo_matrix((np.ones(2*len(a),dtype=bool),(np.r_[a,b],np.r_[b,a])),shape=(n,n)).tocsr()
    count,labels=connected_components(graph,directed=False)
    boundary_degree=np.bincount(unique[counts==1].ravel(),minlength=len(vertices))
    if np.any(boundary_degree>2):
        # Distinct local triangle fans may belong to the same larger component.
        # Connect matching corners only across a shared manifold edge, then
        # duplicate the original coordinate at each disconnected vertex fan.
        rows=[];cols=[]
        for endpoint in (0,1):
            v=unique[paired,endpoint]
            ca=3*a+np.argmax(faces[a]==v[:,None],axis=1)
            cb=3*b+np.argmax(faces[b]==v[:,None],axis=1)
            rows.extend((ca,cb));cols.extend((cb,ca))
        corner_graph=coo_matrix((np.ones(4*len(a),dtype=bool),(np.concatenate(rows),np.concatenate(cols))),shape=(3*n,3*n)).tocsr()
        _,fans=connected_components(corner_graph,directed=False)
        _,first=np.unique(fans,return_index=True)
        vertices=vertices[faces.ravel()[first]]
        faces=fans.reshape(-1,3)
    # Compact each edge-connected patch separately, retaining source face order,
    # winding and exact coordinates. No welding, displacement or face deletion.
    for label in range(count):
        selected=faces[labels==label]
        used,local=np.unique(selected.ravel(),return_inverse=True)
        yield vertices[used].copy(),local.reshape(-1,3)

def solve_unsigned_boundary_field(vertices,faces,*,backend,tolerance):
    """Keep intrinsic edge sources when geometric stitching rejoins a pinch."""
    from test_hybrid_contours import solve_hybrid_boundary_field as original
    try:
        return original(vertices,faces,backend=backend,tolerance=tolerance)
    except ValueError as error:
        if backend!='potpourri_fast_marching_unsigned' or 'Plane intersection graph is branching' not in str(error):raise
    from test_hybrid_contours import build_boundary_curve_segments,generate_boundary_loops,pp3d
    from test_potpourri3d_signed_heat_plane import curve_to_potpourri_payload
    loops=generate_boundary_loops(vertices,faces)
    segments=build_boundary_curve_segments(vertices,faces,tolerance=tolerance)
    if not segments:raise ValueError('No intrinsic boundary edges for unsigned distance sources.')
    # The unsigned solver already receives separate face-local two-point curves.
    # Their zero-distance union does not require a global geometric ordering.
    # Retain face/barycentric identities rather than welding touching fans.
    curves=[list(segment) for segment in segments]
    payloads=[curve_to_potpourri_payload(curve) for curve in curves]
    solver=pp3d.MeshFastMarchingDistanceSolver(vertices,faces)
    distance=np.asarray(solver.compute_distance(payloads,[[0.,0.]]*len(payloads),False),dtype=np.float64)
    if not np.isfinite(distance).all():raise ValueError('Nonfinite intrinsic unsigned boundary distance.')
    print('WARNING: Kept touching boundary fans as separate intrinsic zero-distance edge sources; no surface geometry changed.',flush=True)
    return distance,loops,curves,segments

def signed_fast_marching_field(vertices,faces,plane,*,tolerance=1e-8):
    """Matrix-free signed distance on the same surface and source plane."""
    from test_hybrid_contours import intersect_mesh_with_axis_plane,pp3d
    from test_potpourri3d_signed_heat_plane import curve_to_potpourri_payload
    segments=intersect_mesh_with_axis_plane(vertices,faces,2,plane,tolerance=tolerance)
    if not segments:raise RuntimeError('Plane/mesh intersection produced no segments.')
    payloads=[]
    for a,b in segments:
        # Orient each intrinsic face-local segment consistently with the source
        # plane and mesh winding; geometric welding would rejoin touching fans.
        triangle=vertices[faces[a.face_index]]
        normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])
        tangent=np.cross(normal,np.array([0.,0.,1.]))
        alignment=float(np.dot(b.xyz-a.xyz,tangent))
        if alignment==0:raise ValueError('Source plane segment orientation is degenerate.')
        if alignment<0:a,b=b,a
        payloads.append(curve_to_potpourri_payload([a,b]))
    solver=pp3d.MeshFastMarchingDistanceSolver(vertices,faces)
    field=np.asarray(solver.compute_distance(payloads,[[0.,0.]]*len(payloads),True),dtype=np.float64)
    if not np.isfinite(field).all():raise ValueError('Nonfinite signed fast-marching field.')
    return field
