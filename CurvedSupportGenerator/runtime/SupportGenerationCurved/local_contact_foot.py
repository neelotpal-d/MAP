"""Experimental local contact loft and sampled first-layer growth checks.

No production skeleton/rooting flags are changed by this module.
"""
import time
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


def trunk_connection(skeleton,node,fraction=.75):
    edges=skeleton.lines.reshape(-1,3)[:,1:];children={}
    for upper,lower in edges:children.setdefault(int(lower),[]).append(int(upper))
    path=[int(node)];seen={int(node)}
    while path[-1] in children:
        incoming=children[path[-1]]
        if len(incoming)!=1:break
        other=incoming[0]
        if other in seen:raise ValueError('Cycle in saved skeleton.')
        path.append(other);seen.add(other)
    points=skeleton.points[path];lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
    total=float(lengths.sum())
    if total<.1:raise ValueError('Not enough trunk length for a local foot experiment.')
    target=fraction*total;i=min(int(np.searchsorted(np.cumsum(lengths),target)),len(lengths)-1)
    before=float(lengths[:i].sum())
    return points[i]+(points[i+1]-points[i])*(target-before)/lengths[i],total


def loft_foot(patch,endpoint,connection,up,radius=1.2,steps=6):
    """Loft existing patch connectivity to a section contained in the trunk."""
    up=np.asarray(up,dtype=float);up/=np.linalg.norm(up)
    bottom=np.asarray(patch.points,dtype=float)
    radial=bottom-np.asarray(endpoint);radial-=(radial@up)[:,None]*up
    scale=min(1.,radius/max(np.linalg.norm(radial,axis=1).max(),1e-15))
    top=np.asarray(connection)+scale*radial
    t=np.linspace(0,1,steps+1)
    points=((1-t[:,None,None])*bottom[None,:,:]+t[:,None,None]*top[None,:,:]).reshape(-1,3)
    faces=np.sort(patch.regular_faces,axis=1);n=len(bottom);tetra=[]
    for layer in range(steps):
        a,b,c=(faces+layer*n).T;aa,bb,cc=a+n,b+n,c+n
        tetra.extend((np.column_stack((a,b,c,cc)),np.column_stack((a,b,bb,cc)),np.column_stack((a,aa,bb,cc))))
    tetra=np.vstack(tetra)
    p=points[tetra];det=np.linalg.det(np.stack((p[:,1]-p[:,0],p[:,2]-p[:,0],p[:,3]-p[:,0]),axis=1))
    valid=np.abs(det)>1e-12
    if not valid.all():raise ValueError('Degenerate tetrahedra in contact loft.')
    reverse=det<0
    tetra[reverse,2:4]=tetra[reverse,2:4][:,::-1]
    cells=np.column_stack((np.full(len(tetra),4),tetra))
    volume=pv.UnstructuredGrid(cells.ravel(),np.full(len(tetra),pv.CellType.TETRA,dtype=np.uint8),points)
    return volume,dict(top_section_maximum_radius_mm=float(np.linalg.norm(top-connection,axis=1).max()),
        trunk_radius_mm=radius,loft_tetrahedra=len(tetra),loft_points=len(points),
        limitation='Top section is contained in trunk; full trunk cross-section transition is not certified.')


def previous_layer_backing(points,up,previous,angle=45.,tolerance=.03):
    """Sampled cone reachability to the previous continuous triangle surface."""
    points=np.asarray(points);up=np.asarray(up)
    if previous is None or not previous.n_cells:return np.zeros(len(points),dtype=bool)
    _,closest=previous.find_closest_cell(points,return_closest_point=True)
    _,ids=cKDTree(previous.points).query(points,k=min(16,previous.n_points))
    if ids.ndim==1:ids=ids[:,None]
    candidates=np.concatenate((np.asarray(closest)[:,None,:],previous.points[ids]),axis=1)
    delta=points[:,None,:]-candidates
    height=np.sum(delta*up[:,None,:],axis=2)
    lateral=np.linalg.norm(delta-height[:,:,None]*up[:,None,:],axis=2)
    return ((height>1e-7)&(lateral<=height*np.tan(np.deg2rad(angle))+tolerance)).any(axis=1)


def check_layers(volume,native,ray_query,distance,schedule,gap,min_dot,maximum_layers=5):
    scalar=np.asarray(volume['scalar']);values=np.asarray(schedule).reshape(-1)
    selected=values[(values>scalar.min()+1e-8)&(values<scalar.max()-1e-8)][:maximum_layers]
    previous=None;layers=[];reports=[]
    for index,value in enumerate(selected):
        layer=volume.contour([float(value)],scalars='scalar').triangulate()
        if not layer.n_cells:continue
        points=np.vstack((layer.points,layer.cell_centers().points))
        actual,gradient=native.evaluate_scalars_and_gradients(points)
        gradient=np.asarray(gradient);up=gradient/np.maximum(np.linalg.norm(gradient,axis=1)[:,None],1e-15)
        travel,normal=ray_query(points,-up)
        hit_valid=np.isfinite(travel)&(travel>=0)
        hit=np.flatnonzero(hit_valid);model_ok=np.zeros(len(points),dtype=bool);model_reasons={}
        if len(hit):
            dot=np.sum(normal[hit]*up[hit],axis=1)
            model_phi=native.evaluate_scalars(points[hit]-travel[hit,None]*up[hit])
            facing=dot>min_dot;near=travel[hit]<=(gap+.05)/np.maximum(dot,.1);ordered=model_phi<actual[hit]-1e-8
            model_ok[hit]=facing&near&ordered
            model_reasons=dict(ray_hits=len(hit),facing_hits=int(facing.sum()),near_gap_hits=int(near.sum()),
                earlier_scalar_hits=int(ordered.sum()),minimum_hit_travel_mm=float(travel[hit].min()),
                maximum_hit_travel_mm=float(travel[hit].max()))
        previous_ok=previous_layer_backing(points,up,previous)
        unsupported=~(model_ok|previous_ok)
        signed=distance(points)
        valid_sample=(~unsupported)&(signed>=gap-.005)
        face_ok=valid_sample[:layer.n_points][layer.regular_faces].all(axis=1)&valid_sample[layer.n_points:]
        reports.append(dict(local_layer=index,scalar_value=float(value),triangles=layer.n_cells,
            sampled_points=len(points),model_backed_samples=int(model_ok.sum()),
            previous_layer_backed_samples=int(previous_ok.sum()),unsupported_samples=int(unsupported.sum()),
            model_backing_details=model_reasons,validated_triangles=int(face_ok.sum()),
            unsupported_fraction=float(unsupported.mean()),minimum_sampled_gap_mm=float(signed.min()),
            maximum_scalar_residual=float(np.max(np.abs(actual-value)))))
        layer.cell_data['local_layer']=np.full(layer.n_cells,index,dtype=int)
        layer.cell_data['validated_growth']=face_ok.astype(np.uint8)
        layers.append(layer)
        previous=layer.extract_cells(np.flatnonzero(face_ok)).extract_surface(algorithm='dataset_surface') if face_ok.any() else None
    return layers,dict(checked_layers=len(reports),layers=reports,
        sampled_growth_pass=bool(reports) and all(r['unsupported_samples']==0 for r in reports),
        sampled_gap_pass=bool(reports) and all(r['minimum_sampled_gap_mm']>=gap-.005 for r in reports),
        growth_angle_degrees=45.,lateral_sampling_tolerance_mm=.03,model_gap_allowance_mm=.05,
        limitation='Finite vertices/centroids and nearest-surface candidates; first few native-field layers only, not a full solid/printing certificate.')


def probe_foot(patch,skeleton,node,native,ray_query,distance,schedule,gap,min_dot,radius=1.2):
    started=time.perf_counter();endpoint=skeleton.points[node]
    connection,total=trunk_connection(skeleton,node)
    phi,g=native.evaluate_scalars_and_gradients(connection[None,:]);up=g[0]/np.linalg.norm(g[0])
    threshold=float(phi[0]-.08*np.linalg.norm(g[0]))
    base=patch.clip_scalar(value=threshold,invert=True,scalars='scalar').triangulate()
    if not base.n_cells:return None,[],dict(candidate_found=False,reason='No patch below available trunk connection.')
    base=base.connectivity('closest',closest_point=endpoint).extract_surface(algorithm='dataset_surface').triangulate().clean()
    _,closest=base.find_closest_cell(endpoint,return_closest_point=True)
    attachment=float(np.linalg.norm(endpoint-closest))
    if attachment>.15:return None,[],dict(candidate_found=False,reason='Trimmed patch no longer reaches the saved endpoint.',attachment_distance_mm=attachment)
    try:volume,sizing=loft_foot(base,endpoint,connection,up,radius)
    except ValueError as error:return None,[],dict(candidate_found=False,reason=str(error))
    values=native.evaluate_scalars(volume.points);volume.point_data['scalar']=values
    n=base.n_points;ordered=values.reshape(-1,n)
    reversals=int(np.count_nonzero((np.diff(ordered,axis=0)<=0).any(axis=0)))
    volume.point_data['model_clearance']=distance(volume.points)-gap
    volume=volume.clip_scalar(value=.002,invert=False,scalars='model_clearance')
    if not volume.n_cells:return None,[],dict(candidate_found=False,reason='Loft removed by model-gap trimming.')
    volume.point_data['scalar']=native.evaluate_scalars(volume.points)
    layers,checks=check_layers(volume,native,ray_query,distance,schedule,gap,min_dot)
    return volume,layers,dict(candidate_found=True,available_trunk_length_mm=total,
        connection_point=connection.tolist(),base_area_mm2=float(base.area),attachment_distance_mm=attachment,
        scalar_reversing_loft_columns=reversals,sizing=sizing,first_layer_checks=checks,
        accepted=False,operation_seconds=time.perf_counter()-started,
        status='Diagnostic candidate only; production rooting flags remain unchanged.')
