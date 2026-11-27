"""Exact physical-base and trilinear-isosurface refinement helpers."""
from pathlib import Path
import time
import numpy as np
import pyvista as pv
from . import tree_geometry as tg
from .platform_router.context import save
from .planar_bases import export_bases as _GENERAL_BASE_EXPORT
def project_points(field,points,target):
    """Project inside the saved grid, spending extra work only on slow points.

    Cell-boundary gradient changes can converge linearly. Preserve the same
    residual and step bounds while allowing those few points more iterations.
    """
    points=np.array(points,dtype=float,copy=True)
    bounded=all(hasattr(field,name) for name in ('origin','spacing','shape'))
    if bounded:
        low=np.asarray(field.origin);high=low+field.spacing*(np.asarray(field.shape)-1)
        tolerance=field.spacing*1e-7
        if np.any(points<low-tolerance) or np.any(points>high+tolerance):
            raise ValueError('Initial layer points lie outside the saved field grid.')
        points=np.clip(points,low,high)
    active=np.arange(len(points))
    for _ in range(64):
        if not len(active):break
        current=points[active]
        scalar,gradient=field.evaluate_scalars_and_gradients(current);error=scalar-target
        if not np.isfinite(error).all():break
        pending=abs(error)>=1e-10
        active=active[pending];current=current[pending];error=error[pending];gradient=gradient[pending]
        if not len(active):break
        if not np.isfinite(gradient).all():
            raise ValueError('Layer projection cannot meet its scalar accuracy inside the saved field grid.')
        if bounded:
            direction=error[:,None]*gradient
            blocked=((current<=low+1e-12)&(direction>0))|((current>=high-1e-12)&(direction<0))
            gradient=np.where(blocked,0.,gradient)
        delta=error[:,None]*gradient/np.maximum(np.sum(gradient*gradient,axis=1)[:,None],1e-20)
        length=np.linalg.norm(delta,axis=1);delta*=np.minimum(1.,.25/np.maximum(length,1e-15))[:,None]
        updated=current-delta
        points[active]=np.clip(updated,low,high) if bounded else updated
    if not np.all(abs(field.evaluate_scalars(points)-target)<1e-7):
        raise ValueError('Layer projection cannot meet its scalar accuracy inside the saved field grid.')
    return points


def projection_chart_folds(field, points, projected, faces):
    """Find projection-created folds in certified single-valued Y charts.

    A normal Newton step can move a green-refinement vertex sideways across
    its long, thin neighbour. Scalar residuals and even the 3D normal-dot test
    can still pass. For a locally monotone grid field, reversing the projected
    triangle orientation is a mesh defect, not a genuine field overhang.
    """
    before=np.asarray(points)[faces];after=np.asarray(projected)[faces]
    old=np.cross(before[:,1]-before[:,0],before[:,2]-before[:,0])[:,1]
    new=np.cross(after[:,1]-after[:,0],after[:,2]-after[:,0])[:,1]
    bad=np.zeros(len(faces),dtype=bool)
    candidates=np.flatnonzero((abs(old)>1e-12)&(old*new<=0))
    if not len(candidates) or not all(hasattr(field,k) for k in ('origin','spacing','shape','values')):
        return bad
    # df/dY is bilinear in XZ and constant in Y within each trilinear cell.
    # Bounds on all XZ corners of every crossed cell certify its sign over
    # the entire old-to-new face box; vertex-gradient samples alone do not.
    slopes=np.diff(np.asarray(field.values).reshape(field.shape),axis=1)/field.spacing
    last=np.asarray(field.shape)-2
    for index in candidates:
        vertices=np.vstack((before[index],after[index]))
        lo=np.clip(np.floor((vertices.min(axis=0)-field.origin)/field.spacing).astype(int),0,last)
        hi=np.clip(np.floor((vertices.max(axis=0)-field.origin)/field.spacing).astype(int),0,last)
        values=slopes[lo[0]:hi[0]+2,lo[1]:hi[1]+1,lo[2]:hi[2]+2]
        bad[index]=bool(np.all(values>1e-9) or np.all(values < -1e-9))
    return bad


def project_surface(field, surface, target, maximum_refinements):
    """Refine the unmoved neighbourhood before committing a chart-folding step."""
    refinements=0;initial=np.asarray(surface.points).copy()
    while True:
        points=project_points(field,surface.points,target)
        bad=projection_chart_folds(field,surface.points,points,surface.regular_faces)
        if not bad.any():
            movement=float(np.linalg.norm(points[:len(initial)]-initial,axis=1).max(initial=0))
            surface.points=points
            return surface,refinements,movement
        if refinements>=maximum_refinements:
            raise ValueError('Scalar projection would fold a single-valued layer chart; local refinement budget exhausted.')
        # Keep the original embedding until a valid projection is found.
        # Refining an already folded mesh only subdivides the same overlap.
        surface=tg.subdivide_face_neighbourhood(surface,bad);refinements+=1


def certified_planar_bases(geometry,settings,layer_height,schedule,output):
    """Exact frustum sections when model-clearance follows from height bounds.

    The inscribed disk polygons and their constrained triangulation are inside
    the platform frusta by construction. Their model gap is certified globally
    here; general near-model cases continue to use the ordinary clipped path.
    """
    from shapely.geometry import Point
    from shapely.ops import unary_union
    import shapely
    from SupportGenerationCurved import planar_bases
    plate=settings['plate'];height=settings['height'];up=np.asarray(geometry.direction)
    model_min=float((np.asarray(geometry.mesh.points)@up).min())
    if not np.allclose(up,[0,1,0]) or model_min-(plate+height)<max(geometry.side_gap,geometry.contact_gap)+1e-6:
        return _GENERAL_BASE_EXPORT(geometry,settings,layer_height,schedule,output)
    geometry.prune_pairs=True
    directory=Path(output)/'planar_platform_base_layers';directory.mkdir(exist_ok=True)
    levels=np.minimum(np.arange(1,int(np.ceil(height/layer_height))+1)*layer_height,height)
    pitch=max(float(np.min(np.diff(schedule))),1e-6);records=[];maximum_error=0.
    for number,level in enumerate(levels):
        polygons=[Point(center[0],center[2]).buffer(lower+(upper-lower)*level/height,quad_segs=48)
            for center,lower,upper in zip(settings['centers'],settings['radii'],settings['top_radii'])]
        triangles=shapely.constrained_delaunay_triangles(unary_union(polygons))
        faces=np.array([np.asarray(t.exterior.coords)[:3] for t in triangles.geoms])
        points=np.column_stack((faces[:,:,0].ravel(),np.full(len(faces)*3,plate+level),faces[:,:,1].ravel()))
        cells=np.arange(len(points)).reshape(-1,3)
        down=np.cross(points[cells[:,1]]-points[cells[:,0]],points[cells[:,2]]-points[cells[:,0]])[:,1]<0
        cells[down]=cells[down][:,[0,2,1]]
        surface=pv.PolyData(points,np.column_stack((np.full(len(cells),3),cells)).ravel()).clean()
        xyz=np.asarray(surface.points)[surface.regular_faces]
        probes=np.vstack((surface.points,xyz.mean(axis=1),((xyz+np.roll(xyz,-1,axis=1))*.5).reshape(-1,3)))
        error=float(geometry.evaluate(probes).max());maximum_error=max(maximum_error,error)
        if error>1e-6:
            values=geometry.evaluate(probes);point=probes[int(np.argmax(values))]
            original=getattr(getattr(geometry.evaluate,'__self__',None),'original',None)
            exact=float(original([point])[0]) if original else None
            raise ValueError(f'Physical base outside support solid: {error:g} at {point.tolist()}, array evaluation={exact}, bases={geometry.segments[geometry.platform_base].tolist()}, distance={geometry.bounded_distance([point],geometry.side_gap).tolist()}')
        surface.cell_data['support_role_flags']=np.ones(surface.n_cells,dtype=np.uint8)
        surface.field_data['is_planar_base_layer']=[1];surface.field_data['physical_height']=[plate+level]
        path=directory/f'{number}.vtp';surface.save(path)
        records.append(dict(file=str(path.resolve()),layer=-1,source_layer_index=-1,
            scalar=float(schedule[0]-(len(levels)-number)*pitch),scalar_meaning='print ordering only; geometry uses physical height',
            physical_height=plate+float(level),is_planar_base_layer=True,roles=['platform'],is_platform_layer=True,
            is_contact_layer=False,is_model_base_layer=False,points=surface.n_points,cells=surface.n_cells))
        print(dict(stage='certified_flat_base',layer=number,height_mm=float(level),triangles=surface.n_cells),flush=True)
    save(Path(output)/'planar_platform_base_manifest.json',records)
    save(Path(output)/'planar_base_verification.json',dict(method='Analytic frustum sections; inscribed circle polygons.',
        physical_model_clearance_lower_bound_mm=model_min-(plate+height),maximum_sampled_solid_error_mm=maximum_error,
        tested_vertices_centroids_and_edge_midpoints=True,no_footprint_or_acceptance_tolerance_changed=True))
    return records

def project_layer(record,shape,field,folder,regions,*,surface=None,schedule=None,scalar_error_budget=None,maximum_refinements=6,preserve_projected_chart=False):
    from SupportGenerationCurved.nonplanar import slice_error
    from SupportGenerationCurved.planar_bases import settings_for,remove_from_field_layer
    from SupportGenerationCurved.spatial_roles import classify_patches
    from SupportGenerationCurved.layer_roles import mark_surface
    began=time.perf_counter()
    surface=(pv.read(record['file']) if surface is None else surface).triangulate()
    original=np.asarray(surface.points).copy()
    target=record['scalar'];index=record['layer']
    if schedule is None:schedule=np.loadtxt(OUT/'main_scalar_values.txt')
    gaps=np.diff(schedule);budget=.25*min(gaps[max(0,index-1)],gaps[min(len(gaps)-1,index)])
    if scalar_error_budget is not None:
        if not np.isfinite(scalar_error_budget) or scalar_error_budget<=0:raise ValueError('Scalar error budget must be positive and finite.')
        budget=min(budget,float(scalar_error_budget))
    if not isinstance(maximum_refinements,int) or not 0<=maximum_refinements<=12:raise ValueError('Maximum refinements must be an integer from 0 to 12.')
    if not isinstance(preserve_projected_chart,bool):raise ValueError('Projected-chart policy must be boolean.')
    maximum_move=0.;refinements=0;fold_refinements=0
    for refinement in range(maximum_refinements+1):
        if preserve_projected_chart:
            surface,extra,_=project_surface(field,surface,target,maximum_refinements-refinements)
            refinements+=extra;fold_refinements+=extra;points=np.asarray(surface.points)
        else:
            points=project_points(field,surface.points,target);surface.points=points
        maximum_move=max(maximum_move,float(np.linalg.norm(points[:len(original)]-original,axis=1).max(initial=0)))
        if maximum_move>=.75:raise ValueError(f'Layer {index}: scalar projection would move a vertex {maximum_move:.6g} mm, exceeding the 0.75 mm displacement limit. Regenerate a finer source contour; the saved field and limit were not changed.')
        triangles=points[surface.regular_faces]
        probes=np.concatenate((triangles.mean(axis=1)[:,None],(triangles+np.roll(triangles,-1,axis=1))*.5),axis=1).reshape(-1,3)
        values=np.concatenate([field.evaluate_scalars(probes[k:k+25000]) for k in range(0,len(probes),25000)])
        bad=(abs(values.reshape(-1,4)-target)>budget*.8).any(axis=1)
        if not bad.any():break
        assert refinements<maximum_refinements,'Isosurface facet refinement budget exhausted'
        surface=tg.subdivide_face_neighbourhood(surface,bad);refinements+=1
    bases=settings_for(shape)
    if bases and surface.n_points and surface.bounds[2]<=bases['plate']+bases['height']+1e-8:
        surface=remove_from_field_layer(surface,bases)
    # Clip 5 microns inside the nominal boundary, then verify against the
    # original solid. This avoids chasing micron-size nonconvex chord errors
    # through repeated subdivisions while retaining every original check.
    local=tg.CachedLayerField(shape)
    class Inset:
        def __getattr__(self,name):return getattr(shape,name)
        def evaluate(self,points):return local.evaluate(points)+.005
        def certified_inside_triangles(self,triangles):
            certify=getattr(shape,'certified_inside_triangles',None)
            return certify(triangles,inset=.005) if certify is not None else np.zeros(len(triangles),bool)
    # Inset and original-boundary checks differ by a constant only. Reuse the
    # unshifted samples instead of building two independent geometry caches.
    inset=Inset()
    if all(hasattr(shape,name) for name in ('ra','rb','midpoints')):
        from .layer_extraction import adaptive_candidate
        surface,_=adaptive_candidate(surface,inset,tolerance=.05)
    surface=tg.clip_layer(surface,inset,root_method='adaptive')
    surface,removed=tg.refine_clipped_layer(surface,local,'adaptive')
    # Trimming introduces new vertices and chords after the first isosurface
    # check. Refine that final mesh too; never accept the pre-clip error as
    # evidence for the geometry that will actually be exported.
    post_clip_refinements=0;post_clip_move=0.
    while surface.n_cells:
        measured=slice_error(surface,field,target)
        if max(measured['vertex_scalar_error_max'],measured['interior_scalar_error_max'])<=budget:break
        assert refinements<maximum_refinements,'Post-clipping isosurface refinement budget exhausted'
        xyz=np.asarray(surface.points)[surface.regular_faces]
        probes=np.concatenate((xyz.mean(axis=1)[:,None],(xyz+np.roll(xyz,-1,axis=1))*.5),axis=1)
        bad=(abs(field.evaluate_scalars(probes.reshape(-1,3)).reshape(-1,4)-target)>budget*.8).any(axis=1)
        vertex_bad=abs(field.evaluate_scalars(surface.points)-target)>budget*.8
        bad|=vertex_bad[surface.regular_faces].any(axis=1)
        surface=tg.subdivide_face_neighbourhood(surface,bad);refinements+=1;post_clip_refinements+=1
        if preserve_projected_chart:
            surface,extra,movement=project_surface(field,surface,target,maximum_refinements-refinements)
            refinements+=extra;fold_refinements+=extra;points=np.asarray(surface.points)
        else:
            before=np.asarray(surface.points).copy();points=project_points(field,before,target)
            movement=float(np.linalg.norm(points-before,axis=1).max(initial=0));surface.points=points
        if movement>=.75:raise ValueError(f'Layer {index}: post-clipping scalar projection would move a vertex {movement:.6g} mm, exceeding the 0.75 mm displacement limit. Regenerate a finer source contour.')
        post_clip_move=max(post_clip_move,movement)
        center=np.asarray(surface.points)[surface.regular_faces].mean(axis=1)
        if (local.evaluate(points)>1e-6).any() or (local.evaluate(center)>1e-6).any():
            if all(hasattr(shape,name) for name in ('ra','rb','midpoints')):
                from .layer_extraction import adaptive_candidate
                surface,_=adaptive_candidate(surface,inset,tolerance=.05)
            surface=tg.clip_layer(surface,inset,root_method='adaptive')
        surface,extra=tg.refine_clipped_layer(surface,local,'adaptive');removed+=extra
    flags=classify_patches(surface,record['layer'],regions)
    combined=int(np.bitwise_or.reduce(flags,initial=0))
    roles=[name for name,bit in [('platform',1),('contact',2)] if combined&bit]
    mark_surface(surface,dict(layer=record['layer'],scalar=target,roles=roles,is_platform_layer='platform' in roles,is_contact_layer='contact' in roles))
    surface.field_data['is_model_base_layer']=[bool(combined&4)]
    vertex=local.evaluate(surface.points)
    center=local.evaluate(np.asarray(surface.points)[surface.regular_faces].mean(axis=1)) if surface.n_cells else np.empty(0)
    trim_report=None
    if getattr(shape,'trim_boundary_tolerance_mm',0.):
        from .trim_tolerance import classify_faces,record as record_trim
        bad,_,trim_report=classify_faces(surface,local,include_vertices=True,vertex_values=vertex)
        if bad.any():raise ValueError('Projected support layer exceeds its verified physical trimming tolerance.')
        record_trim(surface,trim_report)
        if trim_report['accepted_faces']:
            import json
            print(json.dumps(dict(stage='support_trim_approximation',layer=record['layer'],
                warning='Retained bounded post-projection support approximation.',**trim_report)),flush=True)
    else:
        assert not (vertex>1e-6).any() and not (center>1e-6).any()
    path=folder/f"{record['layer']}.vtp";surface.save(path)
    updated=dict(record,**slice_error(surface,field,target));updated.update(file=str(path.resolve()),points=surface.n_points,
        cells=surface.n_cells,area=float(surface.area),roles=roles,is_platform_layer='platform' in roles,
        is_contact_layer='contact' in roles,is_model_base_layer=bool(combined&4),
        iso_projection_maximum_displacement_mm=maximum_move,iso_projection_seconds=time.perf_counter()-began,
        removed_crossing_triangles_projection=removed,iso_scalar_error_budget=budget,iso_refinement_passes=refinements,
        iso_post_clip_refinement_passes=post_clip_refinements,iso_post_clip_projection_maximum_step_mm=post_clip_move,
        iso_projection_fold_refinements=fold_refinements,
        iso_projection_preserves_projected_chart=preserve_projected_chart,
        initial_boundary_inset_mm=.005,final_boundary_check_tolerance_mm=1e-6,
        trim_boundary_approximation=trim_report)
    assert max(updated['vertex_scalar_error_max'],updated['interior_scalar_error_max'])<=budget,updated
    return updated
