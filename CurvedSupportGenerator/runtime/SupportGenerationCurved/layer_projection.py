"""Exact physical-base and trilinear-isosurface refinement helpers."""
from pathlib import Path
import time
import numpy as np
import pyvista as pv
from . import tree_geometry as tg
from .platform_router.context import save
from .planar_bases import export_bases as _GENERAL_BASE_EXPORT
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

def project_layer(record,shape,field,folder,regions,*,surface=None,schedule=None,scalar_error_budget=None,maximum_refinements=6):
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
    maximum_move=0.;refinements=0
    for refinement in range(maximum_refinements+1):
        points=np.asarray(surface.points).copy()
        for iteration in range(8):
            scalar,gradient=field.evaluate_scalars_and_gradients(points)
            error=scalar-target
            if abs(error).max(initial=0)<1e-10:break
            delta=error[:,None]*gradient/np.maximum(np.sum(gradient*gradient,axis=1)[:,None],1e-20)
            length=np.linalg.norm(delta,axis=1);delta*=np.minimum(1.,.25/np.maximum(length,1e-15))[:,None]
            points-=delta
        assert abs(field.evaluate_scalars(points)-target).max(initial=0)<1e-7
        maximum_move=max(maximum_move,float(np.linalg.norm(points[:len(original)]-original,axis=1).max(initial=0)))
        assert maximum_move<.75,maximum_move
        surface.points=points
        triangles=points[surface.regular_faces]
        probes=np.concatenate((triangles.mean(axis=1)[:,None],(triangles+np.roll(triangles,-1,axis=1))*.5),axis=1).reshape(-1,3)
        values=np.concatenate([field.evaluate_scalars(probes[k:k+25000]) for k in range(0,len(probes),25000)])
        bad=(abs(values.reshape(-1,4)-target)>budget*.8).any(axis=1)
        if not bad.any():break
        assert refinement<maximum_refinements,'Isosurface facet refinement budget exhausted'
        surface=tg.subdivide_selected_faces(surface,bad);refinements+=1
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
    # Inset and original-boundary checks differ by a constant only. Reuse the
    # unshifted samples instead of building two independent geometry caches.
    surface=tg.clip_layer(surface,Inset(),root_method='adaptive')
    surface,removed=tg.refine_clipped_layer(surface,local,'adaptive')
    flags=classify_patches(surface,record['layer'],regions)
    combined=int(np.bitwise_or.reduce(flags,initial=0))
    roles=[name for name,bit in [('platform',1),('contact',2)] if combined&bit]
    mark_surface(surface,dict(layer=record['layer'],scalar=target,roles=roles,is_platform_layer='platform' in roles,is_contact_layer='contact' in roles))
    surface.field_data['is_model_base_layer']=[bool(combined&4)]
    vertex=local.evaluate(surface.points)
    center=local.evaluate(np.asarray(surface.points)[surface.regular_faces].mean(axis=1)) if surface.n_cells else np.empty(0)
    assert not (vertex>1e-6).any() and not (center>1e-6).any()
    path=folder/f"{record['layer']}.vtp";surface.save(path)
    updated=dict(record,**slice_error(surface,field,target));updated.update(file=str(path.resolve()),points=surface.n_points,
        cells=surface.n_cells,area=float(surface.area),roles=roles,is_platform_layer='platform' in roles,
        is_contact_layer='contact' in roles,is_model_base_layer=bool(combined&4),
        iso_projection_maximum_displacement_mm=maximum_move,iso_projection_seconds=time.perf_counter()-began,
        removed_crossing_triangles_projection=removed,iso_scalar_error_budget=budget,iso_refinement_passes=refinements,
        conservative_boundary_inset_mm=.005)
    assert max(updated['vertex_scalar_error_max'],updated['interior_scalar_error_max'])<=budget,updated
    return updated
