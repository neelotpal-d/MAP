"""Finite interior-hatch ribbon metric, evaluated on final processed segments.

No sidecar interpolation across filtering/decimation: use the exact saved mesh
and final coordinates at export. Walls, links and out-of-domain strips retain
their existing explicit policy and are counted separately.
"""
import json
from pathlib import Path
import numpy as np
import shapely

ROLES={'inner_raster','solid_infill','sparse_infill'}


def _identical_height_contacts(faces,xz,a,b):
    """Topological shared edges/isolated vertices already have identical Y.

    Skip only pairs proven to have disjoint projected interiors, with a common
    edge or solely their common vertex. A boundary segment with other endpoints
    is still checked. Avoid ill-conditioned plane extrapolation on known-equal
    mesh contacts, and avoid GEOS work on ordinary conforming triangulations.
    """
    skip=np.zeros(len(a),bool)
    for start in range(0,len(a),32768):
        end=min(len(a),start+32768);ia,ib=a[start:end],b[start:end]
        shared=faces[ia,:,None]==faces[ib,None,:];count=shared.sum(axis=(1,2))
        candidate=np.flatnonzero(count>0)
        if not len(candidate):continue
        aa,bb=xz[ia[candidate]],xz[ib[candidate]]
        common_a=shared[candidate].any(axis=2);common_b=shared[candidate].any(axis=1)
        flags=[];point_flags=[]
        for own,other,common in ((aa,bb,common_b),(bb,aa,common_a)):
            edge=np.roll(own,-1,axis=1)-own
            area=(own[:,1,0]-own[:,0,0])*(own[:,2,1]-own[:,0,1])-(own[:,1,1]-own[:,0,1])*(own[:,2,0]-own[:,0,0])
            delta=other[:,None,:,:]-own[:,:,None,:]
            cross=(edge[:,:,None,0]*delta[:,:,:,1]-edge[:,:,None,1]*delta[:,:,:,0])*np.sign(area)[:,None,None]
            separated=cross.max(axis=2)<=0
            flags.append(separated.any(axis=1))
            unique_common_zero=((cross==0).sum(axis=2)==1)&(((cross==0)&common[:,None,:]).any(axis=2))
            point_flags.append((separated&unique_common_zero).any(axis=1))
        c=count[candidate]
        proven=(c==3)|((c==2)&(flags[0]|flags[1]))|((c==1)&(point_flags[0]|point_flags[1]))
        skip[start+candidate[proven]]=True
    return skip


def assert_single_sheet(lifter,tolerance=1e-5):
    """Compare affine heights at every overlap vertex, including crossings."""
    if not np.isfinite(tolerance) or tolerance<=0:raise ValueError('Positive finite projection tolerance required.')
    from projection_mesh_repair import _overlap_summary
    _,spread,faces,coordinates=_overlap_summary(lifter,tolerance)
    if spread>tolerance:
        raise ValueError(f'Projected surface has overlapping sheets in XZ with different heights '
            f'(faces {faces}, height difference {spread:.9g} mm at XZ {coordinates}); '
            'split it into single-height charts before projected planning/export.')


def finite_strip_spacings(lifter,region,points,nominal,tags,breaks,directions,*,maximum_spacing=2.):
    p=np.asarray(points,float);w=np.asarray(nominal,float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():raise ValueError('Finite XYZ path required.')
    if w.shape!=(max(0,len(p)-1),) or not np.isfinite(w).all() or (w<=0).any():raise ValueError('Positive per-segment spacing required.')
    if not np.isfinite(maximum_spacing) or maximum_spacing<=0:raise ValueError('Positive surface-spacing limit required.')
    result=np.zeros(len(w));report=dict(qualified_segments=0,excluded_segments=len(w),maximum_physical_spacing_mm=0.)
    if not len(w):return result,report
    d=np.diff(p[:,[0,2]],axis=0);length2=np.linalg.norm(d,axis=1)
    eligible=np.zeros(len(w),bool)
    for role,axis in directions.items():
        if role not in ROLES:continue
        axis=np.asarray(axis,float)
        if axis.shape!=(2,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-12:raise ValueError('Invalid hatch direction metadata.')
        axis/=np.linalg.norm(axis)
        eligible|=(tags[:-1]==role)&(tags[1:]==role)&(length2>1e-10)&(abs(d@axis)>=np.cos(np.deg2rad(.5))*length2)
    if breaks:eligible[np.asarray(sorted(breaks),int)-1]=False
    ids=np.flatnonzero(eligible)
    if not len(ids):return result,report
    tangent=d[ids]/length2[ids,None];perp=np.column_stack((-tangent[:,1],tangent[:,0]))
    left0=p[ids][:,[0,2]]+.5*w[ids,None]*perp
    left1=p[ids+1][:,[0,2]]+.5*w[ids,None]*perp
    right0=p[ids][:,[0,2]]-.5*w[ids,None]*perp
    right1=p[ids+1][:,[0,2]]-.5*w[ids,None]*perp
    quads=shapely.polygons(np.stack((left0,left1,right1,right0),axis=1))
    shapely.prepare(region)
    inside=shapely.covers(region,quads)
    ids=ids[inside]
    if not len(ids):return result,report
    arrays=[a[inside] for a in (left0,left1,right0,right1)]
    # One bulk lift for each side and the centreline; no nearest-face substitute.
    lifted,ambiguous,_=lifter.lift(np.vstack((*arrays,p[ids][:,[0,2]],p[ids+1][:,[0,2]])))
    if ambiguous:raise ValueError('Model hatch ribbon crosses ambiguous projected sheets; regenerate in a valid chart.')
    a,b,c,d3,center0,center1=np.split(lifted,6)
    on_source=(np.linalg.norm(center0-p[ids],axis=1)<=1e-4)&(np.linalg.norm(center1-p[ids+1],axis=1)<=1e-4)
    lengths=np.linalg.norm(center1-center0,axis=1)
    area=.5*np.linalg.norm(np.cross(c-a,d3-a),axis=1)+.5*np.linalg.norm(np.cross(d3-a,b-a),axis=1)
    metric=area/np.maximum(lengths,1e-30)
    # Qualify the entire piecewise-linear mesh strip, not just its corners.
    # All vertices of every intersecting triangle must lie within 0.005 mm of
    # the ribbon plane. Affine triangles then obey the same bound everywhere.
    plane=np.cross(c-a,b-a);plane/=np.maximum(np.linalg.norm(plane,axis=1,keepdims=True),1e-30)
    pairs=np.asarray(lifter.tree.query(quads[inside],predicate='intersects'))
    errors=np.zeros(len(ids))
    if pairs.size:
        tri=lifter.vertices[lifter.faces[lifter.face_indices[pairs[1]]]]
        deviation=abs(np.einsum('nki,ni->nk',tri-a[pairs[0],None],plane[pairs[0]])).max(axis=1)
        np.maximum.at(errors,pairs[0],deviation)
    valid=on_source&(lengths>1e-10)&(errors<=.005)
    if np.any(~np.isfinite(metric[valid])) or np.any(metric[valid]>maximum_spacing+1e-8):
        raise ValueError('Required physical hatch spacing exceeds the configured surface-spacing limit. Refine the layer/orientation or explicitly revise the limit.')
    result[ids[valid]]=metric[valid]
    report.update(qualified_segments=int(valid.sum()),excluded_segments=int(len(w)-valid.sum()),
        maximum_qualified_planarity_error_mm=float(errors[valid].max(initial=0.)),
        maximum_physical_spacing_mm=float(metric[valid].max(initial=0.)),
        scope='First-order normal-height volume for source-matched interior straight strips. Walls/links/holes/modified clearance paths excluded; no bead-envelope or physical calibration certificate.')
    return result,report


def layer_surface_spacings(rows,widths,tags,breaks,layer,metadata,*,maximum_spacing=2.):
    from batch_generate_projected_hybrid_paths import ProjectedSurfaceLifter,_planar_region_from_projected_triangles
    from mesh_distance_workflow import load_obj_mesh
    from extrusion_height import apply_axis_lift
    from surface_metric_source import retained_metric_source
    source,recovery=retained_metric_source(layer,metadata)
    if not source:return np.zeros(max(0,len(rows)-1)),dict(qualified_segments=0,reason='No saved source mesh; regenerate paths to enable surface correction.')
    vertices,faces=load_obj_mesh(source)
    vertices=apply_axis_lift(vertices,axis=str(metadata.get('model_lift_axis','y')),amount=float(metadata.get('model_lift',0.)))
    directions=dict(layer.get('surface_hatch_directions',{}))
    if not directions and layer.get('parallel_axis'):
        value=layer['parallel_axis']
        if value in ('x','z'):directions['inner_raster']=[1.,0.] if value=='x' else [0.,1.]
        elif str(value).startswith('angle:'):
            theta=np.deg2rad(float(value.split(':')[1]));directions['inner_raster']=[float(np.cos(theta)),float(np.sin(theta))]
    nominal=.5*(widths[:-1]+widths[1:])
    if layer.get('surface_metric_components')=='connected_regions':
        from connected_patch_planner import surface_components
        charts=surface_components(vertices,faces)
    else:charts=((vertices,faces),)
    result=np.zeros(max(0,len(rows)-1));maximum_error=0.;count=0
    for chart_index,(points,triangles) in enumerate(charts):
        selected=layer.get('surface_metric_chart_indices')
        if selected is not None and chart_index not in selected:continue
        lifter=ProjectedSurfaceLifter.build(points,triangles)
        region=_planar_region_from_projected_triangles(lifter)
        chart_directions=layer.get('surface_metric_chart_directions',{}).get(str(chart_index),directions)
        values,report=finite_strip_spacings(lifter,region,rows[:,:3],nominal,tags,breaks,chart_directions,maximum_spacing=maximum_spacing)
        overlap=(result>0)&(values>0)
        if np.any(overlap&~np.isclose(result,values,rtol=1e-6,atol=1e-8)):
            raise ValueError('Hatch strip matches disconnected source charts with inconsistent physical spacing.')
        result=np.maximum(result,values);count+=1
        maximum_error=max(maximum_error,report.get('maximum_qualified_planarity_error_mm',0.))
    qualified=int(np.count_nonzero(result))
    return result,dict(qualified_segments=qualified,excluded_segments=len(result)-qualified,
        maximum_physical_spacing_mm=float(result.max(initial=0.)),
        maximum_qualified_planarity_error_mm=maximum_error,source_chart_count=count,
        recovered_patch_reference=recovery is not None,
        scope='Source-matched finite hatch strips; disconnected source charts evaluated separately')
