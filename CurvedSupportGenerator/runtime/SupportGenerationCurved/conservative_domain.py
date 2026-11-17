"""Prepare a conservative support-volume field from any matching P1 model."""
import itertools
import json
from pathlib import Path
import time
import numpy as np

from .field_grid import GridField
from .platform_router.context import read,save,sha
from .platform_router.engine import model_signature
from .collision_tool import MODES, COMPONENTS, validate_axisymmetric


def normalized_tool(tool,display_axis='y',collision_mode='fixed_axis'):
    """Convert the generator's (u, build-up, u-cross-up) box to canonical Y-up."""
    if collision_mode not in MODES:raise ValueError('Choose fixed_axis or field_gradient collision mode.')
    if all(name in tool for name in COMPONENTS):
        if collision_mode=='fixed_axis':
            w=np.asarray(tool.get('build_direction',[0,1,0]),float)
            if w.shape!=(3,) or not np.isfinite(w).all() or np.linalg.norm(w)<1e-12:
                raise ValueError('Invalid fixed tool axis.')
            w=w/np.linalg.norm(w)
            if not (np.allclose(w,[0,1,0]) or (display_axis=='z' and np.allclose(w,[0,0,1]))):
                raise ValueError('Fixed axisymmetric head must align with the prepared build axis.')
        validate_axisymmetric(tool)
        return dict(tool,build_direction=[0,1,0],in_plane_u=[1,0,0],geometry_kind='axisymmetric')
    w=np.asarray(tool.get('build_direction',[0,1,0]),float);u=np.asarray(tool.get('in_plane_u',[1,0,0]),float)
    if w.shape!=(3,) or u.shape!=(3,) or not np.isfinite([w,u]).all() or min(np.linalg.norm(w),np.linalg.norm(u))<1e-12:
        raise ValueError('Tool requires finite nonzero build_direction and in_plane_u.')
    w=w/np.linalg.norm(w);u=u-w*(u@w)
    if np.linalg.norm(u)<1e-12:raise ValueError('Tool in-plane axis is parallel to build-up.')
    u=u/np.linalg.norm(u);v=np.cross(u,w)
    if np.allclose(w,[0,1,0],atol=1e-8):rotation=np.eye(3)
    elif display_axis=='z' and np.allclose(w,[0,0,1],atol=1e-8):rotation=np.array([[1,0,0],[0,0,1],[0,-1,0]])
    else:raise ValueError('Use a Y-up tool box or a fixed tool aligned with the selected model build axis.')
    frame=rotation@np.column_stack((u,w,v))
    if collision_mode=='fixed_axis' and (not np.allclose(np.abs(frame).sum(axis=0),1.,atol=1e-8) or not np.allclose(np.abs(frame).sum(axis=1),1.,atol=1e-8)):
        raise ValueError('This support-field solver needs an axis-aligned fixed tool box.')
    lo=np.array([tool['tool_'+a+'_min'] for a in 'xyz'],float);hi=np.array([tool['tool_'+a+'_max'] for a in 'xyz'],float)
    if not np.isfinite([lo,hi]).all() or np.any(hi<=lo):raise ValueError('Tool box bounds must be finite and increasing.')
    if collision_mode=='field_gradient':
        if lo[1]<0:raise ValueError('The head box must start at or above its tip.')
        result=dict(tool,build_direction=[0,1,0],in_plane_u=frame[:,0].tolist(),geometry_kind='box')
        return result
    corners=np.array(list(itertools.product(*zip(lo,hi))))@frame.T
    low=corners.min(axis=0);high=corners.max(axis=0)
    if low[1]<0:raise ValueError('The fixed head box must start at or above its tip.')
    result=dict(build_direction=[0,1,0],in_plane_u=[1,0,0])
    for i,a in enumerate('xyz'):result['tool_'+a+'_min']=float(low[i]);result['tool_'+a+'_max']=float(high[i])
    return result


def _spaced(points,spacing):
    if not len(points):return np.empty(0,dtype=int)
    _,ids=np.unique(np.floor(np.asarray(points)/spacing).astype(np.int64),axis=0,return_index=True)
    return np.sort(ids)


def prepare(pipeline,folder,settings):
    import pyvista as pv
    from scipy.ndimage import maximum_filter,binary_dilation
    from vtkmodules.vtkFiltersGeneral import vtkOBBTree
    from vtkmodules.vtkCommonCore import vtkPoints,vtkIdList
    from SupportGenerationVoxel.model_field import TetrahedralVertexField,load_closed_mesh
    from .compatible_export import model_volume
    from .conservative_field import progress
    inputs=pipeline['input'];folder=Path(folder).resolve()
    spacing=float(settings.get('spacing',2.));allowance=float(settings.get('lateral_allowance',6.))
    gap=float(settings.get('clearance',.4));anchor_spacing=float(settings.get('anchor_spacing',2.))
    upward=float(settings.get('upward_length',5.));limit=int(settings.get('maximum_points',500000))
    cap=int(settings.get('maximum_model_samples',200000))
    if not np.isfinite([spacing,allowance,gap,anchor_spacing,upward]).all() or min(spacing,anchor_spacing,upward)<=0 or min(allowance,gap)<0 or min(limit,cap)<100:
        raise ValueError('Invalid support-field spacing, gap, sampling or point limits.')
    model=Path(inputs['tetrahedral_volume']).resolve();scalar=inputs.get('tetrahedral_scalar','scalar')
    native=TetrahedralVertexField(model,scalar);surface=load_closed_mesh(inputs['mesh'])
    model_volume(inputs['mesh'],surface,model)
    if surface.bounds[2]<-1e-6:raise ValueError('Place the model at or above the zero-height build plate before support-field optimization.')
    collision_mode=settings.get('collision_mode','field_gradient' if inputs.get('model_generator_axes')==5 else 'fixed_axis')
    tool_path=Path(inputs['tool']).resolve();tool=normalized_tool(read(tool_path),settings.get('display_axis','y'),collision_mode)
    started=time.perf_counter()
    lower=np.floor(np.asarray(native.volume.bounds)[::2]/spacing)*spacing
    upper=np.ceil(np.asarray(native.volume.bounds)[1::2]/spacing)*spacing
    lower-=np.array([allowance+spacing,0,allowance+spacing]);lower[1]=0.
    upper+=np.array([allowance+spacing,2*spacing,allowance+spacing])
    shape=tuple((np.rint((upper-lower)/spacing).astype(int)+1).tolist())
    if np.prod(shape)>limit:raise ValueError(f'Support field needs {np.prod(shape):,} grid points (limit {limit:,}); increase grid spacing or reduce lateral allowance.')
    if folder.exists():raise FileExistsError(f'Choose a fresh support-field output: {folder}')
    folder.mkdir(parents=True);save(folder/'tool.json',tool)
    axes=[lower[a]+np.arange(shape[a])*spacing for a in range(3)]
    xyz=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    tree=vtkOBBTree();tree.SetDataSet(surface);tree.BuildLocator();hits=vtkPoints();hit_ids=vtkIdList()
    height=np.full((shape[0],shape[2]),-np.inf)
    for i,x in enumerate(axes[0]):
        for k,z in enumerate(axes[2]):
            hits.Reset();hit_ids.Reset();tree.IntersectWithLine((x,-spacing,z),(x,upper[1]+spacing,z),hits,hit_ids)
            if hits.GetNumberOfPoints():height[i,k]=max(hits.GetPoint(j)[1] for j in range(hits.GetNumberOfPoints()))
    raw_height=height.copy();radius=int(np.ceil(allowance/spacing))
    height=maximum_filter(height,size=2*radius+1,mode='constant',cval=-np.inf)
    shadow=xyz[:,1]<=np.repeat(height[:,None,:],shape[1],axis=1).ravel()
    progress('support_domain_shadow',shape=shape,grid_points=len(xyz),hit_columns=int(np.isfinite(raw_height).sum()))
    cell_ids,closest=native.volume.find_closest_cell(xyz,return_closest_point=True)
    cell_ids=np.asarray(cell_ids);closest=np.asarray(closest);distance=np.linalg.norm(xyz-closest,axis=1);inside=distance<1e-8
    p0=native.points[native.cells[cell_ids,0]]
    bary123=np.einsum('nij,nj->ni',native.inverse_matrices[cell_ids],closest-p0)
    bary=np.column_stack((1-bary123.sum(axis=1),bary123))
    closest_value=np.sum(native.values[native.cells[cell_ids]]*bary,axis=1);closest_gradient=native.gradients[cell_ids]
    reference=float(np.median(np.linalg.norm(native.gradients,axis=1)))
    if not np.isfinite(reference) or reference<=1e-10:raise ValueError('Model field has a zero/invalid median gradient; optimize a usable slicing field first.')
    t=np.clip(distance/upward,0,1);t=t*t*(3-2*t)
    target=(1-t[:,None])*closest_gradient;target[:,1]+=t*reference
    initial=closest_value+np.einsum('ni,ni->n',xyz-closest,target);initial[inside]=closest_value[inside]
    support=shadow&(distance>=gap)&(xyz[:,1]>0.)
    if collision_mode=='field_gradient':support &= ~inside
    if not support.any():raise ValueError('No potential support points in the requested shadow/grid.')
    active=binary_dilation((shadow|(distance<=3*spacing)).reshape(shape),iterations=1)
    free=active.ravel()&~inside
    triangles=np.asarray(surface.points)[surface.regular_faces];area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)*.5
    rng=np.random.default_rng(42);count=min(200000,max(1000,int(area.sum()/anchor_spacing**2*8)))
    ids=rng.choice(len(triangles),count,p=area/area.sum());uv=rng.random((count,2));flip=uv.sum(axis=1)>1;uv[flip]=1-uv[flip]
    anchors=triangles[ids,0]+uv[:,0,None]*(triangles[ids,1]-triangles[ids,0])+uv[:,1,None]*(triangles[ids,2]-triangles[ids,0])
    anchors=anchors[_spaced(anchors,anchor_spacing)]
    candidates=np.vstack((anchors,native.points,native.points[native.cells].mean(axis=1),xyz[inside]))
    points=candidates[_spaced(candidates,max(.05,spacing*.5))]
    if len(points)>cap:
        raise ValueError(f'{len(points):,} model collision samples exceed limit {cap:,}; increase spacing or the sample cap. Samples are not silently dropped.')
    mv,mg=native.evaluate_scalars_and_gradients(points);av,ag=native.evaluate_scalars_and_gradients(anchors)
    validation=triangles.mean(axis=1);vv,vg=native.evaluate_scalars_and_gradients(validation)
    np.savez_compressed(folder/'domain.npz',origin=lower,spacing=spacing,shape=shape,initial=initial,
        free=free,inside=inside,active=active,target=target,distance=distance,shadow=support,
        height=height,raw_height=raw_height,reference_gradient=reference,potential_points=xyz[support],
        model_points=points,model_values=mv,model_gradients=mg,anchors=anchors,anchor_values=av,anchor_gradients=ag,
        validation=validation,validation_values=vv,validation_gradients=vg)
    centers=(xyz.reshape(shape+(3,))[:-1,:-1,:-1]+spacing*.5).reshape(-1,3)
    center_mask=support.reshape(shape)[:-1,:-1,:-1].reshape(-1)
    holdout=centers[center_mask]
    if collision_mode=='field_gradient' and len(holdout):
        # A support corner does not make its cell centre support. In rotating
        # mode this also determines which gradient defines the tool pose.
        _,closest_holdout=native.volume.find_closest_cell(holdout,return_closest_point=True)
        center_distance=np.linalg.norm(holdout-np.asarray(closest_holdout),axis=1)
        index=np.rint((holdout-lower)/spacing).astype(int)
        index=np.clip(index,0,np.asarray(shape)-1)
        valid=(center_distance>=max(gap,1e-8))&(holdout[:,1]>0)&(holdout[:,1]<=height[index[:,0],index[:,2]])
        holdout=holdout[valid]
    if len(holdout)>100000:holdout=holdout[np.linspace(0,len(holdout)-1,100000,dtype=int)]
    np.savez_compressed(folder/'holdout.npz',points=holdout)
    meta=dict(schema_version=1,model=str(model),model_sha256=sha(model),model_signature=model_signature(model,scalar),
        scalar_name=scalar,mesh=str(Path(inputs['mesh']).resolve()),tool=str(folder/'tool.json'),
        tool_source=str(tool_path),tool_sha256=sha(folder/'tool.json'),spacing_mm=spacing,lateral_allowance_mm=allowance,
        model_clearance_mm=gap,grid_shape=shape,grid_points=len(xyz),model_collision_samples=len(points),
        potential_samples=int(support.sum()),interface_fit_samples=len(anchors),interface_holdout_samples=len(validation),
        hard_fixed_model_grid_nodes=int(inside.sum()),free_grid_nodes=int(free.sum()),reference_gradient_per_mm=reference,
        model_stress_interpolation_used=False,build_up=[0,1,0],platform=0.,tree_or_support_solid_used=False,
        collision_mode=collision_mode,model_generator_axes=inputs.get('model_generator_axes'),
        model_field_provenance=inputs.get('model_field_provenance'),
        tool_frame_convention='project_reference_u_then_u_cross_axis' if collision_mode=='field_gradient' else 'canonical_fixed_y_up',
        collision_domain='Laterally dilated downward model shadow; potential material before tree generation.',
        preparation_seconds=time.perf_counter()-started,
        limitations=['Finite collision/geometry samples; not continuous tool clearance.',
            'Original model field is immutable; its existing collisions cannot be corrected by this solver.',
            'Exterior/interface fitting is soft; no exact C1 continuity claim.'])
    save(folder/'support_field.json',meta)
    GridField(dict(origin=lower,spacing=spacing,shape=shape),initial).volume().save(folder/'initial_grid.vti')
    progress('support_domain_prepared',seconds=time.perf_counter()-started,potential_samples=int(support.sum()),model_samples=len(points))
    return meta


def finish(folder,*,include_holdout=False):
    from .conservative_field import detect
    folder=Path(folder)
    with np.load(folder/'domain.npz') as d:data={k:d[k].copy() for k in d.files}
    with np.load(folder/'optimized_field.npz') as d:values=d['values'].copy()
    meta=read(folder/'support_field.json');field=GridField(data,values)
    pv,pg=field.evaluate_scalars_and_gradients(data['validation']);ref=float(data['reference_gradient'])
    error=pv-data['validation_values'];gradient=np.linalg.norm(pg-data['validation_gradients'],axis=1)/ref
    report=dict(status='complete',model_unchanged=sha(meta['model'])==meta['model_sha256'],
        interface_scalar_RMS=float(np.sqrt(np.mean(error**2))),interface_scalar_max=float(abs(error).max()),
        interface_gradient_relative_RMS=float(np.sqrt(np.mean(gradient**2))),
        hard_fixed_model_grid_error=float(abs(values[data['inside']]-data['initial'][data['inside']]).max(initial=0)),
        optimizer=read(folder/'optimization_complete.json'),limitations=meta['limitations'])
    if include_holdout:_,report['holdout_collisions']=detect(folder,data,values,'independent_holdout',include_centers=True)
    remaining=report['optimizer']['remaining_colliding_roots']
    report['collision_status']='sampled_potential_collisions_remain' if any(remaining.values()) or any(
        item['colliding_roots'] for item in report.get('holdout_collisions',{}).values()) else 'no_sampled_potential_collisions'
    print(json.dumps(dict(stage='support_field_validation',collision_status=report['collision_status'],
        remaining_potential_roots=remaining,interface_scalar_rms=report['interface_scalar_RMS'],
        scope='Potential shadow samples; actual routed solid and toolpaths require verification.')),flush=True)
    if not report['model_unchanged'] or report['hard_fixed_model_grid_error']!=0:raise RuntimeError('Model changed during support-field optimization.')
    field.volume().save(folder/'support_field.vti')
    import pyvista as pyv
    points=data['potential_points'];chosen=np.linspace(0,len(points)-1,min(30000,len(points)),dtype=int)
    preview=pyv.PolyData(points[chosen]);v,g=field.evaluate_scalars_and_gradients(points[chosen])
    preview['scalar']=v;preview['gradient']=g;preview.save(folder/'field_preview.vtp')
    meta.update(domain_sha256=sha(folder/'domain.npz'),optimized_field_sha256=sha(folder/'optimized_field.npz'))
    save(folder/'support_field.json',meta)
    save(folder/'field_report.json',report)
    return report
