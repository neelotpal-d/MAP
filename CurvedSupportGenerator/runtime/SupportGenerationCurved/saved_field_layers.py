"""Slice an existing optimized trilinear field and rooted forest without refitting."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import json
import multiprocessing
import time
import numpy as np
import pyvista as pv
from . import tree_geometry as tg
from .field_grid import GridField
from .platform_router.context import read,save,sha
from .platform_router.engine import load_support_field

_STATE=None


def reinforcement_rows(posts,mesh,plate,clearance,base_height=.6,base_factor=1.8):
    """Bed-connected capsules with a distance bound for their entire volume."""
    if not isinstance(posts,list):raise ValueError('reinforcement_posts must be an array.')
    if not posts:return np.empty((0,9)),np.empty((0,3)),[]
    from .platform_router.acceleration import NativeDistance
    distance=NativeDistance(mesh);rows=[];reports=[]
    for post in posts:
        x,z=map(float,post['center_xz']);top=float(post['top_y']);radius=float(post['radius'])
        if not np.isfinite([x,z,top,radius,base_height,base_factor]).all() or min(radius,base_height)<=0 or base_factor<1 or top<=plate+base_height:
            raise ValueError('Invalid bed reinforcement dimensions.')
        count=max(2,int(np.ceil((top-plate-base_height)/.05))+1)
        ys=np.linspace(plate+base_height,top,count);points=np.column_stack((np.full(count,x),ys,np.full(count,z)))
        # Signed distance is 1-Lipschitz: sample spacing/2 bounds every
        # centreline point; subtracting radius then covers the whole capsule.
        bound=float(distance(points).min())-radius-float(np.diff(ys).max())/2-1e-5
        base_bound=float(np.asarray(mesh.points)[:,1].min())-(plate+base_height)
        if min(bound,base_bound)<clearance:raise ValueError('Bed reinforcement would be clipped by the model clearance.')
        rows.extend([[x,plate+base_height,z,x,top,z,radius,radius,0],
            [x,plate,z,x,plate+base_height,z,radius*base_factor,radius,3]])
        reports.append(dict(post=post,body_clearance_lower_bound_mm=bound,base_clearance_lower_bound_mm=base_bound))
    return np.asarray(rows),np.tile([0.,1.,0.],(len(rows),1)),reports


def geometry(pipeline):
    from .skeleton_geometry import load_skeleton,sweep_rows
    from SupportGenerationVoxel.model_field import load_closed_mesh
    inputs=pipeline['input'];s=pipeline.get('settings',{})
    tg.configure_geometry_threads(int(s.get('vtk_threads_per_worker',1)))
    graph,rooted=load_skeleton(inputs['skeleton_source'],skip_unrooted=bool(s.get('skip_unrooted_contacts',False)))
    mesh=load_closed_mesh(inputs['mesh'])
    separated=rooted.get('routing_mode')=='separated-posts'
    if separated:
        from .separated_posts import shape_for
        shape=shape_for(graph,rooted,mesh,float(s.get('contact_gap',.4)),float(s.get('platform_base_height',.6)))
        roots=np.empty((0,3))
        sizing={k:rooted[k] for k in ('tip_radius_mm','neck_length_mm','body_radius_mm','taper_length_mm','minimum_spacing_mm')}
    else:
        rows,normals,roots,sizing=sweep_rows(graph,rooted['seed_count'],
            base_radius=float(s.get('branch_diameter',2.4))/2,tip_radius=float(s.get('tip_radius',.6)),
            maximum_radius=float(s.get('maximum_trunk_diameter',5.))/2,direction=(0,1,0),plate=float(s.get('plate',0)),
            platform_base_radius_factor=float(s.get('platform_base_radius_factor',1.8)),
            platform_base_height=float(s.get('platform_base_height',.6)),branch_sizing=s.get('branch_sizing','bambu'),
            branch_diameter_angle=float(s.get('branch_diameter_angle',5.)),planar_platform_bases=bool(s.get('planar_platform_bases',True)),
            maximum_platform_radius=float(s.get('maximum_platform_trunk_diameter',8.))/2)
        extra,extra_normals,certificates=reinforcement_rows(s.get('reinforcement_posts',[]),mesh,float(s.get('plate',0)),
            max(float(s.get('side_clearance',.4)),float(s.get('contact_gap',.4))),
            float(s.get('platform_base_height',.6)),float(s.get('platform_base_radius_factor',1.8)))
        if len(extra):
            if not s.get('planar_platform_bases',True):raise ValueError('Bed reinforcements require planar platform bases.')
            rows=np.vstack((rows,extra));normals=np.vstack((normals,extra_normals))
            sizing['bed_reinforcements']=certificates
        shape=tg.TaperedTree(rows,mesh,(0,1,0),float(s.get('plate',0)),
            float(s.get('side_clearance',.4)),float(s.get('contact_gap',.4)),terminal_normals=normals,
            terminal_cap='rounded',distance_backend='certified',prune_pairs=bool(s.get('trim_pair_pruning',True)))
    from .trim_tolerance import configured
    shape.trim_boundary_tolerance_mm=configured(s.get('trim_boundary_tolerance_mm',0.))
    shape.model_contact_roots=roots;shape.platform_roots=[]
    from .grid_distance_certificate import GridDistanceCertificate
    shape.distance_certificate=GridDistanceCertificate(shape.mesh,shape.distance)
    backend='exact arrays';accelerator=None
    if not separated and s.get('native_acceleration',True) and shape.prune_pairs:
        try:
            from .exact_support_bvh import ExactSupportBVH,LexCachedLayerField
            accelerator=ExactSupportBVH(shape,threads=int(s.get('vtk_threads_per_worker',1)))
        except (FileNotFoundError,OSError):
            # Exact vectorized fallback keeps the geometry/acceptance policy.
            print(json.dumps(dict(stage='native_acceleration_unavailable',backend=backend)),flush=True)
        else:
            shape.evaluate=accelerator.evaluate
            shape._exact_bvh=accelerator
            tg.CachedLayerField=LexCachedLayerField
            backend='exact BVH / '+accelerator.cache_backend+' point cache'
    return shape,sizing,rooted,accelerator,backend


def initialize_worker(pipeline,output):
    global _STATE
    from .planar_bases import settings_for
    output=Path(output);s=pipeline.get('settings',{})
    shape,sizing,rooted,fast,backend=geometry(pipeline)
    field=GridField.load(pipeline['input']['support_field_source'])
    # Fixed-axis paths require a single height chart; general five-axis layer
    # normals may legitimately turn through the build-plane projection.
    folder=Path(pipeline['input']['support_field_source'])
    protocol=next((p for p in (folder/'support_field.json',folder/'protocol.json',folder/'accepted/protocol.json') if p.is_file()),None)
    metadata=read(protocol) if protocol is not None else {}
    mode=metadata.get('collision_mode')
    preserve_chart=(mode=='fixed_axis') if mode is not None else metadata.get('model_generator_axes')==3
    _STATE=dict(shape=shape,field=field,fast=fast,output=output,settings=s,
        preserve_projected_chart=preserve_chart,
        active=pv.read(output/'support_field_grid.vtu'),schedule=np.loadtxt(output/'main_scalar_values.txt').reshape(-1),
        regions=read(output/'support_role_regions.json'),roles=read(output/'support_role_indices.json'),bases=settings_for(shape))
    print(json.dumps(dict(stage='layer_worker_ready',backend=backend)),flush=True)


def _generated_layer(state,index,value,correct):
    from .layer_quality import field_normal_consistency,refined_field_grid
    attempts=[]
    verify=(not state.get('preserve_projected_chart',False) and isinstance(state['field'],GridField))
    for factor in (1,2,4):
        active=state['active']
        if factor>1:
            cache=state.setdefault('refined_contour_sources',{})
            if factor not in cache:cache[factor]=refined_field_grid(state['field'],factor)
            active=cache[factor]
        generated=tg.extract_one_layer(state['shape'],active,index,value,state['output']/'tree_layers',1.,.2,'surface_scalar',
            layer_compression='lz4',planar_base_settings=state['bases'],adaptive_layer_tolerance=float(state['settings'].get('adaptive_layer_tolerance',.05)),
            role_regions=state['regions'],layer_role_indices=state['roles'],spatial_platform=True,_return_surface=True)
        if generated['vertex_bad'] or generated['centroid_bad'] or generated.get('geometry_bad_faces',0):raise ValueError(f'Layer {index}: geometry validation failed.')
        if not verify or correct:return generated,dict(factor=factor,attempts=attempts)
        surface=generated['surface'];normal=field_normal_consistency(surface.points,surface.regular_faces,state['field'])
        attempts.append(dict(factor=factor,field_normal_consistency=normal))
        if normal['valid']:
            return generated,dict(factor=factor,spacing_mm=state['field'].spacing/factor,attempts=attempts,
                method='Recontour the same original trilinear field; no field optimization, smoothing or extrapolation.')
        print(json.dumps(dict(stage='layer_normal_recontour',index=index,factor=factor,
            warning='Extracted layer conflicts with saved field normals; retrying a finer source contour.',diagnostics=normal)),flush=True)
    raise ValueError(f'Layer {index}: field-normal consistency failed after bounded contour refinements (1, 2, 4). Inspect the saved field and layer geometry; no normals were forced or check bypassed. Last check: {normal}')


def extract(index):
    from .nonplanar import slice_error
    from .layer_projection import project_layer
    state=_STATE;s=state['settings'];out=state['output'];schedule=state['schedule'];value=float(schedule[index]);t=time.perf_counter();cpu_started=time.process_time()
    correct=s.get('correct_layer_scalar',True)
    if not isinstance(correct,bool):raise ValueError('correct_layer_scalar must be boolean.')
    generated,contour_refinement=_generated_layer(state,index,value,correct)
    surface=generated.pop('surface');record=generated['manifest'];record.update(slice_error(surface,state['field'],value))
    # New interface jobs default to mesh geometry. Absent settings retain the
    # original exact-field contract for saved CLI jobs/reproducibility.
    correct=s.get('correct_layer_scalar',True)
    if not isinstance(correct,bool):raise ValueError('correct_layer_scalar must be boolean.')
    gaps=np.diff(schedule);budget=float(.25*min(gaps[max(0,index-1)],gaps[min(len(gaps)-1,index)]))
    # Geometric source binding and projected root lifting need a physical
    # accuracy bound in addition to a fraction of the scalar layer interval.
    if correct and surface.n_cells:
        points=np.asarray(surface.points)
        minimum_norm=float('inf');minimum_y=float('inf')
        for start in range(0,len(points),25000):
            _,gradient=state['field'].evaluate_scalars_and_gradients(points[start:start+25000])
            minimum_norm=min(minimum_norm,float(np.linalg.norm(gradient,axis=1).min()))
            minimum_y=min(minimum_y,float(gradient[:,1].min()))
        tolerance=float(s.get('layer_field_tolerance_mm',.005))
        if not np.isfinite(tolerance) or tolerance<=0:raise ValueError('Layer field tolerance must be positive and finite.')
        if not np.isfinite(minimum_norm) or minimum_norm<=1e-12:raise ValueError('Support layer has a degenerate field gradient.')
        budget=min(budget,tolerance*minimum_norm)
        # Only fixed-axis paths lift through a vertical height chart.
        # Five-axis normal paths retain the normal-distance accuracy bound.
        if state.get('preserve_projected_chart',False) and minimum_y>1e-9:
            budget=min(budget,.02*minimum_y)
    projected=False
    if correct and surface.n_cells and max(record['vertex_scalar_error_max'],record['interior_scalar_error_max'])>budget:
        projected=True
        record=project_layer(record,state['shape'],state['field'],out/'tree_layers_refined',state['regions'],surface=surface,schedule=schedule,scalar_error_budget=budget,maximum_refinements=12,
            preserve_projected_chart=state.get('preserve_projected_chart',False))
    exceeds=max(record['vertex_scalar_error_max'],record['interior_scalar_error_max'])>budget
    if correct and exceeds:
        raise ValueError(f'Layer {index}: trilinear isosurface error exceeds {budget:g}.')
    record.update(layer_scalar_correction_enabled=correct,layer_scalar_error_budget=budget,
        layer_scalar_error_exceeds_budget=bool(exceeds),layer_field_geometry_validated=bool(correct))
    if not correct and exceeds:
        print(json.dumps(dict(stage='layer_scalar_approximation',index=index,warning='Using extracted mesh; no exact scalar-level claim.',
            vertex_error=record['vertex_scalar_error_max'],interior_error=record['interior_scalar_error_max'],scalar_budget=budget)),flush=True)
    from .layer_quality import mesh_quality,field_normal_consistency
    final_surface=pv.read(record['file']) if projected else surface
    record['mesh_quality']=mesh_quality(final_surface.points,final_surface.regular_faces)
    record['contour_source_refinement']=contour_refinement
    if not state.get('preserve_projected_chart',False) and isinstance(state['field'],GridField):
        checked=field_normal_consistency(final_surface.points,final_surface.regular_faces,state['field'])
        record['field_normal_consistency']=checked
        if not checked['valid']:raise ValueError(f'Layer {index}: final mesh/field normal consistency failed: {checked}')
    if record['mesh_quality']['degenerate_faces']:raise ValueError(f'Layer {index}: degenerate final triangle mesh.')
    result=dict(record=record,seconds=time.perf_counter()-t,cpu_seconds=time.process_time()-cpu_started,extraction_timing=generated['timing'])
    save(out/'layer_metadata'/f'{index}.json',result);return result


def run_job(pipeline,output):
    from SupportGenerationVoxel.model_field import TetrahedralVertexField
    from SupportGenerationVoxel.contacts import derive_layer_values
    from .layers import extend_schedule
    from .online_slicing import plan_from_current_layers
    from .layer_roles import role_indices
    from .spatial_roles import region_specs
    from .planar_bases import settings_for,export_bases
    from .layer_projection import certified_planar_bases
    from .compatible_export import export_compatible
    from .layer_viewers import LayerViewerWriter
    started=time.perf_counter();inputs=pipeline['input'];s=pipeline.get('settings',{});output=Path(output).resolve()
    from .trim_tolerance import configured
    trim_budget=configured(s.get('trim_boundary_tolerance_mm',0.))
    if not isinstance(s.get('correct_layer_scalar',True),bool):raise ValueError('correct_layer_scalar must be boolean.')
    for key,default in (('layer_workers',4),('vtk_threads_per_worker',1)):
        value=s.get(key,default)
        if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError(f'{key} must be a positive integer.')
    gap=float(s.get('contact_gap',.4));side=float(s.get('side_clearance',.4))
    if not np.isfinite([gap,side]).all() or gap<0 or side!=gap:
        raise ValueError('Saved-field support layers currently require finite equal side and contact clearances.')
    source=Path(inputs['skeleton_source']);audit=source/'routing_audit.json'
    saved=read(source/'summary.json');routing_report=read(source/'module_report.json')
    separated=routing_report.get('routing_mode')=='separated-posts'
    if Path(saved['mesh']).resolve()!=Path(inputs['mesh']).resolve():raise ValueError('The saved tree belongs to a different model surface.')
    if not np.isclose(routing_report.get('clearance_mm',np.nan),gap,rtol=0,atol=1e-8):raise ValueError('Layer clearance must match the saved tree.')
    if audit.is_file() and read(audit)['status']!='passed':raise ValueError('Saved routing audit failed; inspect and repair the tree before layers.')
    data,values,meta=load_support_field(inputs['support_field_source'],inputs['tetrahedral_volume'],inputs.get('tetrahedral_scalar','scalar'))
    if output.exists():raise FileExistsError(f'Choose a fresh layer output: {output}')
    output.mkdir(parents=True)
    for name in ('tree_layers','tree_layers_refined','layer_metadata'):(output/name).mkdir()
    save(output/'rebuild_pipeline.json',pipeline)
    field=GridField(data,values);active=field.volume().cast_to_unstructured_grid();active.points=np.asarray(active.points,dtype=float)
    active.save(output/'support_field_grid.vtu')
    native=TetrahedralVertexField(inputs['tetrahedral_volume'],inputs.get('tetrahedral_scalar','scalar'))
    from .compatible_export import model_volume
    model_volume(inputs['mesh'],pv.read(inputs['mesh']),inputs['tetrahedral_volume'],_loaded_volume=native.volume)
    shape,sizing,rooted,fast,backend=geometry(pipeline)
    if inputs.get('shared_layer_values'):
        schedule=np.loadtxt(inputs['shared_layer_values']).reshape(-1);schedule_report=dict(mode='explicit shared schedule',seconds=0.)
    else:
        if inputs.get('layer_values'):
            initial=np.loadtxt(inputs['layer_values']).reshape(-1)
        else:
            ids=np.linspace(0,native.volume.n_points-1,min(20000,native.volume.n_points),dtype=int)
            p=np.asarray(native.volume.points)[ids];v,g=native.evaluate_scalars_and_gradients(p)
            initial=derive_layer_values(v,g,float(s.get('layer_spacing',.2)),0,10000,256)
        initial,_=extend_schedule(initial,[float(values.min()),float(values.max())])
        if s.get('slicing_mode','online')=='online' and s.get('include_support_gradients',True):
            projections=shape.a@shape.direction;base_points=shape.a[projections<=projections.min()+.6]
            schedule,schedule_report=plan_from_current_layers(native.volume,active,native,field,shape,initial[0],initial[-1],
                base_points=base_points,model_spacing=float(s.get('layer_spacing',.2)),support_spacing=float(s.get('support_layer_spacing',.2)),
                maximum_support_spacing=float(s.get('maximum_support_layer_height',.5)))
        else:
            from .support_schedule import sample_support_gradients,refine_support_schedule
            if s.get('include_support_gradients',True):
                v,g,probes=sample_support_gradients(shape,field,spacing=float(s.get('support_gradient_probe_spacing',2.)))
                schedule,schedule_report=refine_support_schedule(initial,v,g,target_spacing=float(s.get('support_layer_spacing',.2)),
                    maximum_spacing=float(s.get('maximum_support_layer_height',.5)))
            else:schedule=initial;schedule_report=dict(mode='model schedule only')
    schedule=np.asarray(schedule,dtype=float)
    if len(schedule)<2 or not np.isfinite(schedule).all() or np.any(np.diff(schedule)<=0):raise ValueError('Require at least two finite increasing shared scalar levels.')
    np.savetxt(output/'main_scalar_values.txt',schedule,fmt='%.17g');np.savetxt(output/'support_scalar_values.txt',schedule,fmt='%.17g')
    save(output/'support_schedule_report.json',schedule_report)
    ids=np.arange(len(schedule));band=int(s.get('contact_layers_below',3))
    roles=role_indices(schedule,field.evaluate_scalars(shape.b[shape.terminal]),platform_layer=-1,contact_layers_below=band)
    feet=field.evaluate_scalars(shape.model_contact_roots);roles['model_base_layer_indices']=sorted(set(int(i) for i in np.searchsorted(schedule,feet) if i<len(schedule)))
    roles['contact_layer_indices']=sorted(set(roles['contact_layer_indices'])|set(roles['model_base_layer_indices']))
    regions=[r for r in region_specs(schedule,ids,shape,field,contact_layers_below=band,margin=1.)
        if r['flag']!=1 or not s.get('planar_platform_bases',True)]
    save(output/'support_role_indices.json',roles);save(output/'support_role_regions.json',regions)
    bases=settings_for(shape)
    if bases:
        certified_planar_bases(shape,bases,float(s.get('platform_base_layer_height',.2)),schedule,output)
    else:save(output/'planar_platform_base_manifest.json',[])
    # Display sampling is independent of the original voxel field and adaptive layer trim.
    spacing=float(s.get('support_preview_spacing',1.))
    solid=tg.sample_solid(shape,spacing=min(spacing,.2) if separated else spacing)
    solid.contour([0.],scalars='support_solid').triangulate().save(output/'support_solid.vtp')
    save(output/'tree_geometry_report.json',dict(contact_gap=shape.contact_gap,side_clearance=shape.side_gap,
        plate=shape.plate,sizing=sizing,rooted=rooted,support_role_regions=regions,backend=backend))
    if fast:fast.close()
    if not s.get('all_layers',True):ids=np.unique(np.linspace(0,len(schedule)-1,min(int(s.get('preview_layers',5)),len(schedule)),dtype=int))
    workers=min(int(s.get('layer_workers',4)),len(ids));results=[]
    setup_seconds=time.perf_counter()-started
    extraction_started=time.perf_counter()
    if workers<=1:
        initialize_worker(pipeline,output)
        for i in ids:
            results.append(extract(int(i)));print(json.dumps(dict(stage='layer_complete',completed=len(results),total=len(ids),index=int(i),seconds=results[-1]['seconds'])),flush=True)
    else:
        with tg.worker_blas_environment(1):
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker,initargs=(pipeline,str(output))) as pool:
                futures={pool.submit(extract,int(i)):int(i) for i in ids}
                for future in as_completed(futures):
                    try:r=future.result()
                    except Exception as error:
                        # Do not finish hundreds of queued layers after the
                        # first failed one. Retain completed diagnostics and
                        # let only already-running tasks finish their writes.
                        cancelled=sum(pending.cancel() for pending in futures)
                        print(json.dumps(dict(stage='layer_failed',index=futures[future],error=str(error),cancelled_pending_layers=cancelled)),flush=True)
                        raise
                    results.append(r);print(json.dumps(dict(stage='layer_complete',completed=len(results),total=len(ids),index=r['record']['layer'],seconds=r['seconds'])),flush=True)
    extraction_seconds=time.perf_counter()-extraction_started
    export_started=time.perf_counter()
    records=sorted((r['record'] for r in results),key=lambda r:r['layer']);save(output/'tree_layer_manifest_refined.json',records)
    approximations=[dict(layer=r['layer'],**r['trim_boundary_approximation']) for r in records
        if r.get('trim_boundary_approximation') and r['trim_boundary_approximation']['accepted_faces']]
    trim_summary=dict(tolerance_mm=trim_budget,accepted_layers=len(approximations),
        accepted_faces=sum(r['accepted_faces'] for r in approximations),
        maximum_deviation_bound_mm=max((r['maximum_deviation_bound_mm'] for r in approximations),default=0.),layers=approximations)
    save(output/'trim_approximation_report.json',trim_summary)
    from .layer_quality import summarize as summarize_layer_quality
    save(output/'support_layer_quality_report.json',summarize_layer_quality(records))
    viewers=LayerViewerWriter(output)
    report=export_compatible(inputs['mesh'],pv.read(inputs['mesh']),schedule,ids,records,output,evaluator=native,
        field_mode='tetrahedral',volume_path=inputs['tetrahedral_volume'],native_model_layers=True,_surface_callback=viewers.add)
    if viewers.model_overrides:
        manifest=read(output/'model_layer_manifest.json');viewers.update_model_records(manifest);save(output/'model_layer_manifest.json',manifest)
    report['viewer']=viewers.finish();save(output/'export_report.json',report)
    compatible=read(output/'compatible_export.json');compatible.update(contact_gap=float(s.get('contact_gap',.4)),side_clearance=float(s.get('side_clearance',.4)),
        plate=float(s.get('plate',0)),nominal_layer_height=float(s.get('layer_spacing',.2)),complete_layers=bool(s.get('all_layers',True)),
        support_field_source=str(inputs['support_field_source']),rebuild_pipeline=str(output/'rebuild_pipeline.json'),
        layer_scalar_correction_enabled=s.get('correct_layer_scalar',True),trim_boundary_approximation=trim_summary)
    if separated:
        compatible['experimental_overhang_only']=True
        compatible['complete_layers']=False
    save(output/'compatible_export.json',compatible)
    summary=dict(status='complete',pipeline=pipeline,shared_levels=len(schedule),exported_indices=ids.tolist(),
        workers=workers,native_threads=int(s.get('vtk_threads_per_worker',1)),backend=backend,
        vtk_threads_effective=tg.effective_geometry_threads(int(s.get('vtk_threads_per_worker',1))),
        layer_scalar_correction_enabled=s.get('correct_layer_scalar',True),trim_boundary_approximation=trim_summary,
        field_representation='Original trilinear voxels; no transfer or tetrahedralization',seconds=time.perf_counter()-started,
        timings=dict(setup_seconds=setup_seconds,extraction_wall_seconds=extraction_seconds,
            export_and_reports_seconds=time.perf_counter()-export_started,
            sum_worker_layer_seconds=sum(r['seconds'] for r in results),
            sum_worker_cpu_seconds=sum(r.get('cpu_seconds',0.) for r in results),
            scope='Backend setup, extraction pool including startup/shutdown, then component export/viewer/reports. Excludes GUI loading.'),
        support_field_sha256=sha(Path(inputs['support_field_source'])/'optimized_field.npz'),excluded_contact_ids=rooted.get('excluded_contact_ids',[]))
    save(output/'summary.json',summary);return summary
