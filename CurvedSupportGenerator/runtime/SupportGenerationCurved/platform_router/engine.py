"""Model-independent platform-first workflow around the promoted algorithms."""
import copy
import json
import math
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np
import pyvista as pv

from .context import RoutingContext,use_context,parameters,read,save,sha
from .acceleration import NativeDistance,FastGridGuide,counts
from .routing_field import NativeRoutingGuide
from ..field_grid import GridField


def model_signature(volume,scalar='scalar'):
    import hashlib
    mesh=pv.read(volume) if isinstance(volume,(str,Path)) else volume
    if not isinstance(mesh,pv.UnstructuredGrid) or np.any(mesh.celltypes!=10):
        raise ValueError('Platform-first routing requires the matching pure tetrahedral model field.')
    values=np.asarray(mesh.point_data[scalar],float).reshape(-1)
    if values.size!=mesh.n_points or not np.isfinite(values).all():raise ValueError('Invalid model scalar.')
    h=hashlib.sha256()
    for a in (np.asarray(mesh.points,dtype='<f8'),np.asarray(mesh.cells,dtype='<i8'),values.astype('<f8')):
        h.update(np.asarray(a.shape,dtype='<i8').tobytes());h.update(a.tobytes(order='C'))
    return h.hexdigest()


def load_support_field(folder,model_path,scalar='scalar'):
    folder=Path(folder).resolve()
    protocol=next((p for p in (folder/'support_field.json',folder/'protocol.json',folder/'accepted/protocol.json') if p.is_file()),None)
    if protocol is None:raise ValueError('Saved support field requires support_field.json or a model-linked protocol.json.')
    meta=read(protocol);original=meta.get('model') or meta.get('tetrahedral_volume')
    if meta.get('field_representation')=='unified_neural_prototype':
        raise ValueError('Experimental neural support fields cannot feed the grid router/layer adapter. Use an optimized grid support field.')
    if not original:raise ValueError('The saved support field does not identify its original model field.')
    original=Path(original);original=original if original.is_absolute() else protocol.parent/original
    if meta.get('model_sha256') and original.is_file() and sha(original)!=meta['model_sha256']:
        raise ValueError('The original model field changed since support-field preparation.')
    for key,name in (('domain_sha256','domain.npz'),('optimized_field_sha256','optimized_field.npz')):
        if meta.get(key) and sha(folder/name)!=meta[key]:raise ValueError(f'Saved support field changed: {name}')
    expected=meta.get('model_signature')
    if expected is None:
        if not original.is_file():raise ValueError(f'The support field model is missing: {original}')
        expected=model_signature(original,meta.get('scalar_name',meta.get('tetrahedral_scalar','scalar')))
    if model_signature(model_path,scalar)!=expected:
        raise ValueError('Support field and selected model differ in coordinates, tetrahedra or scalar values. Use the original matching field.')
    with np.load(folder/'domain.npz') as d:data={k:d[k].copy() for k in d.files}
    with np.load(folder/'optimized_field.npz') as d:values=d['values'].copy()
    GridField(data,values)
    return data,values,meta


def audit_graph(graph,source,guide,distance,settings,*,probes=65):
    n=int(source['skeleton_experiment']['seed_count']);edges=np.asarray(graph.lines).reshape(-1,3)[:,1:]
    p=np.asarray(graph.points);age=np.asarray(graph['contact_path_length'])
    stage=settings['contact_gradient_length']+settings['contact_transition_length'];gap=settings['clearance']
    result=dict(edges=len(edges),probes_per_edge=probes,clearance_violations=0,
        mature_field_cone_violations=0,scalar_increase_violations=0,height_descent_violations=0,
        minimum_clearance_mm=float('inf'),maximum_mature_cone_degrees=0.,examples=[])
    for start in range(0,len(edges),512):
        e=edges[start:start+512];a=p[e[:,0]];delta=p[e[:,1]]-a
        q=a[:,None]+np.linspace(0,1,probes)[None,:,None]*delta[:,None]
        d=distance(q.reshape(-1,3)).reshape(-1,probes)
        if hasattr(guide,'actual_with_distances'):values,g=guide.actual_with_distances(q.reshape(-1,3),d.ravel())
        else:values,g=guide.actual(q.reshape(-1,3))
        values=values.reshape(-1,probes);g=g.reshape(-1,probes,3)
        norm=np.linalg.norm(g,axis=2);length=np.linalg.norm(delta,axis=1)
        cosine=np.sum(-g*delta[:,None],axis=2)/np.maximum(norm*length[:,None],1e-30)
        mature=age[e[:,0]]>=stage-1e-8
        bad_clear=d.min(axis=1)<gap-2e-6
        bad_phi=(~np.isfinite(values).all(axis=1))|(np.diff(values,axis=1).max(axis=1)>1e-10)
        bad_angle=mature&((~np.isfinite(g).all(axis=(1,2)))|(norm.min(axis=1)<1e-10)|
            (cosine.min(axis=1)<np.cos(np.deg2rad(settings['angle']))-1e-5))
        bad_height=delta[:,1]>=-1e-8
        result['clearance_violations']+=int(bad_clear.sum());result['scalar_increase_violations']+=int(bad_phi.sum())
        result['mature_field_cone_violations']+=int(bad_angle.sum());result['height_descent_violations']+=int(bad_height.sum())
        result['minimum_clearance_mm']=min(result['minimum_clearance_mm'],float(d.min(initial=np.inf)))
        if mature.any():result['maximum_mature_cone_degrees']=max(result['maximum_mature_cone_degrees'],float(np.rad2deg(np.arccos(np.clip(cosine[mature].min(),-1,1)))))
        for i in np.flatnonzero(bad_clear|bad_phi|bad_angle|bad_height)[:max(0,20-len(result['examples']))]:
            result['examples'].append(dict(edge=start+int(i),source=p[e[i,0]].tolist(),target=p[e[i,1]].tolist()))
    result.update(counts(graph,n))
    if not np.isfinite(result['minimum_clearance_mm']):result['minimum_clearance_mm']=None
    result['status']='passed' if not any(result[k] for k in ('clearance_violations','scalar_increase_violations','mature_field_cone_violations','height_descent_violations')) else 'failed'
    result['scalar_source']='optimized trilinear support scalar' if guide.has_scalar_grid else 'physical height rank; model-local directions are not an exterior scalar'
    result['scope']='Finite centreline checks; protected contact stages retain their exact generated necks. Swept solids, layer backing, mechanical stability and tool travel require their own checks.'
    return result


def route_model(pipeline,output,argv,*,support_field_source=None,maximum_steps=450,
                minimum_bend_radius=5.,maximum_merge_radius=24.,stronger_merging=True,
                recovery_time_budget=5.,routing_threads=4):
    from .. import model_skeleton
    from ..model_skeleton import BoundaryTaylorField,CachedSurfaceQuery
    from ..tree_geometry import configure_geometry_threads
    from SupportGenerationVoxel.model_field import TetrahedralVertexField,load_closed_mesh
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(f'Choose a fresh support routing output: {output}')
    if not np.isfinite([minimum_bend_radius,maximum_merge_radius,recovery_time_budget]).all() or min(minimum_bend_radius,recovery_time_budget)<=0 or maximum_merge_radius<6:
        raise ValueError('Positive bend/recovery limits and merge radius >=6 mm are required.')
    inputs=copy.deepcopy(pipeline['input'])
    if inputs.get('field_mode')!='tetrahedral':raise ValueError('The platform-first interface route requires a tetrahedral model scalar.')
    model=Path(inputs['tetrahedral_volume']);scalar=inputs.get('tetrahedral_scalar','scalar')
    data=values=meta=None
    support_field_source=support_field_source or inputs.get('support_field_source')
    if support_field_source:data,values,meta=load_support_field(support_field_source,model,scalar)
    mesh=load_closed_mesh(inputs['mesh']);native=TetrahedralVertexField(model,scalar)
    query=CachedSurfaceQuery(mesh);boundary=BoundaryTaylorField(native,mesh,query)
    def guide_factory(local_native,distance,project,direction,inner,outer,**kw):
        if data is not None:return FastGridGuide(data,values,local_native)
        return NativeRoutingGuide(local_native,distance,project,direction,inner,outer,**kw)
    started=time.perf_counter();output.mkdir(parents=True)
    prefix=output/'contact_prefix';args=list(map(str,argv))
    def option(name,default):
        flag='--'+name.replace('_','-')
        return float(args[args.index(flag)+1]) if flag in args else default
    if abs(option('plate',0.))>1e-8:
        raise ValueError('Platform-first routing currently uses a zero-height build plate. Translate the model/plate to zero or choose the progressive router.')
    stage=option('contact_gradient_length',3.)+option('contact_transition_length',3.)
    prefix_steps=max(1,int(math.ceil(stage/option('step',.6)))+2)
    prefix_args=[*args,'--max-steps',str(prefix_steps)]
    # Recovery belongs after the shared corridor pass, not once per short neck.
    prefix_args=[v for v in prefix_args if v!='--reroute-stalled']
    model_skeleton.main(dict(input=inputs),prefix,argv=prefix_args,field_factory=guide_factory)
    saved=read(prefix/'summary.json');settings=saved['arguments'];settings['max_steps']=int(maximum_steps)
    if support_field_source:
        saved['input']['support_field_source']=str(Path(support_field_source).resolve())
        saved['support_field_source']=str(Path(support_field_source).resolve())
    save(prefix/'summary.json',saved)
    configure_geometry_threads(routing_threads)
    def make_field():return guide_factory(boundary,NativeDistance(mesh),lambda p:query(p)[0],[0,1,0],settings['near_band'],settings['vertical_distance'],scalar_band=0.)
    context=RoutingContext(prefix,output,saved,make_field,lambda:NativeDistance(mesh))
    with use_context(context):
        from .fast_platform_pipeline import FastPlatformTrial
        from .corridor_graph_routes import CorridorGraphTrial
        p=SimpleNamespace(output=output/'initial',zone=float(settings['merge_distance']),min_bend_radius=minimum_bend_radius,
            footing_budget=2.,fast_field=False,growing_index=True,certified_guide=False)
        first=FastPlatformTrial(p);first.run();del first
        p=parameters(output/'corridor');p.source=output/'initial';p.min_bend_radius=minimum_bend_radius
        final=CorridorGraphTrial(p);final.run();source=p.output
        graph=pv.read(source/'support_centerlines.vtp');report=read(source/'module_report.json')
        if report['unrooted_contacts'] and '--reroute-stalled' in args:
            from ..stalled_recovery import recover_stalled
            graph,report=recover_stalled(graph,report,final.field,final.distance,final.landing,[0,1,0],0.,
                radius=float(settings.get('reroute_radius',1.5)),time_budget=recovery_time_budget,maximum_components=None,
                search_mode=settings.get('reroute_search','local'),prefer_platform=bool(settings.get('prefer_platform',True)))
            recovered=output/'recovered';recovered.mkdir()
            graph.save(recovered/'support_centerlines.vtp');save(recovered/'module_report.json',report)
            updated=read(source/'summary.json');updated['skeleton_experiment']=report;save(recovered/'summary.json',updated)
            source=recovered
        del final
        if stronger_merging and counts(graph,int(saved['skeleton_experiment']['seed_count']))['platform_roots']>=2:
            from .try_depth_dependent_merging import DepthConsolidation
            p=parameters(output/'merged');p.source=source;p.index_spacing=2.;p.zone=24.;p.max_drop=60.
            p.min_bend_radius=minimum_bend_radius;p.fine_candidates=48;p.depth_step=10.
            p.maximum_radius=maximum_merge_radius;p.reach_policy='cone45'
            merger=DepthConsolidation(p);merger.run_pass();source=p.output;del merger
        graph=pv.read(source/'support_centerlines.vtp');saved=read(source/'summary.json');report=read(source/'module_report.json')
        before_audit=time.perf_counter()
        from ..grid_distance_certificate import GridDistanceCertificate
        audit_guide=make_field();certificate=GridDistanceCertificate(mesh,NativeDistance(mesh))
        # Exact signed-distance bounds avoid re-querying every distant probe.
        # Native direction blending still needs actual distances throughout
        # its complete near-model band; do not cap those at model clearance.
        band=settings['clearance'] if audit_guide.has_scalar_grid else max(settings['clearance'],audit_guide.outer_band)
        audit=audit_graph(graph,context.saved,audit_guide,lambda p:certificate.evaluate(p,band),settings)
        audit['distance_queries']=certificate.report
        audit['minimum_clearance_kind']='Exact inside query band; conservative capped distance beyond the band.'
        audit['seconds']=time.perf_counter()-before_audit
    if pipeline.get('settings',{}).get('adaptive_contact_tips',True):
        from ..contact_coverage import fit_tree_tips
        options=pipeline.get('settings',{})
        graph,tip_report=fit_tree_tips(graph,saved,mesh,native,minimum=float(options.get('minimum_tip_radius',.6)),
            maximum=float(options.get('maximum_tip_radius',2.5)),transition=float(options.get('tip_transition_length',5.)))
        save(output/'tip_sizing_report.json',tip_report)
        report['adaptive_tip_coverage']=tip_report
    graph.save(output/'support_centerlines.vtp')
    report.update(contact_selection=context.saved['skeleton_experiment'].get('contact_selection'),routing_mode='platform_first',field_source='conservative_grid' if data is not None else 'model_local_direction',
        support_field_source=str(support_field_source) if support_field_source else None,
        routing_audit=audit,complete_workflow_seconds=time.perf_counter()-started,
        contact_prefix_regenerated=True,maximum_components=None,
        limitations=report.get('limitations',[])+[audit['scope']])
    saved['skeleton_experiment']=report
    save(output/'module_report.json',report);save(output/'summary.json',saved);save(output/'routing_audit.json',audit)
    save(output/'workflow.json',dict(schema_version=1,algorithm='platform_first',source=str(source),
        model_signature=model_signature(model,scalar),model=str(model),support_field_source=str(support_field_source) if support_field_source else None,
        stronger_merging=stronger_merging,minimum_bend_radius_mm=minimum_bend_radius,
        maximum_merge_radius_mm=maximum_merge_radius,plate_mm=0.,seconds=time.perf_counter()-started))
    print(json.dumps(dict(stage='platform_first_complete',rooted_contacts=report['rooted_contacts'],
        unrooted_contacts=report['unrooted_contacts'],audit=audit['status'],output=str(output))),flush=True)
    return report
