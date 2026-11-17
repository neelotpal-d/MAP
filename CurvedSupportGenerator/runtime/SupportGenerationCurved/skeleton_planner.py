"""Standalone Swan skeleton growth and preview; produces no solids or layers."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtimeEnv import configure_openmp_runtime
configure_openmp_runtime()
import argparse
import json
import time
import numpy as np
import pyvista as pv
from SupportGenerationCurved.config import load_config
from SupportGenerationCurved.output import prepare_output
from SupportGenerationCurved.tree_geometry import load_preserved_tree,configure_geometry_threads
from SupportGenerationCurved.transferred_field import TransferredSupportField
from SupportGenerationCurved.progressive_skeleton import grow


class SkeletonField:
    """Batched finite differences of the actual blended support scalar field."""
    def __init__(self,blended,h=.1):
        self.blended=blended;self.h=h;self.cache={};self.evaluated=0

    def __call__(self,points):
        points=np.asarray(points,dtype=float).reshape(-1,3)
        unique,inverse=np.unique(points,axis=0,return_inverse=True)
        keys=[p.tobytes() for p in unique]
        missing=[i for i,key in enumerate(keys) if key not in self.cache]
        if missing:
            query=unique[missing];offsets=np.vstack((np.zeros((1,3)),np.eye(3)*self.h,-np.eye(3)*self.h))
            stencil=(query[:,None,:]+offsets[None,:,:]).reshape(-1,3)
            scalar=self.blended.evaluate_scalars(stencil).reshape(-1,7)
            gradient=(scalar[:,1:4]-scalar[:,4:7])/(2*self.h)
            for i,value,g in zip(missing,scalar[:,0],gradient):self.cache[keys[i]]=(value,g)
            self.evaluated+=len(missing)
        values=np.asarray([self.cache[key][0] for key in keys])
        gradients=np.asarray([self.cache[key][1] for key in keys]).reshape(-1,3)
        return values[inverse],gradients[inverse]


def show(mesh,skeleton,screenshot=None,mode='near-model'):
    plotter=pv.Plotter(off_screen=screenshot is not None,window_size=(1200,900))
    plotter.set_background('#20242b');plotter.add_mesh(mesh,color='#88bbee',opacity=.22)
    if skeleton.n_cells:plotter.add_mesh(skeleton,color='#60d6ab',line_width=4,render_lines_as_tubes=True)
    kinds=np.asarray(skeleton['node_kind'])
    for kind,color in ((1,'#ff8c42'),(2,'#2d9cdb'),(3,'#ff3344'),(4,'#ffdc55')):
        selected=kinds==kind
        if selected.any():plotter.add_points(skeleton.points[selected],color=color,point_size=7,render_points_as_spheres=True)
    fallbacks=np.asarray(skeleton.point_data.get('landing_is_fallback',np.zeros(skeleton.n_points)),dtype=bool)
    if fallbacks.any():plotter.add_points(skeleton.points[fallbacks],color='#c07aff',point_size=10,render_points_as_spheres=True)
    title={'near-model':'model-local directions, vertical body',
           'field-guided':'transferred scalar gradient, field-preferred growth',
           'transferred':'transferred field cone, vertical preference'}[mode]
    legend='Orange: contacts | blue: roots | yellow: junctions | red: stalled'
    if fallbacks.any():legend+='\nViolet: fallback landing below preferred orientation threshold'
    plotter.add_text(f'Skeleton only | {title}\n{legend}',font_size=11,color='white')
    plotter.add_axes();plotter.camera_position='xy';plotter.camera.up=[0,1,0]
    plotter.reset_camera()
    plotter.show(screenshot=str(screenshot) if screenshot else None)


def main():
    started=time.perf_counter()
    p=argparse.ArgumentParser(description=__doc__)
    base=Path(__file__).resolve().parent
    p.add_argument('--run',type=Path,default=base/'benchmarks/swan_rounded_all/run_001')
    p.add_argument('--config',type=Path,default=base/'config.swan_support.json')
    p.add_argument('--output',type=Path)
    p.add_argument('--routing-mode',choices=('near-model','field-guided','transferred'),default='near-model',
        help='near-model: current local guide and vertical body; field-guided: follow the extended scalar gradient; transferred: legacy extended-field cone with vertical preference.')
    p.add_argument('--near-band',type=float,default=1.,help='Use native directions at actual points within this model distance.')
    p.add_argument('--vertical-distance',type=float,default=3.,help='Beyond this distance use only the global build direction.')
    p.add_argument('--model-query-backend',choices=('auto','vtk','open3d'),default='auto')
    p.add_argument('--model-landing-check',choices=('upward','none'),default='upward')
    p.add_argument('--landing-min-up-dot',type=float,default=.15,help='Required outward-normal alignment; preferred threshold when fallback is enabled.')
    p.add_argument('--landing-fallback-min-up-dot',type=float,help='Optional HARD lower bound when preferred landing attempts fail; disabled by default.')
    p.add_argument('--landing-fallback-min-length',type=float,default=1.2,help='Minimum accumulated contact-to-front path length for fallback, in mm.')
    p.add_argument('--landing-footprint-radius',type=float,default=0.,help='Optional six-point backing ring in mm; zero checks only the centre.')
    p.add_argument('--root-contact-mode',choices=('adaptive','legacy'),default='adaptive',help='Size model-root contacts from small backing stencils.')
    p.add_argument('--root-contact-radius',type=float,default=2.5)
    p.add_argument('--root-min-contact-radius',type=float,default=.6)
    p.add_argument('--clean-output',action='store_true')
    p.add_argument('--step',type=float,default=.6)
    p.add_argument('--angle',type=float,default=45.)
    p.add_argument('--merge-distance',type=float,default=6.)
    p.add_argument('--clearance-escape-band',type=float,default=1.5,help='Near-model band in mm where a blocked front may prefer the local outward normal; 0 disables this preference.')
    p.add_argument('--clearance-escape-weight',type=float,default=.45,help='Outward-normal weight inside the clearance-escape band, before cone projection.')
    p.add_argument('--max-steps',type=int,default=450)
    p.add_argument('--reroute-stalled',action='store_true',help='Try a bounded nearby landing search after normal growth.')
    p.add_argument('--reroute-source',type=Path,help='Recover only this saved failed graph instead of growing a new tree; uses a separate output.')
    p.add_argument('--shift-source',type=Path,help='Copy a saved near-model skeleton and apply only gradual trunk shifting; no growth or stalled recovery.')
    p.add_argument('--trunk-side-shift',type=float,default=0.,help='Maximum gradual model-root shift in mm, fading to zero at the first fork; 0 disables.')
    p.add_argument('--reroute-radius',type=float,default=1.5)
    p.add_argument('--reroute-search',choices=('local','wide'),default='local',help='wide: sample lower/distant patches, rewind up to 64 nodes on the failed trunk, and try bounded field-constrained paths.')
    p.add_argument('--reroute-time-budget',type=float,default=5.,help='Soft recovery time budget after loading; checked between bounded query batches.')
    p.add_argument('--preview',action='store_true')
    p.add_argument('--preview-only',action='store_true')
    p.add_argument('--screenshot',type=Path)
    a=p.parse_args()
    shift_only=a.shift_source is not None
    if not np.isfinite(a.trunk_side_shift) or a.trunk_side_shift<0:
        p.error('--trunk-side-shift must be finite and nonnegative.')
    if shift_only:
        if a.reroute_source is not None or a.reroute_stalled or a.reroute_search!='local':
            p.error('--shift-source is separate from stalled recovery options.')
        if a.trunk_side_shift<=0:p.error('--shift-source requires --trunk-side-shift greater than zero.')
        a.reroute_source=a.shift_source
    if a.trunk_side_shift>0 and (a.routing_mode!='near-model' or a.model_landing_check!='upward'):
        p.error('Gradual trunk shift requires near-model routing and upward landing checks.')
    if a.reroute_search=='wide':
        if not any(x=='--reroute-radius' or x.startswith('--reroute-radius=') for x in sys.argv[1:]):a.reroute_radius=20.
        if not any(x=='--reroute-time-budget' or x.startswith('--reroute-time-budget=') for x in sys.argv[1:]):a.reroute_time_budget=8.
        a.reroute_stalled=not shift_only
    if not np.isfinite(a.reroute_radius) or not np.isfinite(a.reroute_time_budget) or min(a.reroute_radius,a.reroute_time_budget)<=0:
        p.error('Recovery radius and time budget must be finite and positive.')
    if not np.isfinite([a.clearance_escape_band,a.clearance_escape_weight]).all() or a.clearance_escape_band<0 or not 0<=a.clearance_escape_weight<=1:
        p.error('Clearance escape band must be nonnegative and weight must lie in [0,1].')
    recovered_source=None
    if a.reroute_source is not None:
        source=a.reroute_source.resolve()
        recovered_source=json.loads((source/'module_report.json').read_text())
        if recovered_source.get('routing_mode')!='near-model':p.error('Saved-graph recovery currently supports near-model routing only.')
        settings=recovered_source.get('model_landing_check',{})
        for flag,key,attribute in (('--landing-min-up-dot','min_up_dot','landing_min_up_dot'),
            ('--landing-fallback-min-up-dot','hard_min_up_dot','landing_fallback_min_up_dot'),
            ('--landing-fallback-min-length','fallback_minimum_path_length_mm','landing_fallback_min_length'),
            ('--root-min-contact-radius','minimum_contact_radius_mm','root_min_contact_radius'),
            ('--landing-footprint-radius','footprint_radius_mm','landing_footprint_radius')):
            if key in settings and not any(x==flag or x.startswith(flag+'=') for x in sys.argv[1:]):setattr(a,attribute,settings[key])
        if not any(x=='--root-contact-mode' or x.startswith('--root-contact-mode=') for x in sys.argv[1:]):
            a.root_contact_mode='adaptive' if settings.get('adaptive_contact_radius_mm') is not None else 'legacy'
        for flag,key,attribute,container in (('--root-contact-radius','adaptive_contact_radius_mm','root_contact_radius',settings),
            ('--near-band','near_band_mm','near_band',recovered_source),
            ('--vertical-distance','vertical_distance_mm','vertical_distance',recovered_source)):
            if container.get(key) is not None and not any(x==flag or x.startswith(flag+'=') for x in sys.argv[1:]):
                setattr(a,attribute,container[key])
        a.reroute_stalled=not shift_only
    if a.landing_fallback_min_up_dot is not None:
        if a.model_landing_check!='upward':p.error('Landing fallback requires --model-landing-check upward.')
        if not 0<a.landing_fallback_min_up_dot<=a.landing_min_up_dot:
            p.error('Fallback minimum must be positive and no greater than --landing-min-up-dot.')
    if a.output is None:
        directories={'near-model':'swan_skeleton_near_model','field-guided':'swan_skeleton_field_guided','transferred':'swan_skeleton'}
        a.output=base/'benchmarks'/('swan_skeleton_side_shift' if a.trunk_side_shift>0 else 'swan_skeleton_recovered' if a.reroute_source is not None else directories[a.routing_mode])
    if a.reroute_source is not None and (a.output.resolve()==source or a.output.resolve() in source.parents or source in a.output.resolve().parents):
        p.error('Recovery output must be separate from its saved source graph.')
    if a.reroute_stalled and a.model_landing_check!='upward':p.error('Stalled recovery requires upward model landing checks.')
    if a.preview_only:
        summary=json.loads((a.output/'summary.json').read_text())
        show(pv.read(summary['mesh']),pv.read(a.output/'support_centerlines.vtp'),a.screenshot,
             summary['skeleton_experiment'].get('routing_mode','transferred'));return
    from SupportGenerationVoxel.model_field import TrustedModelField
    summary=json.loads((a.run/'summary.json').read_text());cfg=load_config(a.config)
    for source in (a.run,Path(summary['arguments']['tree_source'])):
        if a.output.resolve()==source.resolve() or a.output.resolve() in source.resolve().parents:
            p.error('Output must not contain saved field/source inputs.')
    output=prepare_output(a.output,clean=a.clean_output)
    configure_geometry_threads(8)
    mesh=pv.read(summary['mesh']).triangulate().clean()
    normal=pv.read(cfg['input'].get('normalization_mesh') or summary['mesh'])
    native=TrustedModelField(cfg['input']['checkpoint'],normal.points,cfg['network'])
    settings=summary['arguments']
    geometry,_=load_preserved_tree(Path(settings['tree_source']),mesh,.3,.3,settings['tip_radius'],
        terminal_cap='rounded',branch_sizing=settings['branch_sizing'],branch_diameter=settings['branch_diameter'])
    if not np.all(geometry.model_bridge):raise ValueError('Initial experiment expects a model-bridge-only Swan source.')
    domain=None;query_backend=None
    def closest_query(points):
        ids,closest=mesh.find_closest_cell(points,return_closest_point=True)
        return closest,np.asarray(ids,dtype=int)
    if a.routing_mode=='near-model':
        from SupportGenerationCurved.contact_direction_guide import ContactDirectionGuide
        project=lambda points:mesh.find_closest_cell(points,return_closest_point=True)[1]
        query_backend='vtk'
        if a.model_query_backend!='vtk':
            try:
                from SupportGenerationCurved.mesh_queries import Open3DClosest
                locator=Open3DClosest(mesh,threads=8)
                closest_query=locator.query
                project=lambda points:locator.query(points)[0]
                query_backend='open3d'
            except ImportError:
                if a.model_query_backend=='open3d':raise
        field=ContactDirectionGuide(native,geometry.distance,project,geometry.direction,a.near_band,a.vertical_distance)
    else:
        domain=pv.read(a.run/'envelope_field.vtu')
        field=SkeletonField(TransferredSupportField(domain,native,geometry,mode=summary.get('arguments',{}).get('support_field_mode','native_blend')))
    landing_check=None
    if a.model_landing_check=='upward':
        from SupportGenerationCurved.landing_checks import ModelLandingCheck
        oriented=mesh.compute_normals(cell_normals=True,point_normals=False,
                                      consistent_normals=True,auto_orient_normals=True)
        landing_check=ModelLandingCheck(closest_query,oriented.cell_normals,field,geometry.distance,
            min_up_dot=a.landing_min_up_dot,footprint_radius=a.landing_footprint_radius,
            contact_radius=a.root_contact_radius if a.root_contact_mode=='adaptive' else None,
            minimum_contact_radius=a.root_min_contact_radius,hard_min_up_dot=a.landing_fallback_min_up_dot,
            fallback_min_length=a.landing_fallback_min_length)
    recovery_graph=None;recovery_field_setup=0.
    if recovered_source is not None:
        saved_summary=json.loads((a.reroute_source/'summary.json').read_text())
        if Path(saved_summary['mesh']).resolve()!=Path(summary['mesh']).resolve():p.error('Saved recovery graph belongs to a different model.')
        recovery_graph=pv.read(a.reroute_source/'support_centerlines.vtp')
        setup_started=time.perf_counter()
        # A normal routing pass has already populated these field samples.
        # Saved-input recovery must load them (and initialise the neural runtime)
        # before applying the incremental recovery time budget.
        field(recovery_graph.points)
        recovery_field_setup=time.perf_counter()-setup_started
    loading=time.perf_counter()-started
    print(f'Loaded {len(geometry.b)} contacts; routing={a.routing_mode}; load={loading:.2f}s',flush=True)
    if recovered_source is None:
        skeleton,report=grow(geometry.b,field,geometry.distance,geometry.direction,geometry.plate,
                             step=a.step,angle_degrees=a.angle,merge_distance=a.merge_distance,max_steps=a.max_steps,
                             landing_check=landing_check,
                             clearance_escape_band=a.clearance_escape_band,clearance_escape_weight=a.clearance_escape_weight,
                             growth_preference='field' if a.routing_mode=='field-guided' else 'vertical',
                             progress=lambda status:print(json.dumps(status),flush=True))
    else:
        skeleton=recovery_graph;report=recovered_source.copy()
        report['source_planning_seconds']=report.get('planning_seconds');report['planning_seconds']=0.
        report['recovered_from']=str(a.reroute_source.resolve())
        report['recovery_input_field_setup_seconds']=recovery_field_setup
        a.angle=report['local_cone_angle_degrees']
    if a.reroute_stalled and report['unrooted_contacts']:
        from SupportGenerationCurved.stalled_recovery import recover_stalled
        skeleton,report=recover_stalled(skeleton,report,field,geometry.distance,landing_check,
            geometry.direction,geometry.plate,radius=a.reroute_radius,time_budget=a.reroute_time_budget,search_mode=a.reroute_search)
        report['planning_seconds']+=report['stalled_recovery']['seconds']
    if a.trunk_side_shift>0:
        from SupportGenerationCurved.trunk_shift import shift_trunks
        oriented=mesh.compute_normals(cell_normals=True,point_normals=False,consistent_normals=True,auto_orient_normals=True)
        skeleton,shift_report=shift_trunks(skeleton,field,geometry.distance,landing_check,
            closest_query,oriented.cell_normals,geometry.direction,geometry.plate,
            maximum_shift=a.trunk_side_shift,gap=report['clearance_mm'],angle=a.angle)
        report['trunk_side_shift']=shift_report
        report['planning_seconds']+=shift_report['seconds']
        report['model_landing_check']=landing_check.report()
        fallback=np.asarray(skeleton.point_data.get('landing_is_fallback',np.zeros(skeleton.n_points)),dtype=bool)
        report['fallback_landings']=[dict(node=int(i),point=skeleton.points[i].tolist(),
            achieved_up_dot=float(skeleton['landing_achieved_up_dot'][i])) for i in np.flatnonzero(fallback)]
    if landing_check is not None and a.root_contact_mode=='adaptive':
        edges=skeleton.lines.reshape(-1,3)[:,1:]
        incoming=np.bincount(edges[:,1],minlength=skeleton.n_points)
        outgoing=np.bincount(edges[:,0],minlength=skeleton.n_points)
        root_radii=np.asarray(skeleton.point_data.get('root_contact_radius',np.zeros(skeleton.n_points))).copy()
        for node in np.flatnonzero((incoming>0)&(outgoing==0)):
            root_radii[node]=landing_check.contact_radii.get(np.asarray(skeleton.points[node],dtype=float).tobytes(),root_radii[node])
        skeleton.point_data['root_contact_radius']=root_radii
        selected=root_radii[root_radii>0]
        report['adaptive_root_contacts']=dict(count=len(selected),
            minimum_radius_mm=float(selected.min()) if len(selected) else None,
            maximum_radius_mm=float(selected.max()) if len(selected) else None)
    audit_started=time.perf_counter()
    if skeleton.n_cells:
        edges=skeleton.lines.reshape(-1,3)[:,1:]
        begin=skeleton.points[edges[:,0]];end=skeleton.points[edges[:,1]];delta=end-begin
        probes=begin[:,None,:]+np.linspace(0,1,33)[None,:,None]*delta[:,None,:]
        scalars,gradients=field(probes.reshape(-1,3));scalars=scalars.reshape(-1,33)
        gradients=gradients.reshape(-1,33,3)
        cosines=np.sum(-gradients*(delta/np.linalg.norm(delta,axis=1)[:,None])[:,None,:],axis=2)/np.maximum(np.linalg.norm(gradients,axis=2),1e-15)
        report['dense_audit_cone_violation_edges']=int(np.count_nonzero(cosines.min(axis=1)<np.cos(np.deg2rad(a.angle))-1e-5))
        ordering_key='dense_audit_height_reversal_edges' if a.routing_mode=='near-model' else 'dense_audit_scalar_reversal_edges'
        report[ordering_key]=int(np.count_nonzero((np.diff(scalars,axis=1)>=0).any(axis=1)))
        if hasattr(field,'native_descent_mask'):
            report['dense_audit_near_model_scalar_violation_edges']=int(np.count_nonzero(~field.native_descent_mask(probes)))
        report['dense_audit_clearance_violation_edges']=int(np.count_nonzero((geometry.distance(probes.reshape(-1,3)).reshape(-1,33)<.3-2e-6).any(axis=1)))
    if skeleton.n_points:
        d=geometry.distance(skeleton.points)
        if domain is not None:
            sampled=pv.PolyData(skeleton.points).sample(domain)
            outside=np.asarray(sampled['vtkValidPointMask'])==0
            report['outside_saved_envelope_nodes']=int(outside.sum())
            report['outside_saved_envelope_beyond_native_band']=int(np.count_nonzero(outside&(d>1.)))
        report['minimum_sampled_node_clearance_mm']=float(d.min())
        stalled=np.asarray(skeleton['node_kind'])==3
        if stalled.any():
            _,g=field(skeleton.points[stalled])
            report['stalled_model_distances_mm']=d[stalled].tolist()
            report['stalled_gradient_norms']=np.linalg.norm(g,axis=1).tolist()
    report.update(loading_seconds=loading,audit_seconds=time.perf_counter()-audit_started,
                  unique_field_points=field.evaluated,source_run=str(a.run.resolve()),routing_mode=a.routing_mode,
                  near_band_mm=a.near_band,vertical_distance_mm=a.vertical_distance,
                  exterior_field_used=domain is not None,model_query_backend=query_backend)
    if a.routing_mode=='near-model':
        report['limitations'].append('Direction guide is not an exterior scalar field; future transferred layers must be checked against the skeleton.')
    elif a.routing_mode=='field-guided':
        report['limitations'].append('Uses finite differences of the saved-envelope/native blended scalar, not direct interpolation of extended vectors. Saved-envelope coverage and final-layer compatibility require checking.')
    write_started=time.perf_counter();skeleton.save(output/'support_centerlines.vtp')
    report['serialization_seconds']=time.perf_counter()-write_started
    report['module_total_before_preview_seconds']=time.perf_counter()-started
    report['timing_scope']='Module total includes loading, planning, audit and geometry serialization; excludes interpreter/module import startup and preview.'
    result={'mesh':summary['mesh'],'arguments':summary['arguments'],
            'envelope':{'build_direction':list(geometry.direction)},'skeleton_experiment':report}
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    (output/'module_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    if a.preview or a.screenshot:show(mesh,skeleton,a.screenshot,a.routing_mode)


if __name__=='__main__':main()
