"""Experimental layer planning using existing surface-field solvers.

Both planners can be compared through this entry with the same layer-normal
ray height audit. Explicit travel breaks are retained; machine export is gated.
"""
from pathlib import Path
import json,time,shutil
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np
import igl
from mesh_distance_workflow import load_obj_mesh
from connected_patch_planner import surface_components
from toolpath_export import compute_path_normals
from test_hybrid_contours import (solve_hybrid_boundary_field,solve_parallel_field,
    build_parallel_triangle_cache,stitch_segments_into_fragment_polylines,encode_path_tag_blocks)
from test_hybrid_contours_vectorized import (extract_parallel_segments_with_boundary_values_vectorized,
    clip_parallel_segments_vectorized)
from contour_extraction import extract_isocontour_polylines
from contact_support_height import _triangle_ray_distances

def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf8')
def key(path):return tuple(map(int,Path(path).stem.split('_')))
def split_subdivide(points,step):
    """Keep every triangle crossing. Never replace a curved surface by a chord."""
    p=np.asarray(points,float);result=[]
    for a,b in zip(p[:-1],p[1:]):
        length=np.linalg.norm(b-a)
        if length<1e-10:continue
        count=max(1,int(np.ceil(length/step)))
        result.extend(a+(b-a)*t/count for t in range(count))
    if result:result.append(p[-1])
    return np.asarray(result).reshape(-1,3)

def orient_mesh(V,F,mode,reference=None,reference_is_next=False,field_gradients=None):
    V=np.asarray(V,dtype=np.float64);F=np.asarray(F,dtype=np.int64)
    if mode=='field_gradient':
        from field_orientation import orient_from_gradients
        if field_gradients is None:raise ValueError('Saved slicing-field gradients are required; re-export layers or select an explicit winding orientation.')
        return orient_from_gradients(V,F,field_gradients)
    if mode=='mesh_winding':return V,F,dict(mode=mode,flipped_components=0)
    if mode=='reverse_winding':return V,F[:,[0,2,1]],dict(mode=mode,flipped_components=1)
    if mode!='layer_order':raise ValueError('Unknown normal-direction mode.')
    if reference is None:raise ValueError('Automatic normal direction needs an adjacent layer; for a single layer select Mesh winding or Reverse winding explicitly.')
    RV,RF=load_obj_mesh(reference);tree=igl.AABB();tree.init(RV,RF)
    vertices=[];faces=[];offset=0;flips=0;agreements=[]
    for v,f in surface_components(V,F):
        tri=v[f];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);centers=tri.mean(axis=1)
        _,_,closest=tree.squared_distance(RV,RF,centers)
        delta=(closest-centers) if reference_is_next else (centers-closest)
        score=np.einsum('ij,ij->i',normal,delta)
        informative=abs(score)>1e-10
        if not informative.any():raise ValueError('Adjacent layer does not determine normal direction. Select an explicit winding orientation.')
        positive=float(np.abs(score[score>0]).sum());negative=float(np.abs(score[score<0]).sum())
        agreement=max(positive,negative)/(positive+negative)
        if agreement<.9:raise ValueError(f'Layer-order normal direction is ambiguous ({agreement:.1%} agreement). Check layer order or select an explicit winding orientation.')
        if negative>positive:f=f[:,[0,2,1]];flips+=1
        vertices.append(v);faces.append(f+offset);offset+=len(v);agreements.append(agreement)
    return np.vstack(vertices),np.vstack(faces),dict(mode=mode,flipped_components=flips,
        minimum_weighted_agreement=min(agreements),reference=reference,reference_is_next=reference_is_next)

def frame(vertices,faces,angle):
    tri=vertices[faces];area=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal=area.sum(axis=0)
    if np.linalg.norm(normal)<1e-10:normal=area[np.argmax(np.linalg.norm(area,axis=1))]
    normal/=np.linalg.norm(normal)
    theta=np.deg2rad(angle);target=np.array([np.cos(theta),0.,np.sin(theta)])
    tangent=target-normal*np.dot(target,normal)
    if np.linalg.norm(tangent)<1e-6:
        target=np.eye(3)[np.argmin(abs(normal))];tangent=target-normal*np.dot(target,normal)
    tangent/=np.linalg.norm(tangent);across=np.cross(tangent,normal)
    return np.column_stack((tangent,normal,across))

def solve_surface_infill(local,faces,plane,backend):
    """Retain the requested solver; recover a signed-heat factorization failure."""
    options=dict(level_set_constraint='None',soft_level_set_weight=-1.,tolerance=1e-8)
    try:
        field,_,_=solve_parallel_field(local,faces,'z',2,plane,backend=backend,**options)
        return field,backend,None
    except ValueError as exc:
        if backend!='signed_heat' or 'factorization failed' not in str(exc):raise
        # Both solvers work on the original surface. Never substitute an XZ
        # projection or remove troublesome triangles from a folded layer.
        field,_,_=solve_parallel_field(local,faces,'z',2,plane,backend='vector_heat_stripe',**options)
        print(f'WARNING: Signed-heat factorization failed on a {len(faces)}-triangle patch; '
              'used the surface vector-heat solver. Inspect field diagnostics.',flush=True)
        return field,'vector_heat_stripe',str(exc)

def surface_paths(vertices,faces,settings):
    vertices=np.asarray(vertices,dtype=np.float64);faces=np.asarray(faces,dtype=np.int64)
    width=settings['spacing'];pitch=settings['pitch'];step=settings['resample_spacing']
    wall_width=settings.get('wall_width',width)
    count=settings['boundary_count'];overlap=settings['overlap'];inset=.5*wall_width if settings['inset'] else 0.
    sampling=settings.get('waypoint_sampling')
    if sampling is not None:
        from surface_waypoint_sampling import sampling_options, reduce_surface_strand
        sampling=sampling_options(sampling)
    sample_step=min(step,sampling['maximum_spacing_mm']) if sampling is not None and settings['reduce_waypoints'] else step
    all_parts=[];all_normals=[];all_widths=[];roles=[];diagnostics=[]
    for V,F in surface_components(vertices,faces):
        started=time.perf_counter()
        # Reuse the earlier continuous-boundary fast marching and selected infill
        # solvers. Only their adapter/extraction/serialization is new.
        boundary,loops,_,_=solve_hybrid_boundary_field(V,F,backend='potpourri_fast_marching_unsigned',tolerance=1e-8)
        parts=[];part_roles=[]
        for i in range(count):
            level=inset+i*wall_width
            curves=loops if level==0 else extract_isocontour_polylines(V,F,boundary,level)
            parts.extend(curves);part_roles.extend(['boundary_loop']*len(curves))
        gradient_stats={}
        if settings['pattern']!='contours':
            basis=frame(V,F,settings['angle']);local=(V-V.mean(axis=0))@basis
            plane=float((local[:,2].min()+local[:,2].max())*.5)
            # A source plane through mesh vertices can produce duplicated edge
            # intersections in the existing solver. Choose a nearby generic plane.
            epsilon=max(float(np.ptp(local[:,2]))*1e-6,1e-6)
            for attempt in range(32):
                if np.min(abs(local[:,2]-plane))>1e-7:break
                plane+=epsilon*(attempt+1)
            field,used_backend,fallback_reason=solve_surface_infill(local,F,plane,settings['field_backend'])
            if not np.isfinite(field).all():raise ValueError('Nonfinite surface infill field.')
            # Finite-element gradient diagnostics: field increments approximate
            # physical surface spacing only where gradient magnitude is near1.
            gradient=np.asarray(igl.grad(V,F)@field).reshape(3,-1).T
            norms=np.linalg.norm(gradient,axis=1)
            gradient_stats={'gradient_norm_p05':float(np.percentile(norms,5)),
                'gradient_norm_median':float(np.median(norms)),'gradient_norm_p95':float(np.percentile(norms,95)),
                'infill_backend':used_backend,'infill_fallback_reason':fallback_reason}
            cache=build_parallel_triangle_cache(V,F,field,boundary)
            cutoff=(inset+count*wall_width+.5*(width-wall_width)-overlap) if count else 0.
            levels=np.arange(np.ceil(field.min()/pitch),np.floor(field.max()/pitch)+1)*pitch
            for level in levels:
                segments=extract_parallel_segments_with_boundary_values_vectorized(cache,float(level),tolerance=1e-8)
                clipped=clip_parallel_segments_vectorized(segments,cutoff,tolerance=1e-8)
                curves=stitch_segments_into_fragment_polylines(clipped,tolerance=1e-8)
                parts.extend(curves);part_roles.extend(['sparse_infill' if settings['sparse'] else 'inner_raster']*len(curves))
        kept=0;sampling_reports=[];input_points=0;output_points=0
        for p,role in zip(parts,part_roles):
            p=split_subdivide(p,sample_step)
            if len(p)<2:continue
            input_points+=len(p)
            normals=compute_path_normals(V,F,p)
            if settings['reduce_waypoints'] and sampling is not None:
                ids,report=reduce_surface_strand(p,normals,sampling)
                p=p[ids];normals=normals[ids];sampling_reports.append(report)
            elif settings['reduce_waypoints'] and len(p)>2:
                # Exact collinearity only, bounded segment length and normal turn.
                ids=[0]
                for i in range(1,len(p)-1):
                    a=p[i]-p[ids[-1]];b=p[i+1]-p[i];span=p[i+1]-p[ids[-1]]
                    straight=np.linalg.norm(np.cross(a,b))<=1e-10*max(np.linalg.norm(a)+np.linalg.norm(b),1.)
                    if not(straight and np.dot(a,b)>=0 and np.linalg.norm(span)<=2*step and np.dot(normals[ids[-1]],normals[i+1])>np.cos(np.deg2rad(.25))):ids.append(i)
                ids.append(len(p)-1);p=p[ids];normals=normals[ids]
            all_parts.append(p);all_normals.append(normals)
            output_points+=len(p)
            all_widths.append(np.full(len(p),wall_width if role=='boundary_loop' else width))
            roles.append(role);kept+=1
        diagnostics.append(dict(vertices=len(V),faces=len(F),runs=kept,seconds=time.perf_counter()-started,
            waypoint_sampling=dict(mode='adaptive_surface' if sampling is not None else 'legacy_collinear',
                enabled=settings['reduce_waypoints'],
                input_points=input_points,output_points=output_points,
                strands=sampling_reports),**gradient_stats))
    # Every separate contour/hatch strand is separated by an explicit travel.
    # Do not turn straight-line handoffs across voids into deposited material.
    if not all_parts:return np.empty((0,6)),np.empty(0),np.empty(0,dtype=str),[],diagnostics
    counts=[len(p) for p in all_parts];breaks=np.cumsum(counts)[:-1].tolist()
    rows=np.column_stack((np.vstack(all_parts),np.vstack(all_normals)))
    tags=np.concatenate([np.repeat(role,len(p)) for p,role in zip(all_parts,roles)])
    return rows,np.concatenate(all_widths),tags,breaks,diagnostics

def plan_job(job):
    started=time.perf_counter();mesh=Path(job['mesh']);out=Path(job['output']);s=job['settings']
    V,F=load_obj_mesh(str(mesh));diagnostics=[]
    normal_mode=job.get('normal_orientation','field_gradient')
    gradients=gradient_source=None
    if normal_mode=='field_gradient':
        from field_orientation import load_gradients
        gradients,gradient_source=load_gradients(mesh,len(F))
    V,F,orientation=orient_mesh(V,F,normal_mode,job.get('orientation_reference'),job.get('orientation_reference_is_next',False),field_gradients=gradients)
    if gradient_source:orientation['gradient_source']=gradient_source
    if job['approach']=='surface_field':
        profile=None
        if job.get('entry',{}).get('part')=='support':
            from surface_field_support import support_paths
            rows,widths,tags,breaks,diagnostics,profile=support_paths(V,F,s,job['entry'],job['support_config'])
        else:
            rows,widths,tags,breaks,diagnostics=surface_paths(V,F,s)
        path=out/(mesh.stem+'.txt');np.savetxt(path,rows,fmt='%.10f')
        np.savetxt(out/(mesh.stem+'.widths.txt'),widths,fmt='%.10f')
        tagfile=out/(mesh.stem+'.tags.txt');tagfile.write_text(''.join(f'{a} {b} {tag}\n' for a,b,tag in encode_path_tag_blocks(tags)),encoding='utf8')
        breakfile=out/(mesh.stem+'.breaks.txt');breakfile.write_text(''.join(str(i)+'\n' for i in breaks),encoding='utf8')
        meta=dict(path_file=path.name,path_point_count=len(rows),width_profile_file=mesh.stem+'.widths.txt',
            path_tag_file=tagfile.name,path_break_file=breakfile.name,path_break_count=len(breaks),
            contour_spacing=s['spacing'],parallel_spacing=s['pitch'],width=s['spacing'],
            planner_backend='surface_field',boundary_field_backend='potpourri_fast_marching_unsigned',
            parallel_field_backend=('mixed' if any(d.get('infill_fallback_reason') for d in diagnostics) else s['field_backend']),
            requested_parallel_field_backend=s['field_backend'],field_diagnostics=diagnostics,
            model_half_width_inset=s['inset'],outer_contour_center_offset=.5*s['spacing'] if s['inset'] else 0.,
            infill_wall_overlap=s['overlap'],height_aware_waypoint_reduction=(
                'adaptive 3D chord/normal interpolation/arc length bounds' if s.get('waypoint_sampling') is not None
                else 'exact collinearity only; normal-turn and length bounded'),
            surface_waypoint_sampling=s.get('waypoint_sampling'),
            normal_mode='interpolated mesh winding; no fixed-axis sign flip',connector_policy='explicit travel between independent strands')
        if profile is not None:
            role=profile['role']
            meta.update(layer_role=f'support_{role}',support_profile=profile,support_components=diagnostics,
                outer_contour_center_offset=None,model_half_width_inset=False,
                extrusion_role_scales={f'support_{role}_infill':profile['flow_ratio']} if role=='contact' else {},
                contact_pitch_used_for_extrusion=False,
                parallel_disabled=not any('infill' in tag for tag in tags))
    else:
        from batch_generate_projected_hybrid_paths import process_projected_hybrid_layer
        oriented=out/'oriented_inputs'/mesh.name;oriented.parent.mkdir(exist_ok=True)
        with oriented.open('w') as stream:
            np.savetxt(stream,V,fmt='v %.17g %.17g %.17g');np.savetxt(stream,F+1,fmt='f %d %d %d')
        path,meta=process_projected_hybrid_layer(oriented,out,spacing=s['spacing'],parallel_spacing=s['spacing'],
            resample_spacing=s['resample_spacing'],parallel_resample_spacing=s['resample_spacing'],boundary_count=s['boundary_count'],
            sharp_angle_deg=35,boundary_field_backend='native',parallel_field_backend='signed_heat',axis_mode='fixed_angle',axis='x',
            hatch_angle_degrees=s['angle'],plane_value=None,level_set_constraint='None',soft_level_set_weight=-1,
            contour_max_level_fraction=1.,min_polyline_points=2,mild_smooth_all=False,mild_smooth_iterations=0,mild_smooth_alpha=.2,
            disable_parallel=s['pattern']=='contours',planar_infill=dict(enabled=s['sparse'],density=s['density'],
                angle_degrees=s['angle']-90,alternate_angle_degrees=None,model_half_width_inset=s['inset'],
                wall_attachment_overlap=s['overlap'],point_reduction=dict(enabled=s['reduce_waypoints'])))
    stored=np.loadtxt(path,ndmin=2) if path.stat().st_size else np.empty((0,6))
    broken=set(map(int,(out/meta['path_break_file']).read_text().split())) if meta.get('path_break_file') else set()
    edges=np.array([i not in broken for i in range(1,len(stored))],bool)
    samples=.5*(stored[:-1,:3]+stored[1:,:3])[edges]
    tree=igl.AABB();tree.init(V,F)
    sq,_,_=tree.squared_distance(V,F,samples) if len(samples) else (np.empty(0),None,None)
    distance=np.sqrt(np.maximum(np.asarray(sq).reshape(-1),0))
    meta['path_geometry_audit']=dict(midpoint_samples=len(samples),off_surface_midpoints=int((distance>.01).sum()),maximum_midpoint_distance_mm=float(distance.max()) if len(distance) else 0.,tolerance_mm=.01)
    main,sub=key(mesh)
    meta.update(layer_main=main,layer_sub=sub,source_mesh=mesh.name,path_file=path.name,normal_orientation=orientation,planning_seconds=time.perf_counter()-started)
    if job.get('entry'):
        meta.update(job['entry'])
        meta.update(original_source_mesh=job['entry']['source_mesh'],source_mesh=mesh.name)
        if job['entry']['part']=='model':meta['layer_role']='model'
    return meta

def height_job(job):
    started=time.perf_counter();layer=job['layer'];src=Path(job['path_dir']);out=Path(job['output'])
    file=src/layer['path_file'];rows=np.loadtxt(file,ndmin=2) if file.stat().st_size else np.empty((0,6))
    points=rows[:,:3];normal=rows[:,3:6];normal=normal/np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-30)
    direction=-normal;best=np.full(len(points),np.inf);nearest=np.full(len(points),np.inf)
    source=np.full(len(points),-1,dtype=int);edge_recovered=np.zeros(len(points),bool)
    for idx,path in enumerate(job['references']):
        V,F=load_obj_mesh(path);tree=igl.AABB();tree.init(V,F)
        if not len(points):continue
        hit,_,_=tree.intersect_ray_first(V,F,points,direction);hit=np.asarray(hit).reshape(-1);ids=np.flatnonzero(hit>=0)
        if len(ids):
            distances=_triangle_ray_distances(points[ids],direction[ids],V[F[hit[ids]]])
            valid=(distances>1e-7)&(distances<best[ids]);selected=ids[valid]
            best[selected]=distances[valid];source[selected]=idx
        sq,closest_faces,_=tree.squared_distance(V,F,points)
        distance=np.sqrt(np.maximum(np.asarray(sq).reshape(-1),0))
        nearest=np.minimum(nearest,distance)
        # Shared-edge float ray misses: test the nearest triangle against the
        # ORIGINAL ray in double precision. Never substitute nearest distance.
        candidates=np.flatnonzero((best>distance+1e-6)&np.isfinite(distance))
        if len(candidates):
            candidate_faces=np.asarray(closest_faces).reshape(-1)[candidates]
            checked=_triangle_ray_distances(points[candidates],direction[candidates],V[F[candidate_faces]])
            valid=np.isfinite(checked)&(checked<best[candidates])&(abs(checked-distance[candidates])<=1e-4)
            selected=candidates[valid];best[selected]=checked[valid];source[selected]=idx;edge_recovered[selected]=True
    # Platform only on the first supplied main (or without earlier geometry).
    # Paired first-main parts can be disjoint, yet each starts on the bed.
    # Do not replace a later ray miss
    # through an unsupported gap with a short nearest distance or a fabricated cap.
    platform=np.zeros(len(points),bool)
    if not job['references'] or job.get('allow_platform',False):
        d=np.divide(points[:,1]-job['platform'],normal[:,1],out=np.full(len(points),np.inf),where=normal[:,1]>1e-8)
        valid=(d>1e-7)&(d<best);best[valid]=d[valid];platform[valid]=True
    miss=~np.isfinite(best);too_high=(best>job['maximum_height'])&~miss
    unresolved=miss|too_high;heights=np.where(miss,0.,best)
    name=f"{layer['layer_main']}_{layer['layer_sub']}_5axis.txt"
    np.savetxt(out/name,np.column_stack((rows,heights)),fmt='%.10f')
    for field in ['width_profile_file','path_tag_file','path_break_file']:
        if layer.get(field):shutil.copyfile(src/layer[field],out/layer[field])
    np.savez_compressed(out/(Path(name).stem+'.height_validation.npz'),normal_ray_height=best,
        nearest_distance=nearest,unresolved=unresolved,normal_miss=miss,above_limit=too_high,platform=platform,reference=source,shared_edge_ray_recovered=edge_recovered)
    return dict(layer,output_path_file=name,reference_layers=[Path(p).stem for p in job['references']],
        support_validation=dict(unresolved_points=int(unresolved.sum()),normal_miss_points=int(miss.sum()),above_limit_points=int(too_high.sum()),shared_edge_ray_recovered=int(edge_recovered.sum())),
        height_seconds=time.perf_counter()-started,max_height=float(heights.max()) if len(heights) else 0.)

def parallel_map(fn,jobs,workers):
    if workers==1:return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:return list(pool.map(fn,jobs))

def run(config_path):
    begin=time.perf_counter();config_path=Path(config_path);c=json.loads(config_path.read_text(encoding='utf-8-sig'));base=config_path.parent
    paired=c.get('pipeline_variant')=='support-model' or 'compatible_inputs' in c
    approach=c.get('planning_approach','projected')
    if approach not in {'projected','surface_field'}:raise ValueError('Unknown planning approach.')
    for stage in ['remesh','support_mesh_expand']:
        if c.get(stage,{}).get('enabled'):raise ValueError(f'{stage} is not supported by the experimental surface workflow; prepare meshes before planning.')
    def location(k):return (base/Path(c['paths'][k])).resolve()
    path_dir=location('toolpaths');heights=location('heights')
    if any(p.exists() and any(p.iterdir()) for p in (path_dir,heights)):raise FileExistsError('Experimental planning requires fresh path/height output folders.')
    entries=None;paired_metadata=None
    if paired:
        if approach!='surface_field':raise ValueError('Paired layer-normal planning currently requires Surface-field; use Fixed-axis for the Projected paired planner.')
        from surface_field_support import prepare_inputs
        t=time.perf_counter()
        meshes,entries,paired_metadata=prepare_inputs(c,base,location('combined_input_surfaces'),
            c.get('surface_field',{}).get('normal_orientation','field_gradient'))
        print(f'[surface workflow] Paired schedule/normal sources prepared in {time.perf_counter()-t:.2f} s',flush=True)
    else:
        inputs=location('input_surfaces')
        meshes=sorted([p for p in inputs.glob('*.obj') if len(p.stem.split('_'))==2 and all(v.isdigit() for v in p.stem.split('_'))],key=key)
    if not meshes:raise ValueError('No numeric layer meshes.')
    if c.get('surface_field',{}).get('normal_orientation','field_gradient')=='field_gradient':
        missing=[m.name for m in meshes if not m.with_suffix('.field_gradients.npz').is_file()]
        if missing:raise ValueError('Saved slicing-field gradients missing for '+', '.join(missing[:5])+'. Re-export generated layers or explicitly choose Mesh winding / Reverse winding for imported geometry.')
    h=c['hybrid_contours'];p=h.get('planar_infill',{});mode=h.get('axis_mode','auto_tangent_alternating')
    if mode not in {'fixed','fixed_angle','alternating_angle','auto_tangent_alternating'}:raise ValueError('Unsupported infill direction mode.')
    if h.get('ironing',{}).get('enabled'):raise ValueError('Surface workflow does not implement ironing.')
    if h.get('top_layer_classifier',{}).get('enabled') or h.get('top_layer_overrides',{}).get('classification_json'):
        raise ValueError('Top-surface classification overrides are not supported by this experimental workflow; use solid base/top layer counts.')
    if h.get('progress_overrides') or h.get('material_overrides') or h.get('model_lift',0):
        raise ValueError('Progress/material overrides and model lift are not supported by this experimental workflow.')
    if h.get('reuse_existing_paths') or h.get('projected_width_check',{}).get('enabled'):
        raise ValueError('Path reuse and projected width overrides are not supported by this experimental workflow.')
    pattern=c.get('model_pattern') or ('contours' if h.get('disable_parallel') else 'sparse' if p.get('enabled') else 'hybrid')
    workers=int(h.get('layer_workers',1));width=float(h['spacing']);step=float(h['resample_spacing']);density=float(p.get('density',.25))
    if workers<1 or not np.isfinite([width,step,density]).all() or width<=0 or step<=0 or not 0<density<=1:raise ValueError('Invalid planner settings.')
    if pattern not in {'hybrid','sparse','raster','contours'}:raise ValueError('Unsupported pattern.')
    sampling=c.get('surface_field',{}).get('waypoint_sampling')
    if sampling is not None:
        from surface_waypoint_sampling import sampling_options
        sampling=sampling_options(sampling)
    remesh_report = dict(enabled=False, seconds=0.)
    remesh_config = c.get('surface_field', {}).get('remesh', {})
    if remesh_config.get('enabled'):
        if approach != 'surface_field':
            raise ValueError('Optional surface remeshing is only supported with Surface-field planning.')
        from surface_remesh import prepare_layers
        print(f'[surface workflow] Remeshing {len(meshes)} layer copies with geometry checks', flush=True)
        meshes, remesh_report = prepare_layers(meshes, path_dir.parent/'surface_remeshed', remesh_config, parallel_map, workers)
    mains=sorted({key(m)[0] for m in meshes});ranks={m:i for i,m in enumerate(mains)}
    entry_lookup={(e['layer_main'],e['layer_sub']):e for e in entries} if paired else {}
    model_mains=sorted({e['layer_main'] for e in entries if e['part']=='model'}) if paired else mains
    model_ranks={m:i for i,m in enumerate(model_mains)}
    jobs=[]
    for mesh in meshes:
        main,sub=key(mesh);rank=ranks[main]
        model_rank=model_ranks.get(main,-1)
        sparse=pattern=='sparse' and model_rank>=int(p.get('solid_base_main_layers',0)) and model_rank<len(model_mains)-int(p.get('solid_top_main_layers',0))
        angle=float(h.get('hatch_angle_degrees',0))
        if mode in {'auto_tangent_alternating','alternating_angle'}:angle+=90*(main%2)
        if mode=='fixed':angle=0. if h.get('axis','x')=='x' else 90.
        settings=dict(spacing=width,resample_spacing=step,pitch=width/density if sparse else width,
            density=density,sparse=sparse,pattern=pattern,boundary_count=0 if pattern=='raster' else int(h['boundary_count']),
            overlap=float(h.get('infill_wall_overlap',.1)),inset=bool(h.get('model_half_width_inset',False)),angle=angle,
            field_backend=c.get('surface_field',{}).get('infill_backend','vector_heat_stripe'),reduce_waypoints=bool(h.get('point_reduction',{}).get('enabled',True)))
        if sampling is not None:settings['waypoint_sampling']=sampling
        if settings['boundary_count']<0 or not 0<=settings['overlap']<=width*.5:raise ValueError('Invalid contour count or overlap.')
        if settings['field_backend'] not in {'signed_heat','vector_heat_stripe'}:raise ValueError('Unsupported surface infill solver.')
        normal_mode=c.get('surface_field',{}).get('normal_orientation','field_gradient')
        adjacent=(mains[rank-1] if rank>0 else mains[1] if len(mains)>1 else None) if normal_mode=='layer_order' else None
        reference=[p for p in meshes if key(p)[0]==adjacent] if adjacent is not None else []
        # Multiple subpatches in a reference main are combined, retaining all
        # disconnected surfaces for the direction comparison.
        ref=None
        if reference:
            if len(reference)==1:ref=str(reference[0])
            else:
                import pyvista as pv
                refpath=config_path.parent/f'normal_reference_{adjacent}.obj'
                combined=pv.merge([pv.read(p) for p in reference],merge_points=False).extract_surface(algorithm='dataset_surface')
                combined.save(refpath);ref=str(refpath)
        jobs.append(dict(mesh=str(mesh),output=str(path_dir),settings=settings,approach=approach,
            normal_orientation=normal_mode,
            orientation_reference=ref,orientation_reference_is_next=rank==0))
        if paired:jobs[-1].update(entry=entry_lookup[(main,sub)],support_config=c['support_paths'])
    path_dir.mkdir(parents=True,exist_ok=True);heights.mkdir(parents=True,exist_ok=True)
    print(f'[surface workflow] {approach}, {len(jobs)} layers, {workers} workers',flush=True)
    t=time.perf_counter();layers=parallel_map(plan_job,jobs,workers);planning=time.perf_counter()-t
    meta=dict(batch_kind='experimental_model_paths',planning_approach=approach,width=width,contour_spacing=width,layers=layers,
        remeshing=remesh_report,
        experimental=True,machine_export_supported=False,normal_source='Saved field-gradient oriented mesh; explicit winding overrides and legacy config mode supported',model_lift=0.,model_lift_axis='y')
    if paired:
        meta.update(batch_kind='experimental_surface_model_support_paths',support_model=dict(same_material=True,
            source_metadata=paired_metadata,contact_gap_correction=False,backing_repair=False,hole_repair=False))
    write_json(path_dir/'batch_paths.meta.json',meta)
    heights_cfg=c.get('heights_3axis',{});window=int(heights_cfg.get('previous_main_window',2));nominal=float(heights_cfg.get('nominal_layer_height',.2))
    maximum=float(c.get('surface_field',{}).get('maximum_normal_height',1.5*nominal))
    if window<1 or maximum<=0 or not np.isfinite(maximum):raise ValueError('Invalid normal-height search settings.')
    hj=[]
    mesh_lookup={key(m):m for m in meshes}
    for layer in layers:
        m=layer['layer_main'];previous=mains[max(0,ranks[m]-window):ranks[m]]
        refs=[str(p) for p in meshes if key(p)[0] in previous]
        if paired:
            from surface_field_support import height_references
            refs=height_references(layer,layers,mesh_lookup,mains,ranks,window)
        hj.append(dict(layer=layer,path_dir=str(path_dir),output=str(heights),references=refs,
            platform=float(heights_cfg.get('platform_value',0)),maximum_height=maximum,
            allow_platform=paired and layer['layer_main']==mains[0]))
    height_workers=int(heights_cfg.get('layer_workers',workers))
    if height_workers<1:raise ValueError('Height workers must be positive.')
    t=time.perf_counter();results=parallel_map(height_job,hj,height_workers);height_seconds=time.perf_counter()-t
    meta.update(batch_kind='experimental_layer_normal_heights',height_mode='5axis',height_definition='first positive ray hit along negative layer normal; platform ray only on first layer',
        effective_settings=dict(planning_workers=workers,height_workers=height_workers,per_layer=[j['settings'] for j in jobs],
            short_fragment_filter=False,smoothing=False,solid_continuity_check=False),
        maximum_normal_height=maximum,shared_edge_recovery_distance_tolerance_mm=1e-4,layers=results,support_validation={'unresolved_points':sum(r['support_validation']['unresolved_points'] for r in results)},
        timings=dict(remesh_wall_seconds=remesh_report['seconds'],planning_wall_seconds=planning,height_wall_seconds=height_seconds,total_wall_seconds=time.perf_counter()-begin),
        limitations=['No model/support contact-gap extrusion correction or backing/hole repair.','No swept tool or machine kinematics check.',
            'Normals use geometry and selected orientation policy; validate against slicing field.',
            'First layer of a cropped stack may lack its actual underlying geometry.',
            'Independent strands use travel; no optimized travel or printable transfer motions.'])
    write_json(heights/'batch_5axis.meta.json',meta)
    print('[surface workflow] '+json.dumps(meta['timings']),flush=True)
    return meta
