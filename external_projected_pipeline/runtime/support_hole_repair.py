"""Directed born-small support-hole masks with saved-field lifting.

Source layer meshes remain unchanged. Cached projected regions avoid repeating
their polygon unions in path workers. Model-near/protected lifetimes are kept.
"""

from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
import hashlib,json
from pathlib import Path
import sys,time
import numpy as np
import shapely
from shapely.geometry import Polygon,MultiPolygon
from shapely.ops import unary_union
from scipy.spatial import cKDTree


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def chart_y_shift_budget(layer_height):
    """Bound field-chart correction to a quarter of the source layer target.

    This is a geometry displacement budget, not an extrusion-height limit.
    At the existing 0.2 mm target it remains 0.05 mm; coarser source layers
    must not be forced through the old absolute 0.05 mm ceiling.
    """
    height=float(layer_height)
    if not np.isfinite(height) or height<=0:
        raise ValueError('Source nominal layer height must be positive and finite.')
    return .25*height


def patch_identity(lifter):
    h=hashlib.sha256(np.ascontiguousarray(lifter.vertices,dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(lifter.faces,dtype=np.int64).tobytes())
    return h.hexdigest()


@lru_cache(maxsize=1)
def _geometry_context(field_source,model_source,domain_sha,field_sha,model_sha):
    for path,wanted in ((Path(field_source)/'domain.npz',domain_sha),
                        (Path(field_source)/'optimized_field.npz',field_sha),(Path(model_source),model_sha)):
        if _sha(path)!=wanted:raise ValueError('Support repair source changed: '+str(path))
    runtime=Path(__file__).resolve().parents[2]/'CurvedSupportGenerator/runtime'
    if str(runtime) not in sys.path:sys.path.insert(0,str(runtime))
    from SupportGenerationCurved.field_grid import GridField
    import pyvista as pv
    import igl,vtk
    mesh=pv.read(model_source)
    if not isinstance(mesh,pv.PolyData):mesh=mesh.extract_surface(algorithm='dataset_surface')
    mesh=mesh.triangulate().clean()
    if mesh.n_open_edges:raise ValueError('Support-hole repair requires a closed original model solid.')
    vertices=np.asarray(mesh.points,dtype=np.float64).copy()
    faces=np.asarray(mesh.regular_faces,dtype=np.int64).copy()
    tree=igl.AABB();tree.init(vertices,faces)
    inside=vtk.vtkSelectEnclosedPoints();inside.Initialize(mesh)
    return GridField.load(field_source),(vertices,faces,tree,inside,mesh)


def _context(settings):
    return _geometry_context(settings['field_source'],settings['model_source'],
        settings['domain_sha256'],settings['field_sha256'],settings['model_sha256'])


def _distances(model,points):
    vertices,faces,tree=model[:3]
    squared,_,_=tree.squared_distance(vertices,faces,np.asarray(points))
    return np.sqrt(np.maximum(squared,0)).reshape(-1)


def _confirmed_inside(model,points):
    """Confirm provisional ray-parity positives; near edges VTK can disagree.

    The exact solid-angle test avoids randomly rejecting already trimmed
    vertices because a ray grazes a mesh edge. Closed model is required above.
    """
    import igl
    points=np.asarray(points,float).reshape(-1,3)
    possible=np.array([bool(model[3].IsInsideSurface(*p)) for p in points])
    if possible.any():
        winding=igl.winding_number(model[0],model[1],points[possible])
        if not np.isfinite(winding).all():raise ValueError('Model winding classification is inconclusive.')
        possible[possible]=abs(winding)>.5
    return possible


def track_lifetimes(catalogs,settings):
    holes=[dict(h,scalar=row['scalar']) for row in catalogs for h in row['holes']]
    groups={row['scalar']:[] for row in catalogs}
    for i,h in enumerate(holes):groups[h['scalar']].append(i)
    previous=[];previous_polys=[];status={}
    for scalar,ids in sorted(groups.items()):
        tree=shapely.STRtree([p.buffer(settings['parent_radius_mm']) for p in previous_polys])
        for i in ids:
            h=holes[i];parents=[]
            for local in tree.query(Polygon(h['xy']),predicate='intersects'):
                j=previous[int(local)];p=holes[j]
                if abs(h['center_y']-p['center_y'])<=settings['maximum_sheet_shift_mm']+.5*(h['height_span']+p['height_span']):parents.append(j)
            may_start=all(status[j] for j in parents) if parents else h['small_birth_candidate']
            status[i]=may_start and not h['protected']
            h['parents']=parents
        previous=ids;previous_polys=[Polygon(holes[i]['xy']) for i in ids]
    return holes,status


def _catalog(job):
    path,entry,settings=job
    from support_preflight_cache import Probe
    probe=Probe(path,entry,settings);cached=probe.load()
    if cached is not None:
        cached['catalogue_cache']=probe.stats
        return cached
    from mesh_distance_workflow import load_obj_mesh
    from connected_patch_planner import surface_components
    from batch_generate_projected_hybrid_paths import ProjectedSurfaceLifter,_planar_region_from_projected_triangles,_iter_polygons,_subdivide_3d
    field,model=_context(settings)
    vertices,faces=load_obj_mesh(path);holes=[];regions={}
    patches=list(surface_components(vertices,faces))
    # The planner keeps original vertex/cell ordering for a single body.
    if len(patches)==1:patches=[(vertices,faces)]
    from projection_mesh_repair import validate_field_chart
    field_charts={}
    repaired_mesh=None
    for points,triangles in patches:
        lifter=ProjectedSurfaceLifter.build(points,triangles)
        identity=patch_identity(lifter)
        chart=validate_field_chart(lifter,field,entry['scalar'],field_tolerance=settings['surface_repair_budget_mm'],
            maximum_facet_difference=settings['chart_maximum_y_shift_mm'])
        if chart is not None:field_charts[identity]=chart
        geometry=_planar_region_from_projected_triangles(lifter,validate_projection=False)
        source_polygons=_iter_polygons(geometry)
        polygons=[Polygon(p.exterior.coords,[r.coords for r in p.interiors if Polygon(r).area>1e-10]) for p in source_polygons]
        geometry=polygons[0] if len(polygons)==1 else MultiPolygon(polygons)
        regions[identity]=geometry.wkb_hex
        for polygon in source_polygons:
            for ring in polygon.interiors:
                xy=np.asarray(ring.coords);xyz,_,_=lifter.lift(xy)
                probes=_subdivide_3d(xyz,settings['boundary_sample_mm'])
                distance=float(_distances(model,probes).min())
                bounds=Polygon(xy).bounds
                holes.append(dict(xy=xy.tolist(),ring=xyz.tolist(),file=entry['combined_mesh'],
                    small_birth_candidate=max(bounds[2]-bounds[0],bounds[3]-bounds[1])<settings['birth_width_mm'],
                    protected=distance<=settings['protection_distance_mm'],model_distance_mm=distance,
                    center_y=float(xyz[:,1].mean()),height_span=float(np.ptp(xyz[:,1]))))
    result=dict(file=entry['combined_mesh'],scalar=entry['scalar'],holes=holes,regions=regions,
        repaired_mesh=None,surface_repairs=[],field_charts=field_charts)
    probe.store(result);result['catalogue_cache']=probe.stats
    return result


def prepare_repair(entries,mesh_dir,output_dir,support_config,export_metadata,height_config,*,base_dir,workers=1):
    """Return per-layer context paths and an explicit availability report."""
    option=dict(support_config.get('hole_repair',{}));started=time.perf_counter()
    report=dict(enabled=bool(option.get('enabled',True)),status='disabled',changed_layer_candidates=0)
    if not report['enabled']:return {},report
    root=option.get('field_source') or export_metadata.get('support_field_source')
    model=option.get('model_source') or height_config.get('support_validation',{}).get('model_solid') or export_metadata.get('model_volume_source')
    if not root or not model:
        report.update(status='unavailable',reason='Saved support field and original closed model are required; no field is guessed or extrapolated.')
        return {},report
    def absolute(value):
        path=Path(value);return str((Path(base_dir)/path).resolve()) if not path.is_absolute() else str(path.resolve())
    root,model=absolute(root),absolute(model)
    width=float(support_config.get('width',.5));nominal=float(height_config.get('nominal_layer_height',.2))
    gap=max(float(export_metadata.get('side_clearance',0)),float(export_metadata.get('contact_gap',0)),
            float(height_config.get('contact_support_height',{}).get('gap',0)))
    settings=dict(field_source=root,model_source=model,birth_width_mm=float(option.get('birth_width_mm',4*width)),
        protection_distance_mm=gap+1.25*width,minimum_new_path_clearance_mm=gap+width,
        parent_radius_mm=.5*width,maximum_sheet_shift_mm=max(4*nominal,1.6*width),boundary_sample_mm=min(.25,.5*width))
    chart_height=float(export_metadata.get('nominal_layer_height',nominal))
    settings.update(surface_repair_budget_mm=min(.01,.05*nominal),model_clearance_mm=gap,
        surface_repair_output=str((output_dir/'support_surface_repair').resolve()),
        chart_maximum_y_shift_mm=chart_y_shift_budget(chart_height),
        chart_source_layer_height_mm=chart_height,
        chart_model_clearance_mm=max(0,gap-min(.01,.05*nominal)))
    if not np.isfinite([width,nominal,gap,*[settings[k] for k in ('birth_width_mm','protection_distance_mm','minimum_new_path_clearance_mm','parent_radius_mm','maximum_sheet_shift_mm','boundary_sample_mm')]]).all() or min(width,nominal,settings['birth_width_mm'])<=0 or gap<0:
        raise ValueError('Invalid support-hole repair dimensions.')
    settings.update(domain_sha256=_sha(Path(root)/'domain.npz'),field_sha256=_sha(Path(root)/'optimized_field.npz'),model_sha256=_sha(model))
    enabled=option.get('cache_enabled',True)
    if not isinstance(enabled,bool):raise ValueError('hole_repair.cache_enabled must be boolean.')
    from support_preflight_cache import revision
    settings.update(catalogue_cache_enabled=enabled,
        catalogue_cache_directory=str(Path(__file__).resolve().parents[1]/'.cache/support_preflight'),
        catalogue_cache_revision=revision() if enabled else None)
    supports=[e for e in entries if e['part']=='support' and not e.get('is_planar_base_layer') and not e.get('is_platform_layer')]
    jobs=[(str(mesh_dir/e['combined_mesh']),e,settings) for e in supports]
    catalogs=[]
    if workers<=1:catalogs=[_catalog(job) for job in jobs]
    else:
        pool=ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'))
        pending={pool.submit(_catalog,job):job[0] for job in jobs}
        try:
            for future in as_completed(pending):
                try:catalogs.append(future.result())
                except Exception as exc:
                    for queued in pending:queued.cancel()
                    print(f'[support repair] Preflight failed for {Path(pending[future]).name}: {exc}',flush=True)
                    raise ValueError(f'Support preflight failed for {pending[future]}: {exc}') from exc
                if len(catalogs)%100==0:print(f'[support repair] Catalogued {len(catalogs)}/{len(jobs)} layers',flush=True)
        finally:pool.shutdown(wait=True,cancel_futures=True)
    catalogs.sort(key=lambda r:(r['scalar'],r['file']))
    holes,status=track_lifetimes(catalogs,settings);selected={}
    for i,hole in enumerate(holes):
        if status[i]:selected.setdefault(hole['file'],[]).append(hole)
    folder=output_dir/'support_repair';folder.mkdir(parents=True,exist_ok=True);contexts={}
    for row in catalogs:
        path=folder/(Path(row['file']).stem+'.json')
        path.write_text(json.dumps(dict(settings=settings,regions=row['regions'],holes=selected.get(row['file'],[]),scalar=row['scalar'],field_charts=row['field_charts']),separators=(',',':')))
        contexts[row['file']]=str(path)
    report.update(status='prepared',settings=settings,catalogued_layers=len(catalogs),hole_occurrences=len(holes),
        approved_hole_occurrences=sum(status.values()),changed_layer_candidates=len(selected),seconds=time.perf_counter()-started,
        scope='Born-small, model-distant lifetime planning masks; new path points must pass saved-field and model exclusion checks. No source mesh edits.')
    report['repaired_meshes']={r['file']:r['repaired_mesh'] for r in catalogs if r['repaired_mesh']}
    report['surface_repairs']={r['file']:r['surface_repairs'] for r in catalogs if r['surface_repairs']}
    report['field_charts']={r['file']:r['field_charts'] for r in catalogs if r['field_charts']}
    cache=[r.get('catalogue_cache',{}) for r in catalogs]
    report['catalogue_cache']=dict(enabled=enabled,**{name:sum(r.get(name,0) for r in cache)
        for name in ('hit','miss','invalid','write_failed','hash_seconds')})
    verify_repair_sources(report)
    (folder/'report.json').write_text(json.dumps(report,indent=2))
    return contexts,report


def verify_repair_sources(report):
    """Do not accept workers' cached field data after a source mutation."""
    if report.get('status')!='prepared':return
    s=report['settings']
    for path,key in ((Path(s['field_source'])/'domain.npz','domain_sha256'),
                     (Path(s['field_source'])/'optimized_field.npz','field_sha256'),
                     (Path(s['model_source']),'model_sha256')):
        if _sha(path)!=s[key]:raise ValueError('Support repair source changed during planning: '+str(path))


@lru_cache(maxsize=1)
def _read_context(path):return json.loads(Path(path).read_text())


class FieldRepairLifter:
    def __init__(self,base,mask,context):
        self.base,self.mask,self.context=base,mask,context
        self.field_mask=mask;self.full_field_chart=False;self.chart_y_shift=0.
        self.points=0;self.error=0.;self.clearance=float('inf')
        self.initial_error=0.;self.initial_mesh_samples=0;self.lift_seconds=0.;self.normal_seconds=0.
    def __getattr__(self,name):return getattr(self.base,name)
    def lift(self,points):
        mask=shapely.covers(self.field_mask,shapely.points(points))
        if not mask.any():return self.base.lift(points)
        started=time.perf_counter()
        result=np.empty((len(points),3));ambiguous=snapped=0
        if (~mask).any():result[~mask],ambiguous,snapped=self.base.lift(points[~mask])
        field,model=_context(self.context['settings'])
        local=self.base.tree.nearest(shapely.points(points[mask]));faces=self.base.faces[self.base.face_indices[local]]
        seed=self.base.vertices[faces,1].mean(axis=1);q=np.column_stack((points[mask,0],seed,points[mask,1]))
        scalar=self.context['scalar'];settings=self.context['settings']
        hole=shapely.covers(self.mask,shapely.points(points[mask]))
        material=~hole if self.full_field_chart else np.zeros(len(q),bool)
        if material.any():
            original,_,_=self.base.lift(points[mask][material]);seed[material]=original[:,1];q[material,1]=seed[material]
            # Trilinear Y derivatives are constant within each Y cell. Sample
            # every crossed cell in the small seed bracket, not unrelated empty
            # space in a whole layer's bounding box.
            limit=settings.get('chart_maximum_y_shift_mm',.05)
            count=max(2,int(np.ceil(2*limit/field.spacing))+2)
            lower=seed[material]-limit;upper=seed[material]+limit
            if hasattr(field,'origin') and hasattr(field,'shape'):
                # Check every intersected Y cell inside the saved grid. A
                # wider valid bracket near the boundary must not extrapolate.
                lower=np.maximum(lower,float(field.origin[1]))
                upper=np.minimum(upper,float(field.origin[1]+(field.shape[1]-1)*field.spacing))
            for fraction in np.linspace(0.,1.,count):
                test=q[material].copy();test[:,1]=lower+(upper-lower)*fraction
                _,g=field.evaluate_scalars_and_gradients(test)
                if (g[:,1]<=1e-9).any():raise ValueError('Saved field is not monotone in the projected path root bracket.')
        for _ in range(24):
            value,gradient=field.evaluate_scalars_and_gradients(q);error=value-scalar
            if _==0 and material.any():
                # A first-order normal-distance estimate, not an exact nearest
                # surface distance. Reuse this existing query: no extra sampling.
                initial=abs(error[material])/np.maximum(np.linalg.norm(gradient[material],axis=1),1e-12)
                self.initial_error=max(self.initial_error,float(initial.max(initial=0)))
                self.initial_mesh_samples+=int(material.sum())
            if abs(error).max(initial=0)<1e-10:break
            if (gradient[:,1]<=1e-9).any():raise ValueError('Support hole mask is not a positive-Y field graph.')
            q[:,1]-=np.clip(error/gradient[:,1],-.25,.25)
        value,gradient=field.evaluate_scalars_and_gradients(q)
        physical=abs(value-scalar)/np.maximum(np.linalg.norm(gradient,axis=1),1e-12)
        limits=np.where(material,settings.get('chart_maximum_y_shift_mm',.05),settings['maximum_sheet_shift_mm'])
        if physical.max(initial=0)>1e-5 or (gradient[:,1]<=1e-9).any() or (abs(q[:,1]-seed)>limits).any():
            bad=int(np.argmax(abs(q[:,1]-seed)-limits))
            raise ValueError(f'Support path cannot be lifted onto its saved field sheet: Y shift={abs(q[bad,1]-seed[bad]):.6g} mm, '
                f'limit={limits[bad]:.6g} mm, point={q[bad].tolist()}, field error={physical.max(initial=0):.6g} mm, '
                f'initial normal-error estimate={self.initial_error:.6g} mm.')
        distance=_distances(model,q)
        clearance=np.where(material,settings.get('chart_model_clearance_mm',settings.get('model_clearance_mm',0)),settings['minimum_new_path_clearance_mm'])
        if (distance<clearance).any() or _confirmed_inside(model,q).any():
            bad=int(np.argmin(distance-clearance))
            raise ValueError(f'Support field lift enters model keepout: distance={distance[bad]:.6g} mm, '
                f'required={clearance[bad]:.6g} mm, point={q[bad].tolist()}.')
        self.points+=len(q);self.error=max(self.error,float(physical.max(initial=0)))
        if material.any():self.chart_y_shift=max(self.chart_y_shift,float(abs(q[material,1]-seed[material]).max(initial=0)))
        self.clearance=min(self.clearance,float(distance.min()));result[mask]=q
        self.lift_seconds+=time.perf_counter()-started
        return result,ambiguous,snapped
    def normals(self,points,normals):
        mask=shapely.covers(self.field_mask,shapely.points(points[:,[0,2]]))
        if mask.any():
            started=time.perf_counter()
            field,_=_context(self.context['settings']);_,g=field.evaluate_scalars_and_gradients(points[mask])
            g/=np.maximum(np.linalg.norm(g,axis=1,keepdims=True),1e-12)
            g*=np.where(np.einsum('ij,ij->i',g,normals[mask])<0,-1.,1.)[:,None];normals[mask]=g
            self.normal_seconds+=time.perf_counter()-started
        return normals
    def report(self):return dict(mask_count=self.mask_count,field_lift_points=self.points,
        maximum_physical_field_error_mm=self.error,minimum_model_distance_mm=self.clearance if self.points else None,
        full_field_chart=self.full_field_chart,maximum_chart_y_shift_mm=self.chart_y_shift,
        initial_mesh_samples=self.initial_mesh_samples,
        maximum_initial_normalized_field_error_mm=self.initial_error if self.initial_mesh_samples else None,
        field_lift_seconds=self.lift_seconds,field_normal_seconds=self.normal_seconds)


def bind_repair(lifter,context_path):
    context=_read_context(str(context_path));identity=patch_identity(lifter)
    if identity not in context['regions']:raise ValueError('Support surface changed after repair catalogue construction.')
    region=shapely.from_wkb(context['regions'][identity]);approved=[]
    tree=cKDTree(lifter.vertices)
    for hole in context['holes']:
        distance,_=tree.query(np.asarray(hole['ring']))
        if distance.min()<1e-7:approved.append(Polygon(hole['xy']))
    mask=unary_union(approved);wrapped=FieldRepairLifter(lifter,mask,context);wrapped.mask_count=len(approved)
    if context.get('field_charts',{}).get(identity):
        wrapped.full_field_chart=True;wrapped.field_mask=region.union(mask)
    return wrapped,region.union(mask) if approved else region


def save_reference_addition(lifter,output_dir,stem):
    """Give the height stage the same filled material used by path planning.

    The original layers are untouched. Only field-lifted mask triangles are
    written; height-reference assembly later appends them to the source mesh.
    """
    if lifter.mask.is_empty or lifter.mask.area<=1e-12:return None
    triangles=shapely.get_parts(shapely.constrained_delaunay_triangles(lifter.mask))
    triangles=np.asarray([np.asarray(t.exterior.coords)[:3] for t in triangles if t.area>1e-12])
    if not len(triangles):return None
    # Refine only new mask triangles, keeping both edge length and saved-field
    # chord error bounded. The source adaptive mesh is never resampled here.
    for iteration in range(9):
        corners,_,_=lifter.lift(triangles.reshape(-1,2));corners=corners.reshape(-1,3,3)
        center,_,_=lifter.lift(triangles.mean(axis=1))
        edge=np.linalg.norm(corners-np.roll(corners,1,axis=1),axis=2).max(axis=1)
        error=np.linalg.norm(center-corners.mean(axis=1),axis=1)
        refine=(edge>1.)|(error>.01)
        if not refine.any():break
        if iteration==8 or len(triangles)+3*int(refine.sum())>100_000:
            raise ValueError('Support repair height-reference refinement exceeded its bounded budget.')
        selected=triangles[refine];a,b,c=selected[:,0],selected[:,1],selected[:,2]
        ab,bc,ca=(a+b)/2,(b+c)/2,(c+a)/2
        children=np.stack((np.stack((a,ab,ca),axis=1),np.stack((ab,b,bc),axis=1),
                           np.stack((ca,bc,c),axis=1),np.stack((ab,bc,ca),axis=1)),axis=1).reshape(-1,3,2)
        triangles=np.concatenate((triangles[~refine],children))
    path=Path(output_dir)/(stem+'.repair_reference.obj');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8') as stream:
        np.savetxt(stream,corners.reshape(-1,3),fmt='v %.17g %.17g %.17g')
        np.savetxt(stream,np.arange(len(corners)*3).reshape(-1,3)+1,fmt='f %d %d %d')
    return str(path.resolve())


def assemble_height_references(source_dir,output_dir,layers):
    """Keep synthetic path fill and nearest-surface height references aligned."""
    from mesh_distance_workflow import load_obj_mesh
    import shutil
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    for row in layers:
        name=row['source_mesh'];source=Path(row.get('support_surface_reference_file') or (Path(source_dir)/name));target=output_dir/name
        additions=row.get('support_reference_additions',[])
        if not additions:shutil.copy2(source,target);continue
        meshes=[load_obj_mesh(str(p)) for p in [source,*map(Path,additions)]]
        vertices=[];faces=[];offset=0
        for v,f in meshes:vertices.append(v);faces.append(f+offset);offset+=len(v)
        with target.open('w',encoding='utf8') as stream:
            np.savetxt(stream,np.vstack(vertices),fmt='v %.17g %.17g %.17g')
            np.savetxt(stream,np.vstack(faces)+1,fmt='f %d %d %d')
    return output_dir
