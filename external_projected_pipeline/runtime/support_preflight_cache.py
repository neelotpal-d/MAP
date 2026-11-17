"""Content-addressed JSON cache for verified per-layer support catalogues.

Always hash source bytes. Field/model content, scalar, boundary sampling,
implementation and dependency versions bind a record to its computation.
Width/gap/lifetime decisions are recomputed from the cached measurements.
"""
from pathlib import Path
import gzip,hashlib,importlib.metadata,json,os,platform,sys,time,uuid
import numpy as np

SCHEMA='support-preflight.v1'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def revision():
    runtime=Path(__file__).resolve().parent
    files=[runtime/name for name in ('support_preflight_cache.py','support_hole_repair.py',
        'projection_mesh_repair.py','surface_hatch_metric.py','connected_patch_planner.py',
        'batch_generate_projected_hybrid_paths.py','mesh_distance_workflow.py','heat_method_core.py')]
    files.append(runtime.parents[1]/'CurvedSupportGenerator/runtime/SupportGenerationCurved/field_grid.py')
    h=hashlib.sha256()
    for path in files:
        h.update(path.name.encode());h.update(path.read_bytes())
    for name in ('numpy','shapely','pyvista','scipy','libigl'):
        h.update((name+':'+importlib.metadata.version(name)).encode())
    import shapely
    from vtkmodules.vtkCommonCore import vtkVersion
    h.update(('GEOS:'+shapely.geos_version_string+';VTK:'+vtkVersion.GetVTKSourceVersion()).encode())
    h.update((sys.platform+':'+platform.machine()).encode())
    h.update(str(sys.version_info[:2]).encode())
    return h.hexdigest()

def reclassify(row,settings):
    """Apply current thresholds; do not reuse old width/gap decisions."""
    for chart in row['field_charts'].values():
        if chart is None:continue
        if (not np.isfinite([chart['maximum_vertex_field_error_mm'],chart['minimum_gradient_y'],
                chart['maximum_facet_difference_mm']]).all()
                or chart['maximum_vertex_field_error_mm']>settings['surface_repair_budget_mm']
                or chart['minimum_gradient_y']<=1e-9):
            raise ValueError('Conflicting vertices do not belong to the saved scalar level within its geometry budget.')
        budget=settings['chart_maximum_y_shift_mm']
        if chart['maximum_facet_difference_mm']>budget:
            raise ValueError('Conflicting facets exceed the source-chart displacement budget.')
        chart['source_chart_displacement_budget_mm']=float(budget)
    for hole in row['holes']:
        xy=np.asarray(hole['xy'],float)
        hole['small_birth_candidate']=bool(np.ptp(xy,axis=0).max()<settings['birth_width_mm'])
        hole['protected']=bool(hole['model_distance_mm']<=settings['protection_distance_mm'])
    return row

class Probe:
    def __init__(self,path,entry,settings):
        self.source=Path(path);self.entry=entry;self.settings=settings;self.path=None;self.source_sha=None
        self.stats=dict(hit=False,miss=False,invalid=False,write_failed=False,hash_seconds=0.)
        if not settings.get('catalogue_cache_enabled',False):return
        start=time.perf_counter();self.source_sha=digest(path);self.stats['hash_seconds']=time.perf_counter()-start
        identity=dict(schema=SCHEMA,source_sha256=self.source_sha,file=entry['combined_mesh'],scalar=float(entry['scalar']),
            domain_sha256=settings['domain_sha256'],field_sha256=settings['field_sha256'],model_sha256=settings['model_sha256'],
            boundary_sample_mm=settings['boundary_sample_mm'],revision=settings['catalogue_cache_revision'])
        self.key=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        self.path=Path(settings['catalogue_cache_directory'])/self.key[:2]/(self.key+'.json.gz')

    def load(self):
        if self.path is None:return None
        try:
            with gzip.open(self.path,'rb') as stream:data=stream.read(32*1024*1024+1)
            if len(data)>32*1024*1024:raise ValueError('Oversized catalogue cache entry.')
            check,payload=data.split(b'\n',1)
            if check.decode('ascii')!=hashlib.sha256(payload).hexdigest():raise ValueError('Catalogue cache checksum mismatch.')
            envelope=json.loads(payload)
            if envelope['schema']!=SCHEMA or envelope['key']!=self.key:raise ValueError('Catalogue cache identity mismatch.')
            row=envelope['row']
            if row['file']!=self.entry['combined_mesh'] or row['scalar']!=self.entry['scalar']:raise ValueError('Catalogue layer mismatch.')
            self.stats['hit']=True
            return reclassify(row,self.settings)
        except FileNotFoundError:self.stats['miss']=True
        except (OSError,ValueError,TypeError,KeyError,EOFError):self.stats.update(miss=True,invalid=True)
        return None

    def store(self,row):
        if self.path is None:return
        if digest(self.source)!=self.source_sha:raise ValueError('Support layer changed during preflight: '+str(self.source))
        temporary=self.path.with_name(self.path.name+'.'+str(os.getpid())+'.'+uuid.uuid4().hex+'.tmp')
        try:
            payload=json.dumps(dict(schema=SCHEMA,key=self.key,row=row),sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
            data=hashlib.sha256(payload).hexdigest().encode('ascii')+b'\n'+payload
            self.path.parent.mkdir(parents=True,exist_ok=True)
            temporary.write_bytes(gzip.compress(data,compresslevel=1,mtime=0));os.replace(temporary,self.path)
        except (OSError,ValueError):self.stats['write_failed']=True
        finally:
            if temporary.exists():temporary.unlink()
