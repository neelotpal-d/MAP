"""Isolated representative workloads built from the production MAP kernels."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'CurvedSupportGenerator/runtime'),str(Path(__file__).resolve().parent)]
_LAYER_STATE=None


def write(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf8')
def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def layer_initialize(fixture,output,threads):
    global _LAYER_STATE
    import numpy as np
    import pyvista as pv
    from SupportGenerationCurved import tree_geometry as tg
    from SupportGenerationCurved.grid_distance_certificate import GridDistanceCertificate
    fixture=Path(fixture);geometry_threads=tg.configure_geometry_threads(threads)
    mesh=pv.read(fixture/'model.vtp');rows=np.load(fixture/'branches.npy')
    shape=tg.TaperedTree(rows,mesh,[0,1,0],0.,.4,.4,terminal_cap='rounded',distance_backend='certified',prune_pairs=True)
    shape.distance_certificate=GridDistanceCertificate(mesh,shape.distance)
    backend='exact arrays';accelerator=None
    try:
        from SupportGenerationCurved.exact_support_bvh import ExactSupportBVH,LexCachedLayerField
        accelerator=ExactSupportBVH(shape,threads=threads)
    except (FileNotFoundError,OSError):pass
    else:
        shape.evaluate=accelerator.evaluate;shape._exact_bvh=accelerator
        tg.CachedLayerField=LexCachedLayerField;backend='exact BVH / '+accelerator.cache_backend
    # Production saved-field jobs use explicit double-precision volume points.
    active=pv.read(fixture/'layer_grid.vti').cast_to_unstructured_grid()
    active.points=np.asarray(active.points,dtype=np.float64)
    _LAYER_STATE=(shape,active,Path(output),backend,geometry_threads,accelerator)


def layer_task(task):
    import numpy as np
    from SupportGenerationCurved import tree_geometry as tg
    shape,active,output,backend,budget,_=_LAYER_STATE
    index,value=task
    result=tg.extract_one_layer(shape,active,index,value,output,.4,.2,'surface_scalar',
        layer_compression='lz4',adaptive_layer_tolerance=.05,_return_surface=True)
    if result['vertex_bad'] or result['centroid_bad']:raise ValueError('Calibration layer failed its production geometry checks.')
    surface=result['surface']
    digest=hashlib.sha256(np.round(surface.points,6).astype('<f8').tobytes()+surface.faces.astype('<i8').tobytes()).hexdigest()
    return dict(cells=int(result['manifest']['cells']),points=int(result['manifest']['points']),digest=digest,backend=backend,vtk=budget)


def prepare(folder,samples):
    import numpy as np
    import pyvista as pv
    from connected_patch_planner import _write_obj
    folder.mkdir(parents=True,exist_ok=False)
    model=pv.Sphere(radius=6.,center=(0,18,0),theta_resolution=32,phi_resolution=24).triangulate()
    model.save(folder/'model.vtp')
    rows=np.array([[x,0,0,x,12,0,2.,1.,0] for x in (-4.,0.,4.)])
    np.save(folder/'branches.npy',rows)
    grid=pv.ImageData(dimensions=(48,40,28),spacing=(16/47,12/39,10/27),origin=(-8,0,-5))
    points=grid.points;grid['scalar']=points[:,1]+.1*np.sin(points[:,0]/3)*np.cos(points[:,2]/3)
    grid.save(folder/'layer_grid.vti')
    write(folder/'settings.json',dict(samples=samples,layer_levels=np.linspace(.8,10.8,samples).tolist()))
    paired=folder/'paired';paired.mkdir();manifests={}
    for part in ('model','support'):
        target=paired/(part+'_layers');target.mkdir();records=[]
        xs=np.linspace(-26,-6,37) if part=='model' else np.linspace(6,14,31)
        zs=np.linspace(-8,8,33)
        xx,zz=np.meshgrid(xs,zs);x=xx.ravel();z=zz.ravel()
        faces=[];nx=len(xs)
        for iz in range(len(zs)-1):
            for ix in range(nx-1):
                a=iz*nx+ix;faces.extend([[a,a+1,a+nx],[a+1,a+nx+1,a+nx]])
        faces=np.asarray(faces);cx=x[faces].mean(axis=1);cz=z[faces].mean(axis=1)
        keep=((cx+16)**2+cz**2>2.5**2) if part=='model' else (((cx-10)**2+(cz-4)**2<2.8**2)|((cx-10)**2+(cz+4)**2<2.8**2))
        faces=faces[keep]
        for index in range(samples):
            height=.2*(index+1);warp=.04*np.sin(x/5)*np.cos(z/4)
            vertices=np.column_stack((x,height+warp,z));name=f'{index}_0.obj'
            _write_obj(target/name,vertices,faces)
            records.append(dict(file=name,main_index=index,sub_index=0,layer_scalar=height,
                part=part,is_platform_layer=part=='support' and index==0,
                is_contact_layer=part=='support' and index>=samples-3,
                role_classification='calibration_fixture'))
        manifests[part]=records;write(paired/(part+'_layer_manifest.json'),records)
    write(paired/'compatible_export.json',dict(complete_layers=False,calibration_only=True,contact_gap=.4,side_clearance=.4))
    # Warm only the optional native build/cache. Timed trials still initialize
    # their own processes, meshes and accelerators like normal production jobs.
    warm=folder/'warmup';warm.mkdir();layer_initialize(folder,warm,1)
    warm_paths=folder/'pipeline_warmup';warm_paths.mkdir()
    pipeline(folder,warm_paths,1,1)
    print(json.dumps(dict(prepared=True,samples=samples,layer_backend=_LAYER_STATE[3])),flush=True)


def layers(fixture,output,workers,threads):
    config=json.loads((fixture/'settings.json').read_text(encoding='utf8'))
    tasks=list(enumerate(config['layer_levels']))
    if workers==1:
        layer_initialize(fixture,output,threads);results=list(map(layer_task,tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=layer_initialize,initargs=(fixture,output,threads)) as pool:results=list(pool.map(layer_task,tasks))
    counts=dict(cells=sum(r['cells'] for r in results),points=sum(r['points'] for r in results),layers=len(results))
    if not counts['cells']:raise ValueError('Calibration extraction returned empty geometry.')
    return dict(signature=signature([counts,[r['digest'] for r in results]]),work=counts,backend=results[0]['backend'],vtk=results[0]['vtk'])


def pipeline(fixture,output,workers,threads):
    import numpy as np
    from external_projected_pipeline.api import create_config
    from run_surface_pipeline_projected_manifest_support import run
    config=output/'config.json'
    create_config(config,pipeline='support-model',export_root=fixture/'paired',output_root=output/'outputs',
        spacing=.5,resample_spacing=.5,boundary_count=2,layer_workers=workers,
        overrides={'execution.native_threads_per_worker':threads,'execution.blas_threads_per_process':1,
            'execution.keep_patch_artifacts':False,'execution.io_workers':min(4,workers),
            'support_paths.hole_repair.enabled':False,'support_paths.backing_repair.enabled':False,
            'support_paths.regular.contour_count':1})
    os.environ['IGL_NUM_THREADS']=str(threads)
    os.environ['PROJECTED_IO_WORKERS']=str(min(4,workers))
    os.environ['PROJECTED_KEEP_PATCH_ARTIFACTS']='0'
    # Importing the numerical stage occurs only in this fresh child process.
    result=run(config)
    path=result/'batch_3axis.meta.json';meta=json.loads(path.read_text(encoding='utf8'))
    meta.update(machine_export_supported=False,calibration_only=True)
    write(path,meta)
    counts=dict(layers=len(meta['layers']),points=sum(int(r.get('path_point_count',0)) for r in meta['layers']),
        unresolved=int(meta.get('support_validation',{}).get('unresolved_points',0)))
    # Height metadata names are stable but not all old batches repeat point
    # counts. Use the raw production path metadata for this invariant.
    raw=json.loads((output/'outputs/toolpaths/batch_paths.meta.json').read_text(encoding='utf8'))
    counts['points']=sum(int(r.get('path_point_count',0)) for r in raw['layers'])
    if not counts['points']:raise ValueError('Calibration planning returned no path points.')
    digest=hashlib.sha256()
    for path in sorted(result.glob('*.txt')):
        digest.update(path.name.encode())
        if path.name.endswith(('.tags.txt','.jumps.txt')) or not path.stat().st_size:
            digest.update(path.read_bytes());continue
        data=np.loadtxt(path,ndmin=2)
        if not np.isfinite(data).all():raise ValueError('Calibration produced nonfinite paths/heights.')
        digest.update(np.round(data,6).astype('<f8').tobytes())
    return dict(signature=signature([counts,digest.hexdigest()]),work=counts,backend='production projected model/support + height pipeline')


def distance(fixture,output,workers,threads):
    import numpy as np
    import pyvista as pv
    from SupportGenerationCurved import tree_geometry as tg
    from SupportGenerationCurved.platform_router.acceleration import NativeDistance
    budget=tg.configure_geometry_threads(threads);query=NativeDistance(pv.read(fixture/'model.vtp'))
    rng=np.random.default_rng(847);total=0.;count=0
    # Include small scalar calls and the batched array overload used in routing.
    for size in (16,128,1024,8192):
        points=rng.uniform([-8,0,-8],[8,24,8],(size,3))
        for _ in range(12):total+=float(query(points).sum());count+=size
    return dict(signature=signature(dict(sum=round(total,3),points=count)),work=dict(points=count),
        backend='production NativeDistance routing kernel',vtk=budget)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','layers','pipeline','distance'])
    p.add_argument('fixture',type=Path);p.add_argument('--output',type=Path);p.add_argument('--result',type=Path)
    p.add_argument('--samples',type=int,default=24);p.add_argument('--workers',type=int,default=1);p.add_argument('--threads',type=int,default=1)
    a=p.parse_args()
    if a.stage=='prepare':prepare(a.fixture,a.samples);return
    a.output.mkdir(parents=True,exist_ok=False);began=time.perf_counter()
    result=globals()[a.stage](a.fixture,a.output,a.workers,a.threads)
    result.update(status='passed',kernel_seconds=time.perf_counter()-began)
    write(a.result,result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
