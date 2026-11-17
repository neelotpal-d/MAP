"""Lightweight, recorded native-thread budgets for portable child processes."""
import math
import os


def positive_integer(value,name):
    if isinstance(value,bool) or not isinstance(value,int) or value<1:
        raise ValueError(f'{name} must be a positive integer.')
    return value


def resolve_execution(config,environ=None,cpu_count=None):
    env=os.environ if environ is None else environ
    cpu=max(1,(os.cpu_count() or 1) if cpu_count is None else cpu_count)
    stages={name:positive_integer(config.get(section,{}).get(key,1),section+'.'+key)
        for name,section,key in [('model','hybrid_contours','layer_workers'),
            ('support','support_paths','layer_workers'),('heights','heights_3axis','layer_workers')]}
    settings=config.get('execution',{})
    if not isinstance(settings,dict):raise ValueError('execution must be an object.')
    inherited=env.get('IGL_NUM_THREADS')
    try:inherited=int(inherited) if inherited is not None else None
    except (TypeError,ValueError):inherited=None
    native=settings.get('native_threads_per_worker',inherited if inherited and inherited>0 else
        max(1,min(2,cpu//max(stages.values()))))
    native=positive_integer(native,'execution.native_threads_per_worker')
    blas=positive_integer(settings.get('blas_threads_per_process',1),'execution.blas_threads_per_process')
    keep=settings.get('keep_patch_artifacts',True)
    if not isinstance(keep,bool):raise ValueError('execution.keep_patch_artifacts must be true or false.')
    io=positive_integer(settings.get('io_workers',1),'execution.io_workers')
    return dict(layer_processes=stages,native_threads_per_worker=native,blas_threads_per_process=blas,
        keep_patch_artifacts=keep,
        io_workers=io,
        logical_processors=cpu,maximum_requested_native_threads=max(stages.values())*native,
        oversubscribed=max(stages.values())*native>cpu,
        policy='Stages run sequentially; explicit config then inherited IGL limit, otherwise at most two native threads per worker. BLAS defaults to one.')


def execution_environment(settings,environ=None):
    result=dict(os.environ if environ is None else environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        result[key]=str(settings['blas_threads_per_process'])
    result['IGL_NUM_THREADS']=str(settings['native_threads_per_worker'])
    result['PROJECTED_KEEP_PATCH_ARTIFACTS']='1' if settings['keep_patch_artifacts'] else '0'
    result['PROJECTED_IO_WORKERS']=str(settings['io_workers'])
    result.setdefault('VTK_SMP_MAX_THREADS','1')
    return result
