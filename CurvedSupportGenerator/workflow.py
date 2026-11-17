"""Portable field -> platform-first tree -> saved-field layers entry point."""
from pathlib import Path
import argparse
import faulthandler
import json
import os
import sys


def configure_environment():
    faulthandler.enable()
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VTK_SMP_MAX_THREADS'):
        os.environ[name]='1'
    handles=[]
    if sys.platform=='win32':
        import ctypes
        prefix=Path(sys.executable).resolve().parent
        directories=[p for p in (prefix,prefix/'Library/bin',prefix/'Library/usr/bin',prefix/'Library/mingw-w64/bin',prefix/'Scripts') if p.is_dir()]
        if (prefix/'conda-meta').is_dir():
            os.environ['PATH']=os.pathsep.join(map(str,directories))+os.pathsep+os.environ.get('PATH','')
            if hasattr(os,'add_dll_directory'):handles=[os.add_dll_directory(str(p)) for p in directories]
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetErrorMode.restype=ctypes.c_uint;kernel.SetErrorMode.argtypes=[ctypes.c_uint]
        kernel.SetErrorMode(kernel.GetErrorMode()|0x0001|0x0002)
    return handles


def main(argv=None):
    handles=configure_environment()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('job',type=Path)
    p.add_argument('--stage',choices=('field','neural-field','tree','layers','separated-posts'),required=True);p.add_argument('--output',type=Path)
    p.add_argument('--minimum-bend-radius',type=float,default=5.);p.add_argument('--maximum-merge-radius',type=float,default=24.)
    p.add_argument('--stronger-merging',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--routing-threads',type=int,default=4);p.add_argument('--recovery-budget',type=float,default=5.)
    args,forwarded=p.parse_known_args(argv)
    from cli import load_job
    output,clean,pipeline=load_job(args.job,output_override=args.output,clean_override=False)
    sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
    settings=pipeline.get('settings',{})
    if args.stage=='neural-field':
        if forwarded:p.error('Unknown neural field options: '+' '.join(forwarded))
        import torch
        from SupportGenerationCurved.neural_support import train
        train(pipeline,output,settings)
    elif args.stage=='field':
        if forwarded:p.error('Unknown field options: '+' '.join(forwarded))
        # Load Torch's native libraries before VTK/NumPy on Windows. Qt stays
        # in the parent interpreter; this is the isolated numerical worker.
        import torch
        from SupportGenerationCurved.conservative_domain import prepare,finish
        from SupportGenerationCurved.conservative_field import optimize
        prepare(pipeline,output,settings)
        optimize(output,rounds=int(settings.get('rounds',4)),iterations=int(settings.get('iterations',200)),
            collision_weight=float(settings.get('collision_weight',200.)),settings=settings)
        finish(output,include_holdout=bool(settings.get('independent_validation',True)))
    elif args.stage=='tree':
        from SupportGenerationCurved.platform_router.engine import route_model
        maximum=int(forwarded[forwarded.index('--max-steps')+1]) if '--max-steps' in forwarded else 450
        route_model(pipeline,output,forwarded,maximum_steps=maximum,minimum_bend_radius=args.minimum_bend_radius,
            maximum_merge_radius=args.maximum_merge_radius,stronger_merging=args.stronger_merging,
            recovery_time_budget=args.recovery_budget,routing_threads=args.routing_threads)
    elif args.stage=='separated-posts':
        from SupportGenerationCurved.separated_posts import route
        options=argparse.ArgumentParser(prog='workflow.py --stage separated-posts')
        options.add_argument('--clearance',type=float,default=.4)
        options.add_argument('--overhang-angle',type=float,default=60.)
        options.add_argument('--minimum-support-height',type=float,default=.6)
        options.add_argument('--maximum-contacts',type=int,default=5000)
        route(pipeline,output,**vars(options.parse_args(forwarded)))
    else:
        if forwarded:p.error('Unknown layer options: '+' '.join(forwarded))
        from SupportGenerationCurved.saved_field_layers import run_job
        run_job(pipeline,output)
    (output/'workflow_input.json').write_text(json.dumps(pipeline,indent=2),encoding='utf8')
    print(json.dumps(dict(stage=args.stage,status='complete',output=str(output))),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
