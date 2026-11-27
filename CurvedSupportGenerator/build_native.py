"""Build optional exact support BVH/cache with a local C++ compiler."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import argparse
import json
import platform


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,help='Build into a separate staging folder.')
    parser.add_argument('--static-runtime',action='store_true',help='Windows: link the C/C++ runtime statically; OpenMP still uses the runtime environment.')
    args=parser.parse_args(argv)
    root=Path(__file__).resolve().parent/'runtime/SupportGenerationCurved';environment=os.environ.copy()
    windows=sys.platform=='win32'
    if windows:
        compiler=shutil.which('cl.exe')
        if not compiler:
            finder=Path(os.environ.get('ProgramFiles(x86)','C:/Program Files (x86)'))/'Microsoft Visual Studio/Installer/vswhere.exe'
            if not finder.is_file():raise RuntimeError('Install MSVC C++ build tools or use the exact array fallback.')
            install=subprocess.check_output([str(finder),'-latest','-products','*','-requires','Microsoft.VisualStudio.Component.VC.Tools.x86.x64','-property','installationPath'],text=True).strip()
            vcvars=Path(install)/'VC/Auxiliary/Build/vcvars64.bat'
            output=subprocess.check_output(f'call "{vcvars}" >nul && set',shell=True,text=True)
            environment={k.upper():v for k,v in environment.items()}
            for line in output.splitlines():
                if '=' in line:
                    key,value=line.split('=',1)
                    if key:environment[key.upper()]=value
            compiler=shutil.which('cl.exe',path=environment.get('PATH'))
        if not compiler:raise RuntimeError('MSVC compiler unavailable.')
    else:
        compiler=shutil.which('c++')
        if not compiler:raise RuntimeError('Install a C++ compiler or use the exact array fallback.')
    build_root=args.output_root.resolve() if args.output_root else root/'native_build'
    mac_openmp=Path(sys.prefix)/'lib/libomp.dylib'
    use_openmp=windows or sys.platform!='darwin' or mac_openmp.is_file()
    if sys.platform=='darwin' and not use_openmp:
        print('Pinned OpenMP runtime not found; building serial C++ acceleration.',flush=True)
    commands=[]
    for sub,source,name in (('v3_ordered','support_union_bvh_ordered.cpp','support_union_bvh'),('point_cache_v1','exact_point_cache.cpp','exact_point_cache')):
        destination=(args.output_root.resolve() if args.output_root else root/'native_build')/sub;destination.mkdir(parents=True,exist_ok=True)
        if windows:
            command=[compiler,'/nologo','/O2','/EHsc','/openmp','/LD',*(['/MT'] if args.static_runtime else []),str(root/source),'/Fo'+str(destination/(name+'.obj')),
                '/link','/OUT:'+str(destination/(name+'.dll')),'/IMPLIB:'+str(destination/(name+'.lib'))]
        else:
            flags=['-dynamiclib'] if sys.platform=='darwin' else ['-shared','-fopenmp']
            if sys.platform=='darwin' and use_openmp:
                flags+=['-Xpreprocessor','-fopenmp','-L'+str(mac_openmp.parent),'-lomp',
                        '-Wl,-rpath,'+str(mac_openmp.parent)]
            suffix='.dylib' if sys.platform=='darwin' else '.so'
            command=[compiler,'-O3','-std=c++17','-fPIC',*flags,str(root/source),'-o',str(destination/(name+suffix))]
        try:subprocess.run(command,cwd=destination,env=environment,check=True)
        except subprocess.CalledProcessError:
            if sys.platform!='darwin' or not use_openmp:raise
            print('OpenMP build failed; retrying this library with serial C++ acceleration.',flush=True)
            # Remove only the OpenMP additions. Compiled serial queries and
            # process-level layer parallelism remain available.
            command=[compiler,'-O3','-std=c++17','-fPIC','-dynamiclib',str(root/source),'-o',str(destination/(name+suffix))]
            subprocess.run(command,cwd=destination,env=environment,check=True)
            use_openmp=False
        commands.append(command)
    (build_root/'build-info.json').write_text(json.dumps(dict(platform=sys.platform,compiler=compiler,
        commands=commands,openmp_enabled=('/openmp' in commands[0] or '-fopenmp' in commands[0]),
        cuda_required=False,cpu='baseline; no native CPU tuning',architecture=platform.machine(),
        libc=platform.libc_ver() if sys.platform.startswith('linux') else None),indent=2)+'\n')
    print('Exact support BVH and point cache built.',flush=True)


if __name__=='__main__':main()
