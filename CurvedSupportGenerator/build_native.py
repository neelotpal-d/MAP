"""Build optional exact support BVH/cache with a local C++ compiler."""
from pathlib import Path
import os
import shutil
import subprocess
import sys


def main():
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
    for sub,source,name in (('v3_ordered','support_union_bvh_ordered.cpp','support_union_bvh'),('point_cache_v1','exact_point_cache.cpp','exact_point_cache')):
        destination=root/'native_build'/sub;destination.mkdir(parents=True,exist_ok=True)
        if windows:
            command=[compiler,'/nologo','/O2','/EHsc','/openmp','/LD',str(root/source),'/Fo'+str(destination/(name+'.obj')),
                '/link','/OUT:'+str(destination/(name+'.dll')),'/IMPLIB:'+str(destination/(name+'.lib'))]
        else:
            flags=['-dynamiclib'] if sys.platform=='darwin' else ['-shared','-fopenmp']
            suffix='.dylib' if sys.platform=='darwin' else '.so'
            command=[compiler,'-O3','-std=c++17','-fPIC',*flags,str(root/source),'-o',str(destination/(name+suffix))]
        subprocess.run(command,cwd=destination,env=environment,check=True)
    print('Exact support BVH and point cache built.',flush=True)


if __name__=='__main__':main()
